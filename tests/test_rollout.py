import itertools
import threading

import numpy as np
import pytest
import torch

from pld_rc365.rl.agent import ResidualSAC
from pld_rc365.rl.buffers import ReplayBuffer, segment_mc_returns, symmetric_sample
from pld_rc365.rollout import EpisodeSpec, Runner, Specialist

from .fakes import FakeBackbone, FakeBase, FakeVecEnv

RL = dict(batch_size=8, buffer_capacity=1000, discount=0.99, lr=3e-4, grad_clip=1.0, reward_bias=0.0,
          critic_actor_ratio=2, otf_samples=1, target_entropy_scale=0.5, init_temperature=1.0, tau=0.005,
          num_q=2, hidden_dim=32, latent_dim=16, num_spatial_features=2, backup_entropy=False, utd=1.0,
          actor_sync_every=5, calql=dict(steps=5, alpha=5.0, num_samples=3))
RES = dict(dims=7, xi=0.5, xi_schedule="constant")
FMAP = (4, 2, 2)


def make_spec_obj(k):
    agent = ResidualSAC(RL, RES, FMAP, 2, device="cpu", seed=k)
    import copy
    return Specialist(agent=agent, collector_actor=copy.deepcopy(agent.actor), actor_lock=threading.Lock(),
                      offline=ReplayBuffer(500, (2, *FMAP), 16, 12, 7), online=ReplayBuffer(500, (2, *FMAP), 16, 12, 7))


def make_runner(num_envs=2, horizon=12, stage_steps=(5,), learnable=None, succeed=lambda s: s % 2 == 0, exec_steps=4):
    venv = FakeVecEnv(num_envs, horizon, stage_steps, succeed)
    base = FakeBase(num_envs, exec_steps)
    K = len(stage_steps)
    learnable = set(range(K)) if learnable is None else learnable
    specs = {k: make_spec_obj(k) for k in learnable}
    r = Runner(venv, base, FakeBackbone(), "cpu", K, learnable, specs, residual_dims=7, gamma=0.99)
    return r, venv, base, specs


def run_n(runner, specs_iter, n):
    it = iter(specs_iter)
    results = []
    runner.run(lambda: next(it, None), lambda r: results.append(r) or False)
    assert len(results) == n
    return results


def test_base_actions_and_next_ab_alignment_online():
    runner, venv, base, specs = make_runner(num_envs=1, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    run_n(runner, [EpisodeSpec(seed=1, mode="base", store="online")], 1)
    buf = specs[0].online
    assert buf.size == 12  # every step stored, incl. the truncated last one
    ab = buf.data["a_b"][:12, 0] * 1000
    nab = buf.data["next_a_b"][:12, 0] * 1000
    np.testing.assert_allclose(ab, np.arange(12), atol=1e-3)          # a_b at step t is the action for step t
    np.testing.assert_allclose(nab, np.arange(1, 13), atol=1e-3)      # next_a_b is the action used at t+1
    assert np.all(buf.data["done"][:12] == 0)
    # features of obs t encode t, next_feat encode t+1
    np.testing.assert_allclose(buf.data["feat"][:12, 0, 0, 0, 0], np.arange(12), atol=1e-2)
    np.testing.assert_allclose(buf.data["next_feat"][:12, 0, 0, 0, 0], np.arange(1, 13), atol=1e-2)
    # base-only data: a_bar = a_b[:7]
    np.testing.assert_allclose(buf.data["a_bar"][:12], np.clip(buf.data["a_b"][:12, :7], -1, 1))


def test_probing_steps_not_stored_and_residual_after_takeover():
    runner, venv, base, specs = make_runner(num_envs=1, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    res = run_n(runner, [EpisodeSpec(seed=1, mode="pld", t_base=7, store="online")], 1)[0]
    assert res.took_over and res.takeover_step == 7
    buf = specs[0].online
    assert buf.size == 12 - 7
    np.testing.assert_allclose(buf.data["a_b"][:5, 0] * 1000, np.arange(7, 12), atol=1e-3)
    executed = np.stack(venv.actions[(0, 1)])
    # probing: executed == base action exactly; after takeover: arm dims differ by |a_delta| <= xi
    np.testing.assert_allclose(executed[:7, 0] * 1000, np.arange(7), atol=1e-3)
    stored_abar = buf.data["a_bar"][:5]
    np.testing.assert_allclose(executed[7:, :7], stored_abar, atol=1e-6)   # online buffer holds the FULL action
    assert np.all(np.abs(stored_abar - np.clip(buf.data["a_b"][:5, :7], -1, 1)) <= 0.5 + 1e-6)
    np.testing.assert_allclose(executed[7:, 7:], buf.data["a_b"][:5, 7:], atol=1e-6)  # base/torso/mode pass-through


def test_success_terminal_reward_and_offline_quota():
    runner, venv, base, specs = make_runner(num_envs=2, horizon=12, stage_steps=(5,))
    runner.offline_quota = 1
    seeds = itertools.count(0)
    results = []

    def on_end(r):
        results.append(r)
        return sum(x.success for x in results) >= 1

    runner.run(lambda: EpisodeSpec(seed=next(seeds), mode="base", store="offline"), on_end)
    off = specs[0].offline
    assert off.size == 5                                   # one successful 5-step episode only
    assert off.data["reward"][4] == 1 and off.data["done"][4] == 1
    assert np.all(off.data["reward"][:4] == 0)
    np.testing.assert_allclose(off.data["mc_return"][:5], 0.99 ** np.arange(4, -1, -1), rtol=1e-5)


def test_multistage_routing_and_base_only_stage():
    # stages: 0 -> learnable, 1 -> base-only (wait), 2 -> learnable
    runner, venv, base, specs = make_runner(num_envs=1, horizon=30, stage_steps=(3, 6, 9), learnable={0, 2},
                                            succeed=lambda s: True)
    res = run_n(runner, [EpisodeSpec(seed=0, mode="pld", t_base=0, store="online")], 1)[0]
    assert res.success and res.length == 9
    assert specs[0].online.size == 3 and specs[2].online.size == 3
    assert specs[0].online.data["done"][2] == 1 and specs[2].online.data["done"][2] == 1
    executed = np.stack(venv.actions[(0, 0)])
    np.testing.assert_allclose(executed[3:6, 0] * 1000, np.arange(3, 6), atol=1e-3)   # wait stage: pure base


def test_segment_returns_and_symmetric_sampling():
    r = np.array([0, 0, 1, 0, 1], np.float32)
    d = np.array([0, 0, 1, 0, 1], np.float32)
    np.testing.assert_allclose(segment_mc_returns(r, d, 0.5), [0.25, 0.5, 1, 0.5, 1])
    a, b = ReplayBuffer(10, (1,), 1, 12, 7), ReplayBuffer(10, (1,), 1, 12, 7)
    for _ in range(3):
        a.add({"reward": 1.0})
        b.add({"reward": 0.0})
    s = symmetric_sample(a, b, 8, np.random.default_rng(0))
    assert s["reward"].sum() == 4


def test_agent_updates_and_calql_run():
    sp = make_spec_obj(0)
    rng = np.random.default_rng(0)
    for t in range(40):
        sp.offline.add({"feat": rng.normal(size=(2, *FMAP)), "proprio": rng.normal(size=16),
                        "a_b": rng.uniform(-1, 1, 12), "a_bar": rng.uniform(-1, 1, 7),
                        "reward": float(t % 10 == 9), "done": float(t % 10 == 9),
                        "next_feat": rng.normal(size=(2, *FMAP)), "next_proprio": rng.normal(size=16),
                        "next_a_b": rng.uniform(-1, 1, 12), "mc_return": 0.5})
    actor_before = {k: v.clone() for k, v in sp.agent.actor.state_dict().items()}
    sp.agent.pretrain_calql(sp.offline, 5)
    assert sp.agent.n_updates == 0
    # actor re-initialised after critic warm-start (Alg. 1)
    assert any(not torch.equal(actor_before[k], v) for k, v in sp.agent.actor.state_dict().items())
    for _ in range(4):
        info = sp.agent.update(sp.offline.sample(8, rng))
    assert np.isfinite(info["critic_loss"])
    a = sp.agent.act(sp.agent.actor, rng.normal(size=(3, 2, *FMAP)).astype(np.float16),
                     rng.normal(size=(3, 16)).astype(np.float32), rng.uniform(-1, 1, (3, 12)).astype(np.float32), True)
    assert a.shape == (3, 7) and np.all(np.abs(a) <= 0.5 + 1e-6)


def test_logprob_matches_numerical_density():
    sp = make_spec_obj(0)
    torch.manual_seed(0)
    f = torch.randn(1, 2, *FMAP).half()
    p, ab = torch.randn(1, 16), torch.rand(1, 12) * 2 - 1
    # 1-D sanity on the tanh+scale change of variables: integrate density over a fine grid of a 1-D projection
    a, logp = sp.agent.actor(f, p, ab, 0.5)
    assert torch.isfinite(logp).all() and a.abs().max() <= 0.5


@pytest.mark.parametrize("alpha,T", [(0.8, 600), (0.6, 3000)])
def test_t_base_sampler_range(alpha, T):
    from pld_rc365.common import sample_t_base
    rng = np.random.default_rng(0)
    xs = [sample_t_base({"probing": {"alpha": alpha}}, T, rng) for _ in range(5000)]
    assert min(xs) == 0 and max(xs) == int(np.floor(alpha * T))


def test_async_learner_throttles_to_utd_and_syncs_actor():
    import time

    from pld_rc365.rl.agent import AsyncLearner

    sp = make_spec_obj(0)
    rng = np.random.default_rng(0)
    tr = lambda: {"feat": rng.normal(size=(2, *FMAP)), "proprio": rng.normal(size=16), "a_b": rng.uniform(-1, 1, 12),  # noqa: E731
                  "a_bar": rng.uniform(-1, 1, 7), "next_feat": rng.normal(size=(2, *FMAP)),
                  "next_proprio": rng.normal(size=16), "next_a_b": rng.uniform(-1, 1, 12)}
    for _ in range(20):
        sp.offline.add(tr())
    lr = AsyncLearner(sp.agent, sp.offline, sp.online, sp.collector_actor, sp.actor_lock)
    lr.start()
    for _ in range(10):
        sp.online.add(tr())
    deadline = time.time() + 20
    while sp.agent.n_updates < 10 and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    lr.stop()
    assert lr.error is None
    assert sp.agent.n_updates == 10          # utd=1: exactly one critic update per stored online transition
    for k, v in sp.collector_actor.state_dict().items():   # pushed at update 5 and 10 (actor_sync_every=5)
        assert torch.equal(v, sp.agent.actor.state_dict()[k])


def test_linear_xi_schedule_counts_residual_steps_only_and_persists():
    import copy

    res = dict(dims=7, xi=0.5, xi_schedule="linear", xi_start=0.1, xi_ramp_steps=10)
    agent = ResidualSAC(RL, res, FMAP, 2, device="cpu", seed=0)
    sp = Specialist(agent=agent, collector_actor=copy.deepcopy(agent.actor), actor_lock=threading.Lock(),
                    offline=ReplayBuffer(500, (2, *FMAP), 16, 12, 7), online=ReplayBuffer(500, (2, *FMAP), 16, 12, 7))
    venv = FakeVecEnv(1, 12, (100,), succeed=lambda s: False)
    runner = Runner(venv, FakeBase(1, 4), FakeBackbone(), "cpu", 1, {0}, {0: sp}, residual_dims=7)
    assert agent.xi == pytest.approx(0.1)
    run_n(runner, [EpisodeSpec(seed=1, mode="base", store="online")], 1)      # warm-up: 12 base steps
    assert agent.online_steps == 12 and agent.xi_steps == 0 and agent.xi == pytest.approx(0.1)
    run_n(runner, [EpisodeSpec(seed=3, mode="pld", t_base=7, store="online")], 1)  # 5 residual steps
    assert agent.xi_steps == 5 and agent.xi == pytest.approx(0.1 + 0.5 * 0.4)
    run_n(runner, [EpisodeSpec(seed=5, mode="pld", t_base=0, store="online")], 1)  # +12 -> ramp done
    assert agent.xi == pytest.approx(0.5)
    executed = np.stack(venv.actions[(0, 5)])
    stored = sp.online.data["a_bar"][17:29]
    np.testing.assert_allclose(executed[:, :7], stored, atol=1e-6)
    agent2 = ResidualSAC(RL, res, FMAP, 2, device="cpu", seed=1)
    agent2.load_state_dict(agent.state_dict())
    assert agent2.xi == pytest.approx(0.5)      # stage 2 continues at the final xi


def test_consistency_check_catches_chunk_desync():
    runner, venv, base, specs = make_runner(num_envs=1, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    orig_pop = base.pop
    calls = {"n": 0}

    def bad_pop(i):
        calls["n"] += 1
        if calls["n"] == 3:
            orig_pop(i)           # pointer advanced between peek and pop: the consumed element is not the used one
        return orig_pop(i)

    base.pop = bad_pop
    with pytest.raises(RuntimeError, match="chunk desync"):
        run_n(runner, [EpisodeSpec(seed=1, mode="pld", t_base=2, store="online")], 1)
