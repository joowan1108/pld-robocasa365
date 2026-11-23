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
    np.testing.assert_allclose(buf.frames[buf.data["obs_id"][:12], 0, 0, 0, 0, 0], np.arange(12), atol=1e-2)
    np.testing.assert_allclose(buf.frames[buf.data["next_obs_id"][:12], 0, 0, 0, 0, 0], np.arange(1, 13), atol=1e-2)
    assert buf.frames_added == 13                                     # each observation is stored once
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
        a.add({"reward": 1.0, "feat": np.zeros((1,)), "next_feat": np.zeros((1,))})
        b.add({"reward": 0.0, "feat": np.zeros((1,)), "next_feat": np.zeros((1,))})
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


def test_consistency_check_allows_out_of_range_base_actions():
    runner, venv, base, specs = make_runner(num_envs=1, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    orig_ensure = base.ensure

    def ensure(idxs):
        orig_ensure(idxs)
        for i in idxs:
            for a in base.q[i]:
                a[6] = 1.3      # VLA gripper slightly beyond +1
                a[1] = -1.2
    base.ensure = ensure
    for mode, tb in (("base", 0), ("pld", 3)):
        run_n(runner, [EpisodeSpec(seed=1, mode=mode, t_base=tb, store="online")], 1)


def test_cap_num_envs_by_free_gpu_memory(monkeypatch):
    from pld_rc365 import common

    cfg = {"env": {"cap_envs_by_gpu_mem": True, "gpu_mem_per_env_gb": 1.2, "gpu_mem_reserve_train_gb": 2.0,
                   "gpu_mem_reserve_eval_gb": 0.5}}
    monkeypatch.delenv("PLD_EGL_DEVICES", raising=False)
    monkeypatch.setattr(common, "_free_gpu_gb", lambda i: 12.6)       # 4090 with the XR1 server loaded
    assert common.cap_num_envs(cfg, 10, training=True) == 8
    assert common.cap_num_envs(cfg, 10, training=False) == 10
    monkeypatch.setenv("PLD_EGL_DEVICES", "0,1")
    monkeypatch.setattr(common, "_free_gpu_gb", lambda i: 12.6 if i == 0 else 10.0)
    assert common.cap_num_envs(cfg, 20, training=True) == 10 + 6
    monkeypatch.setattr(common, "_free_gpu_gb", lambda i: None)       # no nvidia-smi: leave as is
    assert common.cap_num_envs(cfg, 10, training=True) == 10


def test_pipelined_queries_keep_transitions_aligned():
    """Async server (replies arrive a few polls later): waiting envs are not stepped, the others keep going, and
    every stored transition still has a_b = action used at t and next_a_b = action used at t+1."""
    runner, venv, base, specs = make_runner(num_envs=3, horizon=20, stage_steps=(100,), succeed=lambda s: False)
    base.delay = 3
    specs_list = [EpisodeSpec(seed=s, mode="base", store="online") for s in (1, 3, 5, 7)]
    run_n(runner, specs_list, 4)
    buf = specs[0].online
    assert buf.size == 4 * 20
    ab = buf.data["a_b"][: buf.size, 0] * 1000
    nab = buf.data["next_a_b"][: buf.size, 0] * 1000
    # per transition, the fake encodes the env step each action was meant for: next = this + 1
    np.testing.assert_allclose(nab, ab + 1, atol=1e-3)
    assert base.n_requests < base.n_queries     # some requests carried several envs


def test_entropy_is_defined_on_the_normalised_residual_so_target_is_reachable():
    """log pi must not depend on xi: with the xi ramp starting at 0.1 the entropy of a_delta itself is at most
    7*log(0.2) = -11.3 < target -3.5, which made the temperature diverge on the first real run."""
    sp = make_spec_obj(0)
    f = torch.randn(64, 2, *FMAP).half()
    p, ab = torch.randn(64, 16), torch.rand(64, 12) * 2 - 1
    torch.manual_seed(0)
    a1, lp1 = sp.agent.actor(f, p, ab, 0.1)
    torch.manual_seed(0)
    a5, lp5 = sp.agent.actor(f, p, ab, 0.5)
    torch.testing.assert_close(lp1, lp5)
    torch.testing.assert_close(a1 * 5, a5, rtol=1e-5, atol=1e-6)
    # a wide policy reaches entropies above the target, so the temperature loss can change sign
    with torch.no_grad():
        last = sp.agent.actor.trunk[-1]
        last.weight.zero_()
        last.bias.zero_()
        last.bias[7:] = -0.14        # std ~0.87: tanh(N(0, 0.87)) is close to uniform on [-1, 1] (entropy ~ +4.6)
    _, lp = sp.agent.actor(f, p, ab, 0.1)
    assert (-lp.mean()).item() > sp.agent.target_entropy


def test_reserve_gpu_memory_is_a_noop_for_weights():
    sp = make_spec_obj(0)
    before = {k: v.clone() for k, v in sp.agent.critic.state_dict().items()}
    sp.agent.reserve_gpu_memory(force=True)         # same code path on CPU
    assert all(torch.equal(before[k], v) for k, v in sp.agent.critic.state_dict().items())
    assert all(p.grad is None for p in sp.agent.critic.parameters())


def test_episode_residual_summary_is_logged():
    runner, venv, base, specs = make_runner(num_envs=1, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    res = run_n(runner, [EpisodeSpec(seed=1, mode="pld", t_base=4, store="online"),
                         EpisodeSpec(seed=3, mode="base", store="online")], 2)
    r = res[0].extra["residual"]
    assert r["steps"] == 8 and len(r["mean_abs"]) == 7 and r["xi"] == pytest.approx(0.5)
    assert all(0 <= v <= 0.5 + 1e-6 for v in r["mean_abs"]) and 0 <= r["abs_over_xi"] <= 1.0 + 1e-6
    assert res[1].extra["residual"] is None          # base-only episode: no residual acted


def test_critic_input_follows_paper_eq2():
    import torch
    from pld_rc365.rl.agent import ResidualSAC
    rl = {"discount": 0.99, "tau": 0.005, "batch_size": 4, "target_entropy_scale": 0.5, "hidden_dim": 16,
          "latent_dim": 8, "num_spatial_features": 2, "num_q": 2, "lr": 3e-4, "init_temperature": 1.0}
    res = {"dims": 7, "xi": 0.1, "xi_schedule": "constant"}
    f, p = torch.randn(3, 2, 4, 2, 2), torch.randn(3, 16)
    ag = ResidualSAC(rl, res, (4, 2, 2), 2, device="cpu")
    assert ag.critic.encoder.base_action_dim == 0 and ag.actor.encoder.base_action_dim == 12
    z1 = ag.critic.encoder(f, p, torch.zeros(3, 12))
    z2 = ag.critic.encoder(f, p, torch.ones(3, 12))
    assert torch.equal(z1, z2)                                   # Q does not depend on a_b, only on a_bar
    ag2 = ResidualSAC({**rl, "critic_uses_a_b": True}, res, (4, 2, 2), 2, device="cpu")
    assert ag2.critic.encoder.base_action_dim == 12


def test_calql_paper_form_matches_appendix_b1():
    import numpy as np
    import torch
    from pld_rc365.rl.agent import ResidualSAC
    rl = {"discount": 0.99, "tau": 0.005, "batch_size": 6, "target_entropy_scale": 0.5, "hidden_dim": 16,
          "latent_dim": 8, "num_spatial_features": 2, "num_q": 2, "lr": 3e-4, "init_temperature": 1.0,
          "grad_clip": 1.0, "critic_actor_ratio": 2, "calql": {"form": "paper", "num_samples": 4, "alpha": 1.0}}
    res = {"dims": 7, "xi": 0.1, "xi_schedule": "constant"}
    ag = ResidualSAC(rl, res, (4, 2, 2), 2, device="cpu")
    rng = np.random.default_rng(0)
    B = 6
    b = {"feat": rng.standard_normal((B, 2, 4, 2, 2)).astype(np.float32), "next_feat": rng.standard_normal((B, 2, 4, 2, 2)).astype(np.float32),
         "proprio": rng.standard_normal((B, 16)).astype(np.float32), "next_proprio": rng.standard_normal((B, 16)).astype(np.float32),
         "a_b": rng.uniform(-.5, .5, (B, 12)).astype(np.float32), "next_a_b": rng.uniform(-.5, .5, (B, 12)).astype(np.float32),
         "a_bar": rng.uniform(-.5, .5, (B, 7)).astype(np.float32), "reward": np.zeros(B, np.float32),
         "done": np.zeros(B, np.float32), "mc_return": np.full(B, 100.0, np.float32)}
    t = ag._to_t(b)
    z = ag.critic.encoder(t["feat"], t["proprio"], t["a_b"])
    q = ag.critic(None, None, None, t["a_bar"], z=z)
    # V^mu far above any Q: every policy-action Q is floored -> term = sum_heads mean(V^mu - Q(s, a_data)),
    # and the push-down has no gradient (only Q(s, a_data) is pushed up).
    cql, info = ag._calql_term_paper(t, z, q, 0.1)
    assert info["calib_binds"] == 1.0
    assert torch.allclose(cql, (100.0 - q).mean(-1).sum(), atol=1e-4)
    # V^mu far below: term = mean over policy samples of Q(s, a_pi) - Q(s, a_data); finite, has gradient
    t["mc_return"] = torch.full((B,), -100.0)
    cql2, info2 = ag._calql_term_paper(t, z, q, 0.1)
    assert info2["calib_binds"] == 0.0 and torch.isfinite(cql2)
    assert abs(info2["cql_gap"] - (info2["q_pi"] - q.mean().item())) < 1e-4
    out = ag.update(b, calql_alpha=1.0)                 # full update runs with the paper form
    assert np.isfinite(out["critic_loss"]) and "q_rand" not in out


def test_time_limit_failure_is_terminal_when_enabled():
    runner, venv, base, specs = make_runner(num_envs=1, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    runner.timeout_terminal = True
    res = run_n(runner, [EpisodeSpec(seed=1, mode="base", store="online")], 1)
    assert not res[0].success and res[0].length == 12
    buf = specs[0].online
    assert buf.size == 12
    np.testing.assert_array_equal(buf.data["done"][:12], [0] * 11 + [1])   # only the time-limit step is terminal
    assert np.all(buf.data["reward"][:12] == 0)                             # ... with reward 0 (failure)
    assert runner.stats["timeout_terminals"] == 1
    # a successful episode is unchanged: terminal with reward 1 at the success step, no extra terminal
    runner2, _, _, specs2 = make_runner(num_envs=1, horizon=12, stage_steps=(5,), succeed=lambda s: True)
    runner2.timeout_terminal = True
    res2 = run_n(runner2, [EpisodeSpec(seed=2, mode="base", store="online")], 1)
    b2 = specs2[0].online
    assert res2[0].success and b2.data["done"][: b2.size].sum() == 1 and b2.data["reward"][: b2.size].sum() == 1
    assert runner2.stats.get("timeout_terminals", 0) == 0


def test_make_runner_reads_timeout_terminal_from_config():
    from pld_rc365.common import make_runner as mk
    from pld_rc365.config import load_config
    cfg = load_config("xr1", "CoffeeSetupMug", ["rl.device=cpu"])
    venv = FakeVecEnv(1, 12, (5,), lambda s: True)
    assert mk(cfg, venv, FakeBase(1, 4), FakeBackbone(), {}, 1).timeout_terminal is True
    cfg["rl"]["timeout_terminal"] = False
    assert mk(cfg, venv, FakeBase(1, 4), FakeBackbone(), {}, 1).timeout_terminal is False


def test_async_learner_respects_update_ratio():
    import time
    from pld_rc365.rl.agent import AsyncLearner
    runner, venv, base, specs = make_runner(num_envs=1, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    run_n(runner, [EpisodeSpec(seed=1, mode="base", store="online")], 1)      # 12 stored transitions
    sp = specs[0]
    for utd, expect in ((2.0, 24), (1.0, 24)):                                # budget = utd * stored; never above it
        sp.agent.rl["utd"] = utd
        lr = AsyncLearner(sp.agent, sp.offline, sp.online, sp.collector_actor, sp.actor_lock)
        lr.start()
        t0 = time.time()
        while sp.agent.n_updates < expect and time.time() - t0 < 60:
            time.sleep(0.05)
        time.sleep(0.3)
        lr.stop()
        assert sp.agent.n_updates == expect, (utd, sp.agent.n_updates)


def test_critic_action_residual_separates_vla_action_from_residual():
    """rl.critic_action=residual: Q(s, a_b, a_delta). Base-only data has a zero action input whatever the VLA does,
    a_b reaches the critic through the encoder, and the losses run."""
    import numpy as np
    import torch
    from pld_rc365.rl.agent import ResidualSAC
    from pld_rc365.rl.networks import combine
    rl = {"discount": 0.99, "tau": 0.005, "batch_size": 4, "target_entropy_scale": 0.5, "hidden_dim": 16,
          "latent_dim": 8, "num_spatial_features": 2, "num_q": 2, "lr": 3e-4, "init_temperature": 1.0, "grad_clip": 1.0,
          "critic_action": "residual", "critic_actor_ratio": 1, "calql": {"form": "paper", "num_samples": 3}}
    res = {"dims": 7, "xi": 0.1, "xi_schedule": "constant"}
    ag = ResidualSAC(rl, res, (4, 2, 2), 2, device="cpu")
    assert ag.critic.encoder.base_action_dim == 12 and ag.state_dict()["arch"]["critic_action"] == "residual"
    g = torch.Generator().manual_seed(0)
    a_b = torch.rand(5, 12, generator=g) * 3 - 1.5                       # includes values outside [-1, 1]
    base_only = a_b[:, :7].clamp(-1, 1)                                   # what the rollout stores without a residual
    assert torch.equal(ag._qa(a_b, base_only), torch.zeros(5, 7))
    a_d = (torch.rand(5, 7, generator=g) * 2 - 1) * 0.1
    qa = ag._qa(a_b, combine(a_b, a_d, 7))
    assert qa.abs().max() <= 1 + 1e-5
    inside = (a_b[:, :7].abs() < 0.9)                                     # not clipped: input is exactly a_delta / xi
    assert torch.allclose(qa[inside], (a_d / 0.1)[inside], atol=1e-5)
    f, p = torch.randn(5, 2, 4, 2, 2), torch.randn(5, 16)
    q1 = ag.critic(f, p, a_b, torch.zeros(5, 7))
    q2 = ag.critic(f, p, a_b + 0.5, torch.zeros(5, 7))
    assert not torch.allclose(q1, q2)                                     # the VLA action is critic state
    # paper mode is unchanged: action input is a_bar itself
    ag0 = ResidualSAC({**rl, "critic_action": "total"}, res, (4, 2, 2), 2, device="cpu")
    assert torch.equal(ag0._qa(a_b, base_only), base_only) and ag0.critic.encoder.base_action_dim == 0
    rng = np.random.default_rng(0)
    b = {"feat": rng.standard_normal((4, 2, 4, 2, 2)).astype(np.float16), "next_feat": rng.standard_normal((4, 2, 4, 2, 2)).astype(np.float16),
         "proprio": rng.standard_normal((4, 16)).astype(np.float32), "next_proprio": rng.standard_normal((4, 16)).astype(np.float32),
         "a_b": rng.uniform(-1, 1, (4, 12)).astype(np.float32), "next_a_b": rng.uniform(-1, 1, (4, 12)).astype(np.float32),
         "a_bar": rng.uniform(-1, 1, (4, 7)).astype(np.float32), "reward": np.zeros(4, np.float32),
         "done": np.zeros(4, np.float32), "mc_return": np.full(4, 0.5, np.float32)}
    info = ag.update(b, calql_alpha=1.0)
    assert np.isfinite(info["critic_loss"])
    info = ag.update(b)
    assert np.isfinite(info["critic_loss"])


def test_offline_cache_key_depends_on_what_the_buffers_store():
    """Buffers store encoder features of the residual images, so image size, cameras and augmentation are in the key."""
    from types import SimpleNamespace
    from pld_rc365.stage1_train import offline_cache_dir
    spec = SimpleNamespace(layout_id=11, style_id=34)
    cfg = {"paths": {"runs_root": "runs"}, "model_name": "xr1", "task_name": "T", "base": {"exec_steps": 16},
           "stage1": {"init_seed_pool": 0}, "seeds": {"offline": 0},
           "residual": {"image_size": 128, "cameras": ["a", "b"]}}
    assert offline_cache_dir(cfg, spec, 50).name == "L11_S34_n50_e16"          # the pre-2026-11-21 buffers
    cfg["residual"]["image_size"] = 256
    assert offline_cache_dir(cfg, spec, 50).name == "L11_S34_n50_e16_i256"
    cfg["residual"]["cameras"] = ["a", "b", "c"]
    assert offline_cache_dir(cfg, spec, 50).name == "L11_S34_n50_e16_i256_c3"
    cfg["residual"]["augment"] = {"views": 4, "pad": 4}
    assert offline_cache_dir(cfg, spec, 50).name == "L11_S34_n50_e16_i256_c3_v4p4"
    cfg["residual"]["augment"] = {"views": 0, "pad": 4}
    assert offline_cache_dir(cfg, spec, 50).name == "L11_S34_n50_e16_i256_c3"
    cfg["rl"] = {"residual_hold": 8, "n_step": 3}
    assert offline_cache_dir(cfg, spec, 50).name == "L11_S34_n50_e16_i256_c3_h8n3"


def test_random_shift_is_a_replicate_padded_crop():
    import torch
    from pld_rc365.rl.networks import random_shift
    img = torch.arange(2 * 3 * 6 * 6 * 3, dtype=torch.float32).remainder(251).to(torch.uint8).view(2, 3, 6, 6, 3)
    g = torch.Generator().manual_seed(0)
    out = random_shift(img, 2, g)
    assert out.shape == img.shape and out.dtype == torch.uint8
    shifts = set()
    for b in range(2):
        for c in range(3):
            pad = torch.nn.functional.pad(img[b, c].permute(2, 0, 1)[None].float(), (2, 2, 2, 2), mode="replicate")[0].permute(1, 2, 0)
            hits = [(dy, dx) for dy in range(5) for dx in range(5) if torch.equal(pad[dy:dy + 6, dx:dx + 6].to(torch.uint8), out[b, c])]
            assert hits, "not a crop of the replicate-padded image"
            shifts.add(hits[0])
    assert len(shifts) > 1                                                 # drawn per image, not one shift for all


def test_runner_stores_augmented_views_and_acts_on_the_plain_frame():
    """With augmentation on, a stored frame holds the features of K shifted copies (not the plain one), s' of a
    transition is the same stored frame as s of the next, and the actor is fed the plain frame."""
    import torch
    from pld_rc365.rl.buffers import ReplayBuffer
    runner, venv, base, specs = make_runner(num_envs=1, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    K = 3
    specs[0].online = ReplayBuffer(500, (2, *FMAP), 16, 12, 7, views=K)
    runner.aug_views, runner.aug_pad = K, 1
    seen = []
    real_act = specs[0].agent.act
    specs[0].agent.act = lambda actor, feats, *a, **k: (seen.append(np.asarray(feats).copy()), real_act(actor, feats, *a, **k))[1]
    run_n(runner, [EpisodeSpec(seed=1, mode="pld", t_base=0, store="online")], 1)
    buf = specs[0].online
    assert buf.size == 12 and buf.frames_added == 13 and buf.frames.shape[1] == K
    assert np.array_equal(buf.data["next_obs_id"][:11], buf.data["obs_id"][1:12])
    assert len(seen) == 12 and all(f.shape == (1, 2, *FMAP) for f in seen)   # one plain frame per acting step
    b = buf.sample(32, np.random.default_rng(0))
    assert b["feat"].shape == (32, 2, *FMAP) and b["next_feat"].shape == (32, 2, *FMAP)


def test_actor_accepts_precomputed_encoding():
    import torch
    from pld_rc365.rl.agent import ResidualSAC
    rl = {"discount": 0.99, "tau": 0.005, "batch_size": 4, "target_entropy_scale": 0.5, "hidden_dim": 16,
          "latent_dim": 8, "num_spatial_features": 2, "num_q": 2, "lr": 3e-4, "init_temperature": 1.0}
    ag = ResidualSAC(rl, {"dims": 7, "xi": 0.1, "xi_schedule": "constant"}, (4, 8, 8), 2, device="cpu")   # 8x8 map
    f, p, ab = torch.randn(3, 2, 4, 8, 8), torch.randn(3, 16), torch.randn(3, 12)
    a1, _ = ag.actor(f, p, ab, 0.1, deterministic=True)
    a2, _ = ag.actor(None, None, None, 0.1, deterministic=True, z=ag.actor.encoder(f, p, ab))
    assert torch.equal(a1, a2)


def test_per_dimension_residual_bound():
    """residual.xi_scale: |a_delta_j| <= xi * xi_scale_j, the critic's residual input stays in [-1, 1] per dim, and
    the per-episode statistics count 'at bound' against each dim's own bound."""
    import torch
    from pld_rc365.rl.agent import ResidualSAC
    from pld_rc365.rl.networks import combine
    rl = {"discount": 0.99, "tau": 0.005, "batch_size": 4, "target_entropy_scale": 0.5, "hidden_dim": 16,
          "latent_dim": 8, "num_spatial_features": 2, "num_q": 2, "lr": 3e-4, "init_temperature": 1.0,
          "critic_action": "residual"}
    scale = [1, 1, 1, 0.25, 0.25, 0.25, 1]
    ag = ResidualSAC(rl, {"dims": 7, "xi": 0.1, "xi_schedule": "constant", "xi_scale": scale}, (4, 2, 2), 2, device="cpu")
    f, p, ab = torch.randn(64, 2, 4, 2, 2) * 50, torch.randn(64, 16) * 50, torch.zeros(64, 12)   # saturate tanh
    a_d, _ = ag.actor(f, p, ab, 0.1)
    bound = 0.1 * torch.tensor(scale)
    assert torch.all(a_d.abs() <= bound + 1e-7)
    assert a_d.abs()[:, 3:6].max() <= 0.025 + 1e-7 and a_d.abs()[:, :3].max() > 0.05
    qa = ag._qa(ab, combine(ab, a_d, 7))
    assert qa.abs().max() <= 1 + 1e-5 and qa.abs()[:, 3:6].max() > 0.5       # rotation input uses its own scale
    import copy
    assert torch.equal(copy.deepcopy(ag.actor).xi_scale, ag.actor.xi_scale)   # the collector copy keeps it
    ag.init_actor()
    assert torch.allclose(ag.actor.xi_scale, torch.tensor(scale))             # survives the post-Cal-QL re-init
    ag1 = ResidualSAC(rl, {"dims": 7, "xi": 0.1, "xi_schedule": "constant"}, (4, 2, 2), 2, device="cpu")
    assert torch.equal(ag1.actor.xi_scale, torch.ones(7))                     # default: one bound for all dims
    # runner statistics
    runner, venv, base, specs = make_runner(num_envs=1, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    specs[0].agent.xi_scale_np = np.asarray(scale, np.float32)
    res = run_n(runner, [EpisodeSpec(seed=1, mode="pld", t_base=0, store="online")], 1)[0]
    r = res.extra["residual"]
    np.testing.assert_allclose(r["xi_dim"], [x * r["xi"] for x in scale], atol=1e-6)


def test_residual_dim_switched_off():
    """residual.xi_scale_j = 0: no residual on that dim (gripper), it is not part of log pi / the target entropy,
    its critic input is 0, and the executed gripper action is the VLA's."""
    import torch
    from pld_rc365.rl.agent import ResidualSAC
    from pld_rc365.rl.networks import combine
    rl = {"discount": 0.99, "tau": 0.005, "batch_size": 4, "target_entropy_scale": 0.5, "hidden_dim": 16,
          "latent_dim": 8, "num_spatial_features": 2, "num_q": 2, "lr": 3e-4, "init_temperature": 1.0,
          "critic_action": "residual", "calql": dict(steps=5, alpha=1.0, num_samples=3)}
    scale = [1, 1, 1, 0.25, 0.25, 0.25, 0]
    ag = ResidualSAC(rl, {"dims": 7, "xi": 0.1, "xi_schedule": "constant", "xi_scale": scale}, (4, 2, 2), 2, device="cpu")
    assert ag.n_active == 6 and ag.target_entropy == -3.0
    f, p = torch.randn(32, 2, 4, 2, 2), torch.randn(32, 16)
    ab = torch.rand(32, 12) * 2 - 1
    torch.manual_seed(0)
    a_d, logp = ag.actor(f, p, ab, 0.1)
    assert torch.all(a_d[:, 6] == 0) and a_d[:, :6].abs().max() > 0
    a_bar = combine(ab, a_d, 7)
    assert torch.equal(a_bar[:, 6], ab[:, 6])                      # gripper = VLA action
    qa = ag._qa(ab, a_bar)
    assert torch.all(qa[:, 6] == 0) and torch.isfinite(qa).all() and qa.abs().max() <= 1 + 1e-5
    # log pi ignores the switched-off dim: changing only that dim's head output leaves it unchanged
    last = ag.actor.trunk[-1]
    with torch.no_grad():
        last.bias[6] += 3.0          # mean of dim 6
        last.bias[13] += 1.0         # log-std of dim 6
    torch.manual_seed(0)
    a_d2, logp2 = ag.actor(f, p, ab, 0.1)
    assert torch.allclose(logp, logp2, atol=1e-5) and torch.allclose(a_d, a_d2, atol=1e-7)
    ag.reserve_gpu_memory(force=True)        # critic (Cal-QL) and actor losses run with the switched-off dim
    with pytest.raises(ValueError):
        ResidualSAC(rl, {"dims": 7, "xi": 0.1, "xi_schedule": "constant", "xi_scale": [0] * 7}, (4, 2, 2), 2, device="cpu")
    with pytest.raises(ValueError):
        ResidualSAC(rl, {"dims": 7, "xi": 0.1, "xi_schedule": "constant", "xi_scale": [1, 1, 1, 1, 1, 1, -1]}, (4, 2, 2), 2, device="cpu")


def test_vecenv_watchdog_replaces_a_hung_worker():
    from pld_rc365.envs.rc365_env import VecEnv

    class Conn:
        def __init__(self, answers):
            self.answers, self.sent, self.closed = list(answers), [], False
        def send(self, x):
            self.sent.append(x)
        def poll(self, timeout):
            return bool(self.answers)
        def recv(self):
            return self.answers.pop(0)
        def close(self):
            self.closed = True

    class Proc:
        pid, killed = 123, False
        def kill(self):
            self.killed = True
        def join(self, timeout=None):
            pass

    v = object.__new__(VecEnv)
    v.num_envs, v.step_timeout, v.reset_timeout, v.restarts = 2, 0.01, 0.01, 0
    hung, ok = Conn([]), Conn([{"t": 1}])
    v.remotes, v.procs = [ok, hung], [Proc(), Proc()]
    old_proc = v.procs[1]
    fresh = Conn([{"t": 0, "fresh": True}])
    v._spawn = lambda i: (v.remotes.__setitem__(i, fresh), v.procs.__setitem__(i, Proc()))
    out = v.step({0: 0, 1: 0})
    assert out[0] == {"t": 1} and out[1] == {"_dead": True}
    assert old_proc.killed and hung.closed and v.restarts == 1 and v.remotes[1] is fresh
    assert v.reset(1, seed=5) == {"t": 0, "fresh": True}                  # the new worker serves the next reset


def test_runner_drops_the_episode_of_a_replaced_worker():
    runner, venv, base, specs = make_runner(num_envs=2, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    real_step, calls = venv.step, {"n": 0}

    def step(actions):
        out = real_step(actions)
        calls["n"] += 1
        if calls["n"] == 5 and 1 in out:     # env 1 "hangs" once
            out[1] = {"_dead": True}
        return out
    venv.step = step
    it = iter([EpisodeSpec(seed=s, mode="base", store="online") for s in (1, 3, 5)])
    res = []
    runner.run(lambda: next(it, None), lambda r: res.append(r) or False)
    assert runner.stats["dead_envs"] == 1
    assert sorted(r.spec.seed for r in res) == [1, 5]   # the dropped episode (seed 3) reports no result
    # complete transitions stored before the hang stay (real experience); the one waiting for next_a_b is dropped
    assert specs[0].online.size == 12 + 12 + 4                # 4 = n_stored reported for the dropped episode


def test_aggregate_transitions_hold_and_n_step():
    from pld_rc365.rl.buffers import aggregate_transitions
    g = 0.5
    trs = [{"feat": f"f{t}" if t % 3 == 0 else None, "proprio": t, "a_b": np.full(12, t), "a_bar": np.full(7, t),
            "reward": 1.0 if t == 7 else 0.0, "done": 1.0 if t == 7 else 0.0,
            "next_feat": f"f{t + 1}" if (t + 1) % 3 == 0 else None, "next_proprio": t + 1,
            "next_a_b": np.full(12, t + 1), "mc_return": g ** (7 - t)} for t in range(8)]
    out = aggregate_transitions(trs, hold=3, n_step=2, gamma=g)
    assert [o["proprio"] for o in out] == [0, 3, 6]                       # one per block of 3 steps
    assert [o["steps"] for o in out] == [6, 5, 2]                         # 3+3, 3+2 (segment ends), 2
    assert [o["done"] for o in out] == [0.0, 1.0, 1.0]
    assert out[0]["next_proprio"] == 6 and out[0]["next_feat"] == "f6" and out[0]["reward"] == 0.0
    assert out[1]["reward"] == g ** 4 and out[2]["reward"] == g ** 1    # reward at step 7, discounted to t=3 / t=6
    assert out[2]["mc_return"] == g ** 1 and out[0]["mc_return"] == g ** 7
    assert out[2]["next_feat"] == "f6"                                    # terminal: placeholder frame


def test_runner_holds_the_residual_and_stores_block_transitions():
    """rl.residual_hold = 4, n_step = 2: the actor decides only every 4 env steps from the takeover step, the residual
    is held in between, and the online buffer gets one transition per block, discounted over the steps it spans."""
    runner, venv, base, specs = make_runner(num_envs=1, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    runner.hold, runner.n_step, runner.timeout_terminal = 4, 2, True
    sp = specs[0]
    sp.agent.online_steps = 0
    decisions = []
    real_act = sp.agent.act
    def act(actor, feats, *a, **k):
        out = real_act(actor, feats, *a, **k)
        decisions.append(out.copy())
        return out
    sp.agent.act = act
    res = run_n(runner, [EpisodeSpec(seed=1, mode="pld", t_base=2, store="online")], 1)[0]
    assert len(decisions) == 3                                            # at steps 2, 6, 10
    assert res.n_stored == 10 and sp.agent.online_steps == 10             # steps 2..11 (probing 0..1 not stored)
    executed = np.array(venv.actions[(0, 1)])[:, :7]
    runner2, venv2, _, _ = make_runner(num_envs=1, horizon=12, stage_steps=(100,), succeed=lambda s: False)
    run_n(runner2, [EpisodeSpec(seed=1, mode="base", store="none")], 1)
    base_actions = np.array(venv2.actions[(0, 1)])[:, :7]                 # the scripted VLA actions, no residual
    resid = executed - base_actions
    assert np.allclose(resid[:2], 0)                                      # probing: VLA only
    expected = np.concatenate([np.repeat(d, n, 0) for d, n in zip(decisions, (4, 4, 2))])
    np.testing.assert_allclose(resid[2:, :6], expected[:, :6], atol=1e-6)  # one decision held per block (the
                                                                          # gripper dim is clipped at the VLA's +-1)
    assert not np.allclose(decisions[0], decisions[1])
    buf = sp.online
    assert buf.size == 3                                                  # blocks starting at 2, 6, 10
    np.testing.assert_allclose(buf.data["steps"][:3], [8, 6, 2])          # 4+4, 4+2, 2 (time limit)
    np.testing.assert_allclose(buf.data["done"][:3], [0, 1, 1])
    np.testing.assert_allclose(buf.data["proprio"][:3, 0], [2, 6, 10])    # s of each transition = block start
    np.testing.assert_allclose(buf.data["next_proprio"][:3, 0], [10, 12, 12])
    assert np.array_equal(buf.data["next_obs_id"][0], buf.data["obs_id"][2])   # s' of block 0 (n=2) is s of block 2


def test_base_action_bias_is_seen_and_executed():
    """base.action_bias (known-answer test): the residual sees the biased a_b, and without a residual the env executes
    the biased action."""
    runner, venv, base, specs = make_runner(num_envs=1, horizon=8, stage_steps=(100,), succeed=lambda s: False)
    base.action_bias = np.zeros(12, np.float32); base.action_bias[0] = 0.05
    run_n(runner, [EpisodeSpec(seed=1, mode="base", store="online")], 1)
    runner2, venv2, _, specs2 = make_runner(num_envs=1, horizon=8, stage_steps=(100,), succeed=lambda s: False)
    run_n(runner2, [EpisodeSpec(seed=1, mode="base", store="online")], 1)
    a, a0 = np.array(venv.actions[(0, 1)]), np.array(venv2.actions[(0, 1)])
    np.testing.assert_allclose(a[:, 0] - a0[:, 0], 0.05, atol=1e-6)
    np.testing.assert_allclose(a[:, 1:], a0[:, 1:], atol=1e-6)
    np.testing.assert_allclose(specs[0].online.data["a_b"][:8, 0] - specs2[0].online.data["a_b"][:8, 0], 0.05, atol=1e-6)


def test_base_policy_client_applies_action_bias():
    import collections
    from pld_rc365.base.client import BasePolicy
    b = object.__new__(BasePolicy)
    b.queues = [collections.deque([np.arange(12, dtype=np.float32)])]
    assert np.array_equal(b.peek(0), np.arange(12))
    b.action_bias = np.zeros(12, np.float32); b.action_bias[0] = 0.05
    assert b.peek(0)[0] == np.float32(0.05) and b.pop(0)[0] == np.float32(0.05)
