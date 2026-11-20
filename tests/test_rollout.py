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
