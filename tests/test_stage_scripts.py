"""End-to-end orchestration of stage1_train / stage2_collect with fake env, base and backbone (no sim, CPU)."""

import json
import sys
from types import SimpleNamespace

import pytest

from .fakes import FakeBackbone, FakeBase, FakeVecEnv

# offline-cache key suffix for the default residual input (256 px, 3 cameras, 4 augmented views, pad 4)
KEY_SUFFIX = "_i256_c3_v4p4"
TINY = [
    "rl.device=cpu", "rl.hidden_dim=16", "rl.latent_dim=8", "rl.num_spatial_features=2", "rl.batch_size=8",
    "rl.buffer_capacity=2000", "rl.offline_successes=2", "rl.calql.steps=3", "rl.calql.num_samples=2",
    "rl.warmup_episodes=2", "rl.actor_sync_every=2", "env.num_envs=2",
    "stage1.sr_window=4", "stage1.stop_sr=0.0", "stage1.final_eval_episodes=3", "stage1.checkpoint_every_episodes=2",
    "eval.seed_base=0", "stage2.target_successes=3", "stage2.marks=[1,3]",
    "task.layout_id=11", "task.style_id=34",
]


@pytest.fixture
def patched(monkeypatch, tmp_path):
    def fake_build_env_and_base(cfg, scene="fixed", task=None, record_root=None, num_envs=None, port=None, host=None,
                                training=False):
        n = int(num_envs or cfg["env"]["num_envs"])
        venv = FakeVecEnv(n, horizon=12, stage_steps=(5,))
        venv.close = lambda: None
        spec = SimpleNamespace(stage_names=list(cfg["task"]["stages"]), layout_id=11, style_id=34,
                               task=cfg["task"]["env_name"])
        return venv, FakeBase(n, exec_steps=4), spec

    def fake_build_backbone(cfg):
        return FakeBackbone(), (4, 2, 2), 2

    import pld_rc365.stage1_train as s1
    import pld_rc365.stage2_collect as s2

    for mod in (s1, s2):
        monkeypatch.setattr(mod, "build_env_and_base", fake_build_env_and_base)
        monkeypatch.setattr(mod, "build_backbone", fake_build_backbone)
    return tmp_path


def test_stage1_then_stage2(patched, monkeypatch):
    import pld_rc365.stage1_train as s1
    import pld_rc365.stage2_collect as s2

    runs = patched / "runs"
    common = ["--model", "xr1", "--task", "CoffeeSetupMug", "--set", *TINY, f"paths.runs_root={runs}",
              f"paths.data_root={runs}/pld_data"]
    monkeypatch.setattr(sys, "argv", ["stage1", *common])
    s1.main()
    rd = runs / "xr1" / "CoffeeSetupMug" / "stage1"
    timing = json.loads((rd / "timing.json").read_text())
    for ph in ("offline_collect", "calql_pretrain", "warmup", "online", "final_eval"):
        assert ph in timing["phases"], ph
    assert timing["extra"]["offline"]["successes"] == 2
    assert timing["extra"]["online"]["stop_reason"].startswith("rolling_sr")
    assert "sr>=0.8_since_online_start" in timing["marks"] or timing["extra"]["online"]["thresholds_not_reached"]
    assert (rd / "specialists_final.pt").exists()
    phases = [json.loads(l)["phase"] for l in (rd / "episodes.jsonl").read_text().splitlines()]
    assert phases.count("warmup") == 2 and phases.count("final_eval") == 3

    monkeypatch.setattr(sys, "argv", ["stage2", *common, "--skip-write"])
    s2.main()
    rd2 = runs / "xr1" / "CoffeeSetupMug" / "stage2"
    t2 = json.loads((rd2 / "timing.json").read_text())
    assert t2["extra"]["collect"]["kept"] == 3
    assert "successes_1" in t2["marks"] and "successes_3" in t2["marks"]
    kept = json.loads((rd2 / "kept_episodes.json").read_text())
    assert len(kept) == 3 and all(k["length"] == 5 for k in kept)


def test_stage1_offline_buffer_saved_and_reused(patched, monkeypatch):
    import pld_rc365.stage1_train as s1

    runs = patched / "runs"
    common = ["--model", "xr1", "--task", "CoffeeSetupMug", "--skip-final-eval", "--set", *TINY,
              f"paths.runs_root={runs}"]
    monkeypatch.setattr(sys, "argv", ["stage1", *common])
    s1.main()
    cache = runs / "xr1" / "CoffeeSetupMug" / "offline_cache" / ("L11_S34_n2_e16" + KEY_SUFFIX)
    meta = json.loads((cache / "meta.json").read_text())
    assert (cache / "stage0.npz").exists() and meta["stats"]["successes"] == 2 and meta["gamma"] == 0.99
    first = json.loads((runs / "xr1" / "CoffeeSetupMug" / "stage1" / "timing.json").read_text())

    # second run (fresh stage1 dir) reuses the cache: no offline episodes, same transitions, reuse recorded
    (runs / "xr1" / "CoffeeSetupMug" / "stage1").rename(runs / "xr1" / "CoffeeSetupMug" / "stage1_first")
    monkeypatch.setattr(sys, "argv", ["stage1", "--model", "xr1", "--task", "CoffeeSetupMug", "--skip-final-eval",
                                      "--set", *TINY, f"paths.runs_root={runs}", "stage1.reuse_offline=true"])
    s1.main()
    rd = runs / "xr1" / "CoffeeSetupMug" / "stage1"
    t2 = json.loads((rd / "timing.json").read_text())
    phases = [json.loads(l)["phase"] for l in (rd / "episodes.jsonl").read_text().splitlines()]
    assert "offline" not in phases and "offline_load" in t2["phases"]
    assert t2["extra"]["offline"]["reused_from"] == str(cache)
    assert t2["extra"]["offline"]["offline_transitions"] == first["extra"]["offline"]["offline_transitions"]
    assert t2["phases"]["offline_collect"]["seconds"] == meta["stats"]["collect_seconds"]

    # incompatible settings must refuse the cache
    (rd).rename(runs / "xr1" / "CoffeeSetupMug" / "stage1_second")
    monkeypatch.setattr(sys, "argv", ["stage1", "--model", "xr1", "--task", "CoffeeSetupMug", "--skip-final-eval",
                                      "--set", *TINY, f"paths.runs_root={runs}", "stage1.reuse_offline=true",
                                      "rl.discount=0.98"])
    with pytest.raises(RuntimeError, match="incompatible"):
        s1.main()


def test_stage1_warmup_cache_saved_and_reused(patched, monkeypatch):
    import pld_rc365.stage1_train as s1

    runs = patched / "runs"
    base = ["stage1", "--model", "xr1", "--task", "CoffeeSetupMug", "--skip-final-eval", "--set", *TINY,
            f"paths.runs_root={runs}"]
    monkeypatch.setattr(sys, "argv", base)
    s1.main()
    cache = runs / "xr1" / "CoffeeSetupMug" / "offline_cache" / ("L11_S34_n2_e16" + KEY_SUFFIX)
    wdirs = sorted(cache.glob("warmup_*"))
    assert len(wdirs) == 1 and (wdirs[0] / "agent0.pt").exists() and (wdirs[0] / "online0.npz").exists()
    wmeta = json.loads((wdirs[0] / "meta.json").read_text())
    n_warm_eps = len((wdirs[0] / "warmup_episodes.jsonl").read_text().splitlines())
    assert n_warm_eps == 2 and wmeta["calql_seconds"] >= 0 and wmeta["warmup_seconds"] > 0

    rd = runs / "xr1" / "CoffeeSetupMug" / "stage1"
    rd.rename(rd.with_name("stage1_first"))
    # different xi -> same cache (xi is not part of the key); no Cal-QL / warm-up is run
    monkeypatch.setattr(sys, "argv", base + ["stage1.reuse_offline=true", "stage1.reuse_warmup=true",
                                             "residual.xi=0.05", "residual.xi_schedule=constant"])
    called = {"calql": 0}
    from pld_rc365.rl import agent as agent_mod
    real = agent_mod.ResidualSAC.pretrain_calql

    def spy(self, *a, **k):
        called["calql"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(agent_mod.ResidualSAC, "pretrain_calql", spy)
    s1.main()
    assert called["calql"] == 0
    timing = json.loads((rd / "timing.json").read_text())
    assert "warmup_load" in timing["phases"] and timing["extra"]["warmup_is_from_cache"]["actor"] == "keep"
    assert timing["phases"]["warmup"]["seconds"] == pytest.approx(wmeta["warmup_seconds"])
    eps = [json.loads(l) for l in (rd / "episodes.jsonl").read_text().splitlines()]
    assert sum(e["phase"] == "warmup" and e.get("from_cache") for e in eps) == n_warm_eps
    assert any(e["phase"] == "online" for e in eps)
    assert sorted(cache.glob("warmup_*")) == wdirs          # nothing new written

    # a change that matters (Cal-QL steps) -> different key -> recomputed and saved separately
    rd.rename(rd.with_name("stage1_second"))
    monkeypatch.setattr(sys, "argv", base + ["stage1.reuse_offline=true", "stage1.reuse_warmup=true",
                                             "rl.calql.steps=4"])
    s1.main()
    assert called["calql"] == 1 and len(sorted(cache.glob("warmup_*"))) == 2


def test_stage1_limits_can_be_changed_while_running(patched, monkeypatch):
    """control.json in the run dir overrides the step budget at the next episode end."""
    import pld_rc365.stage1_train as s1

    runs = patched / "runs"
    rd = runs / "xr1" / "CoffeeSetupMug" / "stage1"
    rd.mkdir(parents=True)
    (rd / "control.json").write_text(json.dumps({"max_online_env_steps": 60, "stop_sr": 2.0}))
    monkeypatch.setattr(sys, "argv", ["stage1", "--model", "xr1", "--task", "CoffeeSetupMug", "--skip-final-eval",
                                      "--set", *TINY, f"paths.runs_root={runs}", "stage1.max_online_env_steps=100000"])
    s1.main()
    timing = json.loads((rd / "timing.json").read_text())
    assert timing["extra"]["online"]["stop_reason"] == "max_online_env_steps"        # not rolling_sr (stop_sr=2.0)
    assert timing["extra"]["limits_override"]["max_online_env_steps"] == 60
    assert 60 <= timing["extra"]["online"]["online_steps"]["0"] < 200


def test_offline_cache_is_keyed_by_exec_steps(patched, monkeypatch):
    import pld_rc365.stage1_train as s1

    runs = patched / "runs"
    for ex in (16, 4):
        monkeypatch.setattr(sys, "argv", ["stage1", "--model", "xr1", "--task", "CoffeeSetupMug", "--skip-final-eval",
                                          "--set", *TINY, f"paths.runs_root={runs}", f"base.exec_steps={ex}",
                                          "stage1.reuse_offline=true"])
        s1.main()
        rd = runs / "xr1" / "CoffeeSetupMug" / "stage1"
        rd.rename(rd.with_name(f"stage1_e{ex}"))
    caches = sorted(p.name for p in (runs / "xr1" / "CoffeeSetupMug" / "offline_cache").iterdir())
    assert caches == ["L11_S34_n2_e16" + KEY_SUFFIX, "L11_S34_n2_e4" + KEY_SUFFIX]      # separate buffers, nothing overwritten


def test_stage1_fixed_initial_state_pool(patched, monkeypatch):
    import pld_rc365.stage1_train as s1

    runs = patched / "runs"
    monkeypatch.setattr(sys, "argv", ["stage1", "--model", "xr1", "--task", "CoffeeSetupMug", "--set", *TINY,
                                      f"paths.runs_root={runs}", "stage1.init_seed_pool=3"])
    s1.main()
    rd = runs / "xr1" / "CoffeeSetupMug" / "stage1"
    eps = [json.loads(l) for l in (rd / "episodes.jsonl").read_text().splitlines()]
    pool = {100000, 100001, 100002}
    assert {e["phase"] for e in eps} >= {"offline", "warmup", "online", "final_eval"}
    assert {e["seed"] for e in eps} <= pool and all(e.get("init") is not None for e in eps)
    assert json.loads((rd / "timing.json").read_text())["extra"]["init_seed_pool"] == sorted(pool)
    assert (runs / "xr1" / "CoffeeSetupMug" / "offline_cache" / ("L11_S34_n2_e16_p3" + KEY_SUFFIX) / "meta.json").exists()
