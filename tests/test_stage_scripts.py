"""End-to-end orchestration of stage1_train / stage2_collect with fake env, base and backbone (no sim, CPU)."""

import json
import sys
from types import SimpleNamespace

import pytest

from .fakes import FakeBackbone, FakeBase, FakeVecEnv

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
    def fake_build_env_and_base(cfg, scene="fixed", task=None, record_root=None, num_envs=None, port=None, host=None):
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
