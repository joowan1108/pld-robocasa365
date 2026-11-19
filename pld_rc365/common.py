"""Construction helpers shared by the stage scripts."""

from __future__ import annotations

import copy
import os
import threading
from pathlib import Path

import numpy as np
import torch

from .base.client import BasePolicy
from .envs.rc365_env import VecEnv, make_spec
from .rl.agent import ResidualSAC
from .rl.buffers import ReplayBuffer
from .rl.networks import FrozenBackbone
from .rollout import Runner, Specialist


def learnable_stage_ids(cfg: dict) -> list[int]:
    names = cfg["task"]["stages"]
    base_only = set(cfg["task"].get("base_only_stages", []))
    return [k for k, n in enumerate(names) if n not in base_only]


def sample_t_base(cfg: dict, horizon: int, rng: np.random.Generator) -> int:
    """PLD base-policy probing: T_base ~ U{0, ..., floor(alpha * T)}, T = task horizon."""
    hi = int(np.floor(float(cfg["probing"]["alpha"]) * horizon))
    return int(rng.integers(0, hi + 1))


def build_env_and_base(cfg: dict, scene: str = "fixed", task: str | None = None, record_root: str | None = None,
                       num_envs: int | None = None, port: int | None = None, host: str | None = None):
    os.environ.setdefault("MUJOCO_GL", cfg["env"].get("mujoco_gl", "egl"))
    spec = make_spec(cfg, scene=scene, task=task)
    n = int(num_envs or cfg["env"]["num_envs"])
    egl = os.environ.get("PLD_EGL_DEVICES")
    venv = VecEnv(spec, n, record_root=record_root, gl=cfg["env"].get("mujoco_gl", "egl"),
                  egl_devices=egl.split(",") if egl else None)
    b = cfg["base"]
    base = BasePolicy(host or b["host"], int(port or b["port"]), n, b["exec_steps"], b["history_indices"])
    return venv, base, spec


def build_backbone(cfg: dict) -> tuple[FrozenBackbone, tuple, int]:
    dev = cfg["rl"]["device"]
    bb = FrozenBackbone().to(dev)
    n_cams = len(cfg["residual"]["cameras"])
    s = int(cfg["residual"]["image_size"])
    with torch.no_grad():
        f = bb(torch.zeros(1, n_cams, s, s, 3, dtype=torch.uint8, device=dev))
    return bb, tuple(f.shape[2:]), n_cams


def build_specialists(cfg: dict, fmap_shape: tuple, n_cams: int, stage_ids: list[int],
                      online_capacity: int | None = None, offline_capacity: int = 200_000) -> dict[int, Specialist]:
    rl, res = cfg["rl"], cfg["residual"]
    out = {}
    for k in stage_ids:
        agent = ResidualSAC(rl, res, fmap_shape, n_cams, device=rl["device"], seed=1000 + k)
        collector = copy.deepcopy(agent.actor).requires_grad_(False)
        cap = int(online_capacity or rl["buffer_capacity"])
        out[k] = Specialist(
            agent=agent,
            collector_actor=collector,
            actor_lock=threading.Lock(),
            offline=ReplayBuffer(offline_capacity, (n_cams, *fmap_shape), 16, 12, int(res["dims"])),
            online=ReplayBuffer(cap, (n_cams, *fmap_shape), 16, 12, int(res["dims"])),
        )
    return out


def sync_collectors(specs: dict[int, Specialist]):
    for sp in specs.values():
        with sp.actor_lock:
            sp.collector_actor.load_state_dict(sp.agent.actor.state_dict())


def save_specialists(specs: dict[int, Specialist], path: Path, extra: dict | None = None):
    torch.save({"specialists": {k: sp.agent.state_dict() for k, sp in specs.items()}, "extra": extra or {}}, path)


def load_specialists(cfg: dict, path: Path, fmap_shape: tuple, n_cams: int) -> dict[int, Specialist]:
    ck = torch.load(path, map_location=cfg["rl"]["device"], weights_only=False)
    ids = sorted(int(k) for k in ck["specialists"])
    specs = build_specialists(cfg, fmap_shape, n_cams, ids, online_capacity=1, offline_capacity=1)
    for k in ids:
        specs[k].agent.load_state_dict(ck["specialists"][k])
    sync_collectors(specs)
    return specs


def make_runner(cfg, venv, base, backbone, specs, num_stages: int) -> Runner:
    return Runner(venv, base, backbone, cfg["rl"]["device"], num_stages, set(learnable_stage_ids(cfg)),
                  specs, residual_dims=int(cfg["residual"]["dims"]), gamma=float(cfg["rl"]["discount"]))


def common_argparser(desc: str):
    import argparse

    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--model", required=True, choices=["rldx", "xr1"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides a.b=c")
    return ap
