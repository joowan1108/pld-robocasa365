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


def _free_gpu_gb(index: int) -> float | None:
    """Free memory on a GPU via nvidia-smi (no CUDA context is created in this process)."""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(index)],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return float(out.splitlines()[0]) / 1024.0
    except Exception:  # noqa: BLE001
        return None


def cap_num_envs(cfg: dict, n: int, training: bool, recording: bool = False) -> int:
    e = cfg["env"]
    if not e.get("cap_envs_by_gpu_mem", False):
        return n
    devs = [int(x) for x in os.environ.get("PLD_EGL_DEVICES", "0").split(",") if x != ""] or [0]
    free = [_free_gpu_gb(d) for d in devs]
    if any(f is None for f in free):
        return n
    reserve = float(e.get("gpu_mem_reserve_train_gb" if training else "gpu_mem_reserve_eval_gb", 1.0))
    # envs that record episodes (stage 2) hold more GPU memory: ~1.45 GB vs ~0.9-1.15 GB measured on an RTX 4090
    per = float(e.get("gpu_mem_per_env_record_gb", 1.5) if recording else e.get("gpu_mem_per_env_gb", 1.2))
    # envs are spread round-robin over the rendering GPUs; the learner lives on the last one
    allowed = sum(max(0, int((f - (reserve if k == len(free) - 1 else 0.0)) // per)) for k, f in enumerate(free))
    if allowed < n:
        print(f"[envs] capping num_envs {n} -> {max(1, allowed)} (free GPU GB {['%.1f' % f for f in free]}, "
              f"{per} GB/env, reserve {reserve} GB)", flush=True)
        return max(1, allowed)
    return n


def build_env_and_base(cfg: dict, scene: str = "fixed", task: str | None = None, record_root: str | None = None,
                       num_envs: int | None = None, port: int | None = None, host: str | None = None,
                       training: bool = False):
    from .utils.threads import limit_threads

    limit_threads()
    os.environ.setdefault("MUJOCO_GL", cfg["env"].get("mujoco_gl", "egl"))
    spec = make_spec(cfg, scene=scene, task=task)
    n = cap_num_envs(cfg, int(num_envs or cfg["env"]["num_envs"]), training, recording=record_root is not None)
    cfg["env"]["num_envs_actual"] = n
    egl = os.environ.get("PLD_EGL_DEVICES")
    venv = VecEnv(spec, n, record_root=record_root, gl=cfg["env"].get("mujoco_gl", "egl"),
                  egl_devices=egl.split(",") if egl else None)
    b = cfg["base"]
    base = BasePolicy(host or b["host"], int(port or b["port"]), n, b["exec_steps"], b["history_indices"],
                      replicas=int(b.get("replicas", 1)))
    if b.get("action_bias") is not None:
        bias = np.zeros(12, np.float32)
        vals = np.asarray(b["action_bias"], np.float32)
        bias[: len(vals)] = vals          # env action order: eef pos 0:3, rot 3:6, gripper 6, base motion 7:11, mode 11
        base.action_bias = bias
        print(f"[base] KNOWN-ANSWER TEST: constant bias {vals.tolist()} added to every VLA action", flush=True)
    return venv, base, spec


def build_backbone(cfg: dict) -> tuple[FrozenBackbone, tuple, int]:
    dev = cfg["rl"]["device"]
    bb = FrozenBackbone().to(dev)
    n_cams = len(cfg["residual"]["cameras"])
    s = int(cfg["residual"]["image_size"])
    with torch.no_grad():
        f = bb(torch.zeros(1, n_cams, s, s, 3, dtype=torch.uint8, device=dev))
    return bb, tuple(f.shape[2:]), n_cams


def claim_gpu_memory(backbone, specs: dict, num_envs: int, cfg: dict) -> None:
    """Make this process allocate its CUDA memory now, before the env workers' EGL renderers fill the GPU.

    Stage 1 does this through the learner; stage 2 / eval only run inference, and their first real forward used to
    come after the renderers had taken everything (stage 2 on an RTX 4090, 8 recording envs + XR1 server:
    "CUDA out of memory. Tried to allocate 12 MiB", 2026-11-20)."""
    dev = torch.device(cfg["rl"]["device"])
    if dev.type != "cuda":
        return
    s, n_cams = int(cfg["residual"]["image_size"]), len(cfg["residual"]["cameras"])
    with torch.no_grad():   # augmented views are encoded in chunks of num_envs, so this is the peak
        backbone(torch.zeros(max(1, num_envs), n_cams, s, s, 3, dtype=torch.uint8, device=dev))
    for sp in specs.values():
        sp.agent.reserve_gpu_memory()


def aug_views(cfg: dict) -> int:
    """Randomly shifted copies encoded and stored per frame for the learner (residual.augment.views; 0 = off)."""
    return int((cfg["residual"].get("augment") or {}).get("views", 0))


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
            offline=ReplayBuffer(offline_capacity, (n_cams, *fmap_shape), 16, 12, int(res["dims"]), views=aug_views(cfg)),
            online=ReplayBuffer(cap, (n_cams, *fmap_shape), 16, 12, int(res["dims"]), views=aug_views(cfg)),
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
    runner = Runner(venv, base, backbone, cfg["rl"]["device"], num_stages, set(learnable_stage_ids(cfg)),
                    specs, residual_dims=int(cfg["residual"]["dims"]), gamma=float(cfg["rl"]["discount"]))
    runner.timeout_terminal = bool(cfg["rl"].get("timeout_terminal", True))
    runner.aug_views = aug_views(cfg)
    runner.hold = max(1, int(cfg["rl"].get("residual_hold", 1)))
    runner.n_step = max(1, int(cfg["rl"].get("n_step", 1)))
    runner.aug_pad = int((cfg["residual"].get("augment") or {}).get("pad", 4))
    runner.max_learner_lag = int(cfg["rl"].get("max_learner_lag", 0) or 0)
    return runner


def common_argparser(desc: str):
    import argparse

    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--model", required=True, choices=["rldx", "xr1"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides a.b=c")
    return ap
