"""RoboCasa365 env harness shared by both VLAs.

Mirrors `robocasa/wrappers/gym_wrapper.py` (the wrapper both RC365 evals use):
  * images are rendered upside down by MuJoCo and flipped (`img[::-1]`) before reaching the policy,
  * state keys = robot0_base_to_eef_pos/quat, robot0_gripper_qpos, robot0_base_pos/quat,
  * 12-D action in env order [eef_pos 3, eef_rot 3, gripper 1, base_motion 4 (base 3 + torso 1), control_mode 1],
    gripper / control_mode thresholded at 0.5 -> {-1, +1},
  * sparse success = env._check_success(), evaluated every step,
  * per-episode seed: env.rng = default_rng(seed) before reset.
but bypasses the gym wrapper so we (a) keep raw robosuite obs (privileged stage predicates) and
(b) can pass `split=None` with a fixed (layout, style) (the gym wrapper's split argument overrides ids).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
STATE_KEYS = {  # RLDX/gym-wrapper name -> robosuite obs key
    "end_effector_position_relative": "robot0_base_to_eef_pos",
    "end_effector_rotation_relative": "robot0_base_to_eef_quat",
    "gripper_qpos": "robot0_gripper_qpos",
    "base_position": "robot0_base_pos",
    "base_rotation": "robot0_base_quat",
}
# LeRobot observation.state (16-D) order from robocasa PandaOmron_modality.json
DATASET_STATE_ORDER = (
    ("base_position", 3),
    ("base_rotation", 4),
    ("end_effector_position_relative", 3),
    ("end_effector_rotation_relative", 4),
    ("gripper_qpos", 2),
)
ACTION_DIM = 12


def threshold_discrete(action12: np.ndarray) -> np.ndarray:
    """Apply the gym-wrapper thresholds so the returned vector is exactly what the env executed."""
    a = np.asarray(action12, dtype=np.float64).copy()
    a[6] = -1.0 if a[6] < 0.5 else 1.0
    a[11] = -1.0 if a[11] < 0.5 else 1.0
    return a


def state_vector16(state: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(state[k], dtype=np.float64).reshape(-1)[:n] for k, n in DATASET_STATE_ORDER])


@dataclass
class EnvSpec:
    task: str
    stage_names: list[str]
    scene: str = "fixed"            # fixed | pretrain | target
    layout_id: int | None = None
    style_id: int | None = None
    obj_instance_split: str | None = "pretrain"
    camera_size: int = 256
    residual_size: int = 128
    residual_cameras: tuple[str, ...] = ("robot0_agentview_left", "robot0_eye_in_hand")
    horizon: int | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class RC365Env:
    def __init__(self, spec: EnvSpec, record_root: str | None = None):
        import robocasa  # noqa: F401  (registers envs)
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        from robocasa.utils.env_utils import create_env

        from .stages import StageMachine

        self.spec = spec
        kw: dict[str, Any] = dict(
            env_name=spec.task,
            camera_names=list(CAMERAS),
            camera_widths=spec.camera_size,
            camera_heights=spec.camera_size,
            render_onscreen=False,
        )
        if spec.scene == "fixed":
            if spec.layout_id is None or spec.style_id is None:
                raise ValueError(f"{spec.task}: fixed scene needs layout_id/style_id (run stage0 --sweep first)")
            kw.update(split=None, layout_ids=int(spec.layout_id), style_ids=int(spec.style_id),
                      obj_instance_split=spec.obj_instance_split)
        elif spec.scene in ("pretrain", "target"):
            kw.update(split=spec.scene)
        else:
            raise ValueError(spec.scene)
        kw.update(spec.extra_kwargs)
        self.env = create_env(**kw)
        self.horizon = int(spec.horizon) if spec.horizon else int(get_task_horizon(spec.task))
        self.stage_machine = StageMachine(spec.task, spec.stage_names)
        self.record_root = Path(record_root) if record_root else None
        self._rec = None
        self.t = 0

    # ------------------------------------------------------------------ obs
    def _packet(self, raw: dict) -> dict:
        import cv2

        images = {c: np.ascontiguousarray(raw[f"{c}_image"][::-1]) for c in CAMERAS}
        state = {k: np.asarray(raw[v], dtype=np.float32).reshape(-1) for k, v in STATE_KEYS.items()}
        rs = self.spec.residual_size
        res_images = np.stack(
            [cv2.resize(images[c], (rs, rs), interpolation=cv2.INTER_AREA) for c in self.spec.residual_cameras]
        )
        return {
            "images": images,
            "state": state,
            "proprio": state_vector16(state).astype(np.float32),
            "res_images": res_images,
        }

    # ------------------------------------------------------------------ API
    def reset(self, seed: int, record: bool = False) -> dict:
        self.env.rng = np.random.default_rng(int(seed))
        raw = self.env.reset()
        self.t = 0
        success = bool(self.env._check_success())
        stage = self.stage_machine.reset(self.env)
        pkt = self._packet(raw)
        pkt.update(
            lang=str(self.env.get_ep_meta().get("lang", "")),
            success=success,
            stage=stage,
            t=0,
            horizon=self.horizon,
            layout_style=(getattr(self.env, "layout_id", None), getattr(self.env, "style_id", None)),
        )
        self._close_recorder(keep=False)
        if record:
            self._open_recorder(seed)
            self._rec_obs(pkt)
        return pkt

    def _env_action(self, a12: np.ndarray) -> np.ndarray:
        from robosuite.controllers.composite.composite_controller import HybridMobileBase

        a = threshold_discrete(a12)
        parts = {
            "right": a[0:6],
            "right_gripper": a[6:7],
            "base": a[7:10],
            "torso": a[10:11],
        }
        out = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            act = np.zeros(cc.action_limits[0].shape)
            for part_name in cc.part_controllers:
                s, e = cc._action_split_indexes[part_name]
                act[s:e] = parts[part_name]
            if isinstance(cc, HybridMobileBase):
                act[-1] = a[11]
            out.append(act)
        return np.concatenate(out)

    def step(self, action12: np.ndarray) -> dict:
        executed = threshold_discrete(action12)
        raw, _, _, _ = self.env.step(self._env_action(action12))
        self.t += 1
        success = bool(self.env._check_success())
        stage = self.stage_machine.update(self.env)
        pkt = self._packet(raw)
        pkt.update(success=success, stage=stage, t=self.t, executed_action=executed.astype(np.float32))
        if self._rec is not None:
            self._rec_action(executed)
            if not (success or self.t >= self.horizon):
                self._rec_obs(pkt)
        return pkt

    # ------------------------------------------------------------------ recording (stage 2)
    # Frames are written to a per-episode uint8 memmap on disk instead of RAM: a 3000-step
    # GetToastedBread episode is ~1.8 GB of 3x256x256 frames.
    def _open_recorder(self, seed: int):
        assert self.record_root is not None, "record_root not set"
        self.record_root.mkdir(parents=True, exist_ok=True)
        d = Path(tempfile.mkdtemp(prefix=f"ep_{seed}_", dir=self.record_root))
        n, s = self.horizon + 1, self.spec.camera_size
        self._rec = {
            "dir": d,
            "frames": np.lib.format.open_memmap(d / "frames.npy", mode="w+", dtype=np.uint8, shape=(n, len(CAMERAS), s, s, 3)),
            "states": [],
            "actions": [],
            "seed": seed,
        }

    def _rec_obs(self, pkt: dict):
        i = len(self._rec["states"])
        for j, c in enumerate(CAMERAS):
            self._rec["frames"][i, j] = pkt["images"][c]
        self._rec["states"].append(state_vector16(pkt["state"]))

    def _rec_action(self, executed: np.ndarray):
        self._rec["actions"].append(np.asarray(executed, dtype=np.float64))

    def finish_episode(self, keep: bool, extra_meta: dict | None = None) -> str | None:
        """Close the recorder. Returns the episode directory if kept (frames.npy, lowdim.npz)."""
        return self._close_recorder(keep=keep, extra_meta=extra_meta)

    def _close_recorder(self, keep: bool, extra_meta: dict | None = None) -> str | None:
        rec, self._rec = self._rec, None
        if rec is None:
            return None
        d = rec["dir"]
        n = len(rec["actions"])
        rec["frames"].flush()
        del rec["frames"]
        if not keep or n == 0:
            import shutil

            shutil.rmtree(d, ignore_errors=True)
            return None
        states = np.stack(rec["states"][:n])
        actions = np.stack(rec["actions"][:n])
        assert len(states) == n, (len(states), n)
        meta = dict(extra_meta or {})
        np.savez(d / "lowdim.npz", states=states, actions=actions, length=n, seed=rec["seed"],
                 lang=self.env.get_ep_meta().get("lang", ""), meta=np.array(repr(meta)))
        return str(d)

    def close(self):
        self._close_recorder(keep=False)
        self.env.close()


# ---------------------------------------------------------------------------- subprocess vec env
def _worker(remote, parent_remote, spec_kwargs: dict, record_root: str | None, gl: str, egl_device: str | None):
    parent_remote.close()
    os.environ.setdefault("MUJOCO_GL", gl)
    if egl_device is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = egl_device
    env = RC365Env(EnvSpec(**spec_kwargs), record_root=record_root)
    try:
        while True:
            cmd, arg = remote.recv()
            if cmd == "reset":
                remote.send(env.reset(**arg))
            elif cmd == "step":
                remote.send(env.step(arg))
            elif cmd == "finish":
                remote.send(env.finish_episode(**arg))
            elif cmd == "horizon":
                remote.send(env.horizon)
            elif cmd == "close":
                env.close()
                remote.send(None)
                break
            else:
                raise ValueError(cmd)
    except KeyboardInterrupt:
        pass


class VecEnv:
    """N RC365Env workers in subprocesses. Synchronous step over the subset of envs given."""

    def __init__(self, spec: EnvSpec, num_envs: int, record_root: str | None = None, gl: str = "egl",
                 egl_devices: list[str] | None = None):
        import multiprocessing as mp
        from dataclasses import asdict

        ctx = mp.get_context("spawn")
        self.num_envs = num_envs
        self.remotes, self.procs = [], []
        for i in range(num_envs):
            remote, work_remote = ctx.Pipe()
            dev = egl_devices[i % len(egl_devices)] if egl_devices else None
            p = ctx.Process(target=_worker, args=(work_remote, remote, asdict(spec), record_root, gl, dev), daemon=True)
            p.start()
            work_remote.close()
            self.remotes.append(remote)
            self.procs.append(p)
        self.remotes[0].send(("horizon", None))
        self.horizon = self.remotes[0].recv()

    def reset(self, idx: int, seed: int, record: bool = False) -> dict:
        self.remotes[idx].send(("reset", {"seed": seed, "record": record}))
        return self.remotes[idx].recv()

    def step(self, actions: dict[int, np.ndarray]) -> dict[int, dict]:
        for i, a in actions.items():
            self.remotes[i].send(("step", a))
        return {i: self.remotes[i].recv() for i in actions}

    def finish(self, idx: int, keep: bool, extra_meta: dict | None = None) -> str | None:
        self.remotes[idx].send(("finish", {"keep": keep, "extra_meta": extra_meta}))
        return self.remotes[idx].recv()

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
                r.recv()
            except Exception:  # noqa: BLE001
                pass
        for p in self.procs:
            p.join(timeout=10)


def make_spec(cfg: dict, scene: str = "fixed", task: str | None = None) -> EnvSpec:
    t = cfg["task"]
    task_name = task or t["env_name"]
    stage_names = t["stages"] if task_name == t["env_name"] else ["success"]
    return EnvSpec(
        task=task_name,
        stage_names=list(stage_names),
        scene=scene,
        layout_id=t.get("layout_id"),
        style_id=t.get("style_id"),
        obj_instance_split=cfg["env"].get("obj_instance_split", "pretrain"),
        camera_size=cfg["env"]["camera_size"],
        residual_size=cfg["residual"]["image_size"],
        residual_cameras=tuple(cfg["residual"]["cameras"]),
        horizon=cfg["env"].get("horizon"),
    )
