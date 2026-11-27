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
PRIV_NAMES = (["t", "success", "grasped"] + [f"eef_pos_{a}" for a in "xyz"] + [f"eef_mat_{i}" for i in range(9)]
              + ["finger_q1", "finger_q2"] + [f"obj_pos_{a}" for a in "xyz"] + [f"obj_quat_{a}" for a in "wxyz"]
              + [f"recep_pos_{a}" for a in "xyz"])
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
# Privileged sim state appended to the stored proprio vector when env.privileged is true (asymmetric critic:
# rl.critic_proprio may select these, the actor never does). World frame; 13 entries after the 16 state entries.
PRIV_STATE_ORDER = (
    ("priv_obj_rel_eef", 3),     # mug body - eef site
    ("priv_recep_rel_obj", 3),   # coffee-machine receptacle_place_site - mug body
    ("priv_recep_rel_eef", 3),   # receptacle - eef
    ("priv_obj_up", 3),          # the mug's z axis (tipping)
    ("priv_grasped", 1),         # OU.check_obj_grasped(env, "obj")
)
PRIV_STATE_DIM = sum(n for _, n in PRIV_STATE_ORDER)


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
    privileged: bool = False        # append PRIV_STATE_ORDER to pkt["proprio"] (16 -> 29 entries)
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
        # Diagnostics only (failure analysis): with PLD_PRIV_DIR set, every step's privileged sim state (PRIV_NAMES)
        # is written per episode to $PLD_PRIV_DIR/priv_<seed>_<pid>_<n>.npz. Off by default; nothing else changes.
        self._priv_dir = os.environ.get("PLD_PRIV_DIR") or None
        self._priv: dict | None = None
        self._priv_n = 0

    # ------------------------------------------------------------------ obs
    def _packet(self, raw: dict) -> dict:
        import cv2

        images = {c: np.ascontiguousarray(raw[f"{c}_image"][::-1]) for c in CAMERAS}
        state = {k: np.asarray(raw[v], dtype=np.float32).reshape(-1) for k, v in STATE_KEYS.items()}
        rs = self.spec.residual_size
        # residual_size == camera size: the residual policy sees the rendered frame as it is (no resampling)
        res_images = np.stack([
            images[c] if images[c].shape[:2] == (rs, rs) else cv2.resize(images[c], (rs, rs), interpolation=cv2.INTER_AREA)
            for c in self.spec.residual_cameras
        ])
        proprio = state_vector16(state)
        if self.spec.privileged:
            proprio = np.concatenate([proprio, self._priv_state()])
        return {
            "images": images,
            "state": state,
            "proprio": proprio.astype(np.float32),
            "res_images": res_images,
        }

    def _priv_state(self) -> np.ndarray:
        """PRIV_STATE_ORDER (world frame). Needs a task with an "obj" and a coffee machine (CoffeeSetupMug)."""
        from robocasa.utils import object_utils as OU

        env, d = self.env, self.env.sim.data
        cm = getattr(env, "coffee_machine", None)
        if "obj" not in getattr(env, "obj_body_id", {}) or cm is None:
            raise ValueError(f"env.privileged: no privileged state defined for {self.spec.task} (needs obj + coffee machine)")
        eef = d.site_xpos[env.robots[0].eef_site_id["right"]]
        bid = env.obj_body_id["obj"]
        obj = d.body_xpos[bid]
        rec = d.site_xpos[env.sim.model.site_name2id(f"{cm.naming_prefix}receptacle_place_site")]
        up = d.body_xmat[bid].reshape(3, 3)[:, 2]
        return np.concatenate([obj - eef, rec - obj, rec - eef, up, [float(OU.check_obj_grasped(env, "obj"))]])

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
        if self._priv_dir:
            self._flush_priv()
            self._priv = {"seed": int(seed), "rows": [self._priv_row(success)]}
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
        if self._priv is not None:
            self._priv["rows"].append(self._priv_row(success))
        return pkt

    # ------------------------------------------------------------------ privileged trace (diagnostics)
    def _priv_row(self, success: bool) -> np.ndarray:
        """One row of PRIV_NAMES (world frame). Object/receptacle entries are NaN when the task has no "obj" or no
        coffee machine."""
        from robocasa.utils import object_utils as OU

        env, d = self.env, self.env.sim.data
        nan3, nan4 = [np.nan] * 3, [np.nan] * 4
        sid = env.robots[0].eef_site_id["right"]
        eef_pos, eef_mat = d.site_xpos[sid], d.site_xmat[sid]
        gq = [d.qpos[env.sim.model.get_joint_qpos_addr(j)] for j in ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2")]
        obj_pos, obj_quat, grasped = nan3, nan4, np.nan
        if "obj" in getattr(env, "obj_body_id", {}):
            bid = env.obj_body_id["obj"]
            obj_pos, obj_quat = d.body_xpos[bid], d.body_xquat[bid]
            grasped = float(OU.check_obj_grasped(env, "obj"))
        rec_pos = nan3
        cm = getattr(env, "coffee_machine", None)
        if cm is not None:
            rec_pos = d.site_xpos[env.sim.model.site_name2id(f"{cm.naming_prefix}receptacle_place_site")]
        return np.concatenate([[self.t, float(success), grasped], eef_pos, eef_mat, gq, obj_pos, obj_quat,
                               rec_pos]).astype(np.float64)

    def _flush_priv(self) -> None:
        p, self._priv = self._priv, None
        if not p or not self._priv_dir:
            return
        Path(self._priv_dir).mkdir(parents=True, exist_ok=True)
        self._priv_n += 1
        np.savez(Path(self._priv_dir) / f"priv_{p['seed']}_{os.getpid()}_{self._priv_n}.npz",
                 rows=np.stack(p["rows"]), names=np.array(PRIV_NAMES), seed=p["seed"])

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
        self._flush_priv()
        self.env.close()


# ---------------------------------------------------------------------------- subprocess vec env
def _worker(remote, parent_remote, spec_kwargs: dict, record_root: str | None, gl: str, egl_device: str | None):
    from ..utils.threads import limit_threads

    limit_threads(1)  # MuJoCo stepping is single-threaded; avoid one OpenMP thread per host core per worker
    parent_remote.close()
    os.environ.setdefault("MUJOCO_GL", gl)
    w_egl = os.environ.get("PLD_WORKER_EGL_ID")
    if w_egl:
        # Raw EGL device index for the sim workers (bw2: EGL order != CUDA order, see scripts/bw2_env.sh). robosuite
        # asserts MUJOCO_EGL_DEVICE_ID is listed in CUDA_VISIBLE_DEVICES; sim workers never use CUDA, so hide it.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["MUJOCO_EGL_DEVICE_ID"] = w_egl
    elif egl_device is not None:
        # robosuite asserts MUJOCO_EGL_DEVICE_ID is listed in CUDA_VISIBLE_DEVICES when the latter is set
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cvd or egl_device in cvd.split(","):
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
    """N RC365Env workers in subprocesses. Synchronous step over the subset of envs given.

    Watchdog: a worker that does not answer a step within `step_timeout` seconds (or a reset within `reset_timeout`)
    is killed and replaced by a fresh one. step() then returns {"_dead": True} for that env and the caller drops the
    episode. Seen on bw2 (B200, 32 envs): one worker spun at 100% CPU inside env.step for 80 min and blocked the whole
    run; the cause is unknown (no ptrace there) and it is rare (the twin run with the same seeds did not hit it)."""

    def __init__(self, spec: EnvSpec, num_envs: int, record_root: str | None = None, gl: str = "egl",
                 egl_devices: list[str] | None = None, step_timeout: float | None = None,
                 reset_timeout: float | None = None):
        import multiprocessing as mp
        from dataclasses import asdict

        self._ctx = mp.get_context("spawn")
        self.num_envs = num_envs
        self._args = (asdict(spec), record_root, gl)
        self._egl = egl_devices
        self.step_timeout = float(step_timeout or os.environ.get("PLD_ENV_STEP_TIMEOUT", 300))
        self.reset_timeout = float(reset_timeout or os.environ.get("PLD_ENV_RESET_TIMEOUT", 900))
        self.restarts = 0
        self.remotes: list = [None] * num_envs
        self.procs: list = [None] * num_envs
        for i in range(num_envs):
            self._spawn(i)
        self.remotes[0].send(("horizon", None))
        self.horizon = self.remotes[0].recv()

    def _spawn(self, i: int) -> None:
        from ..utils.threads import _VARS

        saved = {v: os.environ.get(v) for v in _VARS}
        for v in _VARS:  # children inherit the environment at spawn: 1 thread each
            os.environ[v] = "1"
        try:
            remote, work_remote = self._ctx.Pipe()
            dev = self._egl[i % len(self._egl)] if self._egl else None
            spec_kwargs, record_root, gl = self._args
            p = self._ctx.Process(target=_worker, args=(work_remote, remote, spec_kwargs, record_root, gl, dev), daemon=True)
            p.start()
            work_remote.close()
        finally:
            for v, val in saved.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val
        self.remotes[i], self.procs[i] = remote, p

    def _restart(self, i: int, why: str) -> None:
        p = self.procs[i]
        print(f"[vecenv] env {i}: {why} -- killing worker pid {p.pid} and starting a new one", flush=True)
        p.kill()
        p.join(timeout=30)
        try:
            self.remotes[i].close()
        except Exception:  # noqa: BLE001
            pass
        self.restarts += 1
        self._spawn(i)

    def _recv(self, i: int, timeout: float, what: str):
        """The worker's answer, or None after restarting a worker that missed the deadline or died."""
        r = self.remotes[i]
        try:
            if r.poll(timeout):
                return r.recv()
            self._restart(i, f"no answer to {what} within {timeout:.0f} s")
        except (EOFError, OSError, ConnectionError) as e:
            self._restart(i, f"worker gone during {what} ({type(e).__name__})")
        return None

    def reset(self, idx: int, seed: int, record: bool = False) -> dict:
        for attempt in range(2):
            self.remotes[idx].send(("reset", {"seed": seed, "record": record}))
            out = self._recv(idx, self.reset_timeout, f"reset(seed={seed})")
            if out is not None:
                return out
        raise RuntimeError(f"env {idx}: reset(seed={seed}) failed twice")

    def step(self, actions: dict[int, np.ndarray]) -> dict[int, dict]:
        for i, a in actions.items():
            self.remotes[i].send(("step", a))
        out = {}
        for i in actions:
            r = self._recv(i, self.step_timeout, "step")
            out[i] = {"_dead": True} if r is None else r
        return out

    def finish(self, idx: int, keep: bool, extra_meta: dict | None = None) -> str | None:
        self.remotes[idx].send(("finish", {"keep": keep, "extra_meta": extra_meta}))
        return self._recv(idx, self.reset_timeout, "finish")

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
                if r.poll(30):
                    r.recv()
            except Exception:  # noqa: BLE001
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()


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
        privileged=bool(cfg["env"].get("privileged", False)),
    )
