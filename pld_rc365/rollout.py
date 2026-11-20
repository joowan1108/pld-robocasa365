"""Vectorised rollout engine shared by every stage.

Episode modes
  base : the frozen VLA acts for the whole episode (offline successes, warm-up, base/SFT evaluation).
  pld  : base-policy probing for t < t_base, then the residual specialist of the active stage takes over
         (a_bar = a_b + a_delta). Base-only stages (e.g. the toaster `wait`) stay with the VLA.
Storage
  none    : nothing stored.
  offline : every transition is kept in a per-episode list and pushed to the specialists' offline buffers
            only if the episode succeeds (MC returns computed per stage segment, for Cal-QL).
  online  : transitions go straight to the specialists' online buffers. In `pld` episodes only steps with
            t >= t_base are stored (probing only initialises the state, PLD §3.2).

Per transition (specialist k = stage at s_t):
  r = 1 if the stage index advanced at s_{t+1} (k's sparse stage success), done = r (terminal for k).
  next_a_b = the VLA action actually used at s_{t+1} (peeked once the next query happened).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .base.client import BasePolicy
from .envs.rc365_env import VecEnv, threshold_discrete
from .rl.buffers import ReplayBuffer, segment_mc_returns


@dataclass
class EpisodeSpec:
    seed: int
    mode: str = "base"            # base | pld
    t_base: int = 0               # takeover step for pld
    deterministic: bool = False   # residual mean (stage 2 / eval)
    store: str = "none"           # none | offline | online
    record: bool = False          # stage-2 frame recording
    tag: str = ""


@dataclass
class EpisodeResult:
    spec: EpisodeSpec
    success: bool
    length: int
    stage_reached: int
    took_over: bool               # pld: residual phase started before the episode ended
    takeover_step: int | None
    n_stored: int
    rec_dir: str | None
    wall_s: float
    extra: dict = field(default_factory=dict)


@dataclass
class Specialist:
    agent: object                 # ResidualSAC
    collector_actor: torch.nn.Module
    actor_lock: threading.Lock
    offline: ReplayBuffer
    online: ReplayBuffer
    learner: object | None = None


@dataclass
class _EnvState:
    spec: EpisodeSpec
    pkt: dict
    t: int = 0
    stage: int = 0
    feat: np.ndarray | None = None
    pending: dict | None = None
    took_over: bool = False
    takeover_step: int | None = None
    traj: list = field(default_factory=list)   # offline: (stage, transition)
    n_stored: int = 0
    scene: tuple | None = None                 # (layout_id, style_id) sampled at reset
    init: list | None = None                   # initial proprio (rounded): fingerprint of the reset state
    res_n: int = 0                             # residual-acted steps in this episode
    res_sum: np.ndarray | None = None          # sum of a_delta per dim
    res_abs: np.ndarray | None = None          # sum of |a_delta| per dim
    res_bound: np.ndarray | None = None        # count of |a_delta| > 0.95 xi per dim
    base_abs: np.ndarray | None = None         # sum of |a_b[:d]| over the same steps (VLA action scale)
    res_xi: float | None = None
    t0: float = field(default_factory=time.time)


class Runner:
    def __init__(self, venv: VecEnv, base: BasePolicy, backbone, device: str, num_stages: int,
                 learnable: set[int], specialists: dict[int, Specialist] | None = None,
                 residual_dims: int = 7, gamma: float = 0.99):
        self.venv, self.base, self.backbone = venv, base, backbone
        self.device = torch.device(device)
        self.K = num_stages
        self.learnable = set(learnable)
        self.spec = specialists or {}
        self.d = residual_dims
        self.gamma = gamma
        self.horizon = venv.horizon
        self.stats = {"env_steps": 0, "residual_steps": 0, "probe_steps": 0}
        self.offline_quota: int | None = None   # max successful episodes pushed to offline buffers
        self.offline_accepted = 0
        self.abort_on_stop = True               # drop in-flight episodes once on_end requested a stop
        self.check_consistency = True           # cheap per-step invariants (bug guard, no effect on results)
        self.log_every_s = 120.0                # throughput line on stdout

    # ------------------------------------------------------------------ helpers
    def _features(self, pkts: list[dict]) -> np.ndarray:
        imgs = torch.as_tensor(np.stack([p["res_images"] for p in pkts]), device=self.device)
        return self.backbone(imgs).cpu().numpy()

    def _residual_phase(self, s: _EnvState) -> bool:
        return s.spec.mode == "pld" and s.t >= s.spec.t_base

    def _acts_residual(self, s: _EnvState) -> bool:
        return self._residual_phase(s) and s.stage in self.learnable and s.stage in self.spec

    def _stores_now(self, s: _EnvState) -> bool:
        if s.spec.store == "none" or s.stage not in self.learnable:
            return False
        if s.spec.store == "offline" or s.spec.mode == "base":
            return True
        return self._residual_phase(s)

    def _push(self, s: _EnvState, k: int, tr: dict):
        s.n_stored += 1
        if s.spec.store == "offline":
            s.traj.append((k, tr))
        elif s.spec.store == "online":
            sp = self.spec[k]
            sp.online.add(tr)
            sp.agent.online_steps += 1
            if tr.get("_residual", False):
                sp.agent.xi_steps += 1

    def _flush_offline(self, s: _EnvState, success: bool):
        if s.spec.store != "offline" or not success:
            return
        if self.offline_quota is not None and self.offline_accepted >= self.offline_quota:
            return
        self.offline_accepted += 1
        by_stage: dict[int, list[dict]] = {}
        for k, tr in s.traj:
            by_stage.setdefault(k, []).append(tr)
        for k, trs in by_stage.items():
            r = np.array([t["reward"] for t in trs], dtype=np.float32)
            dn = np.array([t["done"] for t in trs], dtype=np.float32)
            mc = segment_mc_returns(r, dn, self.gamma)
            for t, g in zip(trs, mc):
                t["mc_return"] = g
            self.spec[k].offline.add_many(trs)

    # ------------------------------------------------------------------ main loop
    def run(self, next_spec: Callable[[], EpisodeSpec | None],
            on_end: Callable[[EpisodeResult], bool] | None = None,
            check: Callable[[], None] | None = None) -> None:
        """Run until `next_spec` is exhausted and all envs finish, or `on_end` returns True (stop)."""
        states: dict[int, _EnvState] = {}
        stopping = False

        def start(i: int) -> None:
            sp = None if stopping else next_spec()
            if sp is None:
                states.pop(i, None)
                return
            pkt = self.venv.reset(i, sp.seed, record=sp.record)
            self.base.reset_env(i, pkt)
            ls = pkt.get("layout_style")
            states[i] = _EnvState(spec=sp, pkt=pkt, stage=pkt["stage"],
                                  scene=tuple(int(x) for x in ls) if ls and None not in ls else None,
                                  init=[round(float(v), 4) for v in np.asarray(pkt["proprio"]).ravel()[:10]])

        for i in range(self.venv.num_envs):
            start(i)

        t_log = time.time()
        steps_at_log = self.stats["env_steps"]
        while states:
            if check is not None:
                check()
            # 1. base actions: pipelined queries -- envs without an action wait, the rest keep stepping
            self.base.poll()
            self.base.submit(sorted(states))
            active = [i for i in sorted(states) if self.base.has_action(i)]
            if not active:
                t_w = time.time()
                self.base.poll(block=True)
                self.stats["wait_s"] = self.stats.get("wait_s", 0.0) + time.time() - t_w
                continue
            if self.log_every_s and time.time() - t_log >= self.log_every_s:
                dt = time.time() - t_log
                nreq = max(1, getattr(self.base, "n_requests", 1))
                print(f"[runner] {(self.stats['env_steps'] - steps_at_log) / dt:.1f} env steps/s | "
                      f"envs {len(states)} | mean query batch {self.base.n_queries / nreq:.1f} | "
                      f"waiting {len(states) - len(active)}", flush=True)
                t_log, steps_at_log = time.time(), self.stats["env_steps"]
            a_b = {i: self.base.peek(i) for i in active}
            # 2. finalise transitions waiting for next_a_b
            for i in active:
                s = states[i]
                if s.pending is not None:
                    k, tr = s.pending
                    tr["next_a_b"] = a_b[i]
                    self._push(s, k, tr)
                    s.pending = None
            # 3. features where needed (cached from the previous step when available)
            need = [i for i in active if (self._acts_residual(states[i]) or self._stores_now(states[i]))
                    and states[i].feat is None]
            if need:
                feats = self._features([states[i].pkt for i in need])
                for i, f in zip(need, feats):
                    states[i].feat = f
            # 4. actions
            actions: dict[int, np.ndarray] = {}
            a_bar: dict[int, np.ndarray] = {}
            groups: dict[tuple[int, bool], list[int]] = {}
            for i in active:
                s = states[i]
                if s.spec.mode == "pld" and s.t >= s.spec.t_base and not s.took_over:
                    s.took_over, s.takeover_step = True, s.t
                if self._acts_residual(s):
                    groups.setdefault((s.stage, s.spec.deterministic), []).append(i)
                else:
                    actions[i] = a_b[i]
                    a_bar[i] = np.clip(a_b[i][: self.d], -1.0, 1.0)
                    if s.spec.mode == "pld" and s.t < s.spec.t_base:
                        self.stats["probe_steps"] += 1
            for (k, det), idxs in groups.items():
                sp = self.spec[k]
                with sp.actor_lock:
                    a_d = sp.agent.act(
                        sp.collector_actor,
                        np.stack([states[i].feat for i in idxs]),
                        np.stack([states[i].pkt["proprio"] for i in idxs]),
                        np.stack([a_b[i] for i in idxs]),
                        deterministic=det,
                    )
                xi_now = float(sp.agent.xi)
                for i, ad in zip(idxs, a_d):
                    s_ = states[i]
                    if s_.res_sum is None:
                        z = np.zeros(self.d)
                        s_.res_sum, s_.res_abs, s_.res_bound, s_.base_abs = z.copy(), z.copy(), z.copy(), z.copy()
                    s_.res_n += 1
                    s_.res_sum += ad
                    s_.res_abs += np.abs(ad)
                    s_.res_bound += np.abs(ad) > 0.95 * xi_now
                    s_.base_abs += np.abs(a_b[i][: self.d])
                    s_.res_xi = xi_now
                    ab = np.clip(a_b[i][: self.d] + ad, -1.0, 1.0)
                    act = a_b[i].copy()
                    act[: self.d] = ab
                    actions[i], a_bar[i] = act, ab
                    self.stats["residual_steps"] += 1
            # 5. step
            stores = {i: self._stores_now(states[i]) for i in active}
            acted_residual = {i for _, idxs in groups.items() for i in idxs}
            out = self.venv.step(actions)
            self.stats["env_steps"] += len(active)
            for i in active:
                popped = self.base.pop(i)
                if self.check_consistency:
                    self._check_step(i, popped, a_b[i], actions[i], a_bar[i], out[i])
                self.base.observe(i, out[i])
            # 6. next-state features for transitions being stored
            nf_idx = [i for i in active if stores[i]]
            next_feats = dict(zip(nf_idx, self._features([out[i] for i in nf_idx]))) if nf_idx else {}
            # 7. bookkeeping / transitions / episode ends
            ended: list[int] = []
            for i in active:
                s, nxt = states[i], out[i]
                advanced = nxt["stage"] > s.stage
                if stores[i]:
                    tr = {
                        "feat": s.feat, "proprio": s.pkt["proprio"], "a_b": a_b[i], "a_bar": a_bar[i],
                        "reward": 1.0 if advanced else 0.0, "done": 1.0 if advanced else 0.0,
                        "next_feat": next_feats[i], "next_proprio": nxt["proprio"],
                        "_residual": i in acted_residual,
                    }
                    if advanced:
                        tr["next_a_b"] = np.zeros_like(a_b[i])  # terminal for this specialist: unused
                        self._push(s, s.stage, tr)
                    else:
                        s.pending = (s.stage, tr)
                s.t, s.stage, s.pkt = nxt["t"], nxt["stage"], nxt
                s.feat = next_feats.get(i)
                if nxt["success"] or s.stage >= self.K or s.t >= self.horizon:
                    ended.append(i)
            # 8. finish episodes
            for i in ended:
                s = states[i]
                success = bool(s.pkt["success"])
                if s.pending is not None:  # time-limit truncation: bootstrap with the base action at s_T
                    self.base.ensure([i])
                    k, tr = s.pending
                    tr["next_a_b"] = self.base.peek(i)
                    self._push(s, k, tr)
                    s.pending = None
                self._flush_offline(s, success)
                rec_dir = None
                if s.spec.record:
                    keep = success and self._keep_record(s)
                    rec_dir = self.venv.finish(i, keep, extra_meta=self._record_meta(s))
                res = EpisodeResult(spec=s.spec, success=success, length=s.t, stage_reached=s.stage,
                                    took_over=s.took_over, takeover_step=s.takeover_step,
                                    n_stored=s.n_stored, rec_dir=rec_dir, wall_s=time.time() - s.t0,
                                    extra={"scene": s.scene, "init": s.init,
                                           "residual": self._residual_summary(s)})
                if on_end is not None and on_end(res):
                    stopping = True
                start(i)
            if stopping and self.abort_on_stop and states:
                for i in list(states):
                    if states[i].spec.record:
                        self.venv.finish(i, False)
                    states.pop(i)
        self.base.drain()  # leave no request in flight for the next run() (envs get reset there)

    @staticmethod
    def _residual_summary(s: "_EnvState") -> dict | None:
        """Per-episode residual statistics (per action dim: eef dx dy dz rx ry rz, gripper)."""
        if not s.res_n:
            return None
        n = float(s.res_n)
        r = lambda x: [round(float(v), 4) for v in x]  # noqa: E731
        return {"steps": s.res_n, "xi": s.res_xi, "mean": r(s.res_sum / n), "mean_abs": r(s.res_abs / n),
                "at_bound": r(s.res_bound / n), "base_abs": r(s.base_abs / n),
                "abs_over_xi": round(float((s.res_abs / n).mean() / max(s.res_xi, 1e-8)), 4),
                "abs_over_base": round(float((s.res_abs / n)[:6].sum() / max((s.base_abs / n)[:6].sum(), 1e-8)), 4)}

    def _check_step(self, i, popped, a_b, action, a_bar, pkt):
        """Invariants between what was stored and what was executed (raise on the first violation):
        chunk pointer: the base action used at this step is the queue element consumed for it;
        stored a_bar equals the executed arm+gripper action; base/torso/mode pass through from the VLA;
        the env executed exactly the thresholded version of the action we sent."""
        d = self.d
        if not np.array_equal(np.asarray(popped), a_b):
            raise RuntimeError(f"env {i}: chunk desync, used a_b {a_b} but queue popped {popped}")
        # the VLA may emit values slightly outside [-1, 1]; the OSC controller clips its input and the gripper is
        # thresholded, so the stored a_bar is the clipped version of what was sent
        if not np.allclose(np.clip(action[:d], -1.0, 1.0), a_bar, atol=1e-6):
            raise RuntimeError(f"env {i}: stored a_bar {a_bar} != executed arm action {action[:d]}")
        if not np.array_equal(action[d:], a_b[d:]):
            raise RuntimeError(f"env {i}: base/torso/mode dims changed: {action[d:]} vs a_b {a_b[d:]}")
        ex = pkt.get("executed_action")
        if ex is not None and not np.allclose(ex, threshold_discrete(action), atol=1e-6):
            raise RuntimeError(f"env {i}: env executed {ex} but we sent {action}")

    # stage-2 hooks (overridden via attributes)
    keep_pure_base: bool = True

    def _keep_record(self, s: _EnvState) -> bool:
        return self.keep_pure_base or s.took_over

    def _record_meta(self, s: _EnvState) -> dict:
        return {"t_base": s.spec.t_base, "took_over": s.took_over, "takeover_step": s.takeover_step,
                "pure_base": not s.took_over, "length": s.t, "seed": s.spec.seed}
