"""Sim-free fakes for the rollout engine: a scripted env, a scripted base policy and a dummy backbone."""

from __future__ import annotations

import collections

import numpy as np
import torch


class FakeVecEnv:
    """Each env: state = step counter t. Stage advances at the steps listed in `stage_steps` (per stage)
    if `succeed` is True for that seed (seed % 2 == 0 by default). Horizon fixed."""

    def __init__(self, num_envs: int, horizon: int, stage_steps=(5,), succeed=lambda seed: seed % 2 == 0):
        self.num_envs, self.horizon = num_envs, horizon
        self.stage_steps = list(stage_steps)
        self.succeed = succeed
        self.t = [0] * num_envs
        self.seed = [0] * num_envs
        self.stage = [0] * num_envs
        self.actions = collections.defaultdict(list)   # (env, seed) -> executed actions
        self.finished = []

    def _pkt(self, i):
        t = self.t[i]
        K = len(self.stage_steps)
        return {
            "images": {c: np.full((8, 8, 3), t, np.uint8) for c in
                       ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")},
            "state": {"end_effector_position_relative": np.array([t, 0, 0], np.float32)},
            "proprio": np.full(16, t, np.float32),
            "res_images": np.full((2, 4, 4, 3), t % 256, np.uint8),
            "lang": "do it", "success": self.stage[i] >= K, "stage": self.stage[i], "t": t,
        }

    def reset(self, i, seed, record=False):
        self.t[i], self.seed[i], self.stage[i] = 0, seed, 0
        return self._pkt(i)

    def step(self, actions):
        out = {}
        for i, a in actions.items():
            self.actions[(i, self.seed[i])].append(np.asarray(a).copy())
            self.t[i] += 1
            if self.succeed(self.seed[i]):
                while self.stage[i] < len(self.stage_steps) and self.t[i] >= self.stage_steps[self.stage[i]]:
                    self.stage[i] += 1
            out[i] = self._pkt(i)
        return out

    def finish(self, i, keep, extra_meta=None):
        self.finished.append((i, keep, extra_meta))
        return f"/tmp/ep_{i}" if keep else None


class FakeBase:
    """Base action = [t_query*100 + k, ...] where k is the index inside the chunk: lets tests check that the
    a_b stored for step t and the next_a_b are exactly the executed queue elements."""

    def __init__(self, num_envs, exec_steps=4):
        self.exec_steps = exec_steps
        self.q = [collections.deque() for _ in range(num_envs)]
        self.cur_t = [0] * num_envs
        self.n_queries = 0
        self.query_seconds = 0.0
        self.info = {"kind": "fake"}

    def reset_env(self, i, pkt):
        self.q[i].clear()
        self.cur_t[i] = pkt["t"]

    def observe(self, i, pkt):
        self.cur_t[i] = pkt["t"]

    def ensure(self, idxs):
        for i in idxs:
            if not self.q[i]:
                self.n_queries += 1
                for k in range(self.exec_steps):
                    a = np.zeros(12, np.float32)
                    a[0] = (self.cur_t[i] + k) / 1000.0     # encodes the env step this action is meant for
                    a[6] = 1.0
                    self.q[i].append(a)

    def peek(self, i):
        return self.q[i][0].copy()

    def pop(self, i):
        return self.q[i].popleft()


class FakeBackbone(torch.nn.Module):
    """(B, n, H, W, 3) uint8 -> (B, n, 4, 2, 2) half: encodes the mean pixel so features identify the step."""

    def forward(self, x):
        m = x.float().mean(dim=(2, 3, 4))  # (B, n)
        return m[..., None, None, None].expand(*m.shape, 4, 2, 2).half()
