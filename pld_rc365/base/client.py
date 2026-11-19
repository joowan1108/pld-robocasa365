"""Per-env base-policy bookkeeping on the PLD side: observation history, action-chunk queue, a_b lookahead.

Chunk handling (not specified in the paper): the VLA is queried when the env's queue is empty, the first
`exec_steps` actions of its 16-step chunk are queued (RLDX 8, XR1 16 -- their RC365 eval settings), and the
residual acts per env step on the current queue head a_b,t.
"""

from __future__ import annotations

import collections
import time

import numpy as np

from .protocol import Client

CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


class BasePolicy:
    def __init__(self, host: str, port: int, num_envs: int, exec_steps: int, history_indices=(-6, -4, -2, 0)):
        self.client = Client(host, port)
        self.info = self.client.call({"cmd": "info"})
        self.exec_steps = int(exec_steps)
        self.hist_idx = list(history_indices)
        self.maxlen = 1 - min(self.hist_idx)
        self.num_envs = num_envs
        self.frames = [collections.deque(maxlen=self.maxlen) for _ in range(num_envs)]
        self.states = [collections.deque(maxlen=self.maxlen) for _ in range(num_envs)]
        self.queues = [collections.deque() for _ in range(num_envs)]
        self.langs = [""] * num_envs
        self.n_queries = 0
        self.query_seconds = 0.0

    def reset_env(self, i: int, pkt: dict):
        self.frames[i].clear()
        self.states[i].clear()
        self.queues[i].clear()
        self.langs[i] = pkt["lang"]
        self.observe(i, pkt)

    def observe(self, i: int, pkt: dict):
        self.frames[i].append(pkt["images"])
        self.states[i].append(pkt["state"])

    def _history(self, i: int) -> dict:
        items_f, items_s = list(self.frames[i]), list(self.states[i])
        n = len(items_f)
        # same clamping as both RC365 evaluators: missing history -> earliest available frame
        sel = [max(0, n - 1 + d) for d in self.hist_idx]
        return {
            "video": {c: np.stack([items_f[j][c] for j in sel]) for c in CAMS},
            "state": {k: np.stack([items_s[j][k] for j in sel]) for k in items_s[0]},
            "lang": self.langs[i],
        }

    def ensure(self, idxs) -> None:
        """Query the server (one batched request) for every env in idxs whose queue is empty."""
        need = [i for i in idxs if not self.queues[i]]
        if not need:
            return
        t0 = time.time()
        rep = self.client.call({"cmd": "infer", "obs": [self._history(i) for i in need]})
        self.query_seconds += time.time() - t0
        self.n_queries += len(need)
        chunks = np.asarray(rep["chunks"], dtype=np.float32)
        for i, ch in zip(need, chunks):
            self.queues[i].extend(ch[: self.exec_steps])

    def peek(self, i: int) -> np.ndarray:
        return self.queues[i][0].copy()

    def pop(self, i: int) -> np.ndarray:
        return self.queues[i].popleft()
