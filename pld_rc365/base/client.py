"""Per-env base-policy bookkeeping on the PLD side: observation history, action-chunk queue, a_b lookahead.

Chunk handling (not specified in the paper): the VLA is queried when the env's queue is empty, the first
`exec_steps` actions of its 16-step chunk are queued (RLDX 8, XR1 16 -- their RC365 eval settings), and the
residual acts per env step on the current queue head a_b,t.

Replicas: with N servers (one per GPU, ports port..port+N-1) env i is served by replica i % N and the replicas
are queried concurrently, so N GPUs give ~N x base-policy throughput.
"""

from __future__ import annotations

import collections
import time

import numpy as np

from .protocol import Client

CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


class BasePolicy:
    def __init__(self, host: str, port: int, num_envs: int, exec_steps: int, history_indices=(-6, -4, -2, 0),
                 replicas: int = 1):
        self.replicas = max(1, int(replicas))
        self.clients = [Client(host, port + r) for r in range(self.replicas)]
        infos = [c.call({"cmd": "info"}) for c in self.clients]
        self.info = {**infos[0], "replicas": infos}
        self._in_flight: dict[int, list[int]] = {}   # replica -> envs of its outstanding request
        self._sent_at: dict[int, float] = {}
        self.n_requests = 0
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

    # ------------------------------------------------------------------ async (pipelined) queries
    # One request in flight per replica. Envs whose queue ran out wait (they are not stepped) while the others keep
    # stepping; whatever accumulates during a request is sent as the next batch. Each env still executes exactly its
    # own chunk and replans at its own state -- only the order of computation changes.
    def has_action(self, i: int) -> bool:
        return bool(self.queues[i])

    @property
    def in_flight_envs(self) -> set[int]:
        return {i for envs in self._in_flight.values() for i in envs}

    def submit(self, idxs) -> int:
        """Send requests for envs in idxs with an empty queue, on replicas that are idle. Returns #envs sent."""
        busy = self.in_flight_envs
        groups: dict[int, list[int]] = {}
        for i in idxs:
            if not self.queues[i] and i not in busy:
                groups.setdefault(i % self.replicas, []).append(i)
        sent = 0
        for r, envs in groups.items():
            if r in self._in_flight:
                continue
            self.clients[r].send({"cmd": "infer", "obs": [self._history(i) for i in envs]})
            self._in_flight[r] = envs
            self._sent_at[r] = time.time()
            sent += len(envs)
        return sent

    def poll(self, block: bool = False) -> int:
        """Collect finished replies (all in-flight ones if block). Returns #envs whose queue was filled."""
        filled = 0
        for r in list(self._in_flight):
            if not block and not self.clients[r].ready():
                continue
            rep = self.clients[r].recv()
            envs = self._in_flight.pop(r)
            chunks = np.asarray(rep["chunks"], dtype=np.float32)
            for i, ch in zip(envs, chunks):
                self.queues[i].extend(ch[: self.exec_steps])
            self.query_seconds += time.time() - self._sent_at.pop(r)
            self.n_queries += len(envs)
            self.n_requests += 1
            filled += len(envs)
        return filled

    def drain(self) -> None:
        if self._in_flight:
            self.poll(block=True)

    def ensure(self, idxs) -> None:
        """Synchronous: make sure every env in idxs has a queued action (drains in-flight requests first)."""
        self.drain()
        while any(not self.queues[i] for i in idxs):
            self.submit(idxs)
            self.poll(block=True)

    # base.action_bias (diagnostic "known-answer" test): a constant offset added to every action the VLA returns, as if
    # the base policy itself were biased. The residual sees and corrects the biased action like any other a_b.
    action_bias: np.ndarray | None = None

    def _biased(self, a: np.ndarray) -> np.ndarray:
        return a if self.action_bias is None else a + self.action_bias

    def peek(self, i: int) -> np.ndarray:
        return self._biased(self.queues[i][0].copy())

    def pop(self, i: int) -> np.ndarray:
        return self._biased(self.queues[i].popleft())
