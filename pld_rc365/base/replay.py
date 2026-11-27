"""A base policy that replays recorded VLA actions instead of querying the VLA (residual_demo --replay-from).

For counterfactual videos: the VLA alone was recorded on a seed (its raw 12-D action at every step, from trace.npy);
replaying exactly those actions from the same seed reproduces the episode (the simulation is deterministic), and
adding the residual on top of them from some step on shows what the residual alone changes. After the takeover the
replayed VLA actions are open loop: they no longer react to the state the residual has moved the robot to.

Same interface as BasePolicy as far as the Runner uses it; every env must be reset with a seed that has a recording."""

from __future__ import annotations

import collections

import numpy as np


def actions_from_trace(trace: np.ndarray, d: int = 7) -> np.ndarray:
    """trace rows [t, a_b[:d], a_delta[:d], acted, decided, a_b[d:]] -> (T, 12) raw VLA actions."""
    tr = np.asarray(trace, dtype=np.float32)
    if tr.shape[1] < 1 + 2 * d + 2 + 1:
        raise ValueError(f"trace has {tr.shape[1]} columns: recorded before the full VLA action was kept")
    return np.concatenate([tr[:, 1:1 + d], tr[:, 1 + 2 * d + 2:]], 1)


class ReplayBase:
    def __init__(self, recordings: dict[int, np.ndarray], num_envs: int):
        self.rec = {int(k): np.asarray(v, dtype=np.float32) for k, v in recordings.items()}
        self.queues = [collections.deque() for _ in range(num_envs)]
        self.last = [None] * num_envs
        self.info = {"kind": "replay", "seeds": sorted(self.rec)}
        self.n_queries = self.n_requests = 0
        self.query_seconds = 0.0
        self.action_bias = None
        self.ran_out = collections.Counter()

    def reset_env(self, i: int, pkt: dict):
        seed = int(pkt["seed"])
        if seed not in self.rec:
            raise KeyError(f"no recorded VLA actions for seed {seed}")
        self.queues[i] = collections.deque(self.rec[seed])
        self.last[i] = self.rec[seed][-1]
        self.seed = seed

    def observe(self, i: int, pkt: dict):
        pass

    def _fill(self, i: int):
        if not self.queues[i]:              # episode longer than the recording: hold the last action
            self.queues[i].append(self.last[i].copy())
            self.ran_out[i] += 1

    def has_action(self, i: int) -> bool:
        self._fill(i)
        return True

    @property
    def in_flight_envs(self) -> set[int]:
        return set()

    def submit(self, idxs) -> int:
        return 0

    def poll(self, block: bool = False) -> int:
        return 0

    def drain(self) -> None:
        pass

    def ensure(self, idxs) -> None:
        for i in idxs:
            self._fill(i)

    def peek(self, i: int) -> np.ndarray:
        self._fill(i)
        return self.queues[i][0].copy()

    def pop(self, i: int) -> np.ndarray:
        self._fill(i)
        return self.queues[i].popleft()


class PrefixBase:
    """Branched evaluation (residual_demo --branch-from): replay the first t_base actions of a recorded VLA episode on
    the same seed (the simulation is deterministic, so this reaches exactly the recorded state at t_base), then query
    the live VLA (`inner`, a BasePolicy) from t_base on. The VLA's observation history is fed throughout, so at t_base
    it sees the same frames as in the recorded episode. Two branches started this way (VLA alone / VLA + residual)
    share the state at t_base exactly; they differ only by the residual and the VLA's own sampling noise afterwards.
    reset_env needs pkt["seed"] and pkt["t_base"] (the Runner sets both)."""

    def __init__(self, inner, recordings: dict[int, np.ndarray]):
        self.inner = inner
        self.rec = {int(k): np.asarray(v, dtype=np.float32) for k, v in recordings.items()}
        self.prefix = [collections.deque() for _ in range(len(inner.queues))]

    def __getattr__(self, name):                       # info, n_queries, n_requests, queues, ...
        return getattr(self.inner, name)

    def reset_env(self, i: int, pkt: dict):
        self.inner.reset_env(i, pkt)
        seed, t = int(pkt["seed"]), int(pkt.get("t_base", 0))
        if seed not in self.rec:
            raise KeyError(f"no recorded VLA actions for seed {seed}")
        if t > len(self.rec[seed]):
            raise ValueError(f"seed {seed}: t_base {t} beyond the recorded episode ({len(self.rec[seed])} steps)")
        self.prefix[i] = collections.deque(self.rec[seed][:t])

    def observe(self, i: int, pkt: dict):
        self.inner.observe(i, pkt)

    def has_action(self, i: int) -> bool:
        return bool(self.prefix[i]) or self.inner.has_action(i)

    @property
    def in_flight_envs(self) -> set[int]:
        return self.inner.in_flight_envs

    def submit(self, idxs) -> int:
        return self.inner.submit([i for i in idxs if not self.prefix[i]])

    def poll(self, block: bool = False) -> int:
        return self.inner.poll(block)

    def drain(self) -> None:
        self.inner.drain()

    def ensure(self, idxs) -> None:
        self.inner.ensure([i for i in idxs if not self.prefix[i]])

    def peek(self, i: int) -> np.ndarray:
        return self.prefix[i][0].copy() if self.prefix[i] else self.inner.peek(i)

    def pop(self, i: int) -> np.ndarray:
        return self.prefix[i].popleft() if self.prefix[i] else self.inner.pop(i)
