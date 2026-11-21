"""Replay buffers.

Transition = (feat, proprio, a_b, a_bar, r, done, next_feat, next_proprio, next_a_b, mc_return).
  * a_b      : the frozen VLA's 12-D action at s_t (current element of its action chunk) -- part of the state.
  * a_bar    : the *full executed* arm+gripper action a_b[:7] + a_delta (clipped), NOT the residual alone.
               For base-only data (offline successes, warm-up) a_bar = a_b[:7].
  * next_a_b : the VLA action actually used at s_{t+1}; needed for the TD target Q(s', a_b' + pi_delta(s', a_b')).
  * mc_return: discounted Monte-Carlo return inside the specialist's segment (Cal-QL lower bound; offline only).
Arrays are allocated with np.zeros (lazily committed by the OS), so memory grows with what is stored.
"""

from __future__ import annotations

import threading

import numpy as np

FIELDS = ("feat", "proprio", "a_b", "a_bar", "reward", "done", "next_feat", "next_proprio", "next_a_b", "mc_return")


class ReplayBuffer:
    def __init__(self, capacity: int, feat_shape: tuple, proprio_dim: int, base_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.size = 0
        self.ptr = 0
        self.total_added = 0
        self.lock = threading.Lock()
        c = self.capacity
        self.data = {
            "feat": np.zeros((c, *feat_shape), dtype=np.float16),
            "next_feat": np.zeros((c, *feat_shape), dtype=np.float16),
            "proprio": np.zeros((c, proprio_dim), dtype=np.float32),
            "next_proprio": np.zeros((c, proprio_dim), dtype=np.float32),
            "a_b": np.zeros((c, base_dim), dtype=np.float32),
            "next_a_b": np.zeros((c, base_dim), dtype=np.float32),
            "a_bar": np.zeros((c, act_dim), dtype=np.float32),
            "reward": np.zeros((c,), dtype=np.float32),
            "done": np.zeros((c,), dtype=np.float32),
            "mc_return": np.zeros((c,), dtype=np.float32),
        }

    def add(self, tr: dict):
        with self.lock:
            i = self.ptr
            for k in FIELDS:
                self.data[k][i] = tr.get(k, 0.0)
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self.total_added += 1

    def add_many(self, trs: list[dict]):
        for tr in trs:
            self.add(tr)

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        with self.lock:
            idx = rng.integers(0, self.size, size=n)
            return {k: v[idx] for k, v in self.data.items()}   # fancy indexing already copies

    def __len__(self):
        return self.size

    def save(self, path):
        with self.lock:
            np.savez(path, size=self.size, ptr=self.ptr, total_added=self.total_added,
                     **{k: v[: self.size] for k, v in self.data.items()})

    def load(self, path) -> int:
        """Replace the contents with a buffer saved by save() (shapes must match). Returns #loaded."""
        d = np.load(path)
        n = int(d["size"])
        for k in FIELDS:
            if d[k].shape[1:] != self.data[k].shape[1:]:
                raise ValueError(f"{path}: field {k} shape {d[k].shape[1:]} != buffer {self.data[k].shape[1:]}")
        if n > self.capacity:
            raise ValueError(f"{path}: {n} transitions > capacity {self.capacity}")
        with self.lock:
            for k in FIELDS:
                self.data[k][:n] = d[k][:n]
            self.size = self.ptr = self.total_added = n
        return n


def symmetric_sample(offline: ReplayBuffer, online: ReplayBuffer, batch: int, rng) -> dict[str, np.ndarray]:
    """RLPD / PLD: half the batch from each buffer (all offline while the online buffer is empty)."""
    if len(online) == 0:
        return offline.sample(batch, rng)
    if len(offline) == 0:
        return online.sample(batch, rng)
    a = offline.sample(batch // 2, rng)
    b = online.sample(batch - batch // 2, rng)
    return {k: np.concatenate([a[k], b[k]], axis=0) for k in a}


def segment_mc_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted returns computed backwards, restarting after every done (segment boundary)."""
    out = np.zeros_like(rewards, dtype=np.float32)
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        if dones[t]:
            g = 0.0
        g = float(rewards[t]) + gamma * g
        out[t] = g
    return out
