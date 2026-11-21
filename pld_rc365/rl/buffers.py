"""Replay buffers.

Transition = (feat, proprio, a_b, a_bar, r, done, next_feat, next_proprio, next_a_b, mc_return).
  * a_b      : the frozen VLA's 12-D action at s_t (current element of its action chunk) -- part of the state.
  * a_bar    : the *full executed* arm+gripper action a_b[:7] + a_delta (clipped), NOT the residual alone.
               For base-only data (offline successes, warm-up) a_bar = a_b[:7].
  * next_a_b : the VLA action actually used at s_{t+1}; needed for the TD target Q(s', a_b' + pi_delta(s', a_b')).
  * mc_return: discounted Monte-Carlo return inside the specialist's segment (Cal-QL lower bound; offline only).

Image features live in a frame store: every observation is stored once and transitions point to it (s_{t+1} of one
transition is s_t of the next). A frame holds `views` versions of the frozen-encoder feature maps: with image
augmentation these are the features of `views` randomly shifted copies of the frame, and sampling picks one per
camera, independently for s and s'. Arrays are allocated with np.zeros (lazily committed by the OS), so memory grows
with what is stored.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import torch

FIELDS = ("feat", "proprio", "a_b", "a_bar", "reward", "done", "next_feat", "next_proprio", "next_a_b", "mc_return")
SMALL = ("proprio", "a_b", "a_bar", "reward", "done", "next_proprio", "next_a_b", "mc_return", "steps")
DEFAULTS = {"steps": 1.0}   # env steps between s and s' (the TD target discounts the bootstrap by gamma ** steps)


class FrameRef:
    """Encoder features of one observation, shape (n, n_cam, C, h, w). n = 1: the plain frame. n = 1 + K: the plain
    frame at index 0 (used to act) followed by K augmented views (the ones a buffer with views=K stores).
    `slots` remembers where each buffer stored it, so a frame shared by two transitions is stored once."""

    __slots__ = ("data", "slots")

    def __init__(self, data: np.ndarray):
        self.data = data
        self.slots: dict[int, tuple[int, int]] = {}


class ReplayBuffer:
    def __init__(self, capacity: int, feat_shape: tuple, proprio_dim: int, base_dim: int, act_dim: int,
                 views: int = 1, frame_capacity: int | None = None):
        self.capacity = int(capacity)
        self.feat_shape = tuple(feat_shape)          # (n_cam, C, h, w)
        self.n_cam = int(self.feat_shape[0])
        self.views = max(1, int(views))
        self.size = 0
        self.ptr = 0
        self.total_added = 0
        self.lock = threading.Lock()
        c = self.capacity
        # One new frame per transition plus one per stored segment (its first observation); 2 per transition covers
        # buffers loaded from the pre-frame-store format. A transition whose frames were overwritten is never
        # sampled (see _valid), so a too-small store costs data, not correctness.
        self.frame_capacity = int(frame_capacity or (2 * c if c <= 4096 else c + max(4096, c // 16)))
        self.frames = np.zeros((self.frame_capacity, self.views, *self.feat_shape), dtype=np.float16)
        self._flat = self.frames.reshape(-1, *self.feat_shape[1:])          # (frame, view, cam) flattened
        self._flat_t = torch.from_numpy(self._flat)                          # same memory, for GIL-free gathers
        self.frame_ptr = 0
        self.frames_added = 0
        self.stale_skipped = 0
        self.data = {
            "proprio": np.zeros((c, proprio_dim), dtype=np.float32),
            "next_proprio": np.zeros((c, proprio_dim), dtype=np.float32),
            "a_b": np.zeros((c, base_dim), dtype=np.float32),
            "next_a_b": np.zeros((c, base_dim), dtype=np.float32),
            "a_bar": np.zeros((c, act_dim), dtype=np.float32),
            "reward": np.zeros((c,), dtype=np.float32),
            "done": np.zeros((c,), dtype=np.float32),
            "mc_return": np.zeros((c,), dtype=np.float32),
            "steps": np.ones((c,), dtype=np.float32),
            "obs_id": np.zeros((c,), dtype=np.int64),        # frame slot of s
            "next_obs_id": np.zeros((c,), dtype=np.int64),   # frame slot of s'
            "clock": np.zeros((c,), dtype=np.int64),         # frames_added when the older of the two was written
        }

    # ------------------------------------------------------------------ writing
    def _put(self, fr) -> tuple[int, int]:
        if not isinstance(fr, FrameRef):                     # a bare feature array: one plain frame
            a = np.asarray(fr)
            fr = FrameRef(a[None] if a.ndim == len(self.feat_shape) else a)
        rec = fr.slots.get(id(self))
        if rec is not None and self.frames_added - rec[1] < self.frame_capacity:
            return rec                                       # already stored and not overwritten since
        d = fr.data
        if d.shape[0] == 1 + self.views and self.views > 1:
            d = d[1:]                                        # drop the plain frame, keep the augmented views
        elif d.shape[0] != self.views:
            raise ValueError(f"frame has {d.shape[0]} views, buffer stores {self.views}")
        slot = self.frame_ptr
        self.frames[slot] = d
        rec = (slot, self.frames_added)
        fr.slots[id(self)] = rec
        self.frame_ptr = (slot + 1) % self.frame_capacity
        self.frames_added += 1
        return rec

    def add(self, tr: dict):
        with self.lock:
            so, sn = self._put(tr["feat"]), self._put(tr["next_feat"])
            i = self.ptr
            for k in SMALL:
                self.data[k][i] = tr.get(k, DEFAULTS.get(k, 0.0))
            self.data["obs_id"][i], self.data["next_obs_id"][i] = so[0], sn[0]
            self.data["clock"][i] = min(so[1], sn[1])
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self.total_added += 1

    def add_many(self, trs: list[dict]):
        for tr in trs:
            self.add(tr)

    # ------------------------------------------------------------------ reading
    def _draw(self, n: int, rng: np.random.Generator):
        """Indices of n transitions whose frames are intact, the small fields, and flat frame indices for s and s'."""
        with self.lock:
            idx = rng.integers(0, self.size, size=n)
            ok = self.frames_added - self.data["clock"][idx] < self.frame_capacity
            if not ok.all():                                  # frames overwritten: replace by intact transitions
                self.stale_skipped += int((~ok).sum())
                live = np.flatnonzero(self.frames_added - self.data["clock"][: self.size] < self.frame_capacity)
                if live.size == 0:
                    raise RuntimeError("replay buffer: no transition with intact frames (frame store too small)")
                idx[~ok] = rng.choice(live, size=int((~ok).sum()))
            small = {k: self.data[k][idx] for k in SMALL}
            so, sn = self.data["obs_id"][idx], self.data["next_obs_id"][idx]
        cams = np.arange(self.n_cam)[None]
        pick = (lambda: rng.integers(0, self.views, size=(n, self.n_cam))) if self.views > 1 else (lambda: 0)
        flat = lambda slot: (((slot[:, None] * self.views + pick()) * self.n_cam) + cams).reshape(-1)  # noqa: E731
        return idx, small, flat(so), flat(sn)

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        _, small, fo, fn = self._draw(n, rng)
        small["feat"] = self._flat[fo].reshape(n, *self.feat_shape)
        small["next_feat"] = self._flat[fn].reshape(n, *self.feat_shape)
        return small

    def sample_into(self, n: int, rng: np.random.Generator, out: dict[str, torch.Tensor], offset: int = 0) -> None:
        """Write n sampled transitions into rows [offset, offset+n) of preallocated CPU tensors (e.g. pinned)."""
        if n <= 0:
            return
        _, small, fo, fn = self._draw(n, rng)
        for k, v in small.items():
            out[k][offset:offset + n] = torch.from_numpy(v)
        for key, f in (("feat", fo), ("next_feat", fn)):
            dst = out[key][offset:offset + n].view(n * self.n_cam, *self.feat_shape[1:])
            torch.index_select(self._flat_t, 0, torch.from_numpy(f), out=dst)

    def batch_spec(self) -> dict[str, tuple[tuple, torch.dtype]]:
        spec = {k: (self.data[k].shape[1:], torch.float32) for k in SMALL}
        spec["feat"] = spec["next_feat"] = (self.feat_shape, torch.float16)
        return spec

    def __len__(self):
        return self.size

    # ------------------------------------------------------------------ disk
    def save(self, path):
        with self.lock:
            if self.total_added > self.capacity or self.frames_added > self.frame_capacity:
                raise NotImplementedError("saving a buffer that has wrapped around is not supported")
            np.savez(path, size=self.size, ptr=self.ptr, total_added=self.total_added, views=self.views,
                     frames_added=self.frames_added, frames=self.frames[: self.frames_added],
                     **{k: v[: self.size] for k, v in self.data.items()})

    def load(self, path) -> int:
        """Replace the contents with a buffer saved by save() (shapes must match). Returns #loaded.
        Files written before the frame store existed (feat / next_feat per transition) are read too."""
        d = np.load(path)
        n = int(d["size"])
        if n > self.capacity:
            raise ValueError(f"{path}: {n} transitions > capacity {self.capacity}")
        for k in SMALL:
            if k not in d.files:
                continue
            if d[k].shape[1:] != self.data[k].shape[1:]:
                raise ValueError(f"{path}: field {k} shape {d[k].shape[1:]} != buffer {self.data[k].shape[1:]}")
        with self.lock:
            if "frames" in d.files:
                fr = d["frames"]
                if fr.shape[1:] != self.frames.shape[1:]:
                    raise ValueError(f"{path}: frames {fr.shape[1:]} != buffer {self.frames.shape[1:]} (views, n_cam, C, h, w)")
                nf = fr.shape[0]
                if nf > self.frame_capacity:
                    raise ValueError(f"{path}: {nf} frames > frame capacity {self.frame_capacity}")
                self.frames[:nf] = fr
                for k in ("obs_id", "next_obs_id", "clock"):
                    self.data[k][:n] = d[k][:n]
            else:                                              # old format: one frame per feat and per next_feat
                if self.views != 1:
                    raise ValueError(f"{path}: stored without augmented views, buffer expects {self.views}")
                f, nx = d["feat"], d["next_feat"]
                if f.shape[1:] != self.feat_shape:
                    raise ValueError(f"{path}: field feat shape {f.shape[1:]} != buffer {self.feat_shape}")
                nf = 2 * n
                if nf > self.frame_capacity:
                    raise ValueError(f"{path}: {nf} frames > frame capacity {self.frame_capacity}")
                self.frames[:n, 0], self.frames[n:nf, 0] = f[:n], nx[:n]
                self.data["obs_id"][:n], self.data["next_obs_id"][:n] = np.arange(n), n + np.arange(n)
                self.data["clock"][:n] = 0
            for k in SMALL:
                self.data[k][:n] = d[k][:n] if k in d.files else DEFAULTS.get(k, 0.0)   # files without 'steps': 1
            self.size = self.ptr = self.total_added = n
            self.ptr %= self.capacity
            self.frames_added, self.frame_ptr = nf, nf % self.frame_capacity
        return n


class BatchPrefetcher:
    """Background threads that sample learner batches (half offline, half online, as symmetric_sample), copy them to
    the learner's device through pinned memory and keep a few ready in a queue. The gathers and the host-to-GPU
    copies release the GIL, so they no longer compete with the collector's Python code for the learner's time."""

    def __init__(self, offline: ReplayBuffer, online: ReplayBuffer | None, batch: int, device, threads: int = 3,
                 depth: int = 4, seed: int = 0):
        self.offline, self.online, self.batch = offline, online, int(batch)
        self.device = torch.device(device)
        self.q: queue.Queue = queue.Queue(maxsize=max(1, int(depth)))
        self.stop_flag = threading.Event()
        self.error: BaseException | None = None
        self.threads = [threading.Thread(target=self._work, args=(seed, i), daemon=True) for i in range(max(1, int(threads)))]
        for t in self.threads:
            t.start()

    def _work(self, seed: int, i: int):
        try:
            rng = np.random.default_rng([int(seed), i])
            cuda = self.device.type == "cuda"
            host = {k: torch.empty((self.batch, *shape), dtype=dt) for k, (shape, dt) in self.offline.batch_spec().items()}
            if cuda:
                host = {k: v.pin_memory() for k, v in host.items()}
            while not self.stop_flag.is_set():
                n_on_buf = 0 if self.online is None else len(self.online)
                if len(self.offline) == 0 and n_on_buf == 0:
                    self.stop_flag.wait(0.01)
                    continue
                n_on = 0 if n_on_buf == 0 else (self.batch if len(self.offline) == 0 else self.batch - self.batch // 2)
                self.offline.sample_into(self.batch - n_on, rng, host, 0)
                if n_on:
                    self.online.sample_into(n_on, rng, host, self.batch - n_on)
                b = {k: (v.to(self.device) if cuda else v.clone()) for k, v in host.items()}
                while not self.stop_flag.is_set():
                    try:
                        self.q.put(b, timeout=0.1)
                        break
                    except queue.Full:
                        pass
        except BaseException as e:  # noqa: BLE001
            self.error = e
            self.stop_flag.set()

    def get(self, timeout: float = 120.0) -> dict[str, torch.Tensor]:
        t_end = None
        while True:
            if self.error is not None:
                raise RuntimeError("batch prefetch thread failed") from self.error
            try:
                return self.q.get(timeout=0.2)
            except queue.Empty:
                import time
                t_end = t_end or time.time() + timeout
                if time.time() > t_end:
                    raise TimeoutError("no batch from the prefetch threads") from None

    def close(self):
        self.stop_flag.set()
        for t in self.threads:
            t.join(timeout=10)


def symmetric_sample(offline: ReplayBuffer, online: ReplayBuffer, batch: int, rng) -> dict[str, np.ndarray]:
    """RLPD / PLD: half the batch from each buffer (all offline while the online buffer is empty)."""
    if len(online) == 0:
        return offline.sample(batch, rng)
    if len(offline) == 0:
        return online.sample(batch, rng)
    a = offline.sample(batch // 2, rng)
    b = online.sample(batch - batch // 2, rng)
    return {k: np.concatenate([a[k], b[k]], axis=0) for k in a}


def aggregate_transitions(trs: list[dict], hold: int, n_step: int, gamma: float) -> list[dict]:
    """Turn the per-env-step transitions of one contiguous segment (one specialist stage of one episode, in order,
    starting at a residual decision step) into the learner's transitions:

    1. residual hold: consecutive blocks of `hold` env steps, during which the residual was held fixed, become one
       transition: s, a_b, a_bar of the block's first step; reward = sum_j gamma^j r_j; s', a_b' and done of its
       last step; steps = block length (the last block may be shorter when the segment ends inside it).
    2. n-step: every block j is backed up over blocks j..j+n_step-1 (stopping at a terminal one):
       reward = sum of their discounted rewards, s' / a_b' / done of the last one, steps = their total length.
    The TD target is then  reward + gamma ** steps * (1 - done) * Q(s', a_b', pi(s', a_b')).
    mc_return (offline data) is taken from the first step of each transition."""
    if not trs:
        return []
    hold, n_step = max(1, int(hold)), max(1, int(n_step))
    blocks = []
    for a in range(0, len(trs), hold):
        b = trs[a:a + hold]
        f, last = b[0], b[-1]
        if f.get("feat") is None:
            raise ValueError("first step of a residual block has no features (block alignment bug)")
        blocks.append({
            **{k: f[k] for k in ("feat", "proprio", "a_b", "a_bar") if k in f},
            "mc_return": f.get("mc_return", 0.0), "_residual": f.get("_residual", False),
            "reward": sum(gamma ** j * float(t.get("reward", 0.0)) for j, t in enumerate(b)),
            "done": float(last.get("done", 0.0)), "steps": float(len(b)),
            "next_feat": last.get("next_feat"), "next_proprio": last.get("next_proprio"), "next_a_b": last.get("next_a_b"),
        })
    out = []
    for j in range(len(blocks)):
        R, steps, last = 0.0, 0.0, None
        for m in blocks[j:j + n_step]:
            R += gamma ** steps * m["reward"]
            steps += m["steps"]
            last = m
            if m["done"]:
                break
        tr = {**blocks[j], "reward": R, "steps": steps, "done": last["done"], "next_proprio": last["next_proprio"],
              "next_a_b": last["next_a_b"], "next_feat": last["next_feat"]}
        if tr["next_feat"] is None:        # terminal: s' is never used; any stored frame will do
            if not tr["done"]:
                raise ValueError("non-terminal transition without next-state features")
            tr["next_feat"] = tr["feat"]
        if tr["next_a_b"] is None:
            tr["next_a_b"] = np.zeros_like(np.asarray(tr["a_b"]))
        out.append(tr)
    return out


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
