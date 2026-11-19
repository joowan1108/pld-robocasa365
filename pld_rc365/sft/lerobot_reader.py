"""Dependency-light reader for the LeRobot v2.1 datasets written by stage 2 (pandas + PyAV only), used by the
XR1 trainer (whose venv does not need lerobot). Returns samples in the *policy* conventions:
  video: {cam: uint8 (4, H, W, 3)} at relative steps history_indices (clamped to the episode start)
  state: {key: float32 (4, D)} same steps (RLDX/gym-wrapper key names)
  action: float32 (chunk, 12) in env order [eef_pos, eef_rot, gripper, base_motion, control_mode] + valid mask
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np

CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
STATE_SLICES = {  # lerobot observation.state layout (PandaOmron_modality.json)
    "base_position": (0, 3),
    "base_rotation": (3, 7),
    "end_effector_position_relative": (7, 10),
    "end_effector_rotation_relative": (10, 14),
    "gripper_qpos": (14, 16),
}
# lerobot action layout -> env order
ACTION_TO_ENV = np.r_[5:8, 8:11, 11:12, 0:4, 4:5]


class LeRobotEpisodes:
    def __init__(self, roots: list[str | Path], history_indices=(-6, -4, -2, 0), chunk: int = 16):
        import pandas as pd

        self.hist = list(history_indices)
        self.chunk = chunk
        self.episodes = []  # (root, info, ep_index, states, actions, lang)
        for root in roots:
            root = Path(root)
            if (root / "lerobot").is_dir():
                root = root / "lerobot"
            info = json.loads((root / "meta" / "info.json").read_text())
            tasks = {}
            for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    tasks[d["task_index"]] = d["task"]
            for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                ep = json.loads(line)
                idx = ep["episode_index"]
                chunk_id = idx // info["chunks_size"]
                pq = root / info["data_path"].format(episode_chunk=chunk_id, episode_index=idx)
                df = pd.read_parquet(pq)
                states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
                actions = np.stack(df["action"].to_numpy()).astype(np.float32)[:, ACTION_TO_ENV]
                tdi = int(np.asarray(df["annotation.human.task_description"].iloc[0]).reshape(-1)[0])
                self.episodes.append(dict(root=root, info=info, index=idx, chunk=chunk_id, states=states,
                                          actions=actions, lang=tasks[tdi], fps=info["fps"]))
        self.index = [(e, t) for e in range(len(self.episodes)) for t in range(len(self.episodes[e]["actions"]))]

    def __len__(self):
        return len(self.index)

    def _video_path(self, ep: dict, cam: str) -> Path:
        return ep["root"] / ep["info"]["video_path"].format(
            episode_chunk=ep["chunk"], video_key=f"observation.images.{cam}", episode_index=ep["index"])

    @staticmethod
    def _decode(path: Path, frame_ids: list[int], fps: int) -> list[np.ndarray]:
        import av

        want = sorted(set(frame_ids))
        out: dict[int, np.ndarray] = {}
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            tb = s.time_base
            target = Fraction(want[0], fps)
            c.seek(int(target / tb), stream=s, backward=True, any_frame=False)
            for fr in c.decode(s):
                fi = int(round(float(fr.pts * tb) * fps))
                if fi in want:
                    out[fi] = fr.to_ndarray(format="rgb24")
                if fi >= want[-1]:
                    break
        missing = [f for f in want if f not in out]
        if missing:
            raise RuntimeError(f"could not decode frames {missing} from {path}")
        return [out[f] for f in frame_ids]

    def sample(self, i: int) -> dict:
        e, t = self.index[i]
        ep = self.episodes[e]
        sel = [max(0, t + d) for d in self.hist]
        video = {c: np.stack(self._decode(self._video_path(ep, c), sel, ep["fps"])) for c in CAMS}
        st = ep["states"][sel]
        state = {k: st[:, a:b] for k, (a, b) in STATE_SLICES.items()}
        T = len(ep["actions"])
        idx = np.arange(t, t + self.chunk)
        valid = idx < T
        act = ep["actions"][np.minimum(idx, T - 1)]
        return {"video": video, "state": state, "lang": ep["lang"], "action": act, "valid": valid}
