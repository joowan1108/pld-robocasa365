"""Write recorded PLD episodes as a LeRobot v2.1 dataset in RoboCasa365's own format.

Mirrors robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py (robocasa 1.0.x, lerobot==0.3.3):
  features: 3 x video (256x256, h264/yuv420p, 20 fps), observation.state (16, float64), action (12, float64),
            annotation.human.task_description / task_name (int64 indices into tasks.jsonl), next.reward, next.done
  observation.state order: base_pos(3) base_quat(4) eef_pos_rel(3) eef_quat_rel(4) gripper_qpos(2)
  action order (PandaOmron_modality.json): base_motion(4) control_mode(1) eef_pos(3) eef_rot(3) gripper(1)
  images are upright (the converter's robomimic env flips MuJoCo renders, as the gym wrapper does).
  meta/modality.json + embodiment.json + stats.json via robocasa's add_groot_specific_metadata.
Layout: <out_dir>/lerobot/{data,videos,meta}, matching the `**/lerobot/meta/info.json` glob of the RLDX recipe.

Each recorded episode dir holds frames.npy (T+1, 3, H, W, 3 uint8 memmap; first T used) and lowdim.npz
(states (T,16) in dataset order, actions (T,12) *executed* env actions in env order, lang).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


def write_dataset(episode_dirs: list[str], out_dir: str | Path, task_name: str, overwrite: bool = True) -> Path:
    from robocasa.utils.lerobot_utils import (LerobotDatasetWrapper, add_groot_specific_metadata,
                                              reorder_hdf5_action)

    out_dir = Path(out_dir)
    lerobot_path = out_dir / "lerobot"
    if lerobot_path.exists():
        if not overwrite:
            raise FileExistsError(lerobot_path)
        shutil.rmtree(lerobot_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    modality_path = Path(__import__("robocasa").__file__).parent / "models/assets/groot_dataset_assets/PandaOmron_modality.json"
    modality = json.loads(modality_path.read_text())

    lows = [np.load(Path(d) / "lowdim.npz", allow_pickle=False) for d in episode_dirs]
    langs = [str(l["lang"]) for l in lows]
    task_to_id: dict[str, int] = {}
    for lang in langs:
        task_to_id.setdefault(lang, len(task_to_id))
    task_name_idx = len(task_to_id)

    first = np.load(Path(episode_dirs[0]) / "frames.npy", mmap_mode="r")
    img_shape = tuple(first.shape[2:])
    video_info = {"video.fps": 20, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                  "video.is_depth_map": False, "has_audio": False}
    features = {f"observation.images.{c}": {"dtype": "video", "shape": img_shape,
                                            "names": ["height", "width", "channel"], "video_info": video_info}
                for c in ("robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right")}
    features.update({
        "annotation.human.task_description": {"dtype": "int64", "shape": (1,)},
        "annotation.human.task_name": {"dtype": "int64", "shape": (1,)},
        "observation.state": {"dtype": "float64", "shape": (16,)},
        "action": {"dtype": "float64", "shape": (12,)},
        "next.reward": {"dtype": "float32", "shape": (1,)},
        "next.done": {"dtype": "bool", "shape": (1,)},
    })
    ds = LerobotDatasetWrapper.create(repo_id=lerobot_path, robot_type="PandaOmron", fps=20, features=features,
                                      image_writer_threads=16, image_writer_processes=8)
    for d, low, lang in zip(episode_dirs, lows, langs):
        frames = np.load(Path(d) / "frames.npy", mmap_mode="r")
        states = np.asarray(low["states"], dtype=np.float64)
        actions = reorder_hdf5_action(np.asarray(low["actions"], dtype=np.float64), modality)
        T = int(low["length"])
        assert states.shape == (T, 16) and actions.shape == (T, 12), (states.shape, actions.shape, T)
        for i in range(T):
            ds.add_frame({
                **{f"observation.images.{c}": np.asarray(frames[i, j]) for j, c in enumerate(CAMS)},
                "observation.state": states[i],
                "action": actions[i],
                "annotation.human.task_description": np.array([task_to_id[lang]], dtype=np.int64),
                "next.reward": np.array([1.0 if i == T - 1 else 0.0], dtype=np.float32),
                "next.done": np.array([i == T - 1], dtype=bool),
                "annotation.human.task_name": np.array([task_name_idx], dtype=np.int64),
            }, task=lang)
        ds.save_episode()
    add_groot_specific_metadata(out_dir)
    with open(lerobot_path / "meta" / "tasks.jsonl", "a") as f:
        f.write(json.dumps({"task_index": task_name_idx, "task": task_name}) + "\n")
    images_dir = lerobot_path / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    return lerobot_path
