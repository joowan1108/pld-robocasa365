"""LeRobot v2.1 reader used by the XR1 trainer: frame-accurate history decoding and action reordering."""

import json

import numpy as np
import pytest

av = pytest.importorskip("av")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _write_fake_dataset(root, T=40, size=64):
    from pld_rc365.sft.lerobot_reader import CAMS

    (root / "meta").mkdir(parents=True)
    info = {"fps": 20, "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pick the mug"}) + "\n")
    (root / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": T}) + "\n")
    (root / "data" / "chunk-000").mkdir(parents=True)
    states = np.tile(np.arange(T)[:, None], (1, 16)).astype(np.float64)
    actions = np.tile(np.arange(12)[None], (T, 1)).astype(np.float64) + 100 * np.arange(T)[:, None]
    pd.DataFrame({"observation.state": list(states), "action": list(actions),
                  "annotation.human.task_description": [np.array([0])] * T}).to_parquet(
        root / "data" / "chunk-000" / "episode_000000.parquet")
    for c in CAMS:
        d = root / "videos" / "chunk-000" / f"observation.images.{c}"
        d.mkdir(parents=True)
        with av.open(str(d / "episode_000000.mp4"), "w") as cont:
            s = cont.add_stream("libx264", rate=20)
            s.width = s.height = size
            s.pix_fmt = "yuv420p"
            s.options = {"g": "8", "crf": "0"}
            for t in range(T):
                img = np.full((size, size, 3), (t * 6) % 250, np.uint8)
                for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    cont.mux(p)
            for p in s.encode():
                cont.mux(p)


def test_history_frames_and_chunk(tmp_path):
    from pld_rc365.sft.lerobot_reader import LeRobotEpisodes

    _write_fake_dataset(tmp_path / "lerobot")
    ds = LeRobotEpisodes([tmp_path])
    assert len(ds) == 40
    s = ds.sample(ds.index.index((0, 21)))
    means = s["video"]["robot0_eye_in_hand"].reshape(4, -1).mean(1)
    np.testing.assert_allclose(means, [15 * 6, 17 * 6, 19 * 6, 21 * 6], atol=3)  # t-6, t-4, t-2, t
    assert s["state"]["base_position"][:, 0].tolist() == [15, 17, 19, 21]
    # env order: eef_pos <- lerobot 5:8, ..., control_mode <- lerobot 4
    np.testing.assert_array_equal(s["action"][0] - 2100, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])
    s0 = ds.sample(ds.index.index((0, 1)))
    np.testing.assert_allclose(s0["video"]["robot0_agentview_left"].reshape(4, -1).mean(1), [0, 0, 0, 6], atol=3)
    s_end = ds.sample(ds.index.index((0, 35)))
    assert s_end["valid"].sum() == 5 and np.all(s_end["action"][5:] == s_end["action"][4])
