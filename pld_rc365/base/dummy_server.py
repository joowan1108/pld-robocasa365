"""Model-free stand-in for the base-policy servers, for smoke-testing the env harness + client + runner.

Validates every request against the formats both real servers expect and returns "hold still" chunks
(zero arm/base motion, gripper open, control_mode 0 -> arm mode). Also dumps the first request to disk so its
images can be eyeballed (orientation!).

    python pld_rc365/base/dummy_server.py --port 22000 --dump runs/logs/dummy_first_request.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pld_rc365.base.protocol import Server  # noqa: E402

CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
STATE_DIMS = {"end_effector_position_relative": 3, "end_effector_rotation_relative": 4, "gripper_qpos": 2,
              "base_position": 3, "base_rotation": 4}


def validate(o: dict, size: int = 256, hist: int = 4) -> None:
    for c in CAMS:
        v = o["video"][c]
        assert v.dtype == np.uint8 and v.shape == (hist, size, size, 3), (c, v.dtype, v.shape)
    for k, d in STATE_DIMS.items():
        s = o["state"][k]
        assert s.dtype == np.float32 and s.shape == (hist, d), (k, s.dtype, s.shape)
        assert np.all(np.isfinite(s)), k
    for k in ("end_effector_rotation_relative", "base_rotation"):
        n = np.linalg.norm(o["state"][k], axis=-1)
        assert np.allclose(n, 1.0, atol=1e-3), (k, n)  # unit quaternions
    assert isinstance(o["lang"], str) and len(o["lang"]) > 5, o["lang"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=22000)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()
    state = {"dumped": False, "requests": 0}

    def handler(batch):
        for o in batch:
            validate(o)
        state["requests"] += 1
        if args.dump and not state["dumped"]:
            o = batch[0]
            os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
            np.savez_compressed(args.dump, lang=o["lang"], **{f"video_{c}": o["video"][c] for c in CAMS},
                                **{f"state_{k}": v for k, v in o["state"].items()})
            state["dumped"] = True
        chunk = np.zeros((len(batch), 16, 12), np.float32)
        chunk[..., 6] = -1.0   # gripper open
        return chunk

    Server(args.host, args.port, handler, info={"kind": "dummy", "model_path": None, "seed": None}).serve_forever()


if __name__ == "__main__":
    main()
