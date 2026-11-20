"""RLDX-1 base-policy server. Run inside the RLDX-1 venv (github.com/RLWRLD/RLDX-1, tag v1.0.2):

    cd RLDX-1 && uv run python /path/to/residual/pld_rc365/base/rldx_server.py \
        --model-path RLWRLD/RLDX-1-FT-RC365 --port 21000 --device cuda:0 --seed 0

Inputs follow the RC365 eval (`rldx/eval/rollout_policy.py` + upstream gym wrapper):
video keys robot0_agentview_left/right, robot0_eye_in_hand as uint8 (B, 4, 256, 256, 3) with
delta indices [-6, -4, -2, 0]; state (B, 1, D) for the 5 GENERAL_EMBODIMENT state keys; language
`annotation.human.task_description` as [[str], ...]. Output keys are already un-normalized.
Flow-matching noise is drawn fresh per call from the global torch RNG (seeded once here).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pld_rc365.base.protocol import Server  # noqa: E402

CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
STATE_KEYS = (
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
    "base_position",
    "base_rotation",
)
ACTION_ORDER = ("end_effector_position", "end_effector_rotation", "gripper_close", "base_motion", "control_mode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="RLWRLD/RLDX-1-FT-RC365")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=21000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from pld_rc365.utils.threads import limit_threads

    limit_threads()  # env vars were already set when the pld_rc365 package was imported (before torch)
    import torch

    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "4")))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.policy.rldx_policy import RLDXPolicy

    policy = RLDXPolicy(
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        model_path=args.model_path,
        device=args.device,
        strict=True,
    )
    lang_key = policy.modality_configs["language"].modality_keys[0]
    video_keys = list(policy.modality_configs["video"].modality_keys)
    state_keys = list(policy.modality_configs["state"].modality_keys)
    assert set(video_keys) == set(CAMS), video_keys
    assert set(state_keys) == set(STATE_KEYS), state_keys

    def handler(batch: list[dict]) -> np.ndarray:
        obs = {
            "video": {k: np.stack([b["video"][k] for b in batch]).astype(np.uint8) for k in video_keys},
            # RLDX uses only the current state (delta_indices [0])
            "state": {k: np.stack([b["state"][k][-1:] for b in batch]).astype(np.float32) for k in state_keys},
            "language": {lang_key: [[str(b["lang"])] for b in batch]},
        }
        with torch.inference_mode():
            action, _ = policy.get_action(obs)
        chunk = np.concatenate([np.asarray(action[k], dtype=np.float32) for k in ACTION_ORDER], axis=-1)
        assert chunk.shape[-1] == 12, chunk.shape
        return chunk

    Server(args.host, args.port, handler,
           info={"kind": "rldx", "model_path": args.model_path, "seed": args.seed}).serve_forever()


if __name__ == "__main__":
    main()
