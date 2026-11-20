"""Xiaomi-Robotics-1 (RoboCasa365) base-policy server. Run inside a py3.11 / torch 2.8 /
transformers 4.57.1 / flash-attn 2 venv:

    MIBOT_SERVER_SEED=0 python pld_rc365/base/xr1_server.py \
        --model-path XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 --port 22000

Preprocessing replicates Xiaomi-Robotics-1 `eval_robocasa365/entry.py`:
  * 4 frames / 4 states at relative steps [-6, -4, -2, 0] (obs_history=4, obs_interval=2)
  * each frame center-cropped (0.95) and resized back (bilinear)
  * 14-D state: eef_pos_rel(3), axis-angle(eef_quat_rel)(3), gripper_qpos(2), base_pos(3), axis-angle(base_quat)(3),
    zero-padded to 60, shape (1, 4, 60)
  * chat prompt "Left camera: / Right camera: / Wrist camera:" videos + "Generate robot actions for the task:\\n{instr} /no_cot",
    assistant prefill "<cot></cot>"
  * model(**inputs) with num_steps=5, decode_action (identity on dims 0..11), keep [:, :12]
Note: importing modeling_mibot seeds the global torch RNG from MIBOT_SERVER_SEED (or 7+device); noise is then
drawn fresh per call from that stream, i.e. the policy stays stochastic across calls.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pld_rc365.base.protocol import Server  # noqa: E402
from pld_rc365.utils.rotations import quat_xyzw_to_axis_angle  # noqa: E402

CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
ROBOT_TYPE = "robocasa365"
STATE_DIM = 60


def xr1_state14(state: dict[str, np.ndarray]) -> np.ndarray:
    """(T, D) per key -> (T, 14)."""
    T = len(state["end_effector_position_relative"])
    rows = []
    for t in range(T):
        rows.append(np.concatenate([
            np.asarray(state["end_effector_position_relative"][t], np.float32).reshape(-1),
            quat_xyzw_to_axis_angle(state["end_effector_rotation_relative"][t]),
            np.asarray(state["gripper_qpos"][t], np.float32).reshape(-1),
            np.asarray(state["base_position"][t], np.float32).reshape(-1),
            quat_xyzw_to_axis_angle(state["base_rotation"][t]),
        ]))
    out = np.stack(rows).astype(np.float32)
    assert out.shape[1] == 14, out.shape
    return out


def center_crop(image: np.ndarray, crop_ratio: float):
    from PIL import Image

    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil
    w, h = pil.size
    cw, ch = max(1, int(w * crop_ratio)), max(1, int(h * crop_ratio))
    left, top = (w - cw) // 2, (h - ch) // 2
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return pil.crop((left, top, left + cw, top + ch)).resize((w, h), resampling)


def build_messages(video: dict[str, np.ndarray], instruction: str, crop_ratio: float) -> list[dict]:
    vids = {k: [center_crop(f, crop_ratio) for f in video[k]] for k in CAMS}
    return [
        {"role": "user", "content": [
            {"type": "text", "text": "Left camera: "},
            {"type": "video", "video": vids[CAMS[0]]},
            {"type": "text", "text": "\nRight camera: "},
            {"type": "video", "video": vids[CAMS[1]]},
            {"type": "text", "text": "\nWrist camera: "},
            {"type": "video", "video": vids[CAMS[2]]},
            {"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


def encode_inputs(processor, video: dict[str, np.ndarray], state: dict[str, np.ndarray], lang: str, crop_ratio: float):
    """Tokenize one sample exactly like the eval client. Shared with the SFT trainer."""
    s14 = xr1_state14(state)
    st = np.zeros((1, s14.shape[0], STATE_DIM), dtype=np.float32)
    st[0, :, :14] = s14
    return processor.apply_chat_template(
        build_messages(video, lang, crop_ratio),
        tokenize=True, return_dict=True, return_tensors="pt",
        do_resize=False, state=st, robot_type=ROBOT_TYPE,
    )


def pad_and_stack(items: list[dict], pad_id: int) -> dict:
    """Batch per-sample processor outputs: right-pad input_ids / attention_mask, concatenate the flattened video
    patches + grids, stack state / action_mask. The DiT reads the VLM attention mask for the cache, so right padding
    is exact (tests/test_xr1_tiny.py checks batched == per-sample)."""
    import torch

    L = max(it["input_ids"].shape[1] for it in items)
    ids = torch.full((len(items), L), pad_id, dtype=items[0]["input_ids"].dtype)
    am = torch.zeros((len(items), L), dtype=items[0]["attention_mask"].dtype)
    for j, it in enumerate(items):
        n = it["input_ids"].shape[1]
        ids[j, :n] = it["input_ids"][0]
        am[j, :n] = it["attention_mask"][0]
    return {
        "input_ids": ids, "attention_mask": am,
        "pixel_values_videos": torch.cat([it["pixel_values_videos"] for it in items], 0),
        "video_grid_thw": torch.cat([it["video_grid_thw"] for it in items], 0),
        "state": torch.cat([it["state"] for it in items], 0),
        "action_mask": torch.cat([it["action_mask"] for it in items], 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=22000)
    ap.add_argument("--crop-ratio", type=float, default=0.95)
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--max-batch", type=int, default=16,
                    help="envs per forward pass; 1 = the reference evaluator's per-env calls (much slower)")
    ap.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "eager"],
                    help="MiBoT supports flash_attention_2 or eager only (no sdpa); use eager if flash-attn won't build")
    args = ap.parse_args()

    from pld_rc365.utils.threads import limit_threads

    limit_threads()  # env vars were already set when the pld_rc365 package was imported (before torch)
    import torch

    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "4")))
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, attn_implementation=args.attn, dtype=torch.bfloat16
    ).cuda().to(torch.bfloat16).eval()
    assert ROBOT_TYPE in processor.list_robot_types()

    def to_dev(d):
        return {
            k: (v.to(device=model.device, dtype=model.dtype) if torch.is_floating_point(v) else v.to(model.device))
            if isinstance(v, torch.Tensor) else v
            for k, v in d.items()
        }

    pad_id = processor.tokenizer.pad_token_id

    def handler(batch: list[dict]) -> np.ndarray:
        # Batched forward: a B=1 call costs ~280 ms on a 4090 and is launch-bound (GPU ~5% busy), so the envs that
        # replan together are run as one right-padded batch (same result per env as B=1, up to the noise draw).
        out = []
        for s in range(0, len(batch), args.max_batch):
            part = batch[s:s + args.max_batch]
            encs = [dict(encode_inputs(processor, b["video"], b["state"], b["lang"], args.crop_ratio)) for b in part]
            inputs = to_dev(pad_and_stack(encs, pad_id) if len(encs) > 1 else encs[0])
            with torch.inference_mode():
                actions = model(**inputs, num_steps=args.num_steps, logits_to_keep=1).actions
            actions = processor.decode_action(actions.cpu(), robot_type=ROBOT_TYPE)
            out.append(actions[:, :, :12].float().numpy())
        return np.concatenate(out).astype(np.float32)

    seed = os.environ.get("MIBOT_SERVER_SEED", "7+device")
    Server(args.host, args.port, handler,
           info={"kind": "xr1", "model_path": args.model_path, "seed": seed}).serve_forever()


if __name__ == "__main__":
    main()
