"""Check that our XR1 server builds exactly the same model inputs as the official evaluator.

Feeds the obs histories saved by `scripts/check_harness.py --save` through
  official: Xiaomi-Robotics-1 eval_robocasa365/entry.py (history queues, sample_history, observation_to_state,
            EvalClient._build_messages, processor.apply_chat_template)  on the *gym wrapper* observations
  ours    : pld_rc365.base.client.BasePolicy history selection + pld_rc365.base.xr1_server.encode_inputs  on
            *our harness* packets
at every replanning step (every 16 env steps) and compares all tensors. Needs only the processor files.

    python scripts/check_xr1_inputs.py --npz runs/logs/harness_coffee.npz \
        --entry /opt/src/Xiaomi-Robotics-1/eval_robocasa365 --model-path $XR1_MODEL
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pld_rc365.base.client import BasePolicy  # noqa: E402
from pld_rc365.base.xr1_server import encode_inputs  # noqa: E402

CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
SKEYS = ("end_effector_position_relative", "end_effector_rotation_relative", "gripper_qpos", "base_position",
         "base_rotation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--entry", required=True, help="dir containing the official entry.py (and ../deploy/client.py)")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--replan", type=int, default=16)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor

    sys.path.insert(0, str(Path(args.entry).resolve().parent / "deploy"))
    sys.path.insert(0, str(Path(args.entry).resolve()))
    import entry  # official evaluator module

    proc = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    d = np.load(args.npz)
    T = d[f"g_{CAMS[0]}"].shape[0]
    lang = str(d["lang"])

    # official client object without opening a connection
    client = object.__new__(entry.EvalClient)
    client.processor, client.robot_type, client.crop_ratio = proc, entry.ROBOT_TYPE, 0.95

    # ours: BasePolicy bookkeeping without a server
    bp = object.__new__(BasePolicy)
    bp.hist_idx, bp.maxlen = [-6, -4, -2, 0], 7
    bp.frames, bp.states = [collections.deque(maxlen=7)], [collections.deque(maxlen=7)]
    bp.langs = [lang]

    qlen = (4 - 1) * 2 + 1
    img_q = {f"video.{c}": collections.deque(maxlen=qlen) for c in CAMS}
    st_q = collections.deque(maxlen=qlen)
    n_cmp, bad = 0, []
    for t in range(T):
        gobs = {f"video.{c}": d[f"g_{c}"][t] for c in CAMS}
        gobs.update({f"state.{k}": d[f"gs_{k}"][t] for k in SKEYS})
        for k, v in entry.collect_images(gobs).items():
            img_q[k].append(v)
        st_q.append(entry.observation_to_state(gobs))
        bp.frames[0].append({c: d[f"p_{c}"][t] for c in CAMS})
        bp.states[0].append({k: d[f"ps_{k}"][t] for k in SKEYS})
        if t % args.replan:
            continue
        # official
        states = entry.sample_history(st_q, 4, 2)
        images = {k: entry.sample_history(q, 4, 2) for k, q in img_q.items()}
        st = np.zeros((1, 4, entry.STATE_DIM), np.float32)
        st[0, :, :14] = states
        ref = proc.apply_chat_template(client._build_messages(images, lang), tokenize=True, return_dict=True,
                                       return_tensors="pt", do_resize=False, state=st, robot_type=entry.ROBOT_TYPE)
        # ours
        h = bp._history(0)
        mine = encode_inputs(proc, h["video"], h["state"], h["lang"], 0.95)
        for k in ref:
            a, b = ref[k], mine[k]
            same = torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
            if not same:
                bad.append(f"t={t} {k}: differs")
        n_cmp += 1
    print(f"compared {n_cmp} replanning steps, keys={list(ref.keys())}")
    print("[ok] identical XR1 inputs" if not bad else "\n".join(bad[:20]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
