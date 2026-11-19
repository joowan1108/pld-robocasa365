"""Compare our env harness (RC365Env) with the official RoboCasa gym wrapper that both RC365 evals use.

Same task, split and per-episode seed -> both must produce identical images (all 3 cameras, upright), state,
instruction, success flags and, after identical 12-D actions, identical next observations. Also reports which
kitchens each scene mode samples and the stage predicates along the way. No model needed.

    MUJOCO_GL=egl python scripts/check_harness.py --task CoffeeSetupMug --steps 30
    python scripts/check_harness.py --task GetToastedBread --scenes-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pld_rc365.envs.rc365_env import CAMERAS, STATE_KEYS, EnvSpec, RC365Env  # noqa: E402

TASK_STAGES = {"CoffeeSetupMug": ["success"],
               "GetToastedBread": ["press", "wait", "pick", "place"],
               "SteamInMicrowave": ["veg_in_bowl", "bowl_in_microwave", "door_closed", "turned_on"]}


def gym_obs_to_pkt(o: dict) -> dict:
    return {
        "images": {c: o[f"video.{c}"] for c in CAMERAS},
        "state": {k: np.asarray(o[f"state.{k}"], np.float32) for k in STATE_KEYS},
        "lang": o["annotation.human.task_description"],
    }


def compare(tag: str, a: dict, b: dict) -> list[str]:
    errs = []
    for c in CAMERAS:
        if a["images"][c].shape != b["images"][c].shape or not np.array_equal(a["images"][c], b["images"][c]):
            d = np.abs(a["images"][c].astype(int) - b["images"][c].astype(int)).max() if a["images"][c].shape == b["images"][c].shape else "shape"
            errs.append(f"{tag}: image {c} differs (max abs {d})")
    for k in STATE_KEYS:
        if not np.allclose(a["state"][k], b["state"][k], atol=1e-6):
            errs.append(f"{tag}: state {k} differs {a['state'][k]} vs {b['state'][k]}")
    if "lang" in a and "lang" in b and a["lang"] != b["lang"]:
        errs.append(f"{tag}: lang differs {a['lang']!r} vs {b['lang']!r}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="CoffeeSetupMug")
    ap.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    ap.add_argument("--seed", type=int, default=157)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--scenes-only", action="store_true")
    ap.add_argument("--resets", type=int, default=4)
    ap.add_argument("--fixed", default="11,34", help="layout,style for the fixed-scene check")
    ap.add_argument("--save", default=None, help="npz with both obs histories (for the XR1 preprocessing check)")
    args = ap.parse_args()
    report: dict = {"task": args.task}
    errs: list[str] = []
    stages = TASK_STAGES.get(args.task, ["success"])

    if not args.scenes_only:
        import gymnasium as gym
        import robocasa.wrappers.gym_wrapper  # noqa: F401  (registers robocasa/<Task>)
        from robocasa.utils.env_utils import convert_action

        # official wrapper first, then ours (sequential: a RoboCasa env takes ~6.5 GB of RAM)
        rng = np.random.default_rng(0)
        acts = []
        for _ in range(args.steps):
            a = np.zeros(12, np.float32)
            a[0:6] = rng.uniform(-0.5, 0.5, 6)
            a[6] = rng.choice([-1.0, 1.0])
            a[11] = -1.0
            acts.append(a)
        t0 = time.time()
        g = gym.make(f"robocasa/{args.task}", split=args.split, seed=7)
        go, _ = g.reset(seed=args.seed)
        hist_g, succ_g = [gym_obs_to_pkt(go)], []
        for a in acts:
            go, _, _, _, info = g.step(convert_action(a))
            hist_g.append(gym_obs_to_pkt(go))
            succ_g.append(bool(info["success"]))
        g.close()
        del g
        report["official_s"] = time.time() - t0

        t0 = time.time()
        ours = RC365Env(EnvSpec(task=args.task, stage_names=stages, scene=args.split))
        po = ours.reset(seed=args.seed)
        hist_p, stages_seen = [po], [po["stage"]]
        errs += compare("reset", po, hist_g[0])
        for t, a in enumerate(acts):
            po = ours.step(a)
            errs += compare(f"step{t}", po, hist_g[t + 1])
            if succ_g[t] != po["success"]:
                errs.append(f"step{t}: success differs")
            hist_p.append(po)
            stages_seen.append(po["stage"])
        report["ours_s"] = time.time() - t0
        report["lang"] = hist_p[0]["lang"]
        report["layout_style"] = [int(x) for x in hist_p[0]["layout_style"]]
        report["stages_seen"] = sorted(set(stages_seen))
        if args.save:
            np.savez_compressed(args.save, lang=hist_p[0]["lang"],
                                **{f"g_{c}": np.stack([h["images"][c] for h in hist_g]) for c in CAMERAS},
                                **{f"p_{c}": np.stack([h["images"][c] for h in hist_p]) for c in CAMERAS},
                                **{f"gs_{k}": np.stack([h["state"][k] for h in hist_g]) for k in STATE_KEYS},
                                **{f"ps_{k}": np.stack([h["state"][k] for h in hist_p]) for k in STATE_KEYS})
        ours.close()
        del ours

    # kitchens sampled per scene mode
    scenes = {}
    L, S = (int(x) for x in args.fixed.split(","))
    for mode in ("fixed", "pretrain", "target"):
        try:
            env = RC365Env(EnvSpec(task=args.task, stage_names=stages, scene=mode, layout_id=L, style_id=S))
            ks = [tuple(int(x) for x in env.reset(seed=1000 + i)["layout_style"]) for i in range(args.resets)]
            env.close()
            scenes[mode] = ks
        except Exception as e:  # noqa: BLE001
            scenes[mode] = f"ERROR {e!r}"
    report["scenes"] = {k: [list(x) for x in v] if isinstance(v, list) else v for k, v in scenes.items()}
    if isinstance(scenes.get("fixed"), list) and len(set(scenes["fixed"])) != 1:
        errs.append(f"fixed scene not fixed: {scenes['fixed']}")
    report["errors"] = errs[:20]
    report["n_errors"] = len(errs)
    print(json.dumps(report, indent=2))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
