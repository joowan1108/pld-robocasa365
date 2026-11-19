"""Stage 0 -- pick the fixed (layout, style) per task.

Runs the base VLA for a few episodes in each candidate pretrain-split kitchen and prints success rates.
Choose a scene where BOTH models are in roughly 15-70% (room for the residual to help, enough successes for the
offline buffer), then write it into configs/tasks/<task>.yaml (layout_id / style_id).

  python -m pld_rc365.stage0_sweep --model xr1 --task GetToastedBread --episodes 10
"""

from __future__ import annotations

import json

from .config import load_config, run_dir
from .common import common_argparser
from .eval import evaluate

# robocasa env_utils.KITCHEN_SCENES_5X5 (pretrain-split layouts/styles used by the RC365 MimicGen subsets)
CANDIDATES = [(l, s) for l in (11, 15, 18, 40, 50) for s in (14, 28, 34, 46, 58)]


def main():
    ap = common_argparser("Scene sweep for the fixed-scene setting")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--candidates", nargs="*", default=None, help="L,S pairs, e.g. 11,34 15,14")
    args = ap.parse_args()
    cands = [tuple(int(x) for x in c.split(",")) for c in args.candidates] if args.candidates else CANDIDATES
    out = {}
    for (l, s) in cands:
        cfg = load_config(args.model, args.task, args.set + [f"task.layout_id={l}", f"task.style_id={s}"])
        try:
            r = evaluate(cfg, "fixed", None, args.episodes, cfg["eval"]["seed_base"] + 50_000, None)
            out[f"{l},{s}"] = {"success_rate": r["success_rate"], "mean_success_length": r["mean_success_length"]}
        except Exception as e:  # noqa: BLE001  (e.g. layout excluded for this task)
            out[f"{l},{s}"] = {"error": repr(e)}
        print(f"layout={l} style={s}: {out[f'{l},{s}']}", flush=True)
    rd = run_dir(load_config(args.model, args.task, args.set + ["task.layout_id=0", "task.style_id=0"]), "stage0")
    (rd / "sweep.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
