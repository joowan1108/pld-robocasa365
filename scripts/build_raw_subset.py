"""Write nested LeRobot datasets from a subset of a stage-2 run's raw episode recordings.

Used for the paper's "self-bootstrap" comparison data (D^Base Policy: successful rollouts of the base policy alone):
the kept episodes of a stage-2 run where the VLA succeeded before the residual took over.

  python scripts/build_raw_subset.py --run runs/xr1/CoffeeSetupMug/stage2_alpha08_withpurebase_1254 \
      --select pure_base --name selfboot --sizes 50 100
Output: runs/pld_data/xr1/CoffeeSetupMug_<name><n>/lerobot (+ episodes.json with the selected episodes)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="stage-2 run dir with episodes.jsonl and raw/")
    ap.add_argument("--select", choices=["pure_base", "took_over", "all"], default="pure_base")
    ap.add_argument("--name", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--model", default="xr1")
    ap.add_argument("--task", default="CoffeeSetupMug")
    args = ap.parse_args()
    from pld_rc365.data.lerobot_writer import write_dataset

    run = Path(args.run)
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    kept = [e for e in eps if e.get("kept")]
    if args.select == "pure_base":
        sel = [e for e in kept if not e["took_over"]]
    elif args.select == "took_over":
        sel = [e for e in kept if e["took_over"]]
    else:
        sel = kept
    rows = []
    for e in sel:                                   # completion order, as stage 2 writes its nested subsets
        dirs = sorted((run / "raw").glob(f"ep_{e['seed']}_*"))
        if len(dirs) != 1:
            raise SystemExit(f"seed {e['seed']}: expected one raw dir, found {len(dirs)}")
        rows.append({**e, "dir": str(dirs[0])})
    print(f"{len(kept)} kept episodes, {len(rows)} selected ({args.select})", flush=True)
    for n in args.sizes:
        if len(rows) < n:
            print(f"only {len(rows)} episodes: skipping size {n}", flush=True)
            continue
        out = REPO / "runs" / "pld_data" / args.model / f"{args.task}_{args.name}{n}"
        t0 = time.time()
        write_dataset([r["dir"] for r in rows[:n]], out, task_name=args.task)
        (out / "episodes.json").write_text(json.dumps(rows[:n], indent=2))
        L = [r["length"] for r in rows[:n]]
        print(f"wrote {out} ({n} episodes, mean length {sum(L) / n:.0f}) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
