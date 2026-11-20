"""Stage-1 (residual RL) progress report from runs/<model>/<task>/stage1/{timing.json, episodes.jsonl, learner.jsonl}.

    python scripts/progress.py --model xr1 --task CoffeeSetupMug            # human-readable
    python scripts/progress.py --model xr1 --task CoffeeSetupMug --oneline  # one status line (for monitors)
    python scripts/progress.py ... --json out.json                          # curves for plotting

"Is RL working?" reads off three things:
  * base SR in the fixed scene (offline + warm-up episodes, base policy only) vs the rolling SR of online episodes
    where the residual took over (window 100, Fig. 5 of the paper);
  * the same rolling SR binned over time (should rise);
  * learner health: critic loss finite, Q in [0, 1] for sparse 0/1 reward, entropy/alpha moving, xi ramp.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # partially written last line
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xr1")
    ap.add_argument("--task", default="CoffeeSetupMug")
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--oneline", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()
    rd = Path(args.runs) / args.model / args.task / "stage1"
    timing = json.loads((rd / "timing.json").read_text()) if (rd / "timing.json").exists() else {}
    eps = jsonl(rd / "episodes.jsonl")
    lrn = jsonl(rd / "learner.jsonl")
    now = time.time()
    t0 = timing.get("t0_unix", now)
    phases = timing.get("phases", {})
    phase = "not started"
    for name in ("offline_collect", "calql_pretrain", "warmup", "online", "final_eval"):
        if name in phases:
            phase = f"{name} done"
    by = {}
    for e in eps:
        by.setdefault(e["phase"], []).append(e)
    if eps:
        phase = f"{eps[-1]['phase']} (running)" if not phases.get(eps[-1]["phase"]) else phase

    base_eps = by.get("offline", []) + by.get("warmup", [])
    base_sr = mean(e["success"] for e in base_eps)
    online = by.get("online", [])
    took = [e for e in online if e.get("took_over")]
    roll = online[-1]["rolling_sr"] if online else None
    online_steps = online[-1].get("online_steps") if online else None
    # rolling SR over time (takeover episodes, equal-size bins)
    curve = []
    if took:
        n = max(1, len(took) // args.bins)
        for k in range(0, len(took), n):
            chunk = took[k:k + n]
            curve.append({"episodes": k + len(chunk), "sr": mean(e["success"] for e in chunk),
                          "minutes": (chunk[-1]["time"] - t0) / 60})
    last_l = lrn[-1] if lrn else {}
    online_l = [l for l in lrn if l.get("phase") == "online"]
    calql_l = [l for l in lrn if l.get("phase") == "calql"]

    if args.oneline:
        parts = [f"[{(now - t0) / 60:.0f}m] {phase}", f"base SR {base_sr:.2f} ({len(base_eps)} eps)" if base_sr is not None else "base SR -"]
        if online:
            parts.append(f"online eps {len(online)} (takeover {len(took)})")
            parts.append(f"rolling SR {roll:.2f}" if roll is not None else "rolling SR -")
            if curve:
                parts.append("curve " + " ".join(f"{c['sr']:.2f}" for c in curve[-6:]))
        if online_l:
            l = online_l[-1]
            parts.append(f"updates {l.get('n_updates')} critic {l.get('critic_loss', float('nan')):.3g} "
                         f"Q {l.get('q', float('nan')):.2f} alpha {l.get('alpha', float('nan')):.3g}")
        print(" | ".join(parts))
    else:
        print(f"run dir      : {rd}")
        print(f"elapsed      : {(now - t0) / 60:.1f} min   phase: {phase}")
        for k, v in phases.items():
            print(f"  {k:16s} {v['seconds'] / 60:7.1f} min")
        if timing.get("extra", {}).get("offline"):
            o = timing["extra"]["offline"]
            print(f"offline      : {o['successes']}/{o['episodes']} successes, mean success length {o.get('mean_success_length')}")
        print(f"base SR      : {base_sr if base_sr is None else round(base_sr, 3)}  (offline + warm-up, base policy only, {len(base_eps)} eps)")
        print(f"online       : {len(online)} episodes, {len(took)} with residual takeover; "
              f"rolling SR (last 100 takeover eps) = {roll}")
        print(f"online steps : {online_steps}")
        if curve:
            print("SR over time (takeover episodes, binned):")
            for c in curve:
                bar = "#" * int(round(40 * (c["sr"] or 0)))
                print(f"  ep {c['episodes']:6d}  {c['minutes']:7.1f} min  SR {c['sr']:.2f} {bar}")
        if calql_l:
            c = calql_l[-1]
            print(f"Cal-QL       : last step {c.get('step')} critic {c.get('critic_loss')} cql_gap {c.get('cql_gap')} Q {c.get('q')}")
        if online_l:
            print("learner (last 5 logs):")
            for l in online_l[-5:]:
                print("  " + "  ".join(f"{k}={l[k]:.4g}" if isinstance(l.get(k), float) else f"{k}={l.get(k)}"
                                       for k in ("stage", "n_updates", "critic_loss", "q", "td_target", "actor_loss",
                                                 "entropy", "alpha") if k in l))
        print(f"thresholds   : {timing.get('marks', {})}")
    if args.json:
        Path(args.json).write_text(json.dumps({"base_sr": base_sr, "curve": curve, "online": online, "learner": lrn,
                                               "timing": timing}, default=float))


if __name__ == "__main__":
    main()
