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
    # Fair baseline for the takeover SR: residual takes over only in episodes still running at T_base, which are
    # harder than average. Apply the same T_base distribution to the base-only episodes (length, success).
    fair = None
    alpha = float(json.loads((rd / "config.json").read_text())["probing"]["alpha"]) if (rd / "config.json").exists() else None
    horizon = timing.get("extra", {}).get("horizon")
    if base_eps and alpha is not None and horizon:
        hi = int(alpha * horizon)
        num = den = 0.0
        for e in base_eps:
            p_alive = min(1.0, max(0, min(e["length"], hi + 1)) / (hi + 1))   # P(T_base < length)
            den += p_alive
            num += p_alive * e["success"]
        fair = num / den if den else None
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
        if fair is not None:
            parts.append(f"fair takeover baseline {fair:.2f}")
        if online:
            parts.append(f"online eps {len(online)} (takeover {len(took)})")
            parts.append(f"rolling SR {roll:.2f}" if roll is not None else "rolling SR -")
            if online[-1].get("rolling_sr_all") is not None:   # the paper's metric: all episodes, window 100
                parts.append(f"all-episode SR {online[-1]['rolling_sr_all']:.2f}")
            if curve:
                parts.append("curve " + " ".join(f"{c['sr']:.2f}" for c in curve[-6:]))
        rec = [e["residual"] for e in took[-50:] if e.get("residual")]
        if rec:
            parts.append(f"residual |a_d|/xi {mean(x['abs_over_xi'] for x in rec):.2f} "
                         f"|a_d|/|a_vla| {mean(x['abs_over_base'] for x in rec):.2f} xi {rec[-1]['xi']:.2f}")
        if online_l:
            l = online_l[-1]
            st = online[-1].get("online_steps") if online else None
            st = sum(st.values()) if isinstance(st, dict) else st
            if st:   # achieved update ratio (the learner is throttled to rl.utd but may also fall behind)
                parts.append(f"UTD {l.get('n_updates', 0) / st:.2f}")
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
        if fair is not None:
            print(f"fair baseline: {fair:.3f}  (base-only SR among episodes still running at T_base ~ U[0, alpha*T]; "
                  f"compare with the takeover rolling SR)")
        print(f"online       : {len(online)} episodes, {len(took)} with residual takeover; "
              f"rolling SR (last 100 takeover eps) = {roll}")
        print(f"online steps : {online_steps}")
        if curve:
            print("SR over time (takeover episodes, binned):")
            for c in curve:
                bar = "#" * int(round(40 * (c["sr"] or 0)))
                print(f"  ep {c['episodes']:6d}  {c['minutes']:7.1f} min  SR {c['sr']:.2f} {bar}")
        rec = [e["residual"] for e in took[-50:] if e.get("residual")]
        if rec:
            names = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]
            avg = lambda key: [mean(x[key][j] for x in rec) for j in range(len(names))]  # noqa: E731
            print(f"residual (last {len(rec)} takeover eps, xi now {rec[-1]['xi']:.3f}):")
            print("   dim   mean  mean|.|  at_bound  |VLA a|")
            for j, nm in enumerate(names):
                print(f"  {nm:>4} {avg('mean')[j]:+.3f}  {avg('mean_abs')[j]:.3f}    {avg('at_bound')[j]:.2f}     {avg('base_abs')[j]:.3f}")
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
