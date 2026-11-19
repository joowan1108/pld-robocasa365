"""Collect Stage 1/2/3 timings and success rates into markdown tables (and summary.json).

  python scripts/summarize.py [--runs runs] [--out runs/summary.md]

Expected eval tags (pld_rc365.eval --tag ...): `base` for the original checkpoint, `sft_pld{N}` for SFT models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ("rldx", "xr1")
TASKS = ("CoffeeSetupMug", "GetToastedBread", "SteamInMicrowave")


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def hms(s):
    if s is None:
        return "-"
    s = float(s)
    return f"{s/3600:.2f}h" if s >= 3600 else f"{s/60:.1f}m"


def pct(x):
    return "-" if x is None else f"{100*x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parents[1] / "runs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 200])
    args = ap.parse_args()
    runs = Path(args.runs)
    out_md, summary = [], {"stage1": [], "stage2": [], "stage3": []}

    out_md.append("## Stage 1 — residual specialist (1 task, fixed scene)\n")
    out_md.append("| model | task | base SR (fixed) | offline collect | Cal-QL | warm-up | online | **total** | "
                  "t→80% | t→90% | t→95% | stop | π̄ clean SR |")
    out_md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        for t in TASKS:
            tj = load(runs / m / t / "stage1" / "timing.json")
            if not tj:
                continue
            ph, ex, mk = tj["phases"], tj["extra"], tj["marks"]
            g = lambda n: ph.get(n, {}).get("seconds")  # noqa: E731
            row = {
                "model": m, "task": t, "base_sr": ex.get("offline", {}).get("base_sr_fixed_scene"),
                "offline": g("offline_collect"), "calql": g("calql_pretrain"), "warmup": g("warmup"),
                "online": g("online"), "total": ex.get("stage1_total_seconds"),
                **{f"t{th}": mk.get(f"sr>={th}_since_stage1_start") for th in (0.8, 0.9, 0.95)},
                "stop": ex.get("online", {}).get("stop_reason"),
                "clean_sr": ex.get("final_eval", {}).get("success_rate"),
            }
            summary["stage1"].append(row)
            out_md.append(f"| {m} | {t} | {pct(row['base_sr'])} | {hms(row['offline'])} | {hms(row['calql'])} | "
                          f"{hms(row['warmup'])} | {hms(row['online'])} | **{hms(row['total'])}** | "
                          f"{hms(row['t0.8'])} | {hms(row['t0.9'])} | {hms(row['t0.95'])} | {row['stop']} | "
                          f"{pct(row['clean_sr'])} |")

    out_md.append("\n## Stage 2 — hybrid rollouts → SFT data\n")
    sz = args.sizes
    out_md.append("| model | task | " + " | ".join(f"t→{n} succ" for n in sz) + " | hybrid SR | pure-base frac | "
                  "mean takeover step | LeRobot write |")
    out_md.append("|---|---|" + "---|" * len(sz) + "---|---|---|---|")
    for m in MODELS:
        for t in TASKS:
            tj = load(runs / m / t / "stage2" / "timing.json")
            if not tj:
                continue
            c = tj["extra"].get("collect", {})
            w = sum(v["seconds"] for k, v in tj["phases"].items() if k.startswith("write_lerobot"))
            row = {"model": m, "task": t, **{f"t{n}": tj["marks"].get(f"successes_{n}") for n in sz},
                   "hybrid_sr": c.get("hybrid_sr"), "pure_base": c.get("pure_base_fraction"),
                   "takeover": c.get("mean_takeover_step"), "write": w}
            summary["stage2"].append(row)
            out_md.append(f"| {m} | {t} | " + " | ".join(hms(row[f't{n}']) for n in sz) +
                          f" | {pct(row['hybrid_sr'])} | {pct(row['pure_base'])} | "
                          f"{'-' if row['takeover'] is None else round(row['takeover'])} | {hms(w)} |")

    scenes = (("fixed", "fixed scene (in-dist.)"), ("pretrain", "random pretrain scenes"),
              ("target", "target split (unseen kitchens+objects)"))
    out_md.append("\n## Stage 3 — LoRA r=32 SFT of the VLA on PLD data (3 tasks pooled)\n")
    out_md.append("SR shown as base → SFT (Δ). `fixed` answers *does PLD work in the single training scene*; "
                  "`target` is the OOD axis.\n")
    out_md.append("| model | PLD/task | SFT wall | task | " + " | ".join(lbl for _, lbl in scenes) + " |")
    out_md.append("|---|---|---|---|" + "---|" * len(scenes))

    def cell(b, s_, sc):
        sb, ss = (b or {}).get(sc, {}).get("success_rate"), (s_ or {}).get(sc, {}).get("success_rate")
        d = "" if sb is None or ss is None else f" ({100*(ss-sb):+.1f}pp)"
        return f"{pct(sb)} → {pct(ss)}{d}"

    for m in MODELS:
        for n in sz:
            tj = load(runs / m / "_sft" / f"pld{n}" / "timing.json")
            if not tj:
                continue
            for t in TASKS:
                b = load(runs / m / t / "eval" / "base.json")
                s_ = load(runs / m / t / "eval" / f"sft_pld{n}.json")
                summary["stage3"].append({"model": m, "n": n, "task": t,
                                          "sft_wall": tj["extra"].get("sft_wall_seconds"),
                                          **{f"base_{sc}": (b or {}).get(sc, {}).get("success_rate") for sc, _ in scenes},
                                          **{f"sft_{sc}": (s_ or {}).get(sc, {}).get("success_rate") for sc, _ in scenes}})
                out_md.append(f"| {m} | {n} | {hms(tj['extra'].get('sft_wall_seconds'))} | {t} | "
                              + " | ".join(cell(b, s_, sc) for sc, _ in scenes) + " |")

    out_md.append("\n## Forgetting check — untouched RC365 tasks (random pretrain scenes)\n")
    rows = []
    for m in MODELS:
        b = load(runs / m / "CoffeeSetupMug" / "eval" / "base_other.json")
        if not b:
            continue
        tags = [n for n in sz if (runs / m / "CoffeeSetupMug" / "eval" / f"sft_pld{n}_other.json").exists()]
        sfts = {n: load(runs / m / "CoffeeSetupMug" / "eval" / f"sft_pld{n}_other.json") for n in tags}
        if not rows:
            out_md.append("| model | task | base | " + " | ".join(f"SFT pld{n}" for n in sz) + " |")
            out_md.append("|---|---|---|" + "---|" * len(sz))
        for key in b:
            r = [pct(b[key]["success_rate"])] + [pct(sfts[n][key]["success_rate"]) if n in sfts and key in sfts[n]
                                                 else "-" for n in sz]
            rows.append(key)
            out_md.append(f"| {m} | {key.split('/')[0]} | " + " | ".join(r) + " |")

    out_md.append("\n## All evaluations (every eval tag, e.g. experiment 1 `sft1_pld200`)\n")
    out_md.append("| model | task | tag | fixed (in-dist.) | random pretrain | target (OOD) | distinct kitchens f/p/t |")
    out_md.append("|---|---|---|---|---|---|---|")
    summary["evals"] = []
    for m in MODELS:
        for t in TASKS:
            ev = runs / m / t / "eval"
            for f in sorted(ev.glob("*.json")) if ev.exists() else []:
                if f.stem.endswith("_timing") or f.stem.endswith("_other"):
                    continue
                d = load(f)
                sr = {sc: d.get(sc, {}).get("success_rate") for sc in ("fixed", "pretrain", "target")}
                nk = "/".join(str(d.get(sc, {}).get("n_distinct_scenes", "-")) for sc in ("fixed", "pretrain", "target"))
                summary["evals"].append({"model": m, "task": t, "tag": f.stem, **sr})
                out_md.append(f"| {m} | {t} | {f.stem} | {pct(sr['fixed'])} | {pct(sr['pretrain'])} | "
                              f"{pct(sr['target'])} | {nk} |")

    text = "\n".join(out_md) + "\n"
    print(text)
    out = Path(args.out) if args.out else runs / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
