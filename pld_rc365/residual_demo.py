"""Record episodes for inspection videos: the VLA alone and/or VLA -> residual takeover on chosen seeds.

  python -m pld_rc365.residual_demo --model xr1 --task CoffeeSetupMug --tag vla --seeds 200001,200006 --repeats 2
  python -m pld_rc365.residual_demo --model xr1 --task CoffeeSetupMug --tag xi02 --seeds 200001,200006 \
      --t-base 0,100,200 --specialists runs/xr1/CoffeeSetupMug/stage1/specialists_latest.pt --set residual.xi=0.2 ...

Every episode is kept, failures included: <runs>/<model>/<task>/demo/<tag>/raw/ep_*/ holds frames.npy (T+1, 3 cams,
H, W, 3), lowdim.npz (states, executed actions) and trace.npy (per step: t, a_b[:7], a_delta[:7], residual acted,
residual decided at this step). results.jsonl lists them. Without --specialists the VLA runs alone (mode base).
With it, the VLA acts until t_base and the residual's mean action is added from then on (the stage-2 protocol).
scripts/render_demo.py turns the recordings into videos."""

from __future__ import annotations

import json

import numpy as np

from .common import (build_backbone, build_env_and_base, claim_gpu_memory, common_argparser, load_specialists,
                     make_runner)
from .config import load_config, run_dir
from .rollout import EpisodeResult, EpisodeSpec


def main():
    ap = common_argparser("Record VLA-only / residual-takeover episodes on chosen seeds")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", required=True, help="comma-separated env seeds")
    ap.add_argument("--t-base", default="0", help="comma-separated takeover steps (with --specialists)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--specialists", default=None)
    ap.add_argument("--sample", action="store_true", help="sample the residual instead of using its mean")
    ap.add_argument("--no-record", action="store_true", help="evaluation only: keep no frames, just results.jsonl")
    ap.add_argument("--branch-from", default=None,
                    help="tag of an earlier VLA-alone run of this script (one episode per seed, any outcome): for "
                         "every seed and t_base, replay its first t_base VLA actions (same state at t_base) and then "
                         "continue with the live VLA, once alone (branch 'vla') and once + residual (branch 'res'). "
                         "Pairs whose recorded episode ended before t_base are skipped")
    ap.add_argument("--branches", default="vla,res",
                    help="with --branch-from: which branches to run. 'vla' does not depend on --specialists, so for "
                         "several checkpoints on the same seeds / t_base it can run once and the others use 'res'")
    ap.add_argument("--replay-from", default=None,
                    help="tag of an earlier VLA-alone run of this script: instead of querying the VLA, replay the "
                         "actions of the first FAILED episode of each seed there (counterfactual: same VLA actions, "
                         "residual added from t_base on)")
    args = ap.parse_args()
    cfg = load_config(args.model, args.task, args.set)
    rd = run_dir(cfg, "demo") / args.tag
    rd.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    t_bases = [int(x) for x in args.t_base.split(",") if x] if args.specialists else [None]
    plan = [(s, t, r) for r in range(args.repeats) for t in t_bases for s in seeds]
    branch_src = {}
    if args.branch_from:
        if not args.specialists:
            raise SystemExit("--branch-from needs --specialists")
        src = [json.loads(l) for l in (run_dir(cfg, "demo") / args.branch_from / "results.jsonl").read_text().splitlines()]
        for r in src:
            if r["mode"] == "base" and r["seed"] in seeds and r["seed"] not in branch_src:
                branch_src[r["seed"]] = r
        missing = [s for s in seeds if s not in branch_src]
        if missing:   # e.g. the source episode was dropped (non-finite VLA action): those seeds are left out
            print(f"[demo] no VLA-alone episode in {args.branch_from} for seeds {missing}: skipped", flush=True)
            seeds = [s for s in seeds if s in branch_src]
        branches = tuple(b for b in args.branches.split(",") if b)
        if not branches or set(branches) - {"vla", "res"}:
            raise SystemExit(f"--branches must be a subset of vla,res (got {args.branches!r})")
        skipped = [(s, t) for s in seeds for t in t_bases if t > 0 and branch_src[s]["length"] <= t]
        plan = [(s, t, r, b) for r in range(args.repeats) for t in t_bases for s in seeds
                if (s, t) not in skipped for b in branches]
        print(f"[demo] branched: {len(plan)} episodes; {len(skipped)} (seed, t_base) pairs skipped because the "
              f"recorded VLA episode ended before t_base", flush=True)

    rec = not args.no_record
    tdir = rd / "traces"
    if not rec:
        tdir.mkdir(exist_ok=True)
    # replay: the VLA actions come from the recordings, so no VLA server is needed (or contacted)
    venv, base, spec = build_env_and_base(cfg, record_root=str(rd / "raw") if rec else None,
                                          no_base=bool(args.replay_from))
    specs, backbone = {}, None
    if args.specialists:
        backbone, fmap_shape, n_cams = build_backbone(cfg)
        specs = load_specialists(cfg, args.specialists, fmap_shape, n_cams)
        claim_gpu_memory(backbone, specs, venv.num_envs, cfg)
    sources = {}
    if args.replay_from:
        from .base.replay import ReplayBase, actions_from_trace

        src = [json.loads(l) for l in (run_dir(cfg, "demo") / args.replay_from / "results.jsonl").read_text().splitlines()]
        for r in src:
            if r["mode"] == "base" and not r["success"] and r["seed"] in seeds and r["seed"] not in sources:
                sources[r["seed"]] = r
        missing = [s for s in seeds if s not in sources]
        if missing:
            raise SystemExit(f"no failed VLA-alone recording in {args.replay_from} for seeds {missing}")
        base = ReplayBase({s: actions_from_trace(np.load(f"{r['rec_dir']}/trace.npy"))
                           for s, r in sources.items()}, venv.num_envs)
    if branch_src:
        from .base.replay import PrefixBase, actions_from_trace

        def trace_of(r):
            return np.load(r["trace_path"] if r.get("trace_path") else f"{r['rec_dir']}/trace.npy")

        base = PrefixBase(base, {s: actions_from_trace(trace_of(r)) for s, r in branch_src.items()})
    runner = make_runner(cfg, venv, base, backbone, specs, len(spec.stage_names))
    runner.aug_views, runner.store_images = 0, False
    runner.keep_failed_records = True
    it = iter(plan)
    n_done = [0]

    def next_spec():
        p = next(it, None)
        if p is None:
            return None
        if len(p) == 4:                     # branched: same prefix, then VLA alone or VLA + residual
            s, t, r, b = p
            mode = "base" if b == "vla" else "pld"
            return EpisodeSpec(seed=s, mode=mode, t_base=t, deterministic=not args.sample, record=rec,
                               trace=not rec, tag=f"{b}_r{r}")
        s, t, r = p
        if t is None:
            return EpisodeSpec(seed=s, mode="base", record=rec, trace=not rec, tag=f"r{r}")
        return EpisodeSpec(seed=s, mode="pld", t_base=t, deterministic=not args.sample, record=rec, trace=not rec,
                           tag=f"r{r}")

    out = open(rd / "results.jsonl", "a")

    def on_end(r: EpisodeResult) -> bool:
        n_done[0] += 1
        tpath = None
        if r.extra.get("trace") is not None:
            if r.rec_dir is not None:
                tpath = f"{r.rec_dir}/trace.npy"
            else:
                tpath = str(tdir / f"seed{r.spec.seed}_{r.spec.mode}_t{r.spec.t_base}_{r.spec.tag}_{n_done[0]}.npy")
            np.save(tpath, r.extra["trace"])
        branched = r.spec.tag.startswith(("vla_", "res_"))
        rec = {"seed": r.spec.seed, "mode": r.spec.mode,
               "t_base": r.spec.t_base if (r.spec.mode == "pld" or branched) else None,
               "branch": r.spec.tag.split("_")[0] if branched else None, "trace_path": tpath,
               "proprio_at_t_base": r.extra.get("proprio_at_t_base"),
               "branch_of": (branch_src[r.spec.seed].get("trace_path") or branch_src[r.spec.seed]["rec_dir"]) if branched else None,
               "repeat": r.spec.tag, "success": bool(r.success), "length": r.length, "took_over": r.took_over,
               "takeover_step": r.takeover_step, "rec_dir": r.rec_dir, "scene": r.extra.get("scene"),
               "residual": r.extra.get("residual"), "specialists": args.specialists,
               "xi": float(cfg["residual"]["xi"]), "hold": int(cfg["rl"].get("residual_hold", 1)),
               "replay_of": sources[r.spec.seed]["rec_dir"] if r.spec.seed in sources else None}
        out.write(json.dumps(rec, default=float) + "\n")
        out.flush()
        print(f"[demo] {n_done[0]}/{len(plan)} seed {r.spec.seed} mode {r.spec.mode} t_base {rec['t_base']} "
              f"success {r.success} length {r.length}", flush=True)
        return False

    try:
        runner.run(next_spec, on_end)
    finally:
        venv.close()
        out.close()
    print(f"[demo] done: {n_done[0]} episodes -> {rd}", flush=True)


if __name__ == "__main__":
    main()
