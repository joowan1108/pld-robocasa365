"""Stage 2 -- hybrid PLD rollouts -> SFT dataset (PLD §3.2, Alg. 1 "Base policy SFT" block).

Each episode: base policy for t < T_base (T_base ~ U[0, alpha*T], T = task horizon), then a_base + a_delta
(residual mean by default) for t >= T_base; every step (probing included) is recorded with the executed action.
Successful episodes are kept until `stage2.target_successes`; wall-clock marks at 50/100/200. Then nested
LeRobot datasets (first 50 / 100 / 200 kept episodes, in completion order) are written.

Usage:
  python -m pld_rc365.stage2_collect --model xr1 --task CoffeeSetupMug \
      [--specialists runs/xr1/CoffeeSetupMug/stage1/specialists_final.pt]
"""

from __future__ import annotations

import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .common import (build_backbone, build_env_and_base, claim_gpu_memory, common_argparser, load_specialists,
                     make_runner,
                     sample_t_base)
from .config import load_config, run_dir
from .rollout import EpisodeResult, EpisodeSpec
from .utils.timing import JsonlLogger, Timer


def main():
    ap = common_argparser("PLD stage 2: hybrid-rollout data collection")
    ap.add_argument("--specialists", default=None)
    ap.add_argument("--keep-raw", action="store_true", help="keep raw frame memmaps after writing LeRobot data")
    ap.add_argument("--skip-write", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.model, args.task, args.set)
    rd = run_dir(cfg, "stage2")
    s2 = cfg["stage2"]
    raw_root = rd / "raw"
    timer = Timer(rd / "timing.json", extra={"model": args.model, "task": args.task,
                                              "num_envs": int(cfg["env"]["num_envs"]),
                                              "base_replicas": int(cfg["base"].get("replicas", 1))})
    ep_log = JsonlLogger(rd / "episodes.jsonl")

    spec_path = Path(args.specialists) if args.specialists else run_dir(cfg, "stage1") / "specialists_final.pt"
    # stage2.num_envs (optional) overrides env.num_envs: recording envs need more GPU memory than stage-1 envs
    venv, base, spec = build_env_and_base(cfg, record_root=str(raw_root), training=True,
                                          num_envs=s2.get("num_envs") or None)
    timer.set("num_envs_actual", venv.num_envs)
    horizon = venv.horizon
    backbone, fmap_shape, n_cams = build_backbone(cfg)
    specs = load_specialists(cfg, spec_path, fmap_shape, n_cams)
    claim_gpu_memory(backbone, specs, venv.num_envs, cfg)
    runner = make_runner(cfg, venv, base, backbone, specs, len(spec.stage_names))
    runner.keep_pure_base = bool(s2["keep_pure_base"])
    rng = np.random.default_rng(cfg["seeds"]["stage2"])
    seeds = itertools.count(cfg["seeds"]["stage2"])
    target = int(s2["target_successes"])
    marks = sorted(int(m) for m in s2["marks"])
    kept: list[dict] = []
    st = {"episodes": 0, "successes": 0, "pure_base": 0, "t0": time.time()}
    timer.set("horizon", horizon)
    timer.set("probing_alpha", cfg["probing"]["alpha"])

    def next_spec():
        return EpisodeSpec(seed=next(seeds), mode="pld", t_base=sample_t_base(cfg, horizon, rng),
                           deterministic=bool(s2["residual_deterministic"]), record=True)

    def on_end(r: EpisodeResult) -> bool:
        st["episodes"] += 1
        st["successes"] += int(r.success)
        ep_log.log(seed=r.spec.seed, success=r.success, length=r.length, t_base=r.spec.t_base,
                   took_over=r.took_over, takeover_step=r.takeover_step, stage=r.stage_reached,
                   kept=r.rec_dir is not None, wall_s=r.wall_s, scene=r.extra.get("scene"),
                   residual=r.extra.get("residual"))
        if r.rec_dir is not None:
            kept.append({"dir": r.rec_dir, "seed": r.spec.seed, "t_base": r.spec.t_base, "length": r.length,
                         "took_over": r.took_over, "takeover_step": r.takeover_step, "pure_base": not r.took_over})
            st["pure_base"] += int(not r.took_over)
            n = len(kept)
            if n in marks:
                timer.mark(f"successes_{n}")
                timer.set(f"at_{n}", {"episodes": st["episodes"], "hybrid_sr": st["successes"] / st["episodes"],
                                       "pure_base_fraction": st["pure_base"] / n})
        return len(kept) >= target

    try:
        with timer.phase("collect"):
            runner.run(next_spec, on_end)
        (rd / "kept_episodes.json").write_text(json.dumps(kept, indent=2))
        lengths = [k["length"] for k in kept]
        tos = [k["takeover_step"] for k in kept if k["takeover_step"] is not None]
        timer.set("collect", {
            "episodes": st["episodes"], "kept": len(kept), "hybrid_sr": st["successes"] / max(1, st["episodes"]),
            "pure_base_fraction": st["pure_base"] / max(1, len(kept)),
            "mean_length": float(np.mean(lengths)) if lengths else None,
            "mean_takeover_step": float(np.mean(tos)) if tos else None,
            "runner_stats": dict(runner.stats),
        })
    finally:
        venv.close()
        ep_log.close()

    if args.skip_write:
        return
    data_root = Path(cfg["paths"]["data_root"])
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parents[1] / data_root
    for n in marks:
        if len(kept) < n:
            continue
        out = data_root / args.model / f"{args.task}_pld{n}"
        with timer.phase(f"write_lerobot_{n}"):
            from .data.lerobot_writer import write_dataset

            write_dataset([k["dir"] for k in kept[:n]], out, task_name=args.task)
        (out / "pld_episodes.json").write_text(json.dumps(kept[:n], indent=2))
    if not args.keep_raw:
        shutil.rmtree(raw_root, ignore_errors=True)


if __name__ == "__main__":
    main()
