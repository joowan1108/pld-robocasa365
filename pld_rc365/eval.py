"""Success-rate evaluation on a fixed seed list.

Evaluates whatever the base-policy server currently serves (the original FT-RC365 checkpoint, or an SFT
checkpoint served by the same server script), optionally with Stage-1 specialists on top (clean pi_bar eval:
no probing, residual mean).

Scene axes (default: all three, 50 episodes each, same seed list):
  fixed    : the single (layout, style) used in Stage 1/2 -- does PLD work in the one scene it was trained in?
  pretrain : random RC365 pretrain kitchens (layouts/styles 11-60, pretrain objects) -- other training-distribution scenes
  target   : RC365 target split (held-out kitchens 1-10 + held-out object instances) -- unseen kitchens/objects (OOD)

  # base VLA
  python -m pld_rc365.eval --model xr1 --task CoffeeSetupMug --tag base
  # SFT model (start the server with --model-path <sft ckpt> first)
  python -m pld_rc365.eval --model xr1 --task CoffeeSetupMug --tag sft_pld200
  # forgetting check on untouched RC365 tasks (random pretrain scenes); run for base and every SFT model
  python -m pld_rc365.eval --model xr1 --task CoffeeSetupMug --tag sft_pld200 --other-tasks
"""

from __future__ import annotations

import json

import numpy as np

from .common import (build_backbone, build_env_and_base, claim_gpu_memory, common_argparser, load_specialists,
                     make_runner)
from .config import load_config, run_dir
from .rollout import EpisodeResult, EpisodeSpec
from .utils.timing import Timer


def evaluate(cfg, scene: str, task: str | None, episodes: int, seed_base: int, specialists_path: str | None):
    venv, base, spec = build_env_and_base(cfg, scene=scene, task=task)
    specs, backbone = {}, None
    if specialists_path:
        backbone, fmap_shape, n_cams = build_backbone(cfg)
        specs = load_specialists(cfg, specialists_path, fmap_shape, n_cams)
        claim_gpu_memory(backbone, specs, venv.num_envs, cfg)
    runner = make_runner(cfg, venv, base, backbone, specs, len(spec.stage_names))
    runner.aug_views, runner.store_images = 0, False   # nothing is stored here: no augmented views to encode
    seeds = iter(range(seed_base, seed_base + episodes))
    res: list[EpisodeResult] = []

    def next_spec():
        s = next(seeds, None)
        if s is None:
            return None
        if specs:
            return EpisodeSpec(seed=s, mode="pld", t_base=0, deterministic=True)
        return EpisodeSpec(seed=s, mode="base")

    try:
        runner.run(next_spec, lambda r: res.append(r) or False)
    finally:
        venv.close()
    succ = [r for r in res if r.success]
    scenes = [r.extra.get("scene") for r in res]
    per_scene: dict[str, list[bool]] = {}
    for r, sc in zip(res, scenes):
        per_scene.setdefault(str(sc), []).append(r.success)
    return {
        # robocasa samples (layout, style) per reset; fixed -> 1 scene, pretrain/target -> many
        "n_distinct_scenes": len({sc for sc in scenes if sc is not None}),
        "per_scene_sr": {k: {"episodes": len(v), "success_rate": float(np.mean(v))} for k, v in per_scene.items()},
        "task": spec.task, "scene": scene, "episodes": len(res),
        "success_rate": float(np.mean([r.success for r in res])) if res else None,
        "mean_success_length": float(np.mean([r.length for r in succ])) if succ else None,
        "mean_stage_reached": float(np.mean([r.stage_reached for r in res])) if res else None,
        "server": base.info, "per_episode": [{"seed": r.spec.seed, "success": r.success, "length": r.length,
                                             "stage": r.stage_reached, "scene": r.extra.get("scene")} for r in res],
    }


def main():
    ap = common_argparser("Evaluate base / SFT VLA (optionally + specialists)")
    ap.add_argument("--tag", required=True, help="name for this evaluation (e.g. base, sft_pld100)")
    ap.add_argument("--scenes", nargs="+", default=["fixed", "pretrain", "target"],
                    choices=["fixed", "pretrain", "target"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--specialists", default=None)
    ap.add_argument("--other-tasks", action="store_true", help="evaluate cfg.eval.forgetting_tasks instead")
    args = ap.parse_args()
    cfg = load_config(args.model, args.task, args.set)
    rd = run_dir(cfg, "eval")
    n = int(args.episodes or cfg["eval"]["episodes"])
    timer = Timer(rd / f"{args.tag}_timing.json", extra={"tag": args.tag})
    out = {}
    if args.other_tasks:
        for t in cfg["eval"]["forgetting_tasks"]:
            with timer.phase(f"{t}_pretrain"):
                out[f"{t}/pretrain"] = evaluate(cfg, "pretrain", t, n, cfg["eval"]["seed_base"], None)
    else:
        for sc in args.scenes:
            with timer.phase(sc):
                out[sc] = evaluate(cfg, sc, None, n, cfg["eval"]["seed_base"], args.specialists)
    path = rd / f"{args.tag}{'_other' if args.other_tasks else ''}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    for k, v in out.items():
        print(f"{k}: SR={v['success_rate']:.3f} ({v['episodes']} eps, {v['n_distinct_scenes']} distinct kitchens)")


if __name__ == "__main__":
    main()
