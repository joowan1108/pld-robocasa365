"""Stage 3 -- distill PLD data back into the (previously frozen) VLA with LoRA r=32 SFT, and time it.

Same budget for every data size (paper: identical training budgets): --max-steps / --global-batch-size.
  RLDX : official trainer (RLDX-1 v1.0.2 rldx/experiment/launch_train.py) with its LoRA flags, run inside the
         RLDX repo via `uv run torchrun`. The FT-RC365 snapshot is copied locally with the `processor/` subfolder
         the trainer requires (rldx/model/core/setup.py loads <ckpt>/processor only).
  XR1  : our trainer pld_rc365/sft/xr1_lora_train.py, run with the XR1 venv's python.

  python -m pld_rc365.sft.run_sft --model rldx --n 200 --rldx-repo ~/RLDX-1 --nproc 8
  python -m pld_rc365.sft.run_sft --model xr1  --n 200 --python ~/venvs/xr1/bin/python --nproc 8
Output: runs/<model>/_sft/pld<N>/ for the pooled 3-task run, runs/<model>/_sft/pld<N>_<Task>/ for a task subset
(e.g. experiment 1: --tasks CoffeeSetupMug). Checkpoint + timing.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ..config import REPO_ROOT, load_config
from ..utils.timing import Timer

TASKS = ("CoffeeSetupMug", "GetToastedBread", "SteamInMicrowave")


def prepare_rldx_ckpt(model_path: str, dst: Path) -> Path:
    from huggingface_hub import snapshot_download

    if (dst / "processor" / "processor_config.json").exists():
        return dst
    src = Path(model_path) if Path(model_path).is_dir() else Path(snapshot_download(model_path))
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        target = dst / f.name
        if f.is_file() and not target.exists():
            try:
                target.symlink_to(f.resolve())
            except OSError:
                shutil.copy2(f, target)
    (dst / "processor").mkdir(exist_ok=True)
    for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
        shutil.copy2(src / name, dst / "processor" / name)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rldx", "xr1"])
    ap.add_argument("--n", type=int, required=True, help="PLD episodes per task (nested subset: 50/100/200)")
    ap.add_argument("--data-name", default="pld", help="dataset family: <Task>_<data-name><n> (e.g. selfboot)")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--global-batch-size", type=int, default=64, help="effective batch per optimizer step")
    ap.add_argument("--per-device-batch-size", type=int, default=2,
                    help="micro-batch per GPU; the rest is gradient accumulation (2 is safe on 24 GB cards)")
    ap.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "eager"], help="XR1 only")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--xr1-defaults", action="store_true",
                    help="XR1 only: the XR-1 code base's post-training optimiser settings (xr1/configs/trainer/"
                         "deepspeed.yaml, Xiaomi-Robotics-1 @0dd7aef): AdamW wd 0.1, 500-step linear warm-up "
                         "5e-7 -> 2e-5, cosine to 5e-6, grad clip 1.0, seed 42. Overrides --lr.")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--rldx-backbone-lora-layers", type=int, default=4,
                    help="top-N LLM layers with LoRA (default 4 = RLDX's default tune_top_llm_layers scope)")
    ap.add_argument("--rldx-repo", default=None)
    ap.add_argument("--base-model-path", default=None,
                    help="local checkpoint dir (e.g. /workspace/checkpoints/...); default: HF id from configs/models")
    ap.add_argument("--python", default="python", help="python of the XR1 venv")
    ap.add_argument("--dry-run", action="store_true", help="print the command only")
    args = ap.parse_args()

    cfg = load_config(args.model, args.tasks[0])
    base_model = args.base_model_path or cfg["base"]["model_path"]
    data_root = Path(cfg["paths"]["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    datasets = [data_root / args.model / f"{t}_{args.data_name}{args.n}" / "lerobot" for t in args.tasks]
    for d in datasets:
        if not (d / "meta" / "info.json").exists() and not args.dry_run:
            raise FileNotFoundError(f"missing stage-2 dataset {d}")
    out = Path(cfg["paths"]["runs_root"])
    # pooled 3-task run -> pld<N>; task subset (e.g. experiment 1) -> pld<N>_<Task>[+<Task>] so they never collide
    dn = f"{args.data_name}{args.n}"
    name = dn if sorted(args.tasks) == sorted(TASKS) else f"{dn}_" + "+".join(args.tasks)
    out = (out if out.is_absolute() else REPO_ROOT / out) / args.model / "_sft" / name
    out.mkdir(parents=True, exist_ok=True)

    micro = args.per_device_batch_size * args.nproc
    if args.global_batch_size % micro:
        raise SystemExit(f"--global-batch-size {args.global_batch_size} must be a multiple of "
                         f"per-device {args.per_device_batch_size} x nproc {args.nproc}")
    accum = args.global_batch_size // micro

    if args.model == "rldx":
        if not args.rldx_repo:
            raise SystemExit("--rldx-repo is required for rldx")
        repo = Path(args.rldx_repo).expanduser().resolve()
        base_ckpt = prepare_rldx_ckpt(base_model, REPO_ROOT / "runs" / "rldx" / "_base_ckpt") \
            if not args.dry_run else Path("<local FT-RC365 copy>")
        venv_py = repo / ".venv" / "bin" / "python"
        launcher = ([str(venv_py), "-m", "torch.distributed.run"] if venv_py.exists() else ["uv", "run", "torchrun"])
        cmd = [*launcher, f"--nproc_per_node={args.nproc}", "rldx/experiment/launch_train.py",
               "--n-cog-tokens", "64", "--video-length", "4",
               "--dataset-paths", *map(str, datasets), "--dataset-mix-ratios", *(["1"] * len(datasets)),
               "--dataset-mode", "standard",
               "--embodiment-tag", "GENERAL_EMBODIMENT",
               "--modality-config-path", "rldx/configs/data/robocasa365_config.py",
               "--state-dropout-prob", "0.0",
               "--base-model-path", str(base_ckpt), "--output-dir", str(out / "ckpt"),
               "--num-gpus", str(args.nproc), "--max-steps", str(args.max_steps),
               "--save-steps", str(args.max_steps), "--save-total-limit", "1",
               "--global-batch-size", str(micro), "--gradient-accumulation-steps", str(accum),
               "--learning-rate", str(args.lr),
               "--action-model-use-lora", "--action-model-lora-rank", str(args.lora_rank),
               "--action-model-lora-alpha", str(args.lora_alpha),
               "--backbone-use-lora", "--backbone-lora-rank", str(args.lora_rank),
               "--backbone-lora-alpha", str(args.lora_alpha),
               "--backbone-lora-num-layers", str(args.rldx_backbone_lora_layers),
               "--experiment-name", f"pld_sft_pld{args.n}"]
        cwd = repo
    else:
        cmd = [args.python, "-m", "torch.distributed.run", f"--nproc_per_node={args.nproc}",
               "-m", "pld_rc365.sft.xr1_lora_train",
               "--data", *map(str, datasets), "--out", str(out),
               "--model-path", base_model, "--max-steps", str(args.max_steps),
               "--global-batch-size", str(args.global_batch_size),
               "--per-device-batch-size", str(args.per_device_batch_size), "--attn", args.attn, "--lr", str(args.lr),
               "--lora-rank", str(args.lora_rank), "--lora-alpha", str(args.lora_alpha)]
        if args.xr1_defaults:
            # peak lr: XR-1's 2e-5 (full fine-tuning) unless PLD_SFT_PEAK_LR is set (e.g. 1e-4 for LoRA, user choice
            # 2026-11-22); warm-up start, cosine floor, wd and seed stay XR-1's
            i = cmd.index("--lr")
            cmd[i + 1] = os.environ.get("PLD_SFT_PEAK_LR", "2e-5")
            cmd += ["--weight-decay", "0.1", "--warmup-steps", "500", "--warmup-lr-start", "5e-7", "--min-lr", "5e-6",
                    "--seed", "42"]
            # kept checkpoints and convergence stop (user request 2026-11-23; not used for the pld50 run):
            # trainable weights every PLD_SFT_SAVE_STEPS steps; stop when the 500-step mean loss has not improved by 2%
            # for 2000 steps (PLD_SFT_EARLY_STOP=0 to disable)
            cmd += ["--save-steps", os.environ.get("PLD_SFT_SAVE_STEPS", "1000")]
            if os.environ.get("PLD_SFT_EARLY_STOP", "1") != "0":
                cmd += ["--es-window", "500", "--es-patience", "2000", "--es-min-delta", "0.02", "--es-min-steps", "2000"]
        cwd = REPO_ROOT

    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    timer = Timer(out / "timing.json", extra={"model": args.model, "n_per_task": args.n, "tasks": args.tasks,
                                             "cmd": cmd, "max_steps": args.max_steps,
                                             "global_batch_size": args.global_batch_size,
                                             "per_device_batch_size": args.per_device_batch_size,
                                             "grad_accum": accum, "nproc": args.nproc})
    t0 = time.time()
    with timer.phase("sft_wall"):
        subprocess.run(cmd, cwd=cwd, check=True)
    timer.set("sft_wall_seconds", time.time() - t0)
    inner = out / "sft_timing.json"
    if inner.exists():
        timer.set("trainer_timing", json.loads(inner.read_text()))
    if args.model == "rldx":
        ckpts = sorted((out / "ckpt").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        timer.set("serve_model_path", str(ckpts[-1]) if ckpts else None)
    else:
        timer.set("serve_model_path", str(out / "merged"))
    print(f"[sft] done in {time.time() - t0:.0f}s -> {timer.data['extra']['serve_model_path']}")


if __name__ == "__main__":
    main()
