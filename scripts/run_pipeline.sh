#!/usr/bin/env bash
# End-to-end PLD pipeline for one model on the three tasks.
#   MODEL=xr1 BASE_PY=~/venvs/xr1/bin/python bash scripts/run_pipeline.sh
#   MODEL=rldx RLDX_REPO=~/RLDX-1 bash scripts/run_pipeline.sh
# Assumes: the PLD venv is active (robocasa 1.0.1 + this package); the base-policy server for $MODEL is already
# running on the port in configs/models/$MODEL.yaml (see README "Servers"); fixed scenes are set in configs/tasks/*.yaml.
set -euo pipefail
MODEL="${MODEL:?set MODEL=rldx|xr1}"
TASKS=(CoffeeSetupMug GetToastedBread SteamInMicrowave)
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"   # GPUs used for SFT (1 or 2 on a 4090/5090 box)
PER_DEV="${PER_DEV:-2}"                                  # micro-batch per GPU (2 fits 24 GB; 4 is ok on 32 GB)
ATTN="${ATTN:-flash_attention_2}"                        # XR1: set ATTN=eager if flash-attn won't build
SFT_STEPS="${SFT_STEPS:-10000}"
SFT_BS="${SFT_BS:-64}"

for T in "${TASKS[@]}"; do
  python -m pld_rc365.eval --model "$MODEL" --task "$T" --tag base   # fixed + pretrain + target scenes
  python -m pld_rc365.stage1_train --model "$MODEL" --task "$T"
  python -m pld_rc365.stage2_collect --model "$MODEL" --task "$T"
done

python -m pld_rc365.eval --model "$MODEL" --task CoffeeSetupMug --tag base --other-tasks

for N in 50 100 200; do
  if [ "$MODEL" = "rldx" ]; then
    python -m pld_rc365.sft.run_sft --model rldx --n "$N" --rldx-repo "${RLDX_REPO:?}" --nproc "$NPROC" \
      --max-steps "$SFT_STEPS" --global-batch-size "$SFT_BS" --per-device-batch-size "$PER_DEV" --attn "$ATTN"
  else
    python -m pld_rc365.sft.run_sft --model xr1 --n "$N" --python "${BASE_PY:?}" --nproc "$NPROC" \
      --max-steps "$SFT_STEPS" --global-batch-size "$SFT_BS" --per-device-batch-size "$PER_DEV" --attn "$ATTN"
  fi
  echo ">>> restart the $MODEL server with --model-path \$(jq -r .extra.serve_model_path runs/$MODEL/_sft/pld$N/timing.json), then press enter"
  read -r
  for T in "${TASKS[@]}"; do
    python -m pld_rc365.eval --model "$MODEL" --task "$T" --tag "sft_pld$N"
  done
  python -m pld_rc365.eval --model "$MODEL" --task CoffeeSetupMug --tag "sft_pld$N" --other-tasks
done
python scripts/summarize.py
