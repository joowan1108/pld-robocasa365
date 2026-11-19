#!/usr/bin/env bash
# Experiment 1 (run first): single fixed kitchen -> PLD -> SFT, then ask two questions:
#   (a) does the VLA improve in the same kitchen?                          -> eval axis `fixed`
#   (b) does it keep / improve generalization when the kitchen changes?   -> eval axes `pretrain`, `target` (+ forgetting)
# Default: XR1 x CoffeeSetupMug (fixed scene 11,34; shortest horizon), 200 PLD episodes, single-task LoRA r=32 SFT.
#
# Unattended and resumable: every step is skipped if its result file already exists, and the script stops at the
# first step that fails to produce its result. The base-policy server is started/stopped here (never alongside SFT).
#   source vast.env && bash scripts/run_experiment1.sh     # GPU count auto-detected (1 or 2x RTX 4090)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
MODEL="${MODEL:-xr1}"
TASK="${TASK:-CoffeeSetupMug}"
N="${N:-200}"
GPUS="${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
SFT_STEPS="${SFT_STEPS:-3000}"   # experiment-1 budget (~2.4 epochs of 200 CoffeeSetupMug episodes)
SFT_BS="${SFT_BS:-32}"
PER_DEV="${PER_DEV:-2}"          # 24 GB: try 2, fall back to 1 if XR1 SFT OOMs (or rent a 48 GB card for SFT)
ATTN="${ATTN:-flash_attention_2}"
PY="${PLD_SIM_PY:-python}"
# GPU layout: GPU0 = VLA server. With 2 GPUs the env rendering + residual learner move to GPU1 (no memory
# sharing with the server during rollouts); SFT runs DDP on both after the server is stopped.
if (( GPUS >= 2 )); then
  export PLD_EGL_DEVICES="${PLD_EGL_DEVICES:-1}"
  DEFAULT_SET="rl.device=cuda:1"
else
  DEFAULT_SET=""
fi
SET=(--set ${PLD_SET:-$DEFAULT_SET})
TAG="sft1_pld${N}"
R="runs/$MODEL/$TASK"
if [[ $MODEL == xr1 ]]; then BASE_MODEL="${XR1_MODEL:-XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365}"
else BASE_MODEL="${RLDX_MODEL:-RLWRLD/RLDX-1-FT-RC365}"; fi
SFT_DIR="runs/$MODEL/_sft/pld${N}_${TASK}"

need() { [[ -e $1 ]] || { echo "[x] step did not produce $1 -- stopping"; exit 1; }; }
step() { echo; echo "=== $(date '+%F %T') $*"; }
echo "[i] $MODEL x $TASK, GPUs=$GPUS, PLD overrides: ${SET[*]}, EGL devices: ${PLD_EGL_DEVICES:-default}"

if [[ ! -f $R/eval/base.json || ! -f $R/eval/base_other.json || ! -f $R/stage1/specialists_final.pt \
      || ! -f runs/pld_data/$MODEL/${TASK}_pld${N}/lerobot/meta/info.json ]]; then
  step "start $MODEL base server"; bash scripts/server.sh start "$MODEL" "$BASE_MODEL" 0
fi
if [[ ! -f $R/eval/base.json ]]; then
  step "base eval (fixed/pretrain/target)"; "$PY" -m pld_rc365.eval --model "$MODEL" --task "$TASK" --tag base "${SET[@]}"
fi; need "$R/eval/base.json"
if [[ ! -f $R/eval/base_other.json ]]; then
  step "base forgetting eval"; "$PY" -m pld_rc365.eval --model "$MODEL" --task "$TASK" --tag base --other-tasks "${SET[@]}"
fi; need "$R/eval/base_other.json"
if [[ ! -f $R/stage1/specialists_final.pt ]]; then
  step "stage 1"; "$PY" -m pld_rc365.stage1_train --model "$MODEL" --task "$TASK" "${SET[@]}"
fi; need "$R/stage1/specialists_final.pt"
if [[ ! -f runs/pld_data/$MODEL/${TASK}_pld${N}/lerobot/meta/info.json ]]; then
  step "stage 2"; "$PY" -m pld_rc365.stage2_collect --model "$MODEL" --task "$TASK" "${SET[@]}" "stage2.target_successes=$N"
fi; need "runs/pld_data/$MODEL/${TASK}_pld${N}/lerobot/meta/info.json"
bash scripts/server.sh stop "$MODEL"

if [[ $MODEL == xr1 ]]; then SERVE="$SFT_DIR/merged"; else SERVE=""; fi
if [[ ! -f $SFT_DIR/timing.json ]] || ! grep -q serve_model_path "$SFT_DIR/timing.json"; then
  step "stage 3 SFT ($GPUS GPU, eff. batch $SFT_BS, $SFT_STEPS steps, per-device $PER_DEV)"
  EXTRA=(); [[ $MODEL == rldx ]] && EXTRA=(--rldx-repo "${RLDX_REPO:?}") || EXTRA=(--python "${PLD_XR1_PY:-python}")
  "$PY" -m pld_rc365.sft.run_sft --model "$MODEL" --n "$N" --tasks "$TASK" "${EXTRA[@]}" --base-model-path "$BASE_MODEL" \
    --nproc "$GPUS" --max-steps "$SFT_STEPS" --global-batch-size "$SFT_BS" --per-device-batch-size "$PER_DEV" --attn "$ATTN"
fi
SERVE="$("$PY" -c "import json;print(json.load(open('$SFT_DIR/timing.json'))['extra']['serve_model_path'])")"
need "$SERVE"

if [[ ! -f $R/eval/$TAG.json || ! -f $R/eval/${TAG}_other.json ]]; then
  step "start $MODEL server with SFT checkpoint"; bash scripts/server.sh start "$MODEL" "$SERVE" 0
  [[ -f $R/eval/$TAG.json ]] || "$PY" -m pld_rc365.eval --model "$MODEL" --task "$TASK" --tag "$TAG" "${SET[@]}"
  [[ -f $R/eval/${TAG}_other.json ]] || "$PY" -m pld_rc365.eval --model "$MODEL" --task "$TASK" --tag "$TAG" --other-tasks "${SET[@]}"
  bash scripts/server.sh stop "$MODEL"
fi
need "$R/eval/$TAG.json"
"$PY" scripts/summarize.py
step "experiment 1 done"
