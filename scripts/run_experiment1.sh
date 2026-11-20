#!/usr/bin/env bash
# Experiment 1 (run first): single fixed kitchen -> PLD -> SFT, then ask two questions:
#   (a) does the VLA improve in the same kitchen?                          -> eval axis `fixed`
#   (b) does it keep / improve generalization when the kitchen changes?   -> eval axes `pretrain`, `target` (+ forgetting)
# Default: XR1 x CoffeeSetupMug (fixed scene 11,34; shortest horizon), 200 PLD episodes, single-task LoRA r=32 SFT.
#
# Unattended and resumable: every step is skipped if its result file already exists, and the script stops at the
# first step that fails to produce its result. The base-policy server is started/stopped here (never alongside SFT).
#   source vast.env && bash scripts/run_experiment1.sh     # GPU count / env count auto-detected
# Overrides: GPUS, REPLICAS, NUM_ENVS, ENVS_PER_GPU, PLD_SET="a.b=c ...", PLD_EGL_DEVICES
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
# GPU layout. Stage 1/2 + eval: one base-policy server replica per GPU (env i -> replica i % GPUS, queried
# concurrently), env rendering spread over all GPUs, residual learner on the last GPU. SFT: DDP over all GPUs
# after the servers are stopped. Env count: up to ENVS_PER_GPU per replica, capped by host RAM (~6.5 GB per
# RoboCasa env + ~20 GB for the replay buffer and processes).
REPLICAS="${REPLICAS:-$GPUS}"; (( REPLICAS >= 1 )) || REPLICAS=1
if [[ $MODEL == rldx ]]; then ENVS_PER_GPU="${ENVS_PER_GPU:-3}"; else ENVS_PER_GPU="${ENVS_PER_GPU:-4}"; fi
MEM_GB=$("$PY" - <<'PYMEM'
import os
lim = None
for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # cgroup v2 / v1
    try:
        v = open(f).read().strip()
        lim = None if v == "max" else int(v)
        break
    except Exception:
        pass
tot = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
print(int(min(lim or tot, tot) / 2**30))
PYMEM
)
RAM_ENVS=$(( (MEM_GB - 20) * 10 / 65 )); (( RAM_ENVS >= 1 )) || RAM_ENVS=1
NUM_ENVS="${NUM_ENVS:-$(( ENVS_PER_GPU * REPLICAS < RAM_ENVS ? ENVS_PER_GPU * REPLICAS : RAM_ENVS ))}"
# 1 GPU: leave EGL device selection to robosuite (default device, as on the DSRL runs); 2+: spread envs
if (( GPUS >= 2 )); then export PLD_EGL_DEVICES="${PLD_EGL_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"; fi
DEFAULT_SET="base.replicas=$REPLICAS env.num_envs=$NUM_ENVS rl.device=cuda:$((GPUS > 0 ? GPUS - 1 : 0))"
start_servers() {  # $1 = model path
  for ((r = 0; r < REPLICAS; r++)); do bash scripts/server.sh start "$MODEL" "$1" "$((r % (GPUS > 0 ? GPUS : 1)))" "$r"; done
}
SET=(--set ${PLD_SET:-$DEFAULT_SET})
TAG="sft1_pld${N}"
R="runs/$MODEL/$TASK"
if [[ $MODEL == xr1 ]]; then BASE_MODEL="${XR1_MODEL:-XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365}"
else BASE_MODEL="${RLDX_MODEL:-RLWRLD/RLDX-1-FT-RC365}"; fi
SFT_DIR="runs/$MODEL/_sft/pld${N}_${TASK}"

need() { [[ -e $1 ]] || { echo "[x] step did not produce $1 -- stopping"; exit 1; }; }
step() { echo; echo "=== $(date '+%F %T') $*"; }
echo "[i] $MODEL x $TASK | GPUs=$GPUS replicas=$REPLICAS envs=$NUM_ENVS (host RAM ${MEM_GB} GB) | overrides: ${SET[*]} | EGL: ${PLD_EGL_DEVICES:-default}"

if [[ ! -f $R/eval/base.json || ! -f $R/eval/base_other.json || ! -f $R/stage1/specialists_final.pt \
      || ! -f runs/pld_data/$MODEL/${TASK}_pld${N}/lerobot/meta/info.json ]]; then
  step "start $MODEL base server x$REPLICAS"; start_servers "$BASE_MODEL"
fi
if [[ ! -f $R/eval/base.json ]]; then
  step "base eval (fixed/pretrain/target)"; "$PY" -m pld_rc365.eval --model "$MODEL" --task "$TASK" --tag base "${SET[@]}"
fi; need "$R/eval/base.json"
# Sanity gate before any expensive step: a near-zero base SR almost always means an image/state/action format
# bug in the harness (published XR1 SR on random pretrain kitchens: CoffeeSetupMug 25/50).
BASE_MIN_SR="${BASE_MIN_SR:-0.10}"
"$PY" - "$R/eval/base.json" "$BASE_MIN_SR" <<'PYGATE' || { echo "[x] base SR gate failed -- check $R/eval/base.json; override with BASE_MIN_SR=0"; exit 1; }
import json, sys
d = json.load(open(sys.argv[1])); th = float(sys.argv[2])
sr = {k: v["success_rate"] for k, v in d.items()}
print("[i] base SR:", {k: round(v, 3) for k, v in sr.items()}, "| kitchens:", {k: v.get("n_distinct_scenes") for k, v in d.items()})
sys.exit(0 if sr.get("pretrain", 0.0) >= th or sr.get("fixed", 0.0) >= th else 1)
PYGATE
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
  step "start $MODEL server x$REPLICAS with SFT checkpoint"; start_servers "$SERVE"
  [[ -f $R/eval/$TAG.json ]] || "$PY" -m pld_rc365.eval --model "$MODEL" --task "$TASK" --tag "$TAG" "${SET[@]}"
  [[ -f $R/eval/${TAG}_other.json ]] || "$PY" -m pld_rc365.eval --model "$MODEL" --task "$TASK" --tag "$TAG" --other-tasks "${SET[@]}"
  bash scripts/server.sh stop "$MODEL"
fi
need "$R/eval/$TAG.json"
"$PY" scripts/summarize.py
step "experiment 1 done"
