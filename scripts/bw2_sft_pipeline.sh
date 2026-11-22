#!/usr/bin/env bash
# [bw2] Stage 2 + base eval + Stage 3 (SFT on nested 50/100/200 PLD episodes) + eval of every SFT model, for one
# Stage-1 specialist. Resumable: every step is skipped when its result exists. Run inside tmux (session pld_*):
#   tmux new -d -s pld_exp "SPECIALIST=... PLD_SET='...' bash scripts/bw2_sft_pipeline.sh"
# SFT waits for $PLD_HOME/logs/SFT_GO (written after the trainer smoke test); its content may set PER_DEV=<n>.
# Env: SPECIALIST (stage-1 specialists_final.pt), PLD_SET (the stage-1 run's rl/residual overrides), NUM_ENVS,
#      SFT_STEPS (10000), SFT_BS (48), SIZES ("50 100 200"), EVAL_SCENES ("fixed pretrain target"), TAG
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source scripts/bw2_env.sh
MODEL=xr1; TASK=CoffeeSetupMug
SPECIALIST="${SPECIALIST:?path of the stage-1 specialists_final.pt}"
NUM_ENVS="${NUM_ENVS:-24}"; SFT_STEPS="${SFT_STEPS:-10000}"; SFT_BS="${SFT_BS:-48}"
SIZES="${SIZES:-50 100 200}"; EVAL_SCENES="${EVAL_SCENES:-fixed pretrain target}"; TAG="${TAG:-sftexp}"
mkdir -p "$PLD_HOME/logs"
LOG="$PLD_HOME/logs/${TAG}_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "=== $(date '+%F %T') $*"; }
fail() { say "FAILED: $*"; echo "$*" > "$PLD_HOME/logs/${TAG}_FAILED"; bash scripts/server.sh stop "$MODEL" >/dev/null 2>&1; exit 1; }
R=runs/$MODEL/$TASK
DATA=runs/pld_data/$MODEL
SET=(--set base.replicas=1 "base.port=$PLD_XR1_PORT" "env.num_envs=$NUM_ENVS" env.cap_envs_by_gpu_mem=false
     rl.device=cuda:0 ${PLD_SET:-})
say "bw2 SFT pipeline [$TAG]: specialist $SPECIALIST | sizes $SIZES | SFT $SFT_STEPS steps x batch $SFT_BS | commit $(git log --oneline -1)"
nvidia-smi -i "$GPU" --query-gpu=name,memory.used,memory.total --format=csv,noheader
largest=$(echo $SIZES | tr ' ' '\n' | sort -n | tail -1)

need_base_server=0
[[ -f $DATA/${TASK}_pld${largest}/lerobot/meta/info.json ]] || need_base_server=1
[[ -f $R/eval/base.json ]] || need_base_server=1
if (( need_base_server )); then
  ss -ltn | grep -q ":$PLD_XR1_PORT " && fail "port $PLD_XR1_PORT in use"
  say "start base server"; bash scripts/server.sh start "$MODEL" "" "$GPU" 0 || fail "base server"
fi
if [[ ! -f $DATA/${TASK}_pld${largest}/lerobot/meta/info.json ]]; then
  say "stage 2: collect $largest PLD episodes (marks: $SIZES)"
  "$PLD_SIM_PY" -m pld_rc365.stage2_collect --model "$MODEL" --task "$TASK" --specialists "$SPECIALIST" "${SET[@]}" \
      "stage2.num_envs=$NUM_ENVS" "stage2.target_successes=$largest" "stage2.marks=[$(echo $SIZES | tr ' ' ',')]" \
      || fail "stage 2"
fi
for n in $SIZES; do [[ -f $DATA/${TASK}_pld${n}/lerobot/meta/info.json ]] || fail "missing dataset pld$n"; done
if [[ ! -f $R/eval/base.json ]]; then
  say "base eval ($EVAL_SCENES)"
  "$PLD_SIM_PY" -m pld_rc365.eval --model "$MODEL" --task "$TASK" --tag base --scenes $EVAL_SCENES "${SET[@]}" || fail "base eval"
fi
bash scripts/server.sh stop "$MODEL"

say "waiting for $PLD_HOME/logs/SFT_GO"
until [[ -f $PLD_HOME/logs/SFT_GO ]]; do sleep 60; done
PER_DEV=$(grep -o "PER_DEV=[0-9]*" "$PLD_HOME/logs/SFT_GO" | cut -d= -f2); PER_DEV="${PER_DEV:-8}"
for n in $SIZES; do
  SFT_DIR=runs/$MODEL/_sft/pld${n}_${TASK}
  if [[ ! -f $SFT_DIR/timing.json ]] || ! grep -q serve_model_path "$SFT_DIR/timing.json"; then
    say "SFT pld$n: $SFT_STEPS steps, batch $SFT_BS (per-device $PER_DEV), LoRA r32, XR-1 code-base optimiser"
    "$PLD_SIM_PY" -m pld_rc365.sft.run_sft --model "$MODEL" --n "$n" --tasks "$TASK" --python "$PLD_XR1_PY" \
        --base-model-path "$XR1_MODEL" --nproc 1 --max-steps "$SFT_STEPS" --global-batch-size "$SFT_BS" \
        --per-device-batch-size "$PER_DEV" --xr1-defaults || fail "SFT pld$n"
  fi
  if [[ ! -f $R/eval/sft_pld${n}.json ]]; then
    SERVE="$SFT_DIR/merged"; [[ -f $SERVE/config.json ]] || fail "no merged model in $SERVE"
    say "eval SFT pld$n ($EVAL_SCENES)"
    bash scripts/server.sh start "$MODEL" "$REPO/$SERVE" "$GPU" 0 || fail "server with $SERVE"
    "$PLD_SIM_PY" -m pld_rc365.eval --model "$MODEL" --task "$TASK" --tag "sft_pld${n}" --scenes $EVAL_SCENES "${SET[@]}" \
        || fail "eval pld$n"
    bash scripts/server.sh stop "$MODEL"
  fi
done
"$PLD_SIM_PY" scripts/summarize.py || true
date > "$PLD_HOME/logs/${TAG}_DONE"
say "done"
