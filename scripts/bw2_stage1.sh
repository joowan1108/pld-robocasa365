#!/usr/bin/env bash
# [bw2] Stage 1 (residual RL) for one model x task on the shared B200 server. Run inside a tmux session named pld_*:
#   tmux new -d -s pld_s1 "bash scripts/bw2_stage1.sh"
# Starts the base-policy server on GPU 5 (port PLD_XR1_PORT), runs stage 1, stops the server (by PID file).
# Overrides: MODEL TASK NUM_ENVS PLD_SET="a.b=c ..." TAG PLD_XR1_PORT   Log: $PLD_HOME/logs/stage1[_TAG]_<start time>.log
# Several runs at once: one clone of the repo per run (each has its own runs/ and server PID files), each with its
# own TAG, tmux session and PLD_XR1_PORT, e.g.
#   cd ~/jwjw/pld/residual_xi05 && tmux new -d -s pld_s2 "TAG=xi05 PLD_XR1_PORT=10610 PLD_SET='residual.xi=0.5' bash scripts/bw2_stage1.sh"
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source scripts/bw2_env.sh
MODEL="${MODEL:-xr1}"; TASK="${TASK:-CoffeeSetupMug}"; NUM_ENVS="${NUM_ENVS:-8}"
mkdir -p "$PLD_HOME/logs"
TAG="${TAG:-}"; SFX="${TAG:+_$TAG}"
LOG="$PLD_HOME/logs/stage1${SFX}_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "=== $(date '+%F %T') $*"; }
say "bw2 stage 1${TAG:+ [$TAG]}: $MODEL x $TASK, $NUM_ENVS envs, GPU $GPU, port $PLD_XR1_PORT, repo $REPO, commit $(git log --oneline -1)"
nvidia-smi -i "$GPU" --query-gpu=name,memory.used,memory.total --format=csv,noheader
if ss -ltn | grep -q ":$PLD_XR1_PORT "; then echo "[x] port $PLD_XR1_PORT is in use"; exit 1; fi
if [[ -e runs/$MODEL/$TASK/stage1 ]]; then echo "[x] runs/$MODEL/$TASK/stage1 exists -- archive it first"; exit 1; fi
say "start $MODEL base server"
bash scripts/server.sh start "$MODEL" "" "$GPU" 0 || exit 1
say "stage 1"
"$PLD_SIM_PY" -m pld_rc365.stage1_train --model "$MODEL" --task "$TASK" --set base.replicas=1 "base.port=$PLD_XR1_PORT" \
    "env.num_envs=$NUM_ENVS" env.cap_envs_by_gpu_mem=false rl.device=cuda:0 ${PLD_SET:-}
rc=$?
bash scripts/server.sh stop "$MODEL"
say "stage 1 exited with code $rc"
[[ $rc -eq 0 ]] && date > "$PLD_HOME/logs/STAGE1_DONE$SFX" || echo "exit $rc ($LOG)" > "$PLD_HOME/logs/STAGE1_FAILED$SFX"
exit $rc
