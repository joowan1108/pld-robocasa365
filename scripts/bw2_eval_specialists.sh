#!/usr/bin/env bash
# [bw2] Evaluate several stage-1 specialist checkpoints like stage 1's final eval (residual from step 0, residual
# mean, no probing) with one base-policy server. Run in its own repo clone (own server PID files) inside tmux:
#   CKPTS="a.pt b.pt" PLD_SET="<stage-1 rl/residual overrides>" TAG=c2ckpt PLD_XR1_PORT=10640 bash scripts/bw2_eval_specialists.sh
# Env: CKPTS (paths), SCENES ("fixed"), EPISODES (50), NUM_ENVS (16). Results: runs/xr1/CoffeeSetupMug/eval/spec_<name>.json
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source scripts/bw2_env.sh
MODEL=xr1; TASK=CoffeeSetupMug; CKPTS="${CKPTS:?checkpoint paths}"; SCENES="${SCENES:-fixed}"
EPISODES="${EPISODES:-50}"; NUM_ENVS="${NUM_ENVS:-16}"; TAG="${TAG:-ckpteval}"
mkdir -p "$PLD_HOME/logs"; LOG="$PLD_HOME/logs/${TAG}_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') eval specialists [$TAG]: $CKPTS | scenes $SCENES | $EPISODES eps | commit $(git log --oneline -1)"
ss -ltn | grep -q ":$PLD_XR1_PORT " && { echo "[x] port $PLD_XR1_PORT in use"; exit 1; }
bash scripts/server.sh start "$MODEL" "" "$GPU" 0 || exit 1
rc=0
for c in $CKPTS; do
  name=$(basename "$c" .pt); name=${name#specialists_}
  [[ -f runs/$MODEL/$TASK/eval/spec_${name}.json ]] && { echo "skip $name (done)"; continue; }
  echo "=== $(date '+%F %T') $name"
  "$PLD_SIM_PY" -m pld_rc365.eval --model "$MODEL" --task "$TASK" --tag "spec_${name}" --scenes $SCENES --episodes "$EPISODES" \
      --specialists "$c" --set base.replicas=1 "base.port=$PLD_XR1_PORT" "env.num_envs=$NUM_ENVS" env.cap_envs_by_gpu_mem=false \
      rl.device=cuda:0 ${PLD_SET:-} || { rc=1; echo "[x] eval $name failed"; }
done
bash scripts/server.sh stop "$MODEL"
echo "=== $(date '+%F %T') done (rc $rc)"; date > "$PLD_HOME/logs/${TAG}_DONE"
exit $rc
