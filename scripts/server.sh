#!/usr/bin/env bash
# Start / stop the base-policy server in its own venv (after `source vast.env`).
#   bash scripts/server.sh start xr1|rldx [MODEL_PATH] [GPU]     # waits until ready
#   bash scripts/server.sh stop  xr1|rldx
# MODEL_PATH defaults to $XR1_MODEL / $RLDX_MODEL (the original FT-RC365 checkpoints); pass an SFT checkpoint
# to evaluate it. Ports come from configs/models/<model>.yaml (xr1 22000, rldx 21000). Stopped by PID, not pattern.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:?start|stop}"; MODEL="${2:?xr1|rldx}"
LOGS="$REPO/runs/logs"; mkdir -p "$LOGS"
PIDF="$LOGS/server_$MODEL.pid"; LOG="$LOGS/server_$MODEL.log"

stop() {
  if [[ -f $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    kill "$(cat "$PIDF")"
    for _ in $(seq 60); do kill -0 "$(cat "$PIDF")" 2>/dev/null || break; sleep 1; done
    kill -9 "$(cat "$PIDF")" 2>/dev/null || true
    echo "[ok] stopped $MODEL server"
  fi
  rm -f "$PIDF"
}

if [[ $CMD == stop ]]; then stop; exit 0; fi

MP="${3:-}"; GPU="${4:-0}"
stop
if [[ $MODEL == xr1 ]]; then
  MP="${MP:-${XR1_MODEL:?source vast.env}}"
  CUDA_VISIBLE_DEVICES=$GPU MIBOT_SERVER_SEED=0 nohup "${PLD_XR1_PY:?}" "$REPO/pld_rc365/base/xr1_server.py" \
      --model-path "$MP" --port 22000 --attn "${ATTN:-flash_attention_2}" > "$LOG" 2>&1 &
else
  MP="${MP:-${RLDX_MODEL:?source vast.env}}"
  ( cd "${RLDX_REPO:?}" && CUDA_VISIBLE_DEVICES=$GPU NO_ALBUMENTATIONS_UPDATE=1 nohup "${PLD_RLDX_PY:?}" \
      "$REPO/pld_rc365/base/rldx_server.py" --model-path "$MP" --port 21000 --device cuda:0 --seed 0 > "$LOG" 2>&1 & \
    echo $! > "$PIDF.tmp" )
fi
[[ -f $PIDF.tmp ]] && mv "$PIDF.tmp" "$PIDF" || echo $! > "$PIDF"
echo "[i] $MODEL server pid $(cat "$PIDF") on GPU $GPU, model $MP (log: $LOG)"
for _ in $(seq 900); do
  grep -q "\[server\] ready" "$LOG" 2>/dev/null && { echo "[ok] $MODEL server ready"; exit 0; }
  kill -0 "$(cat "$PIDF")" 2>/dev/null || { echo "[x] server died:"; tail -30 "$LOG"; exit 1; }
  sleep 2
done
echo "[x] server not ready after 30 min"; tail -30 "$LOG"; exit 1
