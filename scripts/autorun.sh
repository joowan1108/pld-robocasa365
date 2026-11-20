#!/usr/bin/env bash
# [instance] Started by the Vast On-start script (scripts/onstart.sh): setup -> downloads -> experiment 1,
# unattended. Re-running (e.g. after an instance restart) resumes: every step skips finished results.
# Log: /workspace/autorun.log. Final marker: /workspace/AUTORUN_DONE or /workspace/AUTORUN_FAILED.
# Choose the run with template env vars: MODEL (xr1|rldx), TASK, N, SFT_STEPS, ... (see run_experiment1.sh).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
LOCK=/workspace/.autorun.pid
if [[ -f $LOCK ]] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then echo "[i] autorun already running (pid $(cat $LOCK))"; exit 0; fi
echo $$ > "$LOCK"
rm -f /workspace/AUTORUN_DONE /workspace/AUTORUN_FAILED
fail() { echo "[x] $(date '+%F %T') $*"; echo "$*" > /workspace/AUTORUN_FAILED; rm -f "$LOCK"; exit 1; }
say()  { echo; echo "=== $(date '+%F %T') $*"; }
MODEL="${MODEL:-xr1}"; export MODEL

say "GPU / EGL check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || fail "no GPU visible"
ldconfig -p | grep -q libEGL_nvidia || fail "libEGL_nvidia missing: template needs -e NVIDIA_DRIVER_CAPABILITIES=all"
for u in https://github.com https://huggingface.co; do
  curl -sSfI --max-time 20 "$u" > /dev/null || echo "[!] cannot reach $u (downloads may fail)"
done
echo "[i] cgroup memory limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo n/a)"

say "downloads (background, log /workspace/download.log)"
if [[ ! -f /workspace/.download/assets.done || ! -f /workspace/.download/$MODEL.done ]]; then
  if ! { [[ -f /workspace/.download/pid ]] && kill -0 "$(cat /workspace/.download/pid)" 2>/dev/null; }; then
    nohup bash scripts/download.sh >> /workspace/download.log 2>&1 &
  fi
fi

say "python deps"
bash scripts/vast_setup.sh || fail "vast_setup.sh failed"
source vast.env

say "waiting for assets + $MODEL checkpoint"
while [[ ! -f /workspace/.download/assets.done || ! -f /workspace/.download/$MODEL.done ]]; do
  [[ -f /workspace/.download/failed ]] && fail "download failed: $(cat /workspace/.download/failed) (see /workspace/download.log)"
  if ! { [[ -f /workspace/.download/pid ]] && kill -0 "$(cat /workspace/.download/pid)" 2>/dev/null; }; then
    nohup bash scripts/download.sh >> /workspace/download.log 2>&1 &
  fi
  sleep 30
done

say "experiment 1"
bash scripts/run_experiment1.sh || fail "run_experiment1.sh stopped (see above)"
date > /workspace/AUTORUN_DONE
rm -f "$LOCK"
say "autorun done -- results in $REPO/runs (download, verify md5, then destroy)"
