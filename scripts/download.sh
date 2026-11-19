#!/usr/bin/env bash
# [instance] RoboCasa assets (~10 GB) and the VLA checkpoints into /workspace/checkpoints.
# Same marker convention as the SkillPatch image's download.sh (/workspace/.download/<item>.done), so whatever
# that template already fetched is skipped. Order: assets -> the model of this run ($MODEL) -> the other one.
# Safe to re-run; Hugging Face downloads resume.
set -euo pipefail
unset VIRTUAL_ENV
STATE=/workspace/.download
CK=/workspace/checkpoints
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$STATE" "$CK" "$HF_HOME"
rm -f "$STATE/failed"
echo $$ > "$STATE/pid"
item=""
trap 'echo "$item" > "$STATE/failed"; echo "[x] download of $item failed"' ERR

fetch_assets() {
  item=assets
  [[ -f $STATE/assets.done ]] && { echo "[ok] assets (already)"; return; }
  (yes y || true) | /opt/venvs/sim/bin/python -m robocasa.scripts.download_kitchen_assets
  touch "$STATE/assets.done"; echo "[ok] assets"
}
fetch_xr1() {
  item=xr1
  [[ -f $STATE/xr1.done ]] && { echo "[ok] xr1 (already)"; return; }
  /opt/venvs/xr1/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365', local_dir='$CK/Xiaomi-Robotics-1-RoboCasa365')"
  touch "$STATE/xr1.done"; echo "[ok] xr1 checkpoint"
}
fetch_rldx() {
  item=rldx
  [[ -f $STATE/rldx.done ]] && { echo "[ok] rldx (already)"; return; }
  /opt/src/RLDX-1/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('RLWRLD/RLDX-1-FT-RC365', local_dir='$CK/RLDX-1-FT-RC365')"
  touch "$STATE/rldx.done"; echo "[ok] rldx checkpoint"
}

fetch_assets
if [[ ${MODEL:-xr1} == rldx ]]; then fetch_rldx; fetch_xr1; else fetch_xr1; fetch_rldx; fi
rm -f "$STATE/pid"
echo "[ok] downloads done"
