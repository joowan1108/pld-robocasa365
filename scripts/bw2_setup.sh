#!/usr/bin/env bash
# [bw2] One-time setup of the PLD environments on the shared B200 server. FailPatch's environments are cloned, never
# modified. Run inside tmux (session name pld_*):  bash scripts/bw2_setup.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C=$HOME/jwjw/miniconda3/bin/conda; E=$HOME/jwjw/miniconda3/envs
[[ -d $E/pld-sim ]] || "$C" create -y -n pld-sim --clone fp-sim || exit 1
[[ -d $E/pld-xr1 ]] || "$C" create -y -n pld-xr1 --clone fp-xr1 || exit 1
# B200 (sm_100) needs the cu128 build; same torch version as fp-sim so the lerobot / torchcodec pins hold
"$E/pld-sim/bin/python" -m pip install --no-input torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 || exit 1
"$E/pld-sim/bin/python" -m pip install --no-input pyzmq timm || exit 1
"$E/pld-xr1/bin/python" -m pip install --no-input pyzmq peft av pandas pyarrow || exit 1
"$E/pld-sim/bin/python" -m pip install --no-input --no-deps -e "$REPO" || exit 1
source "$REPO/scripts/bw2_env.sh"
( cd "$REPO" && "$PLD_SIM_PY" -c "import torch, timm, zmq, pld_rc365.stage1_train; x = torch.zeros(4, device='cuda') + 1; print('[ok] pld-sim', torch.__version__, torch.cuda.get_device_name(0))" ) || exit 1
( cd "$REPO" && "$PLD_XR1_PY" -c "import torch, zmq, flash_attn, transformers; x = torch.zeros(4, device='cuda') + 1; print('[ok] pld-xr1', torch.__version__)" ) || exit 1
echo BW2_SETUP_DONE
