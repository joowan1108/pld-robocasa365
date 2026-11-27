# Paths and GPU pinning on the shared B200 server (bw2, account guest1). Source this before anything runs there:
#   source scripts/bw2_env.sh
# Everything of ours lives under ~/jwjw/pld. Only GPU 5 is ours (shared with FailPatch): every process sees it as
# its single device (cuda:0) and MuJoCo renders on it. Environments pld-sim / pld-xr1 are clones of FailPatch's
# fp-sim / fp-xr1 (scripts/bw2_setup.sh); robocasa / robosuite / the XR-1 checkpoint are the shared read-only copies.
export JW="${JW:-$HOME/jwjw}"
export PLD_HOME="${PLD_HOME:-$JW/pld}"
export GPU="${GPU:-5}"
# EGL enumerates bw2's GPUs in a different order than CUDA / nvidia-smi: EGL device i is GPU (i + 4) % 8 (checked
# with eglQueryDeviceAttribEXT(EGL_CUDA_DEVICE_NV), xi_runs/egl_map.py, 2026-11-25). MUJOCO_EGL_DEVICE_ID=$GPU made
# MuJoCo render on GPU 1. robosuite uses a single-digit MUJOCO_EGL_DEVICE_ID as the raw EGL index but also asserts
# it is listed in CUDA_VISIBLE_DEVICES (robosuite/utils/binding_utils.py), so the EGL index goes to the sim workers
# via PLD_WORKER_EGL_ID (envs/rc365_env.py _worker: hides CUDA there and sets MUJOCO_EGL_DEVICE_ID).
export CUDA_VISIBLE_DEVICES="$GPU" PLD_WORKER_EGL_ID="${PLD_EGL_ID:-$(( (GPU + 4) % 8 ))}"
unset MUJOCO_EGL_DEVICE_ID
# The shared tmux server carries another user's anaconda base environment, whose __EGL_VENDOR_LIBRARY_DIRS points
# glvnd at conda's mesa only: MuJoCo's headless EGL then finds no NVIDIA device.
unset __EGL_VENDOR_LIBRARY_DIRS __EGL_VENDOR_LIBRARY_FILENAMES
unset VIRTUAL_ENV PLD_EGL_DEVICES CONDA_PREFIX CONDA_DEFAULT_ENV
export PLD_SIM_PY="$JW/miniconda3/envs/pld-sim/bin/python"
export PLD_XR1_PY="$JW/miniconda3/envs/pld-xr1/bin/python"
export XR1_MODEL="$JW/checkpoints/Xiaomi-Robotics-1-RoboCasa365"
export HF_HOME="$PLD_HOME/hf_cache"
export PLD_XR1_PORT="${PLD_XR1_PORT:-10600}"     # FailPatch uses 10586-10588; ours: 10600-10699
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
