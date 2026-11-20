"""PLD (Probe, Learn, Distill) on RoboCasa365.

Stage 1: residual RL specialist on top of a frozen VLA (`stage1_train`).
Stage 2: hybrid base-probing + residual takeover data collection (`stage2_collect`).
Stage 3: LoRA SFT of the VLA on the collected data (`sft/`), evaluated with `eval`.
"""

from .utils.threads import default_threads as _dt, set_thread_env as _ste

# Before torch/OpenMP load: cap per-process CPU threads (containers report host cores, see utils/threads.py).
_ste(_dt())
