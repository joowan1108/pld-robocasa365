"""CPU thread limits.

Containers (e.g. Vast.ai) often expose the host's core count (nproc = 255) while the cgroup CPU quota is much
smaller (~30 CPUs). torch / OpenMP / MKL then start one thread per *host* core in every process; with a VLA
server + 10 env workers that is thousands of spinning threads against a 30-CPU quota, and the whole container
gets throttled (observed: 43% of scheduler periods throttled, GPU 4% busy, eval crawling).

Policy: env workers use 1 thread (MuJoCo stepping is single-threaded), the main PLD process and the policy
servers use PLD_TORCH_THREADS (default 4).
"""

from __future__ import annotations

import os

_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
         "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def default_threads() -> int:
    return int(os.environ.get("PLD_TORCH_THREADS", "4"))


def set_thread_env(n: int, override: bool = False) -> None:
    """Set the env vars read by OpenMP/MKL/BLAS at library load (call before importing torch/numpy users)."""
    for v in _VARS:
        if override or v not in os.environ:
            os.environ[v] = str(n)


def limit_threads(n: int | None = None) -> int:
    """Apply limits to already-imported libraries too."""
    n = int(n or default_threads())
    set_thread_env(n)
    try:
        import torch

        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(max(1, min(n, 4)))
        except RuntimeError:
            pass  # can only be set once, before any parallel work
    except ImportError:
        pass
    try:
        import cv2

        cv2.setNumThreads(n)
    except Exception:  # noqa: BLE001
        pass
    return n
