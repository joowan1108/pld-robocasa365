"""Wall-clock bookkeeping for the Stage 1/2/3 measurements.

Every stage writes `timing.json` next to its outputs:
  {"hardware": {...}, "phases": {name: {"start", "end", "seconds"}}, "marks": {name: seconds_since_t0}, ...}
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def hardware_fingerprint() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        info["gpus"] = [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:  # noqa: BLE001
        info["gpus"] = []
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return info


class Timer:
    def __init__(self, out_path: str | Path, extra: dict | None = None):
        self.out_path = Path(out_path)
        self.t0 = time.time()
        self.data: dict[str, Any] = {
            "hardware": hardware_fingerprint(),
            "t0_unix": self.t0,
            "phases": {},
            "marks": {},
            "extra": dict(extra or {}),
        }
        if self.out_path.exists():  # resume: keep previous phases
            try:
                prev = json.loads(self.out_path.read_text())
                self.data["phases"].update(prev.get("phases", {}))
                self.data["marks"].update(prev.get("marks", {}))
                self.data["extra"] = {**prev.get("extra", {}), **self.data["extra"]}
            except Exception:  # noqa: BLE001
                pass

    @contextmanager
    def phase(self, name: str):
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            self.data["phases"][name] = {"start": start, "end": end, "seconds": end - start}
            self.save()

    def add_phase_seconds(self, name: str, seconds: float):
        self.data["phases"][name] = {"start": None, "end": None, "seconds": float(seconds)}
        self.save()

    def mark(self, name: str, since: float | None = None):
        ref = self.t0 if since is None else since
        self.data["marks"][name] = time.time() - ref
        self.save()

    def set(self, key: str, value: Any):
        self.data["extra"][key] = value
        self.save()

    def save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=float))
        tmp.replace(self.out_path)


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)

    def log(self, **kw):
        kw.setdefault("time", time.time())
        self._f.write(json.dumps(kw, default=float) + "\n")

    def close(self):
        self._f.close()
