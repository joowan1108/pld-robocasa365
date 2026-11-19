"""YAML config loading: configs/common.yaml <- configs/models/<model>.yaml <- configs/tasks/<task>.yaml <- CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(s: str) -> Any:
    return yaml.safe_load(s)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `a.b.c=value` overrides (value parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(val)
    return cfg


def load_config(model: str, task: str, overrides: list[str] | None = None) -> dict:
    with open(CONFIG_DIR / "common.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(CONFIG_DIR / "models" / f"{model}.yaml") as f:
        cfg = deep_update(cfg, yaml.safe_load(f))
    with open(CONFIG_DIR / "tasks" / f"{task}.yaml") as f:
        cfg = deep_update(cfg, yaml.safe_load(f))
    cfg = apply_overrides(cfg, overrides)
    cfg["model_name"] = model
    cfg["task_name"] = task
    return cfg


def run_dir(cfg: dict, stage: str) -> Path:
    root = Path(cfg["paths"]["runs_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    d = root / cfg["model_name"] / cfg["task_name"] / stage
    d.mkdir(parents=True, exist_ok=True)
    return d
