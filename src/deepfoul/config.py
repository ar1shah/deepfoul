"""Load configs/*.yaml and pick a machine profile."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file) to the folder holding pyproject.toml."""
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").is_file() and (p / "configs").is_dir():
            return p
    raise FileNotFoundError("Could not find the DeepFoul repo root (pyproject.toml + configs/).")


def load_config(path: Path | None = None) -> dict[str, Any]:
    root = repo_root()
    cfg_path = path or root / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["paths"] = {k: (root / v).resolve() for k, v in cfg["paths"].items()}
    cfg["root"] = root
    return cfg


def pick_profile(cfg: dict[str, Any], vram_gb: float | None) -> str:
    """Return the profile name: an explicit `profile`, else the largest one the VRAM allows."""
    if cfg.get("profile", "auto") != "auto":
        return cfg["profile"]
    if not vram_gb:
        return "cpu"
    fits = [(p["min_vram_gb"], name) for name, p in cfg["profiles"].items()
            if name != "cpu" and vram_gb >= p["min_vram_gb"]]
    return max(fits)[1] if fits else "cpu"
