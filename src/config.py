"""Central configuration loader.

Everything tunable lives in config/config.yaml so that no magic numbers are
buried in the pipeline code.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


class Config(dict):
    """Dict with attribute access and path resolution helpers."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc
        return Config(v) if isinstance(v, dict) else v

    def path(self, *keys: str) -> Path:
        node: Any = self
        for k in keys:
            node = node[k]
        return _resolve(node)


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = _resolve(path or "config/config.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


CFG = load_config()

RAW = _resolve(CFG["data"]["raw_dir"])
INTERIM = _resolve(CFG["data"]["interim_dir"])
PROCESSED = _resolve(CFG["data"]["processed_dir"])
MODELS_DIR = _resolve(CFG["paths"]["models_dir"])
REPORTS = _resolve(CFG["paths"]["reports_dir"])
FIGURES = _resolve(CFG["paths"]["figures_dir"])

for _d in (RAW, INTERIM, PROCESSED, MODELS_DIR, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SEED = int(CFG["project"]["seed"])
TARGET = CFG["project"]["target"]
DELAY_THRESHOLD = int(CFG["project"]["threshold_minutes"])
