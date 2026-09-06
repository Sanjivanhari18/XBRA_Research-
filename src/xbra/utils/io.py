"""I/O helpers — load/save parquet, JSON, and checkpoint files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_trades(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_ground_truth(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_json(obj: Any, path: Path | str) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def checkpoint(data: Any, name: str, output_dir: Path) -> Path:
    """Save an intermediate result to disk so pipeline can resume after crash."""
    out = output_dir / f"{name}.json"
    save_json(data, out)
    return out


def load_checkpoint(name: str, output_dir: Path) -> Any | None:
    p = output_dir / f"{name}.json"
    return load_json(p) if p.exists() else None
