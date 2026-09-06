"""Data access layer — all modules load data through here.

Never import from config.settings paths directly in a module; use this registry
so we can swap between synthetic / real data without touching agent code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import PROCESSED_DIR, RAW_DIR, SYNTHETIC_DIR


class DataRegistry:
    """Lazy-loading registry for all XBRA datasets."""

    _trades_df: Optional[pd.DataFrame]       = None
    _ground_truth: Optional[pd.DataFrame]    = None
    _prices: Optional[pd.DataFrame]          = None
    _sentiment: Optional[pd.DataFrame]       = None

    @classmethod
    def trades(cls) -> pd.DataFrame:
        if cls._trades_df is None:
            path = SYNTHETIC_DIR / "trades.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Trades not found at {path}. Run Phase 0 first:\n"
                    "  python scripts/generate_synthetic.py"
                )
            cls._trades_df = pd.read_parquet(path)
            cls._trades_df["entry_date"] = pd.to_datetime(cls._trades_df["entry_date"])
            cls._trades_df["exit_date"]  = pd.to_datetime(cls._trades_df["exit_date"])
        return cls._trades_df

    @classmethod
    def ground_truth(cls) -> pd.DataFrame:
        if cls._ground_truth is None:
            path = SYNTHETIC_DIR / "ground_truth.csv"
            if not path.exists():
                raise FileNotFoundError(f"Ground truth not found at {path}.")
            cls._ground_truth = pd.read_csv(path)
        return cls._ground_truth

    @classmethod
    def prices(cls) -> pd.DataFrame:
        if cls._prices is None:
            path = RAW_DIR / "ohlcv_close.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"OHLCV data not found at {path}. Run Phase 0 first."
                )
            cls._prices = pd.read_parquet(path)
        return cls._prices

    @classmethod
    def investor_ids(cls) -> list[str]:
        return cls.ground_truth()["investor_id"].tolist()

    @classmethod
    def trades_for(cls, investor_id: str) -> pd.DataFrame:
        return cls.trades()[cls.trades()["investor_id"] == investor_id].copy()

    @classmethod
    def bias_for(cls, investor_id: str) -> str:
        row = cls.ground_truth()[cls.ground_truth()["investor_id"] == investor_id]
        if row.empty:
            raise KeyError(f"Unknown investor_id: {investor_id}")
        return str(row.iloc[0]["ground_truth_bias"])

    @classmethod
    def reset(cls) -> None:
        """Clear all caches (useful in tests)."""
        cls._trades_df = cls._ground_truth = cls._prices = cls._sentiment = None
