"""Phase 1 — Normaliser: fill-level CSV normalisation, deduplication, sanity checks.

Pipeline (in order):
  1. _normalize_columns  — rename broker-specific headers to canonical names
  2. _coerce_types       — parse dates, cast numerics
  3. _deduplicate        — drop exact-duplicate fills (by fill_id, then by value)
  4. _sanity_checks      — drop impossible rows and collect rejection reasons
  5. sort chronologically

Output: a clean fill-level DataFrame ready for position_builder.build_positions().
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Column alias table  (raw broker column name → canonical fill-level name)
# ---------------------------------------------------------------------------

COLUMN_ALIASES: Dict[str, str] = {
    # date / timestamp
    "date": "fill_date", "trade_date": "fill_date", "transaction_date": "fill_date",
    "order_date": "fill_date", "execution_date": "fill_date",
    # symbol
    "ticker": "symbol", "stock": "symbol", "instrument": "symbol",
    "scrip": "symbol", "security": "symbol",
    # action / side
    "action": "action", "side": "action", "buy_sell": "action",
    "transaction_type": "action", "order_type": "action", "type": "action",
    # quantity
    "shares": "quantity", "qty": "quantity", "volume": "quantity",
    "units": "quantity", "num_shares": "quantity",
    # price (also handles "price" after special-char stripping of e.g. "Price ($)")
    "price": "price", "trade_price": "price", "execution_price": "price",
    "avg_price": "price", "filled_price": "price",
    # fees / commission
    "fees": "fees", "commission": "fees", "fee": "fees",
    "brokerage": "fees", "charges": "fees", "transaction_fee": "fees",
    # fill / order identifier (optional)
    "fill_id": "fill_id", "trade_id": "fill_id", "order_id": "fill_id",
    "transaction_id": "fill_id", "ref": "fill_id",
}

# Canonical action strings accepted from brokers
ACTION_MAP: Dict[str, str] = {
    "buy": "buy", "b": "buy", "long": "buy", "purchase": "buy", "bo": "buy",
    "sell": "sell", "s": "sell", "short": "sell", "sale": "sell", "sc": "sell",
}

REQUIRED_FILL_COLS = ["fill_date", "symbol", "action", "quantity", "price"]


# ---------------------------------------------------------------------------
# Internal stats accumulator (not a public schema — see NormalizationReport)
# ---------------------------------------------------------------------------

@dataclass
class _NormStats:
    raw_count: int = 0
    duplicate_count: int = 0
    rejected_count: int = 0
    rejected_reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step helpers (pure — no I/O, no side effects)
# ---------------------------------------------------------------------------

def _clean_col(name: str) -> str:
    """Lowercase, strip, replace every run of non-alphanumeric chars with '_'.

    Examples:
        "Price ($)" → "price"
        "Num Shares" → "num_shares"
        "  Date  "  → "date"
    """
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_col(c) for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fill_date"] = pd.to_datetime(df["fill_date"])
    df["quantity"]  = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["price"]     = pd.to_numeric(df["price"],    errors="coerce").fillna(0.0)
    if "fees" in df.columns:
        df["fees"]  = pd.to_numeric(df["fees"],     errors="coerce").fillna(0.0)
    else:
        df["fees"]  = 0.0
    df["action"]    = (
        df["action"].astype(str).str.lower().str.strip()
        .map(lambda a: ACTION_MAP.get(a, a))
    )
    if "fill_id" not in df.columns:
        df.insert(0, "fill_id", [f"F{i:04d}" for i in range(len(df))])
    return df


def _deduplicate(df: pd.DataFrame, stats: _NormStats) -> pd.DataFrame:
    before = len(df)
    # Primary: identical fill_id
    df = df.drop_duplicates(subset=["fill_id"], keep="first")
    # Secondary: identical fill fingerprint (catches duplicates with different IDs)
    fp_cols = [c for c in ["fill_date", "symbol", "action", "quantity", "price"] if c in df.columns]
    df = df.drop_duplicates(subset=fp_cols, keep="first")
    stats.duplicate_count = before - len(df)
    return df.reset_index(drop=True)


def _sanity_checks(df: pd.DataFrame, stats: _NormStats) -> pd.DataFrame:
    valid = pd.Series(True, index=df.index)

    if "quantity" in df.columns:
        bad = df["quantity"] <= 0
        for _, row in df[bad].iterrows():
            stats.rejected_reasons.append(
                f"Non-positive quantity ({row['quantity']}) "
                f"for {row.get('symbol')} on {row.get('fill_date').date()}"
            )
        valid &= ~bad

    if "price" in df.columns:
        bad = df["price"] <= 0
        for _, row in df[bad].iterrows():
            stats.rejected_reasons.append(
                f"Non-positive price ({row['price']}) "
                f"for {row.get('symbol')} on {row.get('fill_date').date()}"
            )
        valid &= ~bad

    if "action" in df.columns:
        bad = ~df["action"].isin(("buy", "sell"))
        for _, row in df[bad].iterrows():
            stats.rejected_reasons.append(
                f"Unrecognised action ({row['action']!r}) "
                f"for {row.get('symbol')} on {row.get('fill_date').date()}"
            )
        valid &= ~bad

    stats.rejected_count = int((~valid).sum())
    return df[valid].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_fills_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, _NormStats]:
    """Full normalisation pipeline for a raw broker CSV export.

    Inputs:
        df: raw DataFrame as returned by pd.read_csv()

    Outputs:
        (clean_df, stats)
        clean_df: canonical fill-level DataFrame (chronologically sorted)
        stats:    _NormStats capturing counts for NormalizationReport construction

    Raises:
        ValueError if required columns are missing after mapping.
    """
    stats = _NormStats(raw_count=len(df))

    df = _normalize_columns(df)

    missing = [c for c in REQUIRED_FILL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalisation: {missing}")

    df = _coerce_types(df)
    df = _deduplicate(df, stats)
    df = _sanity_checks(df, stats)
    df = df.sort_values("fill_date").reset_index(drop=True)

    return df, stats


def compute_data_quality_score(
    n_raw: int,
    n_accepted: int,
    n_closed: int,
    n_open: int,
    min_trades: int,
) -> float:
    """Composite data quality score in [0, 1].

    Weights:
      50% — fill acceptance rate (how much survived cleaning)
      30% — position closure rate (how complete the position picture is)
      20% — minimum trade count factor (enough data for analysis)
    """
    fill_rate    = n_accepted / n_raw if n_raw > 0 else 0.0
    closure_rate = n_closed / (n_closed + n_open) if (n_closed + n_open) > 0 else 1.0
    trade_factor = min(1.0, n_closed / max(min_trades, 1))
    return round(fill_rate * 0.50 + closure_rate * 0.30 + trade_factor * 0.20, 4)


def validate_investor_profile(profile) -> List[str]:
    """Post-ingestion validation of an InvestorProfile (position-level checks)."""
    errors: List[str] = []
    if not profile.trades:
        errors.append("No closed positions found")
        return errors
    for t in profile.trades:
        if t.exit_date < t.entry_date:
            errors.append(f"Trade {t.trade_id}: exit_date before entry_date")
        if t.entry_price <= 0:
            errors.append(f"Trade {t.trade_id}: entry_price ≤ 0")
        if t.quantity <= 0:
            errors.append(f"Trade {t.trade_id}: quantity ≤ 0")
    return errors
