"""Phase 1 — Normalizer: column harmonisation, type coercion, validation."""

from __future__ import annotations

from typing import List

import pandas as pd

COLUMN_ALIASES = {
    "date": "entry_date", "trade_date": "entry_date", "buy_date": "entry_date",
    "open_date": "entry_date", "transaction_date": "entry_date",
    "ticker": "symbol", "stock": "symbol", "instrument": "symbol", "scrip": "symbol",
    "shares": "quantity", "qty": "quantity", "volume": "quantity", "units": "quantity",
    "buy_price": "entry_price", "open_price": "entry_price", "purchase_price": "entry_price",
    "sell_date": "exit_date", "close_date": "exit_date", "sell_trade_date": "exit_date",
    "sell_price": "exit_price", "close_price": "exit_price", "sale_price": "exit_price",
    "pnl": "realized_pnl", "profit_loss": "realized_pnl", "net_pnl": "realized_pnl",
    "pl": "realized_pnl",
}

REQUIRED_COLS = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "quantity"]


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalisation: {missing}")

    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["exit_price"]  = pd.to_numeric(df["exit_price"],  errors="coerce")
    df["quantity"]    = pd.to_numeric(df["quantity"],    errors="coerce").astype(int)

    df = df.dropna(subset=REQUIRED_COLS)
    df = df[df["exit_date"] >= df["entry_date"]]
    return df.reset_index(drop=True)


def validate_investor_profile(profile) -> List[str]:
    errors: List[str] = []
    if not profile.trades:
        errors.append("No trades found")
        return errors
    for t in profile.trades:
        if t.exit_date < t.entry_date:
            errors.append(f"Trade {t.trade_id}: exit_date before entry_date")
        if t.entry_price <= 0:
            errors.append(f"Trade {t.trade_id}: entry_price ≤ 0")
        if t.quantity <= 0:
            errors.append(f"Trade {t.trade_id}: quantity ≤ 0")
    return errors
