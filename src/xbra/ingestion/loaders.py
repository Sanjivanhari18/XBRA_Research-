"""Phase 1 — Ingestion & Normalization: data loaders."""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import pandas as pd

from src.xbra.ingestion.normalizer import normalize_dataframe
from src.xbra.schemas import BiasType, InvestorProfile, Trade


def _build_trades(df: pd.DataFrame, investor_id: str) -> list[Trade]:
    trades = []
    for _, row in df.iterrows():
        entry_p = float(row["entry_price"])
        exit_p  = float(row["exit_price"])
        qty     = int(row["quantity"])
        pnl     = (exit_p - entry_p) * qty
        e_date  = row["entry_date"].date() if hasattr(row["entry_date"], "date") else row["entry_date"]
        x_date  = row["exit_date"].date()  if hasattr(row["exit_date"],  "date") else row["exit_date"]
        hold    = (pd.Timestamp(x_date) - pd.Timestamp(e_date)).days

        trades.append(Trade(
            trade_id    = str(row.get("trade_id", f"{investor_id}_{uuid.uuid4().hex[:8]}")),
            investor_id = investor_id,
            symbol      = str(row["symbol"]),
            entry_date  = e_date,
            exit_date   = x_date,
            entry_price = round(entry_p, 4),
            exit_price  = round(exit_p,  4),
            quantity    = qty,
            realized_pnl= round(pnl, 2),
            holding_days= max(hold, 0),
            is_winner   = pnl > 0,
        ))
    return trades


def load_from_parquet(investor_id: str, trades_path: Path) -> InvestorProfile:
    df = pd.read_parquet(trades_path)
    inv_df = df[df["investor_id"] == investor_id].copy()
    if inv_df.empty:
        raise KeyError(f"investor_id {investor_id!r} not found in {trades_path}")
    inv_df["entry_date"] = pd.to_datetime(inv_df["entry_date"])
    inv_df["exit_date"]  = pd.to_datetime(inv_df["exit_date"])
    trades = _build_trades(inv_df, investor_id)
    return InvestorProfile(
        investor_id       = investor_id,
        ground_truth_bias = BiasType.NEUTRAL,  # unknown when loading from parquet directly
        n_trades          = len(trades),
        trades            = trades,
    )


def load_from_csv(investor_id: str, csv_path: Path) -> InvestorProfile:
    raw = pd.read_csv(csv_path)
    norm = normalize_dataframe(raw)
    trades = _build_trades(norm, investor_id)
    return InvestorProfile(
        investor_id       = investor_id,
        ground_truth_bias = BiasType.NEUTRAL,
        n_trades          = len(trades),
        trades            = trades,
    )


def load_from_pdf(investor_id: str, pdf_path: Path) -> InvestorProfile:
    """Extract trade data from a brokerage PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed — run: pip install pdfplumber")

    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in (page.extract_tables() or []):
                for row in table:
                    rows.append(row)

    if not rows:
        raise ValueError(f"No tables found in {pdf_path}")

    # First row = header
    header = [str(c).lower().strip() for c in rows[0]]
    df = pd.DataFrame(rows[1:], columns=header)
    norm = normalize_dataframe(df)
    trades = _build_trades(norm, investor_id)
    return InvestorProfile(
        investor_id       = investor_id,
        ground_truth_bias = BiasType.NEUTRAL,
        n_trades          = len(trades),
        trades            = trades,
    )
