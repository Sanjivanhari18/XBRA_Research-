"""Phase 1 — Ingestion: data loaders for the XBRA pipeline.

Entry points:
  load_from_csv     — primary path for user-uploaded broker CSV exports
  load_from_parquet — used by the synthetic data pipeline (already position-level)
  load_from_pdf     — stub; not yet implemented (CSV-only scope for Phase 1)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config.settings import MIN_DATA_QUALITY_THRESHOLD, MIN_TRADES
from src.xbra.ingestion.normalizer import (
    compute_data_quality_score,
    normalize_fills_df,
    validate_investor_profile,
)
from src.xbra.ingestion.position_builder import build_positions
from src.xbra.schemas import (
    BiasType,
    InvestorProfile,
    NormalizationReport,
    Trade,
)
from src.xbra.utils.logging import logger


def load_from_csv(
    investor_id: str,
    csv_path: Path,
) -> Tuple[InvestorProfile, NormalizationReport]:
    """Load a broker CSV export, normalise fills, reconstruct closed positions.

    Inputs:
        investor_id: identifier assigned to this investor session
        csv_path:    path to the raw broker CSV file

    Outputs:
        (InvestorProfile, NormalizationReport)
        InvestorProfile.trades contains one Trade per closed position.
        NormalizationReport is the full audit trail of the cleaning process.

    Raises:
        FileNotFoundError if csv_path does not exist.
        ValueError if required columns are missing after column mapping.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    logger.info(
        f"[Ingestion] {investor_id}: read {len(raw_df)} raw rows from '{csv_path.name}'"
    )

    # Step 1 — Normalise fills (column mapping, dedup, sanity checks)
    clean_df, stats = normalize_fills_df(raw_df)
    logger.info(
        f"[Ingestion] {investor_id}: {stats.raw_count} raw → {len(clean_df)} accepted "
        f"({stats.duplicate_count} duplicates dropped, "
        f"{stats.rejected_count} invalid rows dropped)"
    )
    if stats.rejected_reasons:
        for reason in stats.rejected_reasons:
            logger.warning(f"[Ingestion] {investor_id}: rejected — {reason}")

    # Step 2 — Reconstruct positions via FIFO lot matching
    trades, open_warnings = build_positions(clean_df, investor_id)
    for w in open_warnings:
        logger.warning(f"[PositionBuilder] {investor_id}: {w}")

    n_open = sum(1 for w in open_warnings if "Open position" in w)

    # Step 3 — Compute data quality score
    quality = compute_data_quality_score(
        n_raw=stats.raw_count,
        n_accepted=len(clean_df),
        n_closed=len(trades),
        n_open=n_open,
        min_trades=MIN_TRADES,
    )

    report = NormalizationReport(
        investor_id=investor_id,
        raw_fill_count=stats.raw_count,
        accepted_fill_count=len(clean_df),
        duplicate_fill_count=stats.duplicate_count,
        rejected_fill_count=stats.rejected_count,
        rejected_reasons=stats.rejected_reasons,
        open_position_count=n_open,
        closed_position_count=len(trades),
        data_quality_score=quality,
    )

    logger.info(
        f"[Ingestion] {investor_id}: {len(trades)} closed positions reconstructed, "
        f"data_quality_score={quality:.3f}"
    )
    if quality < MIN_DATA_QUALITY_THRESHOLD:
        logger.warning(
            f"[Ingestion] {investor_id}: data quality score {quality:.3f} is below "
            f"threshold 0.60 — downstream confidence scores will be discounted"
        )

    profile = InvestorProfile(
        investor_id=investor_id,
        ground_truth_bias=BiasType.NEUTRAL,
        n_trades=len(trades),
        trades=trades,
        data_quality_score=quality,
    )

    # Validate the resulting profile (catches any post-reconstruction oddities)
    errs = validate_investor_profile(profile)
    if errs:
        for e in errs:
            logger.error(f"[Ingestion] {investor_id}: validation error — {e}")

    # Phase 4 hook: log report to SQLite here
    # _log_normalization_report(report)  # TODO: Phase 4

    return profile, report


def load_from_parquet(
    investor_id: str,
    trades_path: Path,
) -> Tuple[InvestorProfile, None]:
    """Load investor data from a Parquet file (used by the synthetic data pipeline).

    The synthetic generator produces position-level rows directly, so no
    fill-level normalisation or position reconstruction is needed.
    """
    df = pd.read_parquet(trades_path)
    inv_df = df[df["investor_id"] == investor_id].copy()
    if inv_df.empty:
        raise KeyError(f"investor_id {investor_id!r} not found in {trades_path}")

    inv_df["entry_date"] = pd.to_datetime(inv_df["entry_date"])
    inv_df["exit_date"]  = pd.to_datetime(inv_df["exit_date"])
    trades = _trades_from_position_rows(inv_df, investor_id)

    profile = InvestorProfile(
        investor_id=investor_id,
        ground_truth_bias=BiasType.NEUTRAL,
        n_trades=len(trades),
        trades=trades,
        data_quality_score=1.0,  # synthetic data is always clean
    )
    return profile, None


def load_from_pdf(investor_id: str, pdf_path: Path) -> None:
    """PDF ingestion — not implemented in Phase 1 (CSV-only scope)."""
    raise NotImplementedError(
        "PDF ingestion is not yet implemented. "
        "Please export your trades as a CSV file and use load_from_csv()."
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _trades_from_position_rows(df: pd.DataFrame, investor_id: str) -> list[Trade]:
    """Build Trade objects from position-level rows (parquet / synthetic data format)."""
    trades = []
    for _, row in df.iterrows():
        entry_p = float(row["entry_price"])
        exit_p  = float(row["exit_price"])
        qty     = int(row["quantity"])
        pnl     = float(row.get("realized_pnl", (exit_p - entry_p) * qty))
        e_date  = row["entry_date"].date() if hasattr(row["entry_date"], "date") else row["entry_date"]
        x_date  = row["exit_date"].date()  if hasattr(row["exit_date"],  "date") else row["exit_date"]
        hold    = (pd.Timestamp(x_date) - pd.Timestamp(e_date)).days

        trades.append(Trade(
            trade_id     = str(row.get("trade_id", f"{investor_id}_{uuid.uuid4().hex[:8]}")),
            investor_id  = investor_id,
            symbol       = str(row["symbol"]),
            entry_date   = e_date,
            exit_date    = x_date,
            entry_price  = round(entry_p, 4),
            exit_price   = round(exit_p, 4),
            quantity     = qty,
            realized_pnl = round(pnl, 2),
            holding_days = max(hold, 0),
            is_winner    = pnl > 0,
        ))
    return trades
