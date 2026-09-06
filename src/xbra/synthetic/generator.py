"""Phase 0 — Synthetic Dataset Generator.

Orchestrates:
1. Download OHLCV data for the trade universe (cached after first run)
2. Generate N investor profiles, each with a known bias
3. Simulate trade sequences using real price data as backdrop
4. Save profiles + trades to parquet + a ground-truth CSV

Run directly:
    python -m src.xbra.synthetic.generator
or via CLI:
    python scripts/generate_synthetic.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)  # type: ignore[assignment]
from tqdm import tqdm

# Make repo root importable when running as script
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import (
    BIAS_DISTRIBUTION,
    MARKET_WINDOW_END,
    MARKET_WINDOW_START,
    N_INVESTORS,
    RANDOM_SEED,
    RAW_DIR,
    SYNTHETIC_DIR,
    TRADE_UNIVERSE,
    TRADES_PER_INVESTOR,
)
from src.xbra.schemas import BiasType, InvestorProfile, Trade
from src.xbra.synthetic.bias_profiles import PROFILES, BiasProfile
from src.xbra.synthetic.trade_simulator import TradeSimulator


class SyntheticDatasetGenerator:
    """Generates a reproducible synthetic dataset of biased investor profiles."""

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.rng = np.random.default_rng(seed)
        self.prices: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Step 1: Market data
    # ------------------------------------------------------------------

    def load_prices(self, force_download: bool = False) -> pd.DataFrame:
        cache_path = RAW_DIR / "ohlcv_close.parquet"

        if not force_download and cache_path.exists():
            logger.info("Loading cached OHLCV from {}", cache_path)
            self.prices = pd.read_parquet(cache_path)
            return self.prices

        logger.info("Downloading OHLCV from yfinance for {} tickers ...", len(TRADE_UNIVERSE))
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError("yfinance not installed — run: pip install yfinance")

        raw = yf.download(
            TRADE_UNIVERSE,
            start=MARKET_WINDOW_START,
            end=MARKET_WINDOW_END,
            auto_adjust=True,
            progress=False,
        )

        # Extract closing prices; handle single vs multi-ticker download shape
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].dropna(how="all")
        else:
            close = raw[["Close"]].rename(columns={"Close": TRADE_UNIVERSE[0]})

        # Forward-fill gaps (weekends already excluded by yfinance)
        close = close.ffill().dropna(how="all")
        close.to_parquet(cache_path)
        logger.success("Saved OHLCV to {}", cache_path)

        self.prices = close
        return close

    # ------------------------------------------------------------------
    # Step 2: Build investor assignment list
    # ------------------------------------------------------------------

    def _build_investor_list(self) -> List[tuple[str, str]]:
        """Returns list of (investor_id, bias_label) pairs."""
        pairs: List[tuple[str, str]] = []
        idx = 0
        for bias_label, count in BIAS_DISTRIBUTION.items():
            for _ in range(count):
                inv_id = f"INV_{bias_label.upper()}_{idx:03d}"
                pairs.append((inv_id, bias_label))
                idx += 1

        # If N_INVESTORS > sum(BIAS_DISTRIBUTION), pad with neutral
        while len(pairs) < N_INVESTORS:
            inv_id = f"INV_NEUTRAL_{idx:03d}"
            pairs.append((inv_id, "neutral"))
            idx += 1

        self.rng.shuffle(pairs)  # type: ignore[arg-type]
        return pairs

    # ------------------------------------------------------------------
    # Step 3: Simulate trades per investor
    # ------------------------------------------------------------------

    def _simulate_investor(
        self,
        investor_id: str,
        bias_label: str,
        profile: BiasProfile,
    ) -> InvestorProfile:
        assert self.prices is not None, "Call load_prices() first"

        n_trades = int(self.rng.integers(*TRADES_PER_INVESTOR))
        start_dt = date.fromisoformat(MARKET_WINDOW_START)
        end_dt   = date.fromisoformat(MARKET_WINDOW_END)

        sim = TradeSimulator(
            investor_id    = investor_id,
            profile        = profile,
            prices         = self.prices,
            start_date     = start_dt,
            end_date       = end_dt,
            n_trades_target= n_trades,
            rng            = self.rng,
        )
        trades = sim.generate()

        return InvestorProfile(
            investor_id       = investor_id,
            ground_truth_bias = BiasType(bias_label),
            n_trades          = len(trades),
            trades            = trades,
            metadata          = {
                "profile_label": profile.label,
                "profile_description": profile.description,
                "ground_truth_features": profile.ground_truth_features,
            },
        )

    # ------------------------------------------------------------------
    # Step 4: Save
    # ------------------------------------------------------------------

    def _save(self, profiles: List[InvestorProfile]) -> None:
        SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

        # Flat trades table
        all_trades: List[dict] = []
        for p in profiles:
            for t in p.trades:
                all_trades.append(t.model_dump())

        trades_df = pd.DataFrame(all_trades)
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"]  = pd.to_datetime(trades_df["exit_date"])
        trades_df.to_parquet(SYNTHETIC_DIR / "trades.parquet", index=False)

        # Ground-truth table (for evaluation)
        gt_rows = [
            {
                "investor_id":       p.investor_id,
                "ground_truth_bias": p.ground_truth_bias.value,
                "n_trades":          p.n_trades,
            }
            for p in profiles
        ]
        gt_df = pd.DataFrame(gt_rows)
        gt_df.to_csv(SYNTHETIC_DIR / "ground_truth.csv", index=False)

        # Full profiles as JSON (for inspection)
        profiles_json = [
            {
                "investor_id":       p.investor_id,
                "ground_truth_bias": p.ground_truth_bias.value,
                "n_trades":          p.n_trades,
                "portfolio_value":   p.portfolio_value,
                "metadata":          p.metadata,
            }
            for p in profiles
        ]
        with open(SYNTHETIC_DIR / "profiles.json", "w") as f:
            json.dump(profiles_json, f, indent=2, default=str)

        logger.success(
            "Saved {} profiles, {} total trades to {}",
            len(profiles), len(all_trades), SYNTHETIC_DIR
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self, force_download: bool = False) -> List[InvestorProfile]:
        self.load_prices(force_download=force_download)

        investor_list = self._build_investor_list()
        profiles: List[InvestorProfile] = []

        logger.info("Generating {} investor profiles ...", len(investor_list))
        for investor_id, bias_label in tqdm(investor_list, desc="Investors"):
            profile = PROFILES[bias_label]
            inv = self._simulate_investor(investor_id, bias_label, profile)
            profiles.append(inv)
            logger.debug(
                "{} ({}) — {} trades generated",
                investor_id, bias_label, inv.n_trades,
            )

        self._save(profiles)
        self._print_summary(profiles)
        return profiles

    def _print_summary(self, profiles: List[InvestorProfile]) -> None:
        from collections import Counter
        bias_counts = Counter(p.ground_truth_bias.value for p in profiles)
        total_trades = sum(p.n_trades for p in profiles)
        avg_trades = total_trades / len(profiles) if profiles else 0

        logger.info("=== Dataset Summary ===")
        logger.info("Total investors : {}", len(profiles))
        logger.info("Total trades    : {}", total_trades)
        logger.info("Avg trades/inv  : {:.1f}", avg_trades)
        logger.info("Bias breakdown  : {}", dict(bias_counts))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate XBRA synthetic dataset")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download OHLCV even if cache exists")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    gen = SyntheticDatasetGenerator(seed=RANDOM_SEED)
    gen.generate(force_download=args.force_download)
