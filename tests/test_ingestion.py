"""Tests for Phase 1 — Ingestion & Normalisation.

Covers:
  - Column aliasing and special-character header cleaning
  - Action normalisation (BUY/buy/Buy → 'buy')
  - Deduplication (by fill_id, by value fingerprint)
  - Sanity checks (negative quantity, zero price, unknown action)
  - Position reconstruction via FIFO (basic, partial fills, averaging down)
  - End-to-end load_from_csv on the sample trade log
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# TestFillNormalizer
# ---------------------------------------------------------------------------

class TestFillNormalizer:
    def _make_df(self, **overrides) -> pd.DataFrame:
        """Minimal valid fill row using broker-style column names."""
        row = {
            "Date": "2023-01-10",
            "Ticker": "AAPL",
            "Action": "BUY",
            "Shares": 10,
            "Price ($)": 150.0,
            "Commission": 4.99,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_broker_column_names_map_to_canonical(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df, _ = normalize_fills_df(self._make_df())
        assert "symbol" in df.columns
        assert "fill_date" in df.columns
        assert "action" in df.columns
        assert "quantity" in df.columns
        assert "price" in df.columns
        assert "fees" in df.columns

    def test_special_char_column_name_cleaned(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        # "Price ($)" should clean to "price" → aliased to "price"
        df, _ = normalize_fills_df(self._make_df())
        assert "price" in df.columns

    def test_action_uppercase_normalised_to_lower_buy(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df, _ = normalize_fills_df(self._make_df(Action="BUY"))
        assert df["action"].iloc[0] == "buy"

    def test_action_mixed_case_buy(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df, _ = normalize_fills_df(self._make_df(Action="Buy"))
        assert df["action"].iloc[0] == "buy"

    def test_action_uppercase_sell(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df, _ = normalize_fills_df(self._make_df(Action="SELL", **{"Price ($)": 160.0}))
        assert df["action"].iloc[0] == "sell"

    def test_missing_required_column_raises(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df = pd.DataFrame([{"Date": "2023-01-10", "Ticker": "AAPL"}])
        with pytest.raises(ValueError, match="Missing required columns"):
            normalize_fills_df(df)

    def test_fees_defaults_to_zero_when_absent(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df = pd.DataFrame([{
            "Date": "2023-01-10", "Ticker": "AAPL",
            "Action": "BUY", "Shares": 10, "Price ($)": 150.0,
        }])
        clean, _ = normalize_fills_df(df)
        assert clean["fees"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# TestDeduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def _two_identical_fills(self) -> pd.DataFrame:
        row = {
            "fill_id": "F001",
            "Date": "2023-01-10", "Ticker": "AAPL",
            "Action": "BUY", "Shares": 10, "Price ($)": 150.0, "Commission": 4.99,
        }
        return pd.DataFrame([row, row])

    def test_duplicate_fill_id_removed(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df, stats = normalize_fills_df(self._two_identical_fills())
        assert len(df) == 1
        assert stats.duplicate_count == 1

    def test_duplicate_by_value_removed_even_without_fill_id(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        row = {
            "Date": "2023-01-10", "Ticker": "AAPL",
            "Action": "BUY", "Shares": 10, "Price ($)": 150.0, "Commission": 4.99,
        }
        df, stats = normalize_fills_df(pd.DataFrame([row, row]))
        assert len(df) == 1
        assert stats.duplicate_count == 1

    def test_distinct_fills_not_dropped(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        rows = [
            {"Date": "2023-01-10", "Ticker": "AAPL", "Action": "BUY",
             "Shares": 10, "Price ($)": 150.0, "Commission": 4.99},
            {"Date": "2023-01-11", "Ticker": "AAPL", "Action": "SELL",
             "Shares": 10, "Price ($)": 160.0, "Commission": 4.99},
        ]
        df, stats = normalize_fills_df(pd.DataFrame(rows))
        assert len(df) == 2
        assert stats.duplicate_count == 0


# ---------------------------------------------------------------------------
# TestSanityChecks
# ---------------------------------------------------------------------------

class TestSanityChecks:
    def _make_df(self, **overrides) -> pd.DataFrame:
        row = {
            "Date": "2023-01-10", "Ticker": "AAPL",
            "Action": "BUY", "Shares": 10, "Price ($)": 150.0, "Commission": 4.99,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_negative_quantity_rejected(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df, stats = normalize_fills_df(self._make_df(Shares=-5))
        assert len(df) == 0
        assert stats.rejected_count == 1
        assert any("Non-positive quantity" in r for r in stats.rejected_reasons)

    def test_zero_quantity_rejected(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df, stats = normalize_fills_df(self._make_df(Shares=0))
        assert len(df) == 0
        assert stats.rejected_count == 1

    def test_zero_price_rejected(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df, stats = normalize_fills_df(self._make_df(**{"Price ($)": 0.0}))
        assert len(df) == 0
        assert any("Non-positive price" in r for r in stats.rejected_reasons)

    def test_unknown_action_rejected(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df, stats = normalize_fills_df(self._make_df(Action="HOLD"))
        assert len(df) == 0
        assert any("Unrecognised action" in r for r in stats.rejected_reasons)

    def test_valid_row_not_rejected(self):
        from src.xbra.ingestion.normalizer import normalize_fills_df
        df, stats = normalize_fills_df(self._make_df())
        assert len(df) == 1
        assert stats.rejected_count == 0


# ---------------------------------------------------------------------------
# TestPositionBuilder
# ---------------------------------------------------------------------------

class TestPositionBuilder:
    def _fills(self, rows: list[dict]) -> pd.DataFrame:
        """Build a normalised fills DataFrame from raw row dicts."""
        from src.xbra.ingestion.normalizer import normalize_fills_df
        return normalize_fills_df(pd.DataFrame(rows))[0]

    def test_single_buy_then_sell_creates_one_position(self):
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-01-05", "Ticker": "AAPL", "Action": "BUY",
             "Shares": 10, "Price ($)": 150.0, "Commission": 4.99},
            {"Date": "2023-01-15", "Ticker": "AAPL", "Action": "SELL",
             "Shares": 10, "Price ($)": 165.0, "Commission": 4.99},
        ])
        trades, warnings = build_positions(fills, "INV_TEST")
        assert len(trades) == 1
        assert len(warnings) == 0
        t = trades[0]
        assert t.symbol == "AAPL"
        assert t.is_winner is True
        assert t.holding_days == 10

    def test_partial_fills_create_one_averaged_position(self):
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-01-05", "Ticker": "AAPL", "Action": "BUY",
             "Shares": 50, "Price ($)": 130.0, "Commission": 4.99},
            {"Date": "2023-01-08", "Ticker": "AAPL", "Action": "BUY",
             "Shares": 50, "Price ($)": 132.0, "Commission": 4.99},
            {"Date": "2023-01-15", "Ticker": "AAPL", "Action": "SELL",
             "Shares": 100, "Price ($)": 140.0, "Commission": 4.99},
        ])
        trades, warnings = build_positions(fills, "INV_TEST")
        assert len(trades) == 1
        t = trades[0]
        assert t.quantity == 100
        assert t.entry_price == pytest.approx(131.0)  # avg of 130 and 132
        assert t.is_winner is True

    def test_partial_close_creates_two_position_records(self):
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-02-01", "Ticker": "TSLA", "Action": "BUY",
             "Shares": 20, "Price ($)": 200.0, "Commission": 4.99},
            {"Date": "2023-02-10", "Ticker": "TSLA", "Action": "SELL",
             "Shares": 10, "Price ($)": 195.0, "Commission": 4.99},  # partial, loser
            {"Date": "2023-03-01", "Ticker": "TSLA", "Action": "SELL",
             "Shares": 10, "Price ($)": 190.0, "Commission": 4.99},  # final close, loser
        ])
        trades, warnings = build_positions(fills, "INV_TEST")
        assert len(trades) == 2
        assert all(not t.is_winner for t in trades)

    def test_averaging_down_weighted_avg_entry(self):
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-01-10", "Ticker": "MSFT", "Action": "BUY",
             "Shares": 30, "Price ($)": 240.0, "Commission": 4.99},
            {"Date": "2023-01-20", "Ticker": "MSFT", "Action": "BUY",
             "Shares": 20, "Price ($)": 230.0, "Commission": 4.99},
            {"Date": "2023-03-01", "Ticker": "MSFT", "Action": "SELL",
             "Shares": 50, "Price ($)": 236.0, "Commission": 4.99},
        ])
        trades, _ = build_positions(fills, "INV_TEST")
        # avg entry = (30*240 + 20*230) / 50 = 236.0
        assert trades[0].entry_price == pytest.approx(236.0)

    def test_sell_with_no_open_long_opens_short(self):
        """A SELL with no prior long lot opens a short position (not a warning)."""
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-01-15", "Ticker": "AMZN", "Action": "SELL",
             "Shares": 10, "Price ($)": 100.0, "Commission": 4.99},
        ])
        trades, warnings = build_positions(fills, "INV_TEST")
        # No closed position yet (never covered), but an open short warning appears
        assert len(trades) == 0
        assert any("Open short position" in w for w in warnings)

    def test_unclosed_long_position_produces_warning(self):
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-01-05", "Ticker": "NVDA", "Action": "BUY",
             "Shares": 15, "Price ($)": 200.0, "Commission": 4.99},
        ])
        _, warnings = build_positions(fills, "INV_TEST")
        assert any("Open long position" in w for w in warnings)

    def test_pnl_excludes_commissions(self):
        """P&L must be price-only — no fee deduction."""
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-01-05", "Ticker": "AAPL", "Action": "BUY",
             "Shares": 10, "Price ($)": 100.0, "Commission": 9.99},
            {"Date": "2023-01-15", "Ticker": "AAPL", "Action": "SELL",
             "Shares": 10, "Price ($)": 110.0, "Commission": 9.99},
        ])
        trades, _ = build_positions(fills, "INV_TEST")
        assert trades[0].realized_pnl == pytest.approx(100.0)   # (110-100)*10, no fees


class TestShortSelling:
    def _fills(self, rows: list[dict]) -> pd.DataFrame:
        from src.xbra.ingestion.normalizer import normalize_fills_df
        return normalize_fills_df(pd.DataFrame(rows))[0]

    def test_sell_then_buy_creates_short_position(self):
        """SELL with no prior long lot opens a short; subsequent BUY covers it."""
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-01-05", "Ticker": "TSLA", "Action": "SELL",
             "Shares": 10, "Price ($)": 200.0, "Commission": 0},
            {"Date": "2023-01-20", "Ticker": "TSLA", "Action": "BUY",
             "Shares": 10, "Price ($)": 180.0, "Commission": 0},
        ])
        trades, warnings = build_positions(fills, "INV_TEST")
        assert len(trades) == 1
        assert len(warnings) == 0
        t = trades[0]
        assert t.side == "short"
        assert t.is_winner is True          # shorted at 200, covered at 180 = profit

    def test_short_pnl_profit_when_price_falls(self):
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-02-01", "Ticker": "MSFT", "Action": "SELL",
             "Shares": 20, "Price ($)": 300.0, "Commission": 0},
            {"Date": "2023-02-15", "Ticker": "MSFT", "Action": "BUY",
             "Shares": 20, "Price ($)": 270.0, "Commission": 0},
        ])
        trades, _ = build_positions(fills, "INV_TEST")
        # short pnl = (entry - exit) * qty = (300 - 270) * 20 = 600
        assert trades[0].realized_pnl == pytest.approx(600.0)

    def test_short_pnl_loss_when_price_rises(self):
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-03-01", "Ticker": "NVDA", "Action": "SELL",
             "Shares": 5, "Price ($)": 250.0, "Commission": 0},
            {"Date": "2023-03-20", "Ticker": "NVDA", "Action": "BUY",
             "Shares": 5, "Price ($)": 310.0, "Commission": 0},
        ])
        trades, _ = build_positions(fills, "INV_TEST")
        assert trades[0].realized_pnl == pytest.approx(-300.0)
        assert trades[0].is_winner is False

    def test_partial_short_cover(self):
        """Cover half the short, leaving the other half open."""
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-04-01", "Ticker": "AMZN", "Action": "SELL",
             "Shares": 20, "Price ($)": 130.0, "Commission": 0},
            {"Date": "2023-04-10", "Ticker": "AMZN", "Action": "BUY",
             "Shares": 10, "Price ($)": 120.0, "Commission": 0},
        ])
        trades, warnings = build_positions(fills, "INV_TEST")
        assert len(trades) == 1
        assert trades[0].side == "short"
        assert trades[0].quantity == 10
        assert any("Open short position" in w for w in warnings)

    def test_buy_covers_short_then_opens_long(self):
        """BUY qty > open short qty → covers the short AND opens a new long lot."""
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-05-01", "Ticker": "META", "Action": "SELL",
             "Shares": 10, "Price ($)": 280.0, "Commission": 0},
            {"Date": "2023-05-10", "Ticker": "META", "Action": "BUY",
             "Shares": 20, "Price ($)": 260.0, "Commission": 0},  # covers 10 short + opens 10 long
            {"Date": "2023-05-20", "Ticker": "META", "Action": "SELL",
             "Shares": 10, "Price ($)": 275.0, "Commission": 0},  # closes the new long
        ])
        trades, warnings = build_positions(fills, "INV_TEST")
        assert len(warnings) == 0
        assert len(trades) == 2
        sides = {t.side for t in trades}
        assert "short" in sides
        assert "long" in sides

    def test_unclosed_short_produces_warning(self):
        from src.xbra.ingestion.position_builder import build_positions
        fills = self._fills([
            {"Date": "2023-06-01", "Ticker": "GOOGL", "Action": "SELL",
             "Shares": 8, "Price ($)": 130.0, "Commission": 0},
        ])
        _, warnings = build_positions(fills, "INV_TEST")
        assert any("Open short position" in w for w in warnings)


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.fixture
    def sample_csv(self) -> Path:
        p = Path(__file__).resolve().parents[1] / "docs" / "sample_trade_log.csv"
        if not p.exists():
            pytest.skip(f"Sample CSV not found at {p}")
        return p

    def test_load_from_csv_returns_profile_and_report(self, sample_csv):
        from src.xbra.ingestion.loaders import load_from_csv
        profile, report = load_from_csv("INV_SAMPLE", sample_csv)
        assert profile.investor_id == "INV_SAMPLE"
        assert profile.n_trades > 0
        assert 0.0 <= profile.data_quality_score <= 1.0

    def test_sample_csv_duplicate_detected(self, sample_csv):
        from src.xbra.ingestion.loaders import load_from_csv
        _, report = load_from_csv("INV_SAMPLE", sample_csv)
        assert report.duplicate_fill_count >= 1

    def test_sample_csv_invalid_row_rejected(self, sample_csv):
        from src.xbra.ingestion.loaders import load_from_csv
        _, report = load_from_csv("INV_SAMPLE", sample_csv)
        assert report.rejected_fill_count >= 1

    def test_sample_csv_pipeline_completes_below_min_trades(self, sample_csv):
        """Pipeline must not crash when n_trades < MIN_TRADES; quality score reflects it."""
        from config.settings import MIN_TRADES
        from src.xbra.ingestion.loaders import load_from_csv
        profile, report = load_from_csv("INV_SAMPLE", sample_csv)
        # Sample CSV is intentionally small — verify the quality score
        # penalises the low trade count (trade_factor < 1.0)
        if profile.n_trades < MIN_TRADES:
            assert report.data_quality_score < 1.0   # trade count penalty applied
        assert profile.n_trades > 0                  # pipeline still produced positions

    def test_sample_csv_loss_averse_pattern(self, sample_csv):
        """Winners should be held shorter on average than losers (loss aversion signal)."""
        from src.xbra.ingestion.loaders import load_from_csv
        profile, _ = load_from_csv("INV_SAMPLE", sample_csv)
        winners = [t for t in profile.trades if t.is_winner]
        losers  = [t for t in profile.trades if not t.is_winner]
        if winners and losers:
            avg_winner_hold = sum(t.holding_days for t in winners) / len(winners)
            avg_loser_hold  = sum(t.holding_days for t in losers)  / len(losers)
            assert avg_winner_hold < avg_loser_hold, (
                f"Expected winners held shorter ({avg_winner_hold:.1f}d) "
                f"than losers ({avg_loser_hold:.1f}d)"
            )


# ---------------------------------------------------------------------------
# TestFusion (unchanged — fusion module tests)
# ---------------------------------------------------------------------------

class TestFusion:
    def test_pick_dominant_bias_neutral_when_all_zero(self):
        from src.xbra.fusion.signal_fusion import pick_dominant_bias
        from src.xbra.schemas import BiasType
        result = pick_dominant_bias(
            {"loss_aversion": 0, "overconfidence": 0, "herding": 0, "disposition": 0}
        )
        assert result == BiasType.NEUTRAL

    def test_normalize_scores_max_is_one(self):
        from src.xbra.fusion.signal_fusion import normalize_scores
        scores = {"a": 0.3, "b": 0.6, "c": 0.9}
        norm   = normalize_scores(scores)
        assert max(norm.values()) == pytest.approx(1.0)
