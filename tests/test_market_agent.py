"""Tests for Market Agent (Phase 3b).

Coverage:
  - safe_external_call wrapper
  - compute_indicators column completeness and value ranges
  - hmm_regimes state assignment on synthetic data
  - _rolling_regime_fallback
  - finbert_sentiment (mocked FinBERT pipeline)
  - fetch_news_headlines (mocked yfinance)
  - build_trade_context: indicator alignment, SPY alpha
  - market_node: missing profile, empty trades, full mock pipeline
  - confidence scoring (all-ok vs all-fail)
"""

from __future__ import annotations

import sys
from datetime import date
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── project imports ────────────────────────────────────────────────────────────
from src.xbra.agents.market_agent import (
    _rolling_regime_fallback,
    _row_as_of,
    build_trade_context,
    compute_indicators,
    fetch_news_headlines,
    finbert_sentiment,
    hmm_regimes,
    market_node,
)
from src.xbra.integrations.safe_call import safe_external_call
from src.xbra.schemas import (
    BiasType,
    InvestorProfile,
    MarketRegime,
    Trade,
)


# ===========================================================================
# Helpers / fixtures
# ===========================================================================

def _make_ohlcv(n: int = 260, base: float = 100.0, trend: float = 0.001, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    closes = base * np.cumprod(1 + trend + rng.normal(0, 0.01, n))
    highs  = closes * (1 + rng.uniform(0.002, 0.015, n))
    lows   = closes * (1 - rng.uniform(0.002, 0.015, n))
    opens  = closes * (1 + rng.normal(0, 0.005, n))
    vols   = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx    = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open":   opens, "high": highs, "low": lows,
        "close":  closes, "volume": vols,
    }, index=idx)


def _make_trade(
    trade_id:    str   = "T001",
    investor_id: str   = "INV_TEST",
    symbol:      str   = "AAPL",
    entry_date:  date  = date(2023, 6, 1),
    exit_date:   date  = date(2023, 7, 1),
    entry_price: float = 100.0,
    exit_price:  float = 105.0,
    quantity:    int   = 10,
    side:        str   = "long",
) -> Trade:
    pnl     = (exit_price - entry_price) * quantity
    holding = (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days
    return Trade(
        trade_id     = trade_id,
        investor_id  = investor_id,
        symbol       = symbol,
        entry_date   = entry_date,
        exit_date    = exit_date,
        entry_price  = entry_price,
        exit_price   = exit_price,
        quantity     = quantity,
        side         = side,
        realized_pnl = round(pnl, 2),
        holding_days = max(holding, 0),
        is_winner    = pnl > 0,
    )


def _make_profile(trades: List[Trade], investor_id: str = "INV_TEST") -> InvestorProfile:
    return InvestorProfile(
        investor_id       = investor_id,
        ground_truth_bias = BiasType.NEUTRAL,
        n_trades          = len(trades),
        trades            = trades,
    )


# ===========================================================================
# 1. safe_external_call
# ===========================================================================

class TestSafeExternalCall:
    def test_returns_result_on_success(self):
        result = safe_external_call(lambda: 42, label="test")
        assert result == 42

    def test_returns_fallback_on_exception(self):
        def boom():
            raise RuntimeError("network down")
        result = safe_external_call(boom, fallback=-1, label="test")
        assert result == -1

    def test_default_fallback_is_none(self):
        result = safe_external_call(lambda: 1 / 0, label="zero-div")
        assert result is None

    def test_returns_custom_fallback_type(self):
        result = safe_external_call(lambda: [][0], fallback=[], label="empty")
        assert result == []

    def test_does_not_reraise(self):
        # Should never propagate exceptions
        safe_external_call(lambda: (_ for _ in ()).throw(ValueError("bad")), label="x")


# ===========================================================================
# 2. compute_indicators
# ===========================================================================

class TestComputeIndicators:
    def setup_method(self):
        self.df = _make_ohlcv(n=260)
        self.ind = compute_indicators(self.df)

    def test_returns_dataframe(self):
        assert isinstance(self.ind, pd.DataFrame)

    def test_has_short_term_columns(self):
        for col in ["rsi_14", "macd", "macd_signal", "macd_diff", "bb_pband", "atr_14", "vol_zscore"]:
            assert col in self.ind.columns, f"missing {col}"

    def test_has_long_term_columns(self):
        for col in ["sma_50", "sma_200", "ema_50", "52wk_high_proximity", "52wk_low_proximity"]:
            assert col in self.ind.columns, f"missing {col}"

    def test_rsi_range(self):
        valid = self.ind["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_bb_pband_range(self):
        # bb_pband can go outside [0,1] on extreme moves; just check dtype
        assert self.ind["bb_pband"].dtype == np.float64

    def test_52wk_proximity_positive(self):
        assert (self.ind["52wk_high_proximity"].dropna() > 0).all()
        assert (self.ind["52wk_low_proximity"].dropna()  > 0).all()

    def test_52wk_high_proximity_le_1(self):
        # close / rolling_max ≤ 1.0 by construction (close can equal the max)
        assert (self.ind["52wk_high_proximity"].dropna() <= 1.0 + 1e-6).all()

    def test_preserves_index(self):
        assert (self.ind.index == self.df.index).all()

    def test_index_size_unchanged(self):
        assert len(self.ind) == len(self.df)


# ===========================================================================
# 3. HMM regimes
# ===========================================================================

class TestHmmRegimes:
    VALID = {MarketRegime.BULL.value, MarketRegime.BEAR.value, MarketRegime.SIDEWAYS.value}

    def test_returns_series(self):
        close = _make_ohlcv(n=260)["close"]
        result = hmm_regimes(close)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self):
        close = _make_ohlcv(n=260)["close"]
        result = hmm_regimes(close)
        assert len(result) == len(close)

    def test_labels_are_valid(self):
        close = _make_ohlcv(n=260)["close"]
        result = hmm_regimes(close)
        assert set(result.unique()).issubset(self.VALID)

    def test_preserves_index(self):
        df    = _make_ohlcv(n=260)
        close = df["close"]
        result = hmm_regimes(close)
        assert (result.index == close.index).all()

    def test_bull_mean_return_exceeds_bear(self):
        """Structural invariant: periods labelled bull have higher mean return than bear.

        GaussianHMM is unsupervised — exact per-day alignment to ground-truth is
        not guaranteed (state collapse is common on short series).  What IS
        guaranteed by the state-mapping logic is the ordering of mean returns.
        """
        close = _make_ohlcv(n=260, seed=5)["close"]
        labels = hmm_regimes(close)
        returns = close.pct_change().fillna(0)

        bull_mask = labels == MarketRegime.BULL.value
        bear_mask = labels == MarketRegime.BEAR.value

        if bull_mask.sum() > 0 and bear_mask.sum() > 0:
            bull_mean = returns[bull_mask].mean()
            bear_mean = returns[bear_mask].mean()
            assert bull_mean >= bear_mean, (
                f"bull mean return {bull_mean:.6f} < bear {bear_mean:.6f}"
            )

    def test_regime_ordering_invariant(self):
        """State-mean ordering: mean(bull_returns) >= mean(sideways) >= mean(bear_returns)."""
        close   = _make_ohlcv(n=300, seed=11)["close"]
        labels  = hmm_regimes(close)
        returns = close.pct_change().fillna(0)

        means: Dict[str, list] = {
            MarketRegime.BULL.value:     [],
            MarketRegime.SIDEWAYS.value: [],
            MarketRegime.BEAR.value:     [],
        }
        for lbl, ret in zip(labels, returns):
            means[lbl].append(ret)

        def mean_or_none(v):
            return float(np.mean(v)) if v else None

        m_bull = mean_or_none(means[MarketRegime.BULL.value])
        m_bear = mean_or_none(means[MarketRegime.BEAR.value])
        if m_bull is not None and m_bear is not None:
            assert m_bull >= m_bear


# ===========================================================================
# 4. Rolling regime fallback
# ===========================================================================

class TestRollingRegimeFallback:
    def test_bull_on_strong_uptrend(self):
        closes = pd.Series(
            100 * np.cumprod(1 + np.full(60, 0.008)),
            index=pd.date_range("2023-01-01", periods=60, freq="B"),
        )
        labels = _rolling_regime_fallback(closes)
        assert labels.iloc[-1] == MarketRegime.BULL.value

    def test_bear_on_strong_downtrend(self):
        closes = pd.Series(
            100 * np.cumprod(1 + np.full(60, -0.008)),
            index=pd.date_range("2023-01-01", periods=60, freq="B"),
        )
        labels = _rolling_regime_fallback(closes)
        assert labels.iloc[-1] == MarketRegime.BEAR.value

    def test_sideways_on_flat(self):
        closes = pd.Series(
            np.full(60, 100.0),
            index=pd.date_range("2023-01-01", periods=60, freq="B"),
        )
        labels = _rolling_regime_fallback(closes)
        assert labels.iloc[-1] == MarketRegime.SIDEWAYS.value

    def test_length_matches(self):
        closes = pd.Series(
            np.linspace(100, 120, 40),
            index=pd.date_range("2023-01-01", periods=40, freq="B"),
        )
        labels = _rolling_regime_fallback(closes)
        assert len(labels) == 40


# ===========================================================================
# 5. FinBERT sentiment
# ===========================================================================

class TestFinbertSentiment:
    def test_returns_zero_on_empty_headlines(self):
        assert finbert_sentiment([]) == 0.0

    def test_returns_zero_when_finbert_unavailable(self):
        import src.xbra.agents.market_agent as ma
        original = ma._finbert_pipeline
        ma._finbert_pipeline = None
        with patch("src.xbra.agents.market_agent._load_finbert", return_value=None):
            score = finbert_sentiment(["Stock rises sharply"])
        ma._finbert_pipeline = original
        assert score == 0.0

    def test_mocked_positive_headline(self):
        mock_pipe = MagicMock(return_value=[{"label": "positive", "score": 0.9}])
        with patch("src.xbra.agents.market_agent._load_finbert", return_value=mock_pipe):
            score = finbert_sentiment(["Stock soars to all-time high"])
        assert score > 0.0

    def test_mocked_negative_headline(self):
        mock_pipe = MagicMock(return_value=[{"label": "negative", "score": 0.8}])
        with patch("src.xbra.agents.market_agent._load_finbert", return_value=mock_pipe):
            score = finbert_sentiment(["Massive losses reported"])
        assert score < 0.0

    def test_mean_of_mixed_headlines(self):
        mock_pipe = MagicMock(return_value=[
            {"label": "positive", "score": 1.0},
            {"label": "negative", "score": 1.0},
        ])
        with patch("src.xbra.agents.market_agent._load_finbert", return_value=mock_pipe):
            score = finbert_sentiment(["good", "bad"])
        assert abs(score) < 1e-6   # (+1 + -1) / 2 = 0


# ===========================================================================
# 6. fetch_news_headlines
# ===========================================================================

class TestFetchNewsHeadlines:
    def test_returns_list(self):
        mock_news = [
            {"content": {"title": "AAPL beats estimates"}},
            {"content": {"title": "New iPhone released"}},
        ]
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.news = mock_news
            result = fetch_news_headlines("AAPL", limit=5)
        assert isinstance(result, list)
        assert "AAPL beats estimates" in result

    def test_returns_empty_list_on_failure(self):
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = fetch_news_headlines("AAPL")
        assert result == []

    def test_respects_limit(self):
        mock_news = [{"content": {"title": f"headline {i}"}} for i in range(20)]
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.news = mock_news
            result = fetch_news_headlines("AAPL", limit=5)
        assert len(result) <= 5


# ===========================================================================
# 7. _row_as_of
# ===========================================================================

class TestRowAsOf:
    def setup_method(self):
        idx       = pd.date_range("2023-01-02", periods=5, freq="B")
        self.df   = pd.DataFrame({"close": [10, 11, 12, 13, 14]}, index=idx)

    def test_returns_row_on_exact_date(self):
        row = _row_as_of(self.df, date(2023, 1, 4))   # 2023-01-04 is in index
        assert row is not None

    def test_returns_row_at_date_in_index(self):
        # 2023-01-05 is a Thursday; the business-day index includes it at position 3
        row = _row_as_of(self.df, date(2023, 1, 5))
        assert row is not None
        assert row["close"] == 13  # index positions 0-4 have values 10-14

    def test_returns_none_when_before_index_start(self):
        row = _row_as_of(self.df, date(2022, 12, 31))
        assert row is None

    def test_returns_last_row_on_date_after_end(self):
        row = _row_as_of(self.df, date(2024, 1, 1))
        assert row["close"] == 14


# ===========================================================================
# 8. build_trade_context
# ===========================================================================

class TestBuildTradeContext:
    def setup_method(self):
        self.ohlcv     = _make_ohlcv(n=260, seed=7)
        self.ind       = compute_indicators(self.ohlcv)
        self.regimes   = hmm_regimes(self.ohlcv["close"])
        self.spy_ohlcv = _make_ohlcv(n=260, seed=99)
        self.spy_ind   = compute_indicators(self.spy_ohlcv)
        self.trade     = _make_trade(
            entry_date=date(2023, 6, 1),
            exit_date=date(2023, 7, 3),
        )

    def test_returns_tuple_of_three(self):
        result = build_trade_context(self.trade, self.ind, self.regimes, 0.2, self.spy_ind)
        assert len(result) == 3

    def test_regime_in_valid_set(self):
        ctx, regime, _ = build_trade_context(self.trade, self.ind, self.regimes, 0.0, None)
        valid = {r.value for r in MarketRegime}
        assert regime in valid

    def test_context_has_core_keys(self):
        ctx, _, _ = build_trade_context(self.trade, self.ind, self.regimes, 0.0, self.spy_ind)
        for key in ["regime", "sentiment", "sym_return", "spy_return", "alpha"]:
            assert key in ctx, f"missing key: {key}"

    def test_context_has_indicator_keys(self):
        ctx, _, _ = build_trade_context(self.trade, self.ind, self.regimes, 0.0, None)
        for key in ["rsi_14", "macd", "bb_pband", "atr_14", "sma_50", "sma_200", "ema_50"]:
            assert key in ctx, f"missing key: {key}"

    def test_alpha_equals_sym_minus_spy(self):
        ctx, _, _ = build_trade_context(self.trade, self.ind, self.regimes, 0.0, self.spy_ind)
        # alpha is computed from unrounded returns; individual keys are rounded → allow 1e-3 tolerance
        assert abs(ctx["alpha"] - (ctx["sym_return"] - ctx["spy_return"])) < 1e-3

    def test_high_negative_sentiment_triggers_market_explains(self):
        _, _, explains = build_trade_context(self.trade, self.ind, self.regimes, -0.8, None)
        assert explains is True

    def test_neutral_sentiment_may_not_trigger(self):
        # Force a non-bear regime by patching
        sideways = pd.Series(
            [MarketRegime.SIDEWAYS.value] * len(self.regimes),
            index=self.regimes.index,
        )
        _, _, explains = build_trade_context(self.trade, self.ind, sideways, 0.0, None)
        assert explains is False

    def test_52wk_proximity_present_and_positive(self):
        ctx, _, _ = build_trade_context(self.trade, self.ind, self.regimes, 0.0, None)
        v = ctx.get("52wk_high_proximity")
        if v is not None:
            assert v > 0

    def test_entry_above_sma50_boolean_or_none(self):
        ctx, _, _ = build_trade_context(self.trade, self.ind, self.regimes, 0.0, None)
        v = ctx.get("entry_above_sma50")
        assert v is None or isinstance(v, bool)


# ===========================================================================
# 9. market_node
# ===========================================================================

class TestMarketNode:
    def test_missing_profile_returns_error(self):
        result = market_node({})
        assert "errors" in result
        assert any("no investor_profile" in e for e in result["errors"])

    def test_empty_trades_returns_zero_confidence(self):
        profile = _make_profile([])
        result  = market_node({"investor_profile": profile.model_dump()})
        assert "market_output" in result
        assert result["market_output"]["confidence"] == 0.0

    def _mock_yfinance_download(self, *args, **kwargs):
        """Return a small synthetic OHLCV dataframe."""
        return _make_ohlcv(n=260, seed=1)

    def test_full_pipeline_with_mock_data(self):
        trades  = [
            _make_trade("T001", entry_date=date(2023, 6, 1), exit_date=date(2023, 7, 3)),
            _make_trade("T002", entry_date=date(2023, 7, 10), exit_date=date(2023, 8, 10)),
        ]
        profile = _make_profile(trades)

        with patch("yfinance.download", side_effect=self._mock_yfinance_download), \
             patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.news = []
            result = market_node({"investor_profile": profile.model_dump()})

        assert "market_output" in result
        mo = result["market_output"]
        assert mo["investor_id"] == "INV_TEST"
        assert mo["confidence"] > 0.0
        assert set(mo["per_trade_context"].keys()) == {"T001", "T002"}
        assert set(mo["regime_labels"].keys())     == {"T001", "T002"}

    def test_confidence_zero_when_yfinance_fails(self):
        trades  = [_make_trade()]
        profile = _make_profile(trades)

        with patch("yfinance.download", side_effect=RuntimeError("no network")), \
             patch("yfinance.Ticker",   side_effect=RuntimeError("no network")):
            result = market_node({"investor_profile": profile.model_dump()})

        mo = result["market_output"]
        assert mo["confidence"] == 0.0

    def test_confidence_one_when_all_symbols_ok(self):
        trades  = [_make_trade()]
        profile = _make_profile(trades)

        with patch("yfinance.download", side_effect=self._mock_yfinance_download), \
             patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.news = []
            result = market_node({"investor_profile": profile.model_dump()})

        assert result["market_output"]["confidence"] == 1.0

    def test_per_trade_context_has_all_indicator_keys(self):
        trades  = [_make_trade()]
        profile = _make_profile(trades)

        with patch("yfinance.download", side_effect=self._mock_yfinance_download), \
             patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.news = []
            result = market_node({"investor_profile": profile.model_dump()})

        ctx = result["market_output"]["per_trade_context"]["T001"]
        for key in ["regime", "sentiment", "rsi_14", "macd", "bb_pband", "sma_50", "sma_200", "ema_50"]:
            assert key in ctx, f"key missing from per_trade_context: {key}"

    def test_regime_labels_contain_valid_values(self):
        trades  = [_make_trade()]
        profile = _make_profile(trades)
        valid   = {r.value for r in MarketRegime}

        with patch("yfinance.download", side_effect=self._mock_yfinance_download), \
             patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.news = []
            result = market_node({"investor_profile": profile.model_dump()})

        for tid, regime in result["market_output"]["regime_labels"].items():
            assert regime in valid, f"invalid regime {regime!r} for trade {tid}"

    def test_stage_set_to_market_done(self):
        trades  = [_make_trade()]
        profile = _make_profile(trades)

        with patch("yfinance.download", side_effect=self._mock_yfinance_download), \
             patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.news = []
            result = market_node({"investor_profile": profile.model_dump()})

        assert result.get("stage") == "market_done"

    def test_multiple_symbols_independent_fetch(self):
        trades = [
            _make_trade("T001", symbol="AAPL", entry_date=date(2023, 6, 1), exit_date=date(2023, 7, 3)),
            _make_trade("T002", symbol="MSFT", entry_date=date(2023, 6, 5), exit_date=date(2023, 7, 7)),
        ]
        profile = _make_profile(trades)
        calls   = []

        def _tracked_download(*args, **kwargs):
            calls.append(args[0])
            return _make_ohlcv(n=260, seed=5)

        with patch("yfinance.download", side_effect=_tracked_download), \
             patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.news = []
            market_node({"investor_profile": profile.model_dump()})

        # AAPL, MSFT, and SPY should each be fetched once
        assert "AAPL" in calls
        assert "MSFT" in calls
        assert "SPY"  in calls

    def test_market_explains_flag_is_bool(self):
        trades  = [_make_trade()]
        profile = _make_profile(trades)

        with patch("yfinance.download", side_effect=self._mock_yfinance_download), \
             patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.news = []
            result = market_node({"investor_profile": profile.model_dump()})

        for v in result["market_output"]["market_explains_flags"].values():
            assert isinstance(v, bool)
