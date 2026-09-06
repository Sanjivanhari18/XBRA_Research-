"""Tests for Phase 3c — Risk metrics computation."""

from __future__ import annotations

import pytest


class TestRiskMetrics:
    @pytest.fixture
    def trades_all_win(self):
        return [
            {"trade_id": f"T{i}", "realized_pnl": 200.0, "exit_date": f"2022-0{(i%9)+1}-{(i%28)+1:02d}",
             "symbol": "AAPL", "quantity": 10, "entry_price": 150.0, "is_winner": True}
            for i in range(10)
        ]

    @pytest.fixture
    def trades_mixed(self):
        return [
            {"trade_id": "W1", "realized_pnl": 500.0,  "exit_date": "2022-01-15", "symbol": "AAPL",
             "quantity": 10, "entry_price": 150.0, "is_winner": True},
            {"trade_id": "L1", "realized_pnl": -1500.0, "exit_date": "2022-03-10", "symbol": "MSFT",
             "quantity": 5,  "entry_price": 300.0, "is_winner": False},
            {"trade_id": "W2", "realized_pnl": 300.0,  "exit_date": "2022-05-20", "symbol": "AAPL",
             "quantity": 8,  "entry_price": 160.0, "is_winner": True},
        ]

    def test_all_winning_trades_zero_drawdown(self, trades_all_win):
        from src.xbra.agents.risk_agent import compute_standard_metrics
        m = compute_standard_metrics(trades_all_win)
        assert m["max_drawdown"] == pytest.approx(0.0, abs=0.01)

    def test_large_loss_creates_drawdown(self, trades_mixed):
        from src.xbra.agents.risk_agent import compute_standard_metrics
        m = compute_standard_metrics(trades_mixed)
        assert m["max_drawdown"] < 0

    def test_concentration_single_symbol(self):
        from src.xbra.agents.risk_agent import compute_standard_metrics
        trades = [
            {"trade_id": f"T{i}", "realized_pnl": 100.0, "exit_date": "2022-01-15",
             "symbol": "AAPL", "quantity": 10, "entry_price": 150.0}
            for i in range(5)
        ]
        m = compute_standard_metrics(trades)
        assert m["concentration"] == pytest.approx(1.0)  # 100% in AAPL

    def test_concentration_diversified(self):
        from src.xbra.agents.risk_agent import compute_standard_metrics
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        trades = [
            {"trade_id": f"T{i}", "realized_pnl": 50.0, "exit_date": "2022-01-15",
             "symbol": symbols[i], "quantity": 10, "entry_price": 100.0}
            for i in range(4)
        ]
        m = compute_standard_metrics(trades)
        assert m["concentration"] < 0.5   # diversified → lower Herfindahl


class TestRegimeClassification:
    def test_returns_series_same_length(self):
        import numpy as np
        import pandas as pd
        from src.xbra.agents.market_agent import classify_regime
        prices = pd.Series(
            100 * np.cumprod(1 + np.random.default_rng(0).normal(0.001, 0.01, 100)),
            index=pd.date_range("2022-01-01", periods=100, freq="B"),
        )
        regimes = classify_regime(prices)
        assert len(regimes) == len(prices)

    def test_bull_regime_in_trending_market(self):
        import numpy as np
        import pandas as pd
        from src.xbra.agents.market_agent import classify_regime
        from src.xbra.schemas import MarketRegime
        prices = pd.Series(
            [100 + i * 1.5 for i in range(100)],  # strongly uptrending
            index=pd.date_range("2022-01-01", periods=100, freq="B"),
        )
        regimes = classify_regime(prices)
        # After window warm-up, should see some bull labels
        valid = regimes.dropna()
        assert any(v == MarketRegime.BULL for v in valid)
