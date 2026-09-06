"""Tests for Phase 0 — Synthetic Dataset Generator.

Run with: pytest tests/test_synthetic.py -v
"""

from __future__ import annotations

import pytest


class TestBiasProfiles:
    def test_all_six_profiles_defined(self):
        from src.xbra.synthetic.bias_profiles import PROFILES
        expected = {"loss_averse", "overconfident", "herding", "disposition", "mixed", "neutral"}
        assert set(PROFILES.keys()) == expected

    def test_loss_averse_loser_hold_longer_than_winner(self):
        from src.xbra.synthetic.bias_profiles import PROFILES
        p = PROFILES["loss_averse"]
        assert p.loser_hold_days[0] > p.winner_hold_days[1], (
            "Loss-averse profile: loser hold min should exceed winner hold max"
        )

    def test_neutral_symmetric_holding(self):
        from src.xbra.synthetic.bias_profiles import PROFILES
        p = PROFILES["neutral"]
        assert p.winner_hold_days == p.loser_hold_days

    def test_overconfident_high_position_size(self):
        from src.xbra.synthetic.bias_profiles import PROFILES
        p = PROFILES["overconfident"]
        assert p.position_size_range[0] >= 0.10, "Overconfident min position should be ≥ 10%"

    def test_herding_high_momentum_follow_prob(self):
        from src.xbra.synthetic.bias_profiles import PROFILES
        p = PROFILES["herding"]
        assert p.momentum_follow_prob >= 0.85


class TestTradeSimulator:
    """Integration-light tests — use tiny synthetic prices to avoid network calls."""

    @pytest.fixture
    def mini_prices(self):
        import numpy as np
        import pandas as pd

        dates = pd.date_range("2022-01-03", periods=250, freq="B")
        rng = np.random.default_rng(0)
        data = {
            "AAPL": 150 * np.cumprod(1 + rng.normal(0.0005, 0.015, 250)),
            "MSFT": 300 * np.cumprod(1 + rng.normal(0.0005, 0.015, 250)),
        }
        return pd.DataFrame(data, index=dates)

    def test_generates_correct_number_of_trades(self, mini_prices):
        import numpy as np
        from datetime import date
        from src.xbra.synthetic.bias_profiles import PROFILES
        from src.xbra.synthetic.trade_simulator import TradeSimulator

        sim = TradeSimulator(
            investor_id     = "TEST_INV_001",
            profile         = PROFILES["neutral"],
            prices          = mini_prices,
            start_date      = date(2022, 1, 3),
            end_date        = date(2022, 12, 30),
            n_trades_target = 20,
            rng             = np.random.default_rng(42),
        )
        trades = sim.generate()
        assert len(trades) > 0
        assert len(trades) <= 20

    def test_trade_dates_within_window(self, mini_prices):
        import numpy as np
        from datetime import date
        from src.xbra.synthetic.bias_profiles import PROFILES
        from src.xbra.synthetic.trade_simulator import TradeSimulator

        start, end = date(2022, 1, 3), date(2022, 12, 30)
        sim = TradeSimulator(
            investor_id     = "TEST_INV_002",
            profile         = PROFILES["loss_averse"],
            prices          = mini_prices,
            start_date      = start,
            end_date        = end,
            n_trades_target = 15,
            rng             = np.random.default_rng(7),
        )
        for t in sim.generate():
            assert t.entry_date >= start
            assert t.exit_date  <= end
            assert t.exit_date  >= t.entry_date

    def test_pnl_direction_matches_price_move(self, mini_prices):
        import numpy as np
        from datetime import date
        from src.xbra.synthetic.bias_profiles import PROFILES
        from src.xbra.synthetic.trade_simulator import TradeSimulator

        sim = TradeSimulator(
            investor_id     = "TEST_INV_003",
            profile         = PROFILES["overconfident"],
            prices          = mini_prices,
            start_date      = date(2022, 1, 3),
            end_date        = date(2022, 12, 30),
            n_trades_target = 30,
            rng             = np.random.default_rng(13),
        )
        for t in sim.generate():
            expected_winner = t.exit_price > t.entry_price
            assert t.is_winner == expected_winner, (
                f"Trade {t.trade_id}: is_winner mismatch"
            )


class TestGenerator:
    def test_bias_distribution_adds_up(self):
        from config.settings import BIAS_DISTRIBUTION, N_INVESTORS
        assert sum(BIAS_DISTRIBUTION.values()) <= N_INVESTORS
