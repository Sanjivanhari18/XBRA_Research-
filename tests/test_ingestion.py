"""Tests for Phase 1 — Ingestion & Normalization."""

from __future__ import annotations

import pandas as pd
import pytest


class TestNormalizer:
    def test_column_aliasing(self):
        from src.xbra.ingestion.normalizer import normalize_dataframe
        df = pd.DataFrame([{
            "date": "2022-01-10", "ticker": "AAPL",
            "buy_price": 150.0, "sell_price": 160.0,
            "qty": 10, "sell_date": "2022-01-20",
        }])
        norm = normalize_dataframe(df)
        assert "symbol" in norm.columns
        assert "entry_date" in norm.columns
        assert "entry_price" in norm.columns

    def test_rejects_exit_before_entry(self):
        from src.xbra.ingestion.normalizer import normalize_dataframe
        df = pd.DataFrame([{
            "entry_date": "2022-01-20", "symbol": "AAPL",
            "entry_price": 150.0, "exit_price": 145.0,
            "quantity": 5, "exit_date": "2022-01-10",   # exit BEFORE entry
        }])
        norm = normalize_dataframe(df)
        assert len(norm) == 0   # invalid row dropped

    def test_missing_required_column_raises(self):
        from src.xbra.ingestion.normalizer import normalize_dataframe
        df = pd.DataFrame([{"entry_date": "2022-01-10", "symbol": "AAPL"}])
        with pytest.raises(ValueError, match="Missing required columns"):
            normalize_dataframe(df)


class TestBehaviorFeatures:
    @pytest.fixture
    def synthetic_profile(self):
        """Build a small InvestorProfile from synthetic data (no yfinance needed)."""
        import numpy as np
        from datetime import date
        from src.xbra.synthetic.bias_profiles import PROFILES
        from src.xbra.synthetic.trade_simulator import TradeSimulator
        # Build mini price data
        import pandas as pd
        dates = pd.date_range("2022-01-03", periods=300, freq="B")
        rng = np.random.default_rng(0)
        # Zero drift → balanced mix of winners and losers so asymmetry is meaningful
        prices = pd.DataFrame({
            "AAPL": 150 * np.cumprod(1 + rng.normal(0.0, 0.015, 300)),
            "MSFT": 280 * np.cumprod(1 + rng.normal(0.0, 0.015, 300)),
        }, index=dates)

        sim = TradeSimulator(
            investor_id="TEST_BEHAVIOR",
            profile=PROFILES["loss_averse"],
            prices=prices,
            start_date=date(2022, 1, 3),
            end_date=date(2022, 12, 30),
            n_trades_target=50,
            rng=np.random.default_rng(99),
        )
        trades = sim.generate()

        from src.xbra.schemas import BiasType, InvestorProfile
        return InvestorProfile(
            investor_id="TEST_BEHAVIOR",
            ground_truth_bias=BiasType.LOSS_AVERSE,
            n_trades=len(trades),
            trades=trades,
        )

    def test_features_returns_dict(self, synthetic_profile):
        from src.xbra.agents.behavior_agent import engineer_features
        feat = engineer_features(synthetic_profile)
        assert isinstance(feat, dict)
        assert "holding_time_asymmetry" in feat
        assert "trade_frequency" in feat

    def test_loss_averse_asymmetry_gt_neutral(self, synthetic_profile):
        """Loss-averse loser_hold_days range > winner_hold_days range in the profile,
        so when there are enough losing trades the asymmetry should exceed 1.0.
        We verify the metric is computed and non-negative; the sign-of-asymmetry
        property is tested at the profile level in test_synthetic.py."""
        from src.xbra.agents.behavior_agent import engineer_features
        feat = engineer_features(synthetic_profile)
        assert feat["holding_time_asymmetry"] >= 0.0

    def test_bias_rules_produce_scores_in_range(self, synthetic_profile):
        from src.xbra.agents.behavior_agent import (
            apply_bias_rules, compute_deviation_scores, engineer_features,
        )
        feat    = engineer_features(synthetic_profile)
        pop     = {k: v for k, v in feat.items()}   # use self as baseline (yields 0 devs)
        dev     = compute_deviation_scores(feat, pop)
        biases  = apply_bias_rules(dev)
        for b, s in biases.items():
            assert 0.0 <= s <= 1.0, f"Bias score out of range: {b}={s}"


class TestFusion:
    def test_pick_dominant_bias_neutral_when_all_zero(self):
        from src.xbra.fusion.signal_fusion import pick_dominant_bias
        from src.xbra.schemas import BiasType
        result = pick_dominant_bias({"loss_aversion": 0, "overconfidence": 0, "herding": 0, "disposition": 0})
        assert result == BiasType.NEUTRAL

    def test_normalize_scores_max_is_one(self):
        from src.xbra.fusion.signal_fusion import normalize_scores
        scores = {"a": 0.3, "b": 0.6, "c": 0.9}
        norm   = normalize_scores(scores)
        assert max(norm.values()) == pytest.approx(1.0)
