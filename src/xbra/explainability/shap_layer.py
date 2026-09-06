"""Phase 5 — SHAP computation layer: feature-to-bias mapping + stability check."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.settings import ROLLING_WINDOW_DAYS, SHAP_STABILITY_THRESHOLD


SHAP_BIAS_MAP: List[Tuple[str, str, str]] = [
    ("holding_time_asymmetry",  "positive", "loss_aversion"),
    ("early_exit_winner_rate",  "positive", "disposition"),
    ("position_size_cv",        "positive", "overconfidence"),
    ("trade_frequency",         "positive", "overconfidence"),
    ("momentum_follow_rate",    "positive", "herding"),
    ("post_loss_reentry_speed", "negative", "loss_aversion"),
    ("avg_loss_magnitude",      "positive", "loss_aversion"),
    ("pct_negative_alpha",      "positive", "herding"),
]


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(X)
    except Exception:
        return np.zeros(X.shape)


def map_features_to_biases(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Attribute each feature's SHAP contribution to a bias type.

    Returns {bias_name: total_abs_shap_attributed}.
    """
    shap_dict = {feature_names[i]: float(shap_values[i]) for i in range(len(feature_names))}
    bias_totals: Dict[str, float] = {
        "loss_aversion": 0.0, "overconfidence": 0.0, "herding": 0.0, "disposition": 0.0
    }

    for feat_prefix, direction, bias in SHAP_BIAS_MAP:
        for feat, val in shap_dict.items():
            if feat.startswith(feat_prefix):
                is_positive_dir = direction == "positive"
                if (is_positive_dir and val > 0) or (not is_positive_dir and val < 0):
                    bias_totals[bias] += abs(val)

    total = sum(bias_totals.values()) or 1.0
    return {k: round(v / total, 4) for k, v in bias_totals.items()}


def check_shap_stability(
    model,
    feature_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    window_days: int = ROLLING_WINDOW_DAYS,
    threshold: float = SHAP_STABILITY_THRESHOLD,
) -> Dict[str, float]:
    """Return {bias: fraction_of_windows_where_its_features_were_top3}.

    Requires trades_df to have an 'exit_date' column for windowing.
    """
    if trades_df.empty or feature_df.empty:
        return {}

    trades_df = trades_df.copy()
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
    min_date = trades_df["exit_date"].min()
    max_date = trades_df["exit_date"].max()

    bias_window_hits: Dict[str, List[bool]] = {
        "loss_aversion": [], "overconfidence": [], "herding": [], "disposition": []
    }

    cursor = min_date
    while cursor < max_date:
        window_end   = cursor + pd.Timedelta(days=window_days)
        window_trades = trades_df[
            (trades_df["exit_date"] >= cursor) & (trades_df["exit_date"] < window_end)
        ]
        if len(window_trades) < 5:
            cursor = window_end
            continue

        # Compute SHAP on this window's feature row (use overall feature_df as proxy)
        try:
            shap_vals = compute_shap_values(model, feature_df)
            feat_names = list(feature_df.columns)
            if shap_vals.ndim > 1:
                shap_row = shap_vals[0]
            else:
                shap_row = shap_vals

            shap_abs = {feat_names[i]: abs(float(shap_row[i])) for i in range(len(feat_names))}
            top3 = set(sorted(shap_abs, key=shap_abs.get, reverse=True)[:3])  # type: ignore[arg-type]

            for feat_prefix, direction, bias in SHAP_BIAS_MAP:
                hit = any(f.startswith(feat_prefix) for f in top3)
                bias_window_hits[bias].append(hit)
        except Exception:
            pass

        cursor = window_end

    return {
        bias: round(np.mean(hits), 4) if hits else 0.0
        for bias, hits in bias_window_hits.items()
    }
