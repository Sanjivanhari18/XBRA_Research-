"""Phase 7 — SHAP stability analysis for the paper.

Measures how consistently the top SHAP features identify each bias
across rolling time windows. Reported in Table 2 of the paper:

  Feature | Bias | Appears in top-3 in X% of windows

Stability threshold: 80% (SHAP_STABILITY_THRESHOLD in settings.py).
Only biases meeting the threshold are reported in the final explanation.
"""

from __future__ import annotations

import pandas as pd

from config.settings import ROLLING_WINDOW_DAYS, SHAP_STABILITY_THRESHOLD


def compute_stability_table(
    investor_ids: list[str],
    window_days: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    """For each investor, compute per-feature SHAP stability across rolling windows.

    Returns DataFrame:
        [investor_id, feature, bias, top3_fraction, stable]
    """
    # TODO Phase 7:
    #   For each investor:
    #     split trade history into rolling windows of window_days
    #     for each window: compute SHAP values on the risk model
    #     track which features appear in top-3 per window
    #     stability = count(top-3) / total_windows
    raise NotImplementedError("Phase 7 — compute_stability_table")


def summarize_stability(stability_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across all investors: mean stability per feature."""
    # TODO Phase 7
    raise NotImplementedError("Phase 7 — summarize_stability")
