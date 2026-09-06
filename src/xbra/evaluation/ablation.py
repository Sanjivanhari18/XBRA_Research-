"""Phase 7 — Ablation study (H3: agent-based decomposition beats monolithic baseline).

Runs 5 experimental conditions:
  1. full_pipeline      — all 4 agents (reference)
  2. no_behavior        — skip Behavior Agent
  3. no_market          — skip Market Agent
  4. no_strategy        — skip Strategy Agent
  5. monolithic_xgb     — single XGBoost on raw features (no agent decomposition)

For each condition: run bias classification on all investors,
record macro-F1 and drawdown-prediction R².

Outputs Table 3 of the paper.
"""

from __future__ import annotations

import pandas as pd

CONDITIONS = [
    "full_pipeline",
    "no_behavior",
    "no_market",
    "no_strategy",
    "monolithic_xgb",
]


def run_ablation(investor_ids: list[str]) -> pd.DataFrame:
    """Run all ablation conditions and return results table.

    Returns DataFrame with columns:
        [condition, macro_f1, drawdown_r2, runtime_sec]
    """
    # TODO Phase 7:
    #   For each condition:
    #     - build a modified pipeline (disable the named agent)
    #     - run on all investor_ids
    #     - collect predicted_bias + drawdown prediction
    #     - compute macro-F1 (classification) + R² (regression)
    raise NotImplementedError("Phase 7 — run_ablation")


def monolithic_xgb_baseline(investor_ids: list[str]) -> dict:
    """Single XGBoost trained on raw trade features (no agent decomposition).

    Used as the H3 baseline — if the multi-agent pipeline doesn't beat this,
    the agentic framing doesn't add value.
    """
    # TODO Phase 7: feature extraction from raw trades → XGBoost → predict drawdown
    raise NotImplementedError("Phase 7 — monolithic_xgb_baseline")
