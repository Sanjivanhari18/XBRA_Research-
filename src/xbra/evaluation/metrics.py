"""Phase 7 — Evaluation: bias classification metrics for the paper.

Produces Table 1 of the paper:
  Per-bias Precision / Recall / F1 / Support
  Overall macro-average F1

Ground truth comes from synthetic dataset (known injected bias labels).
Predictions come from FusedBiasVector.predicted_bias for all investors.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

BIAS_CLASSES = ["loss_averse", "overconfident", "herding", "disposition", "mixed", "neutral"]


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> pd.DataFrame:
    """Return per-class precision, recall, F1, and support.

    Args:
        y_true: ground-truth bias labels (from ground_truth.csv)
        y_pred: predicted bias labels (from pipeline FusedBiasVector)

    Returns:
        DataFrame with columns [class, precision, recall, f1, support]
    """
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        raise RuntimeError("scikit-learn not installed")

    report: dict[str, Any] = classification_report(
        y_true, y_pred, labels=BIAS_CLASSES, output_dict=True, zero_division=0
    )
    rows = []
    for cls in BIAS_CLASSES:
        if cls in report:
            rows.append({
                "class":     cls,
                "precision": round(report[cls]["precision"], 3),
                "recall":    round(report[cls]["recall"], 3),
                "f1":        round(report[cls]["f1-score"], 3),
                "support":   int(report[cls]["support"]),
            })
    rows.append({
        "class":     "macro avg",
        "precision": round(report["macro avg"]["precision"], 3),
        "recall":    round(report["macro avg"]["recall"], 3),
        "f1":        round(report["macro avg"]["f1-score"], 3),
        "support":   int(report["macro avg"]["support"]),
    })
    return pd.DataFrame(rows)


def run_bias_classification_eval(pipeline_results_path: str) -> pd.DataFrame:
    """Load pipeline results, compare to ground truth, return metrics table.

    pipeline_results_path: path to a CSV with columns [investor_id, predicted_bias]
    """
    from src.xbra.data.registry import DataRegistry

    gt = DataRegistry.ground_truth()
    preds = pd.read_csv(pipeline_results_path)
    merged = gt.merge(preds, on="investor_id")
    return compute_classification_metrics(
        merged["ground_truth_bias"].tolist(),
        merged["predicted_bias"].tolist(),
    )
