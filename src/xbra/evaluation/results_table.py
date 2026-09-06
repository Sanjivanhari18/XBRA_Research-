"""Phase 7 — Generate all paper result tables as CSV + LaTeX.

Tables produced:
  Table 1: Bias classification (precision / recall / F1 per class)
  Table 2: SHAP feature stability
  Table 3: Ablation study results
  Table 4: Risk decomposition per regime (H2)
  Table 5: Runtime / latency per pipeline stage

Run:
    python -m src.xbra.evaluation.results_table --all
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR

OUTPUT_DIR = DATA_DIR / "paper_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    df.to_latex(OUTPUT_DIR / f"{name}.tex", index=False, float_format="%.3f")


def generate_table1(pipeline_results_path: str) -> pd.DataFrame:
    from src.xbra.evaluation.metrics import run_bias_classification_eval
    df = run_bias_classification_eval(pipeline_results_path)
    _save(df, "table1_bias_classification")
    return df


def generate_table3(investor_ids: list[str]) -> pd.DataFrame:
    from src.xbra.evaluation.ablation import run_ablation
    df = run_ablation(investor_ids)
    _save(df, "table3_ablation")
    return df


def generate_all(pipeline_results_path: str) -> None:
    """Run all evaluation functions and save tables to OUTPUT_DIR."""
    from src.xbra.data.registry import DataRegistry

    investor_ids = DataRegistry.investor_ids()
    generate_table1(pipeline_results_path)
    generate_table3(investor_ids)
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to pipeline_results.csv (investor_id, predicted_bias)")
    args = parser.parse_args()
    generate_all(args.results)
