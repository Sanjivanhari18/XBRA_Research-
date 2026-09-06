"""Run the XBRA pipeline on all synthetic investors and save results for Phase 7 eval.

Usage:
    python scripts/run_pipeline_all.py [--max N]

Outputs:
    data/paper_results/pipeline_results.csv  — investor_id, predicted_bias, max_drawdown, ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from typing import Optional
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.xbra.data.registry import DataRegistry
from config.settings import DATA_DIR


def main(max_investors: Optional[int] = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    investor_ids = DataRegistry.investor_ids()
    if max_investors:
        investor_ids = investor_ids[:max_investors]

    logger.info("Running pipeline on {} investors ...", len(investor_ids))

    results = []
    failed  = []

    for inv_id in tqdm(investor_ids, desc="Pipeline"):
        try:
            from src.xbra.orchestrator.graph import run_pipeline
            state = run_pipeline(inv_id)
            row = {"investor_id": inv_id}

            if state.get("fused_bias"):
                fused = state["fused_bias"]
                row["predicted_bias"]  = fused.get("predicted_bias", "neutral")
                row["loss_aversion"]   = fused.get("loss_aversion", 0)
                row["overconfidence"]  = fused.get("overconfidence", 0)
                row["herding"]         = fused.get("herding", 0)
                row["disposition"]     = fused.get("disposition", 0)

            if state.get("risk_output"):
                risk = state["risk_output"]
                row["max_drawdown"]       = risk.get("max_drawdown", 0)
                row["sharpe_ratio"]       = risk.get("sharpe_ratio", 0)
                row["r_behavioral_pct"]   = risk.get("r_behavioral_pct", 0)
                row["r_market_pct"]       = risk.get("r_market_pct", 0)
                row["r_interaction_pct"]  = risk.get("r_interaction_pct", 0)
                row["peer_percentile"]    = risk.get("peer_percentile", 50)

            if state.get("strategy_output"):
                row["archetype"] = state["strategy_output"].get("archetype", "")

            results.append(row)
        except Exception as e:
            logger.warning("Failed for {}: {}", inv_id, e)
            failed.append(inv_id)

    out_dir = DATA_DIR / "paper_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pipeline_results.csv"

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    logger.success("Saved {} results to {}", len(results), out_path)

    if failed:
        logger.warning("Failed investors ({}): {}", len(failed), failed[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Limit number of investors")
    args = parser.parse_args()
    main(max_investors=args.max)
