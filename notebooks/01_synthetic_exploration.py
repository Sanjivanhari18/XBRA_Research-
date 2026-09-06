"""Notebook 01 — Explore the synthetic dataset.

Run as a script or open in Jupyter (jupytext paired notebook).
Covers:
  - Trade distribution per bias class
  - Holding-time asymmetry validation (do profiles behave as designed?)
  - Basic P&L statistics
  - Feature distributions for the Behavior Agent

Run:
    python notebooks/01_synthetic_exploration.py
    (or: jupytext --to notebook notebooks/01_synthetic_exploration.py)
"""

# %% Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.xbra.data.registry import DataRegistry

# %% Load data
try:
    trades = DataRegistry.trades()
    gt = DataRegistry.ground_truth()
    print(f"Loaded {len(trades):,} trades across {trades['investor_id'].nunique()} investors")
    print(gt["ground_truth_bias"].value_counts())
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Run: python scripts/generate_synthetic.py")
    sys.exit(1)

# %% Holding-time asymmetry by bias class
def compute_holding_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """Compute median holding days for winners vs losers per investor."""
    return (
        df.groupby(["investor_id", "is_winner"])["holding_days"]
        .median()
        .unstack(fill_value=0)
        .rename(columns={True: "winner_days", False: "loser_days"})
        .assign(asymmetry=lambda x: x["loser_days"] / x["winner_days"].replace(0, np.nan))
    )

asy = compute_holding_asymmetry(trades)
asy = asy.merge(gt, on="investor_id")

print("\n--- Holding-time asymmetry by bias class ---")
print(asy.groupby("ground_truth_bias")["asymmetry"].median().round(2))

# %% Plot asymmetry distribution
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, bias in zip(axes.flat, asy["ground_truth_bias"].unique()):
    subset = asy[asy["ground_truth_bias"] == bias]["asymmetry"].dropna()
    ax.hist(subset, bins=15, edgecolor="black", color="steelblue", alpha=0.8)
    ax.axvline(1.0, color="red", linestyle="--", label="symmetric (=1)")
    ax.set_title(f"{bias}\n(median: {subset.median():.2f}×)")
    ax.set_xlabel("loser_days / winner_days")
    ax.legend(fontsize=7)

plt.suptitle("Holding-time Asymmetry Distribution by Bias Class", fontsize=12)
plt.tight_layout()
plt.savefig("notebooks/figures/01_holding_asymmetry.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: notebooks/figures/01_holding_asymmetry.png")

# %% P&L statistics by bias class
pnl = trades.merge(gt, on="investor_id")
print("\n--- Mean realized P&L by bias class ---")
print(pnl.groupby("ground_truth_bias")["realized_pnl"].mean().round(2))
print("\n--- Win rate by bias class ---")
print(pnl.groupby("ground_truth_bias")["is_winner"].mean().round(3))
