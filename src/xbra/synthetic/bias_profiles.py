"""Bias profile definitions — each profile is a bundle of behavioural parameters
that control how a synthetic investor makes decisions.

When the trade simulator draws from these parameters it produces a trade log
that exhibits the intended bias pattern, which the pipeline can then detect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class BiasProfile:
    # Holding-time asymmetry: (winner_days_range, loser_days_range)
    winner_hold_days: Tuple[int, int]
    loser_hold_days:  Tuple[int, int]

    # Position sizing: fraction of portfolio per trade
    position_size_range: Tuple[float, float]  # (min_frac, max_frac)

    # Trade frequency: avg days between trade entries (lower = more active)
    inter_trade_days_range: Tuple[int, int]

    # Profit-taking threshold: sell winner once return ≥ this value
    profit_take_threshold: float   # e.g. 0.05 → +5 %

    # Loss-cut threshold: sell loser once return ≤ this value (negative)
    # None means never cut (hold until exit_days forces close)
    loss_cut_threshold: float | None

    # Herding: probability of entering a trade in the direction of recent momentum
    momentum_follow_prob: float   # 0.0 = random, 1.0 = always follows momentum

    # Disposition: probability of exiting a WINNER early (realising gains)
    early_exit_winner_prob: float

    # Description label (for paper / logging)
    label: str
    description: str = ""

    # Ground-truth bias for evaluation
    ground_truth_features: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Six canonical profiles
# ---------------------------------------------------------------------------

PROFILES: dict[str, BiasProfile] = {

    "loss_averse": BiasProfile(
        winner_hold_days      = (3, 12),    # cut winners quickly
        loser_hold_days       = (25, 90),   # hold losers a long time
        position_size_range   = (0.04, 0.10),
        inter_trade_days_range= (5, 15),
        profit_take_threshold = 0.04,       # exit winner at small gain
        loss_cut_threshold    = None,       # never cuts losses
        momentum_follow_prob  = 0.40,
        early_exit_winner_prob= 0.70,
        label       = "loss_averse",
        description = ("Holds losing trades 3-5× longer than winning trades. "
                       "Exits winners quickly on small gains."),
        ground_truth_features = {
            "holding_time_asymmetry_expected": ">3.0",
            "loss_cut_prob_expected": "<0.05",
        },
    ),

    "overconfident": BiasProfile(
        winner_hold_days      = (1, 7),
        loser_hold_days       = (1, 7),
        position_size_range   = (0.12, 0.30),   # large bets
        inter_trade_days_range= (1, 4),          # trades very frequently
        profit_take_threshold = 0.08,
        loss_cut_threshold    = -0.05,
        momentum_follow_prob  = 0.55,
        early_exit_winner_prob= 0.30,
        label       = "overconfident",
        description = ("High turnover, large concentrated positions. "
                       "Trades frequently regardless of market conditions."),
        ground_truth_features = {
            "turnover_rate_expected": ">30_trades_per_month",
            "avg_position_size_expected": ">0.15",
        },
    ),

    "herding": BiasProfile(
        winner_hold_days      = (5, 20),
        loser_hold_days       = (10, 30),
        position_size_range   = (0.05, 0.12),
        inter_trade_days_range= (3, 10),
        profit_take_threshold = 0.06,
        loss_cut_threshold    = -0.08,
        momentum_follow_prob  = 0.90,   # almost always follows momentum
        early_exit_winner_prob= 0.40,
        label       = "herding",
        description = ("Trades in the direction of recent market momentum. "
                       "Entry timing clusters around market sentiment spikes."),
        ground_truth_features = {
            "momentum_follow_rate_expected": ">0.75",
        },
    ),

    "disposition": BiasProfile(
        winner_hold_days      = (2, 10),    # realises winners fast
        loser_hold_days       = (30, 120),  # holds losers a very long time
        position_size_range   = (0.05, 0.15),
        inter_trade_days_range= (4, 12),
        profit_take_threshold = 0.03,       # exit at tiny gain (realise it)
        loss_cut_threshold    = None,       # never realises the loss
        momentum_follow_prob  = 0.45,
        early_exit_winner_prob= 0.85,       # systematically realises gains
        label       = "disposition",
        description = ("Systematically realises gains (sells winners) while "
                       "holding losers indefinitely — the disposition effect."),
        ground_truth_features = {
            "proportion_gains_realised_expected": ">0.70",
            "proportion_losses_realised_expected": "<0.20",
        },
    ),

    "mixed": BiasProfile(
        winner_hold_days      = (5, 30),
        loser_hold_days       = (10, 45),
        position_size_range   = (0.06, 0.18),
        inter_trade_days_range= (3, 12),
        profit_take_threshold = 0.06,
        loss_cut_threshold    = -0.07,
        momentum_follow_prob  = 0.60,
        early_exit_winner_prob= 0.50,
        label       = "mixed",
        description = ("Exhibits moderate levels of multiple biases — "
                       "mild loss-aversion combined with moderate herding."),
        ground_truth_features = {},
    ),

    "neutral": BiasProfile(
        winner_hold_days      = (10, 40),
        loser_hold_days       = (10, 40),    # symmetric holding
        position_size_range   = (0.04, 0.08),
        inter_trade_days_range= (7, 20),
        profit_take_threshold = 0.10,
        loss_cut_threshold    = -0.08,
        momentum_follow_prob  = 0.50,        # coin-flip on momentum
        early_exit_winner_prob= 0.30,
        label       = "neutral",
        description = ("Relatively efficient trader with no strong bias. "
                       "Symmetric holding times and disciplined position sizing."),
        ground_truth_features = {
            "holding_time_asymmetry_expected": "~1.0",
        },
    ),
}
