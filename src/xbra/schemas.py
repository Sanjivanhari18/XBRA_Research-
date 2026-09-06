"""Pydantic schemas for all inter-agent data contracts.

Every agent receives and returns dicts that conform to these models.
Using TypedDict-compatible Pydantic models so LangGraph state is type-safe.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BiasType(str, Enum):
    LOSS_AVERSE   = "loss_averse"
    OVERCONFIDENT = "overconfident"
    HERDING       = "herding"
    DISPOSITION   = "disposition"
    MIXED         = "mixed"
    NEUTRAL       = "neutral"


class MarketRegime(str, Enum):
    BULL     = "bull"
    BEAR     = "bear"
    SIDEWAYS = "sideways"


class StrategyArchetype(str, Enum):
    DAY_TRADER         = "day_trader"
    SWING_TRADER       = "swing_trader"
    BUY_HOLD_DRIFTER   = "buy_hold_drifter"
    MOMENTUM_CHASER    = "momentum_chaser"


# ---------------------------------------------------------------------------
# Core data units
# ---------------------------------------------------------------------------

class Trade(BaseModel):
    trade_id:       str
    investor_id:    str
    symbol:         str
    entry_date:     date
    exit_date:      date
    entry_price:    float
    exit_price:     float
    quantity:       int
    side:           str = "long"     # long / short
    realized_pnl:   float
    holding_days:   int
    is_winner:      bool

    model_config = ConfigDict(frozen=True)


class InvestorProfile(BaseModel):
    investor_id:       str
    ground_truth_bias: BiasType
    n_trades:          int
    trades:            List[Trade]
    portfolio_value:   float = 100_000.0
    metadata:          Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent inputs / outputs
# ---------------------------------------------------------------------------

class OrchestratorInput(BaseModel):
    investor_id:    str
    investor_data:  InvestorProfile
    user_query:     Optional[str] = None


class BehaviorAgentOutput(BaseModel):
    investor_id:         str
    loss_aversion_score: float = Field(ge=0.0, le=1.0)
    overconfidence_score: float = Field(ge=0.0, le=1.0)
    herding_score:        float = Field(ge=0.0, le=1.0)
    disposition_score:    float = Field(ge=0.0, le=1.0)
    confidence_per_bias:  Dict[str, float] = Field(default_factory=dict)
    evidencing_trade_ids: Dict[str, List[str]] = Field(default_factory=dict)
    llm_summary:          Optional[str] = None
    features:             Dict[str, float] = Field(default_factory=dict)


class MarketAgentOutput(BaseModel):
    investor_id:         str
    per_trade_context:   Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    regime_labels:       Dict[str, MarketRegime] = Field(default_factory=dict)
    sentiment_scores:    Dict[str, float] = Field(default_factory=dict)
    market_explains_flags: Dict[str, bool] = Field(default_factory=dict)
    llm_summary:         Optional[str] = None


class RiskAgentOutput(BaseModel):
    investor_id:          str
    max_drawdown:         float
    sharpe_ratio:         float
    sortino_ratio:        float
    var_95:               float
    concentration:        float
    r_behavioral_pct:     float    # % of drawdown explained by behavioral features
    r_market_pct:         float
    r_interaction_pct:    float
    regime_decomposition: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    peer_percentile:      float
    llm_summary:          Optional[str] = None
    shap_values:          Dict[str, float] = Field(default_factory=dict)


class StrategyAgentOutput(BaseModel):
    investor_id:              str
    archetype:                StrategyArchetype
    consistency_flag:         bool
    turnover_rate:            float
    sector_herfindahl:        float
    archetype_oscillation:    bool
    strategy_adjusted_biases: Dict[str, float] = Field(default_factory=dict)
    llm_summary:              Optional[str] = None


class FusedBiasVector(BaseModel):
    investor_id:         str
    loss_aversion:       float = Field(ge=0.0, le=1.0)
    overconfidence:      float = Field(ge=0.0, le=1.0)
    herding:             float = Field(ge=0.0, le=1.0)
    disposition:         float = Field(ge=0.0, le=1.0)
    dominant_bias:       BiasType
    silhouette_score:    float
    predicted_bias:      BiasType   # final classification (for eval)


class ExplanationOutput(BaseModel):
    investor_id:  str
    per_bias:     Dict[str, Dict[str, str]] = Field(default_factory=dict)
    # per_bias[bias_name] = {"pattern": ..., "evidence": ..., "consequence": ...}
    verified:     bool = False
    full_report:  Optional[str] = None


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------

class XBRAState(BaseModel):
    """Shared mutable state flowing through the LangGraph DAG."""

    investor_id:       str = ""
    investor_profile:  Optional[InvestorProfile] = None
    user_query:        Optional[str] = None

    # Agent outputs (populated as DAG executes)
    behavior_output:  Optional[BehaviorAgentOutput]  = None
    market_output:    Optional[MarketAgentOutput]    = None
    risk_output:      Optional[RiskAgentOutput]      = None
    strategy_output:  Optional[StrategyAgentOutput]  = None

    fused_bias:       Optional[FusedBiasVector]      = None
    explanation:      Optional[ExplanationOutput]    = None

    # Error propagation
    errors:           List[str] = Field(default_factory=list)
    stage:            str = "init"

    model_config = ConfigDict(arbitrary_types_allowed=True)
