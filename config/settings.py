"""Central configuration for XBRA. All tuneable constants live here."""

from pathlib import Path
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

for _d in [RAW_DIR, SYNTHETIC_DIR, PROCESSED_DIR, MODELS_DIR, DOCS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ingestion / data quality
# ---------------------------------------------------------------------------
MIN_TRADES = 20                  # minimum closed positions required for analysis
MIN_DATA_QUALITY_THRESHOLD = 0.70   # below this, warn and proceed with caution

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
ORCHESTRATOR_MODEL = "llama3.2:3b"   # fast routing model
ANALYSIS_MODEL    = "llama3.2:3b"    # full reasoning model (70b not pulled)
LLM_TEMPERATURE   = 0.1
LLM_MAX_TOKENS    = 2048

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
MARKET_WINDOW_START = "2020-01-01"
MARKET_WINDOW_END   = "2024-12-31"
BENCHMARK_TICKER    = "SPY"

# US tickers used as the synthetic trade universe
TRADE_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM",
    "BAC", "JNJ", "XOM", "PFE", "V", "MA", "WMT", "NFLX", "AMD",
    "INTC", "DIS", "BABA", "GS", "UNH", "HD", "CVX", "KO",
]

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
N_INVESTORS          = 60       # total synthetic investor profiles
TRADES_PER_INVESTOR  = (100, 250)  # (min, max) trades per profile
RANDOM_SEED          = 42

BIAS_DISTRIBUTION = {
    "loss_averse":    10,
    "overconfident":  10,
    "herding":        10,
    "disposition":    10,
    "mixed":          10,
    "neutral":        10,
}

# ---------------------------------------------------------------------------
# FinBERT
# ---------------------------------------------------------------------------
FINBERT_MODEL_NAME = "ProsusAI/finbert"
SENTIMENT_WINDOW_DAYS = 3   # news sentiment window around a trade entry

# ---------------------------------------------------------------------------
# Risk / ML
# ---------------------------------------------------------------------------
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
}
SHAP_STABILITY_THRESHOLD = 0.80   # bias reported only if consistent in ≥80% rolling windows
ROLLING_WINDOW_DAYS = 30

# ---------------------------------------------------------------------------
# Clustering (Strategy Agent + Signal Fusion)
# ---------------------------------------------------------------------------
N_STRATEGY_CLUSTERS = 4   # day-trader, swing, buy-hold-drifter, momentum-chaser
KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 500

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = str(ROOT / "chroma_db")
CHROMA_COLLECTION  = "xbra_memory"

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
PDF_OUTPUT_DIR = DATA_DIR / "reports"
PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
