# XBRA Project Context

## What Is This?

XBRA (**Explainable Behavioral Risk Attribution**) is a proof-of-concept agentic AI framework for detecting, quantifying, and explaining behavioral biases in retail investor trading data. It is being built as a systems/software conference paper with a working implementation that generates result tables.

---

## Goals

- Detect and classify behavioral biases (loss aversion, overconfidence, herding, disposition effect) from retail trading records
- Quantify how much each bias contributes to portfolio drawdown
- Generate human-readable explanations via SHAP + LLM narrative
- Produce reproducible paper results (precision/recall/F1, silhouette score, SHAP stability)

---

## Paper Hypotheses

| ID | Hypothesis |
|----|-----------|
| H1 | Behavioral features explain a significant % of portfolio drawdown |
| H2 | Behavioral risk attribution varies across market regimes |
| H3 | Agent-based decomposition outperforms a monolithic single-model baseline (ablation) |

---

## Technology Stack (Zero-Cost)

| Layer | Tool |
|-------|------|
| LLM (analysis) | `llama3.3:70b` via Ollama (local) |
| LLM (routing) | `llama3.2:3b` via Ollama (local) |
| Orchestration | LangGraph |
| ML / Attribution | XGBoost + SHAP |
| Sentiment | FinBERT (local) |
| Market Data | yfinance |
| Vector DB | ChromaDB |
| UI | Streamlit |
| Testing | pytest |

> **No paid APIs.** All models run locally via Ollama.

---

## Implementation Phases

```
Phase 0 → Phase 1 → Phase 2 → Phase 3a ∥ Phase 3b → Phase 3c → Phase 3d → Phase 4 → Phase 5 → Phase 6 → Phase 7
```

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Synthetic data generator (60 investor profiles, 5 archetypes) | **COMPLETE** — 9/9 tests passing |
| 1 | Ingestion & normalization pipeline | Stub defined |
| 2 | Market context agent (yfinance + regime detection) | Stub defined |
| 3a | Behavior detection agent (XGBoost classifier) | Stub defined |
| 3b | Strategy agent (clustering / K-Means) | Stub defined |
| 3c | Signal fusion layer | Stub defined |
| 3d | Risk attribution agent (SHAP) | Stub defined |
| 4 | Explainability layer (SHAP narratives via LLM) | Stub defined |
| 5 | ChromaDB memory & RAG retrieval | Stub defined |
| 6 | PDF report generation | Stub defined |
| 7 | Streamlit UI | Stub defined |

---

## Synthetic Dataset (Phase 0 Output)

60 investor profiles across 5 behavioral archetypes:

| Archetype | IDs | Count |
|-----------|-----|-------|
| Loss Averse | INV_LOSS_AVERSE_000–009 | 10 |
| Overconfident | INV_OVERCONFIDENT_010–019 | 10 |
| Herding | INV_HERDING_020–029 | 10 |
| Disposition Effect | INV_DISPOSITION_030–039 | 10 |
| Mixed | INV_MIXED_040–049 | 10 |
| Neutral (control) | INV_NEUTRAL_050–059 | 10 |

Synthetic profiles stored at: `data/synthetic/profiles.json`

---

## Repository Layout

```
XBRA imple/
├── src/xbra/
│   ├── agents/          # behavior, market, risk, strategy agents
│   ├── data/            # registry
│   ├── evaluation/      # metrics, ablation, SHAP stability, results tables
│   ├── explainability/  # narrative, SHAP layer, verifier
│   ├── fusion/          # signal fusion
│   ├── ingestion/       # loaders, normalizer
│   ├── llm/             # client (Ollama), prompts
│   ├── memory/          # ChromaDB store
│   ├── orchestrator/    # LangGraph graph, router, state
│   ├── report/          # PDF generator, charts, report node
│   ├── synthetic/       # bias profiles, generator, trade simulator
│   ├── ui/              # Streamlit app + components
│   ├── utils/           # IO, logging
│   └── schemas.py       # shared Pydantic schemas
├── config/              # settings
├── scripts/             # generate_synthetic.py, run_pipeline_all.py
├── notebooks/           # 01_synthetic_exploration.py
├── tests/               # pytest suite
├── data/
│   ├── synthetic/       # profiles.json
│   ├── paper_results/   # CSV + LaTeX result tables
│   └── reports/         # generated PDF reports (60 investor PDFs)
├── pyproject.toml
└── requirements.txt
```

---

## Design Principles

- **Reproducibility first** — all results must come from running the pipeline, not hardcoded
- **Zero cost** — no paid APIs or cloud services; everything runs locally
- **Measurable** — every phase outputs metrics that feed the paper's result tables
- **Modular** — each agent has a defined interface; phases can be implemented independently

---

## Python Environment

- Python 3.9.6
- Shared venv (until project venv is set up): `/Users/sanjivan/Documents/Gmu Research/triage-poc/.venv/bin/python3`
- Project-specific packages (yfinance, langgraph, chromadb, shap, xgboost, etc.) need to be installed before Phase 2+

---

## Key Files

| File | Purpose |
|------|---------|
| `src/xbra/schemas.py` | Shared Pydantic models across all agents |
| `src/xbra/orchestrator/graph.py` | LangGraph pipeline definition |
| `src/xbra/synthetic/generator.py` | Synthetic investor profile generator |
| `src/xbra/evaluation/metrics.py` | Precision/recall/F1, silhouette, SHAP stability |
| `scripts/run_pipeline_all.py` | End-to-end pipeline runner |
| `data/paper_results/` | LaTeX + CSV tables ready for paper inclusion |
