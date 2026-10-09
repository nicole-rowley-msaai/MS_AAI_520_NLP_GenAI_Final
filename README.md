# EquityAgent

**An Autonomous Agentic Workflow for Investment Research**

An autonomous Investment Research Agent that turns a stock ticker into a researched, self-evaluated investment brief. The agent plans its own research, pulls live data through tools, routes tasks to specialist analyzers, refines its draft through an evaluator–optimizer loop, and keeps notes that improve future runs.

> Course final project. Analysis only; not investment advice.

## Features

| Requirement | How it's implemented |
|---|---|
| **Planning** | Planner node produces an ordered `ResearchPlan` (step, tool, purpose) |
| **Dynamic tool use** | LLM function calling over yfinance, FRED, NewsAPI, and `search_filings` (RAG over SEC 10-Ks) |
| **Self-reflection** | After each report, the agent critiques its output (weakest section, unsupported claims, confidence) and its process |
| **Learning across runs** | 3–5 lessons per run saved to `memory/notes.json` and read by the planner next time |
| **Prompt chaining** | News: Ingest → Preprocess → Classify → Extract → Summarize |
| **Routing** | Router sends each task to the earnings, news, or market agent, with a logged reason |
| **Evaluator–optimizer** | Draft scored 1–5 on a 5-part rubric; rewritten until the average is ≥ 4 or 3 loops pass |

## Architecture

```
Ticker ─► Planner ◄── memory/notes.json
             │
           Router ─► Earnings agent  (yfinance financials, EDGAR RAG)
                  ├► News agent      (5-step prompt chain)
                  └► Market agent    (prices, technicals, FRED macro)
             │
        Synthesizer ─► Evaluator ─(below threshold)─► Optimizer ─┐
             ▲                                                   │
             └───────────────────────────────────────────────────┘
             │ (passes)
        Final report + reflection ─► writes lessons to memory
```

## Data sources

| Source | Used for |
|---|---|
| Yahoo Finance (`yfinance`) | Prices, ratios, financial statements, headlines |
| SEC EDGAR | 10-K Risk Factors and MD&A, chunked for retrieval |
| FRED API | CPI, Fed funds rate, 10-year yield, unemployment, GDP |
| NewsAPI | Live headlines for the news chain |
| Kaggle: [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) | 4,846 labeled headlines; test set for the classify step |

## Models

| Role | Model |
|---|---|
| News chain and router | GPT-6 Luna |
| Planner, synthesis, optimizer | Claude Sonnet 5 |
| Evaluator | GPT-6 Sol (different family from the writer) |
| Embeddings | `all-MiniLM-L6-v2` |

Model names are set in `src/config.py`.

## Repository structure

```
equity-agent/
├── src/
│   ├── tools/      yfinance, edgar, fred, news wrappers
│   ├── chains/     news_chain.py
│   ├── agents/     planner, router, earnings, news, market, evaluator, optimizer
│   ├── memory/     store.py
│   ├── trace.py    records every step of a run
│   ├── display.py  renders the trace in the notebook
│   ├── config.py   model names and settings
│   └── graph.py    LangGraph wiring
├── notebooks/final_project.ipynb
├── memory/notes.json
├── runs/           saved trace per run
├── data/           cache + Kaggle sample (large files gitignored)
└── tests/
```

## Setup

```bash
git clone <repo-url>
cd equity-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then add your keys
```

Required in `.env`:

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
FRED_API_KEY=
NEWSAPI_KEY=
EDGAR_USER_AGENT="Your Name your@email.com"
```

Download the Kaggle dataset into `data/kaggle/`.

## Usage

```python
from src.graph import run_agent

report, trace = run_agent("NVDA")
```

Or open `notebooks/final_project.ipynb` and run all cells. The notebook prints the research plan, tool calls, routing decisions, each prompt-chain step, evaluator scores and critiques, the final report, reflection, and memory changes.

## Evaluation

Demo tickers: **NVDA, JPM, XOM**.

- Full agent vs a single-prompt baseline
- Rubric score per evaluator iteration
- News classification accuracy, F1, and confusion matrix on a held-out Kaggle sample
- Report figures checked against tool outputs
- Memory on vs off on a repeat run

## Development

- One feature branch per task; pull requests need one reviewer
- PEP 8 enforced with `black` and `ruff` via pre-commit
- `nbstripout` keeps notebook diffs clean
- Run tests with `pytest`

## Team

| Role | Owns |
|---|---|
| Data lead | Tool wrappers, caching, tests |
| NLP lead | Kaggle EDA, news prompt chain, classifier evaluation |
| Agent lead | Planner, router, specialist agents, filing retrieval, LangGraph wiring |
| Evaluation lead | Evaluator–optimizer, reflection, memory, baseline comparison |
