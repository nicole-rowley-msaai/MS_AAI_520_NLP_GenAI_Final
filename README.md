# EquityAgent

**Evidence-Grounded Agentic Equity Research**

**AAI 520: Natural Language Processing and Generative AI | Final Team Project**

EquityAgent turns a stock symbol into an evidence-grounded research brief. A LangGraph agent plans the research, calls financial data and retrieval tools, routes work to earnings, market and news specialists, verifies every figure in their findings against the raw tool data, writes a brief, evaluates and revises it under a quality gate, reflects on its own run, and stores lessons that the next run's planner reads.

**Research team:** Nicole Rowley (agent orchestration and research tools), Ashley Figueroa (financial NLP), and Pawan Bishwokarma (evaluation and integration).

> Academic research demonstration. Generated briefs are analytical outputs, not investment advice. Tool data, LLM judgments, and numeric coverage have limitations.

## Where to start

| Open this | To see |
|---|---|
| [`EquityAgent_Final_Executed.html`](EquityAgent_Final_Executed.html) | **The submission:** the executed notebook (v10.4 run of October 19, 2026) with every output, figure and interpretation |
| [`notebooks/final_project.ipynb`](notebooks/final_project.ipynb) | The same notebook, executable in Colab |
| [`Results/`](Results/README.md) | Every table and figure from that run as CSV, Markdown, JSON and PNG, named by notebook section; `Results/README.md` is the index |
| [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md) | Who built what |
| [`docs/nlp_track.md`](docs/nlp_track.md) | NLP design, diagnostics and experiments |

The submitted run is tagged `v10.4`. The code it ran is commit `09782eb`; the setup cell of the notebook prints that commit.

## Assignment requirements and evidence

The notebook has three parts: **Part A** data and sentiment modeling, **Part B** one research run shown step by step, **Part C** evaluation.

| Required capability | Implementation | Notebook evidence |
|---|---|---|
| **Research planning** | `src/agents/planner.py` | **B2** plan with tasks, tools, purposes and priorities |
| **Dynamic tool use** | `src/tools/` (Yahoo Finance, SEC EDGAR, FRED, NewsAPI) | **B3** every tool call with its arguments and result |
| **Self-reflection** | `src/agents/evaluator.py` (`reflect`) | **B9** weakest section, unsupported claims, missing data, lessons |
| **Learns across runs** | `src/memory/store.py`, `memory/notes.json` | **B10** lessons stored, then read by a second run whose plan cites them; **C1** memory on vs off |
| **Prompt chaining:** ingest → preprocess → classify → extract → summarize | `src/chains/news_chain.py` | **B5** output of every chain step; **A3–A4** classifier evaluation |
| **Routing** | `src/agents/router.py` | **B4** each step's specialist and the stated reason |
| **Evaluator–optimizer:** generate → evaluate → refine | `src/agents/evaluator.py`, `src/agents/evaluation_pipeline.py`, `src/graph.py` | **B7** scores per draft, critiques, numeric audit, gate decision; **C2** policy replay |
| **Data verification** (addition) | `src/agents/verifier.py` | **B6b** every finding figure confirmed, corrected or flagged against raw tool data |

## System architecture

```text
Stock ticker
    |
    v
Planner  <------------------------------ Stored lessons (when enabled)
    |
    v
Router --> Earnings specialist --> Yahoo Finance + SEC EDGAR retrieval
       --> News specialist     --> Ingest > Preprocess > Classify > Extract > Summarize
       --> Market specialist   --> Prices + FRED macro data
    |
    v
Data verification specialist --> every finding figure checked against raw tool output
    |
    v
Synthesizer --> Evaluator + numeric audit --> Quality gate
                       |                           |
                       |     revision required     |
                       +<------ Optimizer <--------+
                       |
                       v
            Kept draft / research brief
                       |
                       v
                 Self-reflection
                       |
                       v
              Persistent lesson storage
```

The LangGraph path is **plan → route → specialists → verify → synthesize → evaluate ⇄ optimize → reflect**. The three specialists run in parallel. The trace records plans, routed tasks, tool calls and raw tool data, verification results, evaluation critiques, gate decisions and memory events.

**Models.** Planner, router and news chain: `gpt-6-luna`. Specialists, synthesizer and optimizer: `claude-sonnet-5`. Evaluator: `gpt-6-sol`, a different model family from the writer so drafts are not graded by the model that wrote them. The verification specialist and the numeric audit use no model. Defaults are in [`src/config.py`](src/config.py) and can be overridden with environment variables.

## Evaluation design

Two evaluation modes are kept so they can be compared:

- **Baseline evaluation** (default): an evaluator–optimizer loop that stops once the rubric average reaches 4.0 after at least one revision.
- **Reviewed evaluation** (`reviewed_evaluation=True`, used for every run in the notebook): adds accuracy and evidence floors of 4/5, a numeric audit that checks recognized figures in the brief against the findings and blocks acceptance on a contradiction, and selection of the best accepted draft rather than the last one.

Three details matter when reading the results. Figures are compared at the precision the brief writes them, so "$215.9B" for a finding of "$215,938M" is consistent, not a contradiction. The audit recognizes only explicitly formatted claims (metric, optional fiscal year, value), so "no contradictions" is not a claim that every figure is correct. And the optimizer is sent only the numeric issues the live policy can reject.

Part C adds two comparisons. The **evidence-matched baseline** (`src/evaluation_lead_benchmarks.py`) is a single-pass brief written from the same findings the agent used; the gap between it and the agent's kept draft isolates what the revision loop and gate add beyond access to the evidence. The **policy replay** in C2 applies both acceptance rules to the same scored drafts; it is a comparison of stopping rules, not a second independent evaluation.

```python
from src.graph import run_agent

report, trace, state = run_agent("NVDA", use_memory=True, reviewed_evaluation=True)

print("Accepted:", state["accepted"])
print("Kept revision:", state["selected_iteration"])
print("Rubric average:", state["evaluation"].average)
print("Verification:", state["verification"])
print(report)
```

`run_agent` returns `(report, trace, state)`. It makes live data and LLM requests; credentials, quotas and provider charges apply.

## Results of the submitted run

From the v10.4 run of October 19, 2026 (`Results/C1_table1_integrated_live_run.csv`). One run per ticker; live data and LLM judgments vary between runs.

| Ticker | No-tools baseline | Same-evidence baseline | Agent first draft | Kept agent draft | Accepted | Finding figures verified vs raw data |
|---|---:|---:|---:|---:|---|---:|
| NVDA | 2.0 / 5 | 4.0 / 5 | 4.0 / 5 | **4.6 / 5** | Yes | 59 / 62 |
| JPM | 1.6 / 5 | 3.6 / 5 | 3.6 / 5 | **4.6 / 5** | Yes | 65 / 83 |
| XOM | 2.0 / 5 | 4.4 / 5 | 4.2 / 5 | **4.4 / 5** | Yes | 51 / 51 |

- **Evidence does most of the work.** Briefs written from the agent's findings scored about 2.5 points above the no-tools baseline. The evaluate-and-revise loop added +0.5 on average over the same-evidence baseline in this run and +0.1 in a second run of the same code, so its contribution is real in direction but not stable in size.
- **Sentiment classification (A3–A4).** On 300 held-out Financial PhraseBank sentences, the LLM's majority-vote macro-F1 was 0.863 zero-shot and 0.846 few-shot, against 0.771 for TF-IDF logistic regression and 0.575 for the keyword rule. Across four runs of the same code the LLM ranged from 0.827 to 0.863 and the two prompts traded places, so the gap between them is within run-to-run variation.
- **Verification and audit.** The verification specialist confirmed 175 of 196 finding figures against raw tool data and corrected none; the unmatched figures are bank measures it does not compute. The numeric audit found no contradictions. Every tool call succeeded.
- **Memory.** With stored lessons the planner wrote an 8-step plan that cites them in four steps, but the brief did not score higher (4.6 vs 4.8). The lessons ask for evidence the current tools cannot retrieve, such as peer multiples and segment revenue; learning is demonstrated at the planning level.

**Validation.** 188 offline tests pass (`python -m pytest -q`); they use mocked models and tools, so passing does not establish that live research claims are correct. Every code cell in the submitted notebook executed with zero errors.

## What changed in v10 (October 19, 2026)

- **Data verification specialist** (`src/agents/verifier.py`): a rule-based agent between the specialists and the synthesizer that checks every figure in the findings against the raw tool output and passes corrections and flags downstream.
- **Evidence-matched baseline** in C1.
- **Numeric audit accepts rounding**; exact matching had rejected correct drafts as contradictions in every earlier run.
- **Memory reads the newest lessons first**, the reflection step is shown the lessons already stored so it adds only new ones, and the optimizer receives only numeric issues that can block acceptance.
- **Self-checking notebook setup:** the Drive working copy is reset to the published commit, modules from an earlier run are dropped, and the import cell stops if the agent code did not load from that checkout.
- **PEP 8** formatting of Python files and notebook code cells at 79 columns.

## Data sources

| Resource | Purpose |
|---|---|
| Yahoo Finance (`yfinance`) | Prices, key statistics, financial statements |
| SEC EDGAR | 10-K retrieval and semantic search over Risk Factors and MD&A |
| FRED | CPI, Fed funds rate, 10-year yield, unemployment, real GDP |
| NewsAPI | Recent headlines (last 30 days) |
| Financial PhraseBank (Malo et al., 2014), via Kaggle | Labeled sentences for classifier evaluation, grouped by annotator agreement |
| `all-MiniLM-L6-v2` | Embeddings for filing retrieval |

## Repository structure

```text
src/
  agents/
    planner.py, router.py, specialists.py, synthesizer.py, evaluator.py
    verifier.py                 # data verification specialist
    evaluation_lead.py          # quality policy, revision budget, draft selection
    evaluation_pipeline.py      # combined rubric and numeric review session
    numeric_validation.py       # claim parser and audit against findings
  chains/                       # news prompt chain, PhraseBank evaluation
  memory/store.py               # persistent lessons (memory/notes.json)
  tools/                        # market, filings, macro, news, retrieval
  graph.py                      # LangGraph workflow and run_agent(...)
  evaluation.py                 # no-tools baseline, scoring helpers
  evaluation_lead_benchmarks.py # evidence-matched
