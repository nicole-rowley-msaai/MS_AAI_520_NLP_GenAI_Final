# EquityAgent

**Evidence-Grounded Agentic Equity Research**

**AAI 520: Natural Language Processing and Generative AI | Final Team Project**

EquityAgent is a LangGraph-based investment research system that transforms a stock symbol into an evidence-grounded research brief. The agent develops a research plan, invokes financial data and retrieval tools, routes work to specialists, verifies every figure in their findings against the raw tool data, synthesizes a brief, evaluates and revises it under a quality gate, reflects on its process, and stores lessons for later runs.

**Research team:** Nicole Rowley (agent orchestration and research tools), Ashley Figueroa (financial NLP), and Pawan Bishwokarma (evaluation and integration).

> Academic research demonstration. Generated briefs are analytical outputs, not investment advice. Tool data, LLM judgments, and numeric coverage have limitations.

## Final deliverables

| File | Purpose |
|---|---|
| [`EquityAgent_Final_Executed.html`](EquityAgent_Final_Executed.html) | **Executed, readable notebook document for the course submission** (v10.4 run of October 19, 2026), including observed outputs, tables, figures, and interpretation |
| [`notebooks/final_project.ipynb`](notebooks/final_project.ipynb) | The executed Jupyter/Colab notebook containing Parts A, B, and C |
| [`Results/table1_integrated_live_run.csv`](Results/table1_integrated_live_run.csv) | Table 1, written by notebook cell C1 during the documented run |
| [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md) | Team responsibilities and implementation contributions |
| [`docs/nlp_track.md`](docs/nlp_track.md) | NLP design, diagnostics, and experiments |
| [`notebooks/04_evaluator_policy_comparison.ipynb`](notebooks/04_evaluator_policy_comparison.ipynb) | Supporting evaluator decision-policy comparison on an earlier run; the final notebook's C2 performs the same replay on the submitted run |

[GitHub repository](https://github.com/nicole-rowley-msaai/MS_AAI_520_NLP_GenAI_Final)

## Assignment requirements and evidence

The consolidated notebook organizes data and NLP experiments in **Part A**, the autonomous research demonstration in **Part B**, and comparative evaluation in **Part C**.

| Required capability | Implementation and notebook evidence |
|---|---|
| **Research planning** | `src/agents/planner.py` creates a structured plan; **B2** displays its steps and hypotheses |
| **Dynamic tool use** | `src/tools/` provides market, macroeconomic, news, and SEC filing retrieval; **B3** shows recorded tool calls |
| **Self-reflection** | `src/agents/evaluator.py` produces a reflective assessment after draft selection; **B9** records conclusions and limitations |
| **Learns across runs** | `src/memory/store.py` persists lessons in `memory/notes.json`; **B10** displays stored notes and the second run's plan, which cites them; **C1** compares runs with and without memory |
| **Prompt Chaining: Ingest → Preprocess → Classify → Extract → Summarize** | `src/chains/news_chain.py`; **A3–A4** evaluate classification and **B5** shows the intermediate chain outputs |
| **Routing** | `src/agents/router.py` directs research steps to earnings, news, or market specialists; **B4** displays choices and rationales |
| **Evaluator–Optimizer** | `src/agents/evaluator.py`, `src/agents/evaluation_pipeline.py`, `src/graph.py`; **B7** shows scoring, audit, feedback and revisions, while **C2** compares acceptance decisions |
| **Data verification** (addition) | `src/agents/verifier.py` checks every figure in the specialists' findings against the raw tool data before synthesis; **B6b** shows the results per specialist |

## System architecture

```text
Stock ticker
    |
    v
Planner  <------------------------------ Prior notes (when enabled)
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
Synthesizer --> Evaluator + numerical audit --> Quality gate
                       |                                |
                       |         revision required      |
                       +<-------- Optimizer <------------+
                       |
                       v
            Selected draft / research brief
                       |
                       v
                 Self-reflection
                       |
                       v
              Persistent lesson storage
```

The LangGraph path is **plan → route → specialists → verify → synthesize → evaluate ⇄ optimize → reflect**. Specialist tasks are dispatched in parallel. The trace records plans, routed tasks, tool results, raw tool data, verification results, evaluation critiques, revision decisions, and memory events.

## Evaluation design

The project retains two modes for reproducible methodological comparison:

- **Baseline evaluation** (default): LLM rubric scores guide an iterative evaluator–optimizer loop under the original average-score stopping policy.
- **Reviewed evaluation** (`reviewed_evaluation=True`): uses rubric acceptance floors for accuracy and evidence, checks *recognized* financial claims against specialist findings, passes confirmed contradictions into revision feedback, and selects the strongest acceptable draft from the evaluated candidates. The implementation records whether the final selection was accepted or reached the revision budget.

The reviewed mode deliberately uses `NumericPolicy(require_recognized_claims=False, reject_unverified=False)` during live agent runs, and only issues that policy can reject are passed to the optimizer. Figures are compared at the precision the brief writes them, so a rounded figure such as "$215.9B" for "$215,938M" is consistent, not a contradiction. **Unrecognized or unverified statements are not certified as accurate.** The parser recognizes only a subset of financial prose; confirmed contradictions among recognized comparisons block acceptance.

Two further comparisons appear in Part C. The **evidence-matched baseline** (`src/evaluation_lead_benchmarks.py`) is a single-pass brief written from the same findings the agent used; the gap between it and the agent's kept draft isolates what the revision loop and quality gate add beyond access to the evidence. The comparator in **C2** replays both acceptance policies on the same scored draft sequence; it is **not** an independent second LLM evaluation.

For enhanced evaluation:

```python
from src.graph import run_agent

report, trace, state = run_agent(
    "NVDA",
    use_memory=True,
    reviewed_evaluation=True,
)

print("Accepted:", state.get("accepted"))
print("Selected revision:", state.get("selected_iteration"))
print("Final rubric average:", state["evaluation"].average)
print("Verification:", state["verification"])
print("Report:", report)
```

`run_agent` returns `(report, trace, state)`. By default it uses the original evaluation mode to preserve the baseline workflow. The call makes external data and LLM requests; credentials, API quotas, and provider charges may apply.

## Recorded benchmark results

Results from the executed v10.4 notebook (run of October 19, 2026, commit `09782eb`). They describe one observed run; live data and LLM judgments vary between runs.

| Ticker | No-tools baseline | Same-evidence baseline | Agent first draft | Kept agent draft | Kept iteration | Accepted | Finding figures verified vs raw data |
|---|---:|---:|---:|---:|---:|---|---:|
| NVDA | 2.0 / 5 | 4.0 / 5 | 4.0 / 5 | **4.6 / 5** | 1 | Yes | 59 / 62 |
| JPM | 1.6 / 5 | 3.6 / 5 | 3.6 / 5 | **4.6 / 5** | 1 | Yes | 65 / 83 |
| XOM | 2.0 / 5 | 4.4 / 5 | 4.2 / 5 | **4.4 / 5** | 1 | Yes | 51 / 51 |

Briefs written from the agent's findings scored about 2.5 points above the no-tools baseline; the evaluate-and-revise loop added +0.5 on average over the same-evidence baseline in this run and +0.1 in a second run of the same code, so the loop's contribution is real in direction but not stable in size. Tool calls succeeded 100% of the time. The verification specialist corrected no figures; the unmatched JPM figures are bank measures it does not compute. The numeric audit found no contradictions. With memory on, the planner read eight stored lessons and cited them in four plan steps, but the memory-on brief did not score higher (4.6 vs 4.8), because the lessons request evidence the current tools cannot retrieve.

**Evaluation distinctions:** Part A compares financial sentiment classification methods and annotator agreement. Part C compares agent output quality with a no-tools baseline and a same-evidence baseline, shows verification and numeric-audit coverage, replays the two acceptance policies, and examines the effect of persistent notes.

**Validation:** The integrated code passes **188 local automated tests**, and the submitted executed notebook has **41 executed code cells with zero recorded cell errors**. Tests use mocked and controlled cases; a passing suite does not establish that all live research claims are correct.

## What changed in v10 (October 19, 2026)

- **Data verification specialist** (`src/agents/verifier.py`): a rule-based agent between the specialists and the synthesizer that checks every figure in the findings against the raw tool output and passes corrections and flags downstream. No LLM calls; results are reproducible.
- **Evidence-matched baseline** in C1, which isolates what the revision loop adds beyond the evidence.
- **Numeric audit accepts rounding**; earlier exact matching rejected correct drafts as contradictions.
- **Memory reads the newest lessons first**, the reflection step is shown which lessons are already stored so it adds only new ones, and the optimizer receives only numeric issues that can block acceptance.
- **Self-checking notebook setup:** the Drive copy is reset to the published commit, stale modules are dropped, and the import cell stops if the agent code did not load from that checkout.
- **PEP 8:** Python files and notebook code cells formatted at 79 columns.

## Data and model configuration

| Resource | Purpose |
|---|---|
| Yahoo Finance (`yfinance`) | Market prices, key metrics, financial statements |
| SEC EDGAR | Filing retrieval and section-level search over 10-K content |
| FRED | Macroeconomic indicators such as rates, inflation, GDP and unemployment |
| NewsAPI | Financial news and current headline retrieval |
| Kaggle Financial News | Labeled sentiment examples and classifier evaluation |
| Financial PhraseBank (Malo et al., 2014) | Financial sentiment evaluation grouped by annotator agreement |
| `all-MiniLM-L6-v2` | Text embeddings for retrieval |

The model defaults are defined in [`src/config.py`](src/config.py) and may be overridden through environment variables. The repository configuration specifies `gpt-6-luna` for chain and routing tasks, `claude-sonnet-5` for writing tasks, and `gpt-6-sol` for evaluation. Actual availability and billing depend on the corresponding model providers and user credentials; consult the executed notebook for the experiment's recorded configuration.

## Repository structure

```text
src/
  agents/
    planner.py, router.py, specialists.py, synthesizer.py, evaluator.py
    verifier.py              # data verification specialist (findings vs raw tool data)
    evaluation_lead.py       # rubric policy, revision budget, candidate selection
    evaluation_pipeline.py   # combined rubric and numeric review session
    numeric_validation.py    # selective financial claim extraction and checks
  chains/                    # news prompt chain and Kaggle evaluation
  memory/
    store.py                 # persistent agent notes
    evaluation_memory.py     # lesson ranking and verification support
  tools/                     # financial data, news, macro, and SEC retrieval
  graph.py                   # LangGraph workflow and run_agent(...)
  evaluation.py              # baseline comparison and evaluation metrics
  evaluation_lead_benchmarks.py   # evidence-matched baseline
  config.py, llm.py, trace.py, display.py
notebooks/
  final_project.ipynb        # consolidated course notebook (executed)
  01_data_and_nlp_starter.ipynb
  02_agent_run.ipynb
  04_evaluator_policy_comparison.ipynb
  build_final.py             # source notebook builder; applies the passes below in order
  notebook_evaluation_integration.py, notebook_publication_polish.py,
  notebook_rubric_finalization.py, notebook_v10.py
Results/
  table1_integrated_live_run.csv
EquityAgent_Final_Executed.html
memory/notes.json
scripts/compare_evaluator_policies.py
docs/CONTRIBUTIONS.md, docs/nlp_track.md
tests/                       # 188 offline tests
```

## Local setup and execution

```bash
git clone https://github.com/nicole-rowley-msaai/MS_AAI_520_NLP_GenAI_Final.git
cd MS_AAI_520_NLP_GenAI_Final
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure applicable credentials in the **local, Git-ignored** `.env` file, or use **Colab Secrets** with notebook access enabled. These include `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `FRED_API_KEY`, `NEWSAPI_KEY`, `EDGAR_USER_AGENT`, and Kaggle authentication (`KAGGLE_API_TOKEN` or the legacy username/key pair). Never commit these credentials to GitHub. The SEC user agent should identify an application and contact email.

To run the integrated notebook from Colab, open [`notebooks/final_project.ipynb`](notebooks/final_project.ipynb) from the `main` branch. The setup cell clones or resets a working copy in Drive to the published commit, prints that commit, and the import cell stops before any agent run if the code was loaded from anywhere else. Live executions use external APIs and take 30 to 60 minutes. **Do not rebuild or clear the notebook outputs when preparing the submitted HTML:** `notebooks/build_final.py` produces an unexecuted source notebook and is not a substitute for the preserved executed document.

Local unit tests:

```bash
python -m pytest -q
```

## Limitations and reproducibility

- Financial information may be cached, delayed, incomplete, or differ across data-provider and filing dates.
- Evaluator scores come from one LLM call per draft and vary between executions; differences of a few tenths between single runs are within that noise. A rubric score is not a guarantee of investment accuracy.
- The verification specialist checks figures against raw tool data, not published filings, computes only common derived values, and does not check news findings.
- Numeric validation covers a limited subset of automatically recognized metrics and periods. Unverified statements require further review.
- The evaluator policy comparison is a **paired decision replay**, not a claim that two independent evaluator models produced comparable judgments on identical reports.
- Memory changes the plan; it has not been shown to improve the brief, because the lessons request evidence the current tools cannot retrieve.
- The executed HTML preserves the reported run. Re-execution produces different external data, scores, and selected drafts.

## Team and attribution

| Contributor | Primary responsibilities |
|---|---|
| **Nicole Rowley** | Research agent orchestration, planning, financial tool integrations, specialist routing, synthesis, baseline workflow, v10 revisions and data verification specialist |
| **Ashley Figueroa** | Financial NLP, prompt chaining, sentiment classifier evaluation, few-shot improvements and diagnostics |
| **Pawan Bishwokarma** | Evaluator quality policy, numeric validation, revision selection, evaluation tests, controlled comparisons, notebook integration |

For further details see [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md). Generative AI tools (ChatGPT and Claude Code) assisted development; the full disclosure appears at the end of the submitted notebook, consistent with the course's academic integrity requirements.
