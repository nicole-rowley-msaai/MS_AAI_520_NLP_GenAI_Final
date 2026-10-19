# EquityAgent

**Evidence-Grounded Agentic Equity Research**

**AAI 520: Natural Language Processing and Generative AI | Final Team Project**

EquityAgent is a LangGraph-based investment research system that transforms a stock symbol into an evidence-grounded research brief. The agent develops a research plan, invokes financial data and retrieval tools, routes work to specialists, synthesizes findings, evaluates and revises drafts, reflects on its process, and stores lessons for later runs. The integrated system also offers additional rubric acceptance checks, selective numerical validation, and retention of the strongest qualifying draft.

**Research team:** Nicole Rowley (agent orchestration and research tools), Ashley Figueroa (financial NLP), and Pawan Bishwokarma (evaluation and integration).

> Academic research demonstration. Generated briefs are analytical outputs, not investment advice. Tool data, LLM judgments, and numeric coverage have limitations.

## Final deliverables

| File | Purpose |
|---|---|
| [`EquityAgent_Final_Executed.html`](EquityAgent_Final_Executed.html) | **Executed, readable notebook document for the course submission**, including observed outputs, tables, figures, and interpretation |
| [`notebooks/final_project.ipynb`](notebooks/final_project.ipynb) | Executable Jupyter/Colab notebook containing Parts A, B, and C |
| [`Results/table1_integrated_live_run.csv`](Results/table1_integrated_live_run.csv) | Integrated research benchmark results from the documented run |
| [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md) | Team responsibilities and implementation contributions |
| [`docs/nlp_track.md`](docs/nlp_track.md) | NLP design, diagnostics, and experiments |
| [`notebooks/04_evaluator_policy_comparison.ipynb`](notebooks/04_evaluator_policy_comparison.ipynb) | Supporting evaluator decision-policy comparison |

[GitHub repository](https://github.com/nicole-rowley-msaai/MS_AAI_520_NLP_GenAI_Final) · [Final executed HTML](EquityAgent_Final_Executed.html)

## Assignment requirements and evidence

The consolidated notebook organizes data and NLP experiments in **Part A**, the autonomous research demonstration in **Part B**, and comparative evaluation in **Part C**.

| Required capability | Implementation and notebook evidence |
|---|---|
| **Research planning** | `src/agents/planner.py` creates a structured plan; **B2** displays its steps and hypotheses |
| **Dynamic tool use** | `src/tools/` provides market, macroeconomic, news, and SEC filing retrieval; **B3** shows recorded tool calls |
| **Self-reflection** | `src/agents/evaluator.py` produces a reflective assessment after draft selection; **B9** records conclusions and limitations |
| **Learns across runs** | `src/memory/store.py` persists lessons in `memory/notes.json`; **B10** displays stored notes and **C1** compares runs with and without memory |
| **Prompt Chaining: Ingest → Preprocess → Classify → Extract → Summarize** | `src/chains/news_chain.py`; **A3–A4** evaluate classification and **B5** shows the intermediate chain outputs |
| **Routing** | `src/agents/router.py` directs research steps to earnings, news, or market specialists; **B4** displays choices and rationales |
| **Evaluator–Optimizer** | `src/agents/evaluator.py`, `src/agents/evaluation_pipeline.py`, `src/graph.py`; **B7** shows scoring, audit, feedback and revisions, while **C2** compares acceptance decisions |

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
Synthesizer --> Evaluator + optional numerical audit --> Quality gate
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

The LangGraph path is **plan → route → specialists → synthesize → evaluate ⇄ optimize → reflect**. Specialist tasks are dispatched in parallel where appropriate. The trace records plans, routed tasks, tool results, evaluation critiques, revision decisions, and memory events.

## Evaluation design

The project retains two modes for reproducible methodological comparison:

- **Baseline evaluation** (default): LLM rubric scores guide an iterative evaluator–optimizer loop under the original average-score stopping policy.
- **Reviewed evaluation** (`reviewed_evaluation=True`): uses rubric acceptance floors for accuracy and evidence, checks *recognized* financial claims against specialist findings, passes confirmed contradictions into revision feedback, and selects the strongest acceptable draft from the evaluated candidates. The implementation records whether the final selection was accepted or reached the revision budget.

The reviewed mode deliberately uses `NumericPolicy(require_recognized_claims=False, reject_unverified=False)` during live agent runs. **Unrecognized or unverified statements are not certified as accurate.** The parser only recognizes a subset of financial prose; confirmed contradictions among recognized comparisons can block acceptance. A higher rubric average does not necessarily mean a better accepted draft if the numeric gate detects a mismatch. The comparator in **C2** replays policies on the same scored draft sequence; it is **not** an independent second LLM evaluation.

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
print("Report:", report)
```

`run_agent` returns `(report, trace, state)`. By default it uses the original evaluation mode to preserve the baseline workflow. The call makes external data and LLM requests; credentials, API quotas, and provider charges may apply.

## Recorded benchmark results

The **completed notebook execution** reported the following results. These figures describe one observed run and should not be treated as deterministic future scores.

| Ticker | Single-prompt baseline | Agent first draft | Selected agent draft | Selected iteration |
|---|---:|---:|---:|---:|
| NVDA | 1.8 / 5 | 4.2 / 5 | **4.4 / 5** | 1 |
| JPM | 2.0 / 5 | 4.4 / 5 | **4.4 / 5** | 1 |
| XOM | 1.6 / 5 | 4.4 / 5 | **4.4 / 5** | 0 |

All three recorded agent outcomes were accepted by the configured evaluation process. The selected XOM draft was **iteration 0**, demonstrating retention of an earlier candidate when a later revision is not preferred. Tool call success was recorded as **100% in these runs**; that is not a general service-level guarantee. The numerical audit recorded 4 claim-check entries for NVDA (including 2 unverified), 1 for JPM, and 0 for XOM. These results therefore do **not** independently verify all report figures.

**Evaluation distinctions:** Part A compares financial sentiment classification methods and annotator agreement. Part C compares agent output quality with a single-prompt baseline, shows evaluation iterations and numerical audit coverage, and examines the impact of persistent notes. The memory-off and memory-on results demonstrate changed planning, not a guaranteed improvement in report score.

**Validation:** The integrated code passed **155 local automated tests**, and the submitted executed notebook preserved **39 executed code cells with zero recorded cell errors**. Tests include mocked and controlled cases; a passing suite does not establish that all live research claims are correct.

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
    evaluation_lead.py       # rubric policy, revision budget, candidate selection
    evaluation_pipeline.py   # combined rubric and numeric review session
    numeric_validation.py    # selective financial claim extraction and checks
  chains/                    # news prompt chain and Kaggle evaluation
  memory/
    store.py                  # persistent agent notes
    evaluation_memory.py     # lesson verification and prioritization support
  tools/                     # financial data, news, macro, and SEC retrieval
  graph.py                   # LangGraph workflow and run_agent(...)
  evaluation.py              # baseline comparison and evaluation metrics
  evaluation_lead_benchmarks.py
  config.py, llm.py, trace.py, display.py
notebooks/
  final_project.ipynb        # consolidated course notebook
  01_data_and_nlp_starter.ipynb
  02_agent_run.ipynb
  04_evaluator_policy_comparison.ipynb
  build_final.py             # source notebook builder
Results/
  table1_integrated_live_run.csv
EquityAgent_Final_Executed.html
memory/notes.json
scripts/compare_evaluator_policies.py
docs/CONTRIBUTIONS.md
tests/
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

To run the integrated notebook from Colab, open [`notebooks/final_project.ipynb`](notebooks/final_project.ipynb). Live executions use external APIs and can take time. **Do not rebuild or clear the notebook outputs when preparing the submitted HTML:** `notebooks/build_final.py` rebuilds source notebook cells and is not a substitute for the preserved executed document.

Local unit tests:

```bash
python -m pytest -q
```

## Limitations and reproducibility

- Financial information may be cached, delayed, incomplete, or differ across data-provider and filing dates.
- Evaluator scores depend on model judgments and can vary between executions. A rubric score is not a guarantee of investment accuracy.
- Numeric validation covers a limited subset of automatically recognized metrics and periods. Unverified statements require further review.
- The evaluator policy comparison is a **paired decision replay**, not a claim that two independent evaluator models produced comparable judgments on identical reports.
- Memory can affect the plan and selected sources; observed changes do not establish improved investment decisions.
- The executed HTML preserves the reported run. Re-execution may produce different external data, scores, and selected drafts.

## Team and attribution

| Contributor | Primary responsibilities |
|---|---|
| **Nicole Rowley** | Research agent orchestration, planning, financial tool integrations, specialist routing, synthesis, baseline workflow |
| **Ashley Figueroa** | Financial NLP, prompt chaining, sentiment classifier evaluation, few-shot improvements and diagnostics |
| **Pawan Bishwokarma** | Evaluator quality policy, numeric validation, revision selection, evaluation tests, controlled comparisons, notebook integration |

For further details see [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md). A brief statement about generative AI assistance with development appears near the end of the submitted notebook, consistent with the course's academic integrity requirements.
