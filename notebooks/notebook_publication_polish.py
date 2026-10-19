"""Publication-facing editorial pass for the AAI 520 integrated notebook.

This modifies report prose and presentation cells only, retaining the complete
experimental code paths from the upstream notebook assembler and evaluator
integration helper. It does not fabricate or cache execution outputs.
"""

from __future__ import annotations

import hashlib


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def change(cell: dict, value: str) -> None:
    cell["source"] = value


def md(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": value}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": value,
        "outputs": [],
        "execution_count": None,
    }


def index(cells: list[dict], prefix: str) -> int:
    matches = [
        i for i, cell in enumerate(cells) if source(cell).startswith(prefix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one notebook cell starting {prefix!r}, got {matches}"
        )
    return matches[0]


TITLE = """# EquityAgent: Evidence-Grounded Agentic Equity Research

**AAI 520: Natural Language Processing and Generative AI | Final project**
University of San Diego · Master of Science in Applied Artificial \
Intelligence · October 2026

**Research team:** Nicole Rowley (agent orchestration), Ashley Figueroa \
(NLP), and Pawan Bishwokarma (evaluation).
**Code and reproducibility:** [EquityAgent GitHub \
repository](https://github.com/nicole-rowley-msaai/MS_AAI_520_NLP_GenAI_Final)

### Abstract

EquityAgent combines financial information retrieval, NLP, and an iterative
language-model research workflow. A LangGraph agent plans research, delegates
tasks to specialists, gathers market, macroeconomic, financial filing, and news
information, and synthesizes an equity brief. The news component benchmarks
financial sentiment classification against keyword and supervised baselines.
The agent component compares evidence-grounded generation with a no-tools
baseline. Finally, an enhanced evaluation layer adds acceptance thresholds
for accuracy and evidence, selective numerical cross-checking, structured
revision feedback, and selection of the strongest qualifying draft. The
experiments cover NVDA, JPM, and XOM and report classifier metrics, research
quality, tool reliability, memory behavior, and evaluation decisions. All
results below are produced by the executable analysis rather than inserted
from a previous run.

**Notebook organization:** Part A examines data and sentiment modeling; Part B
traces the complete agent workflow; Part C evaluates outputs, baseline
comparisons, and the effect of quality gates.
"""

RESEARCH_DESIGN = """## Research questions and experimental design

1. **Financial NLP:** How does an LLM sentiment classifier compare with keyword
   classification and TF–IDF logistic regression on a held-out set of labeled
   financial headlines?
2. **Research agent:** How do an evidence-informed research process and its
   revision loop compare with generation from a single prompt without tools?
3. **Evaluation controls:** Do explicit evidence and accuracy thresholds, plus
   constrained numerical cross-checks, change which generated drafts are
   considered acceptable?

| Component | Data or comparison | Primary evidence |
|---|---|---|
| Sentiment NLP | Financial PhraseBank headlines; held-out test set | \
Accuracy, macro-F1, confusion matrix, agreement-band analysis |
| Agent research | NVDA, JPM, XOM; market, macro, filing, and news findings | \
Tool trace, research brief, five-criterion LLM rubric |
| Quality controls | Original versus reviewed policies applied to the same \
scored revisions | Threshold decisions, recognized numeric claims, selected \
revision |
| Memory | Repeated runs with and without stored lessons | Planning steps, \
notes retrieved, rubric scores |

The original and reviewed policies reuse the **same evaluator model**; the
comparison isolates their acceptance rules, not two separately trained judges.
Agent calls and external data sources are nondeterministic. Results reflect
the data and model responses available at execution time.
"""

RUN_METADATA = """# Record the effective configuration used for this execution.
from datetime import datetime, timezone
from src.config import MODELS

print("Execution date (UTC):", \
datetime.now(timezone.utc).isoformat(timespec="seconds"))
print("Configured models:", {key: MODELS[key] for key in
                             ("chain", "writer", "evaluator", "embeddings")})
"""

PART_A = """# Part A. Financial data and sentiment modeling

## Agent Functions and Capabilities

This section verifies the agent's financial data tools, explores the financial
headline dataset, and benchmarks sentiment classification using keyword rules,
TF–IDF logistic regression, and zero-shot or few-shot LLM prompting. The
classifier study uses a fixed, stratified held-out sample of 300 headlines,
three LLM trials per prompting condition, and a separate agreement-band
analysis. Tool calls expose success and failure states for inspection rather
than assuming every external source is available.
"""

A_INTERPRETATION = """## A5. Interpretation of sentiment results

The computed comparison table above reports both the mean of repeated LLM
classifications and their majority-vote results. The next cell summarizes the
measured differences without assuming the same values will recur on another
run. Confusion matrices and annotator-agreement bands provide necessary
context: accuracy alone can hide class imbalance, and disagreement among
human annotators makes a headline classification intrinsically ambiguous.

The Financial PhraseBank is public and may overlap with model pretraining
data. Therefore these measurements describe performance on this particular
benchmark, **not verified generalization to unseen current news**. The
separate October 14 and October 18 exploratory runs used different processing
conditions and are not treated as a controlled causal comparison here.
"""

A_MEASURED_SUMMARY = """# Directly summarize this notebook's current \
classification results.
metric_cols = ["classifier", "accuracy (mean)", "macro_f1 (mean)"]
display(comparison[metric_cols].sort_values("macro_f1 (mean)", \
ascending=False))

best = comparison.loc[comparison["macro_f1 (mean)"].idxmax()]
print("Highest observed macro-F1:", best["classifier"],
      "| score:", round(float(best["macro_f1 (mean)"]), 3))
print("Accuracy by agreement level is reported in the preceding table and \
plot.")
"""

PART_B = """# Part B. Autonomous research workflow

## Agent Design and Workflows

EquityAgent is implemented as a LangGraph state graph (`src/graph.py`). Each
research request passes through planning, specialist routing, data gathering,
synthesis, iterative evaluation and revision, reflection, and memory storage.
Specialists run independently and can be executed concurrently.

```text
Ticker → Plan → Route → Specialists → Synthesize → Evaluate
                                                 ↓       ↑
                                              Refine → Optimize
                                                 |
                                            Final selection
                                                 ↓
                                          Reflect → Memory
```

| Stage | Implementation | Evidence shown |
|---|---|---|
| Planning and routing | `src/agents/planner.py`, `router.py` | Plan and \
routing decisions |
| Retrieval and specialists | `src/agents/specialists.py`, `src/tools/` | \
Tool trace and retrieved findings |
| News NLP | `src/chains/news_chain.py` | Chained classification and news \
summary |
| Report generation | `src/agents/synthesizer.py` | Research brief |
| Evaluation and revision | `src/agents/evaluator.py`, \
`evaluation_pipeline.py` | Rubric scores, critiques, numeric audit |
| Reflection and memory | `src/memory/store.py` | Lessons written and reused |

The first demonstration uses NVDA and begins with empty **experiment-local**
agent memory. A subsequent run tests whether previously stored lessons alter
the planning process. Clearing the notes file is intentional for this
controlled local experiment and does not change files in the source repository.
"""

B7_NOTE = """## B7. Evidence-based evaluation and revision

The original LLM evaluator scores five criteria from 1 to 5:
completeness, evidence, accuracy, balance, and clarity. The research agent
then revises its initial draft using specific critiques. The enhanced
Evaluation Lead layer adds deterministic acceptance rules to the same
qualitative scores.

| Acceptance rule | Original policy | Enhanced policy |
|---|---|---|
| Minimum average | 4.0 | 4.0 |
| Minimum accuracy | No separate threshold | 4 / 5 |
| Minimum evidence | No separate threshold | 4 / 5 |
| Minimum revision rounds | 1 | 1 |
| Maximum evaluated drafts | 3 | 3 |
| Recognized numeric contradictions | No separate gate | Prevent acceptance |
| Preferred final draft | Last evaluated revision | Highest-scoring accepted \
revision |

In addition, the enhanced policy records numeric claims it can parse and
compares those against corresponding specialist findings. Unrecognized claims
and unresolved evidence remain **unverified**, not correct by assumption.
Parser coverage is incomplete and any discrepancies require human review.

The tables below show the evaluator's decisions for the drafts **actually
generated in this execution**. An exhausted revision budget is not equivalent
to acceptance.
"""

B7_AUDIT = """# Summarize live rubric decisions and numeric audit coverage.
review_rows = []
for item in state["history"]:
    scores = item["scores"]
    review_rows.append({
        "iteration": item["iteration"],
        "average": item["average"],
        "accuracy": scores["accuracy"],
        "evidence": scores["evidence"],
        "passed": item.get("accepted", False),
        "selected": item.get("selected", False),
        "reason": ", ".join(item.get("failures", [])) or "none",
        "numeric_checked": item.get("numeric_checked", 0),
        "numeric_unverified": item.get("numeric_unverified", 0),
        "numeric_contradicted": item.get("numeric_contradicted", 0),
    })
review_df = pd.DataFrame(review_rows)
display(review_df[["iteration", "average", "accuracy", "evidence",
                   "passed", "selected", "reason"]])
display(review_df[["iteration", "numeric_checked", "numeric_unverified",
                   "numeric_contradicted"]])

print("Final decision:", "accepted" if state["accepted"] else "not accepted")
print("Selected revision:", state["selected_iteration"])
print("Selected rubric average:", state["evaluation"].average)
print("Numeric claims checked:", len(state["numeric_audit"].checks))

numeric_rows = state["numeric_audit"].rows()
if numeric_rows:
    display(pd.DataFrame(numeric_rows))
else:
    print("No claims matched the constrained numeric validator.")
"""

B8_REPORT = """if not state["accepted"]:
    print("Quality criteria were not met; the following report is retained \
for diagnostic review.")
display(show.report_markdown(report))
"""

PART_C = """# Part C. Evaluation and comparative analysis

## Evaluation and Iteration

The following experiments assess different aspects of the system, with
limitations appropriate to each comparison. The no-tools baseline examines
the value of access to research findings and the overall agent workflow; it
does **not isolate the revision loop**. The memory comparison uses separate
live agent runs and therefore cannot isolate the causal effect of memory from
model and data variation. The policy replay applies original and enhanced
acceptance criteria to the **same sequence of observed rubric scores**.

An LLM rubric supplies qualitative scores throughout. Numeric spot-checks
are constrained by the parser and the formats of retrieved specialist data.
"""

C1 = """## C1. Report quality, numerical evidence, and memory

Three descriptive analyses are reported:

1. **No-tools versus research agent.** For each ticker, the baseline
   produces a brief without the specialist findings. Both briefs are graded
   against the same collected findings. This measures the combined effect of
   research evidence and agent workflow, not one component in isolation.
2. **Unsupported figure counts.** A string-based check flags figures not
   repeated verbatim in findings. Rounding, scaling, and unit formatting can
   produce false positives; this measure is a review aid, not fact checking.
3. **Memory on versus off.** Plans and rubric scores are compared across
   separate runs. Variation from LLM outputs and data availability remains
   a confounder.

The complete ticker comparison is also saved as a CSV to facilitate the
accompanying written report. The computed table, rather than any earlier
exploratory scores, is the source for the reported results.
"""

C2 = """## C2. Paired analysis of evaluator acceptance rules

This is a **decision-policy ablation**: the original and enhanced acceptance
rules are evaluated against the same observed sequence of drafts and LLM
rubric scores from the NVDA execution above. Both share the same evaluator
model. The comparison does not represent two independent evaluations or a
separate run of the original agent.

The original rule requires the minimum number of revisions and a rubric
average of at least 4.0. The enhanced rule additionally requires accuracy
and evidence scores of at least 4, rejects contradictions for numeric claims
recognized by its parser, and can preserve the best accepted draft across
revisions. A candidate can thus pass on its average score while failing a
more specific quality criterion.

A difference in acceptance logic is an architectural result. A claimed
improvement in final brief quality would require paired generation experiments
and independent verification of financial claims, neither of which is
established by this policy replay alone.
"""

C3 = """## C3. Limitations and threats to validity

- **Data provenance and currency.** Market prices, news, filings, and macro
  series come from external APIs or caches and may vary by run date. Published
  financial statements and the agent's retrieved excerpts should be
  cross-checked before investment use.
- **Judgment variability.** Rubric scores come from an LLM and can change
  between runs. Repeated evaluations and human ratings would be needed to
  estimate scoring reliability.
- **Numeric coverage.** Only a subset of explicitly formatted financial
  claims is recognized. A zero-contradiction audit is not a claim that all
  figures are correct. Missing, ambiguous, or conflicting source evidence
  remains unresolved.
- **Comparison scope.** The no-tools baseline differs in available evidence;
  the policy ablation shares observed scores; the memory comparison involves
  separate LLM runs. These answer different questions and should not be
  combined into one causal conclusion.
- **Benchmark leakage.** A publicly available sentiment dataset may have
  appeared in model pretraining. PhraseBank accuracy does not establish
  robustness across current market news, languages, or evolving companies.
- **Application scope.** Generated briefs are research aids, not verified
  investment recommendations. Human source review remains necessary.
"""

SUMMARY = """## Discussion and conclusion

EquityAgent demonstrates an end-to-end architecture connecting financial data
retrieval, sentiment NLP, specialist research, structured synthesis, iterative
assessment, and memory. Part A evaluates language understanding against
baseline classifiers. Part B exposes each computational step and the evidence
supporting the final report. Part C distinguishes descriptive model scores,
source-grounding checks, and evaluator policy decisions.

The Evaluation Lead contribution is a **decision and quality-control
extension**, not a new LLM judge. It introduces accuracy and evidence floors,
selective financial claim checks, actionable revision feedback, and best
draft retention. Whether those controls improve the correctness of finished
briefs across many companies is an open empirical question that would require
broader financial claim annotation and external validation.

### Reproducibility and implementation

Source code, datasets, tests, and notebook construction are maintained in the
[project \
repository](https://github.com/nicole-rowley-msaai/MS_AAI_520_NLP_GenAI_Final).
The setup cell records the checked-out commit. Model results, table values,
and figures are calculated by the code cells during execution, with API
responses and retrieval dates affecting reproducibility. Secrets are loaded
from Colab's credential store and are not embedded in the notebook.

**Development disclosure:** AI coding assistance was used in developing the
Evaluation Lead extension and synthetic tests; the implementation was
reviewed and exercised with unit and integration tests and a live NVDA run.
Those validation activities do not substitute for executing the complete
notebook and independently checking its financial conclusions.

### References

- Malo et al. (2014), *Good Debt or Bad Debt: Detecting Semantic Orientations
  in Economic Texts* (Financial PhraseBank).
- Brown et al. (2020), *Language Models are Few-Shot Learners*.
- Wang et al. (2023), *Self-Consistency Improves Chain of Thought Reasoning
  in Language Models*.
- Jurafsky and Martin, *Speech and Language Processing*, third-edition draft.
"""


def polish_final_notebook(notebook: dict) -> dict:
    """Convert an integrated development notebook to a results-first report."""
    cells = notebook["cells"]
    if any("AAI520_PUBLICATION_POLISH_V1" in source(c) for c in cells):
        return notebook

    change(cells[index(cells, "# EquityAgent:")], TITLE)
    change(
        cells[index(cells, "<!-- AAI520_EVALUATION_INTEGRATION_V1 -->")],
        "<!-- AAI520_PUBLICATION_POLISH_V1 -->\n" + RESEARCH_DESIGN,
    )
    change(cells[index(cells, "# Part A.")], PART_A)
    cells.insert(index(cells, "# Part A."), code(RUN_METADATA))
    change(
        cells[index(cells, "**Reading the results (run of October")],
        A_INTERPRETATION,
    )
    a_idx = index(cells, "## A5. Interpretation")
    cells.insert(a_idx + 1, code(A_MEASURED_SUMMARY))

    change(cells[index(cells, "# Part B.")], PART_B)
    change(cells[index(cells, "## B7.")], B7_NOTE)
    change(cells[index(cells, "# Live quality-gate audit")], B7_AUDIT)
    change(
        cells[index(cells, "display(show.report_markdown(report))")], B8_REPORT
    )

    change(cells[index(cells, "# Part C.")], PART_C)
    change(cells[index(cells, "## C1.")], C1)
    # Preserve the full CSV while presenting narrower comparison tables.
    comparison_cell = cells[index(cells, "rows = []\nfor t in TICKERS:")]
    comparison_source = source(comparison_cell)
    if not comparison_source.endswith("results"):
        raise ValueError(
            "Unexpected ticker-comparison code; refusing to patch"
        )
    change(
        comparison_cell,
        comparison_source[: -len("results")]
        + 'display(results[["ticker", "baseline_avg", "agent_first_draft",\n'
        '                 "agent_final", "accepted", "selected_iteration"]])\n'
        'display(results[["ticker", "numeric_checked", '
        '"numeric_contradicted",\n'
        '                 "numeric_unverified", "unsupported_baseline",\n'
        '                 "unsupported_agent", "tool_success"]])\n',
    )
    export_cell = cells[index(cells, "from pathlib import Path\ntable1_path")]
    change(
        export_cell,
        source(export_cell).replace(
            'print("These are fresh integrated-run measurements; copy these '
            'values to the report.")\n',
            "",
        ),
    )
    change(cells[index(cells, "## C2.")], C2)
    change(cells[index(cells, "## C3.")], C3)
    change(cells[index(cells, "## Summary and reproducibility")], SUMMARY)

    for prefix in (
        "### Policy difference beyond this one observed trajectory",
        "# Deliberately hypothetical case",
        "### Previously observed integration checkpoint",
    ):
        cells.pop(index(cells, prefix))

    setup = cells[index(cells, "# --- Colab setup")]
    change(
        setup,
        source(setup).replace(
            "# To test a pull request, set BRANCH to its branch and REPO to a "
            "separate\n"
            "# Drive folder, so the team's working copy is not touched.",
            "# Use an isolated copy because the memory experiment resets "
            "local notes.",
        ),
    )

    for position, cell in enumerate(cells):
        if not cell.get("id"):
            cell["id"] = hashlib.sha256(
                f"{position}|{cell['cell_type']}|{source(cell)}".encode()
            ).hexdigest()[:12]
    return notebook
