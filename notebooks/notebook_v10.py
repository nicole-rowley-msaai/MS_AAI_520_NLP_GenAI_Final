"""v10 pass for the EquityAgent final notebook.

Runs last in build_final.py. It adds an evidence-matched baseline to C1:
a single-pass brief written from the *same findings* the agent used,
graded by the same evaluator against the same findings
(``matched_baseline_brief`` from the Evaluation Lead's
``src/evaluation_lead_benchmarks.py``). The existing no-tools baseline
measures the value of evidence plus workflow; the matched baseline
isolates what planning, revision and the quality gate add on top of the
evidence.

It adds section B6b for the data verification specialist
(``src/agents/verifier.py``), puts it in the Part B diagram and table,
reports its counts per ticker in C1, and notes its scope under
limitations.

It also points the setup cell at ``main``, so a fresh Colab run executes
the revised code rather than the old integration branch.

Calling it twice leaves the notebook unchanged.
"""

from __future__ import annotations

import hashlib

MARKER = "matched_baseline_brief"


def _text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _code(content: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": content,
            "outputs": [], "execution_count": None,
            "id": hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]}


def _markdown(content: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": content,
            "id": hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]}


def _locate(cells: list[dict], needle: str) -> int:
    for i, cell in enumerate(cells):
        if needle in _text(cell):
            return i
    raise ValueError(f"v10 pass: expected text not found: {needle!r}")


def _replace(cells: list[dict], needle: str, old: str, new: str) -> None:
    cell = cells[_locate(cells, needle)]
    text = _text(cell)
    if old not in text:
        raise ValueError(f"v10 pass: {old!r} not found in cell {needle!r}")
    cell["source"] = text.replace(old, new, 1)


C1_CELL = "rep, tr, st = run_agent(t, reviewed_evaluation=True)"
PLOT_CELL = 'ax.set_title("Rubric comparison: no-tools baseline'

MATCHED_SUMMARY = r'''# Evidence-matched comparison, computed from the table above.
from IPython.display import Markdown

paired = results[["ticker", "matched_baseline_avg", "agent_final",
                  "unsupported_matched", "unsupported_agent"]].copy()
paired["agent_minus_matched"] = (
    paired["agent_final"] - paired["matched_baseline_avg"]
).round(1)
display(paired)

gaps = paired["agent_minus_matched"]
better = int((gaps > 0).sum())
display(Markdown(
    "**Evidence-matched baseline.** With the same findings, a single-pass "
    "brief scored " + ", ".join(f"{v:.1f}" for v in paired["matched_baseline_avg"])
    + " against the agent's " + ", ".join(f"{v:.1f}" for v in paired["agent_final"])
    + f" ({', '.join(paired['ticker'])}). The agent scored higher on "
    f"{better} of {len(paired)} tickers, by {gaps.mean():+.1f} points on "
    "average. This difference, not the gap to the no-tools baseline, is "
    "what planning, revision and the quality gate add beyond access to the "
    "evidence. One run per ticker and an LLM judge make it indicative, not "
    "statistically established."
))
'''

OLD_DIAGRAM = """Ticker → Plan → Route → Specialists → Synthesize → Evaluate
                                                 ↓       ↑
                                              Refine → Optimize
                                                 |
                                            Final selection
                                                 ↓
                                          Reflect → Memory"""

NEW_DIAGRAM = """Ticker → Plan → Route → Specialists → Verify → Synthesize → Evaluate ⇄ Optimize
                                                                 |
                                                          Final selection
                                                                 ↓
                                                         Reflect → Memory"""

B6B_TEXT = """## B6b. Data verification specialist

Before any brief is written, a separate verification agent
(`src/agents/verifier.py`) re-reads the raw tool output the specialists
received and checks every financial figure in their findings. A figure is
**confirmed** if it equals a raw value, or one computed from raw values
(year-over-year change, margin, ratio as a percent, macro change), at the
precision written; **contradicted** if the finding names a metric and fiscal
year whose raw value differs; and **not found** otherwise, which flags it
for attribution rather than proving it wrong.

The agent is rule-based, so its checks are reproducible and independent of
the models that wrote the findings. Its corrections and flags travel with
the findings to the synthesizer and the evaluator. News findings summarize
article text, which the verifier does not hold, so they are not checked."""

B6B_CODE = r"""# Data verification specialist: each finding checked against raw tool data.
verification = pd.DataFrame(state["verification"])
display(verification[["route", "status", "checked", "confirmed",
                      "not_found", "contradicted"]])
for row in state["verification"]:
    if row["corrections"]:
        print(f"{row['route']} corrections: {row['corrections']}")
    if row["not_found_figures"]:
        print(f"{row['route']} not found in raw data: "
              f"{row['not_found_figures']}")
    if row["note"]:
        print(f"{row['route']}: {row['note']}")
"""


def finalize_v10(notebook: dict) -> dict:
    cells = notebook["cells"]
    if any(MARKER in _text(c) for c in cells):
        return notebook

    # Run the revised code: check out main into a fresh Drive folder.
    _replace(cells, "# --- Colab setup", 'BRANCH = "integration_pawan_ashley"',
             'BRANCH = "main"')
    _replace(cells, "# --- Colab setup",
             '"/content/drive/MyDrive/equity-agent-pawan-integration-review"',
             '"/content/drive/MyDrive/equity-agent-v10"')

    # C1 description and limitations.
    _replace(cells, "## C1. Report quality", "Three descriptive analyses",
             "Four descriptive analyses")
    _replace(cells, "## C1. Report quality", "\n2. **Unsupported figure counts.**",
             "\n2. **Evidence-matched baseline versus research agent.** A "
             "single-pass brief\n   is written from the *same findings* the "
             "agent used, with no planning,\n   revision or quality gate, and "
             "graded the same way. The gap to this\n   baseline isolates what "
             "the agent workflow adds beyond the evidence.\n"
             "3. **Unsupported figure counts.**")
    _replace(cells, "## C1. Report quality", "\n3. **Memory on versus off.**",
             "\n4. **Memory on versus off.**")
    _replace(cells, "## C3. Limitations",
             "The no-tools baseline differs in available evidence;",
             "The no-tools baseline differs in available evidence, which the "
             "evidence-matched\n  baseline removes;")

    # C1 computation: one matched baseline per ticker.
    _replace(cells, C1_CELL, "rows = []\n",
             "from src.evaluation_lead_benchmarks import matched_baseline_brief"
             "\n\nrows = []\n")
    _replace(cells, C1_CELL, '    findings = st["findings"]\n',
             '    findings = st["findings"]\n'
             "    # Same findings, one pass, no revision or gate.\n"
             "    matched = matched_baseline_brief(t, findings)\n"
             "    matched_score = score_against_findings(matched, findings)\n")
    _replace(cells, C1_CELL, '            "baseline_avg": base_score["average"],\n',
             '            "baseline_avg": base_score["average"],\n'
             '            "matched_baseline_avg": matched_score["average"],\n')
    _replace(cells, C1_CELL,
             '            "unsupported_baseline": len(unsupported_figures(base, findings)),\n',
             '            "unsupported_baseline": len(unsupported_figures(base, findings)),\n'
             '            "unsupported_matched": len(\n'
             "                unsupported_figures(matched, findings)\n"
             "            ),\n")
    _replace(cells, C1_CELL, 'display(results[["ticker", "baseline_avg", ',
             'display(results[["ticker", "baseline_avg", "matched_baseline_avg", ')
    _replace(cells, C1_CELL, '"numeric_unverified", "unsupported_baseline",',
             '"numeric_unverified", "unsupported_baseline",\n'
             '                 "unsupported_matched",')

    # Plot: four bars per ticker, then the computed summary after it.
    plot = _locate(cells, PLOT_CELL)
    _replace(cells, PLOT_CELL, "w = 0.27\n", "w = 0.2\n")
    _replace(cells, PLOT_CELL,
             '    ("baseline_avg", "baseline", "#b5b5b5", -w),\n'
             '    ("agent_first_draft", "agent, first draft", "#8fb3e0", 0),\n'
             '    ("agent_final", "agent, final", "#3b6fb6", w),\n',
             '    ("baseline_avg", "no-tools baseline", "#d0d0d0", -1.5 * w),\n'
             '    ("matched_baseline_avg", "same-evidence baseline", "#9a9a9a",'
             " -0.5 * w),\n"
             '    ("agent_first_draft", "agent, first draft", "#8fb3e0",'
             " 0.5 * w),\n"
             '    ("agent_final", "agent, final", "#3b6fb6", 1.5 * w),\n')
    _replace(cells, PLOT_CELL, '"Rubric comparison: no-tools baseline and integrated agent"',
             '"Rubric comparison: baselines and integrated agent"')

    # Data verification specialist between the specialists and synthesis.
    part_b = "# Part B. Autonomous research workflow"
    _replace(cells, part_b, OLD_DIAGRAM, NEW_DIAGRAM)
    _replace(cells, part_b, "| Report generation | `src/agents/synthesizer.py` |",
             "| Data verification | `src/agents/verifier.py` | Figures "
             "confirmed, corrected or flagged against raw tool data |\n"
             "| Report generation | `src/agents/synthesizer.py` |")
    b6 = _locate(cells, "show.findings_table(trace)")
    cells[b6 + 1:b6 + 1] = [_markdown(B6B_TEXT), _code(B6B_CODE)]
    _replace(cells, C1_CELL, '            "tool_success": tool_success_rate(tr),\n',
             '            "tool_success": tool_success_rate(tr),\n'
             '            "figures_checked": sum(\n'
             '                r["checked"] for r in st["verification"]\n'
             "            ),\n"
             '            "figures_confirmed": sum(\n'
             '                r["confirmed"] for r in st["verification"]\n'
             "            ),\n"
             '            "figures_corrected": sum(\n'
             '                r["contradicted"] for r in st["verification"]\n'
             "            ),\n")
    _replace(cells, C1_CELL, '"unsupported_agent", "tool_success"]])',
             '"unsupported_agent", "tool_success"]])\n'
             'display(results[["ticker", "figures_checked", '
             '"figures_confirmed",\n'
             '                 "figures_corrected"]])')
    _replace(cells, "## C3. Limitations", "- **Comparison scope.**",
             "- **Verification scope.** The verification specialist checks "
             "figures against\n  the raw tool data, not the companies' "
             "published filings. A figure it\n  cannot find may still be "
             "correct, and news figures are not checked.\n"
             "- **Comparison scope.**")

    cells.insert(plot + 1, _code(MATCHED_SUMMARY))
    return notebook
