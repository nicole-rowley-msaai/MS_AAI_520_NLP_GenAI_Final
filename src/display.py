"""Render parts of a RunTrace as DataFrames or markdown for the notebook."""

import json

import pandas as pd

from src.trace import RunTrace


def plan_table(trace: RunTrace) -> pd.DataFrame:
    plan = trace.select("planner", "plan")[-1].output
    return pd.DataFrame(plan["steps"])[
        ["id", "task", "tool", "purpose", "priority"]
    ]


def memory_read(trace: RunTrace) -> pd.DataFrame:
    notes = trace.select("memory", "read")[-1].output
    if not notes:
        print("No prior lessons: first run for this ticker and sector.")
    return pd.DataFrame(notes) if notes else pd.DataFrame(columns=["lesson"])


def report_markdown(report: str):
    """The brief as a Markdown display object, safe for Jupyter.

    Jupyter runs MathJax on Markdown output, so any text between two
    dollar signs ("$215.9B ... $96.7B") renders as a garbled formula.
    Escaping the dollars keeps them literal.
    """
    from IPython.display import Markdown

    return Markdown(report.replace("$", r"\$"))


def tool_log(trace: RunTrace) -> pd.DataFrame:
    rows = []
    for e in trace.select("tool", "call"):
        rows.append(
            {
                "tool": e.input.get("tool"),
                "args": {
                    k: v
                    for k, v in e.input.items()
                    if k not in ("tool", "ticker")
                },
                "ok": e.output.get("ok"),
                "rows": e.output.get("rows"),
                "error": e.output.get("error"),
            }
        )
    return pd.DataFrame(rows)


def routing_table(trace: RunTrace) -> pd.DataFrame:
    rows = [e.output for e in trace.select("router", "decision")]
    return pd.DataFrame(rows)[["step_id", "route", "reason"]]


def news_chain_steps(trace: RunTrace, max_items: int = 5) -> None:
    for e in trace.select("news_chain"):
        print(f"\n=== {e.kind.upper()} ===")
        out = e.output
        if (
            isinstance(out, dict)
            and all(k.isdigit() for k in out)
            and len(out) > max_items
        ):
            out = dict(list(out.items())[:max_items])
        print(json.dumps(out, indent=1, default=str)[:3000])


def findings_table(trace: RunTrace) -> pd.DataFrame:
    rows = [e.output for e in trace.select("specialist")]
    return pd.DataFrame(rows)[["route", "summary", "key_figures", "risks"]]


def evaluation_history(final_state) -> pd.DataFrame:
    hist = final_state["history"]
    rows = [
        {"iteration": h["iteration"], "average": h["average"], **h["scores"]}
        for h in hist
    ]
    return pd.DataFrame(rows)


def critiques(final_state) -> None:
    for h in final_state["history"]:
        print(f"\nIteration {h['iteration']} (avg {h['average']}):")
        for c in h["critiques"]:
            print(f"  - {c}")


def plot_scores(final_state, ax=None):
    import matplotlib.pyplot as plt

    df = evaluation_history(final_state)
    ax = ax or plt.subplots(figsize=(5, 3))[1]
    ax.plot(df["iteration"], df["average"], marker="o", color="#3b6fb6")
    ax.axhline(4.0, ls="--", color="#999", lw=1, label="pass threshold")
    ax.set_xticks(df["iteration"])
    ax.set_ylim(1, 5)
    ax.set_xlabel("Evaluator iteration")
    ax.set_ylabel("Average score (1-5)")
    ax.set_title("Score rises with each refinement")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    return ax


def reflection_view(final_state) -> dict:
    return final_state["reflection"].model_dump()


def memory_diff(trace: RunTrace) -> pd.DataFrame:
    new = trace.select("memory", "write")[-1].output
    return pd.DataFrame(new) if new else pd.DataFrame(columns=["lesson"])


def timing_table(trace: RunTrace) -> pd.DataFrame:
    rows = [
        {"step": e.step, "kind": e.kind, "seconds": e.seconds}
        for e in trace.events
        if e.seconds
    ]
    return (
        pd.DataFrame(rows).groupby("step")["seconds"].sum().round(1).to_frame()
    )
