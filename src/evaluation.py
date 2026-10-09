"""Evaluation helpers: the single-prompt baseline and run comparisons."""

import re

from src.agents import evaluator as ev
from src.agents.schemas import Draft, Finding
from src.agents.synthesizer import render
from src.agents.toolkit import call_tool
from src.config import COMPANIES, MODELS
from src.llm import structured

BASELINE_SYSTEM = (
    "You are an equity analyst. Write a research brief on the company from "
    "your own knowledge. Give a thesis, bull case, bear case, fundamentals, "
    "news and catalysts, market and macro context, and a conclusion."
)


def baseline_brief(ticker: str, model: str = MODELS["writer"]) -> Draft:
    """No tools, no plan, no loop: one prompt. The bar the agent must beat."""
    draft = structured(
        model,
        BASELINE_SYSTEM,
        f"Ticker: {ticker} ({COMPANIES.get(ticker, ticker)})",
        Draft,
    )
    draft.ticker = ticker
    return draft


def score_against_findings(draft: Draft, findings: list[Finding]) -> dict:
    """Grade any draft with the same rubric and the same findings."""
    e = ev.evaluate(draft, findings)
    return {
        "average": e.average,
        **{c: getattr(e, c) for c in ev.CRITERIA},
        "critiques": e.critiques,
    }


NUMBER = re.compile(r"\$?\d[\d,]*\.?\d*%?[BMK]?")


def figures_in(text: str) -> set[str]:
    """Numbers mentioned in a text, for the numeric spot-check."""
    return {m.group().rstrip(".") for m in NUMBER.finditer(text)}


def unsupported_figures(draft: Draft, findings: list[Finding]) -> list[str]:
    """Figures in the brief that no finding supports (hallucination check)."""
    brief = figures_in(render(draft))
    support = set()
    for f in findings:
        support |= figures_in(f.summary + " ".join(f.key_figures + f.risks))
    trivial = {str(n) for n in range(0, 11)}
    return sorted(brief - support - trivial)


def tool_success_rate(trace) -> float:
    calls = [e.output for e in trace.select("tool", "call")]
    return (
        round(sum(1 for c in calls if c.get("ok")) / len(calls), 3)
        if calls
        else 0.0
    )


def quick_smoke(ticker: str) -> dict:
    """One call per tool, for the notebook's setup check."""
    out = {}
    for name in (
        "get_price_history",
        "get_key_stats",
        "get_financials",
        "get_macro_snapshot",
        "get_news",
    ):
        r = call_tool(name, ticker)
        out[name] = "ok" if r.ok else r.error
    return out
