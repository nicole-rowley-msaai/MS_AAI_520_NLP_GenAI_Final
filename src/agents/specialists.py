"""Specialist agents: earnings, news, market, and a general fallback.

Each specialist receives the plan steps routed to it, calls its tools,
and writes a Finding with concrete figures. The news specialist runs the
five-step prompt chain instead of a single LLM call.
"""

import json

from src.agents.schemas import Finding, PlanStep, Route
from src.agents.toolkit import call_tool
from src.chains.news_chain import run_news_chain
from src.config import COMPANIES, MODELS
from src.llm import structured
from src.trace import RunTrace

SPECIALIST_SYSTEM = {
    "earnings": (
        "You are an earnings and fundamentals analyst. From the tool output, "
        "report revenue, margins, earnings, cash flow, leverage, and "
        "valuation "
        "multiples with their values. Quote filing excerpts for risks. Only "
        "state figures that appear in the data."
    ),
    "market": (
        "You are a market and macro analyst. From the tool output, report "
        "recent returns, volatility, trend versus moving averages, beta, and "
        "how the macro backdrop (rates, inflation, growth) bears on the "
        "stock. "
        "Only state figures that appear in the data."
    ),
    "general": (
        "You are a general equity analyst. Summarize what the tool output "
        "says that is relevant to an investment view. Only state figures "
        "that appear in the data."
    ),
}


def _dump(obj) -> str:
    return json.dumps(obj, default=str)[:12000]


def run_specialist(
    route: Route, ticker: str, steps: list[PlanStep], trace: RunTrace
) -> Finding:
    if route == "news":
        return _news_specialist(ticker, trace)

    outputs, sources = {}, []
    for step in steps:
        kwargs = {}
        if step.tool == "search_filings":
            kwargs["query"] = step.task
        result = call_tool(step.tool, ticker, trace, **kwargs)
        sources.append(step.tool)
        outputs[f"step {step.id}: {step.tool}"] = (
            result.data if result.ok else f"ERROR: {result.error}"
        )

    user = (
        f"Ticker: {ticker} ({COMPANIES.get(ticker, ticker)})\n"
        f"Tasks: {[s.task for s in steps]}\n\n"
        f"Tool output:\n{_dump(outputs)}"
    )
    finding = structured(
        MODELS["writer"],
        SPECIALIST_SYSTEM[route],
        user,
        Finding,
    )
    finding.route = route
    finding.sources = sorted(set(sources))
    return finding


def _news_specialist(ticker: str, trace: RunTrace) -> Finding:
    chain_log: list[dict] = []
    digest = run_news_chain(
        ticker, COMPANIES.get(ticker, ticker), trace=chain_log
    )
    ingest = chain_log[0]["output"] if chain_log else {}
    trace.log(
        "tool",
        "call",
        {"tool": "get_news", "ticker": ticker},
        {
            "ok": True,
            "rows": ingest.get("n", 0),
            "error": None,
            "preview": ingest.get("sample", []),
        },
    )
    for entry in chain_log:
        trace.log("news_chain", entry["step"], output=entry["output"])
    return Finding(
        route="news",
        summary=digest.summary,
        key_figures=digest.top_drivers,
        risks=digest.risks,
        sources=["get_news", "news prompt chain"],
    )
