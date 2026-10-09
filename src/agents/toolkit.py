"""The tools the agent can call, by name, with descriptions for the planner.

Every call goes through `call_tool`, which wraps the result in a ToolResult
(ok/error) and logs it to the trace. search_filings is imported lazily so
the rest of the agent works without chromadb installed.
"""

from src.config import BENCHMARK, COMPANIES
from src.tools import ToolResult, safe_call
from src.tools.macro import get_macro_snapshot
from src.tools.market import (
    compute_technicals, get_financials, get_key_stats, get_price_history,
)
from src.tools.news import get_news
from src.trace import RunTrace

TOOL_DESCRIPTIONS = {
    "get_price_history": "Daily prices for 5 years; used for returns, "
                         "volatility, moving averages, beta vs SPY.",
    "get_key_stats": "Market cap, P/E, P/B, margins, ROE, debt/equity, beta.",
    "get_financials": "Annual income statement, balance sheet, cash flow.",
    "get_macro_snapshot": "Latest CPI, Fed funds, 10-year yield, "
                          "unemployment, GDP with one-year-ago values.",
    "get_news": "Recent headlines about the company (last 30 days).",
    "search_filings": "Semantic search over the latest 10-K Risk Factors "
                      "and MD&A; pass a focused query.",
}


def describe_tools() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items())


def _technicals(ticker: str) -> dict:
    prices = get_price_history(ticker)
    bench = get_price_history(BENCHMARK)
    return compute_technicals(prices, bench)


def _search_filings(ticker: str, query: str, k: int = 3) -> list[dict]:
    from src.tools.retrieval import search_filings  # needs chromadb

    return search_filings(ticker, query, k=k)


def call_tool(name: str, ticker: str, trace: RunTrace | None = None,
              **kwargs) -> ToolResult:
    """Run one named tool for a ticker; never raises."""
    company = COMPANIES.get(ticker, ticker)
    dispatch = {
        "get_price_history": lambda: _technicals(ticker),
        "get_key_stats": lambda: get_key_stats(ticker),
        "get_financials": lambda: _summarize_financials(get_financials(ticker)),
        "get_macro_snapshot": lambda: get_macro_snapshot().to_dict("records"),
        "get_news": lambda: get_news(ticker, company),
        "search_filings": lambda: _search_filings(
            ticker, kwargs.get("query", "key risks and results of operations"),
        ),
    }
    if name not in dispatch:
        result = ToolResult(tool=name, ok=False, error=f"Unknown tool: {name}")
    else:
        result = safe_call(name, dispatch[name])
    if trace is not None:
        trace.log("tool", "call",
                  {"tool": name, "ticker": ticker, **kwargs},
                  {"ok": result.ok, "rows": result.meta.get("rows"),
                   "error": result.error,
                   "preview": _preview(result.data)})
    return result


def _summarize_financials(statements: dict) -> dict:
    """Last two fiscal years of the headline lines, as plain numbers."""
    wanted = {
        "income": ["Total Revenue", "Operating Income", "Net Income",
                   "Diluted EPS"],
        "balance": ["Total Assets", "Total Debt", "Stockholders Equity",
                    "Cash And Cash Equivalents"],
        "cashflow": ["Operating Cash Flow", "Free Cash Flow",
                     "Capital Expenditure"],
    }
    out = {}
    for name, lines in wanted.items():
        df = statements.get(name)
        if df is None or df.empty:
            continue
        cols = list(df.columns[:2])
        for line in lines:
            if line in df.index:
                out[line] = {
                    str(c.date() if hasattr(c, "date") else c):
                    _num(df.loc[line, c]) for c in cols
                }
    return out


def _num(x):
    try:
        return None if x != x else float(x)  # NaN check
    except TypeError:
        return None


def _preview(data, n: int = 3):
    if isinstance(data, list):
        return data[:n]
    if isinstance(data, dict):
        return dict(list(data.items())[:8])
    return data
