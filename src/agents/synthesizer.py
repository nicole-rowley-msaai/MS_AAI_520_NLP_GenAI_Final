"""Synthesizer: combines specialist findings into a draft research brief."""

from src.agents.schemas import Draft, Finding
from src.config import COMPANIES, MODELS
from src.llm import structured

SYNTH_SYSTEM = (
    "You are a senior equity analyst writing a research brief for a "
    "portfolio manager. Use only the findings provided. Every figure must "
    "come from a finding. Give a balanced bull and bear case, then a clear "
    "conclusion. This is analysis, not investment advice."
)


def findings_text(findings: list[Finding]) -> str:
    parts = []
    for f in findings:
        parts.append(
            f"## {f.route.upper()} FINDING\n{f.summary}\n"
            f"Key figures: {f.key_figures}\nRisks: {f.risks}\n"
            f"Sources: {f.sources}"
        )
    return "\n\n".join(parts)


def synthesize(
    ticker: str, findings: list[Finding], model: str = MODELS["writer"]
) -> Draft:
    user = (
        f"Ticker: {ticker} ({COMPANIES.get(ticker, ticker)})\n\n"
        + findings_text(findings)
    )
    draft = structured(model, SYNTH_SYSTEM, user, Draft)
    draft.ticker = ticker
    return draft


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {b}" for b in items)


def render(draft: Draft) -> str:
    """Markdown version of the brief for the notebook and the report file."""
    bullets = _bullets
    return (
        f"# {draft.ticker} research brief\n\n"
        f"**Thesis:** {draft.thesis}\n\n"
        f"## Bull case\n{bullets(draft.bull_case)}\n\n"
        f"## Bear case\n{bullets(draft.bear_case)}\n\n"
        f"## Fundamentals\n{draft.fundamentals}\n\n"
        f"## News and catalysts\n{draft.news_and_catalysts}\n\n"
        f"## Market and macro\n{draft.market_and_macro}\n\n"
        f"## Conclusion\n{draft.conclusion}\n\n"
        "*Analysis only; not investment advice.*"
    )
