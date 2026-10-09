"""Planner: turns a ticker into an ordered research plan, using memory."""

from src.agents.schemas import ResearchPlan
from src.agents.toolkit import describe_tools
from src.config import COMPANIES, MODELS
from src.llm import structured
from src.memory.store import format_notes

PLANNER_SYSTEM = (
    "You are the research planner for an equity analyst. Given a ticker, "
    "produce an ordered plan of 5 to 8 steps. Each step names one tool and "
    "says what it should find and why it matters. Cover fundamentals, news, "
    "market behaviour, macro context, and filing risks. Apply the lessons "
    "from past runs when they are relevant to this company or sector."
)


def make_plan(
    ticker: str,
    sector: str | None,
    notes: list[dict],
    model: str = MODELS["writer"],
) -> ResearchPlan:
    user = (
        f"Ticker: {ticker} ({COMPANIES.get(ticker, ticker)})\n"
        f"Sector: {sector or 'unknown'}\n\n"
        f"Available tools:\n{describe_tools()}\n\n"
        f"Lessons from past runs:\n{format_notes(notes)}"
    )
    plan = structured(model, PLANNER_SYSTEM, user, ResearchPlan)
    plan.ticker = ticker
    return plan
