"""Router: sends each plan step to the specialist that should handle it."""

from src.agents.schemas import PlanStep, RouteDecision
from src.config import MODELS
from src.llm import structured

ROUTER_SYSTEM = (
    "You route research tasks to one specialist analyst:\n"
    "- earnings: financial statements, ratios, guidance, 10-K filing text\n"
    "- news: recent headlines, events, sentiment\n"
    "- market: price action, volatility, technicals, macro context\n"
    "- general: anything that fits none of the above\n"
    "Return the route and a one-sentence reason."
)

# Deterministic fallback keyed on the tool, used if the LLM call fails.
TOOL_ROUTES = {
    "get_financials": "earnings", "get_key_stats": "earnings",
    "search_filings": "earnings", "get_news": "news",
    "get_price_history": "market", "get_macro_snapshot": "market",
}


def route_step(step: PlanStep, model: str = MODELS["chain"]) -> RouteDecision:
    user = f"Task {step.id}: {step.task}\nTool: {step.tool}\nPurpose: {step.purpose}"
    try:
        decision = structured(model, ROUTER_SYSTEM, user, RouteDecision)
        decision.step_id = step.id
        return decision
    except Exception as exc:  # noqa: BLE001 - never let routing stop a run
        return RouteDecision(
            step_id=step.id,
            route=TOOL_ROUTES.get(step.tool, "general"),
            reason=f"Rule-based fallback after LLM error: {type(exc).__name__}",
        )
