"""Typed outputs for every agent step, so the trace can print each one."""

from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["earnings", "news", "market", "general"]
ToolName = Literal[
    "get_price_history",
    "get_key_stats",
    "get_financials",
    "get_macro_snapshot",
    "get_news",
    "search_filings",
]


# Planning ------------------------------------------------------------------
class PlanStep(BaseModel):
    id: int
    task: str = Field(description="What to find out, in one sentence")
    tool: ToolName
    purpose: str = Field(description="Why this matters for the brief")
    priority: Literal["high", "medium", "low"] = "medium"


class ResearchPlan(BaseModel):
    ticker: str
    hypothesis: str = Field(description="The key question for this stock")
    steps: list[PlanStep]


# Routing -------------------------------------------------------------------
class RouteDecision(BaseModel):
    step_id: int
    route: Route
    reason: str


# Specialists ----------------------------------------------------------------
class Finding(BaseModel):
    route: Route
    summary: str = Field(description="2-4 sentences with concrete figures")
    key_figures: list[str] = Field(
        description="e.g. 'Revenue FY2025: $130.5B'"
    )
    risks: list[str]
    sources: list[str] = Field(
        description="Tool names or filing sections used"
    )


# Synthesis -----------------------------------------------------------------
class Draft(BaseModel):
    ticker: str
    thesis: str = Field(
        max_length=600, description="A decision summary in 3-4 sentences"
    )
    bull_case: list[str] = Field(max_length=5, description="At most 5 points")
    bear_case: list[str] = Field(max_length=5, description="At most 5 points")
    fundamentals: str
    news_and_catalysts: str
    market_and_macro: str
    conclusion: str


# Evaluation ----------------------------------------------------------------
CRITERIA = ("completeness", "evidence", "accuracy", "balance", "clarity")


class Evaluation(BaseModel):
    completeness: int = Field(ge=1, le=5)
    evidence: int = Field(ge=1, le=5)
    accuracy: int = Field(ge=1, le=5)
    balance: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    critiques: list[str] = Field(description="Specific, fixable problems")

    @property
    def average(self) -> float:
        return round(
            sum(getattr(self, c) for c in CRITERIA) / len(CRITERIA), 2
        )


# Reflection and learning ----------------------------------------------------
class Reflection(BaseModel):
    weakest_section: str
    unsupported_claims: list[str]
    missing_data: list[str]
    failed_tools: list[str]
    what_to_change: list[str]
    confidence: Literal["low", "medium", "high"]
    lessons: list[str] = Field(
        description="3-5 short, reusable lessons for future runs"
    )
