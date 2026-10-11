"""End-to-end agent test with a fake LLM and fake tools (no network)."""

import pytest

from src.agents.schemas import (
    Draft,
    Evaluation,
    Finding,
    Reflection,
    ResearchPlan,
    RouteDecision,
    PlanStep,
)
from src.chains.schemas import (
    Classification,
    ClassificationBatch,
    Extraction,
    ExtractionBatch,
    NewsDigest,
)
from src.memory import store
from src.tools import ToolResult


class FakeLLM:
    """Returns a plausible object for each schema; scores rise per call."""

    def __init__(self):
        self.eval_calls = 0

    def __call__(self, model, system, user, schema):
        if schema is ResearchPlan:
            return ResearchPlan(
                ticker="X",
                hypothesis="h",
                steps=[
                    PlanStep(
                        id=1,
                        task="fundamentals",
                        tool="get_financials",
                        purpose="p",
                    ),
                    PlanStep(id=2, task="news", tool="get_news", purpose="p"),
                    PlanStep(
                        id=3,
                        task="prices",
                        tool="get_price_history",
                        purpose="p",
                    ),
                ],
            )
        if schema is RouteDecision:
            tool = user.split("Tool: ")[1].split("\n")[0]
            route = {"get_financials": "earnings", "get_news": "news"}.get(
                tool, "market"
            )
            return RouteDecision(step_id=0, route=route, reason="fake")
        if schema is Finding:
            return Finding(
                route="general",
                summary="Revenue $10B.",
                key_figures=["Revenue $10B"],
                risks=["r"],
                sources=[],
            )
        if schema is Draft:
            return Draft(
                ticker="X",
                thesis="t",
                bull_case=["b"],
                bear_case=["c"],
                fundamentals="Revenue $10B",
                news_and_catalysts="n",
                market_and_macro="m",
                conclusion="z",
            )
        if schema is Evaluation:
            self.eval_calls += 1
            s = 3 if self.eval_calls == 1 else 4
            return Evaluation(
                completeness=s,
                evidence=s,
                accuracy=s,
                balance=s,
                clarity=s,
                critiques=["fix"],
            )
        if schema is Reflection:
            return Reflection(
                weakest_section="news",
                unsupported_claims=[],
                missing_data=[],
                failed_tools=[],
                what_to_change=[],
                confidence="medium",
                lessons=["Lesson A", "Lesson B"],
            )
        if schema is ClassificationBatch:
            return ClassificationBatch(
                items=[
                    Classification(id=0, topic="other", sentiment="neutral")
                ]
            )
        if schema is ExtractionBatch:
            return ExtractionBatch(
                items=[Extraction(id=0, entities=[], figures=[], events=[])]
            )
        if schema is NewsDigest:
            return NewsDigest(
                ticker="X",
                net_sentiment="neutral",
                top_drivers=["d"],
                risks=["r"],
                summary="s",
            )
        raise AssertionError(f"unexpected schema {schema}")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    fake = FakeLLM()
    for mod in (
        "src.agents.planner",
        "src.agents.router",
        "src.agents.specialists",
        "src.agents.synthesizer",
        "src.agents.evaluator",
        "src.chains.news_chain",
    ):
        monkeypatch.setattr(f"{mod}.structured", fake)
    import src.agents.toolkit as tk
    import src.graph as g

    monkeypatch.setattr(
        tk,
        "call_tool",
        lambda name, ticker, trace=None, **kw: (
            trace.log(
                "tool",
                "call",
                {"tool": name, "ticker": ticker},
                {"ok": True, "rows": 1, "error": None, "preview": {}},
            )
            or ToolResult(
                tool=name, ok=True, data={"Revenue": 1e10}, meta={"rows": 1}
            )
        ),
    )
    monkeypatch.setattr("src.agents.specialists.call_tool", tk.call_tool)
    monkeypatch.setattr(
        "src.chains.news_chain.get_news",
        lambda t, c: [
            {
                "title": "X beats",
                "description": "",
                "source": "",
                "published": "2099-01-01T00:00:00Z",
                "url": "",
            }
        ],
    )
    monkeypatch.setattr(g, "get_key_stats", lambda t: {"sector": "Tech"})
    notes = tmp_path / "notes.json"
    monkeypatch.setattr(store, "NOTES_PATH", notes)
    monkeypatch.setattr(store.load_notes, "__defaults__", (notes,))
    monkeypatch.setattr(store.add_lessons, "__defaults__", (notes,))
    return g, fake, notes


def test_full_run_loops_once_then_passes(wired):
    g, fake, notes = wired
    report, trace, state = g.run_agent("X", save=False)

    assert "research brief" in report
    assert [h["average"] for h in state["history"]] == [3.0, 4.0]
    assert state["iteration"] == 1
    assert fake.eval_calls == 2

    steps = {e.step for e in trace.events}
    assert {
        "memory",
        "planner",
        "router",
        "tool",
        "news_chain",
        "specialist",
        "synthesizer",
        "evaluator",
        "optimizer",
        "reflection",
    } <= steps
    routes = {e.output["route"] for e in trace.select("router", "decision")}
    assert routes == {"earnings", "news", "market"}

    assert notes.exists()
    saved = store.load_notes(notes)
    assert [n["lesson"] for n in saved] == ["Lesson A", "Lesson B"]
    assert all(n["sector"] == "Tech" for n in saved)


def test_memory_off_reads_and_writes_nothing(wired):
    g, _, notes = wired
    _, trace, state = g.run_agent("X", use_memory=False, save=False)
    assert trace.select("memory", "read")[0].output == []
    assert state["new_notes"] == []
    assert not notes.exists()


def test_second_run_reads_first_runs_lessons(wired):
    g, _, _ = wired
    g.run_agent("X", save=False)
    _, trace, _ = g.run_agent("X", save=False)
    read = trace.select("memory", "read")[0].output
    assert [n["lesson"] for n in read] == ["Lesson A", "Lesson B"]


def test_unsupported_figures_flags_hallucinated_numbers():
    from src.evaluation import unsupported_figures

    f = [
        Finding(
            route="earnings",
            summary="Revenue $10B, margin 25%.",
            key_figures=[],
            risks=[],
            sources=[],
        )
    ]
    d = Draft(
        ticker="X",
        thesis="Revenue $10B and EPS $4.20",
        bull_case=[],
        bear_case=[],
        fundamentals="margin 25%",
        news_and_catalysts="",
        market_and_macro="",
        conclusion="",
    )
    assert unsupported_figures(d, f) == ["$4.20"]


def test_debt_to_equity_computed_from_balance_sheet():
    from src.agents.toolkit import _debt_to_equity

    summary = {
        "Total Debt": {"2026-01-25": 11.04e9, "2025-01-26": 10.27e9},
        "Stockholders Equity": {"2026-01-25": 157.29e9, "2025-01-26": 79.3e9},
    }
    ratio = _debt_to_equity(summary)
    assert ratio == {"2026-01-25": 0.07, "2025-01-26": 0.13}


def test_report_markdown_escapes_dollars():
    from src.display import report_markdown

    md = report_markdown("Revenue $215.9B and FCF $96.7B")
    assert md.data == r"Revenue \$215.9B and FCF \$96.7B"


def test_long_thesis_does_not_fail_validation():
    d = Draft(
        ticker="X",
        thesis="w" * 700,
        bull_case=list("abcdefg"),
        bear_case=[],
        fundamentals="",
        news_and_catalysts="",
        market_and_macro="",
        conclusion="",
    )
    assert len(d.thesis) == 700
    assert d.bull_case == list("abcde")
