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
    NewsDigestDraft,
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
            s = 5  # passes on the first draft; the loop must still run once
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
        if schema is NewsDigestDraft:
            return NewsDigestDraft(
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
                {"tool": name, "ticker": ticker, **kw},
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


def test_full_run_revises_at_least_once(wired):
    g, fake, notes = wired
    report, trace, state = g.run_agent("X", save=False)

    assert "research brief" in report
    assert [h["average"] for h in state["history"]] == [5.0, 5.0]
    assert state["iteration"] == 1  # one forced revision, then pass
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

    # specialists ran in parallel and every tool call is tagged by route
    wall = trace.select("specialists", "wall")[0]
    assert wall.output["parallel"] == 3
    tagged = {e.input.get("route") for e in trace.select("tool", "call")}
    assert tagged == {"earnings", "news", "market"}
    assert [f.route for f in state["findings"]] == [
        "earnings",
        "news",
        "market",
    ]

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


def test_unsupported_figures_skips_labelled_calculations():
    from src.evaluation import unsupported_figures

    f = [
        Finding(
            route="earnings",
            summary="Revenue $215.9B, up from $130.5B.",
            key_figures=[],
            risks=[],
            sources=[],
        )
    ]
    d = Draft(
        ticker="X",
        thesis="Revenue $215.9B, a calculated ~65% increase; EPS $4.90",
        bull_case=[],
        bear_case=[],
        fundamentals="",
        news_and_catalysts="",
        market_and_macro="",
        conclusion="",
    )
    assert unsupported_figures(d, f) == ["$4.90"]


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


def test_structured_retries_on_malformed_reply(monkeypatch):
    import src.llm as llm

    calls = []

    def flaky(model, system, user, schema):
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("truncated")
        return Reflection(
            weakest_section="",
            unsupported_claims=[],
            missing_data=[],
            failed_tools=[],
            what_to_change=[],
            confidence="low",
            lessons=[],
        )

    monkeypatch.setattr(llm, "_anthropic", flaky)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    out = llm.structured("claude-x", "s", "u", Reflection)
    assert isinstance(out, Reflection) and len(calls) == 2


def test_structured_falls_back_to_other_provider(monkeypatch):
    import src.llm as llm

    calls = []

    def fake_anthropic(model, system, user, schema):
        calls.append(model)
        raise ValueError("stopped early")

    def fake_openai(model, system, user, schema):
        calls.append(model)
        return schema(step_id=1, route="news", reason="ok")

    monkeypatch.setattr(llm, "_anthropic", fake_anthropic)
    monkeypatch.setattr(llm, "_openai", fake_openai)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    out = llm.structured("claude-x", "s", "u", RouteDecision)
    assert out.route == "news"
    assert calls[:3] == ["claude-x"] * 3 and len(calls) == 4


def test_net_sentiment_is_counted_not_judged():
    from src.chains.news_chain import net_sentiment, sentiment_counts
    from src.chains.schemas import Article

    arts = [Article(id=i, title="t") for i in range(10)]
    labels = {i: {"sentiment": "positive"} for i in range(6)}
    labels.update({i: {"sentiment": "negative"} for i in range(6, 10)})
    counts = sentiment_counts(labels, arts)
    assert counts == {"positive": 6, "neutral": 0, "negative": 4}
    assert net_sentiment(counts) == "positive"
    assert net_sentiment({"positive": 5, "neutral": 0, "negative": 4}) == (
        "neutral"
    )


def test_memory_reads_newest_lessons_when_there_are_many():
    notes = [
        {"ticker": "X", "sector": "Tech", "date": f"2026-10-{day:02d}",
         "lesson": f"lesson {day}"}
        for day in range(1, 13)
    ]
    read = store.relevant_notes("X", "Tech", notes, limit=8)
    assert [n["lesson"] for n in read] == [
        f"lesson {day}" for day in range(12, 4, -1)
    ]


def test_memory_ranks_ticker_before_sector_before_general():
    notes = [
        {"ticker": "", "sector": "", "date": "2026-10-19", "lesson": "general"},
        {"ticker": "Y", "sector": "Tech", "date": "2026-10-18",
         "lesson": "sector"},
        {"ticker": "X", "sector": "Tech", "date": "2026-10-01",
         "lesson": "ticker"},
        {"ticker": "Z", "sector": "Energy", "date": "2026-10-19",
         "lesson": "unrelated"},
    ]
    read = store.relevant_notes("X", "Tech", notes)
    assert [n["lesson"] for n in read] == ["ticker", "sector", "general"]


def test_reviewed_run_sends_optimizer_only_blocking_numeric_feedback(wired):
    # The fake drafts contain no recognized figures. Under the live policy
    # that cannot block acceptance, so the optimizer must not be told about it.
    g, fake, _ = wired
    prompts = []
    original = fake.__class__.__call__

    def recording(self, model, system, user, schema):
        prompts.append(user)
        return original(self, model, system, user, schema)

    fake.__class__.__call__ = recording
    try:
        g.run_agent("X", save=False, reviewed_evaluation=True)
    finally:
        fake.__class__.__call__ = original
    assert prompts
    assert not any("No numerical statements matched" in p for p in prompts)


def _reflection_prompts(fake, run):
    prompts = []
    original = fake.__class__.__call__

    def recording(self, model, system, user, schema):
        if schema is Reflection:
            prompts.append(user)
        return original(self, model, system, user, schema)

    fake.__class__.__call__ = recording
    try:
        run()
    finally:
        fake.__class__.__call__ = original
    return prompts


def test_reflection_sees_stored_lessons_and_is_told_not_to_restate(wired):
    g, fake, _ = wired
    first = _reflection_prompts(fake, lambda: g.run_agent("X", save=False))
    assert "LESSONS ALREADY STORED (do not restate)\n(none)" in first[0]
    second = _reflection_prompts(fake, lambda: g.run_agent("X", save=False))
    assert "- Lesson A\n- Lesson B" in second[0]

    from src.agents.evaluator import REFLECT_SYSTEM

    assert "do not restate them" in REFLECT_SYSTEM


def test_reflection_with_memory_off_sees_no_lessons(wired):
    g, fake, _ = wired
    g.run_agent("X", save=False)  # stores Lesson A and B
    prompts = _reflection_prompts(
        fake, lambda: g.run_agent("X", use_memory=False, save=False)
    )
    assert "(none)" in prompts[0]


def test_verifier_runs_between_specialists_and_synthesizer(wired):
    g, _, _ = wired
    _, trace, state = g.run_agent("X", save=False)
    steps = [e.step for e in trace.events]
    last_specialist = max(i for i, s in enumerate(steps) if s == "specialist")
    verifier = steps.index("verifier")
    assert last_specialist < verifier < steps.index("synthesizer")

    # Specialists recorded the raw tool data the verifier read.
    assert trace.select("specialist", "tool_data")
    report = trace.select("verifier", "report")[0].output
    routes = {r["route"] for r in report["rows"]}
    assert routes == {"earnings", "news", "market"}

    # Verified copies go downstream; the specialists' originals are kept.
    earnings = state["findings"][0]
    assert earnings.key_figures[-1].startswith("VERIFICATION:")
    assert not any(
        k.startswith("VERIFICATION")
        for k in state["specialist_findings"][0].key_figures
    )
    assert [r["route"] for r in state["verification"]] == [
        "earnings", "news", "market",
    ]


def test_findings_table_shows_only_findings(wired):
    # Specialists also log their raw tool data under the same step name;
    # those events must not appear as empty rows in the B6 table.
    from src import display as show

    g, _, _ = wired
    _, trace, _ = g.run_agent("X", save=False)
    assert trace.select("specialist", "tool_data")
    table = show.findings_table(trace)
    assert len(table) == 3
    assert not table.isna().any().any()
    assert set(table["route"]) == {"earnings", "news", "market"}
