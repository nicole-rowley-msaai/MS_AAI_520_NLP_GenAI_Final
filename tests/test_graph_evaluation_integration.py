"""Offline integration tests of the opt-in LangGraph evaluator nodes.

Tests use real Draft, Finding, Evaluation, and RunTrace objects, but no LLM
or market API calls. AI-assisted development disclosed in project appendix.
"""

import pytest

from src.agents.schemas import Draft, Evaluation, Finding, Reflection
from src.trace import RunTrace
import src.graph as graph


def report(amount: str, *, thesis: str = "initial") -> Draft:
    return Draft(
        ticker="NVDA", thesis=thesis,
        bull_case=["growth"], bear_case=["valuation"],
        fundamentals=f"Revenue FY2025: {amount}",
        news_and_catalysts="news", market_and_macro="macro",
        conclusion="conclusion",
    )


def grade(score: int = 5, *, accuracy: int | None = None) -> Evaluation:
    return Evaluation(
        completeness=score, evidence=score,
        accuracy=score if accuracy is None else accuracy,
        balance=score, clarity=score, critiques=["check figures"],
    )


def finding() -> Finding:
    return Finding(
        route="earnings", summary="Revenue FY2025: $100B",
        key_figures=["Revenue FY2025: $100B"], risks=[], sources=["test"],
    )


def setup(monkeypatch, initial: Draft) -> dict:
    monkeypatch.setattr(graph, "synthesize", lambda ticker, findings: initial)
    monkeypatch.setattr(
        graph.ev, "reflect",
        lambda draft, findings, tools, **kwargs: Reflection(
            weakest_section="news", unsupported_claims=[], missing_data=[],
            failed_tools=[], what_to_change=[], confidence="medium",
            lessons=["Review source data"],
        ),
    )
    state = {
        "ticker": "NVDA", "trace": RunTrace("NVDA"), "use_memory": False,
        "reviewed_evaluation": True, "findings": [finding()],
    }
    state.update(graph.node_synthesize(state))
    return state


def evaluate(state):
    state.update(graph.node_evaluate(state))
    return graph.should_refine(state)


def optimize(state):
    state.update(graph.node_optimize(state))


def finish(state):
    state.update(graph.node_reflect(state))
    return state


def test_keeps_best_valid_draft_even_when_revision_scores_lower(monkeypatch):
    state = setup(monkeypatch, report("$100B", thesis="best"))
    monkeypatch.setattr(
        graph.ev, "evaluate",
        lambda draft, findings: grade(5 if draft.thesis == "best" else 4),
    )
    monkeypatch.setattr(
        graph.ev, "optimize",
        lambda draft, evaluation, findings: report("$100B", thesis="worse"),
    )
    assert evaluate(state) == "refine"
    optimize(state)
    assert evaluate(state) == "finish"
    finish(state)
    assert state["accepted"] is True
    assert state["selected_iteration"] == 0
    assert state["draft"].thesis == "best"
    assert state["evaluation"].average == 5.0
    assert [h["selected"] for h in state["history"]] == [True, False]
    assert "best" in state["report"]
    assert state["trace"].select("evaluator", "selection")


def test_numeric_contradiction_reaches_optimizer_and_is_corrected(monkeypatch):
    state = setup(monkeypatch, report("$150B"))
    monkeypatch.setattr(graph.ev, "evaluate", lambda draft, findings: grade(5))
    feedback_received = []

    def fix(draft, evaluation, findings):
        feedback_received.extend(evaluation.critiques)
        return report("$100B", thesis="corrected")

    monkeypatch.setattr(graph.ev, "optimize", fix)
    assert evaluate(state) == "refine"
    assert "numeric_contradictions" in state["history"][0]["failures"]
    optimize(state)
    assert any("Numeric contradiction" in msg for msg in feedback_received)
    assert evaluate(state) == "finish"
    finish(state)
    assert state["accepted"] is True
    assert state["selected_iteration"] == 1
    assert state["history"][0]["numeric_contradicted"] == 1
    assert state["history"][1]["numeric_contradicted"] == 0
    assert "$100B" in state["report"]


def test_unrecognized_claims_are_warnings_not_numeric_approval(monkeypatch):
    state = setup(monkeypatch, report("$100B", thesis="no numeric syntax"))
    state["draft"].fundamentals = "Growing margins but no explicit numbers"
    monkeypatch.setattr(graph.ev, "evaluate", lambda d, f: grade(5))
    monkeypatch.setattr(
        graph.ev, "optimize", lambda d, e, f: d.model_copy(deep=True)
    )
    assert evaluate(state) == "refine"
    assert state["history"][0]["numeric_checked"] == 0
    optimize(state)
    assert evaluate(state) == "finish"
    finish(state)
    assert state["accepted"] is True
    assert state["numeric_audit"].checks == ()
    assert state["trace"].select("evaluator", "numeric_audit")


def test_exhaustion_is_not_misreported_as_a_pass(monkeypatch):
    state = setup(monkeypatch, report("$150B"))
    monkeypatch.setattr(graph.ev, "evaluate", lambda d, f: grade(5))
    monkeypatch.setattr(
        graph.ev, "optimize", lambda d, e, f: d.model_copy(deep=True)
    )
    assert evaluate(state) == "refine"
    optimize(state)
    assert evaluate(state) == "refine"
    optimize(state)
    assert evaluate(state) == "finish"
    finish(state)
    assert state["accepted"] is False
    assert (
        state["stop_reason"]
        == "evaluation_budget_exhausted_without_acceptance"
    )
    assert len(state["history"]) == 3
    assert state["trace"].select("evaluator", "gate")[-1].output == "exhausted"


def test_original_evaluator_path_remains_available(monkeypatch):
    state = setup(monkeypatch, report("$150B"))
    state["reviewed_evaluation"] = False
    monkeypatch.setattr(graph.ev, "evaluate", lambda d, f: grade(5))
    assert evaluate(state) == "refine"
    state["iteration"] = 1
    assert evaluate(state) == "finish"
    finish(state)
    assert state["draft"].fundamentals == "Revenue FY2025: $150B"
    assert "accepted" not in state
    assert "numeric_checked" not in state["history"][-1]


def test_low_accuracy_is_not_accepted_without_numeric_claims(monkeypatch):
    state = setup(monkeypatch, report("$100B"))
    state["draft"].fundamentals = "Qualitative opinion only"
    monkeypatch.setattr(
        graph.ev, "evaluate", lambda d, f: grade(5, accuracy=1)
    )
    monkeypatch.setattr(graph.ev, "optimize", lambda d, e, f: d)
    for expected in ["refine", "refine", "finish"]:
        assert evaluate(state) == expected
        if expected == "refine":
            optimize(state)
    finish(state)
    assert state["accepted"] is False
    assert "accuracy_below_threshold" in state["history"][0]["failures"]
