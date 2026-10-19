"""Offline tests of the independently developed evaluation workflow."""

import pytest

from src.agents.evaluation_lead import (
    EvaluationSession,
    QualityPolicy,
    run_evaluation_cycle,
)
from src.agents.schemas import Draft, Evaluation, Finding


def draft(name: str = "original") -> Draft:
    return Draft(
        ticker="NVDA",
        thesis=name,
        bull_case=["Revenue growth"],
        bear_case=["Competition"],
        fundamentals="Revenue $100B",
        news_and_catalysts="News",
        market_and_macro="Macro",
        conclusion="Conclusion",
    )


def grade(
    average_score: int = 5,
    *,
    accuracy: int | None = None,
    evidence: int | None = None,
) -> Evaluation:
    return Evaluation(
        completeness=average_score,
        evidence=average_score if evidence is None else evidence,
        accuracy=average_score if accuracy is None else accuracy,
        balance=average_score,
        clarity=average_score,
        critiques=["Improve sourcing"],
    )


def test_rejects_high_average_with_low_accuracy():
    result = QualityPolicy().failures(grade(5, accuracy=1))
    assert "accuracy_below_threshold" in result
    assert "average_below_threshold" not in result  # average is 4.2


def test_rejects_low_evidence_even_when_average_passes():
    result = QualityPolicy().failures(grade(5, evidence=2))
    assert result == ("evidence_below_threshold",)


def test_passing_evaluation_has_no_failures():
    assert QualityPolicy().failures(grade(4)) == ()


@pytest.mark.parametrize(
    "settings",
    [
        {"min_average": 6.0},
        {"min_accuracy": 0},
        {"min_evidence": 6},
        {"max_evaluations": 0},
        {"min_revisions": 3, "max_evaluations": 3},
    ],
)
def test_invalid_policy_rejected(settings):
    with pytest.raises(ValueError):
        QualityPolicy(**settings)


def test_one_revision_required_even_if_initial_draft_passes():
    session = EvaluationSession()
    session.record(draft(), grade(5))
    assert session.should_refine() is True
    with pytest.raises(RuntimeError):
        session.finish()


def test_best_accepted_draft_survives_worse_revision():
    session = EvaluationSession()
    session.record(draft("best"), grade(5))
    session.record(draft("worse"), grade(4))
    assert session.should_refine() is False
    outcome = session.finish()
    assert outcome.accepted is True
    assert outcome.selected_iteration == 0
    assert outcome.draft.thesis == "best"
    assert outcome.stop_reason == "quality_gate_met"
    assert [row["selected"] for row in outcome.rows()] == [True, False]


def test_improvement_selected_when_it_outscores_original():
    session = EvaluationSession()
    session.record(draft("weak"), grade(3))
    session.record(draft("strong"), grade(5))
    outcome = session.finish()
    assert outcome.accepted is True
    assert outcome.selected_iteration == 1
    assert outcome.draft.thesis == "strong"


def test_exhaustion_not_reported_as_success():
    session = EvaluationSession()
    for i in range(3):
        session.record(draft(f"weak{i}"), grade(3, accuracy=2))
    assert session.should_refine() is False
    outcome = session.finish()
    assert outcome.accepted is False
    assert outcome.stop_reason == "max_evaluations_reached"
    assert outcome.evaluation.accuracy == 2


def test_saves_snapshot_not_mutable_reference():
    original = draft()
    session = EvaluationSession()
    session.record(original, grade(5))
    original.thesis = "mutated"
    session.record(draft("revision"), grade(4))
    assert session.finish().draft.thesis == "original"


def test_refuses_cross_ticker_candidate():
    session = EvaluationSession()
    session.record(draft(), grade(4))
    other = draft()
    other.ticker = "JPM"
    with pytest.raises(ValueError, match="same ticker"):
        session.record(other, grade(4))


def test_cannot_exceed_evaluation_budget():
    session = EvaluationSession()
    for _ in range(3):
        session.record(draft(), grade(3))
    with pytest.raises(RuntimeError, match="budget exhausted"):
        session.record(draft(), grade(5))


def test_integrated_loop_with_fake_callbacks_and_no_api():
    grades = iter([grade(3), grade(5), grade(4)])
    calls = {"evaluate": 0, "optimize": 0}

    def fake_evaluate(current, findings):
        calls["evaluate"] += 1
        return next(grades)

    def fake_optimize(current, assessment, findings):
        calls["optimize"] += 1
        return draft(f"revision {calls['optimize']}")

    result = run_evaluation_cycle(
        draft("original"),
        [],
        evaluate_fn=fake_evaluate,
        optimize_fn=fake_optimize,
    )
    assert calls == {"evaluate": 2, "optimize": 1}
    assert result.selected_iteration == 1
    assert result.draft.thesis == "revision 1"
    assert result.accepted is True


def test_loop_handles_failing_revisions_until_budget():
    def fake_evaluate(current, findings):
        return grade(5, accuracy=1)

    def fake_optimize(current, assessment, findings):
        return draft("still inaccurate")

    result = run_evaluation_cycle(
        draft(),
        [],
        evaluate_fn=fake_evaluate,
        optimize_fn=fake_optimize,
    )
    assert result.accepted is False
    assert len(result.history) == 3
    assert "accuracy_below_threshold" in result.rows()[0]["failures"]


def test_optimizer_cannot_change_ticker():
    def fake_evaluate(current, findings):
        return grade(3)

    def fake_optimize(current, assessment, findings):
        other = draft()
        other.ticker = "XOM"
        return other

    with pytest.raises(ValueError, match="changed the stock ticker"):
        run_evaluation_cycle(
            draft(), [], evaluate_fn=fake_evaluate, optimize_fn=fake_optimize
        )
