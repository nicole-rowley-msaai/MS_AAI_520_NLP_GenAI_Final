"""Offline tests for combined model scoring and numeric auditing."""

import pytest

from src.agents.evaluation_lead import QualityPolicy
from src.agents.evaluation_pipeline import (
    NumericPolicy,
    ReviewedEvaluationSession,
    numeric_feedback,
    run_reviewed_cycle,
)
from src.agents.numeric_validation import audit_numeric_claims
from src.agents.schemas import Draft, Evaluation, Finding


def draft(figures="Revenue FY2025: $100B", thesis="original"):
    return Draft(
        ticker="NVDA",
        thesis=thesis,
        bull_case=["Strong demand"],
        bear_case=["Competition"],
        fundamentals=figures,
        news_and_catalysts="News",
        market_and_macro="Macro",
        conclusion="Conclusion",
    )


def findings(*figures):
    return [
        Finding(
            route="earnings",
            summary="Example from a synthetic source",
            key_figures=list(figures or ["Revenue FY2025: $100B"]),
            risks=[],
            sources=["synthetic_fixture"],
        )
    ]


def grade(value=5, *, accuracy=None, evidence=None):
    return Evaluation(
        completeness=value,
        evidence=value if evidence is None else evidence,
        accuracy=value if accuracy is None else accuracy,
        balance=value,
        clarity=value,
        critiques=["Review evidence"],
    )


def session():
    return ReviewedEvaluationSession(
        quality_policy=QualityPolicy(min_revisions=0, max_evaluations=3)
    )


def record(current, score=None, source=None, review=None):
    source = source or findings()
    assessment = score if score is not None else grade()
    review = review if review is not None else session()
    return review.record(
        current, assessment, audit_numeric_claims(current, source)
    )


def test_good_scores_and_supported_figures_pass():
    review = session()
    record(draft(), review=review)
    assert not review.should_refine()
    outcome = review.finish()
    assert outcome.accepted
    assert outcome.stop_reason == "accepted_by_both_checks"


def test_high_llm_score_does_not_override_numeric_contradiction():
    review = session()
    candidate = record(draft("Revenue FY2025: $150B"), review=review)
    assert candidate.evaluation.average == 5
    assert not candidate.passed
    assert "numeric_contradictions" in candidate.failures
    assert review.should_refine()


def test_high_llm_score_does_not_override_unverified_period():
    review = session()
    candidate = record(draft("Revenue FY2024: $100B"), review=review)
    assert not candidate.passed
    assert "numeric_unverified" in candidate.failures


def test_high_llm_score_and_no_checked_claims_is_not_a_pass():
    review = session()
    candidate = record(draft("Strong performance."), review=review)
    assert not candidate.passed
    assert "numeric_no_recognized_claims" in candidate.failures


def test_quality_policy_still_rejects_low_accuracy():
    review = session()
    candidate = record(draft(), score=grade(5, accuracy=1), review=review)
    assert not candidate.passed
    assert "accuracy_below_threshold" in candidate.failures


def test_quality_policy_still_rejects_low_evidence():
    review = session()
    candidate = record(draft(), score=grade(5, evidence=1), review=review)
    assert "evidence_below_threshold" in candidate.failures


def test_best_accepted_candidate_survives_worse_revision():
    review = ReviewedEvaluationSession()
    record(draft(thesis="best"), grade(5), review=review)
    record(draft(thesis="lower"), grade(4), review=review)
    result = review.finish()
    assert result.accepted
    assert result.draft.thesis == "best"
    assert result.selected_iteration == 0
    assert result.rows()[0]["selected"]


def test_rejected_high_score_does_not_replace_lower_scoring_valid_draft():
    review = ReviewedEvaluationSession()
    record(draft("Revenue FY2025: $150B", "invalid"), grade(5), review=review)
    record(draft("Revenue FY2025: $100B", "valid"), grade(4), review=review)
    result = review.finish()
    assert result.accepted
    assert result.draft.thesis == "valid"


def test_no_candidate_passes_returns_explicit_failure():
    review = session()
    for i in range(3):
        record(draft("Revenue FY2025: $150B", f"revision {i}"), review=review)
    result = review.finish()
    assert not result.accepted
    assert (
        result.stop_reason == "evaluation_budget_exhausted_without_acceptance"
    )
    assert len(result.history) == 3


def test_feedback_contains_observed_error_and_expected_value():
    audit = audit_numeric_claims(draft("Revenue FY2025: $150B"), findings())
    feedback = numeric_feedback(audit)
    assert "Revenue FY2025" in feedback[0]
    assert "100000000000" in feedback[0]


def test_feedback_reports_unverified_and_missing_coverage():
    assert (
        "unresolved"
        in numeric_feedback(
            audit_numeric_claims(draft("Revenue FY2024: $100B"), findings())
        )[0]
    )
    assert (
        "No numerical statements"
        in numeric_feedback(
            audit_numeric_claims(
                draft("Prose without recognized financial metrics"), findings()
            )
        )[0]
    )


def test_end_to_end_offline_loop_corrects_numeric_error():
    received = []

    def evaluator(current, evidence):
        return grade(5)

    def optimizer(current, assessment, evidence):
        received.extend(assessment.critiques)
        return draft("Revenue FY2025: $100B", "fixed")

    result = run_reviewed_cycle(
        draft("Revenue FY2025: $150B", "incorrect"),
        findings(),
        evaluate_fn=evaluator,
        optimize_fn=optimizer,
    )
    assert result.accepted
    assert result.selected_iteration == 1
    assert result.draft.thesis == "fixed"
    assert "numeric_contradictions" in result.history[0].failures
    assert any("Numeric contradiction" in message for message in received)


def test_end_to_end_offline_loop_retains_best_scoring_report():
    def evaluator(current, evidence):
        return grade(5 if current.thesis == "strong" else 4)

    def optimizer(current, assessment, evidence):
        return draft(thesis="weaker")

    result = run_reviewed_cycle(
        draft(thesis="strong"),
        findings(),
        evaluate_fn=evaluator,
        optimize_fn=optimizer,
    )
    assert result.selected_iteration == 0
    assert result.draft.thesis == "strong"


def test_end_to_end_loop_limits_bad_numerics_to_budget():
    calls = []

    def evaluator(current, evidence):
        return grade()

    def optimizer(current, assessment, evidence):
        calls.append(assessment)
        return draft("Revenue FY2025: $150B")

    result = run_reviewed_cycle(
        draft("Revenue FY2025: $150B"),
        findings(),
        evaluate_fn=evaluator,
        optimize_fn=optimizer,
    )
    assert not result.accepted
    assert len(result.history) == 3
    assert len(calls) == 2


def test_rejects_optimizer_ticker_change():
    def evaluator(current, evidence):
        return grade(3)

    def optimizer(current, assessment, evidence):
        d = draft()
        d.ticker = "XOM"
        return d

    with pytest.raises(ValueError, match="changed the stock ticker"):
        run_reviewed_cycle(
            draft(), findings(), evaluate_fn=evaluator, optimize_fn=optimizer
        )


def test_record_snapshots_cannot_be_changed_by_callers():
    review = session()
    current = draft()
    record(current, review=review)
    current.thesis = "mutated"
    assert review.candidates[0].draft.thesis == "original"


def test_configurable_coverage_policy_is_explicit():
    review = ReviewedEvaluationSession(
        quality_policy=QualityPolicy(min_revisions=0),
        numeric_policy=NumericPolicy(require_recognized_claims=False),
    )
    candidate = record(draft("No explicit financial figures"), review=review)
    assert candidate.passed
    # Such approval is only score-policy approval, not a factual validation.
    assert not candidate.audit.checks


def test_configurable_unverified_tolerance_is_explicit():
    review = ReviewedEvaluationSession(
        quality_policy=QualityPolicy(min_revisions=0),
        numeric_policy=NumericPolicy(reject_unverified=False),
    )
    candidate = record(draft("Revenue FY2024: $100B"), review=review)
    assert candidate.passed
    assert len(candidate.audit.unverified) == 1


def test_rows_expose_numeric_coverage_and_failures():
    review = session()
    record(draft("Revenue FY2025: $150B"), review=review)
    record(draft("Revenue FY2025: $100B"), review=review)
    result = review.finish()
    rows = result.rows()
    assert rows[0]["numeric_contradicted"] == 1
    assert rows[1]["numeric_contradicted"] == 0
    assert rows[1]["selected"] is True


def test_session_requires_evaluation_before_finish_or_refine():
    review = session()
    with pytest.raises(ValueError):
        review.should_refine()
    with pytest.raises(ValueError):
        review.finish()


def test_session_does_not_allow_premature_finish():
    review = ReviewedEvaluationSession()
    record(draft(), review=review)
    assert review.should_refine()
    with pytest.raises(RuntimeError):
        review.finish()


LIVE_POLICY = NumericPolicy(
    require_recognized_claims=False, reject_unverified=False
)


def test_live_policy_feedback_keeps_contradictions():
    audit = audit_numeric_claims(draft("Revenue FY2025: $150B"), findings())
    feedback = numeric_feedback(audit, LIVE_POLICY)
    assert len(feedback) == 1
    assert "contradiction" in feedback[0]


def test_live_policy_feedback_omits_nonblocking_unverified_figures():
    audit = audit_numeric_claims(draft("Revenue FY2024: $100B"), findings())
    assert audit.unverified
    assert numeric_feedback(audit, LIVE_POLICY) == []


def test_live_policy_feedback_omits_coverage_note():
    audit = audit_numeric_claims(
        draft("Prose without recognized financial metrics"), findings()
    )
    assert numeric_feedback(audit, LIVE_POLICY) == []


def test_strict_policy_feedback_matches_unfiltered_feedback():
    for text in ("Revenue FY2024: $100B", "Prose without metrics"):
        audit = audit_numeric_claims(draft(text), findings())
        assert numeric_feedback(audit, NumericPolicy()) == numeric_feedback(
            audit
        )
