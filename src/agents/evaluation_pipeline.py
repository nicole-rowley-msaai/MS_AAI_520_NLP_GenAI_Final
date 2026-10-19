"""Compose rubric evaluation, numeric auditing, and report revision selection.

Independent evaluation workflow for the AAI 520 EquityAgent project.
The LLM assigns qualitative scores; the numeric audit evaluates only a narrow
subset of explicit finance statements against specialist findings. This module
reports any lack of numeric coverage rather than declaring the report correct.

AI-assisted development must be disclosed in the final course deliverable.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from src.agents.evaluation_lead import EvaluationSession, QualityPolicy
from src.agents.numeric_validation import NumericAudit, audit_numeric_claims
from src.agents.schemas import Draft, Evaluation, Finding


@dataclass(frozen=True)
class NumericPolicy:
    """Control how incomplete numeric evidence affects approval.

    Strict settings reject missing or unresolved financial claims.
    This is appropriate for controlled benchmark fixtures; it may require human
    review when live prose falls outside the narrow claim parser's coverage.
    """

    require_recognized_claims: bool = True
    reject_unverified: bool = True


@dataclass(frozen=True)
class ReviewedDraft:
    """Snapshot of one report, its model scores, and deterministic audit."""

    iteration: int
    draft: Draft
    evaluation: Evaluation
    audit: NumericAudit
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class ReviewedOutcome:
    """Best available candidate with honest accepted/exhausted status."""

    draft: Draft
    evaluation: Evaluation
    audit: NumericAudit
    selected_iteration: int
    accepted: bool
    stop_reason: str
    history: tuple[ReviewedDraft, ...]

    def rows(self) -> list[dict]:
        """Table friendly results for the evaluation notebook."""
        return [
            {
                "iteration": item.iteration,
                "average": item.evaluation.average,
                "accuracy": item.evaluation.accuracy,
                "evidence": item.evaluation.evidence,
                "numeric_checked": len(item.audit.checks),
                "numeric_contradicted": len(item.audit.contradictions),
                "numeric_unverified": len(item.audit.unverified),
                "accepted": item.passed,
                "failures": list(item.failures),
                "selected": item.iteration == self.selected_iteration,
            }
            for item in self.history
        ]


@dataclass
class ReviewedEvaluationSession:
    """Track scores and numeric audits without changing the original graph."""

    quality_policy: QualityPolicy = field(default_factory=QualityPolicy)
    numeric_policy: NumericPolicy = field(default_factory=NumericPolicy)
    score_session: EvaluationSession = field(init=False)
    candidates: list[ReviewedDraft] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.score_session = EvaluationSession(self.quality_policy)

    def record(
        self, draft: Draft, evaluation: Evaluation, audit: NumericAudit
    ) -> ReviewedDraft:
        """Record an immutable snapshot with combined acceptance reasons."""
        score = self.score_session.record(draft, evaluation)
        reasons = list(score.failures)
        if audit.contradictions:
            reasons.append("numeric_contradictions")
        if audit.unverified and self.numeric_policy.reject_unverified:
            reasons.append("numeric_unverified")
        if not audit.checks and self.numeric_policy.require_recognized_claims:
            reasons.append("numeric_no_recognized_claims")
        item = ReviewedDraft(
            iteration=score.iteration,
            draft=score.draft,
            evaluation=score.evaluation,
            audit=audit,
            failures=tuple(reasons),
        )
        self.candidates.append(item)
        return item

    def should_refine(self) -> bool:
        """Honor minimum revisions and maximum evaluations before stopping."""
        if not self.candidates:
            raise ValueError("record an evaluation before deciding to refine")
        count = len(self.candidates)
        if count >= self.quality_policy.max_evaluations:
            return False
        if count <= self.quality_policy.min_revisions:
            return True
        return not any(item.passed for item in self.candidates)

    def finish(self) -> ReviewedOutcome:
        """Select the strongest accepted report, not an inferior revision."""
        if not self.candidates:
            raise ValueError("cannot finish an empty evaluation session")
        if self.should_refine():
            raise RuntimeError("evaluation cycle is not finished")

        passing = [item for item in self.candidates if item.passed]
        if passing:
            chosen = max(
                passing,
                key=lambda item: (
                    item.evaluation.average,
                    item.evaluation.accuracy,
                    item.evaluation.evidence,
                ),
            )
        else:
            # A rejected draft is retained for inspection, not represented
            # as a passing financial report.
            chosen = max(
                self.candidates,
                key=lambda item: (
                    -len(item.failures),
                    -len(item.audit.contradictions),
                    -len(item.audit.unverified),
                    item.evaluation.accuracy,
                    item.evaluation.average,
                ),
            )
        return ReviewedOutcome(
            draft=chosen.draft.model_copy(deep=True),
            evaluation=chosen.evaluation.model_copy(deep=True),
            audit=chosen.audit,
            selected_iteration=chosen.iteration,
            accepted=bool(passing),
            stop_reason=(
                "accepted_by_both_checks"
                if passing
                else "evaluation_budget_exhausted_without_acceptance"
            ),
            history=tuple(self.candidates),
        )


def numeric_feedback(
    audit: NumericAudit, policy: NumericPolicy | None = None
) -> list[str]:
    """Turn machine detected problems into specific revision instructions.

    The optimizer gets these critiques in addition to the LLM critiques.
    We do not silently alter the LLM's original numeric scores.

    With a ``policy``, only problems that policy can reject are sent:
    contradictions always; unverified figures only if ``reject_unverified``;
    the no-coverage note only if ``require_recognized_claims``. Sending
    nonblocking items made the optimizer rewrite figures to suit the
    parser rather than the reader. Without a policy, everything is sent.
    """
    send_unverified = policy is None or policy.reject_unverified
    send_coverage = policy is None or policy.require_recognized_claims
    feedback = []
    for check in audit.contradictions:
        expected = (
            f"; expected value {check.expected}"
            if check.expected is not None
            else ""
        )
        feedback.append(
            f"Numeric contradiction in {check.metric}: {check.excerpt!r}"
            f"{expected}. Confirm metric, period, units and calculation."
        )
    for check in audit.unverified if send_unverified else ():
        feedback.append(
            f"Numeric evidence unresolved for {check.metric}: "
            f"{check.excerpt!r} ({check.reason}). "
            "Locate comparable findings or mark the claim as unverified."
        )
    if send_coverage and not audit.checks:
        feedback.append(
            "No numerical statements matched the limited numeric validator. "
            "State metric, fiscal period, and units explicitly; "
            "do not treat lack of matches as proof of accuracy."
        )
    return feedback


def run_reviewed_cycle(
    initial_draft: Draft,
    findings: list[Finding],
    *,
    evaluate_fn: Callable[[Draft, list[Finding]], Evaluation],
    optimize_fn: Callable[[Draft, Evaluation, list[Finding]], Draft],
    quality_policy: QualityPolicy | None = None,
    numeric_policy: NumericPolicy | None = None,
    audit_fn: Callable[[Draft, list[Finding]], NumericAudit] = (
        audit_numeric_claims
    ),
) -> ReviewedOutcome:
    """Run combined evaluation using injected LLM or offline test callbacks.

    No model calls occur unless the caller explicitly supplies model backed
    evaluate_fn and optimize_fn. This function is NOT wired into graph.py.
    """
    session = ReviewedEvaluationSession(
        quality_policy=quality_policy or QualityPolicy(),
        numeric_policy=numeric_policy or NumericPolicy(),
    )
    current = initial_draft.model_copy(deep=True)
    while True:
        evaluation = evaluate_fn(current, findings)
        audit = audit_fn(current, findings)
        session.record(current, evaluation, audit)
        if not session.should_refine():
            return session.finish()
        revised_feedback = evaluation.model_copy(deep=True)
        revised_feedback.critiques.extend(
            numeric_feedback(audit, session.numeric_policy)
        )
        updated = optimize_fn(current, revised_feedback, findings)
        if updated.ticker != initial_draft.ticker:
            raise ValueError("optimizer changed the stock ticker")
        current = updated.model_copy(deep=True)
