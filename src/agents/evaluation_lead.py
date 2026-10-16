"""Independent evaluation policy and revision selection for EquityAgent.

Designed to work with the project's existing Draft, Finding, and Evaluation
schemas. All policy decisions are deterministic; the injected evaluator and
optimizer may be LLM-backed or offline test doubles.

The quality gate checks reported evaluator scores. It is NOT a substitute for
independent verification of raw SEC filings or financial calculations.

AAI 520 student contribution: Evaluation Lead. Disclose AI assistance in the
project documentation as required by the course instructions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from src.agents.schemas import Draft, Evaluation, Finding


@dataclass(frozen=True)
class QualityPolicy:
    """Minimum score requirements and revision budget.

    max_evaluations counts the initial draft and all evaluated revisions.
    min_revisions counts optimizer calls, not evaluator calls.
    """

    min_average: float = 4.0
    min_accuracy: int = 4
    min_evidence: int = 4
    min_revisions: int = 1
    max_evaluations: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.min_average <= 5:
            raise ValueError("min_average must be between 1 and 5")
        if not 1 <= self.min_accuracy <= 5:
            raise ValueError("min_accuracy must be between 1 and 5")
        if not 1 <= self.min_evidence <= 5:
            raise ValueError("min_evidence must be between 1 and 5")
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be at least 1")
        if not 0 <= self.min_revisions < self.max_evaluations:
            raise ValueError("min_revisions must be below max_evaluations")

    def failures(self, evaluation: Evaluation) -> tuple[str, ...]:
        """Return explicit reasons an evaluator score does not pass."""
        reasons = []
        if evaluation.average < self.min_average:
            reasons.append("average_below_threshold")
        if evaluation.accuracy < self.min_accuracy:
            reasons.append("accuracy_below_threshold")
        if evaluation.evidence < self.min_evidence:
            reasons.append("evidence_below_threshold")
        return tuple(reasons)


@dataclass(frozen=True)
class Candidate:
    """Snapshot of a draft and its quality assessment at one iteration."""

    iteration: int
    draft: Draft
    evaluation: Evaluation
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class EvaluationOutcome:
    """Final choice, with explicit distinction between passed and exhausted."""

    draft: Draft
    evaluation: Evaluation
    selected_iteration: int
    accepted: bool
    stop_reason: str
    history: tuple[Candidate, ...]

    def rows(self) -> list[dict]:
        """Simple structured output for notebooks, tables, and charts."""
        return [
            {
                "iteration": c.iteration,
                "average": c.evaluation.average,
                "accuracy": c.evaluation.accuracy,
                "evidence": c.evaluation.evidence,
                "passed": c.passed,
                "failures": list(c.failures),
                "selected": c.iteration == self.selected_iteration,
            }
            for c in self.history
        ]


@dataclass
class EvaluationSession:
    """Tracks revisions and selects the best candidate without losing work."""

    policy: QualityPolicy = field(default_factory=QualityPolicy)
    candidates: list[Candidate] = field(default_factory=list)

    def record(self, draft: Draft, evaluation: Evaluation) -> Candidate:
        """Record an evaluated draft, copying to prevent mutation."""
        if len(self.candidates) >= self.policy.max_evaluations:
            raise RuntimeError("evaluation budget exhausted")
        if self.candidates and draft.ticker != self.candidates[0].draft.ticker:
            raise ValueError("all candidate drafts must use the same ticker")
        candidate = Candidate(
            iteration=len(self.candidates),
            draft=draft.model_copy(deep=True),
            evaluation=evaluation.model_copy(deep=True),
            failures=self.policy.failures(evaluation),
        )
        self.candidates.append(candidate)
        return candidate

    def should_refine(self) -> bool:
        """Apply revision requirements, acceptance rules, and the budget."""
        if not self.candidates:
            raise ValueError("record at least one evaluation first")
        count = len(self.candidates)
        if count >= self.policy.max_evaluations:
            return False
        if count <= self.policy.min_revisions:
            return True
        return not any(candidate.passed for candidate in self.candidates)

    def finish(self) -> EvaluationOutcome:
        """Select the highest scoring passing draft, or flag no valid draft."""
        if not self.candidates:
            raise ValueError("cannot finish before any evaluation")
        if self.should_refine():
            raise RuntimeError("evaluation loop has not completed")

        passing = [c for c in self.candidates if c.passed]
        if passing:
            # Highest average among accepted drafts, then accuracy/evidence.
            # max() keeps the earliest candidate on an exact tie.
            selected = max(
                passing,
                key=lambda c: (
                    c.evaluation.average,
                    c.evaluation.accuracy,
                    c.evaluation.evidence,
                ),
            )
        else:
            # No output passed: show the least deficient draft, but never
            # misrepresent it as a passing final report.
            selected = max(
                self.candidates,
                key=lambda c: (
                    -len(c.failures),
                    c.evaluation.accuracy,
                    c.evaluation.evidence,
                    c.evaluation.average,
                ),
            )
        return EvaluationOutcome(
            draft=selected.draft.model_copy(deep=True),
            evaluation=selected.evaluation.model_copy(deep=True),
            selected_iteration=selected.iteration,
            accepted=bool(passing),
            stop_reason=(
                "quality_gate_met" if passing else "max_evaluations_reached"
            ),
            history=tuple(self.candidates),
        )


def run_evaluation_cycle(
    initial_draft: Draft,
    findings: list[Finding],
    *,
    evaluate_fn: Callable[[Draft, list[Finding]], Evaluation],
    optimize_fn: Callable[[Draft, Evaluation, list[Finding]], Draft],
    policy: QualityPolicy | None = None,
) -> EvaluationOutcome:
    """Run evaluate/optimize with injected functions, then select best draft.

    For the live system, pass ev.evaluate and ev.optimize. Unit tests can
    inject deterministic functions so no external APIs are needed.
    """
    session = EvaluationSession(policy=policy or QualityPolicy())
    current = initial_draft.model_copy(deep=True)
    while True:
        evaluation = evaluate_fn(current, findings)
        session.record(current, evaluation)
        if not session.should_refine():
            return session.finish()
        current = optimize_fn(current, evaluation, findings)
        if current.ticker != initial_draft.ticker:
            raise ValueError("optimizer changed the stock ticker")
