"""Evidence-matched, paired comparison of one-pass and reviewed reports.

Both candidates are evaluated using the same frozen Finding list and evaluator
callback. It removes differences in *access to evidence*,
but it does not isolate planning or prove improved real-world investments.

No API calls occur unless a caller explicitly invokes the live baseline or
passes LLM-backed callbacks. Disclose AI assistance as required by the course.
"""

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json

from src.agents.schemas import CRITERIA, Draft, Evaluation, Finding
from src.agents.synthesizer import findings_text
from src.config import COMPANIES, MODELS
from src.llm import structured

MATCHED_BASELINE_SYSTEM = (
    "You are an equity analyst writing a research brief for a portfolio "
    "manager. Use only the provided specialist findings as evidence. "
    "Do not use outside knowledge or retrieve other data. Give a thesis, "
    "balanced bull and bear cases, fundamentals, news and catalysts, "
    "market and macro context, and a conclusion. This is a single-pass "
    "baseline without feedback, revisions, or tool use. "
    "Do not invent numerical figures."
)


def matched_baseline_brief(
    ticker: str, findings: list[Finding], model: str = MODELS["writer"]
) -> Draft:
    """Generate a one-pass baseline using exactly the given findings.

    This intentionally performs one LLM request and no retrieval or revision.
    """
    ticker = ticker.upper()
    if not findings:
        raise ValueError("a matched baseline requires nonempty findings")
    prompt = (
        f"Ticker: {ticker} ({COMPANIES.get(ticker, ticker)})\n\n"
        f"FINDINGS\n{findings_text(findings)}"
    )
    draft = structured(model, MATCHED_BASELINE_SYSTEM, prompt, Draft)
    draft.ticker = ticker
    return draft


def evidence_fingerprint(findings: list[Finding]) -> str:
    """Hash serialized findings to identify which evidence snapshot was used.

    This fingerprint is for reproducibility, not proof of data authenticity.
    """
    payload = [finding.model_dump(mode="json") for finding in findings]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PairedScores:
    """Snapshots of two grader outputs for one ticker and evidence set."""

    ticker: str
    evidence_sha256: str
    baseline: Evaluation
    reviewed: Evaluation

    @property
    def average_delta(self) -> float:
        return round(self.reviewed.average - self.baseline.average, 2)

    def rows(self) -> list[dict]:
        """Long-format rows for tables and charting."""
        rows = []
        for label, evaluation in (
            ("matched_baseline", self.baseline),
            ("reviewed_agent", self.reviewed),
        ):
            rows.append(
                {
                    "ticker": self.ticker,
                    "candidate": label,
                    "evidence_sha256": self.evidence_sha256,
                    "average": evaluation.average,
                    **{name: getattr(evaluation, name) for name in CRITERIA},
                }
            )
        return rows


def compare_evidence_matched(
    ticker: str,
    findings: list[Finding],
    reviewed_draft: Draft,
    *,
    make_baseline: Callable[[str, list[Finding]], Draft] = (
        matched_baseline_brief
    ),
    judge: Callable[[Draft, list[Finding]], Evaluation],
) -> PairedScores:
    """Generate a matched baseline, then grade both against equal evidence.

    `reviewed_draft` must already have been produced separately (e.g. after the
    agent's evaluation/optimizer cycle). Both drafts are independently judged
    using the same callback and an immutable copy of the findings snapshot.
    The grader may have model bias and stochastic variability.
    """
    ticker = ticker.upper()
    if not findings:
        raise ValueError("comparison needs nonempty findings")
    if reviewed_draft.ticker.upper() != ticker:
        raise ValueError("reviewed draft ticker mismatch")

    snapshot = [finding.model_copy(deep=True) for finding in findings]
    fingerprint = evidence_fingerprint(snapshot)
    baseline = make_baseline(
        ticker, [finding.model_copy(deep=True) for finding in snapshot]
    )
    if baseline.ticker.upper() != ticker:
        raise ValueError("baseline draft ticker mismatch")

    bscore = judge(
        baseline.model_copy(deep=True),
        [finding.model_copy(deep=True) for finding in snapshot],
    )
    rscore = judge(
        reviewed_draft.model_copy(deep=True),
        [finding.model_copy(deep=True) for finding in snapshot],
    )
    if evidence_fingerprint(snapshot) != fingerprint:
        raise RuntimeError("the evidence snapshot changed during comparison")
    return PairedScores(
        ticker=ticker,
        evidence_sha256=fingerprint,
        baseline=bscore.model_copy(deep=True),
        reviewed=rscore.model_copy(deep=True),
    )


def compare_saved_agent(
    ticker: str, findings: list[Finding], agent_draft: Draft
) -> PairedScores:
    """Live helper: one matched baseline and two evaluator model calls.

    Requires the user's configured model credentials and may incur charges.
    """
    from src.agents.evaluator import evaluate

    return compare_evidence_matched(
        ticker, findings, agent_draft, judge=evaluate
    )
