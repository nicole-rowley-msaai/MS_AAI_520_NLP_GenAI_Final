"""Offline tests for evidence-matched baseline comparison."""

import pytest

from src.agents.schemas import Draft, Evaluation, Finding
from src.evaluation_lead_benchmarks import (
    PairedScores,
    compare_evidence_matched,
    evidence_fingerprint,
    matched_baseline_brief,
)


def sample_draft(ticker="NVDA", thesis="one pass"):
    return Draft(
        ticker=ticker,
        thesis=thesis,
        bull_case=["Revenue supports the thesis"],
        bear_case=["Valuation risk"],
        fundamentals="Revenue FY2025: $100B",
        news_and_catalysts="No verified catalyst",
        market_and_macro="Macro context",
        conclusion="Research only",
    )


def sample_finding(summary="Revenue FY2025: $100B"):
    return Finding(
        route="earnings",
        summary=summary,
        key_figures=["Revenue FY2025: $100B"],
        risks=["Valuation risk"],
        sources=["get_financials"],
    )


def score(value):
    return Evaluation(
        completeness=value,
        evidence=value,
        accuracy=value,
        balance=value,
        clarity=value,
        critiques=[],
    )


def test_matched_baseline_calls_structured_once(monkeypatch):
    calls = []

    def fake_structured(model, system, user, schema):
        calls.append((model, system, user, schema))
        return sample_draft()

    monkeypatch.setattr(
        "src.evaluation_lead_benchmarks.structured", fake_structured
    )
    out = matched_baseline_brief("nvda", [sample_finding()], "writer")
    assert out.ticker == "NVDA"
    assert len(calls) == 1
    assert "Revenue FY2025: $100B" in calls[0][2]
    assert "only the provided specialist findings" in calls[0][1]
    assert calls[0][3] is Draft


def test_no_findings_cannot_be_benchmarked():
    with pytest.raises(ValueError, match="nonempty"):
        matched_baseline_brief("NVDA", [])
    with pytest.raises(ValueError, match="nonempty"):
        compare_evidence_matched(
            "NVDA", [], sample_draft(), judge=lambda *x: score(3)
        )


def test_fingerprint_deterministic():
    assert evidence_fingerprint([sample_finding()]) == evidence_fingerprint(
        [sample_finding()]
    )
    assert len(evidence_fingerprint([sample_finding()])) == 64


def test_fingerprint_changes_with_evidence():
    assert evidence_fingerprint([sample_finding()]) != evidence_fingerprint(
        [sample_finding("Revenue FY2025: $200B")]
    )


def test_evidence_matched_comparison_is_paired_and_scored_once_each():
    calls = []
    findings = [sample_finding()]
    agent = sample_draft(thesis="reviewed")

    def baseline(ticker, evidence):
        calls.append(("baseline", ticker, evidence[0].summary))
        return sample_draft(thesis="baseline")

    def judge(draft, evidence):
        calls.append(("judge", draft.thesis, evidence[0].summary))
        return score(4 if draft.thesis == "reviewed" else 2)

    result = compare_evidence_matched(
        "NVDA", findings, agent, make_baseline=baseline, judge=judge
    )
    assert len(calls) == 3
    assert calls[1][2] == calls[2][2]
    assert result.average_delta == 2
    assert [r["candidate"] for r in result.rows()] == [
        "matched_baseline", "reviewed_agent"
    ]


def test_no_fake_improvement_if_scores_equal():
    result = compare_evidence_matched(
        "NVDA", [sample_finding()], sample_draft(),
        make_baseline=lambda *args: sample_draft(),
        judge=lambda *args: score(4),
    )
    assert result.average_delta == 0


def test_negative_delta_reports_regression():
    result = compare_evidence_matched(
        "NVDA", [sample_finding()], sample_draft(thesis="reviewed"),
        make_baseline=lambda *args: sample_draft(thesis="baseline"),
        judge=lambda draft, _: score(2 if draft.thesis == "reviewed" else 5),
    )
    assert result.average_delta == -3


def test_wrong_agent_ticker_rejected():
    with pytest.raises(ValueError, match="reviewed"):
        compare_evidence_matched(
            "JPM", [sample_finding()], sample_draft("NVDA"),
            judge=lambda *args: score(4),
        )


def test_wrong_baseline_ticker_rejected():
    with pytest.raises(ValueError, match="baseline"):
        compare_evidence_matched(
            "NVDA", [sample_finding()], sample_draft(),
            make_baseline=lambda *args: sample_draft("JPM"),
            judge=lambda *args: score(4),
        )


def test_mutating_callback_cannot_change_original_findings():
    inputs = [sample_finding()]

    def baseline(ticker, f):
        f[0].summary = "changed during baseline"
        return sample_draft()

    def judge(draft, f):
        f[0].summary = "changed during judge"
        return score(4)

    result = compare_evidence_matched(
        "NVDA", inputs, sample_draft(), make_baseline=baseline, judge=judge
    )
    assert inputs[0].summary == "Revenue FY2025: $100B"
    assert result.evidence_sha256 == evidence_fingerprint(inputs)


def test_rows_include_all_five_criteria_and_same_fingerprint():
    result = compare_evidence_matched(
        "NVDA", [sample_finding()], sample_draft(),
        make_baseline=lambda *args: sample_draft(),
        judge=lambda *args: score(3),
    )
    rows = result.rows()
    assert len(rows) == 2
    assert rows[0]["evidence_sha256"] == rows[1]["evidence_sha256"]
    for key in ("completeness", "evidence", "accuracy", "balance", "clarity"):
        assert key in rows[0]


def test_does_not_implicitly_call_llm(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("unexpected external model call")

    monkeypatch.setattr("src.evaluation_lead_benchmarks.structured", fail)
    compare_evidence_matched(
        "NVDA", [sample_finding()], sample_draft(),
        make_baseline=lambda *args: sample_draft(),
        judge=lambda *args: score(4),
    )


def test_rows_reflect_numeric_scores():
    result = PairedScores(
        ticker="XOM", evidence_sha256="0" * 64,
        baseline=score(3), reviewed=score(5),
    )
    assert result.rows()[0]["average"] == 3
    assert result.rows()[1]["average"] == 5
    assert result.average_delta == 2
