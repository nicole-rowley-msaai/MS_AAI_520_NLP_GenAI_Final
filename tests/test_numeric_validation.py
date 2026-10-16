"""Offline financial-claim validator tests using synthetic records."""

from decimal import Decimal

import pytest

from src.agents.numeric_validation import (
    audit_numeric_claims,
    extract_reported_claims,
)
from src.agents.schemas import Draft, Finding


def draft(text: str) -> Draft:
    return Draft(
        ticker="NVDA",
        thesis="Research summary",
        bull_case=["Strong demand"],
        bear_case=["Competition"],
        fundamentals=text,
        news_and_catalysts="No additional claims",
        market_and_macro="No additional claims",
        conclusion="Review this evidence",
    )


def findings(*figures: str) -> list[Finding]:
    return [
        Finding(
            route="earnings",
            summary="Statement of reported figures",
            key_figures=list(figures),
            risks=[],
            sources=["synthetic_fixture"],
        )
    ]


def check(text: str, *figures: str):
    return audit_numeric_claims(draft(text), findings(*figures))


def test_match_metric_period_value():
    result = check("Revenue FY2025: $100B", "Revenue FY2025: $100B")
    assert len(result.checks) == 1
    assert result.checks[0].status == "supported"
    assert result.passed is True


def test_detects_conflicting_value_for_same_metric_and_period():
    result = check("Revenue FY2025: $150B", "Revenue FY2025: $100B")
    assert result.checks[0].status == "contradicted"
    assert result.checks[0].expected == Decimal(100000000000)
    assert result.passed is False


def test_comparable_units_convert_billion_and_million():
    result = check("Revenue FY2025: $1.5B", "Revenue FY2025: $1500M")
    assert result.checks[0].status == "supported"


def test_wrong_period_is_not_falsely_counted_as_supported():
    result = check("Revenue FY2024: $150B", "Revenue FY2025: $150B")
    assert result.checks[0].status == "unverified"


def test_wrong_metric_is_not_falsely_counted_as_supported():
    result = check("Net income FY2025: $100B", "Revenue FY2025: $100B")
    assert result.checks[0].status == "unverified"


def test_unverified_does_not_mean_verified():
    result = check("Revenue FY2025: $100B", "Revenue FY2024: $100B")
    assert len(result.unverified) == 1
    assert result.passed is False  # missing period match needs review


def test_conflicting_sources_are_marked_unverified():
    result = check(
        "Revenue FY2025: $100B",
        "Revenue FY2025: $100B",
        "Revenue FY2025: $120B",
    )
    assert result.checks[0].status == "unverified"
    assert result.checks[0].reason == "conflicting_values_in_findings"


def test_percentage_metric_matches():
    result = check("Operating margin FY2025: 25%", "Operating margin FY2025: 25%")
    assert result.checks[0].status == "supported"


def test_percentage_metric_mismatch():
    result = check("Operating margin FY2025: 30%", "Operating margin FY2025: 25%")
    assert result.checks[0].status == "contradicted"


def test_explicit_correct_derived_growth():
    result = check("Revenue increased from $100B to $120B, a calculated increase of 20%")
    derived = [c for c in result.checks if c.kind == "derived_growth"]
    assert len(derived) == 1
    assert derived[0].status == "supported"
    assert derived[0].expected == Decimal(20)


def test_explicit_wrong_derived_growth_is_not_exempt():
    result = check("Revenue increased from $100B to $120B, a calculated increase of 35%")
    derived = [c for c in result.checks if c.kind == "derived_growth"]
    assert derived[0].status == "contradicted"
    assert derived[0].reason == "incorrect_percentage_change"


def test_derived_growth_normalizes_units():
    result = check("Revenue increased from $1B to $1250M, growth of 25%")
    derived = [c for c in result.checks if c.kind == "derived_growth"]
    assert derived[0].status == "supported"


def test_decline_uses_negative_growth():
    result = check("Revenue declined from $120B to $90B, a calculated change of -25%")
    derived = [c for c in result.checks if c.kind == "derived_growth"]
    assert derived[0].status == "supported"


def test_zero_denominator_is_marked_unverified():
    result = check("Revenue grew from $0B to $20B, growth of 50%")
    derived = [c for c in result.checks if c.kind == "derived_growth"]
    assert derived[0].status == "unverified"


def test_no_recognized_statements_means_no_claim_of_verification():
    result = check("Valuation looks expensive", "Revenue FY2025: $100B")
    assert result.checks == ()
    assert result.passed is False


def test_extracts_fcf_synonym():
    result = extract_reported_claims("FCF FY2025: $96.7B")
    assert len(result) == 1
    assert result[0].metric == "free cash flow"


def test_audit_rows_have_explicit_outcomes():
    result = check("Revenue FY2025: $150B", "Revenue FY2025: $100B")
    row = result.rows()[0]
    assert row["kind"] == "reported"
    assert row["status"] == "contradicted"
    assert row["observed"] == "150000000000"


def test_negative_tolerance_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        audit_numeric_claims(
            draft("Revenue FY2025: $100B"),
            findings("Revenue FY2025: $100B"),
            growth_tolerance_pct_points=Decimal("-0.5"),
        )


def test_rounded_growth_within_half_percentage_point():
    result = check("Revenue increased from $100B to $117B, growth of 17.2%")
    derived = [c for c in result.checks if c.kind == "derived_growth"]
    assert derived[0].status == "supported"


def test_catch_negative_revenue_mismatch():
    result = check("Net income FY2025: -$10M", "Net income FY2025: $10M")
    assert result.checks[0].status == "contradicted"


def test_equal_number_different_unit_is_unverified():
    result = check("Operating margin FY2025: 25%", "Operating margin FY2025: $25M")
    assert result.checks[0].status == "unverified"
