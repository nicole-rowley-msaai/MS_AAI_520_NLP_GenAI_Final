"""Data verification specialist: findings checked against raw tool data."""

from src.agents.schemas import Finding
from src.agents.verifier import (
    annotate,
    check_finding,
    summary_rows,
    verify_findings,
)


def years(new, old):
    return {"2026-01-25": new, "2025-01-26": old}


RAW = {
    "earnings": {
        "step 1: get_key_stats": {
            "marketCap": 5170596937728,
            "trailingPE": 27.07,
            "profitMargins": 0.5561,
            "targetMeanPrice": 328.72,
            "numberOfAnalystOpinions": 59,
        },
        "step 2: get_financials": {
            "Total Revenue": years(215938e6, 130497e6),
            "Operating Income": years(130387e6, 81453e6),
            "Net Income": years(120067e6, 72880e6),
            "Free Cash Flow": years(96676e6, 60853e6),
        },
        "step 3: search_filings": [
            {"text": "We recorded a $4.5 billion H20 inventory charge."}
        ],
    },
    "market": {
        "step 4: get_price_history": {
            "last_close": 214.13,
            "return_1m": -0.0238,
            "vol_annualized": 0.3727,
        },
        "step 5: get_macro_snapshot": [
            {"series": "DGS10", "latest": 5.10, "one_year_ago": 4.05}
        ],
    },
}
POOLED = {f"{r} / {k}": v for r, d in RAW.items() for k, v in d.items()}


def finding(text, route="earnings", figures=()):
    return Finding(
        route=route,
        summary=text,
        key_figures=list(figures),
        risks=[],
        sources=[],
    )


def test_rounded_figures_match_raw_values():
    check = check_finding(
        finding("Revenue was $215.9B and free cash flow $96.7B."), POOLED
    )
    assert check.confirmed == ["$215.9B", "$96.7B"]
    assert not check.not_found and not check.contradicted


def test_computed_values_are_recognized():
    check = check_finding(
        finding(
            "Revenue grew 65.5% YoY with a 60.4% operating margin; profit "
            "margin 55.61%; market cap $5.17T; upside to target 53.5%; "
            "10-year yield up 105 bps; 1-month return 2.38%."
        ),
        POOLED,
    )
    assert check.not_found == []
    assert check.checked == 7


def test_numbers_inside_filing_text_count_as_raw_data():
    check = check_finding(finding("A $4.5B H20 charge."), POOLED)
    assert check.confirmed == ["$4.5B"]


def test_wrong_value_for_explicit_metric_and_year_is_contradicted():
    check = check_finding(finding("Net income FY2026: $130.1B"), POOLED)
    assert len(check.contradicted) == 1
    assert check.contradicted[0]["expected"] == 120067e6
    assert check.confirmed == []  # the wrong figure is not also "confirmed"


def test_figure_absent_from_data_is_not_found_not_contradicted():
    check = check_finding(finding("Gross margin was 71.1%."), POOLED)
    assert check.not_found == ["71.1%"]
    assert check.contradicted == []


def test_plain_integers_and_years_are_not_treated_as_figures():
    check = check_finding(
        finding("59 analysts cover FY2026; RTX 5090 shortages; 10-K."), POOLED
    )
    assert check.checked == 0


def test_news_findings_are_marked_not_checked():
    check = check_finding(finding("Buyback of $150bn", route="news"), POOLED)
    assert check.status == "not_checked"
    assert "article text" in check.note


def test_annotate_appends_correction_and_attribution_lines():
    f = finding("Net income FY2026: $130.1B. Gross margin 71.1%.")
    out = annotate(f, check_finding(f, POOLED))
    added = out.key_figures[-3:]
    assert added[0].startswith("VERIFICATION: 0 of 2 figures")
    assert "CORRECTION: net income FY2026 is $120.067B" in added[1]
    assert added[2].endswith("attribute or omit: 71.1%")
    assert f.key_figures == []  # the specialist's own finding is unchanged


def test_verify_pools_raw_data_across_specialists():
    # Upside to the analyst target needs the market specialist's last close.
    f = finding("Target $328.72 implies 53.5% upside.")
    _, checks = verify_findings([f], RAW)
    assert checks[0].not_found == []


def test_summary_rows_describe_each_finding():
    f = finding("Net income FY2026: $130.1B; revenue $215.9B")
    rows = summary_rows(verify_findings([f], RAW)[1])
    assert rows[0]["confirmed"] == 1 and rows[0]["contradicted"] == 1
    assert "$130.1B -> $120.067B" in rows[0]["corrections"]
