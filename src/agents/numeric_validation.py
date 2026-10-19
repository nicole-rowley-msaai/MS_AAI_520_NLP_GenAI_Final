"""Conservative numerical cross-checks for equity research briefs.

Checks explicit metric/period/value statements against specialist Finding data,
plus explicitly stated percentage changes computed from two endpoints.

This is NOT a comprehensive fact checker: it does not fetch authoritative
source documents, interpret free-form finance prose, or validate valuations.
Claims outside the supported syntax receive no judgment and must not be
considered verified. Source disagreements and missing evidence are marked
unverified, not labeled false.

AI assistance was used to develop this code and synthetic test cases.
The student must review, modify as appropriate, and disclose its contribution.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal
import re

from src.agents.schemas import Draft, Finding
from src.agents.synthesizer import render

Status = Literal["supported", "contradicted", "unverified"]
Kind = Literal["reported", "derived_growth"]

# Intentionally narrow patterns: opaque or ambiguous prose is not verified.
_METRIC_NAMES = (
    "free cash flow",
    "operating margin",
    "gross margin",
    "net margin",
    "net income",
    "total debt",
    "revenue",
    "fcf",
    "eps",
)
_METRIC = "(?:" + "|".join(re.escape(x) for x in _METRIC_NAMES) + ")"
_PERIOD = r"(?:FY\s?20\d{2}|20\d{2}|TTM)"
_AMOUNT = (
    r"(?P<amount>-?\$\s?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:billion|million|thousand|[BMK])?|-?\d[\d,]*(?:\.\d+)?\s?%)"
)
_REPORTED = re.compile(
    rf"(?<![A-Za-z])(?:(?P<before>{_PERIOD})\s+)?"
    rf"(?P<metric>{_METRIC})\s*"
    rf"(?:\(\s*(?P<paren>{_PERIOD})\s*\)|(?P<after>{_PERIOD}))?\s*"
    rf"(?:\:|=|\b(?:was|is|of)\b)\s*{_AMOUNT}",
    re.IGNORECASE,
)
_VALUE = r"-?\$\s?\d[\d,]*(?:\.\d+)?\s*(?:billion|million|thousand|[BMK])?"
_GROWTH = re.compile(
    rf"(?<![A-Za-z])(?P<metric>{_METRIC})\s+"
    rf"(?:(?:increased|grew|rose|declined|fell|dropped|changed)\s+)?"
    rf"from\s+(?P<old>{_VALUE})\s+to\s+(?P<new>{_VALUE})"
    rf"(?P<tail>(?:\d+\.\d+|[^.\n;]){{0,110}})",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"(?<![\w.])(?P<percent>-?\d[\d,]*(?:\.\d+)?)\s?%")
_VALUE_PARTS = re.compile(
    r"^(?P<sign>-?)\$\s?(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>billion|million|thousand|[BMK])?$",
    re.IGNORECASE,
)
_PCT_PARTS = re.compile(r"^(?P<num>-?\d[\d,]*(?:\.\d+)?)\s?%$")
_SCALE = {
    None: Decimal(1),
    "": Decimal(1),
    "k": Decimal(1000),
    "thousand": Decimal(1000),
    "m": Decimal(1000000),
    "million": Decimal(1000000),
    "b": Decimal(1000000000),
    "billion": Decimal(1000000000),
}
_ALIAS = {"fcf": "free cash flow"}


@dataclass(frozen=True)
class FinancialClaim:
    """One explicit numerical statement parsed from text."""

    metric: str
    period: str | None
    value: Decimal
    unit: Literal["USD", "percent"]
    excerpt: str
    # Half of the last written digit, in the same units as ``value``:
    # "$215.9B" can stand for anything from $215.85B to $215.95B.
    tolerance: Decimal = Decimal(0)


@dataclass(frozen=True)
class NumericCheck:
    """A review decision based only on explicitly recognized evidence."""

    kind: Kind
    status: Status
    metric: str
    period: str | None
    observed: Decimal
    expected: Decimal | None
    reason: str
    excerpt: str


@dataclass(frozen=True)
class NumericAudit:
    """Results only for syntax this limited parser recognized."""

    checks: tuple[NumericCheck, ...]

    @property
    def contradictions(self) -> tuple[NumericCheck, ...]:
        return tuple(c for c in self.checks if c.status == "contradicted")

    @property
    def unverified(self) -> tuple[NumericCheck, ...]:
        return tuple(c for c in self.checks if c.status == "unverified")

    @property
    def passed(self) -> bool:
        """All recognized checks passed, with at least one check performed.

        This does NOT establish that unrecognized claims or raw data are true.
        """
        return bool(self.checks) and all(c.status == "supported" for c in self.checks)

    def rows(self) -> list[dict]:
        """Rows suitable for a notebook table or a CSV export."""
        return [
            {
                "kind": c.kind,
                "status": c.status,
                "metric": c.metric,
                "period": c.period,
                "observed": str(c.observed),
                "expected": str(c.expected) if c.expected is not None else None,
                "reason": c.reason,
                "excerpt": c.excerpt,
            }
            for c in self.checks
        ]


def _amount(text: str) -> tuple[Decimal, Literal["USD", "percent"]] | None:
    """Normalize supported monetary units and percentage values."""
    text = text.strip()
    pct = _PCT_PARTS.fullmatch(text)
    if pct:
        try:
            return Decimal(pct["num"].replace(",", "")), "percent"
        except InvalidOperation:
            return None
    money = _VALUE_PARTS.fullmatch(text)
    if money:
        try:
            sign = Decimal(-1) if money["sign"] else Decimal(1)
            number = Decimal(money["num"].replace(",", ""))
            scale = _SCALE[money["scale"].lower() if money["scale"] else None]
            return sign * number * scale, "USD"
        except (InvalidOperation, KeyError):
            return None
    return None


def _rounding_tolerance(text: str) -> Decimal:
    """Half a unit in the last written digit, scaled to the stated unit.

    "$215.9B" -> $0.05B, "$96,676M" -> $0.5M, "25%" -> 0.5 points.
    A brief that rounds a source figure is consistent with it, not a
    contradiction of it.
    """
    text = text.strip()
    pct = _PCT_PARTS.fullmatch(text)
    if pct:
        num, scale = pct["num"], Decimal(1)
    else:
        money = _VALUE_PARTS.fullmatch(text)
        if not money:
            return Decimal(0)
        num = money["num"]
        scale = _SCALE[money["scale"].lower() if money["scale"] else None]
    num = num.replace(",", "").lstrip("-")
    decimals = len(num.split(".", 1)[1]) if "." in num else 0
    return Decimal(5) * Decimal(10) ** -(decimals + 1) * scale


def _consistent(a: FinancialClaim, b: FinancialClaim) -> bool:
    """True when two figures can be the same number at different rounding."""
    return abs(a.value - b.value) <= max(a.tolerance, b.tolerance)


def _distinct_values(sources: list[FinancialClaim]) -> list[FinancialClaim]:
    """Collapse source figures that are the same number written differently.

    "$215.9B" in a summary and "$215,938M" in key figures are one value,
    not conflicting evidence. The most precise representative is kept.
    """
    groups: list[FinancialClaim] = []
    for source in sorted(sources, key=lambda c: c.tolerance):
        if not any(_consistent(source, kept) for kept in groups):
            groups.append(source)
    return groups


def _period(text: str | None) -> str | None:
    if not text:
        return None
    value = text.upper().replace(" ", "")
    # FY2025 and bare 2025 are treated as different precision levels.
    return value


def extract_reported_claims(text: str) -> tuple[FinancialClaim, ...]:
    """Extract a constrained form such as 'Revenue FY2025: $1.5B'.

    Supported syntax has an explicit metric, optional period, a separator
    (colon, equals, was, is, of), and a currency or percent value. This
    intentionally omits many real research statements.
    """
    found: list[FinancialClaim] = []
    for match in _REPORTED.finditer(text):
        parsed = _amount(match["amount"])
        if parsed is None:
            continue
        metric = match["metric"].lower()
        found.append(
            FinancialClaim(
                metric=_ALIAS.get(metric, metric),
                period=_period(match["before"] or match["paren"] or match["after"]),
                value=parsed[0],
                unit=parsed[1],
                excerpt=match.group(),
                tolerance=_rounding_tolerance(match["amount"]),
            )
        )
    return tuple(found)


def _reported_checks(
    draft_text: str, findings: list[Finding]
) -> list[NumericCheck]:
    source_text = "\n".join(
        part
        for finding in findings
        for part in [finding.summary, *finding.key_figures]
    )
    source_claims = extract_reported_claims(source_text)
    checks: list[NumericCheck] = []
    for claim in extract_reported_claims(draft_text):
        same_metric = [
            source
            for source in source_claims
            if source.metric == claim.metric and source.unit == claim.unit
        ]
        comparable = [s for s in same_metric if s.period == claim.period]
        candidates = _distinct_values(comparable)
        expected: Decimal | None = None
        if not candidates and claim.period is None:
            # The brief did not name a period. Accept a match with any
            # period's figure, but never call it a contradiction: without a
            # period there is nothing definite to contradict.
            match = next(
                (s for s in same_metric if _consistent(claim, s)), None
            )
            if match is not None:
                status: Status = "supported"
                expected = match.value
                reason = "matching_value_period_not_stated"
            else:
                status = "unverified"
                reason = "no_comparable_metric_period_unit_in_findings"
        elif not candidates:
            status = "unverified"
            reason = "no_comparable_metric_period_unit_in_findings"
        elif len(candidates) > 1:
            status = "unverified"
            reason = "conflicting_values_in_findings"
        else:
            expected = candidates[0].value
            if _consistent(claim, candidates[0]):
                status = "supported"
                reason = (
                    "matching_metric_period_and_value"
                    if claim.value == expected
                    else "matching_within_rounding"
                )
            else:
                status = "contradicted"
                reason = "value_differs_from_findings"
        checks.append(
            NumericCheck(
                kind="reported",
                status=status,
                metric=claim.metric,
                period=claim.period,
                observed=claim.value,
                expected=expected,
                reason=reason,
                excerpt=claim.excerpt,
            )
        )
    return checks


def _derived_checks(
    draft_text: str, tolerance_pct_points: Decimal
) -> list[NumericCheck]:
    checks: list[NumericCheck] = []
    for match in _GROWTH.finditer(draft_text):
        pct_match = _PERCENT.search(match["tail"])
        if not pct_match:
            continue
        old = _amount(match["old"])
        new = _amount(match["new"])
        if not old or not new or old[1] != new[1] or old[1] != "USD":
            continue
        actual = Decimal(pct_match["percent"].replace(",", ""))
        if old[0] == 0:
            checks.append(
                NumericCheck(
                    kind="derived_growth",
                    status="unverified",
                    metric=_ALIAS.get(match["metric"].lower(), match["metric"].lower()),
                    period=None,
                    observed=actual,
                    expected=None,
                    reason="growth_undefined_for_zero_base",
                    excerpt=match.group(),
                )
            )
            continue
        expected = (new[0] - old[0]) / abs(old[0]) * Decimal(100)
        passed = abs(actual - expected) <= tolerance_pct_points
        checks.append(
            NumericCheck(
                kind="derived_growth",
                status="supported" if passed else "contradicted",
                metric=_ALIAS.get(match["metric"].lower(), match["metric"].lower()),
                period=None,
                observed=actual,
                expected=expected,
                reason=(
                    "arithmetic_within_tolerance"
                    if passed
                    else "incorrect_percentage_change"
                ),
                excerpt=match.group(),
            )
        )
    return checks


def audit_numeric_claims(
    draft: Draft,
    findings: list[Finding],
    *,
    growth_tolerance_pct_points: Decimal = Decimal("0.5"),
) -> NumericAudit:
    """Audit recognized report figures against specialist findings.

    Only exact metric, period, and unit matches count as source support.
    No claim is made about unrecognized prose or ultimate source truth.
    A computed growth percentage is checked for arithmetic only; its
    endpoint values must still be verified separately.
    """
    if growth_tolerance_pct_points < 0:
        raise ValueError("growth tolerance must be nonnegative")
    text = render(draft)
    return NumericAudit(
        checks=tuple(
            _reported_checks(text, findings)
            + _derived_checks(text, growth_tolerance_pct_points)
        )
    )
