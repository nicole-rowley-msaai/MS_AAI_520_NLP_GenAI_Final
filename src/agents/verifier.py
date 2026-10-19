"""Data verification specialist: checks specialist findings against raw data.

Sits between the specialists and the synthesizer. The specialists turn
tool output into prose findings with an LLM; this agent re-reads the raw
tool output the specialists received and checks every financial figure
each finding states:

- **confirmed**: the figure equals a raw value from any tool in the run,
  or a value computed from raw values (year-over-year change, margin,
  percentage form of a ratio, macro change), at the precision it is
  written ("$215.9B" confirms $215,938M). A figure with a magnitude word
  also matches a raw value equal to its written number, for sources that
  already report in that scale (FRED real GDP 24408.011, in billions,
  confirms "$24,408.011B");
- **contradicted**: the finding names a metric and fiscal year the raw
  data has, and the value differs beyond rounding;
- **not found**: no raw or computed value matches. This is not proof of
  error; the figure may be derived in a way this agent does not compute.

News findings are summarized from article text, which this agent does
not hold, so they are reported as not checked.

The agent is rule-based, with no LLM, so its judgments are reproducible
and independent of the models that wrote the findings. It passes the
synthesizer copies of the findings with verification lines appended:
corrections for contradicted figures and a list of figures to attribute
or omit. Nothing in a specialist's own text is rewritten.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from src.agents.numeric_validation import extract_reported_claims
from src.agents.schemas import Finding

# A figure: optional sign and $, a number, then a scale or unit. Plain
# integers ("59 analysts", "RTX 5090") are counts or names, not figures,
# and are skipped unless they carry $, a unit, or a decimal point.
_FIGURE = re.compile(
    r"(?<![\w./])(?P<sign>[-−])?(?P<cur>\$)?\s?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<scale>trillion|billion|million|thousand|tn|bn|mn|bps|[TBMK%x])"
    r"(?![A-Za-z]))?"
)
_SCALE = {
    "t": 10**12,
    "tn": 10**12,
    "trillion": 10**12,
    "b": 10**9,
    "bn": 10**9,
    "billion": 10**9,
    "m": 10**6,
    "mn": 10**6,
    "million": 10**6,
    "k": 10**3,
    "thousand": 10**3,
}
_TRIVIAL = 10  # unit-less integers up to 10 carry no checkable claim

# Metric names the numeric parser recognizes -> raw financial line.
_LINE = {
    "revenue": "Total Revenue",
    "net income": "Net Income",
    "free cash flow": "Free Cash Flow",
    "total debt": "Total Debt",
    "eps": "Diluted EPS",
}
_MARGIN = {"operating margin": "Operating Income", "net margin": "Net Income"}


@dataclass(frozen=True)
class Figure:
    text: str
    value: float  # absolute value in base units ($, %, x or bps)
    unit: str  # "usd", "pct", "x", "bps" or "num"
    tolerance: float  # half of the last written digit, in base units
    # For figures with a magnitude word (T, B, M, K): the number as
    # written, before scaling, and its tolerance. Some sources already
    # report in a scale, e.g. FRED real GDP in billions (24408.011), which
    # a finding correctly writes as "$24,408.011B".
    written: float | None = None
    written_tolerance: float = 0.0


@dataclass
class FindingCheck:
    route: str
    status: str  # "checked" or "not_checked"
    confirmed: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    contradicted: list[dict] = field(default_factory=list)
    note: str = ""

    @property
    def checked(self) -> int:
        return (
            len(self.confirmed) + len(self.not_found) + len(self.contradicted)
        )


# Parsing ------------------------------------------------------------------


def _figures(text: str) -> list[Figure]:
    out: dict[tuple, Figure] = {}
    for m in _FIGURE.finditer(text or ""):
        num, scale, cur = m["num"], (m["scale"] or ""), m["cur"]
        if not (cur or scale or "." in num):
            continue
        try:
            number = float(num.replace(",", ""))
        except ValueError:
            continue
        if not (cur or scale) and number <= _TRIVIAL:
            continue
        decimals = len(num.split(".", 1)[1]) if "." in num else 0
        half = 0.5 * 10**-decimals
        key = scale.lower()
        if key in _SCALE:
            unit, factor = "usd" if cur else "num", _SCALE[key]
        elif key == "%":
            unit, factor = "pct", 1
        elif key == "x":
            unit, factor = "x", 1
        elif key == "bps":
            unit, factor = "bps", 1
        else:
            unit, factor = ("usd" if cur else "num"), 1
        fig = Figure(
            m.group().strip(),
            number * factor,
            unit,
            half * factor,
            written=number if factor != 1 else None,
            written_tolerance=half if factor != 1 else 0.0,
        )
        out.setdefault((unit, round(fig.value, 6)), fig)
    return list(out.values())


# Reference values from raw tool data ----------------------------------------


def _numbers(obj, sink: list[float]) -> None:
    """Every number in the raw data, including numbers inside text."""
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if obj == obj:  # skip NaN
            sink.append(float(obj))
    elif isinstance(obj, str):
        sink.extend(f.value for f in _figures(obj))
        sink.extend(
            float(n.replace(",", ""))
            for n in re.findall(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?", obj)
        )
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, sink)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers(v, sink)


def _years(series: dict) -> list[tuple[int, float]]:
    """{'2026-01-31': v, ...} -> [(2026, v), ...], newest first."""
    out = []
    for date, v in series.items():
        if isinstance(v, (int, float)) and v == v:
            try:
                out.append((int(str(date)[:4]), float(v)))
            except ValueError:
                continue
    return sorted(out, reverse=True)


def financial_lines(raw: dict) -> dict[str, dict[int, float]]:
    """Fiscal-year series for the summarized statement lines in raw data."""
    lines: dict[str, dict[int, float]] = {}
    for output in raw.values():
        if not isinstance(output, dict):
            continue
        for name, series in output.items():
            if isinstance(series, dict) and series:
                years = _years(series)
                if years and len(years) == len(series):
                    lines.setdefault(name, dict(years))
    return lines


def reference_values(raw: dict) -> list[float]:
    """Raw numbers plus values commonly computed from them."""
    refs: list[float] = []
    _numbers(raw, refs)
    refs += [v * 100 for v in list(refs) if 0 < abs(v) < 1.5]  # ratio -> %

    lines = financial_lines(raw)
    revenue = lines.get("Total Revenue", {})
    for series in lines.values():
        years = sorted(series.items(), reverse=True)
        for (_, new), (_, old) in zip(years, years[1:]):
            if old:
                refs.append((new - old) / abs(old) * 100)  # YoY %
            refs.append(new - old)
        for year, v in series.items():
            if revenue.get(year):
                refs.append(v / revenue[year] * 100)  # margin %

    for output in raw.values():  # macro records: latest vs one year ago
        if isinstance(output, list):
            for row in output:
                if isinstance(row, dict) and {"latest", "one_year_ago"} <= set(
                    row
                ):
                    new, old = row["latest"], row["one_year_ago"]
                    if isinstance(new, (int, float)) and isinstance(
                        old, (int, float)
                    ):
                        refs += [new - old, (new - old) * 100]
                        if old:
                            refs.append((new - old) / abs(old) * 100)

    prices = [
        v for k, v in _flat_items(raw) if k in ("last_close", "currentPrice")
    ]
    targets = [v for k, v in _flat_items(raw) if k == "targetMeanPrice"]
    for p in prices:
        for t in targets:
            if p:
                refs.append((t - p) / p * 100)  # upside to target %
    return [abs(v) for v in refs]


def _flat_items(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                yield k, float(v)
            else:
                yield from _flat_items(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _flat_items(v)


# Checking -----------------------------------------------------------------


def _matches(fig: Figure, refs: list[float]) -> bool:
    """True if the figure, or its written number, equals a reference value.

    The written number is tried only for figures with a magnitude word, to
    accept sources that already report in that scale.
    """
    tol = max(fig.tolerance, abs(fig.value) * 1e-6)
    if any(abs(fig.value - r) <= tol for r in refs):
        return True
    if fig.written is None:
        return False
    tol = max(fig.written_tolerance, abs(fig.written) * 1e-6)
    return any(abs(fig.written - r) <= tol for r in refs)


def _expected(claim, lines: dict) -> float | None:
    """Raw value for an explicit metric and fiscal year, if available."""
    if not claim.period or claim.period == "TTM":
        return None
    try:
        year = int(re.sub(r"\D", "", claim.period))
    except ValueError:
        return None
    if claim.metric in _LINE and claim.unit == "USD":
        return lines.get(_LINE[claim.metric], {}).get(year)
    if claim.metric in _MARGIN and claim.unit == "percent":
        part = lines.get(_MARGIN[claim.metric], {}).get(year)
        revenue = lines.get("Total Revenue", {}).get(year)
        if part is not None and revenue:
            return part / revenue * 100
    return None


def _written_tolerance(excerpt: str) -> float:
    figs = _figures(excerpt)
    return figs[-1].tolerance if figs else 0.0


def check_finding(finding: Finding, raw: dict) -> FindingCheck:
    if finding.route == "news":
        return FindingCheck(
            route=finding.route,
            status="not_checked",
            note="News figures come from article text, which the verifier "
            "does not hold.",
        )
    text = "\n".join([finding.summary, *finding.key_figures])
    refs = reference_values(raw)
    lines = financial_lines(raw)
    result = FindingCheck(route=finding.route, status="checked")

    contradicted_values = set()
    for claim in extract_reported_claims(text):
        expected = _expected(claim, lines)
        if expected is None:
            continue
        tol = max(_written_tolerance(claim.excerpt), abs(expected) * 1e-6)
        if abs(float(claim.value) - expected) > tol:
            result.contradicted.append(
                {
                    "text": claim.excerpt,
                    "metric": claim.metric,
                    "period": claim.period,
                    "expected": expected,
                    "unit": claim.unit,
                }
            )
            contradicted_values.add(round(abs(float(claim.value)), 6))

    for fig in _figures(text):
        if round(fig.value, 6) in contradicted_values:
            continue
        (result.confirmed if _matches(fig, refs) else result.not_found).append(
            fig.text
        )
    return result


def _fmt(value: float, unit: str) -> str:
    if unit == "percent":
        return f"{value:.2f}%"
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.3f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    return f"${value:,.2f}"


def annotate(finding: Finding, check: FindingCheck) -> Finding:
    """Copy of the finding with verification lines added to key figures."""
    if check.status != "checked" or not check.checked:
        return finding.model_copy(deep=True)
    lines = [
        f"VERIFICATION: {len(check.confirmed)} of {check.checked} figures "
        "match the raw tool data."
    ]
    for c in check.contradicted:
        lines.append(
            f"VERIFICATION CORRECTION: {c['metric']} {c['period']} is "
            f"{_fmt(c['expected'], c['unit'])} in the raw tool data, "
            f"not as stated in '{c['text']}'."
        )
    if check.not_found:
        lines.append(
            "VERIFICATION: not found in the raw tool data; attribute or omit: "
            + "; ".join(check.not_found[:10])
        )
    out = finding.model_copy(deep=True)
    out.key_figures = [*out.key_figures, *lines]
    return out


def verify_findings(
    findings: list[Finding], raw_by_route: dict[str, dict]
) -> tuple[list[Finding], list[FindingCheck]]:
    """Check each finding against all raw tool data from this run.

    The run's data is pooled because a finding may combine figures from
    another specialist's tools (upside to the analyst target uses the
    market specialist's last close). Contradictions still need an
    explicit metric and fiscal year present in the data.
    """
    pooled = {
        f"{route} / {step}": output
        for route, outputs in raw_by_route.items()
        for step, output in outputs.items()
    }
    checks = [check_finding(f, pooled) for f in findings]
    return [annotate(f, c) for f, c in zip(findings, checks)], checks


def summary_rows(checks: list[FindingCheck]) -> list[dict]:
    """One row per finding, for the notebook and the trace."""
    return [
        {
            "route": c.route,
            "status": c.status,
            "checked": c.checked,
            "confirmed": len(c.confirmed),
            "not_found": len(c.not_found),
            "contradicted": len(c.contradicted),
            "not_found_figures": "; ".join(c.not_found),
            "corrections": "; ".join(
                f"{x['text']} -> {_fmt(x['expected'], x['unit'])}"
                for x in c.contradicted
            ),
            "note": c.note,
        }
        for c in checks
    ]


def as_dict(check: FindingCheck) -> dict:
    return {**asdict(check), "checked": check.checked}


__all__ = [
    "FindingCheck",
    "annotate",
    "check_finding",
    "reference_values",
    "summary_rows",
    "verify_findings",
]
