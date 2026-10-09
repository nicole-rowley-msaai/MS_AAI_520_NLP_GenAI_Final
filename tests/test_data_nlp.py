"""Offline tests: no API keys or network needed."""

import pandas as pd
import pytest

from src.chains.kaggle import (
    clean_kaggle,
    held_out_sample,
    keyword_baseline,
    load_kaggle,
    score,
)
from src.chains.news_chain import preprocess
from src.chains.schemas import Article
from src.tools.filings import chunk_text, extract_section
from src.tools.market import compute_technicals


def test_preprocess_dedupes_strips_and_filters():
    arts = [
        Article(
            id=0,
            title="<b>Nvidia</b> beats estimates",
            published="2099-01-01T00:00:00Z",
        ),
        Article(
            id=1,
            title="Nvidia beats estimates!",
            published="2099-01-01T00:00:00Z",
        ),
        Article(id=2, title="Apple launches phone"),
        Article(
            id=3, title="Nvidia old news", published="2001-01-01T00:00:00Z"
        ),
    ]
    out = preprocess(arts, "Nvidia Corporation", "NVDA")
    assert [a.id for a in out] == [0]
    assert out[0].title == "Nvidia beats estimates"


def test_extract_section_skips_table_of_contents():
    text = (
        "Table of contents Item 1A. Risk Factors 12 Item 1B. Unresolved 30 "
        "Item 1A. Risk Factors "
        + "Supply chain risk. " * 50
        + "Item 1B. Unresolved staff comments"
    )
    sec = extract_section(text, r"item\s*1a[\.\s]", r"item\s*1b[\.\s]")
    assert "Supply chain risk" in sec and len(sec) > 500


def test_chunk_text_overlaps():
    chunks = chunk_text(
        " ".join(str(i) for i in range(700)), size=300, overlap=50
    )
    assert len(chunks) == 3
    assert chunks[1].split()[0] == "250"


def test_kaggle_load_clean_split(tmp_path):
    rows = [
        ("positive", "Profit rose 10%"),
        ("negative", "Sales fell"),
        ("neutral", "Company holds meeting"),
    ] * 40 + [("neutral", "Company holds meeting")]
    p = tmp_path / "all-data.csv"
    pd.DataFrame(rows).to_csv(p, header=False, index=False, encoding="latin-1")
    df = clean_kaggle(load_kaggle(p))
    assert len(df) == 3  # duplicates removed
    big = pd.DataFrame(
        {
            "sentiment": ["positive", "neutral", "negative"] * 100,
            "headline": [f"h{i}" for i in range(300)],
        }
    )
    test = held_out_sample(big, n=30)
    expected = {"positive": 10, "neutral": 10, "negative": 10}
    assert test["sentiment"].value_counts().to_dict() == expected


def test_keyword_baseline_and_score():
    assert keyword_baseline("Operating profit rose sharply") == "positive"
    assert keyword_baseline("Net sales fell 5%") == "negative"
    s = score(["positive", "negative"], ["positive", "neutral"])
    assert s["accuracy"] == 0.5


def test_compute_technicals():
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    prices = pd.DataFrame(
        {"Close": [100 + i * 0.1 for i in range(300)]}, index=idx
    )
    t = compute_technicals(prices, benchmark=prices)
    assert t["beta_vs_spy"] == pytest.approx(1.0)
    assert t["sma_50"] < t["last_close"]


def test_extract_sections_falls_back_to_exhibit():
    from src.tools.filings import extract_sections

    body = (
        "Item 7. Management's Discussion and Analysis 45 Item 7A. "
        "Quantitative 90 Item 1A. Risk Factors "
        + "Risk text. " * 600
        + "Item 1B. Unresolved Item 7. Incorporated by reference. Item 8."
    )
    exhibit = (
        "Contents Management’s discussion and analysis 20 "
        "Management’s discussion and analysis "
        + "MD&A body. " * 700
        + "Management’s report on internal control"
    )
    secs = extract_sections({"text": body, "exhibit_text": exhibit})
    assert len(secs["risk_factors"].split()) > 500
    assert len(secs["mdna"].split()) > 500 and "MD&A body" in secs["mdna"]


def test_relevance_filter_drops_low_relevance(monkeypatch):
    import src.chains.news_chain as nc
    from src.chains.schemas import (
        Classification,
        ClassificationBatch,
        Extraction,
        ExtractionBatch,
        NewsDigest,
    )

    def fake_structured(model, system, user, schema):
        ids = [
            int(ln[1 : ln.index("]")])
            for ln in user.splitlines()
            if ln.startswith("[")
        ]
        if schema is ClassificationBatch:
            return ClassificationBatch(
                items=[
                    Classification(
                        id=i,
                        topic="other",
                        sentiment="neutral",
                        relevance="high" if i == 0 else "low",
                    )
                    for i in ids
                ]
            )
        if schema is ExtractionBatch:
            return ExtractionBatch(
                items=[
                    Extraction(id=i, entities=[], figures=[], events=[])
                    for i in ids
                ]
            )
        return NewsDigest(
            ticker="NVDA",
            net_sentiment="neutral",
            top_drivers=[],
            risks=[],
            summary="ok",
        )

    monkeypatch.setattr(nc, "structured", fake_structured)
    monkeypatch.setattr(
        nc,
        "get_news",
        lambda t, c: [
            {
                "title": "Nvidia beats",
                "description": "",
                "source": "",
                "published": "2099-01-01T00:00:00Z",
                "url": "",
            },
            {
                "title": "Nvidia roundup",
                "description": "",
                "source": "",
                "published": "2099-01-01T00:00:00Z",
                "url": "",
            },
        ],
    )
    trace = []
    nc.run_news_chain("NVDA", "Nvidia", trace=trace)
    steps = {s["step"]: s["output"] for s in trace}
    assert steps["relevance_filter"] == {"n_in": 2, "n_out": 1}
    assert list(steps["extract"]) == [0]


def test_secret_is_read_at_call_time(monkeypatch):
    from src.config import secret

    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="EDGAR_USER_AGENT"):
        secret("EDGAR_USER_AGENT")
    monkeypatch.setenv("EDGAR_USER_AGENT", "Test User test@example.com")
    assert secret("EDGAR_USER_AGENT") == "Test User test@example.com"


def test_mdna_heading_fallback_in_main_document():
    """JPMorgan style: Item 7 is a cross-reference; the MD&A sits under its
    plain heading later in the same document, with no EX-13 exhibit."""
    from src.tools.filings import extract_sections

    body = (
        "Item 1A. Risk Factors "
        + "Risk text. " * 600
        + "Item 1B. Unresolved staff comments "
        + "Item 7. Management’s Discussion and Analysis appears on "
        "pages 46-160. Item 8. Financial statements. "
        + "Management’s discussion and analysis "
        + "MD&A body. " * 700
        + "Management’s report on internal control over financial reporting"
    )
    secs = extract_sections({"text": body, "exhibit_text": ""})
    assert len(secs["mdna"].split()) > 1000
    assert "MD&A body" in secs["mdna"]


def test_all_filings_follows_pagination(monkeypatch):
    import src.tools.filings as fl

    pages = {
        fl.SUBMISSIONS_URL.format(cik="0000034088"): {
            "filings": {
                "recent": {
                    "form": ["4", "8-K"],
                    "accessionNumber": ["a", "b"],
                    "primaryDocument": ["x", "y"],
                    "filingDate": ["2026-10-01", "2026-09-01"],
                },
                "files": [{"name": "CIK0000034088-submissions-001.json"}],
            }
        },
        fl.SUBMISSIONS_BASE
        + "CIK0000034088-submissions-001.json": {
            "form": ["10-K"],
            "accessionNumber": ["c"],
            "primaryDocument": ["z"],
            "filingDate": ["2026-02-25"],
        },
    }

    class Resp:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    monkeypatch.setattr(fl, "_get", lambda url: Resp(pages[url]))
    forms = [f for f, *_ in fl._all_filings("0000034088")]
    assert forms == ["4", "8-K", "10-K"]


def test_classify_accepts_custom_system_prompt(monkeypatch):
    import src.chains.news_chain as nc
    from src.chains.schemas import Article, Classification, ClassificationBatch

    seen = {}

    def fake_structured(model, system, user, schema):
        seen["system"] = system
        return ClassificationBatch(
            items=[Classification(id=0, topic="other", sentiment="neutral")]
        )

    monkeypatch.setattr(nc, "structured", fake_structured)
    nc.classify(
        [Article(id=0, title="x")], system=nc.CLASSIFY_SENTIMENT_SYSTEM
    )
    assert seen["system"] == nc.CLASSIFY_SENTIMENT_SYSTEM
    assert "relevance is low" not in seen["system"].lower()


def test_get_latest_10k_falls_back_to_predecessor_cik(monkeypatch):
    """Exxon style: the ticker's current registrant has no 10-K; the prior
    entity in PREDECESSOR_CIKS does."""
    import src.tools.filings as fl

    new_cik, old_cik = "0002115436", "0000034088"
    filings = {
        new_cik: [("8-K", "0002115436-26-000001", "a.htm", "2026-08-01")],
        old_cik: [
            ("10-K", "0000034088-26-000010", "xom-20251231.htm", "2026-02-25")
        ],
    }
    monkeypatch.setattr(fl, "get_cik", lambda t: new_cik)
    monkeypatch.setattr(fl, "PREDECESSOR_CIKS", {"XOM": old_cik})
    monkeypatch.setattr(fl, "_all_filings", lambda cik: iter(filings[cik]))
    monkeypatch.setattr(fl, "_get_annual_report_exhibit", lambda *a: "")

    class Resp:
        text = "<html>Item 1A. Risk Factors body</html>"

    monkeypatch.setattr(fl, "_get", lambda url: Resp())
    filing = fl.get_latest_10k.__wrapped__("XOM")
    assert filing["cik"] == old_cik
    assert filing["filed"] == "2026-02-25"
    assert "xom-20251231.htm" in filing["url"]
