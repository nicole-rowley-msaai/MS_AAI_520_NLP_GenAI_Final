"""Offline tests for the NLP-track changes: no API keys or network needed."""

import pandas as pd

from src.chains import kaggle
from src.chains.schemas import (
    Article,
    Classification,
    ClassificationBatch,
    Extraction,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=["sentiment", "headline"])


# Cleaning ------------------------------------------------------------------
def test_clean_kaggle_dedupes_on_normalized_text():
    df = _frame(
        [
            ("positive", "Profit rose 10 % ."),
            ("positive", "profit rose 10%"),
            ("neutral", "Company holds meeting"),
        ]
    )
    out = kaggle.clean_kaggle(df)
    assert len(out) == 2
    assert out["headline"].iloc[0] == "Profit rose 10 % ."  # original kept


def test_clean_kaggle_drops_conflicting_labels():
    df = _frame(
        [
            ("positive", "Sales were flat"),
            ("neutral", "Sales were flat"),
            ("negative", "Sales fell"),
        ]
    )
    out = kaggle.clean_kaggle(df)
    assert list(out["headline"]) == ["Sales fell"]
    report = kaggle.cleaning_report(df)
    assert report["conflicting_label_groups"] == 1
    assert report["exact_duplicates"] == 1


# Splits and few-shot examples ----------------------------------------------
def _big():
    return pd.DataFrame(
        {
            "sentiment": ["positive", "neutral", "negative"] * 100,
            "headline": [f"headline {i}" for i in range(300)],
        }
    )


def test_train_test_matches_held_out_sample_and_is_disjoint():
    df = _big()
    train, test = kaggle.train_test(df, n=30)
    assert test.equals(kaggle.held_out_sample(df, n=30))
    assert len(train) == 270
    assert not set(train["headline"]) & set(test["headline"])


def test_fewshot_examples_are_balanced_and_from_train():
    train, test = kaggle.train_test(_big(), n=30)
    examples = kaggle.fewshot_examples(train, per_label=2)
    labels = [label for _, label in examples]
    assert labels.count("positive") == labels.count("negative") == 2
    assert not {text for text, _ in examples} & set(test["headline"])


def test_fewshot_examples_prefer_unanimous_sentences():
    train, _ = kaggle.train_test(_big(), n=30)
    unanimous = {
        kaggle.normalize_text(h): "100%"
        for h, s in zip(train["headline"], train["sentiment"])
        if int(h.split()[1]) % 2 == 0
    }
    examples = kaggle.fewshot_examples(train, 2, agreement=unanimous)
    assert all(int(text.split()[1]) % 2 == 0 for text, _ in examples)


# Baselines and voting ------------------------------------------------------
def test_tfidf_baseline_learns_from_train():
    words = {"positive": "surged", "negative": "plunged", "neutral": "said"}
    rows = [
        (label, f"Company {word} report number {i}")
        for i in range(40)
        for label, word in words.items()
    ]
    df = _frame(rows)
    train, test = kaggle.train_test(df, n=30)
    preds = kaggle.tfidf_baseline(train, test)
    assert kaggle.score(test["sentiment"], preds)["accuracy"] == 1.0


def test_majority_vote_and_tie_rule():
    runs = [
        ["positive", "positive", "negative"],
        ["positive", "neutral", "positive"],
        ["negative", "negative", "neutral"],
    ]
    assert kaggle.majority_vote(runs) == ["positive", "neutral", "neutral"]


# Agreement bands -------------------------------------------------------------
def test_load_agreement_uses_strictest_band(tmp_path):
    files = {
        "Sentences_AllAgree.txt": ["Sales rose .@positive"],
        "Sentences_75Agree.txt": ["Sales rose .@positive", "B c .@neutral"],
        "Sentences_66Agree.txt": ["Sales rose .@positive", "B c .@neutral"],
        "Sentences_50Agree.txt": [
            "Sales rose .@positive",
            "B c .@neutral",
            "D e .@negative",
        ],
    }
    for name, lines in files.items():
        (tmp_path / name).write_text("\n".join(lines), encoding="latin-1")
    bands = kaggle.load_agreement(tmp_path)
    assert bands[kaggle.normalize_text("Sales rose")] == "100%"
    assert bands[kaggle.normalize_text("B c")] == "75-99%"
    assert bands[kaggle.normalize_text("D e")] == "50-65%"
    assert kaggle.load_agreement(tmp_path / "missing") == {}


def test_score_by_agreement_splits_by_band():
    test = _frame(
        [
            ("positive", "Sales rose"),
            ("negative", "Sales fell"),
            ("neutral", "Unknown sentence"),
        ]
    )
    bands = {
        kaggle.normalize_text("Sales rose"): "100%",
        kaggle.normalize_text("Sales fell"): "50-65%",
    }
    out = kaggle.score_by_agreement(
        test, ["positive", "positive", "neutral"], bands
    ).set_index("band")
    assert out.loc["100%", "accuracy"] == 1.0
    assert out.loc["50-65%", "accuracy"] == 0.0
    assert out.loc["unmatched", "n"] == 1


# News chain -----------------------------------------------------------------
def test_preprocess_matches_whole_words_and_aliases():
    from src.chains.news_chain import preprocess

    when = "2099-01-01T00:00:00Z"
    arts = [
        Article(id=0, title="J.P. Morgan raises outlook", published=when),
        Article(id=1, title="Lumpy markets jump", published=when),
        Article(id=2, title="JPM shares rise", published=when),
        Article(id=3, title="jpm-utils 1.0 released", published=when),
    ]
    out = preprocess(arts, "JPMorgan", "JPM")
    assert [a.id for a in out] == [0, 2]  # lower-case ticker is not a match


def test_preprocess_drops_package_feeds_and_reports_reasons():
    from src.chains.news_chain import preprocess

    when = "2099-01-01T00:00:00Z"
    arts = [
        Article(
            id=0,
            title="nvidia-nvcomp-cu13 6.0.0.12",
            description="NVIDIA nvcomp",
            source="Pypi.org",
            url="https://pypi.org/project/nvidia-nvcomp-cu13/",
            published=when,
        ),
        Article(id=1, title="Nvidia beats", published=when),
        Article(id=2, title="Nvidia beats!", published=when),
    ]
    stats = {}
    out = preprocess(arts, "Nvidia", "NVDA", stats=stats)
    assert [a.id for a in out] == [1]
    assert stats["non_news"] == 1 and stats["duplicate"] == 1


def test_extraction_keeps_only_numeric_figures():
    e = Extraction(
        id=0,
        entities=["NVIDIA"],
        figures=["dozens of AAA games", "revenue $26.0B", "up 4%"],
        events=[],
    )
    assert e.figures == ["revenue $26.0B", "up 4%"]


def test_classify_batches_in_parallel_with_examples(monkeypatch):
    import src.chains.news_chain as nc

    systems = []

    def fake_structured(model, system, user, schema):
        systems.append(system)
        ids = [
            int(ln[1 : ln.index("]")])
            for ln in user.splitlines()
            if ln.startswith("[")
        ]
        items = [
            Classification(id=i, topic="other", sentiment="positive")
            for i in ids
            if i != 3  # the model skips one item
        ]
        items.append(  # and invents one that was never sent
            Classification(id=999, topic="other", sentiment="negative")
        )
        return ClassificationBatch(items=items)

    monkeypatch.setattr(nc, "structured", fake_structured)
    arts = [Article(id=i, title=f"t{i}") for i in range(45)]
    labels = nc.classify(arts, examples=[("Sales rose", "positive")])
    assert len(systems) == 3  # 45 items in batches of 20
    assert all('"Sales rose" -> positive' in s for s in systems)
    assert 999 not in labels
    cov = nc.coverage(labels, arts)
    assert cov == {"sent": 45, "labeled": 44, "missing_ids": [3]}
    counts = nc.sentiment_counts(labels, arts)
    assert counts == {"positive": 44, "neutral": 0, "negative": 0}


def test_chain_logs_coverage_and_date_span(monkeypatch):
    import src.chains.news_chain as nc
    from src.chains.schemas import ExtractionBatch, NewsDigestDraft

    def fake_structured(model, system, user, schema):
        if schema is ClassificationBatch:
            first = Classification(id=0, topic="other", sentiment="neutral")
            return ClassificationBatch(items=[first])
        if schema is ExtractionBatch:
            return ExtractionBatch(items=[])
        return NewsDigestDraft(
            ticker="NVDA",
            net_sentiment="neutral",
            top_drivers=[],
            risks=[],
            summary="ok",
        )

    news = [
        {
            "title": f"Nvidia item {i}",
            "description": "",
            "source": "",
            "published": f"2099-01-0{i + 1}T00:00:00Z",
            "url": "",
        }
        for i in range(2)
    ]
    monkeypatch.setattr(nc, "structured", fake_structured)
    monkeypatch.setattr(nc, "get_news", lambda t, c: news)
    trace = []
    nc.run_news_chain("NVDA", "Nvidia", trace=trace)
    steps = {s["step"]: s["output"] for s in trace}
    assert steps["ingest"]["earliest"] == "2099-01-01T00:00:00Z"
    assert steps["ingest"]["latest"] == "2099-01-02T00:00:00Z"
    assert steps["classify_coverage"]["missing_ids"] == [1]
    assert steps["preprocess"]["duplicate"] == 0
