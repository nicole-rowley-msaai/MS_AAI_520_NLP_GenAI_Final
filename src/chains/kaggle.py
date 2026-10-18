"""Kaggle financial news dataset: load, clean, split, score the classifier.

The Kaggle file `all-data.csv` is the Financial PhraseBank 50%-agreement
set (Malo et al., 2014). The PhraseBank release in
`data/kaggle/FinancialPhraseBank/` adds the stricter agreement subsets,
which this module uses to break classifier accuracy down by how much the
human annotators agreed.

In-text citations refer to the AAI-520 module readings; full references
are in docs/nlp_track.md.
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from src.chains.schemas import Article
from src.config import DATA_DIR, KAGGLE_CSV

LABELS = ["negative", "neutral", "positive"]
SEED = 42

PHRASEBANK_DIR = DATA_DIR / "kaggle" / "FinancialPhraseBank"
# Strictest subset first: a sentence's band is the first file that has it.
# The subsets are nested (AllAgree is inside 75Agree, and so on), as the
# PhraseBank README describes (Malo et al., 2014).
AGREEMENT_FILES = {
    "100%": "Sentences_AllAgree.txt",
    "75-99%": "Sentences_75Agree.txt",
    "66-74%": "Sentences_66Agree.txt",
    "50-65%": "Sentences_50Agree.txt",
}


# Loading and cleaning ------------------------------------------------------
def load_kaggle(path=KAGGLE_CSV) -> pd.DataFrame:
    """all-data.csv has no header and is latin-1 encoded."""
    df = pd.read_csv(
        path, header=None, names=["sentiment", "headline"], encoding="latin-1"
    )
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df["headline"] = (
        df["headline"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


def normalize_text(text: str) -> str:
    """Matching key: case-folded, punctuation and whitespace removed.

    Used only to detect duplicates and to join files, never as classifier
    input. Case folding helps matching, as in information retrieval, but
    case carries signal for sentiment classification and is "generally
    not done" there (Jurafsky & Martin, 2026, Section 2.6).
    """
    return re.sub(r"[^a-z0-9]+", "", str(text).casefold())


def cleaning_report(df: pd.DataFrame) -> dict:
    """Counts behind each cleaning rule, for the EDA write-up."""
    key = df["headline"].map(normalize_text)
    labels_per_key = df.groupby(key)["sentiment"].nunique()
    return {
        "rows": len(df),
        "exact_duplicates": int(df.duplicated("headline").sum()),
        "normalized_duplicates": int(key.duplicated().sum()),
        "conflicting_label_groups": int((labels_per_key > 1).sum()),
    }


def clean_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows, invalid labels, conflicts, and duplicates.

    Duplicates are found on the normalized key, so "Profit rose ." and
    "profit rose" count as one sentence (text normalization; Jurafsky &
    Martin, 2026, Sections 2.4 and 2.6). A sentence that appears with two
    different labels is dropped rather than kept at random, because no
    single gold label exists for it.
    """
    df = df[df["headline"].str.len() > 0]
    df = df[df["sentiment"].isin(LABELS)]
    key = df["headline"].map(normalize_text)
    consistent = df.groupby(key)["sentiment"].transform("nunique") == 1
    df, key = df[consistent], key[consistent]
    return df[~key.duplicated()].reset_index(drop=True)


# Splits --------------------------------------------------------------------
def train_test(df: pd.DataFrame, n: int = 300):
    """Fixed, stratified split: (train, test) with `n` test rows.

    The test rows are exactly those `held_out_sample` returns. Few-shot
    examples and supervised baselines draw only from `train`, as Brown et
    al. (2020, Section 2.4) condition each test example on examples from
    the training set, never from the evaluation set.
    """
    train, test = train_test_split(
        df, test_size=n, stratify=df["sentiment"], random_state=SEED
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def held_out_sample(df: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    """Fixed, stratified sample used as the classify-step test set."""
    return train_test(df, n)[1]


def to_articles(df: pd.DataFrame) -> list[Article]:
    return [Article(id=i, title=h) for i, h in enumerate(df["headline"])]


def fewshot_examples(
    train: pd.DataFrame,
    per_label: int = 2,
    agreement: dict[str, str] | None = None,
    seed: int = SEED,
) -> list[tuple[str, str]]:
    """Balanced demonstrations for few-shot prompting, from `train` only.

    Few-shot prompting conditions the model on K worked examples with no
    weight updates (Brown et al., 2020, Section 2; Stoffelbauer, 2023).
    When agreement bands are given, examples come from sentences every
    annotator agreed on, so each demonstration has an unambiguous label.
    K is fixed in advance rather than tuned on the test set.
    """
    pool = train
    if agreement:
        bands = train["headline"].map(
            lambda h: agreement.get(normalize_text(h))
        )
        unanimous = train[bands == "100%"]
        counts = unanimous["sentiment"].value_counts()
        if all(counts.get(label, 0) >= per_label for label in LABELS):
            pool = unanimous
    picks = []
    for label in LABELS:
        rows = pool[pool["sentiment"] == label]
        sample = rows.sample(n=per_label, random_state=seed)
        picks += [(h, label) for h in sample["headline"]]
    return picks


# Baselines -----------------------------------------------------------------
# A tiny keyword baseline the LLM classifier must beat.
POS = (
    r"\b(rise|rose|gain|grew|growth|increase|profit|beat|record|up"
    r"|improve|strong)\w*"
)
NEG = (
    r"\b(fall|fell|drop|declin|loss|decreas|cut|down|weak|lower|miss"
    r"|layoff)\w*"
)


def keyword_baseline(headline: str) -> str:
    p = len(re.findall(POS, headline, flags=re.I))
    n = len(re.findall(NEG, headline, flags=re.I))
    return "positive" if p > n else "negative" if n > p else "neutral"


def tfidf_baseline(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Supervised bag-of-n-grams baseline trained on the train split.

    Jurafsky and Martin (2026, Section 9.4) describe two ways to apply a
    pretrained model to a downstream task: prompt it, or train on labeled
    data. The LLM classifier is the first; this is a cheap classical
    stand-in for the second, so the LLM is compared with a model that
    learned from the same labels and not only with keywords. Built with
    scikit-learn (Pedregosa et al., 2011); class_weight="balanced" offsets
    the majority neutral class.
    """
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(train["headline"], train["sentiment"])
    return [str(p) for p in model.predict(test["headline"])]


def majority_vote(runs: list[list[str]]) -> list[str]:
    """Per-item majority label across repeated LLM runs.

    The same idea as self-consistency (Wang et al., 2023): keep the answer
    most independent samples agree on. It reuses runs the notebook already
    makes, so it costs no extra API calls. When the top labels tie, the
    item is labeled neutral, the majority class.
    """
    out = []
    for votes in zip(*runs):
        ranked = Counter(votes).most_common()
        tied = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
        out.append("neutral" if tied else ranked[0][0])
    return out


# Scoring -------------------------------------------------------------------
def score(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS),
        "report": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0
        ),
    }


def load_agreement(folder: Path = PHRASEBANK_DIR) -> dict[str, str]:
    """Normalized sentence -> strictest agreement band that contains it.

    Each PhraseBank sentence has 5 to 8 annotations, and the release
    groups sentences by the strength of the majority (Malo et al., 2014).
    Returns {} if any file is missing, so callers can skip the analysis.
    """
    bands: dict[str, str] = {}
    for band, name in AGREEMENT_FILES.items():
        path = Path(folder) / name
        if not path.exists():
            return {}
        for line in path.read_text(encoding="latin-1").splitlines():
            if "@" in line:
                sentence = line.rsplit("@", 1)[0]
                bands.setdefault(normalize_text(sentence), band)
    return bands


def score_by_agreement(
    test: pd.DataFrame, preds: list[str], agreement: dict[str, str]
) -> pd.DataFrame:
    """Accuracy and macro-F1 per annotator-agreement band.

    If the classifier is far more accurate where every annotator agreed,
    many of its remaining "errors" are sentences people also disagreed on.
    """
    frame = pd.DataFrame(
        {
            "band": [
                agreement.get(normalize_text(h), "unmatched")
                for h in test["headline"]
            ],
            "true": list(test["sentiment"]),
            "pred": list(preds),
        }
    )
    rows = []
    for band in [*AGREEMENT_FILES, "unmatched"]:
        part = frame[frame["band"] == band]
        if part.empty:
            continue
        macro_f1 = f1_score(
            part["true"],
            part["pred"],
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        rows.append(
            {
                "band": band,
                "n": len(part),
                "accuracy": round(
                    accuracy_score(part["true"], part["pred"]), 3
                ),
                "macro_f1": round(macro_f1, 3),
            }
        )
    return pd.DataFrame(rows)
