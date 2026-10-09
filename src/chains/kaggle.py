"""Kaggle financial news dataset: load, clean, split, score the classifier."""

import re

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.chains.schemas import Article
from src.config import KAGGLE_CSV

LABELS = ["negative", "neutral", "positive"]
SEED = 42


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


def clean_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows, invalid labels, and exact duplicate headlines."""
    df = df[df["headline"].str.len() > 0]
    df = df[df["sentiment"].isin(LABELS)]
    return df.drop_duplicates(subset="headline").reset_index(drop=True)


def held_out_sample(df: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    """Fixed, stratified sample used as the classify-step test set."""
    _, test = train_test_split(
        df, test_size=n, stratify=df["sentiment"], random_state=SEED
    )
    return test.reset_index(drop=True)


def to_articles(df: pd.DataFrame) -> list[Article]:
    return [Article(id=i, title=h) for i, h in enumerate(df["headline"])]


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


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS),
        "report": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0
        ),
    }
