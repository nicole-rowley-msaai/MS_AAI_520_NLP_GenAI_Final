"""Typed outputs for each step of the news prompt chain."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["positive", "neutral", "negative"]
Topic = Literal[
    "earnings", "m&a", "legal", "product", "macro", "management", "other"
]


class Article(BaseModel):
    id: int
    title: str
    description: str = ""
    source: str = ""
    published: str = ""
    url: str = ""


Relevance = Literal["high", "low"]


class Classification(BaseModel):
    id: int
    topic: Topic
    sentiment: Sentiment
    relevance: Relevance = Field(
        default="high",
        description="high if the item is mainly about the company; low for "
        "market roundups or items that only mention it in passing",
    )


class ClassificationBatch(BaseModel):
    items: list[Classification]


class Extraction(BaseModel):
    id: int
    entities: list[str] = Field(
        description="Companies, people, products named"
    )
    figures: list[str] = Field(
        description="Numbers with units, e.g. 'revenue $26.0B'"
    )
    events: list[str] = Field(
        description="Dated events, e.g. '2026-09-12: guidance raised'"
    )

    @field_validator("figures")
    @classmethod
    def _figures_have_numbers(cls, figures: list[str]) -> list[str]:
        """Keep only items that contain a digit.

        A figure is a number token such as a price or percentage, which
        tokenizers keep whole (Jurafsky & Martin, 2026, Section 2.5.1).
        Phrases like "dozens of AAA games" are not figures. The validator
        runs after parsing, so the JSON schema sent to the model is
        unchanged.
        """
        return [f for f in figures if any(ch.isdigit() for ch in f)]


class ExtractionBatch(BaseModel):
    items: list[Extraction]


class NewsDigestDraft(BaseModel):
    """What the summarizer model writes. OpenAI strict mode accepts only
    required fields of plain types, so the counted fields live in the
    subclass below and are filled in by the chain."""

    ticker: str
    net_sentiment: Sentiment
    top_drivers: list[str] = Field(
        description="3-5 key drivers, each citing article ids"
    )
    risks: list[str]
    summary: str = Field(description="One short paragraph")


class SentimentCounts(BaseModel):
    positive: int = 0
    neutral: int = 0
    negative: int = 0


class NewsDigest(NewsDigestDraft):
    sentiment_counts: SentimentCounts = Field(
        default_factory=SentimentCounts,
        description="Articles per label among the relevant items; counted "
        "by the chain, not judged by the model",
    )
