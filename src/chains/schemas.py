"""Typed outputs for each step of the news prompt chain."""

from typing import Literal

from pydantic import BaseModel, Field

Sentiment = Literal["positive", "neutral", "negative"]
Topic = Literal["earnings", "m&a", "legal", "product", "macro", "management", "other"]


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
    entities: list[str] = Field(description="Companies, people, products named")
    figures: list[str] = Field(description="Numbers with units, e.g. 'revenue $26.0B'")
    events: list[str] = Field(description="Dated events, e.g. '2026-09-12: guidance raised'")


class ExtractionBatch(BaseModel):
    items: list[Extraction]


class NewsDigest(BaseModel):
    ticker: str
    net_sentiment: Sentiment
    top_drivers: list[str] = Field(description="3-5 key drivers, each citing article ids")
    risks: list[str]
    summary: str = Field(description="One short paragraph")
