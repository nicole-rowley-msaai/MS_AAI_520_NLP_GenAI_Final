"""News prompt chain: Ingest -> Preprocess -> Classify -> Extract -> Summarize.

Each step is its own function and returns typed output, so the notebook
can print what every step produced.
"""

import html
import re
from datetime import datetime, timedelta, timezone

from src.chains.schemas import (
    Article,
    ClassificationBatch,
    ExtractionBatch,
    NewsDigest,
)
from src.config import MODELS, NEWS_LOOKBACK_DAYS
from src.llm import structured
from src.tools.news import get_news

BATCH = 20


# 1. Ingest ---------------------------------------------------------------
def ingest(ticker: str, company: str) -> list[Article]:
    raw = get_news(ticker, company)
    return [Article(id=i, **a) for i, a in enumerate(raw)]


# 2. Preprocess -----------------------------------------------------------
def _strip(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower())[:80]


def _is_recent(published: str, days: int) -> bool:
    if not published:
        return True  # keep undated items rather than guess
    try:
        ts = datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts >= datetime.now(timezone.utc) - timedelta(days=days)


def preprocess(
    articles: list[Article],
    company: str,
    ticker: str,
    days: int = NEWS_LOOKBACK_DAYS,
) -> list[Article]:
    """Strip HTML, drop duplicates, keep recent items naming the company."""
    name_tokens = {company.lower(), ticker.lower(), company.split()[0].lower()}
    seen, kept = set(), []
    for a in articles:
        a = a.model_copy(
            update={
                "title": _strip(a.title),
                "description": _strip(a.description),
            }
        )
        key = _norm_title(a.title)
        text = f"{a.title} {a.description}".lower()
        if not a.title or key in seen:
            continue
        if not _is_recent(a.published, days):
            continue
        if not any(tok in text for tok in name_tokens):
            continue
        seen.add(key)
        kept.append(a)
    return kept


# 3. Classify -------------------------------------------------------------
SENTIMENT_RULE = (
    "Sentiment is judged from the company's shareholders' point of view. Use "
    "neutral when the item is factual with no clear effect on the stock."
)

# Live chain: topic + sentiment + relevance for one company's news feed.
CLASSIFY_SYSTEM = (
    "You label financial news for an equity analyst. For each item return its "
    f"topic, its sentiment, and its relevance. {SENTIMENT_RULE} Relevance is "
    "low ONLY for market roundups, daily summaries, or lists that cover many "
    "companies. News about the company's products, customers, partners, "
    "suppliers, or share price is high, even if another company is named "
    "first."
)

# Benchmark: sentiment only. The Kaggle headlines have no target company, so
# the relevance instruction would only distract from the graded task.
CLASSIFY_SENTIMENT_SYSTEM = (
    "You label financial news headlines for an equity analyst. For each item "
    f"return its topic and its sentiment. {SENTIMENT_RULE} "
    "Set relevance to high."
)


def _fmt(items: list[Article]) -> str:
    return "\n".join(
        f"[{a.id}] {a.title}. {a.description}".strip() for a in items
    )


def classify(
    articles: list[Article],
    model: str = MODELS["chain"],
    system: str = CLASSIFY_SYSTEM,
) -> dict[int, dict]:
    """Topic, sentiment, and relevance per article id.

    Pass system=CLASSIFY_SENTIMENT_SYSTEM to score the Kaggle benchmark.
    """
    out = {}
    for i in range(0, len(articles), BATCH):
        batch = articles[i : i + BATCH]
        res = structured(model, system, _fmt(batch), ClassificationBatch)
        out.update({c.id: c.model_dump() for c in res.items})
    return out


# 4. Extract --------------------------------------------------------------
EXTRACT_SYSTEM = (
    "Extract facts from each news item. Only include what the text states; "
    "never infer numbers. Give figures with units and events with dates "
    "when given."
)


def extract(
    articles: list[Article], model: str = MODELS["chain"]
) -> dict[int, dict]:
    out = {}
    for i in range(0, len(articles), BATCH):
        batch = articles[i : i + BATCH]
        res = structured(model, EXTRACT_SYSTEM, _fmt(batch), ExtractionBatch)
        out.update({e.id: e.model_dump() for e in res.items})
    return out


# 5. Summarize ------------------------------------------------------------
SUMMARIZE_SYSTEM = (
    "Write a news digest for an equity analyst from the labeled, extracted "
    "items below. Cite article ids like [3] for every driver and risk."
)


def summarize(
    ticker: str,
    articles: list[Article],
    labels: dict,
    facts: dict,
    model: str = MODELS["chain"],
) -> NewsDigest:
    lines = []
    for a in articles:
        lab, fx = labels.get(a.id, {}), facts.get(a.id, {})
        lines.append(
            f"[{a.id}] {a.title} | topic={lab.get('topic')} "
            f"sentiment={lab.get('sentiment')} "
            f"| figures={fx.get('figures')} events={fx.get('events')}"
        )
    user = f"Ticker: {ticker}\n\n" + "\n".join(lines)
    return structured(model, SUMMARIZE_SYSTEM, user, NewsDigest)


# Full chain --------------------------------------------------------------
def run_news_chain(
    ticker: str, company: str, trace: list | None = None
) -> NewsDigest:
    """Run all five steps; append each step's output to `trace` if given."""

    def log(step, data):
        if trace is not None:
            trace.append({"step": step, "output": data})

    raw = ingest(ticker, company)
    log("ingest", {"n": len(raw), "sample": [a.title for a in raw[:5]]})
    clean = preprocess(raw, company, ticker)
    log("preprocess", {"n_in": len(raw), "n_out": len(clean)})
    if not clean:
        return NewsDigest(
            ticker=ticker,
            net_sentiment="neutral",
            top_drivers=[],
            risks=[],
            summary="No recent relevant news found.",
        )
    labels = classify(clean)
    log("classify", labels)
    relevant = [
        a for a in clean if labels.get(a.id, {}).get("relevance") == "high"
    ]
    log("relevance_filter", {"n_in": len(clean), "n_out": len(relevant)})
    if not relevant:
        relevant = clean  # better a noisy digest than an empty one
    facts = extract(relevant)
    log("extract", facts)
    digest = summarize(ticker, relevant, labels, facts)
    log("summarize", digest.model_dump())
    return digest
