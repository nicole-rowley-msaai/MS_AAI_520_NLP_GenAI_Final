"""News prompt chain: Ingest -> Preprocess -> Classify -> Extract -> Summarize.

Each step is its own function and returns typed output, so the notebook
can print what every step produced. In-text citations refer to the
AAI-520 module readings; full references are in docs/nlp_track.md.
"""

import html
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from src.chains.schemas import (
    Article,
    ClassificationBatch,
    ExtractionBatch,
    NewsDigest,
    NewsDigestDraft,
)
from src.config import MODELS, NEWS_LOOKBACK_DAYS
from src.llm import structured
from src.tools.news import get_news

BATCH = 20
MAX_WORKERS = 4  # concurrent LLM calls per step; batches are independent


# 1. Ingest ---------------------------------------------------------------
def ingest(ticker: str, company: str) -> list[Article]:
    raw = get_news(ticker, company)
    return [Article(id=i, **a) for i, a in enumerate(raw)]


def date_span(articles: list[Article]) -> dict:
    """Earliest and latest publish time, so the notebook shows how many
    days of news the 100-article page actually covers."""
    dates = sorted(a.published for a in articles if a.published)
    return {"earliest": dates[0], "latest": dates[-1]} if dates else {}


# 2. Preprocess -----------------------------------------------------------
# Several written forms of one company name map to a single normal form
# (Jurafsky & Martin, 2026, Section 2.6, on USA and US).
NAME_ALIASES = {
    "NVDA": ("Nvidia",),
    "JPM": ("JPMorgan", "JP Morgan", "J.P. Morgan"),
    "XOM": ("Exxon", "ExxonMobil", "Exxon Mobil"),
}

# Software release feeds and code hosts that NewsAPI indexes; their items
# are package versions, not news ("nvidia-nvcomp-cu13 6.0.0.12").
NON_NEWS_DOMAINS = ("pypi.org", "github.com", "npmjs.com", "sourceforge.net")


def _strip(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _norm_title(title: str) -> str:
    # Case-folded key for duplicate detection only (Section 2.6).
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


def mention_pattern(company: str, ticker: str) -> re.Pattern:
    """Whole-word match for the company's names and its ticker.

    Names match in any case, but the ticker must be upper case, because
    case can separate meanings, as with US the country and us the pronoun
    (Jurafsky & Martin, 2026, Section 2.6). Word boundaries stop a short
    ticker from matching inside another word, which the earlier substring
    test allowed.
    """
    names = {company, company.split()[0], *NAME_ALIASES.get(ticker, ())}
    alternatives = "|".join(
        re.escape(n) for n in sorted(names, key=len, reverse=True) if n
    )
    return re.compile(
        rf"(?i:\b(?:{alternatives})\b)|\b{re.escape(ticker.upper())}\b"
    )


def _is_non_news(article: Article) -> bool:
    host = urlparse(article.url).netloc.lower()
    source = article.source.lower()
    return any(
        host == d or host.endswith("." + d) or source == d
        for d in NON_NEWS_DOMAINS
    )


def preprocess(
    articles: list[Article],
    company: str,
    ticker: str,
    days: int = NEWS_LOOKBACK_DAYS,
    stats: dict | None = None,
) -> list[Article]:
    """Strip HTML, drop duplicates and non-news, keep recent items that
    name the company.

    Pass a dict as `stats` to receive how many items each rule removed.
    """
    mentions = mention_pattern(company, ticker)
    dropped = {"duplicate": 0, "non_news": 0, "stale": 0, "no_mention": 0}
    seen, kept = set(), []
    for a in articles:
        a = a.model_copy(
            update={
                "title": _strip(a.title),
                "description": _strip(a.description),
            }
        )
        key = _norm_title(a.title)
        if not a.title or key in seen:
            dropped["duplicate"] += 1
        elif _is_non_news(a):
            dropped["non_news"] += 1
        elif not _is_recent(a.published, days):
            dropped["stale"] += 1
        elif not mentions.search(f"{a.title} {a.description}"):
            dropped["no_mention"] += 1
        else:
            seen.add(key)
            kept.append(a)
    if stats is not None:
        stats.update(dropped)
    return kept


# 3. Classify -------------------------------------------------------------
# The Financial PhraseBank annotators were told to judge only what the
# sentence states, from an investor's view of the stock price, and to
# call economically irrelevant sentiment neutral (Malo et al., 2014,
# dataset README). Using the same definition as the gold labels is the
# fix for the over-polarized neutral class seen in the A3 benchmark.
SENTIMENT_RULE = (
    "Judge sentiment from an investor's point of view: could this news "
    "move the company's stock price up (positive) or down (negative)? Use "
    "only the information the text states; do not infer an effect it does "
    "not mention. Use neutral for factual or routine items with no stated "
    "effect on the stock, and for sentiment that is not economically "
    "relevant."
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


def format_examples(examples: list[tuple[str, str]] | None) -> str:
    """Few-shot demonstrations appended to the system prompt.

    In-context examples show the model the task and output format with no
    weight updates (Brown et al., 2020, Section 2), and help when zero-shot
    instructions alone fall short (Stoffelbauer, 2023).
    """
    if not examples:
        return ""
    lines = "\n".join(f'- "{text}" -> {label}' for text, label in examples)
    return f"\n\nLabeled examples (sentiment only):\n{lines}"


def _fmt(items: list[Article]) -> str:
    return "\n".join(
        f"[{a.id}] {a.title}. {a.description}".strip() for a in items
    )


def _run_batches(call, articles: list[Article]) -> dict[int, dict]:
    """Apply `call` to each batch of articles, concurrently, and merge the
    results by article id. Ids the model invents are ignored."""
    batches = [
        articles[i : i + BATCH] for i in range(0, len(articles), BATCH)
    ]
    if len(batches) <= 1:
        results = [call(b) for b in batches]
    else:
        workers = min(MAX_WORKERS, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(call, batches))
    sent = {a.id for a in articles}
    out = {}
    for res in results:
        for item in res.items:
            if item.id in sent:
                out[item.id] = item.model_dump()
    return out


def coverage(labels: dict, articles: list[Article]) -> dict:
    """How many items came back labeled; missing ids are listed, not
    silently counted as neutral."""
    missing = [a.id for a in articles if a.id not in labels]
    return {
        "sent": len(articles),
        "labeled": len(articles) - len(missing),
        "missing_ids": missing,
    }


def classify(
    articles: list[Article],
    model: str = MODELS["chain"],
    system: str = CLASSIFY_SYSTEM,
    examples: list[tuple[str, str]] | None = None,
) -> dict[int, dict]:
    """Topic, sentiment, and relevance per article id.

    Pass system=CLASSIFY_SENTIMENT_SYSTEM to score the Kaggle benchmark,
    and `examples` (from kaggle.fewshot_examples) for few-shot prompting.
    """
    prompt = system + format_examples(examples)
    return _run_batches(
        lambda batch: structured(
            model, prompt, _fmt(batch), ClassificationBatch
        ),
        articles,
    )


# 4. Extract --------------------------------------------------------------
EXTRACT_SYSTEM = (
    "Extract facts from each news item. Only include what the text states; "
    "never infer numbers. Give figures with units and events with dates "
    "when given."
)


def extract(
    articles: list[Article], model: str = MODELS["chain"]
) -> dict[int, dict]:
    return _run_batches(
        lambda batch: structured(
            model, EXTRACT_SYSTEM, _fmt(batch), ExtractionBatch
        ),
        articles,
    )


# 5. Summarize ------------------------------------------------------------
SUMMARIZE_SYSTEM = (
    "Write a news digest for an equity analyst from the labeled, extracted "
    "items below. Cite article ids like [3] for every driver and risk. "
    "The net sentiment is given; it is the majority label over the "
    "relevant articles, so use it as stated and explain it in the summary."
)

NET_MARGIN = 0.15  # positive/negative needs this share-lead over the other


def sentiment_counts(labels: dict, articles: list[Article]) -> dict:
    """Articles per label. Unlabeled articles are left out rather than
    counted as neutral; `coverage` reports them."""
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for a in articles:
        label = labels.get(a.id, {}).get("sentiment")
        if label in counts:
            counts[label] += 1
    return counts


def net_sentiment(counts: dict) -> str:
    """Majority rule with a margin, so a 40/38 split is called neutral."""
    n = sum(counts.values()) or 1
    lead = (counts["positive"] - counts["negative"]) / n
    if lead > NET_MARGIN:
        return "positive"
    if lead < -NET_MARGIN:
        return "negative"
    return "neutral"


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
    counts = sentiment_counts(labels, articles)
    net = net_sentiment(counts)
    user = (
        f"Ticker: {ticker}\nSentiment counts: {counts}\n"
        f"Net sentiment: {net}\n\n" + "\n".join(lines)
    )
    draft = structured(model, SUMMARIZE_SYSTEM, user, NewsDigestDraft)
    return NewsDigest(
        **{**draft.model_dump(), "net_sentiment": net},  # counted, not judged
        sentiment_counts=counts,
    )


# Full chain --------------------------------------------------------------
def run_news_chain(
    ticker: str, company: str, trace: list | None = None
) -> NewsDigest:
    """Run all five steps; append each step's output to `trace` if given."""

    def log(step, data):
        if trace is not None:
            trace.append({"step": step, "output": data})

    raw = ingest(ticker, company)
    log(
        "ingest",
        {
            "n": len(raw),
            "sample": [a.title for a in raw[:5]],
            **date_span(raw),
        },
    )
    dropped: dict = {}
    clean = preprocess(raw, company, ticker, stats=dropped)
    log("preprocess", {"n_in": len(raw), "n_out": len(clean), **dropped})
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
    log("classify_coverage", coverage(labels, clean))
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
