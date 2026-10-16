"""News: live headlines from NewsAPI, with yfinance headlines as a backup.

Both sources return the same keys, matching chains.schemas.Article:
    title, description, source, published, url
"""

from datetime import date, datetime, timedelta, timezone

import requests
import yfinance as yf

from src.config import NEWS_LOOKBACK_DAYS, secret
from src.tools.cache import memory

NEWSAPI_URL = "https://newsapi.org/v2/everything"


@memory.cache
def _newsapi(query: str, days: int, day: str) -> list[dict]:
    # `day` is only part of the cache key, so results refresh once a day.
    api_key = secret("NEWSAPI_KEY")
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    # qInTitle: the company must be in the headline, which drops market
    # roundups that only mention it in passing.
    resp = requests.get(NEWSAPI_URL, timeout=20, params={
        "qInTitle": query, "from": since, "language": "en",
        "sortBy": "publishedAt", "pageSize": 100, "apiKey": api_key,
    })
    resp.raise_for_status()
    return [{
        "title": a.get("title") or "",
        "description": a.get("description") or "",
        "source": (a.get("source") or {}).get("name") or "",
        "published": a.get("publishedAt") or "",
        "url": a.get("url") or "",
    } for a in resp.json().get("articles", [])]


def _yfinance_news(ticker: str) -> list[dict]:
    out = []
    for item in yf.Ticker(ticker).news or []:
        c = item.get("content", item)  # newer yfinance nests under "content"
        provider, link = c.get("provider"), c.get("canonicalUrl")
        out.append({
            "title": c.get("title") or "",
            "description": c.get("summary") or c.get("description") or "",
            "source": provider.get("displayName", "") if isinstance(provider, dict)
            else c.get("publisher") or "",
            "published": str(c.get("pubDate") or ""),
            "url": link.get("url", "") if isinstance(link, dict) else c.get("link") or "",
        })
    return out


def get_news(ticker: str, company: str, days: int = NEWS_LOOKBACK_DAYS) -> list[dict]:
    """Recent headlines for a company; falls back to yfinance if NewsAPI fails."""
    try:
        items = _newsapi(f'"{company}" OR {ticker}', days, date.today().isoformat())
        if items:
            return items
    except Exception:  # noqa: BLE001 - use the backup source instead
        pass
    return _yfinance_news(ticker)
