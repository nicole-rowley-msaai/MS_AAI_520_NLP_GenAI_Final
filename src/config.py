"""Project settings: API keys, model names, tickers, and paths.

Keys are read from a .env file at the repo root (see .env.example).
Change model names here so the whole team switches at once.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# API keys ------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "")  # "Name email@x.com"


def secret(name: str) -> str:
    """Read a key at call time, so it works even if it was set after import.

    Colab notebooks load secrets into os.environ in a setup cell; if any
    module imported src.config before that cell ran, the module-level
    constants above are empty. Tools call this instead.
    """
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to .env, or in Colab add it to "
            "Secrets and re-run the setup cell."
        )
    return value


# Models: confirm exact API ids in each provider's docs ----------------------
MODELS = {
    "chain": os.getenv("CHAIN_MODEL", "gpt-6-luna"),  # news chain, router
    "writer": os.getenv(
        "WRITER_MODEL", "claude-sonnet-5"
    ),  # planner, synthesis
    "evaluator": os.getenv("EVALUATOR_MODEL", "gpt-6-sol"),
    "embeddings": "all-MiniLM-L6-v2",
}

# Scope ---------------------------------------------------------------------
TICKERS = ["NVDA", "JPM", "XOM"]
COMPANIES = {"NVDA": "Nvidia", "JPM": "JPMorgan", "XOM": "Exxon"}
BENCHMARK = "SPY"
PRICE_PERIOD = "5y"
MACRO_START = "2021-01-01"
NEWS_LOOKBACK_DAYS = 30

# When a company reorganizes, its ticker moves to a new SEC registrant that
# has no filing history. Map the ticker to the prior entity's CIK so the
# latest 10-K can still be found. Exxon became ExxonMobil Holdings Corp
# (CIK 2115436) in mid-2026; Exxon Mobil Corp (CIK 34088) filed the 10-K.
PREDECESSOR_CIKS = {"XOM": "0000034088"}

FRED_SERIES = {
    "CPIAUCSL": "CPI, all urban consumers",
    "FEDFUNDS": "Effective federal funds rate",
    "DGS10": "10-year Treasury yield",
    "UNRATE": "Unemployment rate",
    "GDPC1": "Real GDP",
}

# Paths ---------------------------------------------------------------------
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
KAGGLE_CSV = DATA_DIR / "kaggle" / "all-data.csv"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
