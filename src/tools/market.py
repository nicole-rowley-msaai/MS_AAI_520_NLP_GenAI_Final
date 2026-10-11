"""Yahoo Finance: prices, key stats, financial statements, and technicals."""

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import PRICE_PERIOD
from src.tools.cache import memory

STAT_KEYS = [
    "shortName",
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "profitMargins",
    "returnOnEquity",
    "dividendYield",
    "beta",
    # Analyst consensus, so the writer can judge valuation against the
    # street's view rather than calling multiples "rich" unanchored.
    "targetMeanPrice",
    "recommendationKey",
    "numberOfAnalystOpinions",
]


@memory.cache
def get_price_history(ticker: str, period: str = PRICE_PERIOD) -> pd.DataFrame:
    """Daily Open, High, Low, Close, Volume, Dividends, Stock Splits."""
    df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No price data for {ticker}")
    df.index = df.index.tz_localize(None)
    return df


@memory.cache
def get_key_stats(ticker: str) -> dict:
    """A stable subset of yfinance .info fields."""
    info = yf.Ticker(ticker).info
    return {k: info.get(k) for k in STAT_KEYS}


@memory.cache
def get_financials(ticker: str) -> dict[str, pd.DataFrame]:
    """Annual income statement, balance sheet, and cash-flow statement."""
    t = yf.Ticker(ticker)
    return {
        "income": t.income_stmt,
        "balance": t.balance_sheet,
        "cashflow": t.cashflow,
    }


def compute_technicals(prices: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    """Summary numbers for the market agent, from price history alone."""
    close = prices["Close"]
    ret = close.pct_change().dropna()
    bench = benchmark["Close"].pct_change().dropna()
    both = pd.concat([ret, bench], axis=1, join="inner").dropna()
    beta = (
        np.cov(both.iloc[:, 0], both.iloc[:, 1])[0, 1] / both.iloc[:, 1].var()
    )

    def ret_over(days: int) -> float:
        return round(float(close.iloc[-1] / close.iloc[-days - 1] - 1), 4)

    return {
        "last_close": round(float(close.iloc[-1]), 2),
        "return_1m": ret_over(21),
        "return_1y": ret_over(252) if len(close) > 252 else None,
        "vol_annualized": round(float(ret.tail(252).std() * np.sqrt(252)), 4),
        "sma_50": round(float(close.tail(50).mean()), 2),
        "sma_200": round(float(close.tail(200).mean()), 2),
        "beta_vs_spy": round(float(beta), 3),
    }
