"""FRED: macroeconomic series for the market agent."""

import pandas as pd
from fredapi import Fred

from src.config import FRED_SERIES, MACRO_START, secret
from src.tools.cache import memory


@memory.cache
def get_macro_series(series_id: str, start: str = MACRO_START) -> pd.Series:
    """One FRED series, e.g. 'DGS10' for the 10-year Treasury yield."""
    fred = Fred(api_key=secret("FRED_API_KEY"))
    s = fred.get_series(series_id, observation_start=start)
    return s.dropna().rename(series_id)


def get_macro_snapshot(start: str = MACRO_START) -> pd.DataFrame:
    """Latest value and value one year earlier for each configured series."""
    rows = []
    for sid, label in FRED_SERIES.items():
        s = get_macro_series(sid, start)
        year_ago = s[s.index <= s.index[-1] - pd.DateOffset(years=1)]
        rows.append(
            {
                "series": sid,
                "label": label,
                "date": s.index[-1].date(),
                "latest": round(float(s.iloc[-1]), 3),
                "one_year_ago": (
                    round(float(year_ago.iloc[-1]), 3)
                    if len(year_ago)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)
