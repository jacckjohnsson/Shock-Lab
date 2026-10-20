"""Download prices from Yahoo Finance and macro series from FRED."""

from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd
import requests

# FRED series used as macro factors.
FRED_SERIES = {
    "oil": "DCOILWTICO",  # WTI crude oil spot price, $/barrel
    "rates": "DGS10",  # 10-year Treasury yield, percent
    "dollar": "DTWEXBGS",  # Broad trade-weighted US dollar index
}
MARKET_TICKER = "SPY"  # proxy for "the market" (S&P 500)
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def default_window(years: int = 10) -> tuple[date, date]:
    end = date.today()
    return end - timedelta(days=365 * years + years // 4), end


def fetch_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Daily adjusted close prices, one column per ticker."""
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
    )
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no data. Check the tickers and your internet connection.")
    close = raw["Close"] if "Close" in raw else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(axis=1, how="all").sort_index()


def parse_fred_csv(text: str, name: str) -> pd.Series:
    """Parse FRED's CSV download. Missing values are written as '.'."""
    df = pd.read_csv(io.StringIO(text))
    series = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    series.index = pd.to_datetime(df.iloc[:, 0])
    series.name = name
    return series.dropna().sort_index()


def fetch_fred_series(series_id: str, name: str, timeout: float = 30) -> pd.Series:
    resp = requests.get(FRED_CSV_URL.format(series_id=series_id), timeout=timeout)
    resp.raise_for_status()
    return parse_fred_csv(resp.text, name)


def fetch_macro(start: date, end: date) -> pd.DataFrame:
    """Oil price, 10-year yield and dollar index levels, one column each."""
    cols = [fetch_fred_series(sid, name) for name, sid in FRED_SERIES.items()]
    df = pd.concat(cols, axis=1)
    return df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
