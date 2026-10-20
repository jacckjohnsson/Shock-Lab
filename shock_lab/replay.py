"""Historical replays: compare what the model would have predicted with what actually happened."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import (
    FACTORS,
    build_factor_returns,
    fit_all,
    last_years,
    predicted_return,
    stock_returns,
    to_frequency,
)

REPLAYS = {
    "2008 financial crisis (Sep 2008 to Mar 2009)": ("2008-09-02", "2009-03-09"),
    "COVID crash (Feb to Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 rate shock (Jan to Oct 2022)": ("2022-01-03", "2022-10-12"),
}
MAX_START_GAP_DAYS = 7  # a ticker must have a price within a week of the start date


def _endpoints(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float] | None:
    s = series.loc[start:end].dropna()
    if s.empty or (s.index[0] - start).days > MAX_START_GAP_DAYS:
        return None
    return float(s.iloc[0]), float(s.iloc[-1])


def period_factor_moves(market: pd.Series, macro: pd.DataFrame, start, end) -> pd.Series:
    """Total factor moves over a period, in the same units as a shock."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    moves = {}
    for name, series, kind in [
        ("market", market, "pct"),
        ("oil", macro["oil"], "pct"),
        ("rates", macro["rates"], "diff"),
        ("dollar", macro["dollar"], "pct"),
    ]:
        ends = _endpoints(series, start, end)
        if ends is None:
            raise ValueError(f"No {name} data at the start of this period.")
        a, b = ends
        moves[name] = b - a if kind == "diff" else b / a - 1
    return pd.Series(moves)[FACTORS]


def run_replay(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.Series,
    macro: pd.DataFrame,
    start,
    end,
    years: float,
    freq: str = "daily",
) -> tuple[pd.DataFrame, pd.Series]:
    """Fit betas only on data BEFORE the event, then predict the event.

    Returns (per-holding table, factor moves during the event). Dollar columns
    apply the % moves to today's position values.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    moves = period_factor_moves(market, macro, start, end)

    factors = build_factor_returns(to_frequency(market, freq), to_frequency(macro, freq))
    returns = stock_returns(to_frequency(prices, freq))
    factors_before = last_years(factors, years, end=start)
    models, _ = fit_all(last_years(returns, years, end=start), factors_before)

    last_prices = prices.ffill().iloc[-1]
    rows = []
    for _, h in holdings.iterrows():
        t = h["Ticker"]
        if t not in prices.columns:
            continue
        value = float(h["Shares"]) * float(last_prices[t])
        ends = _endpoints(prices[t], start, end)
        actual = ends[1] / ends[0] - 1 if ends else np.nan
        pred = predicted_return(models[t], moves)[0] if t in models else np.nan
        rows.append(
            {
                "Ticker": t,
                "Value today ($)": value,
                "Predicted (%)": pred * 100,
                "Actual (%)": actual * 100,
                "Predicted ($)": value * pred,
                "Actual ($)": value * actual,
            }
        )
    return pd.DataFrame(rows), moves


def replay_accuracy(table: pd.DataFrame) -> dict[str, float]:
    """Simple accuracy stats on holdings that have both a prediction and an actual."""
    ok = table.dropna(subset=["Predicted (%)", "Actual (%)"])
    if ok.empty:
        return {"n": 0}
    err = ok["Predicted (%)"] - ok["Actual (%)"]
    out = {
        "n": len(ok),
        "mean_abs_error_pts": float(err.abs().mean()),
        "predicted_total": float(ok["Predicted ($)"].sum()),
        "actual_total": float(ok["Actual ($)"].sum()),
        "value": float(ok["Value today ($)"].sum()),
    }
    if len(ok) >= 3:
        out["correlation"] = float(np.corrcoef(ok["Predicted (%)"], ok["Actual (%)"])[0, 1])
    return out
