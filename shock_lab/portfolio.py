"""Clean and validate the holdings the user types in."""

from __future__ import annotations

import pandas as pd

MAX_HOLDINGS = 30


def clean_holdings(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (holdings with Ticker/Shares, list of warnings).

    Upper-cases tickers, drops blank or non-positive rows, merges duplicates,
    and enforces the 30-holding limit.
    """
    warnings: list[str] = []
    df = raw.copy()
    df["Ticker"] = df["Ticker"].astype("string").str.strip().str.upper()
    df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")
    df = df[df["Ticker"].notna() & (df["Ticker"] != "")]
    bad = df[~(df["Shares"] > 0)]
    if len(bad):
        warnings.append(f"Ignored rows without a positive share count: {', '.join(bad['Ticker'])}")
    df = df[df["Shares"] > 0]
    if df["Ticker"].duplicated().any():
        warnings.append("Merged duplicate tickers by adding their shares.")
        df = df.groupby("Ticker", as_index=False, sort=False)["Shares"].sum()
    if len(df) > MAX_HOLDINGS:
        warnings.append(f"Only the first {MAX_HOLDINGS} holdings are used.")
        df = df.head(MAX_HOLDINGS)
    return df[["Ticker", "Shares"]].reset_index(drop=True).astype({"Ticker": str}), warnings
