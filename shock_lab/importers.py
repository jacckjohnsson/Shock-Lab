"""Read holdings from a CSV: a simple Ticker/Shares file or a broker positions export.

Broker exports (Fidelity, Schwab, Vanguard, E*TRADE and similar) usually have a few title
lines before the header, cash and total rows mixed in with positions, and notes at the
bottom. This finds the header row, keeps rows that look like ticker symbols, and
skips the rest.
"""

from __future__ import annotations

import csv
import io
import re

import pandas as pd

SYMBOL_HEADERS = ("symbol", "ticker")
QUANTITY_HEADERS = ("quantity", "shares", "qty")
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.\-][A-Z0-9]{1,3})?$")
# Money market funds that brokers list as positions but behave like cash.
CASH_LIKE = {"SPAXX", "FDRXX", "FZFXX", "SPRXX", "FCASH", "CORE", "VMFXX", "VMMXX", "SWVXX", "SNVXX", "SNSXX"}

TEMPLATE_CSV = "Ticker,Shares\nAAPL,20\nXOM,30\nJPM,15\n"


def _find_column(header: list[str], names: tuple[str, ...]) -> int | None:
    cells = [h.strip().strip('"').lower() for h in header]
    for i, c in enumerate(cells):
        if any(n == c or c.startswith(n) or f" {n}" in c or f"({n}" in c for n in names):
            return i
    return None


def _to_number(text: str) -> float | None:
    cleaned = re.sub(r"[,$\s]", "", text or "")
    if cleaned.startswith("(") and cleaned.endswith(")"):  # accounting-style negative
        cleaned = "-" + cleaned[1:-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_holdings_csv(content: bytes | str) -> tuple[pd.DataFrame, list[str]]:
    """Return (DataFrame with Ticker and Shares, list of notes for the user)."""
    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    rows = list(csv.reader(io.StringIO(text)))

    header_idx = sym_col = qty_col = None
    for i, row in enumerate(rows[:50]):
        s, q = _find_column(row, SYMBOL_HEADERS), _find_column(row, QUANTITY_HEADERS)
        if s is not None and q is not None:
            header_idx, sym_col, qty_col = i, s, q
            break
    if header_idx is None:
        raise ValueError(
            "Couldn't find a symbol column and a shares/quantity column. "
            "The file needs headers like 'Symbol' or 'Ticker', and 'Quantity' or 'Shares'."
        )

    tickers, shares, skipped, cash = [], [], [], []
    for row in rows[header_idx + 1 :]:
        if len(row) <= max(sym_col, qty_col):
            continue
        raw_sym = row[sym_col].strip().strip('"')
        sym = raw_sym.rstrip("*").upper()
        if not sym:
            continue
        if sym in CASH_LIKE or raw_sym.endswith("**"):
            cash.append(sym)
            continue
        qty = _to_number(row[qty_col])
        if not TICKER_RE.match(sym) or qty is None:
            skipped.append(raw_sym)
            continue
        tickers.append(sym)
        shares.append(qty)

    notes = []
    if cash:
        notes.append(f"Left out cash and money market positions: {', '.join(dict.fromkeys(cash))}")
    if skipped:
        shown = list(dict.fromkeys(skipped))
        more = f" and {len(shown) - 6} more" if len(shown) > 6 else ""
        notes.append(f"Skipped rows that aren't stock or ETF positions: {', '.join(shown[:6])}{more}")
    return pd.DataFrame({"Ticker": tickers, "Shares": shares}), notes
