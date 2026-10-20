"""Shock Lab: what happens to my portfolio if oil, rates, the dollar or the market move?

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from shock_lab import data
from shock_lab.model import (
    FACTOR_LABELS,
    FACTORS,
    build_factor_returns,
    fit_all,
    holding_impacts,
    implied_factor_moves,
    portfolio_summary,
)
from shock_lab.portfolio import MAX_HOLDINGS, clean_holdings
from shock_lab.sample import SAMPLE_BETAS, make_sample_data
from shock_lab.shocks import PRESETS

DISCLAIMER = (
    "**For education only, not investment advice.** Estimates come from a linear model fit to "
    "past daily data. Real markets can react very differently, especially in a crisis. "
    "Do not make investment decisions based on this tool alone."
)

st.set_page_config(page_title="Shock Lab", layout="wide")
st.title("Shock Lab")
st.caption("What happens to my portfolio if oil, interest rates, the dollar or the stock market move?")
st.warning(DISCLAIMER)


def money(x: float) -> str:
    return f"{'-' if x < 0 else '+'}${abs(x):,.0f}"


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading prices and macro data…")
def load_live(tickers: tuple[str, ...], years: int):
    start, end = data.default_window(years)
    prices = data.fetch_prices(sorted(set(tickers) | {data.MARKET_TICKER}), start, end)
    macro = data.fetch_macro(start, end)
    market = prices[data.MARKET_TICKER]
    return market, macro, prices[[t for t in tickers if t in prices.columns]]


@st.cache_data
def load_sample():
    return make_sample_data()


# ---- Sidebar: data source and scenario ----
with st.sidebar:
    st.header("Data")
    use_sample = st.toggle("Use made-up sample data (works offline)", value=False)
    years = st.slider("Years of history", 3, 10, 10, disabled=use_sample)

    st.header("Scenario")
    preset = st.selectbox("Preset shock", list(PRESETS) + ["Custom"])
    base = PRESETS.get(preset, {})
    st.caption("Adjust any factor. Leave a factor at 0 to let the model fill it in (see below).")
    shock = {
        "market": st.slider(FACTOR_LABELS["market"], -40.0, 40.0, base.get("market", 0.0) * 100, 1.0) / 100,
        "oil": st.slider(FACTOR_LABELS["oil"], -60.0, 60.0, base.get("oil", 0.0) * 100, 1.0) / 100,
        "rates": st.slider(FACTOR_LABELS["rates"], -3.0, 3.0, base.get("rates", 0.0), 0.25),
        "dollar": st.slider(FACTOR_LABELS["dollar"], -20.0, 20.0, base.get("dollar", 0.0) * 100, 0.5) / 100,
    }
    co_move = st.toggle(
        "Include typical co-moves of the factors I didn't set",
        value=True,
        help="Example: when oil jumps, stocks and the dollar have historically moved too. "
        "On = estimate those moves from history. Off = hold them at 0.",
    )

# ---- Holdings input ----
st.subheader(f"1. Your holdings (up to {MAX_HOLDINGS})")
default_rows = (
    pd.DataFrame({"Ticker": list(SAMPLE_BETAS), "Shares": [50, 20, 40, 60]})
    if use_sample
    else pd.DataFrame({"Ticker": ["AAPL", "XOM", "JPM", "NEE", "DAL"], "Shares": [20, 30, 15, 25, 40]})
)
edited = st.data_editor(
    default_rows,
    num_rows="dynamic",
    key=f"holdings_{use_sample}",
    column_config={
        "Ticker": st.column_config.TextColumn(required=True),
        "Shares": st.column_config.NumberColumn(min_value=0.0, step=1.0, required=True),
    },
    width="stretch",
)
holdings, warnings = clean_holdings(edited)
for w in warnings:
    st.info(w)
if holdings.empty:
    st.stop()

if not st.button("Run scenario", type="primary"):
    st.stop()

# ---- Data and model ----
try:
    if use_sample:
        market, macro, prices = load_sample()
        prices = prices[[t for t in holdings["Ticker"] if t in prices.columns]]
    else:
        market, macro, prices = load_live(tuple(holdings["Ticker"]), years)
except Exception as e:  # network or data errors
    st.error(f"Could not load data: {e}")
    st.stop()

missing = [t for t in holdings["Ticker"] if t not in prices.columns]
if missing:
    st.warning(f"No price data for: {', '.join(missing)}")

factors = build_factor_returns(market, macro)
models, errors = fit_all(prices, factors)
for t, msg in errors.items():
    st.warning(msg)
if not models:
    st.error("No holdings could be modeled.")
    st.stop()

moves = implied_factor_moves(shock, factors) if co_move else pd.Series(shock)[FACTORS]
impacts = holding_impacts(models, holdings, prices.ffill().iloc[-1], moves)
summary = portfolio_summary(impacts)

# ---- Results ----
st.subheader("2. Scenario")
scenario_tbl = pd.DataFrame(
    {
        "Factor": [FACTOR_LABELS[f] for f in FACTORS],
        "You set": [shock[f] * (1 if f == "rates" else 100) for f in FACTORS],
        "Used in model": [moves[f] * (1 if f == "rates" else 100) for f in FACTORS],
    }
)
st.dataframe(scenario_tbl.style.format({"You set": "{:+.2f}", "Used in model": "{:+.2f}"}), hide_index=True)

st.subheader("3. Estimated impact")
c1, c2, c3 = st.columns(3)
c1.metric("Portfolio value", f"${summary['value']:,.0f}")
c2.metric("Estimated change", money(summary["impact"]))
c3.metric("Estimated change (%)", f"{summary['impact_pct']:+.2f}%")

chart = px.bar(
    impacts.sort_values("Impact ($)"),
    x="Impact ($)",
    y="Ticker",
    orientation="h",
    color=impacts.sort_values("Impact ($)")["Impact ($)"] > 0,
    color_discrete_map={True: "#2e7d32", False: "#c62828"},
    labels={"color": "Gain"},
)
chart.update_layout(showlegend=False, height=max(300, 28 * len(impacts)))
st.plotly_chart(chart, width="stretch")

st.dataframe(
    impacts.style.format(
        {
            "Shares": "{:,.2f}",
            "Price": "${:,.2f}",
            "Value ($)": "${:,.0f}",
            "Impact (%)": "{:+.2f}%",
            "Low (%)": "{:+.2f}%",
            "High (%)": "{:+.2f}%",
            "Impact ($)": money,
            "R²": "{:.2f}",
        }
    ),
    hide_index=True,
    width="stretch",
)
st.caption(
    "Low/High is a rough 95% range from uncertainty in the estimated betas only. "
    "R² shows how much of a stock's daily movement the four factors explain (0 to 1)."
)

with st.expander("Model details: betas for each holding"):
    beta_tbl = pd.DataFrame({t: m.betas for t, m in models.items()}).T
    beta_tbl.columns = [FACTOR_LABELS[f] for f in FACTORS]
    beta_tbl["Days of data"] = [m.n_obs for m in models.values()]
    st.dataframe(beta_tbl.style.format("{:.3f}", subset=beta_tbl.columns[:-1]))
    st.caption(
        "A beta of 0.30 on oil means the stock has tended to move about 0.3% for each 1% move in oil, "
        "holding the other factors fixed. The rates beta is per 1 percentage point change in the 10-year yield."
    )

st.caption(DISCLAIMER)
