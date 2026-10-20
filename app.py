"""Shock Lab: what happens to my portfolio if oil, rates, the dollar or the market move?

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from shock_lab import data
from shock_lab.model import (
    FACTOR_LABELS,
    FACTORS,
    PERIODS_PER_YEAR,
    STRESS_QUANTILE,
    build_factor_returns,
    fit_all,
    holding_impacts,
    implied_factor_moves,
    last_years,
    portfolio_summary,
    rolling_betas,
    stock_returns,
    stress_periods,
    to_frequency,
)
from shock_lab.portfolio import MAX_HOLDINGS, clean_holdings
from shock_lab.replay import REPLAYS, replay_accuracy, run_replay
from shock_lab.sample import SAMPLE_BETAS, make_sample_data
from shock_lab.shocks import PRESETS

DISCLAIMER = (
    "**For education only, not investment advice.** Estimates come from a linear model fit to "
    "past data. Real markets can react very differently, especially in a crisis. "
    "Do not make investment decisions based on this tool alone."
)
GREEN, RED, GREY = "#2e7d32", "#c62828", "#9e9e9e"

st.set_page_config(page_title="Shock Lab", layout="wide")
st.title("Shock Lab")
st.caption("What happens to my portfolio if oil, interest rates, the dollar or the stock market move?")
st.warning(DISCLAIMER)


def money(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{'-' if x < 0 else '+'}${abs(x):,.0f}"


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:+.2f}%"


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading prices and macro data…")
def load_live(tickers: tuple[str, ...]):
    start, end = data.history_window()
    prices = data.fetch_prices(sorted(set(tickers) | {data.MARKET_TICKER}), start, end)
    macro = data.fetch_macro(start, end)
    market = prices[data.MARKET_TICKER]
    return market, macro, prices[[t for t in tickers if t in prices.columns]]


@st.cache_data
def load_sample():
    return make_sample_data()


# ---- Sidebar: data, model and scenario ----
with st.sidebar:
    st.header("Data")
    use_sample = st.toggle("Use made-up sample data (works offline)", value=False)

    st.header("Model")
    years = st.select_slider(
        "History used to fit betas",
        options=[1, 2, 3, 5, 10],
        value=5,
        format_func=lambda y: f"Last {y} year{'s' if y > 1 else ''}",
        help="Shorter = reflects how stocks behave lately but is noisier. Longer = more stable but slower to adapt.",
    )
    freq = st.radio(
        "Return frequency",
        ["daily", "weekly"],
        format_func=str.capitalize,
        horizontal=True,
        help="Weekly returns reduce noise from stock and macro data closing at different times of day.",
    )
    if freq == "weekly" and years < 2:
        years = 2
        st.caption("Weekly mode needs at least 2 years of data, so 2 years are used.")

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

# Keep results on screen while the user changes settings, once they've clicked Run.
if st.button("Run scenario", type="primary"):
    st.session_state["ran"] = True
if not st.session_state.get("ran"):
    st.stop()

# ---- Data and models ----
try:
    if use_sample:
        market, macro, prices = load_sample()
        prices = prices[[t for t in holdings["Ticker"] if t in prices.columns]]
    else:
        market, macro, prices = load_live(tuple(holdings["Ticker"]))
except Exception as e:  # network or data errors
    st.error(f"Could not load data: {e}")
    st.stop()

missing = [t for t in holdings["Ticker"] if t not in prices.columns]
if missing:
    st.warning(f"No price data for: {', '.join(missing)}. Check the ticker symbols.")

all_factors = build_factor_returns(to_frequency(market, freq), to_frequency(macro, freq))
all_returns = stock_returns(to_frequency(prices, freq))
factors = last_years(all_factors, years)
returns = last_years(all_returns, years)
models, errors = fit_all(returns, factors)
for t, msg in errors.items():
    st.warning(msg)
if not models:
    st.error("No holdings could be modeled. Try a longer history window.")
    st.stop()

# Stress betas need more data, so they use the full history.
stress_idx = stress_periods(all_factors)
stress_models, _ = fit_all(all_returns.loc[all_returns.index.isin(stress_idx)], all_factors.loc[stress_idx])

moves = implied_factor_moves(shock, factors) if co_move else pd.Series(shock)[FACTORS]
impacts = holding_impacts(models, holdings, prices.ffill().iloc[-1], moves, stress_models)
normal = portfolio_summary(impacts)
stress = portfolio_summary(impacts.dropna(subset=["Stress impact ($)"]), "Stress impact ($)")

tab_scn, tab_replay, tab_model = st.tabs(["Scenario", "Historical replays", "Model details"])

# ---- Scenario tab ----
with tab_scn:
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
    c1.metric("Portfolio value", f"${normal['value']:,.0f}")
    c2.metric("Normal times", money(normal["impact"]), f"{normal['impact_pct']:+.2f}%", delta_color="off")
    c3.metric("Stress times", money(stress["impact"]), f"{stress['impact_pct']:+.2f}%", delta_color="off")
    st.caption(
        f"**Normal times** uses betas from the last {years} year(s). **Stress times** uses betas fit only on "
        f"the worst {STRESS_QUANTILE:.0%} of market {freq} moves since 2006, when stocks often move together more."
    )

    chart_df = impacts.sort_values("Impact ($)")
    fig = go.Figure()
    fig.add_bar(y=chart_df["Ticker"], x=chart_df["Impact ($)"], orientation="h", name="Normal times",
                marker_color=[GREEN if v > 0 else RED for v in chart_df["Impact ($)"]])
    fig.add_bar(y=chart_df["Ticker"], x=chart_df["Stress impact ($)"], orientation="h", name="Stress times",
                marker_color=GREY)
    fig.update_layout(barmode="group", height=max(320, 44 * len(chart_df)), xaxis_title="Estimated impact ($)",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        impacts.style.format(
            {
                "Shares": "{:,.2f}",
                "Price": "${:,.2f}",
                "Value ($)": "${:,.0f}",
                "Impact (%)": pct,
                "Low (%)": pct,
                "High (%)": pct,
                "Impact ($)": money,
                "Stress impact (%)": pct,
                "Stress impact ($)": money,
                "R²": "{:.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Low/High is a rough 95% range from uncertainty in the estimated betas only. "
        "R² shows how much of a stock's movement the four factors explain (0 to 1)."
    )

# ---- Historical replays tab ----
with tab_replay:
    st.markdown(
        "How well would this model have predicted real events? For each event, betas are fit **only on the "
        f"{years} year(s) before it started**, then fed the factor moves that actually happened. "
        "Dollar amounts apply those % moves to your positions' value today."
    )
    event = st.selectbox("Event", list(REPLAYS))
    start, end = REPLAYS[event]
    try:
        table, ev_moves = run_replay(holdings, prices, market, macro, start, end, years, freq)
    except ValueError as e:
        st.info(f"Not enough data for this event: {e}")
    else:
        st.caption(
            "What the factors did: "
            + ", ".join(
                f"{FACTOR_LABELS[f].split(' (')[0]} {ev_moves[f]:+.2f} pts" if f == "rates"
                else f"{FACTOR_LABELS[f].split(' (')[0]} {ev_moves[f] * 100:+.1f}%"
                for f in FACTORS
            )
        )
        acc = replay_accuracy(table)
        if acc["n"]:
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted (portfolio)", money(acc["predicted_total"]),
                      f"{acc['predicted_total'] / acc['value'] * 100:+.1f}%", delta_color="off")
            r2.metric("Actual (portfolio)", money(acc["actual_total"]),
                      f"{acc['actual_total'] / acc['value'] * 100:+.1f}%", delta_color="off")
            r3.metric("Avg. miss per holding", f"{acc['mean_abs_error_pts']:.1f} pts")

            plot = table.dropna(subset=["Predicted (%)", "Actual (%)"])
            lim = max(plot[["Predicted (%)", "Actual (%)"]].abs().max().max() * 1.1, 5)
            sc = px.scatter(plot, x="Predicted (%)", y="Actual (%)", text="Ticker")
            sc.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim, line=dict(dash="dot", color=GREY))
            sc.update_traces(textposition="top center")
            sc.update_layout(height=420, xaxis_range=[-lim, lim], yaxis_range=[-lim, lim])
            st.plotly_chart(sc, width="stretch")
            st.caption("Dots on the dotted line were predicted perfectly. Above the line = did better than predicted.")
        skipped = table[table["Predicted (%)"].isna() | table["Actual (%)"].isna()]["Ticker"].tolist()
        if skipped:
            st.info(f"Not enough history before this event for: {', '.join(skipped)}")
        st.dataframe(
            table.style.format(
                {"Value today ($)": "${:,.0f}", "Predicted (%)": pct, "Actual (%)": pct,
                 "Predicted ($)": money, "Actual ($)": money}
            ),
            hide_index=True,
            width="stretch",
        )

# ---- Model details tab ----
with tab_model:
    beta_tbl = pd.DataFrame({t: m.betas for t, m in models.items()}).T
    beta_tbl.columns = [FACTOR_LABELS[f] for f in FACTORS]
    beta_tbl["R²"] = [m.r_squared for m in models.values()]
    beta_tbl["Periods of data"] = [m.n_obs for m in models.values()]
    st.markdown(f"**Betas ({freq}, last {years} year(s))**")
    st.dataframe(beta_tbl.style.format("{:.3f}", subset=beta_tbl.columns[:-1]))
    st.caption(
        "A beta of 0.30 on oil means the stock has tended to move about 0.3% for each 1% move in oil, "
        "holding the other factors fixed. The rates beta is per 1 percentage point change in the 10-year yield."
    )

    st.markdown("**How betas changed over time** (1-year rolling window, full history)")
    rc1, rc2 = st.columns(2)
    tick = rc1.selectbox("Holding", list(models))
    factor = rc2.selectbox("Factor", FACTORS, format_func=lambda f: FACTOR_LABELS[f])
    rb = rolling_betas(all_returns[tick], all_factors, PERIODS_PER_YEAR[freq])
    if rb.empty:
        st.info("Not enough history for a rolling chart.")
    else:
        line = px.line(rb, y=factor, labels={factor: f"Beta to {FACTOR_LABELS[factor]}", "index": ""})
        line.add_hline(y=models[tick].betas[factor], line_dash="dot", line_color=GREY,
                       annotation_text=f"Used now ({models[tick].betas[factor]:.2f})")
        line.update_layout(height=360)
        st.plotly_chart(line, width="stretch")

st.caption(DISCLAIMER)
