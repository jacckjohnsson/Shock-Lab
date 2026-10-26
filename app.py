"""Shock Lab: what happens to my portfolio if oil, rates, the dollar or the market move?

Run with:  streamlit run app.py
"""

from __future__ import annotations

from dataclasses import replace
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from shock_lab import brand, data, landing
from shock_lab.brand import AMBER, BORDER, CYAN, GREEN, LAVENDER, MUTED, RED, TEXT
from shock_lab.compare import compare_scenarios
from shock_lab.frontier import RETURN_METHODS, compare_portfolios, expected_returns, frontier_curve, model_portfolios
from shock_lab.hedging import (
    CONTRACT_SIZE,
    HEDGES,
    HedgeSetup,
    compare_hedges,
    hedge_pnl,
    payoff_table,
    put_greeks_table,
    vix_market_beta,
)
from shock_lab.importers import TEMPLATE_CSV, parse_holdings_csv
from shock_lab.model import (
    DRIVERS,
    FACTOR_LABELS,
    FACTORS,
    PERIODS_PER_YEAR,
    SCENARIO_VARS,
    STRESS_QUANTILE,
    build_factor_returns,
    display_units,
    fit_all,
    holding_impacts,
    implied_factor_moves,
    last_years,
    move_text,
    portfolio_summary,
    rolling_betas,
    stock_returns,
    stress_periods,
    to_frequency,
)
from shock_lab.portfolio import MAX_HOLDINGS, clean_holdings
from shock_lab.montecarlo import HORIZONS, simulate, stats_table
from shock_lab.replay import REPLAYS, replay_accuracy, run_replay
from shock_lab.report import to_excel, to_pdf
from shock_lab.risk import concentration, stock_covariance
from shock_lab.sample import SAMPLE_BETAS, make_sample_data
from shock_lab.shocks import PRESETS

DISCLAIMER = (
    "For education only, not investment advice. Estimates come from a linear model fit to "
    "past data. Real markets can react very differently, especially in a crisis. "
    "Do not make investment decisions based on this tool alone."
)
HEDGE_COLORS = {"No hedge": TEXT, "Index puts": AMBER, "Put spread": CYAN, "Inverse ETF": LAVENDER, "Raise cash": MUTED}

st.set_page_config(page_title="Shock Lab", page_icon=str(Path(__file__).parent / "assets" / "icon.png"), layout="wide")
st.html(brand.CSS)
st.session_state.setdefault("use_sample", False)


def start(demo: bool = False) -> None:
    st.session_state["started"] = True
    if demo:  # show results right away with the made-up portfolio
        st.session_state["use_sample"] = True
        st.session_state["ran"] = True


# ---- Welcome screen (the landing page), shown until someone starts ----
if not st.session_state.get("started"):
    st.html(landing.CSS)
    st.html(landing.brand_bar_html())
    hero_l, hero_r = st.columns([1, 1.1], gap="large", vertical_alignment="center")
    with hero_l:
        st.html(landing.hero_html())
        c1, c2 = st.columns(2)
        c1.button("Run a stress test", type="primary", on_click=start, width="stretch")
        c2.button("Try a demo portfolio", on_click=start, args=(True,), width="stretch")
        st.html(landing.hero_note_html())
    with hero_r:
        st.html(landing.hero_shot_html())
    st.html(landing.body_html())
    _, mid, _ = st.columns([2, 1, 2])
    mid.button("Launch Shock Lab", type="primary", on_click=start, key="start_bottom", width="stretch")
    st.html(landing.footer_html())
    st.stop()

st.html(brand.header_html())
steps_slot = st.empty()


def money(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{'-' if x < 0 else '+'}${abs(x):,.0f}"


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:+.2f}%"


def gain_loss_color(v) -> str:
    if not isinstance(v, (int, float)) or pd.isna(v) or v == 0:
        return ""
    return f"color: {GREEN}" if v > 0 else f"color: {RED}"


def show_table(df: pd.DataFrame, formats: dict, signed: list[str] | None = None, **kwargs) -> None:
    styler = df.style.format(formats)
    cols = [c for c in (signed or []) if c in df.columns]
    if cols:
        styler = styler.map(gain_loss_color, subset=cols)
    st.dataframe(styler, width="stretch", **kwargs)


def show_chart(fig: go.Figure, height: int | None = None, **layout) -> None:
    fig = brand.style_chart(fig, height)
    if layout:
        fig.update_layout(**layout)
    st.plotly_chart(fig, theme=None, width="stretch")


def step(n: int, title: str, subtitle: str = "") -> None:
    st.html(brand.step_html(n, title, subtitle))


def describe_moves(mv: pd.Series) -> str:
    return ", ".join(move_text(v, mv[v]) for v in SCENARIO_VARS if v in mv.index)


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading prices and macro data…")
def load_live(tickers: tuple[str, ...]):
    start, end = data.history_window()
    prices = data.fetch_prices(sorted(set(tickers) | {data.MARKET_TICKER, data.GOLD_TICKER}), start, end)
    macro = data.fetch_macro(start, end)
    market = prices[data.MARKET_TICKER]
    if data.GOLD_TICKER not in prices:
        raise ValueError("Yahoo Finance returned no gold (GLD) prices.")
    macro = macro.reindex(macro.index.union(prices.index))
    macro["gold"] = prices[data.GOLD_TICKER]
    return market, macro, prices[[t for t in tickers if t in prices.columns]]


@st.cache_data
def load_sample():
    return make_sample_data()


# ---- Sidebar: model settings ----
with st.sidebar:
    st.html(f'<img src="{brand.logo_data_uri()}" width="56" height="28" alt=""/> '
            f'<span style="font-family:{brand.MONO};letter-spacing:.14em;font-weight:600">SHOCK LAB</span>')
    st.button("← Back to the overview", on_click=lambda: st.session_state.update(started=False))
    st.header("Model settings")
    use_sample = st.toggle("Use made-up sample data (works offline)", key="use_sample")
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
    co_move = st.toggle(
        "Include typical co-moves of factors you don't set",
        value=True,
        help="Example: when oil jumps, stocks and the dollar have historically moved too. "
        "On = estimate those moves from history. Off = hold them at 0.",
    )

ran = st.session_state.get("ran", False)
steps_slot.html(brand.steps_html(len(brand.STEPS) if ran else 2))

# ---- Step 1: holdings ----
step(1, "Your holdings", f"Up to {MAX_HOLDINGS} stocks or ETFs. Type them in, or upload a file from your broker.")
mode = st.radio(
    "How do you want to add holdings?", ["Type them in", "Upload a CSV"], horizontal=True, label_visibility="collapsed"
)
column_config = {
    "Ticker": st.column_config.TextColumn(required=True),
    "Shares": st.column_config.NumberColumn(min_value=0.0, step=1.0, required=True),
}
if mode == "Type them in":
    default_rows = (
        pd.DataFrame({"Ticker": list(SAMPLE_BETAS), "Shares": [50, 20, 40, 60]})
        if use_sample
        else pd.DataFrame({"Ticker": ["AAPL", "XOM", "JPM", "NEE", "DAL"], "Shares": [20, 30, 15, 25, 40]})
    )
    edited = st.data_editor(
        default_rows, num_rows="dynamic", key=f"holdings_{use_sample}", column_config=column_config, width="stretch"
    )
else:
    st.caption(
        "Upload a positions export from your broker (Fidelity, Schwab, Vanguard, E*TRADE and others), "
        "or any CSV with a Ticker/Symbol column and a Shares/Quantity column. Your file stays on this computer."
    )
    up_col, tmpl_col = st.columns([3, 1])
    uploaded = up_col.file_uploader("Holdings CSV", type=["csv", "txt"], label_visibility="collapsed")
    tmpl_col.download_button("Download a template", TEMPLATE_CSV, "shock_lab_holdings_template.csv", "text/csv")
    if uploaded is None:
        st.stop()
    try:
        parsed, notes = parse_holdings_csv(uploaded.getvalue())
    except ValueError as e:
        st.error(str(e))
        st.stop()
    for n in notes:
        st.info(n)
    st.caption(f"Found {len(parsed)} positions. You can edit them below before running.")
    edited = st.data_editor(
        parsed, num_rows="dynamic", key=f"upload_{uploaded.name}_{uploaded.size}", column_config=column_config,
        width="stretch",
    )
holdings, warnings = clean_holdings(edited)
for w in warnings:
    st.info(w)
if holdings.empty:
    st.stop()

# ---- Step 2: scenario ----
step(2, "Choose a scenario", "Pick a preset or set your own moves. Anything left at 0 is filled in from history.")
preset_names = list(PRESETS) + ["Custom"]
preset = st.pills("Preset", preset_names, default="S&P 500 -15%", label_visibility="collapsed") or "Custom"
base = PRESETS.get(preset, {})
# (low, high, step) in the units shown on the slider
SLIDER_RANGES = {
    "market": (-40.0, 40.0, 1.0), "oil": (-60.0, 60.0, 1.0), "rates": (-3.0, 3.0, 0.25), "dollar": (-20.0, 20.0, 0.5),
    "gold": (-30.0, 30.0, 1.0), "vix": (-15.0, 40.0, 1.0), "credit": (-1.5, 4.0, 0.25),
}
shock = {}
for row in (FACTORS[:4], FACTORS[4:] + DRIVERS):
    for col, v in zip(st.columns(4), row):
        lo, hi, stp = SLIDER_RANGES[v]
        unit = display_units(v)
        shock[v] = col.slider(FACTOR_LABELS[v], lo, hi, base.get(v, 0.0) * unit, stp, key=f"sl_{v}_{preset}") / unit
st.caption("VIX and credit spreads are stress gauges, not betas: a move in either is translated into the S&P, oil, "
           "rates, dollar and gold moves that have typically come with it.")
is_preset = preset in PRESETS and shock == {v: base.get(v, 0.0) for v in SCENARIO_VARS}
scenario_name = preset if is_preset else "your custom scenario"

saved = st.session_state.setdefault("saved_scenarios", {})
n1, n2, n3 = st.columns([3, 1, 1], vertical_alignment="bottom")
name = n1.text_input("Save this scenario to compare later", placeholder="e.g. Oil spike + rate hike")
if n2.button("Save scenario", disabled=not name.strip(), width="stretch"):
    saved[name.strip()] = dict(shock)
    st.toast(f"Saved \"{name.strip()}\". It appears in step 7.")
if saved and n3.button("Clear saved", width="stretch"):
    saved.clear()

# Keep results on screen while settings change, once Run has been clicked.
if st.button("Run analysis", type="primary"):
    st.session_state["ran"] = True
    st.rerun()
if not ran:
    st.stop()

# ---- Data and models ----
try:
    if use_sample:
        market, macro, prices = load_sample()
        prices = prices[[t for t in holdings["Ticker"] if t in prices.columns]]
    else:
        market, macro, prices = load_live(tuple(holdings["Ticker"]))
except Exception as e:  # network or data errors
    st.error(f"Could not load data: {e}. Turn on the sample data in the sidebar to try the app without live data.")
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

moves = implied_factor_moves(shock, factors, co_move).reindex(SCENARIO_VARS).fillna(0.0)
last_prices = prices.ffill().iloc[-1]
impacts = holding_impacts(models, holdings, last_prices, moves, stress_models)
normal = portfolio_summary(impacts)
stress = portfolio_summary(impacts.dropna(subset=["Stress impact ($)"]), "Stress impact ($)")

# ---- Step 3: impact ----
step(3, "Estimated impact", f"Betas from the last {years} year(s) of {freq} returns.")
summary_slot = st.empty()  # filled in after the hedges are priced in step 4

m1, m2, m3 = st.columns(3)
m1.metric("Portfolio value", f"${normal['value']:,.0f}")
m2.metric("Normal times", money(normal["impact"]), f"{normal['impact_pct']:+.2f}%", delta_color="off")
m3.metric("Stress times", money(stress["impact"]), f"{stress['impact_pct']:+.2f}%", delta_color="off")

left, right = st.columns([2, 1])
with left:
    chart_df = impacts.sort_values("Impact ($)")
    fig = go.Figure()
    fig.add_bar(y=chart_df["Ticker"], x=chart_df["Impact ($)"], orientation="h", name="Normal times",
                marker_color=[GREEN if v > 0 else RED for v in chart_df["Impact ($)"]])
    fig.add_bar(y=chart_df["Ticker"], x=chart_df["Stress impact ($)"], orientation="h", name="Stress times",
                marker_color=CYAN, opacity=0.8)
    fig.update_layout(barmode="group", xaxis_title="Estimated impact ($)")
    show_chart(fig, max(320, 44 * len(chart_df)))
with right:
    st.markdown("**Factor moves**")
    scenario_tbl = pd.DataFrame(
        {
            "Factor": [FACTOR_LABELS[v] for v in SCENARIO_VARS],
            "You set": [shock[v] * display_units(v) for v in SCENARIO_VARS],
            "Used": [moves[v] * display_units(v) for v in SCENARIO_VARS],
        }
    )
    show_table(scenario_tbl, {"You set": "{:+.2f}", "Used": "{:+.2f}"}, ["You set", "Used"], hide_index=True)
    st.caption(
        f"**Stress times** uses betas fit only on the worst {STRESS_QUANTILE:.0%} of market {freq} moves since 2006, "
        "when stocks tend to move together more."
    )

show_table(
    impacts,
    {"Shares": "{:,.2f}", "Price": "${:,.2f}", "Value ($)": "${:,.0f}", "Impact (%)": pct, "Low (%)": pct,
     "High (%)": pct, "Impact ($)": money, "Stress impact (%)": pct, "Stress impact ($)": money, "R²": "{:.2f}"},
    ["Impact (%)", "Impact ($)", "Stress impact (%)", "Stress impact ($)"],
    hide_index=True,
)
st.caption(
    "Low/High is a rough 95% range from uncertainty in the estimated betas only. "
    "R² shows how much of a stock's movement the five factors explain (0 to 1)."
)

st.markdown("**Concentration: is one holding running the show?**")
ppy = PERIODS_PER_YEAR[freq]
risk_tickers = [t for t in impacts["Ticker"] if t in models]
position_values = impacts.set_index("Ticker")["Value ($)"]
cov = stock_covariance(models, risk_tickers, factors, ppy)
conc_table, conc, conc_warnings = concentration(position_values, cov)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Yearly volatility", f"{conc['vol']:.1f}%",
          help="How much the portfolio typically swings in a year (one standard deviation), from the factor model.")
c2.metric("Acts like", f"{conc['effective_n']:.1f} positions",
          help="1 / sum of squared weights. Ten equal holdings = 10. One big holding pulls this toward 1.")
c3.metric("Largest position", f"{conc['top_weight']:.0f}%")
c4.metric("Largest share of risk", f"{conc['top_risk']:.0f}%")
if conc_warnings:
    st.warning("\n".join(f"- {w}" for w in conc_warnings))
else:
    st.success("No single holding dominates this portfolio's value or risk.")
cl, cr = st.columns([1, 1])
with cl:
    conc_fig = go.Figure()
    conc_fig.add_bar(x=conc_table["Ticker"], y=conc_table["Weight (%)"], name="Share of value", marker_color=AMBER)
    conc_fig.add_bar(x=conc_table["Ticker"], y=conc_table["Share of risk (%)"], name="Share of risk",
                     marker_color=LAVENDER)
    conc_fig.update_layout(barmode="group", yaxis_title="%")
    show_chart(conc_fig, 300)
with cr:
    show_table(conc_table, {"Value ($)": "${:,.0f}", "Weight (%)": "{:.1f}%", "Share of risk (%)": "{:.1f}%",
                            "Volatility (%/yr)": "{:.1f}%"}, hide_index=True)
st.caption(
    "Share of risk = how much of the portfolio's day-to-day swings come from each holding. A volatile stock can "
    "carry far more risk than its share of value. Warnings appear above 25% of value, 35% of risk, or when the "
    "portfolio acts like fewer than 4 equal positions."
)

# ---- Step 4: hedging ----
step(4, "Hedge it", "How common hedges would change this result. A model of how hedges behave, not a recommendation.")
vix_series = macro["vix"].dropna() if "vix" in macro else pd.Series(dtype=float)
tbill_series = macro["tbill"].dropna() if "tbill" in macro else pd.Series(dtype=float)
if vix_series.empty:
    st.info("VIX data is unavailable, so a VIX of 20 is assumed.")
vix_now = float(vix_series.iloc[-1]) if not vix_series.empty else 20.0
rate_now = float(tbill_series.iloc[-1]) / 100 if not tbill_series.empty else 0.04
daily_mkt = last_years(market.pct_change(fill_method=None).dropna().to_frame("m"), years)["m"]
vbeta = vix_market_beta(vix_series.reindex(market.index).ffill(), daily_mkt) if not vix_series.empty else -120.0

h1, h2, h3, h4, h5 = st.columns(5)
hedge_ratio = h1.slider("Share of market exposure to hedge", 0, 100, 50, 5, format="%d%%") / 100
strike_pct = h2.select_slider(
    "Put strike", options=[0.85, 0.90, 0.95, 1.00], value=0.95,
    format_func=lambda k: "At the money" if k == 1.0 else f"{(1 - k):.0%} below",
)
expiry_days = h3.select_slider(
    "Put expiry", options=[30, 90, 180, 365], value=90,
    format_func=lambda d: {30: "1 month", 90: "3 months", 180: "6 months", 365: "1 year"}[d],
)
horizon_label = h4.selectbox("Shock happens", ["Instantly", "In 1 week", "In 1 month"])
regime = h5.selectbox("Betas", ["Normal times", "Stress times"], key="hedge_regime")

use_models = stress_models if regime == "Stress times" else models
tickers = [t for t in impacts["Ticker"] if t in use_models]
values = impacts.set_index("Ticker").loc[tickers, "Value ($)"].values
B = pd.DataFrame({t: use_models[t].betas for t in tickers}).T[FACTORS].values


def portfolio_impact_for(market_move: float) -> float:
    m = {"market": market_move}
    mv = implied_factor_moves(m, factors, co_move)
    return float(values @ (B @ mv[FACTORS].values))


dollar_beta = portfolio_impact_for(0.01) / 0.01
scenario_impact = float(values @ (B @ moves[FACTORS].values))
spot = float(market.dropna().iloc[-1])
setup = HedgeSetup(
    spot=spot, vix=vix_now, rate=rate_now, dollar_beta=dollar_beta, hedge_ratio=hedge_ratio,
    strike_pct=strike_pct, expiry_days=expiry_days,
    horizon_days={"Instantly": 0, "In 1 week": 7, "In 1 month": 30}[horizon_label], vix_beta=vbeta,
)
mkt_move = float(moves["market"])
# A VIX shock you set overrides the usual VIX reaction to the market move.
vix_after = float(np.clip(vix_now + (shock["vix"] if shock["vix"] else vbeta * mkt_move), 9, 80))
setup.vix_extra = shock["vix"] - vbeta * mkt_move if shock["vix"] else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Exposure per 1% S&P", f"${dollar_beta / 100:,.0f}",
          help="Dollars the portfolio tends to move for each 1% move in the S&P 500, including the typical "
          "co-moves of oil, rates and the dollar if that setting is on.")
k2.metric("SPY price", f"${spot:,.2f}")
k3.metric("VIX now → scenario", f"{vix_now:.0f} → {vix_after:.0f}")
k4.metric("S&P move in scenario", f"{mkt_move * 100:+.1f}%")

if abs(mkt_move) < 0.005:
    st.info("This scenario barely moves the S&P 500, so S&P hedges do little here. "
            "Try the S&P 500 -15% preset, or turn on co-moves in the sidebar.")
n_contracts, exact = setup.contracts()
over_hedged = exact > 0 and abs(n_contracts - exact) / exact > 0.25
if over_hedged:
    st.warning(
        f"One SPY option covers about ${CONTRACT_SIZE * spot:,.0f} of the S&P 500. Hedging {hedge_ratio:.0%} "
        f"of your exposure needs {exact:.2f} contracts, so {n_contracts} contract(s) "
        f"{'over' if n_contracts > exact else 'under'}-hedges by a lot. For smaller portfolios, the "
        "inverse ETF or raising cash can be sized to the exact dollar amount."
    )

hedge_table = compare_hedges(setup, mkt_move, scenario_impact, 0.10, portfolio_impact_for(0.10))
up_col = [c for c in hedge_table.columns if c.startswith("Hedge if S&P")][0]
show_table(
    hedge_table,
    {"Upfront cost ($)": "${:,.0f}", "Cash needed ($)": "${:,.0f}", "Hedge gain/loss ($)": money,
     "Portfolio + hedge ($)": money, "Loss offset (%)": lambda v: "n/a" if pd.isna(v) else f"{v:.0f}%", up_col: money},
    ["Hedge gain/loss ($)", "Portfolio + hedge ($)", up_col],
    hide_index=True,
)
st.caption(
    "Upfront cost = option premium paid today. Cash needed = money tied up in the hedge. "
    "The last column shows what each hedge costs if the S&P rises 10% instead: the price of protection."
)

grid = np.round(np.arange(-0.30, 0.2001, 0.01), 2)
pay = payoff_table(setup, portfolio_impact_for, grid)
pay_fig = px.line(pay, x="S&P 500 move (%)", y="Change ($)", color="Hedge", category_orders={"Hedge": HEDGES},
                  color_discrete_map=HEDGE_COLORS, labels={"Change ($)": "Portfolio + hedge change ($)"})
pay_fig.update_traces(selector=dict(name="No hedge"), line=dict(width=3, dash="dot"))
pay_fig.add_vline(x=mkt_move * 100, line_dash="dot", line_color=AMBER, annotation_text="This scenario",
                  annotation_font_color=AMBER)
pay_fig.add_hline(y=0, line_color=BORDER, line_width=1)
show_chart(pay_fig, 420)
st.caption("Each line is your portfolio plus that hedge across different S&P 500 moves. "
           "Flatter on the left = more protection. Lower on the right = more upside given up.")

st.markdown("**Option greeks for the put hedges**")
show_table(
    put_greeks_table(setup),
    {"Implied vol used (%)": "{:.1f}", "Delta (SPY shares)": "{:+.1f}", "Dollar delta ($ per 1% S&P move)": money,
     "Gamma (delta change per $1 SPY)": "{:.3f}", "Vega ($ per 1 vol point)": "${:,.0f}", "Theta ($ per day)": money},
    hide_index=True,
)
st.caption(
    "Delta: how many SPY shares the position acts like. Gamma: how fast delta grows as SPY moves. "
    "Vega: gain if volatility rises 1 point. Theta: value lost per day as time passes. "
    "Prices use Black-Scholes with the VIX as volatility plus a simple skew for lower strikes, so real quotes will differ."
)

# ---- Step 5: Monte Carlo ----
step(5, "Simulate thousands of futures",
     "Monte Carlo: replay random stretches of real market history through your portfolio to see the range of outcomes.")
mc1, mc2, mc3 = st.columns([1, 2, 2])
horizon = mc1.selectbox("Horizon", list(HORIZONS), index=1)
full_label = "All history since 2006 (includes 2008 and 2020)"
source = mc2.radio("Draw market history from", [full_label, f"Last {years} year(s) only"], horizontal=True)
hedge_options = [h for h in HEDGES if h != "No hedge"]
overlay = mc3.selectbox("Compare with a hedge from step 4", ["None"] + hedge_options, index=1)
trading_days, calendar_days = HORIZONS[horizon]
mc_history = all_factors if source == full_label else factors
try:
    sim = simulate(models, position_values, mc_history, trading_days, freq)
except ValueError as e:
    st.info(str(e))
    sim = None

mc_stats = pd.DataFrame()
if sim is not None:
    results = {"No hedge": sim.pnl}
    if overlay != "None":
        if overlay in ("Index puts", "Put spread") and calendar_days > expiry_days:
            st.info(f"The {expiry_days}-day put expires before the {horizon} horizon. Pick a longer put expiry in "
                    "step 4 to compare it here.")
        else:
            mc_setup = replace(setup, horizon_days=calendar_days, vix_extra=0.0)
            results[overlay] = sim.pnl + hedge_pnl(overlay, mc_setup, sim.market_move, sim.pnl)
    mc_stats = stats_table(results, sim.value)
    base_stats = mc_stats.iloc[0]
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Typical outcome", money(base_stats["Typical outcome ($)"]))
    q2.metric("Bad case (1 in 20)", money(base_stats["Bad case, 1 in 20 ($)"]),
              help="95% value at risk: 1 in 20 simulations lost more than this.")
    q3.metric("Average of worst 5%", money(base_stats["Average of worst 5% ($)"]),
              help="Expected shortfall (CVaR): the average loss in the worst 1 in 20 simulations.")
    q4.metric("Chance of losing 10%+", f"{base_stats['Chance of losing 10%+ (%)']:.0f}%")

    fl, fr = st.columns([3, 2])
    with fl:
        b = sim.bands
        fan = go.Figure()
        fan.add_scatter(x=b["Trading days"], y=b["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fan.add_scatter(x=b["Trading days"], y=b["p5"], fill="tonexty", fillcolor="rgba(245,166,35,0.15)",
                        line=dict(width=0), name="90% of outcomes")
        fan.add_scatter(x=b["Trading days"], y=b["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fan.add_scatter(x=b["Trading days"], y=b["p25"], fill="tonexty", fillcolor="rgba(245,166,35,0.35)",
                        line=dict(width=0), name="Middle 50%")
        fan.add_scatter(x=b["Trading days"], y=b["p50"], line=dict(color=AMBER, width=2), name="Typical")
        fan.add_hline(y=0, line_color=BORDER, line_width=1)
        fan.update_layout(xaxis_title="Trading days from today", yaxis_title="Portfolio change ($)")
        show_chart(fan, 400, legend=dict(orientation="h", y=-0.22, x=0))
    with fr:
        # Shared bins; the unhedged result is filled and a hedge is an outline, so overlaps don't mix colors.
        edges = np.histogram_bin_edges(np.concatenate(list(results.values())), bins=60)
        hist = go.Figure()
        for name, pnl in results.items():
            counts, _ = np.histogram(pnl, bins=edges)
            if name == "No hedge":
                hist.add_bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges), name=name,
                             marker_color=MUTED, marker_line_width=0, opacity=0.7)
            else:
                hist.add_scatter(x=edges, y=np.append(counts, counts[-1]), name=name, mode="lines",
                                 line=dict(color=HEDGE_COLORS[name], width=2.5, shape="hv"))
        hist.add_vline(x=base_stats["Bad case, 1 in 20 ($)"], line_dash="dot", line_color=RED,
                       annotation_text="1 in 20", annotation_font_color=RED)
        hist.update_layout(bargap=0, xaxis_title=f"Change after {horizon} ($)", yaxis_title="Simulations")
        show_chart(hist, 380)
    show_table(
        mc_stats,
        {c: money for c in mc_stats.columns if c.endswith("($)")} | {c: "{:.1f}%" for c in mc_stats.columns
                                                                    if c.endswith("(%)")},
        [c for c in mc_stats.columns if c.endswith("($)")],
        hide_index=True,
    )
    st.caption(
        f"{len(sim.pnl):,} simulations, each built from random week-long stretches of real {freq} factor moves, so "
        "crashes, calm spells and how oil, rates, the dollar and gold moved together all carry over. Each stock adds "
        "its own random noise. Average returns are removed, so this shows risk, not expected gain. Hedges are "
        "valued at the horizon with the same option model as step 4."
    )

# ---- Step 6: efficient frontier ----
step(6, "Optimize the mix",
     "A light mean-variance view: where your portfolio sits and what the same stocks could do in other weights.")
frontier_out = None
if len(risk_tickers) < 2:
    st.info("Add at least two holdings to see an efficient frontier.")
else:
    o1, o2 = st.columns([2, 1])
    method = o1.radio("Expected returns", RETURN_METHODS, horizontal=True,
                      help="Market-implied: T-bill rate + market beta × 5% equity premium (steady, CAPM). "
                      "Historical: each stock's average return over the fit window (swings a lot).")
    min_cap = int(5 * np.ceil(100 / len(risk_tickers) / 5))  # the cap must allow weights to add up to 100%
    cap = o2.slider("Max weight in one stock", min(max(min_cap, 10), 100), 100, max(40, min_cap), 5,
                    format="%d%%") / 100
    mu = expected_returns(models, risk_tickers, rate_now, method, returns, ppy)
    ports = model_portfolios(position_values.reindex(risk_tickers), mu, cov, rate_now, cap)
    imp = impacts.set_index("Ticker")
    port_summary, port_weights = compare_portfolios(
        ports, mu, cov, rate_now, normal["value"], imp["Impact (%)"], imp["Stress impact (%)"]
    )
    curve = frontier_curve(mu.values, cov.values, cap)
    frontier_out = (port_summary, port_weights)

    PORT_STYLE = {
        "Your portfolio": (TEXT, "circle", 16),
        "Same expected return, less risk": (GREEN, "diamond", 14),
        "Lowest risk": (LAVENDER, "square", 13),
        "Best return per unit of risk": (AMBER, "star", 18),
    }
    ef = go.Figure()
    ef.add_scatter(x=curve["Volatility (%/yr)"], y=curve["Return (%/yr)"], mode="lines", name="Efficient frontier",
                   line=dict(color=AMBER, width=3))
    stock_vol = np.sqrt(np.diag(cov.values)) * 100
    ef.add_scatter(x=stock_vol, y=mu.values * 100, mode="markers+text", text=risk_tickers, textposition="top center",
                   name="Single stocks", marker=dict(color=MUTED, size=8), textfont=dict(color=MUTED, size=10))
    for _, r in port_summary.iterrows():
        color, symbol, size = PORT_STYLE[r["Portfolio"]]
        ef.add_scatter(x=[r["Volatility (%/yr)"]], y=[r["Expected return (%/yr)"]], mode="markers", name=r["Portfolio"],
                       marker=dict(color=color, symbol=symbol, size=size, line=dict(color="#0B1F3A", width=1)))
    ef.update_layout(xaxis_title="Risk: volatility (%/yr)", yaxis_title="Expected return (%/yr)")
    show_chart(ef, 480, legend=dict(orientation="h", y=-0.18, x=0), margin=dict(l=10, r=10, t=40, b=10))

    show_table(
        port_summary,
        {"Expected return (%/yr)": "{:.1f}%", "Volatility (%/yr)": "{:.1f}%", "Sharpe ratio": "{:.2f}",
         "Scenario, normal ($)": money, "Scenario, stress ($)": money},
        ["Scenario, normal ($)", "Scenario, stress ($)"],
        hide_index=True,
    )
    st.markdown("**Weights (% of portfolio)**")
    show_table(port_weights, "{:.1f}%")
    st.caption(
        "Built only from stocks you already own, long-only, with no stock above the max weight. Risk comes from the "
        "factor model. The scenario columns show how each mix would fare in the scenario from step 2. These are "
        "textbook model portfolios for learning, not recommendations: mean-variance is very sensitive to expected "
        "returns, which no one knows. Try switching to historical averages to see how much the answer moves."
    )

# The summary card at the top of step 3, now that hedges, simulations and concentration are done.
hardest = None
losers = impacts[impacts["Impact ($)"] < 0]
if len(losers):
    w = losers.loc[losers["Impact ($)"].idxmin()]
    hardest = (w["Ticker"], float(w["Impact ($)"]), float(w["Impact (%)"]))
card_lines = []
ht = hedge_table.set_index("Hedge")
if mkt_move <= -0.005 and scenario_impact < 0:
    puts = ht.loc["Index puts"]
    inv = ht.loc["Inverse ETF"]
    card_lines.append(
        f"Hedging <span class='mono accent'>{hedge_ratio:.0%}</span> of your market exposure with "
        f"<span class='mono'>{n_contracts}</span> SPY put{'s' if n_contracts != 1 else ''} would cost about "
        f"<span class='mono'>${puts['Upfront cost ($)']:,.0f}</span> today and offset about "
        f"<span class='mono accent'>{puts['Loss offset (%)']:.0f}%</span> of this loss"
        + (" (whole contracts over-hedge a portfolio this size)." if over_hedged else ".")
    )
    card_lines.append(
        f"An inverse S&P ETF sized to the same exposure would offset about "
        f"<span class='mono accent'>{inv['Loss offset (%)']:.0f}%</span> with no premium, but gives up gains if "
        "the market rises. Details in step 4."
    )
elif mkt_move >= 0.005:
    card_lines.append("This scenario lifts the market, so market hedges would cost you money here.")
if sim is not None:
    bad = mc_stats.iloc[0]["Bad case, 1 in 20 ($)"]
    card_lines.append(
        f"Over {horizon}, about 1 in 20 simulated futures lost more than "
        f"<span class='mono neg'>${abs(bad):,.0f}</span> ({abs(bad) / sim.value * 100:.1f}%). Details in step 5."
    )
if conc_warnings:
    card_lines.append(f"<span class='accent'>Concentration:</span> {escape(conc_warnings[0])}")
summary_slot.html(
    brand.summary_card_html(
        scenario_name, normal["value"], normal["impact"], normal["impact_pct"], stress["impact"],
        stress["impact_pct"], hardest, card_lines,
    )
)


# ---- Step 7: compare scenarios ----
step(7, "Compare scenarios", "The presets, your saved scenarios and the current one, side by side.")
current_label = "Current scenario"
options = list(PRESETS) + list(saved) + [current_label]
picked = st.multiselect("Scenarios to compare", options, default=list(PRESETS) + list(saved))
scenarios = {n: (shock if n == current_label else saved.get(n, PRESETS.get(n))) for n in picked}
if not scenarios:
    st.info("Pick at least one scenario.")
    cmp_summary, cmp_holdings = pd.DataFrame(), pd.DataFrame()
else:
    cmp_summary, cmp_holdings = compare_scenarios(scenarios, models, stress_models, holdings, last_prices, factors, co_move)
    bars = go.Figure()
    bars.add_bar(x=cmp_summary["Scenario"], y=cmp_summary["Normal ($)"], name="Normal times",
                 marker_color=[GREEN if v > 0 else RED for v in cmp_summary["Normal ($)"]])
    bars.add_bar(x=cmp_summary["Scenario"], y=cmp_summary["Stress ($)"], name="Stress times", marker_color=CYAN,
                 opacity=0.8)
    bars.update_layout(barmode="group", yaxis_title="Estimated portfolio impact ($)")
    show_chart(bars, 380)
    show_table(
        cmp_summary, {"Normal ($)": money, "Normal (%)": pct, "Stress ($)": money, "Stress (%)": pct},
        ["Normal ($)", "Normal (%)", "Stress ($)", "Stress (%)"], hide_index=True,
    )
    st.markdown("**Estimated impact per holding ($, normal times)**")
    show_table(cmp_holdings, money, list(cmp_holdings.columns))

# ---- Step 8: check the model ----
step(8, "Check the model", "Betas behind every estimate, how they changed, and how the model did in real crises.")
b1, b2 = st.columns([1, 1])
with b1:
    beta_tbl = pd.DataFrame({t: m.betas for t, m in models.items()}).T
    beta_tbl.columns = [FACTOR_LABELS[f] for f in FACTORS]
    beta_tbl["R²"] = [m.r_squared for m in models.values()]
    beta_tbl["Periods"] = [m.n_obs for m in models.values()]
    st.markdown(f"**Betas ({freq}, last {years} year(s))**")
    show_table(beta_tbl, {c: "{:.3f}" for c in beta_tbl.columns[:-1]})
    st.caption(
        "A beta of 0.30 on oil means the stock has tended to move about 0.3% for each 1% move in oil, "
        "holding the other factors fixed. The rates beta is per 1 percentage point change in the 10-year yield."
    )
with b2:
    st.markdown("**How betas changed over time** (1-year rolling window)")
    rc1, rc2 = st.columns(2)
    tick = rc1.selectbox("Holding", list(models))
    factor = rc2.selectbox("Factor", FACTORS, format_func=lambda f: FACTOR_LABELS[f])
    rb = rolling_betas(all_returns[tick], all_factors, PERIODS_PER_YEAR[freq])
    if rb.empty:
        st.info("Not enough history for a rolling chart.")
    else:
        line = px.line(rb, y=factor, labels={factor: "Beta", "index": ""}, color_discrete_sequence=[AMBER])
        line.add_hline(y=models[tick].betas[factor], line_dash="dot", line_color=MUTED,
                       annotation_text=f"Used now ({models[tick].betas[factor]:.2f})", annotation_font_color=MUTED,
                       annotation_position="bottom left")
        show_chart(line, 300)

st.markdown("**Historical replays: would the model have called it?**")
st.caption(
    f"Betas are fit only on the {years} year(s) before each event, then fed the factor moves that actually happened. "
    "Dollar amounts apply those % moves to your positions' value today."
)
replay_table, replay_event = None, None
event = st.selectbox("Event", list(REPLAYS))
start, end = REPLAYS[event]
try:
    table, ev_moves = run_replay(holdings, prices, market, macro, start, end, years, freq)
except ValueError as e:
    st.info(f"Not enough data for this event: {e}")
else:
    replay_table, replay_event = table, event
    st.caption("What the factors did: " + describe_moves(ev_moves))
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
        sc = px.scatter(plot, x="Predicted (%)", y="Actual (%)", text="Ticker", color_discrete_sequence=[AMBER])
        sc.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim, line=dict(dash="dot", color=MUTED))
        sc.update_traces(textposition="top center", marker=dict(size=11))
        sc.update_layout(xaxis_range=[-lim, lim], yaxis_range=[-lim, lim])
        show_chart(sc, 420)
        st.caption("Dots on the dotted line were predicted perfectly. Above the line = did better than predicted.")
    skipped = table[table["Predicted (%)"].isna() | table["Actual (%)"].isna()]["Ticker"].tolist()
    if skipped:
        st.info(f"Not enough history before this event for: {', '.join(skipped)}")
    show_table(
        table,
        {"Value today ($)": "${:,.0f}", "Predicted (%)": pct, "Actual (%)": pct, "Predicted ($)": money,
         "Actual ($)": money},
        ["Predicted (%)", "Actual (%)", "Predicted ($)", "Actual ($)"],
        hide_index=True,
    )

# ---- Step 9: export ----
step(9, "Export", "Download your results to keep, share or open in Excel.")
settings = [
    ("Holdings", f"{len(impacts)} positions"),
    ("Betas fit on", f"last {years} year(s), {freq} returns"),
    ("Stress betas", f"worst {STRESS_QUANTILE:.0%} of market periods since 2006"),
    ("Co-moves of unset factors", "included" if co_move else "held at 0"),
    ("Monte Carlo", f"{horizon}, history from {'2006 on' if source == full_label else f'the last {years} year(s)'}"),
    ("Data", "made-up sample data" if use_sample else "Yahoo Finance and FRED"),
]
summary_rows = [
    ("Scenario", scenario_name),
    ("Factor moves used", describe_moves(moves)),
    ("Portfolio value", f"${normal['value']:,.0f}"),
    ("Estimated change, normal times", f"{money(normal['impact'])} ({normal['impact_pct']:+.2f}%)"),
    ("Estimated change, stress times", f"{money(stress['impact'])} ({stress['impact_pct']:+.2f}%)"),
    ("Yearly volatility", f"{conc['vol']:.1f}% (acts like {conc['effective_n']:.1f} equal positions)"),
]
if not mc_stats.empty:
    summary_rows.append((f"Bad case over {horizon} (1 in 20)", money(mc_stats.iloc[0]["Bad case, 1 in 20 ($)"])))
summary_rows += [("Concentration warning", w) for w in conc_warnings]
beta_export = pd.DataFrame({t: m.betas for t, m in models.items()}).T.rename(columns=FACTOR_LABELS)
beta_export.insert(0, "Ticker", beta_export.index)
sheets = {"Scenario": impacts, "Concentration": conc_table, "Betas": beta_export.reset_index(drop=True),
          "Hedging": hedge_table}
if not mc_stats.empty:
    sheets["Monte Carlo"] = mc_stats
    sheets["Monte Carlo bands"] = sim.bands.rename(columns=lambda c: f"{c[1:]}th percentile ($)" if c[0] == "p" else c)
if frontier_out is not None:
    sheets["Frontier portfolios"] = frontier_out[0]
    sheets["Frontier weights (%)"] = frontier_out[1].reset_index()
if not cmp_summary.empty:
    sheets["Compare scenarios"] = cmp_summary
    sheets["Compare by holding"] = cmp_holdings.reset_index(names="Ticker")
if replay_table is not None:
    sheets["Replay"] = replay_table

stamp = pd.Timestamp.today().strftime("%Y-%m-%d")
e1, e2, e3 = st.columns(3)
e1.download_button("Scenario results (CSV)", impacts.to_csv(index=False), f"shock_lab_scenario_{stamp}.csv",
                   "text/csv", width="stretch")
e2.download_button("All tables (Excel)", to_excel(sheets), f"shock_lab_{stamp}.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
pdf_tables = [("Current scenario", impacts.drop(columns=["Shares", "Low (%)", "High (%)"]))]
if not cmp_summary.empty:
    pdf_tables.append(("Scenario comparison", cmp_summary))
pdf_tables.append(("Concentration", conc_table))
pdf_tables.append(("Hedges in this scenario", hedge_table.drop(columns=["What it means"])))
if not mc_stats.empty:
    pdf_tables.append((f"Monte Carlo, {horizon}", mc_stats))
if frontier_out is not None:
    pdf_tables.append(("Efficient frontier: model portfolios", frontier_out[0]))
if replay_table is not None:
    pdf_tables.append((f"Historical replay: {replay_event}", replay_table))
e3.download_button("Report (PDF)", to_pdf("Shock Lab report", settings, summary_rows, pdf_tables, DISCLAIMER),
                   f"shock_lab_report_{stamp}.pdf", "application/pdf", type="primary", width="stretch")
st.caption("The Excel file includes every table above: scenario, concentration, betas, hedges, Monte Carlo, "
           "frontier portfolios, scenario comparison and the replay you last viewed.")

st.html(f'<div class="sl-disclaimer" style="margin-top:28px">{brand.WATERMARK_TEXT}</div>')
