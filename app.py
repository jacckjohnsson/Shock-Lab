"""Shock Lab: what happens to my portfolio if oil, rates, the dollar or the market move?

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from shock_lab import brand, data
from shock_lab.brand import AMBER, BORDER, CYAN, GREEN, LAVENDER, MUTED, RED, TEXT
from shock_lab.compare import compare_scenarios
from shock_lab.hedging import (
    CONTRACT_SIZE,
    HEDGES,
    HedgeSetup,
    compare_hedges,
    payoff_table,
    put_greeks_table,
    vix_market_beta,
)
from shock_lab.importers import TEMPLATE_CSV, parse_holdings_csv
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
from shock_lab.report import to_excel, to_pdf
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


def show_chart(fig: go.Figure, height: int | None = None) -> None:
    st.plotly_chart(brand.style_chart(fig, height), theme=None, width="stretch")


def step(n: int, title: str, subtitle: str = "") -> None:
    st.html(brand.step_html(n, title, subtitle))


def describe_moves(mv: pd.Series) -> str:
    return ", ".join(
        f"{FACTOR_LABELS[f].split(' (')[0]} {mv[f]:+.2f} pts" if f == "rates"
        else f"{FACTOR_LABELS[f].split(' (')[0]} {mv[f] * 100:+.1f}%"
        for f in FACTORS
    )


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


# ---- Sidebar: model settings ----
with st.sidebar:
    st.html(f'<img src="{brand.logo_data_uri()}" width="56" height="28" alt=""/> '
            f'<span style="font-family:{brand.MONO};letter-spacing:.14em;font-weight:600">SHOCK LAB</span>')
    st.header("Model settings")
    use_sample = st.toggle("Use made-up sample data (works offline)", value=False)
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
steps_slot.html(brand.steps_html(7 if ran else 2))

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
step(2, "Choose a scenario", "Pick a preset or set your own moves. Factors left at 0 are filled in from history.")
preset_names = list(PRESETS) + ["Custom"]
preset = st.pills("Preset", preset_names, default="S&P 500 -15%", label_visibility="collapsed") or "Custom"
base = PRESETS.get(preset, {})
s1, s2, s3, s4 = st.columns(4)
shock = {
    "market": s1.slider(FACTOR_LABELS["market"], -40.0, 40.0, base.get("market", 0.0) * 100, 1.0,
                        key=f"sl_market_{preset}") / 100,
    "oil": s2.slider(FACTOR_LABELS["oil"], -60.0, 60.0, base.get("oil", 0.0) * 100, 1.0, key=f"sl_oil_{preset}") / 100,
    "rates": s3.slider(FACTOR_LABELS["rates"], -3.0, 3.0, base.get("rates", 0.0), 0.25, key=f"sl_rates_{preset}"),
    "dollar": s4.slider(FACTOR_LABELS["dollar"], -20.0, 20.0, base.get("dollar", 0.0) * 100, 0.5,
                        key=f"sl_dollar_{preset}") / 100,
}
is_preset = preset in PRESETS and shock == {f: base.get(f, 0.0) for f in FACTORS}
scenario_name = preset if is_preset else "your custom scenario"

saved = st.session_state.setdefault("saved_scenarios", {})
n1, n2, n3 = st.columns([3, 1, 1], vertical_alignment="bottom")
name = n1.text_input("Save this scenario to compare later", placeholder="e.g. Oil spike + rate hike")
if n2.button("Save scenario", disabled=not name.strip(), width="stretch"):
    saved[name.strip()] = dict(shock)
    st.toast(f"Saved \"{name.strip()}\". It appears in step 5.")
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
            "Factor": [FACTOR_LABELS[f] for f in FACTORS],
            "You set": [shock[f] * (1 if f == "rates" else 100) for f in FACTORS],
            "Used": [moves[f] * (1 if f == "rates" else 100) for f in FACTORS],
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
    "R² shows how much of a stock's movement the four factors explain (0 to 1)."
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
    mv = implied_factor_moves(m, factors) if co_move else pd.Series(m).reindex(FACTORS).fillna(0.0)
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
vix_after = float(np.clip(vix_now + vbeta * mkt_move, 9, 80))

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

# The summary card at the top of step 3, now that the hedges are priced.
hardest = None
losers = impacts[impacts["Impact ($)"] < 0]
if len(losers):
    w = losers.loc[losers["Impact ($)"].idxmin()]
    hardest = (w["Ticker"], float(w["Impact ($)"]), float(w["Impact (%)"]))
hedge_lines = []
ht = hedge_table.set_index("Hedge")
if mkt_move <= -0.005 and scenario_impact < 0:
    puts = ht.loc["Index puts"]
    inv = ht.loc["Inverse ETF"]
    hedge_lines.append(
        f"Hedging <span class='mono accent'>{hedge_ratio:.0%}</span> of your market exposure with "
        f"<span class='mono'>{n_contracts}</span> SPY put{'s' if n_contracts != 1 else ''} would cost about "
        f"<span class='mono'>${puts['Upfront cost ($)']:,.0f}</span> today and offset about "
        f"<span class='mono accent'>{puts['Loss offset (%)']:.0f}%</span> of this loss"
        + (" (whole contracts over-hedge a portfolio this size)." if over_hedged else ".")
    )
    hedge_lines.append(
        f"An inverse S&P ETF sized to the same exposure would offset about "
        f"<span class='mono accent'>{inv['Loss offset (%)']:.0f}%</span> with no premium, but gives up gains if "
        "the market rises. Details in step 4."
    )
elif mkt_move >= 0.005:
    hedge_lines.append("This scenario lifts the market, so market hedges would cost you money here.")
summary_slot.html(
    brand.summary_card_html(
        scenario_name, normal["value"], normal["impact"], normal["impact_pct"], stress["impact"],
        stress["impact_pct"], hardest, hedge_lines,
    )
)

# ---- Step 5: compare scenarios ----
step(5, "Compare scenarios", "The presets, your saved scenarios and the current one, side by side.")
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

# ---- Step 6: check the model ----
step(6, "Check the model", "Betas behind every estimate, how they changed, and how the model did in real crises.")
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

# ---- Step 7: export ----
step(7, "Export", "Download your results to keep, share or open in Excel.")
settings = [
    ("Holdings", f"{len(impacts)} positions"),
    ("Betas fit on", f"last {years} year(s), {freq} returns"),
    ("Stress betas", f"worst {STRESS_QUANTILE:.0%} of market periods since 2006"),
    ("Co-moves of unset factors", "included" if co_move else "held at 0"),
    ("Data", "made-up sample data" if use_sample else "Yahoo Finance and FRED"),
]
summary_rows = [
    ("Scenario", scenario_name),
    ("Factor moves used", describe_moves(moves)),
    ("Portfolio value", f"${normal['value']:,.0f}"),
    ("Estimated change, normal times", f"{money(normal['impact'])} ({normal['impact_pct']:+.2f}%)"),
    ("Estimated change, stress times", f"{money(stress['impact'])} ({stress['impact_pct']:+.2f}%)"),
]
beta_export = pd.DataFrame({t: m.betas for t, m in models.items()}).T.rename(columns=FACTOR_LABELS)
beta_export.insert(0, "Ticker", beta_export.index)
sheets = {"Scenario": impacts, "Betas": beta_export.reset_index(drop=True), "Hedging": hedge_table}
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
pdf_tables.append(("Hedges in this scenario", hedge_table.drop(columns=["What it means"])))
if replay_table is not None:
    pdf_tables.append((f"Historical replay: {replay_event}", replay_table))
e3.download_button("Report (PDF)", to_pdf("Shock Lab report", settings, summary_rows, pdf_tables, DISCLAIMER),
                   f"shock_lab_report_{stamp}.pdf", "application/pdf", type="primary", width="stretch")
st.caption("The Excel file includes the scenario, betas, scenario comparison, hedges and the replay you last viewed.")

st.html(f'<div class="sl-disclaimer" style="margin-top:28px">{brand.WATERMARK_TEXT}</div>')
