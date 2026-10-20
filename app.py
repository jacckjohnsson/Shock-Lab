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
from shock_lab.compare import compare_scenarios
from shock_lab.importers import TEMPLATE_CSV, parse_holdings_csv
from shock_lab.portfolio import MAX_HOLDINGS, clean_holdings
from shock_lab.report import to_excel, to_pdf
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

    saved = st.session_state.setdefault("saved_scenarios", {})
    with st.expander("Save this scenario to compare later"):
        name = st.text_input("Name", placeholder="e.g. Oil spike + rate hike")
        if st.button("Save scenario", disabled=not name.strip()):
            saved[name.strip()] = dict(shock)
            st.success(f"Saved \"{name.strip()}\". See the Compare scenarios tab.")
        if saved and st.button("Clear saved scenarios"):
            saved.clear()

# ---- Holdings input ----
st.subheader(f"1. Your holdings (up to {MAX_HOLDINGS})")
mode = st.radio(
    "How do you want to add holdings?",
    ["Type them in", "Upload a CSV"],
    horizontal=True,
    label_visibility="collapsed",
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

last_prices = prices.ffill().iloc[-1]
replay_table, replay_event = None, None

tab_scn, tab_cmp, tab_replay, tab_model, tab_export = st.tabs(
    ["Scenario", "Compare scenarios", "Historical replays", "Model details", "Export"]
)

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
        replay_table, replay_event = table, event
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

# ---- Compare scenarios tab ----
with tab_cmp:
    current_label = "Current sidebar scenario"
    options = list(PRESETS) + list(saved) + [current_label]
    picked = st.multiselect("Scenarios to compare", options, default=list(PRESETS) + list(saved))
    scenarios = {n: (shock if n == current_label else saved.get(n, PRESETS.get(n))) for n in picked}
    if not scenarios:
        st.info("Pick at least one scenario.")
        cmp_summary, cmp_holdings = pd.DataFrame(), pd.DataFrame()
    else:
        cmp_summary, cmp_holdings = compare_scenarios(
            scenarios, models, stress_models, holdings, last_prices, factors, co_move
        )
        bars = go.Figure()
        bars.add_bar(x=cmp_summary["Scenario"], y=cmp_summary["Normal ($)"], name="Normal times",
                     marker_color=[GREEN if v > 0 else RED for v in cmp_summary["Normal ($)"]])
        bars.add_bar(x=cmp_summary["Scenario"], y=cmp_summary["Stress ($)"], name="Stress times", marker_color=GREY)
        bars.update_layout(barmode="group", height=380, yaxis_title="Estimated portfolio impact ($)",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(bars, width="stretch")
        st.dataframe(
            cmp_summary.style.format({"Normal ($)": money, "Normal (%)": pct, "Stress ($)": money, "Stress (%)": pct}),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Estimated impact per holding ($, normal times)**")
        st.dataframe(cmp_holdings.style.format(money), width="stretch")
        st.caption("Save your own scenarios from the sidebar to add them here.")

# ---- Export tab ----
with tab_export:
    st.markdown("Download your results to keep, share or open in Excel.")
    settings = [
        ("Holdings", f"{len(impacts)} positions"),
        ("Betas fit on", f"last {years} year(s), {freq} returns"),
        ("Stress betas", f"worst {STRESS_QUANTILE:.0%} of market periods since 2006"),
        ("Co-moves of unset factors", "included" if co_move else "held at 0"),
        ("Data", "made-up sample data" if use_sample else "Yahoo Finance and FRED"),
    ]
    scenario_desc = ", ".join(
        f"{FACTOR_LABELS[f].split(' (')[0]} {moves[f]:+.2f} pts" if f == "rates"
        else f"{FACTOR_LABELS[f].split(' (')[0]} {moves[f] * 100:+.1f}%"
        for f in FACTORS
    )
    summary_rows = [
        ("Scenario (factor moves used)", scenario_desc),
        ("Portfolio value", f"${normal['value']:,.0f}"),
        ("Estimated change, normal times", f"{money(normal['impact'])} ({normal['impact_pct']:+.2f}%)"),
        ("Estimated change, stress times", f"{money(stress['impact'])} ({stress['impact_pct']:+.2f}%)"),
    ]
    beta_export = pd.DataFrame({t: m.betas for t, m in models.items()}).T.rename(columns=FACTOR_LABELS)
    beta_export.insert(0, "Ticker", beta_export.index)
    sheets = {"Scenario": impacts, "Betas": beta_export.reset_index(drop=True)}
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
    if replay_table is not None:
        pdf_tables.append((f"Historical replay: {replay_event}", replay_table))
    e3.download_button("Report (PDF)", to_pdf("Shock Lab report", settings, summary_rows, pdf_tables, DISCLAIMER),
                       f"shock_lab_report_{stamp}.pdf", "application/pdf", width="stretch")
    st.caption("The Excel file includes the scenario, betas, scenario comparison and the replay you last viewed.")

st.caption(DISCLAIMER)
