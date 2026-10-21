"""The welcome screen shown before the app: the same content as site/index.html, built for st.html.

Screenshots are served by Streamlit's static file serving from the repo's static/ folder.
"""

from __future__ import annotations

from .brand import AMBER, BORDER, CYAN, GREEN, MONO, MUTED, NAVY, PANEL, RED, SANS, TEXT, logo_data_uri

STATIC = "app/static/"

CSS = f"""
<style>
.lp {{ font-family: {SANS}; color: {TEXT}; }}
.lp-bar {{ display: flex; align-items: center; gap: 10px; font-family: {MONO}; font-weight: 600; letter-spacing: .14em;
  margin: 0 0 8px; }}
.lp-eyebrow, .lp-label {{ font-family: {MONO}; color: {AMBER}; letter-spacing: .14em; text-transform: uppercase;
  font-size: .78rem; margin: 0 0 12px; }}
.lp h1 {{ font-size: clamp(2.1rem, 4.4vw, 3.5rem); line-height: 1.08; margin: 0 0 18px; padding: 0; font-weight: 600;
  letter-spacing: -.01em; color: {TEXT}; }}
.lp h1 em {{ font-style: normal; color: {AMBER}; }}
.lp-lede {{ font-size: 1.15rem; color: {MUTED}; line-height: 1.6; margin: 0 0 8px; }}
.lp-note {{ font-size: .82rem; color: {MUTED}; margin: 6px 0 0; }}
.lp-shot {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; background: {PANEL};
  box-shadow: 0 24px 60px rgba(0,0,0,.35); }}
.lp-shot img {{ display: block; width: 100%; height: auto; }}
.lp-strip {{ display: flex; flex-wrap: wrap; gap: 10px 28px; justify-content: center; padding: 14px 16px; margin: 36px 0 8px;
  border-top: 1px solid {BORDER}; border-bottom: 1px solid {BORDER}; background: {PANEL}; font-family: {MONO};
  font-size: .88rem; color: {MUTED}; border-radius: 8px; }}
.lp-strip b {{ color: {TEXT}; }}
.lp-up {{ color: {GREEN}; }} .lp-down {{ color: {RED}; }}
.lp-sec {{ margin: 64px 0 0; }}
.lp h2 {{ font-size: clamp(1.6rem, 3vw, 2.2rem); line-height: 1.15; margin: 0 0 10px; padding: 0; color: {TEXT}; }}
.lp-sub {{ color: {MUTED}; font-size: 1.05rem; max-width: 62ch; margin: 0 0 28px; line-height: 1.6; }}
.lp-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }}
.lp-card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 22px; }}
.lp-card h3 {{ margin: 0 0 8px; padding: 0; font-size: 1.08rem; color: {TEXT}; }}
.lp-card p {{ margin: 0; color: {MUTED}; font-size: .95rem; line-height: 1.55; }}
.lp-num {{ display: inline-block; font-family: {MONO}; font-weight: 600; color: {NAVY}; background: {AMBER};
  border-radius: 6px; padding: 1px 9px; margin-bottom: 12px; }}
.lp-k {{ font-family: {MONO}; color: {AMBER}; font-size: .76rem; letter-spacing: .1em; margin-bottom: 8px; }}
.lp-feature {{ display: grid; grid-template-columns: 1fr 1.25fr; gap: 44px; align-items: center; margin: 72px 0 0; }}
.lp-feature.flip .lp-copy {{ order: 2; }}
.lp-feature.wide {{ grid-template-columns: 1fr; gap: 22px; }}
.lp-feature h3 {{ font-size: 1.5rem; margin: 0 0 10px; padding: 0; line-height: 1.2; color: {TEXT}; }}
.lp-feature p, .lp-feature li {{ color: {MUTED}; line-height: 1.6; }}
.lp-feature ul {{ margin: 8px 0 0; padding-left: 18px; }}
.lp-feature li b {{ color: {TEXT}; }}
.lp-cta {{ text-align: center; margin: 72px 0 12px; padding: 40px 16px 8px; border-top: 1px solid {BORDER}; }}
.lp-disclaimer {{ border-left: 3px solid {AMBER}; padding: 6px 14px; color: {MUTED}; font-size: .88rem; margin: 28px 0 8px; }}
.lp-cyan {{ color: {CYAN}; }}
@media (max-width: 900px) {{
  .lp-grid, .lp-feature {{ grid-template-columns: 1fr; gap: 20px; }}
  .lp-feature.flip .lp-copy {{ order: 0; }}
  .lp-feature, .lp-sec {{ margin-top: 48px; }}
}}
</style>
"""


def _shot(name: str, alt: str) -> str:
    return f'<div class="lp-shot"><img src="{STATIC}{name}" alt="{alt}" loading="lazy"/></div>'


def brand_bar_html() -> str:
    return f'<div class="lp lp-bar"><img src="{logo_data_uri()}" width="52" height="26" alt=""/>SHOCK LAB</div>'


def hero_html() -> str:
    return (
        '<div class="lp">'
        '<p class="lp-eyebrow">Institutional stress tests for everyone</p>'
        "<h1>What happens to <em>your</em> portfolio if the market shakes?</h1>"
        '<p class="lp-lede">Enter your holdings, pick a shock, like oil +20%, rates +1 point or the S&amp;P 500 −15%, '
        "and see what it would do to every position and what it would cost to protect it. These are the same tools "
        "risk desks use, explained in plain English.</p></div>"
    )


def hero_note_html() -> str:
    return ('<p class="lp lp-note">No sign-up. Type tickers in or upload your broker\'s CSV. '
            "Or try a made-up demo portfolio first.</p>")


def hero_shot_html() -> str:
    return _shot("summary-card.png", "Shock Lab summary card for an S&amp;P 500 −15% scenario")


def body_html() -> str:
    strip = "".join(
        f'<span><b>{name}</b> <span class="lp-{cls}">{move}</span></span>'
        for name, move, cls in [
            ("S&amp;P 500", "−15%", "down"), ("Crude oil", "+20%", "up"), ("10-yr yield", "+1.00 pt", "up"),
            ("US dollar", "+5%", "up"), ("Gold", "+10%", "up"), ("VIX", "+15 pts", "up"), ("Credit spreads", "+1.00 pt", "up"),
        ]
    )
    steps = "".join(
        f'<div class="lp-card"><span class="lp-num">{i}</span><h3>{title}</h3><p>{text}</p></div>'
        for i, (title, text) in enumerate(
            [
                ("Add your holdings", "Type in up to 30 tickers and share counts, or upload a positions export from "
                 "Fidelity, Schwab, Vanguard, E*TRADE and others."),
                ("Pick a shock", "Use a preset or drag the sliders. Anything you leave at zero is filled in with how it "
                 "has typically moved alongside your shock."),
                ("See the damage, and your options", "Get a plain-English summary, then dig into every holding, hedge, "
                 "simulation and model detail. Nothing is hidden."),
            ],
            start=1,
        )
    )
    checks = "".join(
        f'<div class="lp-card"><div class="lp-k">{k}</div><h3>{title}</h3><p>{text}</p></div>'
        for k, title, text in [
            ("REPLAYS", "Would it have called 2008?", "Betas are fit only on data from before 2008, COVID and 2022, "
             "then compared with what actually happened."),
            ("UNCERTAINTY", "Ranges, not false precision", "Each estimate carries a 95% range from uncertainty in the "
             "betas, and rolling charts show how betas drift over time."),
            ("DATA", "Public, well-known sources", "Prices from Yahoo Finance; oil, yields, the dollar, credit spreads, "
             "VIX and T-bills from the St. Louis Fed (FRED)."),
            ("EXPORT", "Take it with you", "Download a branded PDF report, an Excel workbook with every table, or a CSV."),
            ("PRIVATE", "No account needed", "Holdings are used for your session only. Nothing is stored or shared."),
            ("OPEN METHOD", "The formulas are documented", "Regression, co-moves, stress betas, Black-Scholes, "
             "bootstrap and mean-variance, written out step by step."),
        ]
    )
    return f"""
<div class="lp">
  <div class="lp-strip">{strip}</div>

  <div class="lp-sec">
    <p class="lp-label">How it works</p>
    <h2>Three steps, about a minute</h2>
    <p class="lp-sub">Shock Lab learns how each of your stocks has reacted to the market, oil, interest rates, the dollar
      and gold, then applies that to the scenario you choose.</p>
    <div class="lp-grid">{steps}</div>
  </div>

  <div class="lp-feature">
    <div class="lp-copy">
      <p class="lp-label">Stress test</p>
      <h3>Every holding, in normal markets and in a selloff</h3>
      <p>A factor regression for each stock estimates its sensitivity to the market, oil, rates, the dollar and gold.
        Each result also comes with a second estimate fit only on the market's worst days, when stocks tend to fall together.</p>
      <ul>
        <li><b>7 presets</b> plus custom scenarios you can save and compare side by side</li>
        <li><b>VIX and credit-spread shocks</b> translated into the market moves that usually come with them</li>
        <li><b>Concentration check:</b> share of value vs. share of risk for every position</li>
      </ul>
    </div>
    {_shot("impact.png", "Estimated dollar impact per holding, normal vs stress times")}
  </div>

  <div class="lp-feature wide">
    <div class="lp-copy">
      <p class="lp-label">Hedge it</p>
      <h3>What protection would cost, before you need it</h3>
      <p>Compare index puts, put spreads, an inverse ETF or simply raising cash. Option prices come from Black-Scholes
        with the live VIX and a volatility skew, with the full greeks shown, and the app warns when one SPY contract
        over-hedges a small portfolio.</p>
    </div>
    {_shot("hedge-payoff.png", "Payoff of the portfolio plus each hedge across S&amp;P 500 moves")}
  </div>

  <div class="lp-feature">
    <div class="lp-copy">
      <p class="lp-label">Simulate</p>
      <h3>Thousands of possible futures, drawn from real history</h3>
      <p>Monte Carlo replays random week-long stretches of real markets since 2006, including 2008 and 2020, through your
        portfolio. You get the typical outcome, a 1-in-20 bad case, the average of the worst 5%, and the odds of losing
        10% or more, with or without a hedge.</p>
    </div>
    {_shot("monte-carlo.png", "Fan chart of simulated portfolio outcomes")}
  </div>

  <div class="lp-feature wide">
    <div class="lp-copy">
      <p class="lp-label">Optimize</p>
      <h3>Where your portfolio sits on the efficient frontier</h3>
      <p>A light mean-variance view built only from the stocks you already own: the lowest-risk mix, the same expected
        return with less risk, and the best return per unit of risk, plus how each would fare in your scenario.</p>
    </div>
    {_shot("frontier.png", "Efficient frontier with your portfolio and model portfolios")}
  </div>

  <div class="lp-sec">
    <p class="lp-label">Built to be checked</p>
    <h2>Transparent math, tested against real crises</h2>
    <p class="lp-sub">Every beta, standard error, greek and assumption is on screen. And the model is graded on its own
      track record.</p>
    <div class="lp-grid">{checks}</div>
  </div>

  <div class="lp-cta">
    <h2>Find out before the market does</h2>
    <p class="lp-sub" style="margin:0 auto 8px">Run your first stress test in about a minute, with no sign-up.</p>
  </div>
</div>
"""


def footer_html() -> str:
    return (
        '<div class="lp lp-disclaimer">For education only, not investment advice. Estimates come from models fit to '
        "past data, and real markets can react very differently, especially in a crisis. Do not make investment "
        "decisions based on this tool alone.</div>"
    )
