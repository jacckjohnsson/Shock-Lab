"""Shock Lab look and feel: colors, logo, chart styling and page fragments."""

from __future__ import annotations

import base64
from html import escape

import plotly.graph_objects as go

# Navy terminal palette
NAVY = "#0B1F3A"
PANEL = "#112A4D"
BORDER = "#1E3A66"
TEXT = "#E6EDF5"
MUTED = "#8FA3BF"
AMBER = "#F5A623"  # brand accent
CYAN = "#4FC3F7"  # "stress times"
GREEN = "#2ECC71"  # gains
RED = "#FF5C5C"  # losses
LAVENDER = "#B39DDB"
MONO = "'IBM Plex Mono', 'SFMono-Regular', Menlo, Consolas, monospace"
SANS = "'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, sans-serif"

NAME = "Shock Lab"
TAGLINE = "Institutional stress tests for everyone."
WATERMARK_TEXT = "Shock Lab · Model estimates for education, not investment advice."

# Seismograph trace: flat, one sharp shock, aftershocks, flat again. Coordinates in a 64x32 box.
SPIKE_POINTS = [(2, 16), (20, 16), (24, 16), (28, 3), (33, 29), (37, 9), (41, 21), (44, 14), (47, 16), (62, 16)]


def logo_svg(color: str = AMBER, stroke: float = 3.0) -> str:
    pts = " ".join(f"{x},{y}" for x, y in SPIKE_POINTS)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 32" width="64" height="32">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def logo_data_uri(color: str = AMBER) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(logo_svg(color).encode()).decode()


# ---- Charts ----

def style_chart(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Navy terminal styling plus a small logo watermark above the top-right corner of the plot."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO, color=TEXT, size=12),
        colorway=[AMBER, CYAN, GREEN, RED, LAVENDER, MUTED],
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(family=MONO, color=TEXT)),
    )
    if height:
        fig.update_layout(height=height)
    axis = dict(gridcolor=BORDER, zerolinecolor=MUTED, linecolor=BORDER, tickfont=dict(color=MUTED), automargin=True,
                title_font=dict(color=MUTED, size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.add_layout_image(
        source=logo_data_uri(), xref="paper", yref="paper", x=1.0, y=1.01, sizex=0.06, sizey=0.06,
        xanchor="right", yanchor="bottom", opacity=0.55, layer="above",
    )
    fig.add_annotation(
        text="SHOCK LAB", xref="paper", yref="paper", x=0.935, y=1.01, xanchor="right", yanchor="bottom",
        showarrow=False, font=dict(family=MONO, size=10, color=MUTED), opacity=0.6,
    )
    return fig


# ---- Page fragments (rendered with st.html, so no markdown or LaTeX processing) ----

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');
html, body, [data-testid="stAppViewContainer"] {{ font-family: {SANS}; }}
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{ font-family: {MONO}; }}
[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 10px 14px; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; text-transform: uppercase; letter-spacing: .06em; font-size: .75rem; }}
.sl-header {{ display: flex; align-items: center; gap: 14px; margin: 4px 0 2px; }}
.sl-word {{ font-family: {MONO}; font-weight: 600; font-size: 1.9rem; letter-spacing: .14em; color: {TEXT}; }}
.sl-tag {{ color: {MUTED}; font-size: .95rem; margin-top: 2px; }}
.sl-disclaimer {{ color: {MUTED}; font-size: .78rem; border-left: 3px solid {AMBER}; padding: 4px 10px; margin: 10px 0 6px; }}
.sl-steps {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 4px; }}
.sl-steps a {{ text-decoration: none; font-family: {MONO}; font-size: .78rem; padding: 4px 10px;
  border: 1px solid {BORDER}; border-radius: 999px; color: {MUTED}; }}
.sl-steps a.on {{ color: {NAVY}; background: {AMBER}; border-color: {AMBER}; }}
.sl-step {{ display: flex; align-items: baseline; gap: 12px; margin: 28px 0 4px; padding-top: 10px;
  border-top: 1px solid {BORDER}; }}
.sl-num {{ font-family: {MONO}; font-weight: 600; color: {NAVY}; background: {AMBER}; border-radius: 6px;
  padding: 1px 8px; font-size: .95rem; }}
.sl-title {{ font-weight: 600; font-size: 1.25rem; color: {TEXT}; }}
.sl-sub {{ color: {MUTED}; font-size: .9rem; margin: 0 0 6px; }}
.sl-card {{ background: {PANEL}; border: 1px solid {BORDER}; border-left: 4px solid {AMBER}; border-radius: 10px;
  padding: 18px 22px; margin: 8px 0 14px; }}
.sl-card .k {{ color: {MUTED}; font-family: {MONO}; font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; }}
.sl-card .big {{ font-family: {MONO}; font-size: 2.1rem; font-weight: 600; margin: 4px 0 8px; }}
.sl-card p {{ margin: 6px 0; color: {TEXT}; line-height: 1.5; }}
.sl-card .mono {{ font-family: {MONO}; }}
.neg {{ color: {RED}; }} .pos {{ color: {GREEN}; }} .stress {{ color: {CYAN}; }} .accent {{ color: {AMBER}; }}
</style>
"""

STEPS = ["Holdings", "Scenario", "Impact", "Hedge", "Compare", "Check the model", "Export"]


def header_html() -> str:
    return (
        f'<div class="sl-header"><img src="{logo_data_uri()}" width="72" height="36" alt="Shock Lab logo"/>'
        f'<div><div class="sl-word">SHOCK LAB</div><div class="sl-tag">{escape(TAGLINE)}</div></div></div>'
        '<div class="sl-disclaimer">For education only, not investment advice. Estimates come from a linear model '
        "fit to past data, and real markets can react very differently, especially in a crisis.</div>"
    )


def steps_html(reached: int) -> str:
    links = "".join(
        f'<a class="{"on" if i <= reached else ""}" href="#step-{i}">{i}&nbsp;{escape(s)}</a>'
        for i, s in enumerate(STEPS, start=1)
    )
    return f'<div class="sl-steps">{links}</div>'


def step_html(n: int, title: str, subtitle: str = "") -> str:
    sub = f'<div class="sl-sub">{escape(subtitle)}</div>' if subtitle else ""
    return (
        f'<div id="step-{n}" class="sl-step"><span class="sl-num">{n}</span>'
        f'<span class="sl-title">{escape(title)}</span></div>{sub}'
    )


def money_html(x: float) -> str:
    cls = "neg" if x < 0 else "pos"
    return f'<span class="mono {cls}">{"-" if x < 0 else "+"}${abs(x):,.0f}</span>'


def summary_card_html(
    scenario_name: str,
    value: float,
    normal: float,
    normal_pct: float,
    stress: float,
    stress_pct: float,
    hardest: tuple[str, float, float] | None,
    hedge_lines: list[str],
) -> str:
    """The plain-English result. Every input string is escaped; hedge_lines may contain trusted span markup."""
    verb = "lose" if normal < 0 else "gain"
    parts = [
        f'<div class="sl-card"><div class="k">If {escape(scenario_name)}</div>',
        f'<div class="big {"neg" if normal < 0 else "pos"}">{"-" if normal < 0 else "+"}${abs(normal):,.0f} '
        f'<span style="font-size:1.1rem">({normal_pct:+.1f}%)</span></div>',
        f"<p>Your <span class='mono'>${value:,.0f}</span> portfolio would likely {verb} about "
        f"{money_html(normal)} ({normal_pct:+.1f}%) in normal markets, and about "
        f"<span class='mono stress'>{'-' if stress < 0 else '+'}${abs(stress):,.0f}</span> ({stress_pct:+.1f}%) "
        "if stocks behave the way they have in past selloffs.</p>",
    ]
    if hardest:
        t, d, p = hardest
        parts.append(f"<p>Hit hardest: <span class='mono accent'>{escape(t)}</span> at {money_html(d)} ({p:+.1f}%).</p>")
    parts += [f"<p>{line}</p>" for line in hedge_lines]
    parts.append("</div>")
    return "".join(parts)
