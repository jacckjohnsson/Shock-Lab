"""Monte Carlo: thousands of possible futures for the portfolio, replayed from real market history.

Factor returns are drawn in blocks of consecutive days from history (a block bootstrap), so the
simulations keep real-world fat tails, volatility clustering and how oil, rates, the dollar, gold
and stocks moved together. Each stock then moves by its betas plus its own random noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import FACTORS, FactorModel

# Horizon name -> (trading days, calendar days)
HORIZONS = {"1 month": (21, 30), "3 months": (63, 90), "6 months": (126, 180), "1 year": (252, 365)}
N_SIMS = 4000
BLOCK_DAYS = 5  # draw history a week at a time
PERIOD_DAYS = {"daily": 1, "weekly": 5}


@dataclass
class Simulation:
    pnl: np.ndarray  # portfolio $ change at the horizon, one per simulation
    market_move: np.ndarray  # S&P 500 total return over the horizon, one per simulation
    bands: pd.DataFrame  # percentiles of the $ change at each period, for a fan chart
    value: float  # starting portfolio value


def simulate(
    models: dict[str, FactorModel],
    values: pd.Series,
    factor_history: pd.DataFrame,
    horizon_days: int,
    freq: str = "daily",
    n_sims: int = N_SIMS,
    seed: int = 0,
) -> Simulation:
    """Simulate a buy-and-hold portfolio over `horizon_days` trading days.

    Factor history is de-meaned first, so the simulations assume no average gain or loss:
    they show risk, not expected return. Stock alphas are left out for the same reason.
    """
    rng = np.random.default_rng(seed)
    tickers = [t for t in values.index if t in models]
    v0 = values.reindex(tickers).astype(float).values
    B = pd.DataFrame({t: models[t].betas for t in tickers}).T[FACTORS].values
    resid = np.array([models[t].residual_vol for t in tickers])

    hist = factor_history[FACTORS].dropna()
    hist = (hist - hist.mean()).values
    step = PERIOD_DAYS[freq]
    horizon = max(1, round(horizon_days / step))
    block = max(1, round(BLOCK_DAYS / step))
    if len(hist) < block * 2:
        raise ValueError("Not enough market history to simulate.")

    n_blocks = -(-horizon // block)
    starts = rng.integers(0, len(hist) - block + 1, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_sims, -1)[:, :horizon]
    paths = hist[idx]  # (n_sims, horizon, factors)

    growth = np.ones((n_sims, len(tickers)))
    totals = np.empty((n_sims, horizon + 1))
    totals[:, 0] = v0.sum()
    for t in range(horizon):
        r = paths[:, t, :] @ B.T + rng.normal(size=(n_sims, len(tickers))) * resid
        growth *= 1 + np.clip(r, -0.95, None)
        totals[:, t + 1] = growth @ v0
    change = totals - v0.sum()
    market_move = np.prod(1 + paths[:, :, FACTORS.index("market")], axis=1) - 1

    q = [5, 25, 50, 75, 95]
    bands = pd.DataFrame(np.percentile(change, q, axis=0).T, columns=[f"p{p}" for p in q])
    bands.insert(0, "Trading days", np.arange(horizon + 1) * step)
    return Simulation(pnl=change[:, -1], market_move=market_move, bands=bands, value=float(v0.sum()))


def risk_stats(pnl: np.ndarray, value: float) -> dict[str, float]:
    """Value at risk, expected shortfall and loss odds, all in dollars (losses positive) or %."""
    q05, q01 = np.percentile(pnl, [5, 1])
    return {
        "var95": float(-q05),
        "cvar95": float(-pnl[pnl <= q05].mean()),
        "var99": float(-q01),
        "median": float(np.median(pnl)),
        "p_loss": float((pnl < 0).mean() * 100),
        "p_loss_10": float((pnl < -0.10 * value).mean() * 100),
        "best5": float(np.percentile(pnl, 95)),
    }


def stats_table(results: dict[str, np.ndarray], value: float) -> pd.DataFrame:
    """One row per variant (e.g. no hedge vs. a hedge) with the headline risk numbers."""
    rows = []
    for name, pnl in results.items():
        s = risk_stats(pnl, value)
        rows.append(
            {
                "Portfolio": name,
                "Typical outcome ($)": s["median"],
                "Bad case, 1 in 20 ($)": -s["var95"],
                "Average of worst 5% ($)": -s["cvar95"],
                "Very bad case, 1 in 100 ($)": -s["var99"],
                "Good case, 1 in 20 ($)": s["best5"],
                "Chance of any loss (%)": s["p_loss"],
                "Chance of losing 10%+ (%)": s["p_loss_10"],
            }
        )
    return pd.DataFrame(rows)
