"""Synthetic market data for offline demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Made-up "true" betas: market, oil, rates (per point), dollar, gold.
SAMPLE_BETAS = {
    "OILCO": [0.9, 0.45, 0.01, -0.2, 0.1],
    "TECHCO": [1.3, -0.05, -0.04, -0.1, 0.0],
    "BANKCO": [1.1, 0.05, 0.03, 0.0, -0.05],
    "UTILCO": [0.5, 0.0, -0.05, 0.05, 0.15],
}


def make_sample_data(n_days: int = 5400, seed: int = 0) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return (market prices, macro levels, stock prices) generated from known betas."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-16", periods=n_days)
    cov = np.array(
        [
            [1.0e-4, 2.0e-5, 1.0e-5, -1.0e-5, 5.0e-6],
            [2.0e-5, 4.0e-4, 2.0e-5, -2.0e-5, 2.0e-5],
            [1.0e-5, 2.0e-5, 4.0e-3, 5.0e-6, -1.0e-5],
            [-1.0e-5, -2.0e-5, 5.0e-6, 1.0e-5, -1.2e-5],
            [5.0e-6, 2.0e-5, -1.0e-5, -1.2e-5, 1.0e-4],
        ]
    )
    f = rng.multivariate_normal(np.zeros(5), cov, size=n_days)
    market = pd.Series(400 * np.cumprod(1 + f[:, 0]), index=idx, name="SPY")
    macro = pd.DataFrame(
        {
            "oil": 70 * np.cumprod(1 + f[:, 1]),
            "rates": 3.0 + np.cumsum(f[:, 2]),
            "dollar": 115 * np.cumprod(1 + f[:, 3]),
            "gold": 180 * np.cumprod(1 + f[:, 4]),
        },
        index=idx,
    )
    stocks = {}
    for t, b in SAMPLE_BETAS.items():
        r = f[1:] @ np.array(b) + rng.normal(0, 0.008, n_days - 1)
        stocks[t] = 100 * np.cumprod(np.r_[1.0, 1 + r])
    # VIX: mean-reverts to 18 and jumps when the market falls (about +1.2 points per -1%).
    vix = np.empty(n_days)
    vix[0] = 18.0
    shocks = rng.normal(0, 0.8, n_days)
    for i in range(1, n_days):
        vix[i] = np.clip(vix[i - 1] + 0.05 * (18 - vix[i - 1]) - 120 * f[i, 0] + shocks[i], 9, 80)
    macro["vix"] = vix
    # Credit spread: mean-reverts to 2 points and widens when the market falls (about +0.6 per -10%).
    credit = np.empty(n_days)
    credit[0] = 2.0
    noise = rng.normal(0, 0.015, n_days)
    for i in range(1, n_days):
        credit[i] = np.clip(credit[i - 1] + 0.02 * (2.0 - credit[i - 1]) - 6.0 * f[i, 0] + noise[i], 0.5, 8.0)
    macro["credit"] = credit
    macro["tbill"] = 4.0
    return market, macro, pd.DataFrame(stocks, index=idx)
