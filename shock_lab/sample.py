"""Synthetic market data for offline demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Made-up "true" betas: market, oil, rates (per point), dollar.
SAMPLE_BETAS = {
    "OILCO": [0.9, 0.45, 0.01, -0.2],
    "TECHCO": [1.3, -0.05, -0.04, -0.1],
    "BANKCO": [1.1, 0.05, 0.03, 0.0],
    "UTILCO": [0.5, 0.0, -0.05, 0.05],
}


def make_sample_data(n_days: int = 2520, seed: int = 0) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return (market prices, macro levels, stock prices) generated from known betas."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-16", periods=n_days)
    cov = np.array(
        [
            [1.0e-4, 2.0e-5, 1.0e-5, -1.0e-5],
            [2.0e-5, 4.0e-4, 2.0e-5, -2.0e-5],
            [1.0e-5, 2.0e-5, 4.0e-3, 5.0e-6],
            [-1.0e-5, -2.0e-5, 5.0e-6, 1.0e-5],
        ]
    )
    f = rng.multivariate_normal(np.zeros(4), cov, size=n_days)
    market = pd.Series(400 * np.cumprod(1 + f[:, 0]), index=idx, name="SPY")
    macro = pd.DataFrame(
        {
            "oil": 70 * np.cumprod(1 + f[:, 1]),
            "rates": 3.0 + np.cumsum(f[:, 2]),
            "dollar": 115 * np.cumprod(1 + f[:, 3]),
        },
        index=idx,
    )
    stocks = {}
    for t, b in SAMPLE_BETAS.items():
        r = f[1:] @ np.array(b) + rng.normal(0, 0.008, n_days - 1)
        stocks[t] = 100 * np.cumprod(np.r_[1.0, 1 + r])
    return market, macro, pd.DataFrame(stocks, index=idx)
