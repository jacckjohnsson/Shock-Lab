"""A light mean-variance optimizer, built only from the stocks you already hold.

Textbook mean-variance is famously touchy: small changes in expected returns swing the weights to
extremes. To keep it sensible, risk comes from the factor model's covariance, expected returns
default to "market-implied" (CAPM: T-bill rate + market beta x equity premium), weights are
long-only, and no stock can exceed a cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .model import FactorModel

EQUITY_PREMIUM = 0.05  # assumed yearly return of stocks over T-bills
RETURN_METHODS = ["Market-implied (steadier)", "Historical average (noisy)"]
FRONTIER_POINTS = 25


def expected_returns(
    models: dict[str, FactorModel],
    tickers: list[str],
    rf: float,
    method: str,
    returns: pd.DataFrame | None = None,
    periods_per_year: int = 252,
) -> pd.Series:
    """Yearly expected return per stock, as decimals."""
    if method.startswith("Historical"):
        if returns is None:
            raise ValueError("Historical average needs the return history.")
        return returns[tickers].mean() * periods_per_year
    return pd.Series({t: rf + models[t].betas["market"] * EQUITY_PREMIUM for t in tickers})


def stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> dict[str, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return {"return": ret, "vol": vol, "sharpe": (ret - rf) / vol if vol > 0 else np.nan}


def _solve(objective, n: int, cap: float, constraints: list) -> np.ndarray:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + constraints
    res = minimize(objective, np.full(n, 1 / n), method="SLSQP", bounds=[(0.0, cap)] * n, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, cap)
    return w / w.sum()


def max_return(mu: np.ndarray, cap: float) -> np.ndarray:
    """Highest-return weights under the cap: fill the best stocks first."""
    w = np.zeros_like(mu)
    left = 1.0
    for i in np.argsort(-mu):
        w[i] = min(cap, left)
        left -= w[i]
        if left <= 1e-12:
            break
    return w


def min_variance(cov: np.ndarray, cap: float, mu: np.ndarray | None = None, target: float | None = None) -> np.ndarray:
    """Lowest-risk weights, optionally with expected return of at least `target`."""
    cons = [] if target is None else [{"type": "ineq", "fun": lambda w: w @ mu - target}]
    return _solve(lambda w: w @ cov @ w, len(cov), cap, cons)


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float, cap: float) -> np.ndarray:
    """Best expected return per unit of risk (highest Sharpe ratio)."""
    def neg_sharpe(w):
        vol = np.sqrt(max(w @ cov @ w, 1e-16))
        return -(w @ mu - rf) / vol
    return _solve(neg_sharpe, len(cov), cap, [])


def frontier_curve(mu: np.ndarray, cov: np.ndarray, cap: float, n_points: int = FRONTIER_POINTS) -> pd.DataFrame:
    """Lowest risk for each expected return from the minimum-variance mix up to the highest possible."""
    lo = float(min_variance(cov, cap) @ mu)
    hi = float(max_return(mu, cap) @ mu)
    rows = []
    for target in np.linspace(lo, hi, n_points):
        w = min_variance(cov, cap, mu, target)
        rows.append({"Return (%/yr)": float(w @ mu) * 100, "Volatility (%/yr)": float(np.sqrt(w @ cov @ w)) * 100})
    return pd.DataFrame(rows)


def model_portfolios(
    current: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float, cap: float
) -> dict[str, np.ndarray]:
    """Your portfolio plus three textbook alternatives using the same stocks.

    current: position $ by ticker, in the same order as mu and cov.
    """
    m, c = mu.values, cov.values
    w0 = (current / current.sum()).values
    target = min(float(w0 @ m), float(max_return(m, cap) @ m))
    return {
        "Your portfolio": w0,
        "Same expected return, less risk": min_variance(c, cap, m, target),
        "Lowest risk": min_variance(c, cap),
        "Best return per unit of risk": max_sharpe(m, c, rf, cap),
    }


def compare_portfolios(
    portfolios: dict[str, np.ndarray],
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    value: float,
    impact_pct: pd.Series,
    stress_pct: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary per portfolio, weights per ticker). Scenario columns apply each stock's % impact to the mix."""
    rows = []
    for name, w in portfolios.items():
        s = stats(w, mu.values, cov.values, rf)
        rows.append(
            {
                "Portfolio": name,
                "Expected return (%/yr)": s["return"] * 100,
                "Volatility (%/yr)": s["vol"] * 100,
                "Sharpe ratio": s["sharpe"],
                "Scenario, normal ($)": value * float(w @ impact_pct.reindex(mu.index).fillna(0).values) / 100,
                "Scenario, stress ($)": value * float(w @ stress_pct.reindex(mu.index).fillna(0).values) / 100,
            }
        )
    weights = pd.DataFrame({name: w * 100 for name, w in portfolios.items()}, index=mu.index)
    weights.index.name = "Ticker"
    return pd.DataFrame(rows), weights
