"""Portfolio risk from the factor model: covariance, risk contributions and concentration warnings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import FACTORS, FactorModel

MAX_WEIGHT = 0.25  # warn when one holding is more than 25% of the portfolio's value
MAX_RISK_SHARE = 0.35  # ...or drives more than 35% of its ups and downs
MIN_EFFECTIVE_HOLDINGS = 4.0  # ...or the portfolio acts like fewer than 4 equal positions


def stock_covariance(
    models: dict[str, FactorModel], tickers: list[str], factors: pd.DataFrame, periods_per_year: int
) -> pd.DataFrame:
    """Annualized covariance of stock returns implied by the factor model.

    Cov = B Cov(factors) B' + D, where D holds each stock's own (residual) variance. Building it from
    the factors instead of raw stock-by-stock history keeps it stable even with many holdings.
    """
    B = pd.DataFrame({t: models[t].betas for t in tickers}).T[FACTORS].values
    fcov = factors[FACTORS].cov().values
    resid = np.diag([models[t].residual_vol ** 2 for t in tickers])
    return pd.DataFrame((B @ fcov @ B.T + resid) * periods_per_year, index=tickers, columns=tickers)


def portfolio_vol(weights: np.ndarray, cov: pd.DataFrame | np.ndarray) -> float:
    c = np.asarray(cov)
    return float(np.sqrt(max(weights @ c @ weights, 0.0)))


def risk_contributions(weights: np.ndarray, cov: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Share of portfolio variance from each holding: w_i (Cov w)_i / w'Cov w. Sums to 1."""
    c = np.asarray(cov)
    total = weights @ c @ weights
    if total <= 0:
        return np.zeros_like(weights)
    return weights * (c @ weights) / total


def concentration(values: pd.Series, cov: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """(per-holding table, portfolio stats, plain-English warnings). values: position $ by ticker."""
    tickers = list(cov.index)
    v = values.reindex(tickers).astype(float)
    w = (v / v.sum()).values
    rc = risk_contributions(w, cov)
    table = pd.DataFrame(
        {
            "Ticker": tickers,
            "Value ($)": v.values,
            "Weight (%)": w * 100,
            "Share of risk (%)": rc * 100,
            "Volatility (%/yr)": np.sqrt(np.diag(cov.values)) * 100,
        }
    ).sort_values("Share of risk (%)", ascending=False, ignore_index=True)
    stats = {
        "vol": portfolio_vol(w, cov) * 100,
        "effective_n": float(1 / np.sum(w**2)),
        "top_weight": float(w.max() * 100),
        "top_risk": float(rc.max() * 100),
    }
    warnings = []
    for _, row in table.iterrows():
        if row["Weight (%)"] > MAX_WEIGHT * 100:
            warnings.append(f"{row['Ticker']} is {row['Weight (%)']:.0f}% of your portfolio's value.")
        if row["Share of risk (%)"] > MAX_RISK_SHARE * 100:
            warnings.append(f"{row['Ticker']} drives about {row['Share of risk (%)']:.0f}% of your portfolio's ups and downs.")
    if stats["effective_n"] < MIN_EFFECTIVE_HOLDINGS:
        warnings.append(
            f"By size, your portfolio acts like about {stats['effective_n']:.1f} equal positions, "
            "so a single stock's news can move the whole thing."
        )
    return table, stats, warnings
