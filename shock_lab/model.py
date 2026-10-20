"""Factor returns, per-stock factor regressions and shock math."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTORS = ["market", "oil", "rates", "dollar"]
FACTOR_LABELS = {
    "market": "S&P 500 (%)",
    "oil": "Crude oil (%)",
    "rates": "10-yr yield (pct points)",
    "dollar": "US dollar (%)",
}
MIN_OBSERVATIONS = 250  # about one year of trading days


def build_factor_returns(market_prices: pd.Series, macro_levels: pd.DataFrame) -> pd.DataFrame:
    """Turn price/level series into daily factor moves.

    market, oil, dollar: simple % returns (as decimals).
    rates: change in yield, in percentage points (1.0 = +1%).
    """
    macro = macro_levels.reindex(market_prices.index).ffill()
    out = pd.DataFrame(index=market_prices.index)
    out["market"] = market_prices.pct_change()
    # WTI briefly went negative in April 2020, which makes % changes meaningless.
    oil = macro["oil"].where(macro["oil"] > 0)
    out["oil"] = oil.pct_change(fill_method=None)
    out["rates"] = macro["rates"].diff()
    out["dollar"] = macro["dollar"].pct_change(fill_method=None)
    out = out.replace([np.inf, -np.inf], np.nan)
    # Drop absurd one-day oil moves that remain around the negative-price episode.
    out.loc[out["oil"].abs() > 0.5, "oil"] = np.nan
    return out.dropna()


@dataclass
class FactorModel:
    ticker: str
    alpha: float
    betas: pd.Series  # indexed by FACTORS
    beta_cov: pd.DataFrame  # covariance of the beta estimates
    r_squared: float
    residual_vol: float  # daily
    n_obs: int


def fit_factor_model(ticker: str, stock_returns: pd.Series, factors: pd.DataFrame) -> FactorModel:
    data = pd.concat([stock_returns.rename("y"), factors[FACTORS]], axis=1).dropna()
    if len(data) < MIN_OBSERVATIONS:
        raise ValueError(f"{ticker}: only {len(data)} days of overlapping data (need {MIN_OBSERVATIONS}).")
    X = sm.add_constant(data[FACTORS])
    res = sm.OLS(data["y"], X).fit()
    cov = res.cov_params().loc[FACTORS, FACTORS]
    return FactorModel(
        ticker=ticker,
        alpha=float(res.params["const"]),
        betas=res.params[FACTORS].astype(float),
        beta_cov=cov,
        r_squared=float(res.rsquared),
        residual_vol=float(np.sqrt(res.mse_resid)),
        n_obs=int(res.nobs),
    )


def fit_all(prices: pd.DataFrame, factors: pd.DataFrame) -> tuple[dict[str, FactorModel], dict[str, str]]:
    """Fit one model per ticker. Returns (models, errors by ticker)."""
    returns = prices.pct_change(fill_method=None)
    models, errors = {}, {}
    for t in prices.columns:
        try:
            models[t] = fit_factor_model(t, returns[t], factors)
        except ValueError as e:
            errors[t] = str(e)
    return models, errors


def implied_factor_moves(shock: dict[str, float], factors: pd.DataFrame) -> pd.Series:
    """Fill in the factors you didn't shock with their expected co-moves.

    If x is the vector of shocked factors, the expected move of the others is
    E[y | x] = Cov(y, x) Cov(x, x)^-1 x  (conditional mean of a multivariate normal).
    """
    shocked = [f for f in FACTORS if shock.get(f, 0.0) != 0.0]
    moves = pd.Series(0.0, index=FACTORS)
    if not shocked:
        return moves
    cov = factors[FACTORS].cov()
    x = np.array([shock[f] for f in shocked])
    others = [f for f in FACTORS if f not in shocked]
    moves[shocked] = x
    if others:
        weights = cov.loc[others, shocked].values @ np.linalg.inv(cov.loc[shocked, shocked].values)
        moves[others] = weights @ x
    return moves


def holding_impacts(
    models: dict[str, FactorModel],
    holdings: pd.DataFrame,
    last_prices: pd.Series,
    factor_moves: pd.Series,
) -> pd.DataFrame:
    """Estimated % and $ impact per holding with a ~95% range from beta uncertainty.

    holdings: columns Ticker, Shares.
    """
    s = factor_moves[FACTORS].values
    rows = []
    for _, h in holdings.iterrows():
        t = h["Ticker"]
        if t not in models:
            continue
        m = models[t]
        ret = float(m.betas[FACTORS].values @ s)
        se = float(np.sqrt(max(s @ m.beta_cov.values @ s, 0.0)))
        value = float(h["Shares"]) * float(last_prices[t])
        rows.append(
            {
                "Ticker": t,
                "Shares": float(h["Shares"]),
                "Price": float(last_prices[t]),
                "Value ($)": value,
                "Impact (%)": ret * 100,
                "Low (%)": (ret - 1.96 * se) * 100,
                "High (%)": (ret + 1.96 * se) * 100,
                "Impact ($)": value * ret,
                "R²": m.r_squared,
            }
        )
    return pd.DataFrame(rows)


def portfolio_summary(impacts: pd.DataFrame) -> dict[str, float]:
    total_value = impacts["Value ($)"].sum()
    total_impact = impacts["Impact ($)"].sum()
    return {
        "value": float(total_value),
        "impact": float(total_impact),
        "impact_pct": float(total_impact / total_value * 100) if total_value else 0.0,
    }
