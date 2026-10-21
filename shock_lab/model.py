"""Factor returns, per-stock factor regressions and shock math."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# Regression factors: every stock gets a beta on each of these.
FACTORS = ["market", "oil", "rates", "dollar", "gold"]
# Stress gauges you can shock but that are not regression factors. VIX and credit spreads move
# almost in lockstep with the market, so as extra regressors they would make the betas unstable.
# Instead, a VIX or credit shock is translated into the factor moves that have typically come with it.
DRIVERS = ["vix", "credit"]
SCENARIO_VARS = FACTORS + DRIVERS
POINT_VARS = {"rates", "vix", "credit"}  # moves measured in points; the rest are % changes
FACTOR_LABELS = {
    "market": "S&P 500 (%)",
    "oil": "Crude oil (%)",
    "rates": "10-yr yield (pct points)",
    "dollar": "US dollar (%)",
    "gold": "Gold (%)",
    "vix": "VIX (points)",
    "credit": "Credit spread (pct points)",
}
MIN_OBSERVATIONS = 60
PERIODS_PER_YEAR = {"daily": 252, "weekly": 52}
STRESS_QUANTILE = 0.20  # "stress" = the worst 20% of market periods


def to_frequency(levels: pd.DataFrame | pd.Series, freq: str) -> pd.DataFrame | pd.Series:
    """Daily levels as-is, or the last value of each week (Friday)."""
    if freq == "weekly":
        return levels.resample("W-FRI").last()
    return levels


def build_factor_returns(market_prices: pd.Series, macro_levels: pd.DataFrame) -> pd.DataFrame:
    """Turn price/level series into factor moves for each period.

    market, oil, dollar, gold: simple % returns (as decimals).
    rates, credit: change in percentage points (1.0 = +1%). vix: change in VIX points.
    The stress gauges (vix, credit) are included when the macro data has them.
    """
    macro = macro_levels.reindex(market_prices.index).ffill()
    out = pd.DataFrame(index=market_prices.index)
    out["market"] = market_prices.pct_change(fill_method=None)
    # WTI briefly went negative in April 2020, which makes % changes meaningless.
    oil = macro["oil"].where(macro["oil"] > 0)
    out["oil"] = oil.pct_change(fill_method=None)
    out["rates"] = macro["rates"].diff()
    out["dollar"] = macro["dollar"].pct_change(fill_method=None)
    out["gold"] = macro["gold"].pct_change(fill_method=None)
    for d in DRIVERS:
        if d in macro:
            out[d] = macro[d].diff()
    out = out.replace([np.inf, -np.inf], np.nan)
    # Drop absurd oil moves that remain around the negative-price episode.
    out.loc[out["oil"].abs() > 0.5, "oil"] = np.nan
    return out.dropna()


def stock_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None)


def last_years(df: pd.DataFrame, years: float, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rows in the `years` before `end` (exclusive), or before the last row if end is None."""
    if end is None:
        end = df.index.max() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(days=round(365.25 * years))
    return df.loc[(df.index >= start) & (df.index < end)]


def stress_periods(factors: pd.DataFrame, quantile: float = STRESS_QUANTILE) -> pd.Index:
    """Dates when the market return was in its worst `quantile`."""
    cutoff = factors["market"].quantile(quantile)
    return factors.index[factors["market"] <= cutoff]


@dataclass
class FactorModel:
    ticker: str
    alpha: float
    betas: pd.Series  # indexed by FACTORS
    beta_cov: pd.DataFrame  # covariance of the beta estimates
    r_squared: float
    residual_vol: float  # per period
    n_obs: int


def fit_factor_model(ticker: str, returns: pd.Series, factors: pd.DataFrame) -> FactorModel:
    data = pd.concat([returns.rename("y"), factors[FACTORS]], axis=1, join="inner").dropna()
    if len(data) < MIN_OBSERVATIONS:
        raise ValueError(f"{ticker}: only {len(data)} periods of overlapping data (need {MIN_OBSERVATIONS}).")
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


def fit_all(returns: pd.DataFrame, factors: pd.DataFrame) -> tuple[dict[str, FactorModel], dict[str, str]]:
    """Fit one model per ticker. Returns (models, errors by ticker)."""
    models, errors = {}, {}
    for t in returns.columns:
        try:
            models[t] = fit_factor_model(t, returns[t], factors)
        except ValueError as e:
            errors[t] = str(e)
    return models, errors


def rolling_betas(returns: pd.Series, factors: pd.DataFrame, window: int) -> pd.DataFrame:
    """Betas re-estimated over a moving window (e.g. 252 days = 1 year)."""
    data = pd.concat([returns.rename("y"), factors[FACTORS]], axis=1, join="inner").dropna()
    if len(data) < window + 1:
        return pd.DataFrame(columns=FACTORS)
    res = RollingOLS(data["y"], sm.add_constant(data[FACTORS]), window=window).fit()
    return res.params[FACTORS].dropna()


def implied_factor_moves(shock: dict[str, float], factors: pd.DataFrame, co_move: bool = True) -> pd.Series:
    """Fill in the variables you didn't shock with their expected co-moves.

    If x is the vector of shocked variables, the expected move of the others is
    E[y | x] = Cov(y, x) Cov(x, x)^-1 x  (conditional mean of a multivariate normal).

    co_move=False holds unset factors at 0, except that a VIX or credit shock is always
    translated into factor moves (on its own it has no beta to act through).
    """
    available = [v for v in SCENARIO_VARS if v in factors.columns]
    moves = pd.Series(0.0, index=available)
    shocked = [v for v in available if shock.get(v, 0.0) != 0.0]
    if not shocked:
        return moves
    moves[shocked] = [shock[v] for v in shocked]
    given = shocked if co_move else [v for v in shocked if v in DRIVERS]
    if not given:
        return moves
    others = [v for v in available if v not in shocked] if co_move else [f for f in FACTORS if f not in shocked]
    if others:
        cov = factors[available].cov()
        weights = cov.loc[others, given].values @ np.linalg.pinv(cov.loc[given, given].values)
        moves[others] = weights @ moves[given].values
    return moves


def move_text(var: str, move: float) -> str:
    """'S&P 500 -15.0%' or 'VIX +10.00 pts'."""
    label = FACTOR_LABELS[var].split(" (")[0]
    return f"{label} {move:+.2f} pts" if var in POINT_VARS else f"{label} {move * 100:+.1f}%"


def display_units(var: str) -> float:
    """Multiplier from model units to what the sliders and tables show."""
    return 1.0 if var in POINT_VARS else 100.0


def predicted_return(model: FactorModel, factor_moves: pd.Series) -> tuple[float, float]:
    """(estimated return, its standard error from beta uncertainty)."""
    s = factor_moves[FACTORS].values.astype(float)
    ret = float(model.betas[FACTORS].values @ s)
    se = float(np.sqrt(max(s @ model.beta_cov.values @ s, 0.0)))
    return ret, se


def holding_impacts(
    models: dict[str, FactorModel],
    holdings: pd.DataFrame,
    last_prices: pd.Series,
    factor_moves: pd.Series,
    stress_models: dict[str, FactorModel] | None = None,
) -> pd.DataFrame:
    """Estimated % and $ impact per holding with a ~95% range from beta uncertainty.

    holdings: columns Ticker, Shares. If stress_models is given, adds the same
    scenario estimated with betas fit only on stress periods.
    """
    rows = []
    for _, h in holdings.iterrows():
        t = h["Ticker"]
        if t not in models:
            continue
        ret, se = predicted_return(models[t], factor_moves)
        value = float(h["Shares"]) * float(last_prices[t])
        row = {
            "Ticker": t,
            "Shares": float(h["Shares"]),
            "Price": float(last_prices[t]),
            "Value ($)": value,
            "Impact (%)": ret * 100,
            "Low (%)": (ret - 1.96 * se) * 100,
            "High (%)": (ret + 1.96 * se) * 100,
            "Impact ($)": value * ret,
        }
        if stress_models is not None:
            if t in stress_models:
                s_ret, _ = predicted_return(stress_models[t], factor_moves)
                row["Stress impact (%)"] = s_ret * 100
                row["Stress impact ($)"] = value * s_ret
            else:
                row["Stress impact (%)"] = np.nan
                row["Stress impact ($)"] = np.nan
        row["R²"] = models[t].r_squared
        rows.append(row)
    return pd.DataFrame(rows)


def portfolio_summary(impacts: pd.DataFrame, column: str = "Impact ($)") -> dict[str, float]:
    total_value = impacts["Value ($)"].sum()
    total_impact = impacts[column].sum()
    return {
        "value": float(total_value),
        "impact": float(total_impact),
        "impact_pct": float(total_impact / total_value * 100) if total_value else 0.0,
    }
