import numpy as np
import pandas as pd
import pytest

from shock_lab.data import parse_fred_csv
from shock_lab.model import (
    FACTORS,
    build_factor_returns,
    fit_all,
    fit_factor_model,
    holding_impacts,
    implied_factor_moves,
    portfolio_summary,
)
from shock_lab.portfolio import MAX_HOLDINGS, clean_holdings
from shock_lab.sample import SAMPLE_BETAS, make_sample_data
from shock_lab.shocks import PRESETS


@pytest.fixture(scope="module")
def sample():
    market, macro, prices = make_sample_data()
    factors = build_factor_returns(market, macro)
    models, errors = fit_all(prices, factors)
    return market, macro, prices, factors, models, errors


def test_regression_recovers_known_betas(sample):
    *_, models, errors = sample
    assert not errors
    for t, true_betas in SAMPLE_BETAS.items():
        m = models[t]
        se = np.sqrt(np.diag(m.beta_cov.values))
        # Each estimate should be within 4 standard errors of the true beta.
        assert (np.abs(m.betas[FACTORS].values - true_betas) < 4 * se).all(), t
        np.testing.assert_allclose(m.betas[["market", "oil"]].values, true_betas[:2], atol=0.05)


def test_factor_returns_units():
    idx = pd.bdate_range("2024-01-01", periods=3)
    market = pd.Series([100.0, 110.0, 99.0], index=idx)
    macro = pd.DataFrame({"oil": [50.0, 55.0, 55.0], "rates": [4.0, 4.25, 4.0], "dollar": [100.0, 100.0, 102.0]}, index=idx)
    f = build_factor_returns(market, macro)
    assert f.loc[idx[1], "market"] == pytest.approx(0.10)
    assert f.loc[idx[1], "oil"] == pytest.approx(0.10)
    assert f.loc[idx[1], "rates"] == pytest.approx(0.25)  # percentage points, not %
    assert f.loc[idx[2], "dollar"] == pytest.approx(0.02)


def test_negative_oil_prices_are_dropped_not_crashing():
    idx = pd.bdate_range("2020-04-15", periods=5)
    market = pd.Series([100.0, 101, 102, 103, 104], index=idx)
    macro = pd.DataFrame({"oil": [20.0, 10.0, -37.6, 10.0, 12.0], "rates": 1.0, "dollar": 100.0}, index=idx)
    f = build_factor_returns(market, macro)
    assert np.isfinite(f.values).all()
    assert idx[2] not in f.index and idx[3] not in f.index


def test_macro_gaps_are_forward_filled():
    idx = pd.bdate_range("2024-01-01", periods=4)
    market = pd.Series([100.0, 101, 102, 103], index=idx)
    macro = pd.DataFrame({"oil": [50.0, 51.0, 52.0], "rates": [4.0, 4.1, 4.2], "dollar": [100.0, 101.0, 102.0]}, index=idx[[0, 1, 3]])
    f = build_factor_returns(market, macro)
    assert idx[2] in f.index and f.loc[idx[2], "oil"] == 0.0


def test_too_little_history_raises():
    idx = pd.bdate_range("2024-01-01", periods=50)
    factors = pd.DataFrame(np.random.default_rng(1).normal(size=(50, 4)) * 0.01, index=idx, columns=FACTORS)
    with pytest.raises(ValueError, match="overlapping data"):
        fit_factor_model("NEW", pd.Series(0.01, index=idx), factors)


def test_direct_shock_impact_is_beta_times_move(sample):
    _, _, prices, _, models, _ = sample
    holdings = pd.DataFrame({"Ticker": ["OILCO"], "Shares": [10]})
    last = prices.iloc[-1]
    moves = pd.Series({"market": 0.0, "oil": 0.20, "rates": 0.0, "dollar": 0.0})
    out = holding_impacts(models, holdings, last, moves).iloc[0]
    beta_oil = models["OILCO"].betas["oil"]
    assert out["Impact (%)"] == pytest.approx(beta_oil * 20)
    assert out["Impact ($)"] == pytest.approx(10 * last["OILCO"] * beta_oil * 0.20)
    assert out["Low (%)"] < out["Impact (%)"] < out["High (%)"]


def test_implied_moves_match_conditional_mean():
    rng = np.random.default_rng(2)
    cov = np.array([[1.0, 0.5, 0, 0], [0.5, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    factors = pd.DataFrame(rng.multivariate_normal(np.zeros(4), cov, size=200_000), columns=FACTORS)
    moves = implied_factor_moves({"oil": 2.0}, factors)
    assert moves["oil"] == 2.0
    assert moves["market"] == pytest.approx(1.0, abs=0.02)  # 0.5 / 1.0 * 2
    assert moves["rates"] == pytest.approx(0.0, abs=0.02)


def test_implied_moves_no_shock_is_zero(sample):
    factors = sample[3]
    assert (implied_factor_moves({}, factors) == 0).all()


def test_portfolio_summary():
    impacts = pd.DataFrame({"Value ($)": [1000.0, 3000.0], "Impact ($)": [-100.0, 60.0]})
    s = portfolio_summary(impacts)
    assert s == {"value": 4000.0, "impact": -40.0, "impact_pct": -1.0}


def test_clean_holdings():
    raw = pd.DataFrame({"Ticker": [" aapl", "AAPL", "msft", "", None, "XOM"], "Shares": [5, 5, "x", 3, 2, 0]})
    out, warnings = clean_holdings(raw)
    assert out.to_dict("list") == {"Ticker": ["AAPL"], "Shares": [10.0]}
    assert len(warnings) == 2


def test_clean_holdings_limit():
    raw = pd.DataFrame({"Ticker": [f"T{i}" for i in range(40)], "Shares": 1})
    out, warnings = clean_holdings(raw)
    assert len(out) == MAX_HOLDINGS and warnings


def test_parse_fred_csv_handles_missing_values():
    text = "observation_date,DGS10\n2024-01-01,.\n2024-01-02,3.95\n2024-01-03,3.91\n"
    s = parse_fred_csv(text, "rates")
    assert list(s.values) == [3.95, 3.91]
    assert s.name == "rates"


def test_presets_cover_the_four_factors():
    assert len(PRESETS) == 4
    assert {k for p in PRESETS.values() for k in p} == set(FACTORS)


def test_fetch_prices_handles_yfinance_multiindex(monkeypatch):
    import sys
    import types
    from datetime import date

    from shock_lab.data import fetch_prices

    idx = pd.date_range("2024-01-01", periods=3, tz="America/New_York")
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["AAPL", "SPY"]], names=["Price", "Ticker"])
    raw = pd.DataFrame(np.arange(12.0).reshape(3, 4), index=idx, columns=cols)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: raw))
    out = fetch_prices(["AAPL", "SPY"], date(2024, 1, 1), date(2024, 1, 4))
    assert list(out.columns) == ["AAPL", "SPY"]
    assert out.index.tz is None
