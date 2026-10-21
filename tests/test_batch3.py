"""Gold/VIX/credit scenarios, concentration, Monte Carlo and the efficient frontier."""

import numpy as np
import pandas as pd
import pytest

from shock_lab.frontier import (
    RETURN_METHODS,
    compare_portfolios,
    expected_returns,
    frontier_curve,
    max_return,
    model_portfolios,
)
from shock_lab.hedging import HedgeSetup, hedge_pnl
from shock_lab.model import (
    FACTORS,
    SCENARIO_VARS,
    build_factor_returns,
    fit_all,
    implied_factor_moves,
    move_text,
    stock_returns,
)
from shock_lab.montecarlo import risk_stats, simulate, stats_table
from shock_lab.risk import concentration, risk_contributions, stock_covariance
from shock_lab.sample import SAMPLE_BETAS, make_sample_data


@pytest.fixture(scope="module")
def setup():
    market, macro, prices = make_sample_data()
    factors = build_factor_returns(market, macro)
    returns = stock_returns(prices)
    models, _ = fit_all(returns, factors)
    tickers = list(SAMPLE_BETAS)
    values = pd.Series([8000.0, 2000.0, 4000.0, 3000.0], index=tickers)
    cov = stock_covariance(models, tickers, factors, 252)
    return factors, returns, models, values, cov, market


def test_factor_returns_include_gold_and_stress_gauges(setup):
    factors = setup[0]
    assert list(factors.columns) == SCENARIO_VARS
    # Sample VIX rises about 1.2 points per -1% market day; credit widens.
    assert np.corrcoef(factors["market"], factors["vix"])[0, 1] < -0.5
    assert np.corrcoef(factors["market"], factors["credit"])[0, 1] < -0.3


def test_gold_beta_is_recovered(setup):
    models = setup[2]
    for t, b in SAMPLE_BETAS.items():
        assert models[t].betas["gold"] == pytest.approx(b[FACTORS.index("gold")], abs=0.05), t


def test_vix_shock_translates_into_a_market_drop(setup):
    factors = setup[0]
    moves = implied_factor_moves({"vix": 15.0}, factors)
    assert moves["vix"] == 15.0
    assert moves["market"] < -0.05
    # Even with co-moves off, a VIX shock still acts through the factors; unset gauges stay at 0.
    off = implied_factor_moves({"vix": 15.0}, factors, co_move=False)
    assert off["market"] == pytest.approx(moves["market"], rel=1e-6)
    assert off["credit"] == 0.0


def test_co_moves_off_keeps_plain_factor_shocks_alone(setup):
    factors = setup[0]
    moves = implied_factor_moves({"oil": 0.2}, factors, co_move=False)
    assert moves["oil"] == 0.2
    assert (moves.drop("oil") == 0).all()


def test_move_text_units():
    assert move_text("vix", 15.0) == "VIX +15.00 pts"
    assert move_text("gold", -0.1) == "Gold -10.0%"


def test_risk_contributions_sum_to_one(setup):
    *_, values, cov, _ = setup
    w = (values / values.sum()).values
    rc = risk_contributions(w, cov)
    assert rc.sum() == pytest.approx(1.0)


def test_concentration_warns_on_a_dominant_holding(setup):
    *_, values, cov, _ = setup
    table, stats, warnings = concentration(values, cov)
    assert table["Weight (%)"].sum() == pytest.approx(100)
    assert stats["effective_n"] == pytest.approx(1 / ((values / values.sum()) ** 2).sum())
    assert any("OILCO" in w and "value" in w for w in warnings)
    # An evenly spread portfolio of different stocks raises no value warning.
    even = pd.Series(1000.0, index=values.index)
    _, _, w2 = concentration(even, cov)
    assert not any("of your portfolio's value" in w for w in w2)


def test_monte_carlo_is_reproducible_and_sensible(setup):
    factors, _, models, values, cov, _ = setup
    a = simulate(models, values, factors, 252, seed=3, n_sims=2000)
    b = simulate(models, values, factors, 252, seed=3, n_sims=2000)
    np.testing.assert_array_equal(a.pnl, b.pnl)
    s = risk_stats(a.pnl, a.value)
    # Close to the normal approximation: 1.645 x yearly volatility x value.
    w = (values / values.sum()).values
    vol = np.sqrt(w @ cov.values @ w)
    assert s["var95"] == pytest.approx(1.645 * vol * a.value, rel=0.25)
    assert s["cvar95"] > s["var95"] > 0
    assert list(a.bands.columns) == ["Trading days", "p5", "p25", "p50", "p75", "p95"]
    assert (a.bands["p5"] <= a.bands["p95"]).all()


def test_weekly_monte_carlo_runs(setup):
    factors, _, models, values, *_ = setup
    weekly = build_factor_returns(*[x.resample("W-FRI").last() for x in make_sample_data()[:2]])
    sim = simulate(models, values, weekly, 63, freq="weekly", n_sims=500)
    assert len(sim.bands) == round(63 / 5) + 1


def test_puts_cut_the_simulated_tail(setup):
    factors, _, models, values, _, market = setup
    sim = simulate(models, values, factors, 63, n_sims=3000)
    h = HedgeSetup(spot=float(market.iloc[-1]), vix=18, rate=0.04, dollar_beta=float(values.sum()),
                   hedge_ratio=1.0, strike_pct=0.95, expiry_days=90, horizon_days=90)
    hedged = sim.pnl + hedge_pnl("Index puts", h, sim.market_move, sim.pnl)
    t = stats_table({"No hedge": sim.pnl, "Puts": hedged}, sim.value).set_index("Portfolio")
    assert t.loc["Puts", "Very bad case, 1 in 100 ($)"] > t.loc["No hedge", "Very bad case, 1 in 100 ($)"]


def test_frontier_portfolios(setup):
    _, returns, models, values, cov, _ = setup
    tickers = list(values.index)
    mu = expected_returns(models, tickers, 0.04, RETURN_METHODS[0])
    assert mu["TECHCO"] > mu["UTILCO"]  # higher market beta, higher market-implied return
    cap = 0.4
    ports = model_portfolios(values, mu, cov, 0.04, cap)
    c, m = cov.values, mu.values
    var = {k: w @ c @ w for k, w in ports.items()}
    for name, w in ports.items():
        assert w.sum() == pytest.approx(1.0)
        if name != "Your portfolio":
            assert (w >= -1e-9).all() and (w <= cap + 1e-6).all(), name
    assert var["Lowest risk"] <= min(var.values()) + 1e-10
    same = ports["Same expected return, less risk"]
    assert same @ m >= ports["Your portfolio"] @ m - 1e-6
    assert var["Same expected return, less risk"] <= var["Your portfolio"]
    sharpe = {k: (w @ m - 0.04) / np.sqrt(var[k]) for k, w in ports.items()}
    assert sharpe["Best return per unit of risk"] >= max(sharpe.values()) - 1e-6
    curve = frontier_curve(m, c, cap, n_points=8)
    assert curve["Return (%/yr)"].is_monotonic_increasing
    assert curve["Volatility (%/yr)"].iloc[0] == pytest.approx(np.sqrt(var["Lowest risk"]) * 100, rel=1e-3)
    summary, weights = compare_portfolios(ports, mu, cov, 0.04, 17000, pd.Series(-10.0, index=tickers),
                                          pd.Series(-12.0, index=tickers))
    assert summary["Scenario, normal ($)"].tolist() == pytest.approx([-1700.0] * 4)
    assert weights.sum().tolist() == pytest.approx([100.0] * 4)


def test_max_return_respects_cap():
    w = max_return(np.array([0.1, 0.2, 0.05]), 0.4)
    assert w.tolist() == pytest.approx([0.4, 0.4, 0.2])


def test_historical_expected_returns(setup):
    _, returns, models, values, *_ = setup
    mu = expected_returns(models, list(values.index), 0.04, RETURN_METHODS[1], returns, 252)
    assert mu["OILCO"] == pytest.approx(returns["OILCO"].mean() * 252)


def test_vix_shock_raises_put_value_in_scenario_only():
    base = HedgeSetup(spot=500, vix=18, rate=0.04, dollar_beta=50_000, hedge_ratio=1.0)
    spiked = HedgeSetup(spot=500, vix=18, rate=0.04, dollar_beta=50_000, hedge_ratio=1.0, vix_extra=10.0)
    assert spiked.premium(spiked.strike) == base.premium(base.strike)  # today's price is unchanged
    assert hedge_pnl("Index puts", spiked, -0.05, 0.0) > hedge_pnl("Index puts", base, -0.05, 0.0)
