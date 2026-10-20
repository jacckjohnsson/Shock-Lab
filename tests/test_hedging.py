import numpy as np
import pandas as pd
import pytest

from shock_lab.hedging import (
    CONTRACT_SIZE,
    HEDGES,
    HedgeSetup,
    compare_hedges,
    hedge_cost,
    hedge_pnl,
    payoff_table,
    put_greeks_table,
    vix_market_beta,
)
from shock_lab.options import bs_put, bs_put_greeks, put_implied_vol
from shock_lab.sample import make_sample_data


def test_bs_put_textbook_value():
    # Hull's classic example: S=K=100, T=1, r=5%, vol=20% -> put = 5.5735
    assert bs_put(100, 100, 1.0, 0.05, 0.20) == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity_with_dividends():
    from scipy.stats import norm

    S, K, T, r, q, v = 420.0, 400.0, 0.5, 0.04, 0.013, 0.22
    d1 = (np.log(S / K) + (r - q + v**2 / 2) * T) / (v * np.sqrt(T))
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - v * np.sqrt(T))
    assert call - bs_put(S, K, T, r, v, q) == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T))


def test_expired_put_is_intrinsic_and_vectorized():
    out = bs_put(np.array([90.0, 110.0]), 100, 0.0, 0.05, 0.2)
    np.testing.assert_allclose(out, [10.0, 0.0])


def test_greeks_match_finite_differences():
    S, K, T, r, v, q = 420.0, 399.0, 0.25, 0.04, 0.19, 0.013
    g = bs_put_greeks(S, K, T, r, v, q)
    h = 0.01
    assert g["delta"] == pytest.approx((bs_put(S + h, K, T, r, v, q) - bs_put(S - h, K, T, r, v, q)) / (2 * h), rel=1e-4)
    gamma_fd = (bs_put(S + 1, K, T, r, v, q) - 2 * bs_put(S, K, T, r, v, q) + bs_put(S - 1, K, T, r, v, q))
    assert g["gamma"] == pytest.approx(gamma_fd, rel=1e-3)
    assert g["vega"] == pytest.approx(bs_put(S, K, T, r, v + 0.01, q) - bs_put(S, K, T, r, v, q), rel=0.02)
    assert g["theta"] == pytest.approx(bs_put(S, K, T - 1 / 365, r, v, q) - bs_put(S, K, T, r, v, q), rel=0.02)
    assert -1 < g["delta"] < 0


def test_skew():
    assert put_implied_vol(0.18, 100, 100) == pytest.approx(0.18)
    assert put_implied_vol(0.18, 90, 100) == pytest.approx(0.24)  # +0.6 pts per 1% OTM
    assert put_implied_vol(0.18, 110, 100) == pytest.approx(0.18)  # no skew above spot


@pytest.fixture
def setup():
    # $100k portfolio with market beta 1 -> $1,000 per 1% move
    return HedgeSetup(spot=500.0, vix=18.0, rate=0.04, dollar_beta=100_000.0, hedge_ratio=1.0,
                      strike_pct=0.95, expiry_days=90, vix_beta=-120.0)


def test_contract_sizing(setup):
    assert setup.contracts() == (2, pytest.approx(2.0))
    small = HedgeSetup(spot=500.0, vix=18.0, rate=0.04, dollar_beta=10_000.0, hedge_ratio=0.5)
    n, exact = small.contracts()
    assert n == 1 and exact == pytest.approx(0.1)  # can't buy a tenth of a contract
    none = HedgeSetup(spot=500.0, vix=18.0, rate=0.04, dollar_beta=10_000.0, hedge_ratio=0.0)
    assert none.contracts()[0] == 0


def test_put_hedge_pnl(setup):
    cost, cash = hedge_cost("Index puts", setup)
    assert cost == cash > 0
    # No move, no time passing: the hedge is worth what you paid.
    assert hedge_pnl("Index puts", setup, 0.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    # A 20% crash: 2 contracts of $475 puts end up deep in the money.
    crash = hedge_pnl("Index puts", setup, -0.20, -20_000)
    assert crash > 2 * CONTRACT_SIZE * (475 - 400) - cost
    # If the market rallies, you lose at most the premium.
    assert -cost <= hedge_pnl("Index puts", setup, 0.20, 20_000) < 0


def test_put_spread_is_cheaper_and_capped(setup):
    put_cost, _ = hedge_cost("Index puts", setup)
    spread_cost, _ = hedge_cost("Put spread", setup)
    assert 0 < spread_cost < put_cost
    max_payout = 2 * CONTRACT_SIZE * (setup.strike - setup.spread_strike)
    assert hedge_pnl("Put spread", setup, -0.50, -50_000) <= max_payout - spread_cost + 1e-6


def test_time_decay_costs_money(setup):
    setup.horizon_days = 30
    assert hedge_pnl("Index puts", setup, 0.0, 0.0) < 0


def test_inverse_etf_and_raise_cash(setup):
    setup.hedge_ratio = 0.5
    assert hedge_pnl("Inverse ETF", setup, -0.10, -10_000) == pytest.approx(5_000)
    assert hedge_pnl("Raise cash", setup, -0.10, -10_000) == pytest.approx(5_000)
    assert hedge_cost("Inverse ETF", setup) == (0.0, 50_000.0)


def test_compare_and_payoff_tables(setup):
    table = compare_hedges(setup, -0.15, -15_000, 0.10, 10_000)
    assert list(table["Hedge"]) == HEDGES
    row = table.set_index("Hedge").loc["Raise cash"]
    assert row["Loss offset (%)"] == pytest.approx(100.0)
    np.testing.assert_allclose(table["Portfolio + hedge ($)"], -15_000 + table["Hedge gain/loss ($)"])
    grid = np.array([-0.2, 0.0, 0.2])
    pay = payoff_table(setup, lambda m: 100_000 * m, grid)
    assert len(pay) == len(HEDGES) * 3
    no_hedge = pay[pay["Hedge"] == "No hedge"]["Change ($)"].values
    np.testing.assert_allclose(no_hedge, [-20_000, 0, 20_000])
    greeks = put_greeks_table(setup)
    assert (greeks["Delta (SPY shares)"] < 0).all()


def test_vix_beta_from_sample_data():
    market, macro, _ = make_sample_data()
    beta = vix_market_beta(macro["vix"], market.pct_change())
    assert beta == pytest.approx(-120, rel=0.15)
