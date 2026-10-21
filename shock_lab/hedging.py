"""How different hedges would behave against a market shock.

Everything here is sized from the portfolio's sensitivity to the S&P 500 ("dollar beta":
dollars gained or lost per 100% move in the index) and hedged with SPY, so a hedge only
offsets the part of a scenario that moves the stock market.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .options import bs_put, bs_put_greeks, put_implied_vol

CONTRACT_SIZE = 100  # one listed option covers 100 shares
DIVIDEND_YIELD = 0.013  # SPY, approximate
INVERSE_ETF_FEE = 0.0089  # SH expense ratio per year
SPREAD_WIDTH = 0.10  # put spread sells a put 10% below the bought strike
VIX_FLOOR, VIX_CAP = 9.0, 80.0

HEDGES = ["No hedge", "Index puts", "Put spread", "Inverse ETF", "Raise cash"]


def vix_market_beta(vix: pd.Series, market_returns: pd.Series) -> float:
    """VIX points per 1.0 (100%) market return, from daily changes. Usually about -100 to -150."""
    data = pd.concat([vix.diff().rename("dvix"), market_returns.rename("m")], axis=1, join="inner").dropna()
    if len(data) < 60:
        return -120.0
    return float(np.polyfit(data["m"], data["dvix"], 1)[0])


def shocked_vix(vix_now: float, market_move, beta: float):
    return np.clip(vix_now + beta * np.asarray(market_move), VIX_FLOOR, VIX_CAP)


@dataclass
class HedgeSetup:
    spot: float  # SPY price
    vix: float  # VIX level, vol points (e.g. 18)
    rate: float  # risk-free rate, decimal
    dollar_beta: float  # $ portfolio change per 1.0 market move
    hedge_ratio: float = 0.5  # share of the market exposure to hedge
    strike_pct: float = 0.95  # put strike as % of spot
    expiry_days: int = 90
    horizon_days: int = 0  # how long until the shock (0 = instantly)
    vix_beta: float = -120.0
    vix_extra: float = 0.0  # extra VIX points in the scenario beyond what the market move implies (a VIX shock)

    @property
    def strike(self) -> float:
        return self.strike_pct * self.spot

    @property
    def spread_strike(self) -> float:
        return (self.strike_pct - SPREAD_WIDTH) * self.spot

    def contracts(self) -> tuple[int, float]:
        """(whole contracts used, exact number needed). Never zero when there is exposure to hedge."""
        exact = max(self.hedge_ratio * self.dollar_beta, 0.0) / (CONTRACT_SIZE * self.spot)
        n = int(round(exact))
        if n == 0 and exact > 0:
            n = 1
        return n, exact

    def put_value(self, strike: float, market_move=0.0, days_passed: int = 0, scenario: bool = False):
        """Value per share of a put after the market moves and some days pass.

        scenario=True also applies vix_extra, the part of a VIX shock the market move doesn't explain.
        """
        S = self.spot * (1 + np.asarray(market_move))
        vix = self.vix + (self.vix_extra if scenario else 0.0)
        atm = shocked_vix(vix, market_move, self.vix_beta) / 100
        T = max(self.expiry_days - days_passed, 0) / 365
        return bs_put(S, strike, T, self.rate, put_implied_vol(atm, strike, S), DIVIDEND_YIELD)

    def premium(self, strike: float) -> float:
        return float(self.put_value(strike))


def hedge_cost(name: str, h: HedgeSetup) -> tuple[float, float]:
    """(upfront cost you never get back unless the hedge pays, cash tied up)."""
    n, _ = h.contracts()
    if name == "Index puts":
        cost = n * CONTRACT_SIZE * h.premium(h.strike)
        return cost, cost
    if name == "Put spread":
        cost = n * CONTRACT_SIZE * (h.premium(h.strike) - h.premium(h.spread_strike))
        return cost, cost
    if name == "Inverse ETF":
        return 0.0, h.hedge_ratio * h.dollar_beta
    return 0.0, 0.0


def hedge_pnl(name: str, h: HedgeSetup, market_move, portfolio_impact) -> np.ndarray:
    """Gain or loss on the hedge itself (premium included) for a market move.

    portfolio_impact: the unhedged portfolio change for the same move (needed for "Raise cash").
    """
    m = np.asarray(market_move, dtype=float)
    n, _ = h.contracts()
    t = h.horizon_days
    if name == "Index puts":
        return n * CONTRACT_SIZE * (h.put_value(h.strike, m, t, scenario=True) - h.premium(h.strike))
    if name == "Put spread":
        now = h.premium(h.strike) - h.premium(h.spread_strike)
        later = h.put_value(h.strike, m, t, scenario=True) - h.put_value(h.spread_strike, m, t, scenario=True)
        return n * CONTRACT_SIZE * (later - now)
    if name == "Inverse ETF":
        notional = h.hedge_ratio * h.dollar_beta
        return -m * notional - INVERSE_ETF_FEE * notional * t / 365
    if name == "Raise cash":
        return -h.hedge_ratio * np.asarray(portfolio_impact, dtype=float)
    return np.zeros_like(m)


def describe(name: str, h: HedgeSetup) -> str:
    n, _ = h.contracts()
    months = round(h.expiry_days / 30)
    if name == "Index puts":
        return f"Buy {n} SPY put{'s' if n != 1 else ''}, strike ${h.strike:,.0f}, ~{months} month(s)"
    if name == "Put spread":
        return f"Buy {n} ${h.strike:,.0f} put(s), sell {n} ${h.spread_strike:,.0f} put(s), ~{months} month(s)"
    if name == "Inverse ETF":
        return f"Hold ${h.hedge_ratio * h.dollar_beta:,.0f} of a -1x S&P 500 ETF (e.g. SH)"
    if name == "Raise cash":
        return f"Sell {h.hedge_ratio:.0%} of every holding"
    return "Keep the portfolio as is"


def compare_hedges(h: HedgeSetup, market_move: float, portfolio_impact: float, up_move: float = 0.10,
                   up_impact: float | None = None) -> pd.DataFrame:
    """One row per hedge: cost, what it does in the scenario, and what it costs if the market rises instead."""
    rows = []
    for name in HEDGES:
        cost, cash = hedge_cost(name, h)
        pnl = float(hedge_pnl(name, h, market_move, portfolio_impact))
        up = float(hedge_pnl(name, h, up_move, up_impact if up_impact is not None else 0.0))
        after = portfolio_impact + pnl
        rows.append(
            {
                "Hedge": name,
                "What it means": describe(name, h),
                "Upfront cost ($)": cost,
                "Cash needed ($)": cash,
                "Hedge gain/loss ($)": pnl,
                "Portfolio + hedge ($)": after,
                "Loss offset (%)": (pnl / -portfolio_impact * 100) if portfolio_impact < 0 else np.nan,
                f"Hedge if S&P +{up_move:.0%} ($)": up,
            }
        )
    return pd.DataFrame(rows)


def payoff_table(h: HedgeSetup, portfolio_impact_fn: Callable[[float], float], moves: np.ndarray) -> pd.DataFrame:
    """Portfolio + hedge value change across a range of S&P 500 moves (long format for charting)."""
    base = np.array([portfolio_impact_fn(m) for m in moves])
    frames = []
    for name in HEDGES:
        total = base + hedge_pnl(name, h, moves, base)
        frames.append(pd.DataFrame({"S&P 500 move (%)": moves * 100, "Change ($)": total, "Hedge": name}))
    return pd.concat(frames, ignore_index=True)


def put_greeks_table(h: HedgeSetup) -> pd.DataFrame:
    """Position greeks today for the put and put-spread hedges."""
    n, _ = h.contracts()
    T = h.expiry_days / 365
    shares = n * CONTRACT_SIZE

    def leg(strike: float) -> dict[str, float]:
        iv = float(put_implied_vol(h.vix / 100, strike, h.spot))
        g = bs_put_greeks(h.spot, strike, T, h.rate, iv, DIVIDEND_YIELD)
        return {"iv": iv, **g}

    long_leg, short_leg = leg(h.strike), leg(h.spread_strike)
    rows = []
    for name, legs in [("Index puts", [(long_leg, 1)]), ("Put spread", [(long_leg, 1), (short_leg, -1)])]:
        g = {k: sum(sign * lg[k] for lg, sign in legs) for k in ("delta", "gamma", "vega", "theta")}
        rows.append(
            {
                "Hedge": name,
                "Implied vol used (%)": long_leg["iv"] * 100,
                "Delta (SPY shares)": g["delta"] * shares,
                "Dollar delta ($ per 1% S&P move)": g["delta"] * shares * h.spot * 0.01,
                "Gamma (delta change per $1 SPY)": g["gamma"] * shares,
                "Vega ($ per 1 vol point)": g["vega"] * shares,
                "Theta ($ per day)": g["theta"] * shares,
            }
        )
    return pd.DataFrame(rows)
