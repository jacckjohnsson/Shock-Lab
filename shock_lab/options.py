"""Black-Scholes pricing and greeks for European puts, plus a simple volatility skew."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Out-of-the-money index puts trade at higher implied volatility than at-the-money ones
# ("skew"). Rule of thumb: about +0.6 vol points for each 1% the strike is below spot.
SKEW_VOL_PER_PCT_OTM = 0.006


def put_implied_vol(atm_vol, strike, spot, skew: float = SKEW_VOL_PER_PCT_OTM):
    """ATM vol plus skew for puts below spot. Vols are decimals (0.18 = 18%)."""
    otm_pct = np.maximum(0.0, (1 - np.asarray(strike) / np.asarray(spot)) * 100)
    return np.asarray(atm_vol) + skew * otm_pct


def _d1_d2(S, K, T, r, sigma, q):
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def bs_put(S, K, T, r, sigma, q: float = 0.0):
    """Price of a European put per share. T in years; r, q, sigma as decimals."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    expired = T <= 0
    T_safe = np.where(expired, 1.0, T)
    d1, d2 = _d1_d2(S, K, T_safe, r, sigma, q)
    price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * np.exp(-q * T_safe) * norm.cdf(-d1)
    out = np.where(expired, np.maximum(K - S, 0.0), price)
    return out.item() if out.ndim == 0 else out


def bs_put_greeks(S, K, T, r, sigma, q: float = 0.0) -> dict[str, float]:
    """Delta (per share), gamma (delta change per $1), vega ($ per 1 vol point), theta ($ per calendar day)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q, disc_r, sqrt_t = np.exp(-q * T), np.exp(-r * T), np.sqrt(T)
    pdf = norm.pdf(d1)
    theta_year = -S * disc_q * pdf * sigma / (2 * sqrt_t) + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    return {
        "delta": float(-disc_q * norm.cdf(-d1)),
        "gamma": float(disc_q * pdf / (S * sigma * sqrt_t)),
        "vega": float(S * disc_q * pdf * sqrt_t / 100),
        "theta": float(theta_year / 365),
    }
