"""Preset scenarios. Values are moves: % as decimals; rates, VIX and credit spreads in points."""

PRESETS = {
    "Oil +20%": {"oil": 0.20},
    "Rates +1% (10-yr yield +1 point)": {"rates": 1.0},
    "Dollar +5%": {"dollar": 0.05},
    "S&P 500 -15%": {"market": -0.15},
    "Gold +10%": {"gold": 0.10},
    "VIX +15 points (panic)": {"vix": 15.0},
    "Credit spreads +1 point": {"credit": 1.0},
}
