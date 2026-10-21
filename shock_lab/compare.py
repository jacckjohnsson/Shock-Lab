"""Run several scenarios on the same portfolio and line them up."""

from __future__ import annotations

import pandas as pd

from .model import FactorModel, holding_impacts, implied_factor_moves, portfolio_summary


def compare_scenarios(
    scenarios: dict[str, dict[str, float]],
    models: dict[str, FactorModel],
    stress_models: dict[str, FactorModel],
    holdings: pd.DataFrame,
    last_prices: pd.Series,
    factors: pd.DataFrame,
    co_move: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary with one row per scenario, $ impact per holding with one column per scenario)."""
    summary_rows, by_holding = [], {}
    for name, shock in scenarios.items():
        moves = implied_factor_moves(shock, factors, co_move)
        impacts = holding_impacts(models, holdings, last_prices, moves, stress_models)
        normal = portfolio_summary(impacts)
        stress = portfolio_summary(impacts.dropna(subset=["Stress impact ($)"]), "Stress impact ($)")
        worst = impacts.loc[impacts["Impact ($)"].idxmin()] if len(impacts) else None
        summary_rows.append(
            {
                "Scenario": name,
                "Normal ($)": normal["impact"],
                "Normal (%)": normal["impact_pct"],
                "Stress ($)": stress["impact"],
                "Stress (%)": stress["impact_pct"],
                "Hit hardest": worst["Ticker"] if worst is not None and worst["Impact ($)"] < 0 else "None",
            }
        )
        by_holding[name] = impacts.set_index("Ticker")["Impact ($)"]
    return pd.DataFrame(summary_rows), pd.DataFrame(by_holding)
