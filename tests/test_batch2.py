import io

import numpy as np
import pandas as pd
import pytest

from shock_lab.compare import compare_scenarios
from shock_lab.importers import TEMPLATE_CSV, parse_holdings_csv
from shock_lab.model import build_factor_returns, fit_all, stock_returns, stress_periods
from shock_lab.portfolio import clean_holdings
from shock_lab.report import to_excel, to_pdf
from shock_lab.sample import SAMPLE_BETAS, make_sample_data
from shock_lab.shocks import PRESETS

FIDELITY = """Account Number,Account Name,Symbol,Description,Quantity,Last Price,Current Value
Z123,Individual,AAPL,APPLE INC,20,$230.10,"$4,602.00"
Z123,Individual,SPAXX**,HELD IN MONEY MARKET,,,"$1,250.00"
Z123,Individual,BRK.B,BERKSHIRE HATHAWAY INC CL B,"1,000",$450.00,"$450,000.00"
Z123,Individual,Pending Activity,,,,"-$50.00"
Z456,Roth IRA,AAPL,APPLE INC,5,$230.10,"$1,150.50"

"The data and information in this spreadsheet is provided to you solely for your use..."
"Brokerage services are provided by Fidelity Brokerage Services LLC"
"""

SCHWAB = '''"Positions for account Individual ...123 as of 10:15 AM ET, 2026/10/20"

"Symbol","Description","Qty (Quantity)","Price","Mkt Val (Market Value)"
"MSFT","MICROSOFT CORP","12","$500.00","$6,000.00"
"VOO","VANGUARD S&P 500 ETF","3.5","$550.00","$1,925.00"
"Cash & Cash Investments","--","--","--","$300.00"
"Account Total","--","--","--","$8,225.00"
'''


def test_template_parses():
    df, notes = parse_holdings_csv(TEMPLATE_CSV)
    assert df.to_dict("list") == {"Ticker": ["AAPL", "XOM", "JPM"], "Shares": [20.0, 30.0, 15.0]}
    assert notes == []


def test_fidelity_export():
    df, notes = parse_holdings_csv(FIDELITY.encode("utf-8-sig"))
    assert df.to_dict("list") == {"Ticker": ["AAPL", "BRK.B", "AAPL"], "Shares": [20.0, 1000.0, 5.0]}
    assert any("SPAXX" in n for n in notes)
    assert any("Pending Activity" in n for n in notes)
    # Cleaning merges the two AAPL lots and converts BRK.B for Yahoo Finance.
    clean, _ = clean_holdings(df)
    assert clean.to_dict("list") == {"Ticker": ["AAPL", "BRK-B"], "Shares": [25.0, 1000.0]}


def test_schwab_export_with_title_lines():
    df, notes = parse_holdings_csv(SCHWAB)
    assert df.to_dict("list") == {"Ticker": ["MSFT", "VOO"], "Shares": [12.0, 3.5]}
    assert any("Account Total" in n for n in notes)


def test_lowercase_and_shares_header():
    df, _ = parse_holdings_csv("ticker,shares\nnvda,4\n")
    assert df.to_dict("list") == {"Ticker": ["NVDA"], "Shares": [4.0]}


def test_missing_columns_raises():
    with pytest.raises(ValueError, match="symbol column"):
        parse_holdings_csv("Name,Amount\nApple,5\n")


@pytest.fixture(scope="module")
def fitted():
    market, macro, prices = make_sample_data()
    factors = build_factor_returns(market, macro)
    returns = stock_returns(prices)
    models, _ = fit_all(returns, factors)
    idx = stress_periods(factors)
    stress_models, _ = fit_all(returns.loc[returns.index.isin(idx)], factors.loc[idx])
    holdings = pd.DataFrame({"Ticker": list(SAMPLE_BETAS), "Shares": [50, 20, 40, 60]})
    return models, stress_models, holdings, prices.iloc[-1], factors


def test_compare_scenarios(fitted):
    models, stress_models, holdings, last, factors = fitted
    scenarios = {**PRESETS, "Mine": {"oil": 0.3, "rates": 0.5}}
    summary, by_holding = compare_scenarios(scenarios, models, stress_models, holdings, last, factors, co_move=False)
    assert list(summary["Scenario"]) == list(scenarios)
    assert list(by_holding.columns) == list(scenarios)
    # Totals match the per-holding columns.
    np.testing.assert_allclose(summary["Normal ($)"].values, by_holding.sum().values)
    # A market drop hurts every sample stock (all have positive market betas).
    assert (by_holding["S&P 500 -15%"] < 0).all()
    assert summary.set_index("Scenario").loc["S&P 500 -15%", "Hit hardest"] in SAMPLE_BETAS


def test_excel_round_trip():
    df = pd.DataFrame({"Ticker": ["AAPL"], "Impact ($)": [-12.5]})
    data = to_excel({"Scenario": df, "A very long sheet name that Excel would reject": df})
    back = pd.read_excel(io.BytesIO(data), sheet_name=None)
    assert list(back) == ["Scenario", "A very long sheet name that Exc"]
    assert back["Scenario"].iloc[0]["Impact ($)"] == -12.5


def test_pdf_is_valid():
    df = pd.DataFrame({"Ticker": ["AAPL", "XOM"], "Value ($)": [1000.0, 2000.0], "Impact ($)": [-50.0, np.nan]})
    pdf = to_pdf("Shock Lab report", [("Holdings", "2")], [("Change", "-$50")], [("Scenario", df)], "**Not advice.**")
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000
