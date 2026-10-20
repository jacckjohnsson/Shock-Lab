# Shock Lab

**What happens to my portfolio if oil jumps 20%, rates rise 1 point, the dollar strengthens, or the S&P 500 falls 15%?**

Shock Lab lets you type in up to 30 stocks or ETFs with share counts, pick a scenario, and see the estimated
dollar and percent impact on each holding and on the whole portfolio. It uses the same basic tool risk teams
at banks and hedge funds use for this question: a multi-factor regression.

> For education only, not investment advice. Estimates come from a linear model fit to past data.
> Real markets can behave very differently, especially in a crisis.

## Run it on your computer

You need Python 3.11 or newer ([python.org/downloads](https://www.python.org/downloads/)).

```bash
# 1. Go into the project folder
cd shock-lab

# 2. Create and turn on a virtual environment (keeps this project's packages separate)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the packages
pip install -r requirements.txt

# 4. Start the app (it opens in your browser at http://localhost:8501)
streamlit run app.py

# 5. Run the tests
pytest
```

No API keys are needed. Prices come from Yahoo Finance (via `yfinance`) and macro data from FRED's public
CSV downloads. If you are offline, turn on **"Use made-up sample data"** in the sidebar.

## Using the app

**Add your holdings** either way:
- **Type them in:** best for a handful of positions.
- **Upload a CSV:** a positions export from your broker (Fidelity, Schwab, Vanguard, E*TRADE and others) or
  any file with a `Ticker`/`Symbol` column and a `Shares`/`Quantity` column. Shock Lab finds the header row,
  skips cash, money market funds, totals and notes, merges the same stock held in several accounts, and
  converts class shares like `BRK.B` to Yahoo's `BRK-B`. You can edit the result before running. A template
  is available in the app.

**Tabs:**
- **Scenario:** the sidebar scenario, normal vs stress estimates, per holding and in total.
- **Compare scenarios:** the four presets plus any scenarios you saved from the sidebar, side by side.
- **Historical replays:** predicted vs actual for 2008, 2020 and 2022.
- **Model details:** betas and how they changed over time.
- **Export:** the current scenario as CSV, every table as an Excel workbook, or a one-page PDF report.

## How it works

### 1. Data

| Factor | Source | How a daily move is measured |
| --- | --- | --- |
| Market | SPY (S&P 500 ETF), Yahoo Finance | % return |
| Oil | FRED `DCOILWTICO` (WTI crude, $/barrel) | % change |
| Rates | FRED `DGS10` (10-year Treasury yield) | change in percentage points |
| Dollar | FRED `DTWEXBGS` (broad trade-weighted dollar index) | % change |

Data is downloaded back to 2006 (so the 2008 crisis can be replayed). FRED series that skip a trading day are
carried forward.

**Model settings (sidebar):**
- **History used to fit betas:** 1, 2, 3, 5 or 10 years. Shorter windows react to how a stock behaves lately but
  are noisier.
- **Return frequency:** daily or weekly. Stock prices close at 4pm New York time, but FRED's macro series are
  recorded at slightly different times, which blurs daily betas. Weekly returns (Friday to Friday) reduce that. Oil prices went
negative in April 2020, so % changes around those days are dropped.

### 2. A factor regression for each stock

For each holding *i*, Shock Lab fits an ordinary least squares regression of its daily return on the four factors:

$$
r_{i,t} = \alpha_i + \beta_{i}^{mkt}\, f^{mkt}_t + \beta_{i}^{oil}\, f^{oil}_t + \beta_{i}^{rates}\, \Delta y_t + \beta_{i}^{usd}\, f^{usd}_t + \varepsilon_{i,t}
$$

Each **beta** says how much the stock has tended to move for a 1-unit move in that factor, *holding the others
fixed*. An oil beta of 0.30 means about +0.3% for every +1% in oil. The rates beta is per 1 percentage point
change in the 10-year yield. **R²** shows how much of the stock's daily movement the four factors explain.

### 3. Turning a shock into an impact

A scenario is a vector of factor moves **s** (for example oil = +0.20). The estimated return of holding *i* is

$$
\widehat{R}_i = \boldsymbol\beta_i^\top \mathbf{s}
$$

and the dollar impact is `shares × last price × R̂ᵢ`. The portfolio impact is the sum over holdings.

**Range.** The betas are estimates with standard errors. The Low/High columns are
$\widehat{R}_i \pm 1.96\sqrt{\mathbf{s}^\top \Sigma_{\beta_i} \mathbf{s}}$, where $\Sigma_{\beta_i}$ is the
covariance of the beta estimates. This only captures estimation error, not the stock's own random noise.

### 4. Normal times vs stress times

Correlations change in a crisis: in selloffs, stocks tend to move together more. Shock Lab fits a second set of
betas using **only the worst 20% of market periods since 2006** and shows the same scenario with those betas next
to the normal estimate. A big gap between the two means a holding behaves differently when markets are falling.
(Because it keeps only 1 in 5 periods, the stress model always uses the full history.)

### 5. Historical replays (is the model any good?)

The **Historical replays** tab tests the model on real events: the 2008 financial crisis, the COVID crash of
2020 and the 2022 rate shock. For each event it:

1. fits betas using **only data from before the event started** (no peeking at the answer),
2. measures what the four factors actually did during the event,
3. predicts each holding's return as β·(actual factor moves), and
4. compares that with the holding's actual return.

The scatter plot shows predicted vs actual; dots on the diagonal were predicted perfectly. "Avg. miss per holding"
is the mean absolute error in percentage points. Holdings that didn't trade yet, or lacked enough history before
the event, are listed separately.

### 6. Rolling betas

The **Model details** tab charts each beta re-estimated over a moving 1-year window, so you can see how a stock's
sensitivity to oil or rates has changed over time, and compare it with the beta currently in use.

### 7. Factors move together

Oil rarely jumps 20% while everything else stands still. With **"Include typical co-moves"** on, the factors
you didn't set are filled in with their expected value given the ones you did, using the historical factor
covariance Σ (the conditional mean of a multivariate normal):

$$
\mathbb{E}[\mathbf{f}_{other} \mid \mathbf{f}_{set} = \mathbf{x}] = \Sigma_{other,set}\, \Sigma_{set,set}^{-1}\, \mathbf{x}
$$

The **Scenario** table in the app shows both what you set and what the model used.

## Limitations (good things to improve next)

- **Linear and scaled up.** Betas are fit on small daily moves and applied to large shocks. Real reactions to a
  20% oil spike may not be proportional.
- **Betas still assumed constant within the window.** Exponentially weighted regressions would put more
  weight on recent data without a hard cutoff.
- **Stress definition is simple.** "Worst 20% of market periods" could be replaced with VIX levels or
  recession dates.
- **Replays use one beta for the whole event.** Real betas shift during a crisis, and the four factors miss
  things like credit spreads and company news.
- **Data licensing.** `yfinance` is fine for a personal project, but not licensed for a paid product.

## Project layout

```
app.py                  Streamlit app (inputs, scenario, results)
shock_lab/data.py       Downloads from Yahoo Finance and FRED
shock_lab/model.py      Factor returns, regressions, shock math
shock_lab/portfolio.py  Cleans the holdings you type in
shock_lab/shocks.py     Preset scenarios
shock_lab/replay.py     Historical replays (2008, 2020, 2022)
shock_lab/compare.py    Several scenarios side by side
shock_lab/importers.py  Reads holdings CSVs and broker exports
shock_lab/report.py     Excel and PDF exports
shock_lab/sample.py     Made-up data with known betas (offline demo and tests)
tests/                  pytest tests
```
