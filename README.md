# Shock Lab

**What happens to my portfolio if oil jumps 20%, rates rise 1 point, the dollar strengthens, gold rallies, the VIX spikes, credit spreads widen, or the S&P 500 falls 15%?**

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

**Then follow the steps** (everything stays on one page, and every detail stays visible):
1. **Holdings:** type them in or upload a CSV.
2. **Scenario:** a preset or your own moves in the S&P 500, oil, the 10-year yield, the dollar, gold, the VIX
   and credit spreads. Click **Run analysis**.
3. **Impact:** a plain-English summary card, normal vs stress estimates per holding, and a **concentration**
   check (share of value vs share of risk, with warnings).
4. **Hedge:** how index puts, a put spread, an inverse ETF or raising cash would change the result, with greeks.
5. **Simulate:** Monte Carlo over 1 month to 1 year: typical, bad (1 in 20) and very bad (1 in 100) outcomes,
   with or without a hedge.
6. **Optimize:** a light efficient frontier built from the stocks you own, and how each model mix fares in
   your scenario.
7. **Compare:** the presets plus any scenarios you saved, side by side.
8. **Check the model:** betas, how they changed over time, and replays of 2008, 2020 and 2022.
9. **Export:** CSV, an Excel workbook with every table, or a branded PDF report.

## How it works

### 1. Data

| Factor | Source | How a daily move is measured |
| --- | --- | --- |
| Market | SPY (S&P 500 ETF), Yahoo Finance | % return |
| Oil | FRED `DCOILWTICO` (WTI crude, $/barrel) | % change |
| Rates | FRED `DGS10` (10-year Treasury yield) | change in percentage points |
| Dollar | FRED `DTWEXBGS` (broad trade-weighted dollar index) | % change |
| Gold | GLD (gold ETF), Yahoo Finance | % return |
| VIX *(stress gauge)* | FRED `VIXCLS` | change in VIX points |
| Credit spread *(stress gauge)* | FRED `BAA10Y` (Baa corporate yield minus 10-year Treasury) | change in percentage points |

The VIX and the 3-month T-bill (`DTB3`) are also used to price options in the hedging step.

Data is downloaded back to 2006 (so the 2008 crisis can be replayed). FRED series that skip a trading day are
carried forward.

**Model settings (sidebar):**
- **History used to fit betas:** 1, 2, 3, 5 or 10 years. Shorter windows react to how a stock behaves lately but
  are noisier.
- **Return frequency:** daily or weekly. Stock prices close at 4pm New York time, but FRED's macro series are
  recorded at slightly different times, which blurs daily betas. Weekly returns (Friday to Friday) reduce that. Oil prices went
negative in April 2020, so % changes around those days are dropped.

### 2. A factor regression for each stock

For each holding *i*, Shock Lab fits an ordinary least squares regression of its daily return on the five factors:

$$
r_{i,t} = \alpha_i + \beta_{i}^{mkt}\, f^{mkt}_t + \beta_{i}^{oil}\, f^{oil}_t + \beta_{i}^{rates}\, \Delta y_t + \beta_{i}^{usd}\, f^{usd}_t + \beta_{i}^{gold}\, f^{gold}_t + \varepsilon_{i,t}
$$

Each **beta** says how much the stock has tended to move for a 1-unit move in that factor, *holding the others
fixed*. An oil beta of 0.30 means about +0.3% for every +1% in oil. The rates beta is per 1 percentage point
change in the 10-year yield. **R²** shows how much of the stock's daily movement the five factors explain.

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
2. measures what the factors actually did during the event,
3. predicts each holding's return as β·(actual factor moves), and
4. compares that with the holding's actual return.

The scatter plot shows predicted vs actual; dots on the diagonal were predicted perfectly. "Avg. miss per holding"
is the mean absolute error in percentage points. Holdings that didn't trade yet, or lacked enough history before
the event, are listed separately.

### 6. Rolling betas

The **Model details** tab charts each beta re-estimated over a moving 1-year window, so you can see how a stock's
sensitivity to oil or rates has changed over time, and compare it with the beta currently in use.

### 7. Hedging

The **Hedging** tab shows how common hedges would behave in the current scenario. It is a model of how
hedges work, not a recommendation.

**Sizing.** Every hedge is sized from the portfolio's *market exposure*: the dollars it tends to move per 1% move
in the S&P 500 (its beta-weighted, or "dollar", beta). With co-moves on, this includes the oil, rate and dollar
moves that usually come with a market move. The hedge ratio slider picks how much of that exposure to cover.

| Hedge | How it's modeled |
| --- | --- |
| Index puts | Buy SPY puts. Contracts = ratio × exposure ÷ (100 × SPY price), rounded to whole contracts |
| Put spread | Buy a put and sell one 10% lower: cheaper, but protection stops at the lower strike |
| Inverse ETF | Hold a -1x S&P 500 ETF (like SH) worth ratio × exposure; gains 1% for each 1% the S&P falls |
| Raise cash | Sell the same share of every holding, which cuts every exposure, not just the market |

**Option pricing.** Puts are priced with Black-Scholes (with SPY's ~1.3% dividend yield):

$$
P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1), \quad
d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

- **Volatility σ** comes from the VIX (FRED `VIXCLS`), plus a simple *skew*: +0.6 vol points for each 1% the
  strike is below the current price, since lower-strike puts trade at higher volatility.
- **Volatility in a crash.** Shock Lab regresses daily VIX changes on S&P 500 returns (typically about +1.2 VIX
  points per -1% day) and moves the VIX with the scenario's market move. This is why puts gain more in a crash
  than the price move alone suggests.
- **Rate r** is the 3-month T-bill rate (FRED `DTB3`).
- **Timing.** "Shock happens instantly / in 1 week / in 1 month" lets time decay (theta) eat into the puts first.

The greeks (delta, gamma, vega, theta) for the put positions are shown in an expander.

**Small portfolios.** One SPY option covers 100 shares, roughly $60,000 of the S&P 500. The app warns when whole
contracts badly over- or under-hedge a small portfolio; the inverse ETF and raising cash can be sized exactly.

### 8. Factors move together

Oil rarely jumps 20% while everything else stands still. With **"Include typical co-moves"** on, the factors
you didn't set are filled in with their expected value given the ones you did, using the historical factor
covariance Σ (the conditional mean of a multivariate normal):

$$
\mathbb{E}[\mathbf{f}_{other} \mid \mathbf{f}_{set} = \mathbf{x}] = \Sigma_{other,set}\, \Sigma_{set,set}^{-1}\, \mathbf{x}
$$

The **Factor moves** table in the app shows both what you set and what the model used.

### 9. VIX and credit spreads: stress gauges, not betas

The VIX and credit spreads move almost in lockstep with the stock market, so adding them as extra regression
factors would make every stock's betas unstable (the regression can't tell them apart from the market). Instead
they are **scenario drivers**: a VIX or credit shock is turned into the S&P, oil, rates, dollar and gold moves
that have typically come with it (the same conditional-mean formula as above), and stocks react through their
usual betas. This happens even with co-moves turned off, since a gauge has no beta of its own. A VIX shock you
set also raises put prices in the hedging step beyond what the market move alone would.

### 10. Concentration

The model gives a covariance matrix for your holdings without needing stock-by-stock history:

$$
\Sigma = B\, \Sigma_f\, B^\top + D
$$

where *B* holds the betas, $\Sigma_f$ is the factor covariance and *D* holds each stock's own residual
variance. Each holding's **share of risk** is $w_i (\Sigma w)_i / w^\top \Sigma w$ (these add to 100%). The app
warns when one holding is over 25% of value or 35% of risk, or when the portfolio "acts like" fewer than 4 equal
positions ($1/\sum w_i^2$).

### 11. Monte Carlo

4,000 possible futures are built by stitching together random **week-long blocks of real factor history** (a block
bootstrap), either since 2006 (includes 2008 and 2020) or only the fit window. Using real blocks keeps fat tails,
volatility clustering and how the factors move together. Each stock moves by its betas plus normal noise sized to
its residual volatility, and positions are held without rebalancing. Average factor returns are removed first,
so the result shows **risk, not expected return**. Reported: the typical (median) outcome, value at risk at 95%
and 99%, expected shortfall (average of the worst 5%), and the chance of losing anything or 10%+. A hedge from
step 4 can be overlaid, valued at the horizon with the same option model.

### 12. Efficient frontier (light)

Classic mean-variance optimization over the stocks you already own, long-only, with a cap on any one stock.
Risk comes from the factor-model covariance above. Expected returns are either **market-implied** (CAPM:
T-bill rate + market beta × a 5% equity premium, which is steady) or each stock's **historical average** (very
noisy, included to show how sensitive the optimizer is). The app shows the frontier, your portfolio, the lowest-risk
mix, the mix with your expected return but less risk, and the highest Sharpe ratio mix, plus how each would fare
in your scenario. These are textbook illustrations, not recommendations.

## Limitations (good things to improve next)

- **Linear and scaled up.** Betas are fit on small daily moves and applied to large shocks. Real reactions to a
  20% oil spike may not be proportional.
- **Betas still assumed constant within the window.** Exponentially weighted regressions would put more
  weight on recent data without a hard cutoff.
- **Stress definition is simple.** "Worst 20% of market periods" could be replaced with VIX levels or
  recession dates.
- **Replays use one beta for the whole event.** Real betas shift during a crisis, and the factors miss
  company news.
- **Monte Carlo is only as wide as history.** Bootstrapping can't produce a crash worse than any week since 2006,
  and stock noise is assumed normal.
- **Mean-variance needs expected returns no one knows.** The market-implied default is a reasonable anchor, not a
  forecast. Shrinkage or Black-Litterman views would be natural next steps.
- **Hedging uses model prices.** Real option quotes, bid/ask spreads and the shape of volatility across
  expiries will differ. Inverse ETFs reset daily, so over long periods they don't exactly track -1x.
- **Data licensing.** `yfinance` is fine for a personal project, but not licensed for a paid product.

## Project layout

```
app.py                  Streamlit app (the 9-step guided flow)
shock_lab/data.py       Downloads from Yahoo Finance and FRED
shock_lab/model.py      Factor returns, regressions, shock math
shock_lab/portfolio.py  Cleans the holdings you type in
shock_lab/shocks.py     Preset scenarios
shock_lab/replay.py     Historical replays (2008, 2020, 2022)
shock_lab/compare.py    Several scenarios side by side
shock_lab/importers.py  Reads holdings CSVs and broker exports
shock_lab/report.py     Excel and PDF exports
shock_lab/options.py    Black-Scholes prices, greeks and skew
shock_lab/hedging.py    Hedge sizing, costs and payoffs
shock_lab/risk.py       Factor-model covariance, risk contributions, concentration warnings
shock_lab/montecarlo.py Block-bootstrap Monte Carlo and risk stats (VaR, expected shortfall)
shock_lab/frontier.py   Light mean-variance optimizer and efficient frontier
shock_lab/brand.py      Colors, logo, chart styling and page fragments
shock_lab/landing.py    The welcome screen shown before the app
shock_lab/sample.py     Made-up data with known betas (offline demo and tests)
tests/                  pytest tests
static/                 Screenshots for the welcome screen (served by Streamlit)
```
