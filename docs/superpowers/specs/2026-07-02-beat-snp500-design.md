# beat-snp500 — Design Spec

**Date:** 2026-07-02
**Status:** Approved by Roman (strategy + architecture sections)

## 1. Goal

A portfolio project whose centerpiece is a **rigorous, bias-controlled quant strategy** that
picks the 10 most promising S&P 500 stocks, with a daily-updating public web dashboard.
Primary audience: recruiters and hiring managers evaluating quant/DS skill. The MLOps
pipeline (scheduling, validation, versioning, monitoring) supports the strategy but is not
the star.

The project supersedes the exploratory notebook
`Quantitative_investment_strategy (1).ipynb`, which is kept in `notebooks/` as provenance.
Its known defects — survivorship bias, three lookahead leaks, unadjusted prices, unstable
cluster-label selection, log/simple return mixing, understated costs — are each explicitly
fixed here.

## 2. Decisions made (with user)

| Decision | Choice |
|---|---|
| Primary showcase | Quant strategy performance; MLOps secondary |
| Models | Champion/challenger: LightGBM ranking model vs repaired K-Means strategy |
| Cadence | Daily re-scoring shown on UI; tracked portfolios rebalance monthly |
| Data | Free (yfinance + Wikipedia point-in-time membership), residual bias documented |
| UI & hosting | Streamlit Community Cloud + GitHub Actions cron; $0 total |

## 3. Strategy & backtest methodology

### 3.1 Universe (point-in-time)

- Reconstruct historical S&P 500 membership from Wikipedia's "Selected changes" table on
  the List of S&P 500 companies page; produce a `(date, ticker)` membership table.
- At each month-end `t`, the eligible universe = index members at `t` with price data
  available from Yahoo. Stocks delisted with no Yahoo history are unavoidably missing;
  this **residual survivorship bias is quantified where possible** (count of missing
  members per month) **and documented** on the dashboard's limitations page.
- Liquidity filter: top 150 by rolling 12-month average dollar volume, computed only from
  data ≤ `t`.

### 3.2 Data

- Daily OHLCV from yfinance with `auto_adjust=True` (split- and dividend-adjusted) —
  one consistent price basis for features AND returns.
- Local Parquet cache under `data/`; daily job fetches only the incremental tail.
  Daily updates are incremental with basis-shift detection on a 5-day overlap; a
  monthly full refresh re-downloads history so dividend/split adjustments stay consistent.
- Fama-French 5 factors (monthly) via `pandas_datareader` from the Ken French library.
- Price history from ~2008 onward (subject to membership-table coverage). Warm-up eats
  ~36 months (12m momentum + 24m betas) and the champion's rolling training window
  another 36, so this yields **≥ 8 years of out-of-sample walk-forward months** — the
  reported backtest period.

### 3.3 Features (monthly, strictly data ≤ t)

- Momentum: 1/3/6/12-month total returns from adjusted close, **winsorized
  cross-sectionally within each month** at 0.5%/99.5% (no full-history quantiles → no
  lookahead).
- Volatility: Garman-Klass (20d, annualized).
- Technicals: RSI(14), ATR(14) **normalized by close** (cross-sectionally comparable),
  Bollinger band width (20, 2), MACD histogram. Explicit column names owned by our code —
  no dependence on pandas_ta's generated names.
- Factor exposures: rolling 24-month FF5 betas via RollingOLS, **lagged one month**;
  rows with missing betas are dropped, never mean-imputed (mean-imputation leaked future
  data in the notebook). Exception: because Ken French publishes factors with a 1–2 month
  lag, a stock's last known betas may be forward-filled up to 3 months (backward-looking
  carry-forward — no leakage) so current-month scoring stays possible.

### 3.4 Label (champion training only)

- Forward 1-month total return from adjusted close, `t → t+1`. Only exists in training
  folds; never used at inference.

### 3.5 Champion — LightGBM ranking model

- Predicts next-month cross-sectional return rank (LGBMRegressor on rank-transformed
  label; simple, robust choice — LambdaRank is out of scope).
- **Walk-forward protocol:** rolling 36-month training window, predict month `t+1`, step
  one month, retrain. No hyperparameter search on future data; a small fixed grid tuned
  only on the first training window, then frozen.
- Signal evaluation: Spearman rank IC per month (mean, IR of IC), top-minus-bottom decile
  spread, top-10 hit rate vs universe median.

### 3.6 Challenger — repaired K-Means strategy

- Monthly K-Means (k=4) on standardized features, as in the notebook, **but** the
  momentum cluster is identified each month by centroid characteristics (highest
  composite momentum score = mean of standardized 3/6/12m returns), never by label index.
- Pick = top 10 stocks in that cluster by composite momentum.

### 3.7 Portfolio construction & backtest

- Both models emit exactly 10 tickers at each month-end.
- Weighting: equal weight (primary, most defensible) and max-Sharpe (PyPortfolioOpt,
  bounds 5–20%) as a variant.
- Timing: signal computed at month-end close of `t`; execution assumed at that close;
  returns accrue from the next trading day. Assumption stated on the methodology page.
- Costs: at each rebalance, cost = 10 bps × Σ|Δweightᵢ| (every unit of weight traded
  pays the one-way cost once; full portfolio replacement ≈ 20 bps).
- Returns math: **simple returns everywhere** for portfolio aggregation; compounding via
  cumulative product.
- Benchmarks: SPY total return; 1,000-draw bootstrap of random 10-stock equal-weight
  portfolios from the same point-in-time universe (the strategy must beat the bootstrap
  distribution, not just SPY).
- Metrics: CAGR, annualized vol, Sharpe (excess over the Fama-French risk-free rate,
  already in the factor download), max drawdown,
  Calmar, monthly turnover, and per-calendar-year sub-period table.
- Engine: small hand-rolled vectorized backtester in `src/backtest/`, unit-tested against
  hand-computed toy cases. No zipline/vectorbt dependency.

## 4. System architecture

### 4.1 Repo layout

```
beat-snp500/
├── src/beat_snp500/
│   ├── data/        # yfinance cache, membership table, FF factors, validation
│   ├── features/    # indicator + return feature computation
│   ├── models/      # champion (LightGBM), challenger (K-Means), registry I/O
│   ├── backtest/    # walk-forward engine, costs, metrics
│   └── portfolio/   # top-10 selection, weighting
├── app/             # Streamlit dashboard
├── data/            # Parquet cache + daily outputs (committed)
├── models/          # versioned model artifacts + registry.json
├── notebooks/       # original notebook (provenance)
├── tests/
├── scripts/         # entrypoints: run_daily.py, run_monthly.py, run_backtest.py
└── .github/workflows/  # daily.yml (cron), ci.yml (tests on PR)
```

### 4.2 Daily job (GitHub Actions cron, weekdays ~22:30 UTC)

1. Incremental price update → Parquet cache.
2. **Validation gate:** stale-price check, missing-ticker count, NaN-rate thresholds.
   Failure aborts the run loudly (failed workflow = notification); garbage is never
   published.
3. Re-score all eligible stocks with the current model artifacts → top-10 leaderboard per
   model (JSON) with per-stock feature snapshots.
4. Update live paper-track: daily returns of the currently-held monthly portfolios vs
   SPY since go-live.
5. Commit outputs to the repo (idempotent: same-day re-runs overwrite same-keyed files).

### 4.3 Monthly step (first weekday of month, same workflow)

- Walk-forward retrain of the champion (roll window forward one month); re-cluster for
  the challenger; emit new 10-stock portfolios; record turnover and applied costs;
  append model version + training window + validation IC to `models/registry.json`.

### 4.4 Streamlit app (Community Cloud, reads committed data)

1. **Today's Top 10** — leaderboard per model with scores and feature breakdown per pick.
2. **Live performance** — paper-tracked portfolios vs SPY since launch (this page is the
   forward test, the most honest evidence the project produces over time).
3. **Backtest report** — equity curves vs SPY and the random-portfolio bootstrap band, IC
   time series, drawdowns, metrics and per-year tables, champion vs challenger.
4. **Methodology & limitations** — data sources, execution/cost assumptions, residual
   survivorship bias quantification, "not investment advice."

### 4.5 Error handling

- yfinance calls: retry with exponential backoff; per-ticker failures tolerated up to the
  validation gate's threshold.
- Wikipedia membership scrape: schema-checked; on parse failure, fall back to last
  committed membership table and flag staleness in the UI.
- All daily outputs written atomically (temp file + rename) and keyed by date.

### 4.6 Testing & CI

- Unit tests: each feature against known-value fixtures; backtest engine against
  hand-computed toy portfolios; cost/turnover math.
- **Leakage test:** compute features on full history vs history truncated at `t`; values
  at `t` must be identical.
- CI workflow runs the suite on every PR; daily workflow runs a smoke subset before
  publishing.

## 5. Out of scope (YAGNI)

- Intraday data, daily-turnover strategies, short selling, options.
- Paid data feeds, cloud databases, container orchestration, model-serving APIs.
- Deep learning models; LambdaRank; hyperparameter re-search each retrain.
- Authentication or multi-user features on the dashboard.

## 6. Success criteria

1. Backtest reproducible end-to-end from a clean clone with one command.
2. No lookahead: leakage tests pass; every feature/label documented with its information
   timestamp.
3. Dashboard updates automatically every trading day without manual intervention.
4. Champion vs challenger comparison with signal-quality metrics (IC, decile spread), not
   just equity curves.
5. Limitations page states residual survivorship bias and cost assumptions plainly.
6. Honest outcome: if the strategies do NOT beat SPY after costs, the dashboard reports
   that faithfully — the project's value is the rigor, not the alpha claim.

## 7. Build phases

1. Data layer: price cache, point-in-time membership, FF factors, validation.
2. Features + leakage tests.
3. Backtest engine + metrics + baselines (SPY, random bootstrap).
4. Challenger (K-Means, repaired) end-to-end through the backtest.
5. Champion (LightGBM walk-forward) end-to-end through the backtest.
6. Daily/monthly pipeline (GitHub Actions) + registry + live paper-track.
7. Streamlit app (4 pages).
8. Docs, README, limitations writeup, polish.
