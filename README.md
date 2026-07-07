# beat-snp500

A quant research project that ranks S&P 500 stocks daily and paper-tracks two
10-stock strategies against SPY, with a bias-controlled walk-forward backtest.

**Live dashboard:** _add your Streamlit Community Cloud URL here_

## What it does

- **Champion:** LightGBM model trained walk-forward on monthly cross-sectional
  features, predicting next-month return ranks. Top 10 picks, monthly rebalance.
- **Challenger:** K-Means clustering (k=4); the momentum cluster is identified by
  centroid behaviour each month and its top 10 stocks are selected.
- **Backtest hygiene:** point-in-time index membership, adjusted prices, leak-tested
  features, turnover-based transaction costs, SPY + random-portfolio benchmarks.
- **Pipeline:** GitHub Actions updates data, re-scores stocks daily, retrains and
  rebalances monthly; artifacts are committed to the repo and rendered by Streamlit.

## Quick start

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest -q                                # full test suite
    python scripts/run_daily.py --force-rebalance   # builds caches + first model (~15 min)
    python scripts/run_backtest.py           # full walk-forward backtest (~30 min)
    streamlit run app/streamlit_app.py

## Repository layout

    src/beat_snp500/   data, features, models, backtest, portfolio, jobs
    scripts/           run_daily.py, run_backtest.py
    app/               Streamlit dashboard
    data/, models/     committed artifacts written by the pipeline
                       (data/prices.parquet is a local/CI cache, not committed —
                       it's gitignored and rebuilt from yfinance on a cold start;
                       CI persists it between runs via actions/cache instead)
    notebooks/         original exploratory notebook (superseded)
    docs/superpowers/  design spec and implementation plan

## Honesty box

The dashboard's *Methodology & Limitations* tab documents residual survivorship
bias, execution assumptions, and factor-lag handling. If the strategies do not
beat SPY after costs, the dashboard says so. Nothing here is investment advice.
