from pathlib import Path

import numpy as np
import pandas as pd

from beat_snp500 import config
from beat_snp500.backtest.bootstrap import monthly_holding_returns, random_portfolio_bootstrap
from beat_snp500.backtest.engine import run_backtest
from beat_snp500.backtest.metrics import perf_metrics, yearly_returns
from beat_snp500.data.prices import close_matrix
from beat_snp500.features.pipeline import build_feature_panel
from beat_snp500.io_utils import atomic_write_json, atomic_write_parquet
from beat_snp500.models.champion import (champion_picks, decile_spread,
                                          spearman_ic, walk_forward_scores)
from beat_snp500.models.kmeans import kmeans_picks
from beat_snp500.portfolio.weights import max_sharpe_weights


def _with_max_sharpe(picks: dict, close: pd.DataFrame) -> dict:
    return {t: max_sharpe_weights(close, list(w), asof=t) for t, w in picks.items()}


def _equity(daily: pd.Series, start) -> pd.Series:
    r = daily[daily.index >= start]
    return (1 + r).cumprod()


def survivorship_stats(membership: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Per month-end: how many point-in-time index members actually have any
    price history in `close`, quantifying residual survivorship bias (members
    delisted without Yahoo Finance history are silently absent from `close`)."""
    valid = {t for t in close.columns if close[t].notna().any()}
    has_data = membership["ticker"].isin(valid)
    out = (membership.assign(has_data=has_data)
           .groupby("date").agg(n_members=("ticker", "size"), n_with_data=("has_data", "sum")))
    out["missing_frac"] = 1 - out["n_with_data"] / out["n_members"]
    return out.reset_index()


def run_report(prices: pd.DataFrame, membership: pd.DataFrame, factors: pd.DataFrame,
               out_dir: Path, top_n: int = config.UNIVERSE_SIZE,
               train_window: int = config.TRAIN_WINDOW_MONTHS,
               n_draws: int = 1000, as_of=None) -> dict:
    out_dir = Path(out_dir)
    # exclude the current (possibly partial) calendar month so the last label is clean
    as_of = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    prices = prices[prices["date"] < as_of.replace(day=1)]
    close = close_matrix(prices)

    panel = build_feature_panel(prices, membership, factors, top_n=top_n)

    scores = walk_forward_scores(panel, train_window=train_window)
    ic = spearman_ic(scores, panel["fwd_return_1m"])
    spread = decile_spread(scores, panel["fwd_return_1m"])
    picks = {
        "champion": champion_picks(scores),
        "kmeans": kmeans_picks(panel),
    }
    picks["champion_ms"] = _with_max_sharpe(picks["champion"], close)
    picks["kmeans_ms"] = _with_max_sharpe(picks["kmeans"], close)

    results = {name: run_backtest(p, close) for name, p in picks.items()}
    start = results["champion"].daily_returns.index.min()

    rf_annual = float((1 + factors["rf"]).prod() ** (12 / len(factors)) - 1)
    daily = {name: res.daily_returns for name, res in results.items()}
    if "SPY" in close.columns:
        daily["spy"] = close["SPY"].pct_change().dropna()

    curves, metrics, yearly, turnover = [], {}, {}, {}
    for name, r in daily.items():
        eq = _equity(r, start)
        curves.append(pd.DataFrame({"date": eq.index, "series": name, "equity": eq.values}))
        metrics[name] = perf_metrics(r[r.index >= start], rf_annual=rf_annual)
        yearly[name] = {int(y): float(v) for y, v in yearly_returns(r[r.index >= start]).items()}
    for name, res in results.items():
        turnover[name] = float(res.turnover.mean()) if len(res.turnover) else float("nan")

    membership_period = membership[membership["date"] < as_of.replace(day=1)]
    surv = survivorship_stats(membership_period, close)
    survivorship_summary = {
        "mean_missing_frac": float(surv["missing_frac"].mean()) if len(surv) else float("nan"),
        "max_missing_frac": float(surv["missing_frac"].max()) if len(surv) else float("nan"),
    }

    champ_months = sorted(picks["champion"])
    universe = {t: panel.xs(t, level="date").index.tolist() for t in champ_months}
    boot = random_portfolio_bootstrap(
        universe, monthly_holding_returns(close, champ_months), n_draws=n_draws)
    champ_cagr = metrics["champion"]["cagr"]
    boot_summary = {
        "cagr_p05": float(np.quantile(boot["cagr"], 0.05)),
        "cagr_p50": float(np.quantile(boot["cagr"], 0.50)),
        "cagr_p95": float(np.quantile(boot["cagr"], 0.95)),
        "champion_cagr_percentile": float((boot["cagr"] < champ_cagr).mean()),
    }

    atomic_write_parquet(pd.concat(curves, ignore_index=True), out_dir / "equity_curves.parquet")
    atomic_write_json(
        metrics | {"yearly": yearly, "turnover": turnover, "survivorship": survivorship_summary},
        out_dir / "metrics.json")
    atomic_write_parquet(
        pd.DataFrame({"ic": ic, "decile_spread": spread}).rename_axis("date").reset_index(),
        out_dir / "ic_monthly.parquet")
    atomic_write_parquet(surv, out_dir / "survivorship.parquet")
    # auditable per-month picks for all four pick sets, so any sub-period
    # concentration (e.g. a strategy leaning on a handful of names) can be
    # inspected directly rather than inferred from aggregate metrics.
    atomic_write_json(
        {name: {str(pd.Timestamp(t).date()): weights for t, weights in p.items()}
         for name, p in picks.items()},
        out_dir / "picks.json")
    atomic_write_parquet(boot["band"].reset_index(), out_dir / "bootstrap.parquet")
    atomic_write_json(boot_summary, out_dir / "bootstrap_summary.json")
    return metrics
