import numpy as np
import pandas as pd


def perf_metrics(daily_returns: pd.Series, rf_annual: float = 0.0) -> dict:
    r = daily_returns.dropna()
    n = len(r)
    if n == 0:
        return {k: float("nan") for k in
                ["total_return", "cagr", "ann_vol", "sharpe",
                 "max_drawdown", "calmar"]} | {"n_days": 0}
    total = float((1 + r).prod() - 1)
    years = n / 252
    cagr = (1 + total) ** (1 / years) - 1
    vol = float(r.std(ddof=0) * np.sqrt(252))
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    ex = r - rf_daily
    sharpe = float(ex.mean() / ex.std(ddof=0) * np.sqrt(252)) if ex.std(ddof=0) > 0 else float("nan")
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    mdd = float(dd.min())
    calmar = float(cagr / abs(mdd)) if mdd < 0 else float("nan")
    return {"total_return": total, "cagr": float(cagr), "ann_vol": vol,
            "sharpe": sharpe, "max_drawdown": mdd, "calmar": calmar, "n_days": n}


def yearly_returns(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns).groupby(daily_returns.index.year).prod() - 1
