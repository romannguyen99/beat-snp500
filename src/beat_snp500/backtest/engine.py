from dataclasses import dataclass, field

import pandas as pd

from beat_snp500 import config


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    skipped: list = field(default_factory=list)


def run_backtest(picks: dict, close: pd.DataFrame,
                 cost_bps: float = config.COST_BPS_ONE_WAY) -> BacktestResult:
    dates = sorted(picks)
    cost_rate = cost_bps / 1e4
    parts, turns, costs = [], {}, {}
    skipped: list = []
    prev_end_weights = pd.Series(dtype=float)

    for i, t in enumerate(dates):
        window_end = dates[i + 1] if i + 1 < len(dates) else close.index.max()
        window_idx = close.index[(close.index > t) & (close.index <= window_end)]
        wanted = [k for k in picks[t] if k in close.columns]
        if len(window_idx) == 0 or not wanted:
            skipped.append(t)
            continue

        hist = close[wanted].loc[:t].ffill()
        if hist.empty:
            skipped.append(t)
            continue
        base = hist.iloc[-1].dropna()
        avail = base.index.tolist()
        if not avail:
            skipped.append(t)
            continue

        w = pd.Series({k: picks[t][k] for k in avail}, dtype=float)
        w = w / w.sum()

        growth = close.loc[window_idx, avail].ffill().div(base[avail], axis=1).fillna(1.0)
        value = growth.mul(w, axis=1).sum(axis=1)
        rets = value.div(value.shift(1).fillna(1.0)).sub(1.0)

        turnover = float(w.sub(prev_end_weights, fill_value=0.0).abs().sum())
        cost = cost_rate * turnover
        rets.iloc[0] -= cost
        turns[t], costs[t] = turnover, cost

        prev_end_weights = (w * growth.iloc[-1]) / value.iloc[-1]
        parts.append(rets)

    if not parts:
        empty = pd.Series(dtype=float)
        return BacktestResult(empty, empty.copy(), empty.copy(), skipped=skipped)

    dr = pd.concat(parts)
    dr = dr[~dr.index.duplicated(keep="first")].sort_index().rename("strategy")
    return BacktestResult(dr, pd.Series(turns).sort_index(), pd.Series(costs).sort_index(),
                          skipped=skipped)
