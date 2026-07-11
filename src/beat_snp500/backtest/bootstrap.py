import numpy as np
import pandas as pd

from beat_snp500 import config


def monthly_holding_returns(close: pd.DataFrame, month_ends) -> pd.DataFrame:
    mclose = close.groupby(close.index + pd.offsets.MonthEnd(0)).last()
    fwd = mclose.pct_change(fill_method=None).shift(-1)
    return fwd.reindex(pd.DatetimeIndex(month_ends))


def random_portfolio_bootstrap(universe: dict, holding_rets: pd.DataFrame,
                               n_draws: int = 1000, n_picks: int | dict = 10,
                               cost_bps: float = config.COST_BPS_ONE_WAY,
                               seed: int = config.SEED) -> dict:
    rng = np.random.default_rng(seed)
    # full replacement each month → Σ|Δw| ≈ 2 → cost = 2 × one-way rate
    cost_per_month = 2.0 * cost_bps / 1e4
    dates = sorted(universe)
    monthly = np.full((n_draws, len(dates)), np.nan)

    for j, t in enumerate(dates):
        elig = [x for x in universe[t] if x in holding_rets.columns]
        rets_t = holding_rets.loc[t, elig].dropna()
        if rets_t.empty:
            monthly[:, j] = 0.0
            continue
        want = n_picks[t] if isinstance(n_picks, dict) else n_picks
        k = min(want, len(rets_t))
        for d in range(n_draws):
            sample = rng.choice(rets_t.index.to_numpy(), size=k, replace=False)
            monthly[d, j] = rets_t[sample].mean() - cost_per_month

    equity = np.cumprod(1 + monthly, axis=1)
    band = pd.DataFrame(
        {"p05": np.quantile(equity, 0.05, axis=0),
         "p50": np.quantile(equity, 0.50, axis=0),
         "p95": np.quantile(equity, 0.95, axis=0)},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    years = len(dates) / 12
    cagr = equity[:, -1] ** (1 / years) - 1
    return {"band": band, "cagr": cagr}
