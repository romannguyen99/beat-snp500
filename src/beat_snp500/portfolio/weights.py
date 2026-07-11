import pandas as pd

from beat_snp500 import config


def equal_weights(tickers) -> dict[str, float]:
    return {t: 1.0 / len(tickers) for t in tickers}


def max_sharpe_weights(close: pd.DataFrame, tickers, asof,
                       lookback_days: int = 252,
                       bounds: tuple = (0.05, 0.20)) -> dict[str, float]:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier

    have = [t for t in tickers if t in close.columns]
    px = close.loc[close.index <= asof, have].tail(lookback_days)
    px = px.dropna(axis=1, thresh=max(2, int(len(px) * 0.5)))
    if px.shape[1] < 2:
        return equal_weights(tickers)
    try:
        mu = expected_returns.mean_historical_return(px)
        cov = risk_models.sample_cov(px)
        ef = EfficientFrontier(mu, cov, weight_bounds=bounds, solver="SCS")
        ef.max_sharpe()
        w = {k: float(v) for k, v in ef.clean_weights().items() if v > 0}
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}
    except Exception:
        return equal_weights(list(px.columns))


def conviction_weights(signals: dict[str, float],
                       cap: float = config.WEIGHT_CAP) -> dict[str, float]:
    """Signal-proportional weights with a per-stock cap (water-filling).

    Selection thresholds are >= 0 so signals must be strictly positive.
    Feasibility (len * cap >= 1) is guaranteed by MIN_PICKS * WEIGHT_CAP = 1;
    with that guard the redistribution loop always terminates with sum == 1.
    """
    if not signals:
        return {}
    w = pd.Series(signals, dtype=float)
    if (w <= 0).any():
        raise ValueError("conviction_weights requires strictly positive signals")
    if len(w) * cap < 1.0 - 1e-12:
        raise ValueError(f"cap {cap} infeasible for {len(w)} names")
    w = w / w.sum()
    for _ in range(len(w)):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = w < cap - 1e-12
        w[under] += excess * w[under] / float(w[under].sum())
    return {t: float(v) for t, v in w.items()}
