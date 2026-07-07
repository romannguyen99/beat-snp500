import pandas as pd


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
