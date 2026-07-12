import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from beat_snp500 import config
from beat_snp500.portfolio.weights import conviction_weights

MOM_COLS = ["return_3m", "return_6m", "return_12m"]


def _composite(Xz: pd.DataFrame, month_df: pd.DataFrame,
              mom_mode: str) -> pd.Series:
    if mom_mode == "mean_3_6_12":
        return Xz[MOM_COLS].mean(axis=1)
    if mom_mode == "12_1":       # classic momentum skipping the reversal month
        raw = (1 + month_df["return_12m"]) / (1 + month_df["return_1m"]) - 1
    elif mom_mode == "vol_scaled":
        raw = month_df["return_12m"] / month_df["gk_vol"].replace(0.0, np.nan)
    else:
        raise ValueError(f"unknown mom_mode: {mom_mode}")
    raw = raw.loc[Xz.index]
    sd = raw.std(ddof=0)
    return ((raw - raw.mean()) / sd).fillna(0.0) if sd else raw * 0.0


def _best_cluster(composite: pd.Series, labels: pd.Series, select_rule: str):
    g = composite.groupby(labels)
    if select_rule == "mean":
        return g.mean().idxmax()
    if select_rule == "risk_adj":
        return (g.mean() / g.std(ddof=0).replace(0.0, np.nan)).idxmax()
    raise ValueError(f"unknown select_rule: {select_rule}")


def cluster_month(month_df: pd.DataFrame, k: int = config.K_CLUSTERS,
                  seed: int = config.SEED) -> tuple[pd.DataFrame, pd.Series]:
    """Standardize BASE_FEATURES and K-means-label one month's cross-section."""
    X = month_df[config.BASE_FEATURES].dropna()
    if len(X) < k:
        return pd.DataFrame(), pd.Series(dtype=int)
    Xz = pd.DataFrame(StandardScaler().fit_transform(X.values),
                      index=X.index, columns=X.columns)
    labels = pd.Series(
        KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xz.values),
        index=X.index,
    )
    return Xz, labels


def kmeans_must_buys(month_df: pd.DataFrame, k: int = config.K_CLUSTERS,
                     threshold: float = config.MUST_BUY_Z_KMEANS,
                     min_picks: int = config.MIN_PICKS,
                     max_picks: int = config.MAX_PICKS,
                     seed: int = config.SEED,
                     mom_mode: str = config.KMEANS_MOM_MODE,
                     select_rule: str = config.KMEANS_SELECT_RULE) -> dict[str, float]:
    """{ticker: momentum z-score} for the must-buy set; {} means hold."""
    Xz, labels = cluster_month(month_df, k=k, seed=seed)
    if Xz.empty:
        return {}
    composite = _composite(Xz, month_df.loc[Xz.index], mom_mode)
    # the momentum cluster is identified by its behaviour, never by label index
    best_cluster = _best_cluster(composite, labels, select_rule)
    members = composite[labels == best_cluster]
    must = members[members > threshold].sort_values(ascending=False)
    if len(must) < min_picks:
        return {}
    return must.head(max_picks).to_dict()


def kmeans_picks(panel: pd.DataFrame, **kwargs) -> dict:
    out = {}
    for t, month_df in panel.groupby(level="date"):
        signals = kmeans_must_buys(month_df.droplevel("date"), **kwargs)
        if signals:
            out[t] = conviction_weights(signals)
    return out
