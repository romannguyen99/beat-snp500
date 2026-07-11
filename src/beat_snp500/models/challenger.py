import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from beat_snp500 import config
from beat_snp500.portfolio.weights import equal_weights

MOM_COLS = ["return_3m", "return_6m", "return_12m"]


def kmeans_top10(month_df: pd.DataFrame, n_picks: int = config.N_PICKS,
                 k: int = config.K_CLUSTERS, seed: int = config.SEED) -> list[str]:
    X = month_df[config.FEATURES].dropna()
    if len(X) < max(k, n_picks):
        return []
    Xz = pd.DataFrame(StandardScaler().fit_transform(X.values),
                      index=X.index, columns=X.columns)
    labels = pd.Series(
        KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xz.values),
        index=X.index,
    )
    composite = Xz[MOM_COLS].mean(axis=1)
    # the momentum cluster is identified by its behaviour, never by label index
    best_cluster = composite.groupby(labels).mean().idxmax()
    members = composite[labels == best_cluster].sort_values(ascending=False)
    if len(members) < n_picks:
        return []
    return members.head(n_picks).index.tolist()


def challenger_picks(panel: pd.DataFrame,
                     n_picks: int = config.N_PICKS) -> dict:
    out = {}
    for t, month_df in panel.groupby(level="date"):
        picks = kmeans_top10(month_df.droplevel("date"), n_picks=n_picks)
        if picks:
            out[t] = equal_weights(picks)
    return out
