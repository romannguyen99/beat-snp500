"""Sanity check #3: decompose the kmeans model's edge into two pieces —
(A) cluster selection: does the cluster it picks (highest composite momentum)
    actually earn a higher average forward return than the rest of the
    universe that month? (a level effect, not a ranking)
(B) within-cluster ranking: given you're already in that cluster, does
    ranking members by composite momentum predict *which* of them do best?
    (an IC, directly comparable to sanity_check_feature_ic.py's unconditional
    momentum IC)

Mirrors kmeans.py's cluster_month clustering exactly so this reflects what
kmeans_must_buys actually does, just with the intermediate cluster labels
exposed instead of only the final must-buy set.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from beat_snp500 import config
from beat_snp500.data.factors import load_ff5
from beat_snp500.features.pipeline import build_feature_panel
from beat_snp500.models.kmeans import MOM_COLS


def cluster_month(month_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, int]:
    """month_df indexed by ticker. Returns (composite, labels, best_cluster)."""
    X = month_df[config.BASE_FEATURES]
    Xz = pd.DataFrame(StandardScaler().fit_transform(X.values),
                      index=X.index, columns=X.columns)
    labels = pd.Series(
        KMeans(n_clusters=config.K_CLUSTERS, n_init=10, random_state=config.SEED)
        .fit_predict(Xz.values),
        index=X.index)
    composite = Xz[MOM_COLS].mean(axis=1)
    best_cluster = composite.groupby(labels).mean().idxmax()
    return composite, labels, best_cluster


def month_by_date_ic(rows: list[pd.DataFrame]) -> pd.Series:
    """rows: per-month DataFrames with columns ['composite', 'fwd'], one per date."""
    out = {}
    for t, df in rows:
        df = df.dropna()
        if len(df) >= 3:
            out[t] = df["composite"].corr(df["fwd"], method="spearman")
    return pd.Series(out)


def main() -> int:
    prices = pd.read_parquet(config.PRICES_PARQUET)
    membership = pd.read_parquet(config.MEMBERSHIP_PARQUET)
    factors = load_ff5(config.FACTORS_PARQUET)

    as_of = pd.Timestamp.today().normalize()
    prices = prices[prices["date"] < as_of.replace(day=1)]
    panel = build_feature_panel(prices, membership, factors, top_n=config.UNIVERSE_SIZE)

    uncond_rows, cond_rows = [], []
    cluster_sizes, cluster_excess = [], []

    for t, month_df in panel.groupby(level="date"):
        flat = month_df.droplevel("date")
        composite, labels, best_cluster = cluster_month(flat)
        # index-aligned join, not positional — avoids order-mismatch bugs
        both = pd.concat({"composite": composite, "fwd": flat["fwd_return_1m"]}, axis=1)

        uncond_rows.append((t, both))

        members = labels[labels == best_cluster].index
        cond_rows.append((t, both.loc[members]))

        cluster_sizes.append(len(members))
        cluster_excess.append(both.loc[members, "fwd"].mean() - both["fwd"].mean())

    uncond_ic = month_by_date_ic(uncond_rows)
    cond_ic = month_by_date_ic(cond_rows)

    print(f"{'':30s}{'mean IC':>10s}{'std IC':>10s}{'IC IR':>10s}")
    print(f"{'unconditional momentum':30s}{uncond_ic.mean():10.4f}{uncond_ic.std():10.4f}"
          f"{uncond_ic.mean()/uncond_ic.std():10.4f}")
    print(f"{'within momentum-cluster':30s}{cond_ic.mean():10.4f}{cond_ic.std():10.4f}"
          f"{cond_ic.mean()/cond_ic.std():10.4f}")
    print()
    sizes = pd.Series(cluster_sizes)
    excess = pd.Series(cluster_excess)
    universe_n = panel.groupby(level="date").size().mean()
    t_stat = excess.mean() / excess.std() * len(excess) ** 0.5
    print(f"momentum-cluster size: mean {sizes.mean():.0f} / universe ~{universe_n:.0f} "
          f"(min {sizes.min()}, max {sizes.max()})")
    print(f"cluster mean fwd return minus universe mean fwd return: "
          f"{excess.mean():.4%} per month (t-stat {t_stat:.2f}, n={len(excess)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
