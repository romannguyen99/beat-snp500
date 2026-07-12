import numpy as np
import pandas as pd

from beat_snp500.features.betas import BETA_COLS, FACTOR_COLS
from beat_snp500.models.kmeans import cluster_month

CLUSTER_REL_COLS = ["return_3m_cz", "return_6m_cz", "return_12m_cz"]


def vol_scaled_momentum(monthly: pd.DataFrame) -> pd.Series:
    return (monthly["return_12m"]
            / monthly["gk_vol"].replace(0.0, np.nan)).rename("mom_vol_scaled")


def residual_momentum(monthly: pd.DataFrame, factors: pd.DataFrame,
                      window: int = 12) -> pd.Series:
    """Blitz/Huij/Martens residual momentum: FF5 residual returns summed over
    months t-(window-1)..t-1 (skip month t, the reversal month), scaled by
    their standard deviation. Uses the lagged rolling betas already in the
    panel; strictly backward-looking."""
    df = monthly[["return_1m", *BETA_COLS]].join(
        factors[FACTOR_COLS + ["rf"]], on="date")
    explained = sum(df[b] * df[f] for b, f in zip(BETA_COLS, FACTOR_COLS))
    resid = df["return_1m"] - df["rf"] - explained

    def per(s: pd.Series) -> pd.Series:
        roll = s.shift(1).rolling(window - 1)
        return roll.sum() / roll.std()

    return (resid.groupby(level="ticker", group_keys=False).apply(per)
            .sort_index().rename("resid_mom"))


def cluster_relative_momentum(panel: pd.DataFrame) -> pd.DataFrame:
    """Momentum z-scored against K-means cluster peers, per month — the
    within-cluster ranking effect documented in improvement/improve_v1.md."""
    def z(g: pd.Series) -> pd.Series:
        sd = g.std(ddof=0)
        return (g - g.mean()) / sd if sd else g * 0.0

    parts = []
    for t, month_df in panel.groupby(level="date"):
        month = month_df.droplevel("date")
        Xz, labels = cluster_month(month)
        if Xz.empty:
            continue
        rel = (Xz[["return_3m", "return_6m", "return_12m"]]
               .groupby(labels).transform(z))
        rel.columns = CLUSTER_REL_COLS
        rel.index = pd.MultiIndex.from_product(
            [[t], rel.index], names=["date", "ticker"])
        parts.append(rel)
    return (pd.concat(parts).sort_index() if parts
            else pd.DataFrame(columns=CLUSTER_REL_COLS))
