import pandas as pd

from beat_snp500 import config
from beat_snp500.features.betas import BETA_COLS, rolling_ff5_betas
from beat_snp500.features.monthly import (
    add_forward_return, add_momentum, apply_membership, daily_indicators,
    liquidity_filter, monthly_panel,
)


def build_feature_panel(prices: pd.DataFrame, membership: pd.DataFrame,
                        factors: pd.DataFrame,
                        top_n: int = config.UNIVERSE_SIZE) -> pd.DataFrame:
    monthly = monthly_panel(daily_indicators(prices))
    monthly = add_momentum(monthly)
    monthly = add_forward_return(monthly)
    monthly = monthly.join(rolling_ff5_betas(monthly["return_1m"], factors))
    # bridge Ken French publication lag: carry a stock's last known betas forward
    # a bounded number of months (backward-looking, no leakage)
    monthly[BETA_COLS] = (monthly[BETA_COLS]
                          .groupby(level="ticker")
                          .ffill(limit=config.BETA_FFILL_LIMIT))
    monthly = apply_membership(monthly, membership)
    monthly = liquidity_filter(monthly, top_n=top_n)
    return monthly[monthly[config.FEATURES].notna().all(axis=1)]
