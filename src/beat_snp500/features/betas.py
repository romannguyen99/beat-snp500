import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from beat_snp500 import config

FACTOR_COLS = ["mkt_rf", "smb", "hml", "rmw", "cma"]
BETA_COLS = ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma"]


def rolling_ff5_betas(return_1m: pd.Series, factors: pd.DataFrame,
                      window: int = config.BETA_WINDOW_MONTHS) -> pd.DataFrame:
    df = return_1m.rename("ret").to_frame().join(factors[FACTOR_COLS], on="date", how="left")

    def per(x: pd.DataFrame) -> pd.DataFrame:
        x = x.dropna()
        if len(x) < window:
            return pd.DataFrame(index=x.index, columns=BETA_COLS, dtype=float)
        exog = sm.add_constant(x[FACTOR_COLS])
        params = (RollingOLS(x["ret"], exog, window=window, min_nobs=window)
                  .fit(params_only=True).params)
        params = params.drop(columns="const")
        params.columns = BETA_COLS
        return params

    betas = df.groupby(level="ticker", group_keys=False).apply(per)
    return betas.groupby(level="ticker").shift(1)  # use betas known at t-1 to describe t
