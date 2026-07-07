from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = DATA_DIR / "outputs"
BACKTEST_DIR = OUTPUTS_DIR / "backtest"

PRICES_PARQUET = DATA_DIR / "prices.parquet"
MEMBERSHIP_PARQUET = DATA_DIR / "membership.parquet"
FACTORS_PARQUET = DATA_DIR / "ff5_factors.parquet"
REGISTRY_JSON = MODELS_DIR / "registry.json"

HISTORY_START = "2008-01-01"
EXTRA_TICKERS = ["SPY"]

N_PICKS = 10
UNIVERSE_SIZE = 150
K_CLUSTERS = 4
TRAIN_WINDOW_MONTHS = 36
BETA_WINDOW_MONTHS = 24
BETA_FFILL_LIMIT = 3
MOMENTUM_LAGS = (1, 3, 6, 12)
WINSOR_PCT = 0.005
COST_BPS_ONE_WAY = 10.0
SEED = 0

FEATURES = [
    "return_1m", "return_3m", "return_6m", "return_12m",
    "gk_vol", "rsi", "atr_norm", "bb_width", "macd_hist",
    "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
]
