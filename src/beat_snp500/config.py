from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = DATA_DIR / "outputs"
BACKTEST_DIR = OUTPUTS_DIR / "backtest"

PRICES_PARQUET = DATA_DIR / "prices.parquet"
MEMBERSHIP_PARQUET = DATA_DIR / "membership.parquet"
FACTORS_PARQUET = DATA_DIR / "ff5_factors.parquet"

MLRUNS_DIR = ROOT / "mlruns"
MLFLOW_REGISTRY_DB = MODELS_DIR / "mlflow_registry.db"

HISTORY_START = "2008-01-01"
EXTRA_TICKERS = ["SPY"]

UNIVERSE_SIZE = 150
K_CLUSTERS = 4
TRAIN_WINDOW_MONTHS = 36
BETA_WINDOW_MONTHS = 24
BETA_FFILL_LIMIT = 3
MOMENTUM_LAGS = (1, 3, 6, 12)
WINSOR_PCT = 0.005
COST_BPS_ONE_WAY = 10.0
SEED = 0

BASE_FEATURES = [
    "return_1m", "return_3m", "return_6m", "return_12m",
    "gk_vol", "rsi", "atr_norm", "bb_width", "macd_hist",
    "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma",
]

# model inputs; validated extras get appended here (spec §4b), while
# BASE_FEATURES stays the clustering space for K-means
FEATURES = list(BASE_FEATURES) + [
    # ADOPTED 2026-07: beat baseline on dev (IC +0.0015 vs -0.0031) and
    # holdout (-0.0127 vs -0.0240) — see data/outputs/tuning/feature_eval.json
    "mom_vol_scaled",
]

MIN_PICKS = 5          # fewer must-buys than this -> hold previous portfolio
MAX_PICKS = 10         # more than this -> keep the highest-signal names
WEIGHT_CAP = 0.20      # per-stock cap; 5 * 0.20 = 1.0 keeps the floor fully invested
MUST_BUY_Z_KMEANS = 0.0
KMEANS_MOM_MODE = "mean_3_6_12"   # {"mean_3_6_12", "12_1", "vol_scaled"}
KMEANS_SELECT_RULE = "mean"       # {"mean", "risk_adj"}
MUST_BUY_Z_LGBM = 1.0
CHAMPION = "kmeans"    # role pointer (re-rated 2026-07); the other model is challenger
DEV_END = "2019-12-31"  # tuning/feature selection uses months <= this; one holdout pass after
