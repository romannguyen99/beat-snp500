import numpy as np
import pandas as pd

from beat_snp500.features.technical import (
    atr_norm, bb_width, garman_klass_vol, macd_hist, rsi,
)

N = 60
UP = pd.Series(np.linspace(100, 200, N))
DOWN = pd.Series(np.linspace(200, 100, N))
FLAT = pd.Series(np.full(N, 100.0))


def test_rsi_direction_and_bounds():
    assert rsi(UP).iloc[-1] > 70
    assert rsi(DOWN).iloc[-1] < 30
    r = rsi(UP).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_neutral_when_flat():
    assert rsi(FLAT).iloc[-1] == 50.0


def test_atr_norm_zero_when_flat():
    assert atr_norm(FLAT, FLAT, FLAT).iloc[-1] == 0.0


def test_bb_width_zero_when_flat():
    assert bb_width(FLAT).iloc[-1] == 0.0


def test_macd_hist_zero_when_flat():
    assert abs(macd_hist(FLAT).iloc[-1]) < 1e-12


def test_gk_vol_zero_when_flat():
    assert garman_klass_vol(FLAT, FLAT, FLAT, FLAT).iloc[-1] == 0.0


def test_warmup_is_nan():
    assert np.isnan(rsi(UP).iloc[0])
    assert np.isnan(bb_width(UP).iloc[0])
    assert np.isnan(garman_klass_vol(UP, UP * 1.01, UP * 0.99, UP).iloc[0])
