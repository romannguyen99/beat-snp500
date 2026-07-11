import numpy as np
import pandas as pd
import pytest

from beat_snp500.portfolio.weights import equal_weights, max_sharpe_weights


def test_equal_weights():
    w = equal_weights(["A", "B", "C", "D"])
    assert w == {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}


def make_close(n_tickers=10, n_days=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=n_days)
    data = {f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n_days)))
            for i in range(n_tickers)}
    return pd.DataFrame(data, index=idx)


def test_max_sharpe_respects_bounds_and_sums_to_one():
    close = make_close()
    w = max_sharpe_weights(close, list(close.columns), asof=close.index[-1])
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    # SCS is a first-order solver; bounds hold to solver precision, not machine precision
    assert all(0.05 - 1e-4 <= v <= 0.20 + 1e-4 for v in w.values())


def test_fallback_to_equal_on_bad_input():
    close = make_close(n_tickers=1)
    w = max_sharpe_weights(close, ["T0", "MISSING"], asof=close.index[-1])
    assert w == {"T0": 0.5, "MISSING": 0.5}


from beat_snp500.portfolio.weights import conviction_weights


def test_conviction_weights_proportional_and_sums_to_one():
    w = conviction_weights({"A": 2.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["A"] == pytest.approx(2 * w["B"])


def test_conviction_weights_caps_and_redistributes():
    w = conviction_weights({"A": 100.0, "B": 1.0, "C": 1.0, "D": 1.0,
                            "E": 1.0, "F": 1.0})
    assert w["A"] == pytest.approx(0.20)
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(v <= 0.20 + 1e-9 for v in w.values())


def test_conviction_weights_floor_case_forces_equal_20pct():
    # 5 names, extreme conviction spread: cap forces exactly 5 x 20%
    w = conviction_weights({"A": 100.0, "B": 50.0, "C": 1.0, "D": 1.0, "E": 1.0})
    assert all(v == pytest.approx(0.20) for v in w.values())


def test_conviction_weights_rejects_nonpositive_signals():
    with pytest.raises(ValueError):
        conviction_weights({"A": 1.0, "B": 0.0, "C": 1.0, "D": 1.0, "E": 1.0})


def test_conviction_weights_rejects_infeasible_cap():
    with pytest.raises(ValueError):
        conviction_weights({"A": 1.0, "B": 1.0}, cap=0.20)  # 2 * 0.2 < 1


def test_conviction_weights_empty_returns_empty():
    assert conviction_weights({}) == {}
