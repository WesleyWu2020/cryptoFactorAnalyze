import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


FACTOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "factor_analyse"
    / "unsubmit"
    / "crypto_alpha_volatility_efficiency_path_quality.py"
)


def _load_factor():
    spec = importlib.util.spec_from_file_location("path_quality_factor", FACTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _panel(n_days=60, n_instruments=2):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    columns = [f"COIN{i}USDT" for i in range(n_instruments)]
    base = np.arange(100.0, 100.0 + n_days)
    close = pd.DataFrame(
        np.column_stack([base, base + np.sin(np.arange(n_days))]),
        index=dates,
        columns=columns,
    )
    high = close * 1.02
    low = close * 0.98
    return {"close": close, "high": high, "low": low}


def test_path_quality_factor_matches_causal_formula():
    module = _load_factor()
    data = _panel()
    actual = module.calc_factor(data)

    window = module.SETTING["params"]["window_days"]
    close = data["close"]
    prev_close = close.shift(1)
    roc = close / close.shift(window) - 1.0
    range_pct = ((data["high"] - data["low"]) / prev_close).rolling(
        window, min_periods=window
    ).mean()
    daily_ret = close / prev_close - 1.0
    path_length = daily_ret.abs().rolling(window, min_periods=window).sum()
    path_efficiency = (roc.abs() / (path_length + module._EPS)).clip(0.0, 1.0)
    expected = (roc / (range_pct + module._EPS)) * (0.5 + 0.5 * path_efficiency)

    pd.testing.assert_frame_equal(actual, expected.replace([np.inf, -np.inf], np.nan))


def test_path_quality_factor_is_prefix_stable():
    module = _load_factor()
    data = _panel(n_days=70)
    full = module.calc_factor(data)
    cutoff = data["close"].index[50]
    truncated = module.calc_factor(
        {name: frame.loc[:cutoff] for name, frame in data.items()}
    )

    pd.testing.assert_frame_equal(
        full.loc[:cutoff], truncated, check_exact=False, atol=1e-12, rtol=1e-12
    )
