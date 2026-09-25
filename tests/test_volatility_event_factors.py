import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


FACTOR_NAMES = [
    "crypto_alpha_breakout_vol_squeeze_release",
    "crypto_alpha_vol_spike_liquidity_reversal",
]


def _load_factor(name):
    path = Path(__file__).resolve().parents[1] / "factor_analyse" / "unsubmit" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _panel(n_days=180):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    columns = ["BTCUSDT", "COIN1USDT", "COIN2USDT", "COIN3USDT"]
    t = np.arange(n_days, dtype=float)
    close = pd.DataFrame(
        np.column_stack([
            100 + 0.15 * t,
            80 + 0.12 * t + 2 * np.sin(t / 5),
            120 - 0.08 * t + 3 * np.cos(t / 7),
            60 + 0.08 * t + np.sin(t / 3),
        ]),
        index=dates,
        columns=columns,
    )
    high = close * 1.02
    low = close * 0.98
    quote_volume = pd.DataFrame(1_000_000.0, index=dates, columns=columns)
    quote_volume["COIN1USDT"] *= 2
    quote_volume["COIN2USDT"] *= 0.5
    return {"close": close, "high": high, "low": low, "quote_volume": quote_volume}


@pytest.mark.parametrize("name", FACTOR_NAMES)
def test_event_factor_contract_and_prefix_stability(name):
    module = _load_factor(name)
    data = _panel()
    result = module.calc_factor(data)

    assert module.TYPE == "regular"
    assert module.META["factor_name"] == name
    assert result.index.equals(data["close"].index)
    assert result.columns.equals(data["close"].columns)
    assert np.isfinite(result.to_numpy(dtype=float)).sum() > 0

    cutoff = data["close"].index[140]
    prefix = module.calc_factor({key: value.loc[:cutoff] for key, value in data.items()})
    pd.testing.assert_frame_equal(
        result.loc[:cutoff], prefix, check_exact=False, atol=1e-12, rtol=1e-12
    )
