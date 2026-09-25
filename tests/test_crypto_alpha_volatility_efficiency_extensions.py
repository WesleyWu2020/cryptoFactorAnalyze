import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


FACTOR_NAMES = [
    "crypto_alpha_volatility_efficiency_taker_confirm",
    "crypto_alpha_volatility_efficiency_funding_crowding",
    "crypto_alpha_volatility_efficiency_downside_risk",
    "crypto_alpha_volatility_efficiency_btc_regime",
    "crypto_alpha_volatility_efficiency_liquidity_adjusted",
]


def _load_factor(name):
    path = (
        Path(__file__).resolve().parents[1]
        / "factor_analyse"
        / "unsubmit"
        / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _panel(n_days=70):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    columns = ["BTCUSDT", "COIN1USDT", "COIN2USDT"]
    trend = np.linspace(100.0, 140.0, n_days)
    close = pd.DataFrame(
        np.column_stack([trend, trend * 0.9 + np.sin(np.arange(n_days)), trend * 1.1]),
        index=dates,
        columns=columns,
    )
    high = close * 1.02
    low = close * 0.98
    quote_volume = pd.DataFrame(1_000_000.0, index=dates, columns=columns)
    quote_volume["COIN1USDT"] *= 2.0
    taker_buy_quote_volume = quote_volume * 0.52
    funding = pd.DataFrame(0.0001, index=dates, columns=columns)
    funding["COIN1USDT"] = 0.0002
    return {
        "close": close,
        "high": high,
        "low": low,
        "quote_volume": quote_volume,
        "taker_buy_quote_volume": taker_buy_quote_volume,
        "funding": funding,
    }


@pytest.mark.parametrize("name", FACTOR_NAMES)
def test_extension_is_causal_and_has_standard_contract(name):
    module = _load_factor(name)
    data = _panel()
    result = module.calc_factor(data)

    assert module.TYPE == "regular"
    assert module.META["factor_name"] == name
    assert result.index.equals(data["close"].index)
    assert result.columns.equals(data["close"].columns)

    cutoff = data["close"].index[50]
    truncated = {key: value.loc[:cutoff] for key, value in data.items()}
    prefix = module.calc_factor(truncated)
    pd.testing.assert_frame_equal(
        result.loc[:cutoff], prefix, check_exact=False, atol=1e-12, rtol=1e-12
    )
