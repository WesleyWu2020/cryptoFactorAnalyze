import numpy as np

from gp.minute_gp_system.engines.unified_v2.config import INDICATOR_INDEX
from gp.minute_gp_system.engines.unified_v2.indicators import (
    derive_indicators,
    derive_indicators_subset,
)


DIRECTIONAL_FIELDS = (
    "up_volume_pressure",
    "down_volume_pressure",
    "new_long_pressure",
    "new_short_pressure",
    "short_cover_pressure",
    "long_liquidation_pressure",
)


def _raw_panel():
    close = np.array(
        [[
            [100.0, 100.0, 100.0, 100.0],
            [102.0, 98.0, 102.0, 98.0],
            [101.0, 99.0, 101.0, 99.0],
        ]],
        dtype=np.float32,
    )
    open_ = np.array(
        [[[100.0, 100.0, 100.0, 100.0]] * 3],
        dtype=np.float32,
    )
    volume = np.array(
        [[[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]]],
        dtype=np.float32,
    )
    oi = np.array(
        [[[100.0, 100.0, 100.0, 100.0], [105.0, 105.0, 95.0, 95.0], [110.0, 110.0, 90.0, 90.0]]],
        dtype=np.float32,
    )
    return {
        "open": open_,
        "high": np.maximum(open_, close) + 1.0,
        "low": np.minimum(open_, close) - 1.0,
        "close": close,
        "volume": volume,
        "open_interest": oi,
    }


def _field(full, name):
    return np.asarray(full[:, INDICATOR_INDEX[name]])


def test_directional_state_fields_split_volume_and_oi_quadrants():
    full = derive_indicators(_raw_panel())

    up = _field(full, "up_volume_pressure")
    down = _field(full, "down_volume_pressure")
    assert up[0, 1, 0] > 0.0
    assert down[0, 1, 1] > 0.0
    assert down[0, 1, 0] == 0.0
    assert up[0, 1, 1] == 0.0

    expected_active_coin = {
        "new_long_pressure": 0,
        "new_short_pressure": 1,
        "short_cover_pressure": 2,
        "long_liquidation_pressure": 3,
    }
    for name, active_coin in expected_active_coin.items():
        values = _field(full, name)[0]
        assert np.all(values[:, active_coin] > 0.0)
        assert np.all(values[:, np.arange(4) != active_coin] == 0.0)


def test_subset_matches_full_for_directional_state_fields():
    raw = _raw_panel()
    indices = [INDICATOR_INDEX[name] for name in DIRECTIONAL_FIELDS]
    subset = np.asarray(derive_indicators_subset(raw, indices))
    full = np.asarray(derive_indicators(raw))

    for local_idx, global_idx in enumerate(indices):
        np.testing.assert_allclose(
            subset[:, local_idx],
            full[:, global_idx],
            equal_nan=True,
        )
