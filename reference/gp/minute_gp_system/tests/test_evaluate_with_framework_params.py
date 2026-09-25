import numpy as np
import pandas as pd

from gp.minute_gp_system.engines.unified_v2 import evaluate_with_framework as ewf
from gp.minute_gp_system.engines.unified_v2.config import N_PARAMS


class DummyLoader:
    n_active_periods = 2
    n_tradable_coins = 2


def test_load_gp_entries_batch_accepts_current_param_schema(monkeypatch):
    entry = {
        "params": np.zeros(N_PARAMS, dtype=np.int32).tolist(),
        "formula": "mean(A=open, w=45, sl=None, mask=open:none)",
    }

    monkeypatch.setattr(ewf, "_create_factor_loader", lambda **kwargs: DummyLoader())
    monkeypatch.setattr(
        ewf,
        "_prepare_loader_metadata",
        lambda loader, h5_path, minutes_per_period: (
            pd.date_range("2026-01-01", periods=2),
            np.array(["BTCUSDT", "ETHUSDT"]),
        ),
    )
    monkeypatch.setattr(
        ewf,
        "_compute_factor_batch",
        lambda entries, loader, backend, batch_size: np.ones((1, 2, 2), dtype=np.float32),
    )

    frames = ewf.load_gp_entries_batch([entry], h5_path="dummy.h5")

    factor_df, used_entry = frames[0]
    assert used_entry is entry
    assert factor_df.shape == (2, 2)
