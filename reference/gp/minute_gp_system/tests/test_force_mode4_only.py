import importlib
import types

import numpy as np


def _reload_evolution(monkeypatch):
    monkeypatch.setenv("GP_FORCE_MODE4_ONLY", "1")
    import gp.minute_gp_system.engines.unified_v2.evolution as evolution
    return importlib.reload(evolution)


def test_force_mode4_repair_projects_all_rows_to_mode4(monkeypatch):
    evolution = _reload_evolution(monkeypatch)
    pop = np.zeros((4, evolution.N_PARAMS), dtype=np.int32)
    pop[:, 6] = [0, 1, 2, 3]

    repaired = evolution.SearchSpaceRepair()._do(None, pop)

    assert np.all(repaired[:, 6] == 3)


def test_force_mode4_candidate_slate_filters_non_mode4(monkeypatch):
    evolution = _reload_evolution(monkeypatch)
    mode2 = np.zeros(evolution.N_PARAMS, dtype=np.int32)
    mode2[6] = 1
    mode4 = np.zeros(evolution.N_PARAMS, dtype=np.int32)
    mode4[6] = 3
    mode4[13] = 0
    mode4[14] = 0
    result = types.SimpleNamespace(
        X=np.vstack([mode2, mode4]),
        F=-np.array([
            [10.0, 10.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32),
        pop=None,
    )

    slate, _, meta = evolution._build_candidate_slate(
        result,
        target_min=1,
        target_max=2,
        candidate_pool_size=2,
    )

    assert len(slate) == 1
    assert np.all(slate[:, 6] == 3)
    assert meta["force_mode4_only"] is True
