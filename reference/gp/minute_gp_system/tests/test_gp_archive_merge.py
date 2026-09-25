import json
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path("/root/crypto-research/users/wesleywu")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "factor_platform" not in sys.modules:
    factor_platform = types.ModuleType("factor_platform")
    factor_platform.__path__ = []
    config_mod = types.ModuleType("factor_platform.config")
    config_mod.DEFAULT_DB_PATH = Path("/tmp/factor_platform_default.db")
    config_mod.DEFAULT_GP_PHENOTYPE_ARCHIVE_PATH = Path("/tmp/factor_platform_default_archive.json")
    sys.modules["factor_platform"] = factor_platform
    sys.modules["factor_platform.config"] = config_mod

from gp.minute_gp_system.engines.unified_v2 import build_external_behavior_archive as archive_mod


def _write_archive(path: Path, behavior_signatures, labels, metadata, phenotypes=None):
    payload = {
        "metadata_json": np.asarray(json.dumps(metadata), dtype=np.str_),
        "labels": np.asarray(labels, dtype=np.str_),
        "behavior_signatures": np.asarray(behavior_signatures, dtype=np.float16),
    }
    if phenotypes is not None:
        payload["phenotypes"] = np.asarray(phenotypes, dtype=np.float16)
    with open(path, "wb") as f:
        np.savez_compressed(f, **payload)


def test_merge_pareto_into_archive_appends_and_creates_backup(tmp_path, monkeypatch):
    archive_path = tmp_path / "archive.npz"
    pareto_path = tmp_path / "pareto_results.json"
    h5_path = tmp_path / "dummy.h5"
    h5_path.write_text("")

    _write_archive(
        archive_path,
        behavior_signatures=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        labels=["old_r1"],
        metadata={"n_entries": 1, "labels": ["old_r1"], "behavior_shape": [1, 3]},
        phenotypes=np.asarray([[9.0, 8.0]], dtype=np.float32),
    )
    pareto_path.write_text(
        json.dumps(
            [
                {"rank": 1, "params": [1, 2, 3, 4, 5], "final_low_corr_gate": {"passed": True}},
                {"rank": 2, "params": [6, 7, 8, 9, 10], "final_low_corr_gate": {"passed": False}},
            ]
        )
    )

    monkeypatch.setattr(archive_mod, "set_backend", lambda *args, **kwargs: None)
    monkeypatch.setattr(archive_mod, "CryptoDataLoader", lambda **kwargs: object())
    monkeypatch.setattr(
        archive_mod,
        "CryptoFactorProblemV2",
        lambda loader, **kwargs: type(
            "ProblemStub",
            (),
            {
                "loader": loader,
                "period_returns": None,
                "period_tradable_mask": None,
                "_fitness_context_is": None,
                "_fitness_eval_kwargs": {},
                "eval_batch_size": 8,
            },
        )(),
    )
    monkeypatch.setattr(archive_mod, "build_behavior_context", lambda *args, **kwargs: {"ctx": True})
    monkeypatch.setattr(
        archive_mod,
        "reconstruct_archive_features",
        lambda params, problem, **kwargs: (
            np.ones((1, 3), dtype=np.float32),
            np.asarray([[0, 1, 0]], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(archive_mod, "standardize_phenotypes", lambda arr: np.asarray(arr))

    appended = archive_mod.merge_pareto_into_archive(
        archive_path=str(archive_path),
        pareto_path=str(pareto_path),
        h5_path=str(h5_path),
        period_minutes=1440,
        start_date="20220102",
        end_date="20241231",
        run_label="seed242",
    )

    assert appended == 1
    assert os.path.exists(str(archive_path) + ".bak")

    with np.load(archive_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        assert data["labels"].tolist() == ["old_r1", "seed242_r1"]
        assert data["behavior_signatures"].shape == (2, 3)
        assert metadata["n_entries"] == 2
        assert metadata["labels"] == ["old_r1", "seed242_r1"]


def test_merge_pareto_into_archive_skips_dimension_mismatch(tmp_path, monkeypatch, capsys):
    archive_path = tmp_path / "archive.npz"
    pareto_path = tmp_path / "pareto_results.json"
    h5_path = tmp_path / "dummy.h5"
    h5_path.write_text("")

    _write_archive(
        archive_path,
        behavior_signatures=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        labels=["old_r1"],
        metadata={"n_entries": 1, "labels": ["old_r1"], "behavior_shape": [1, 3]},
    )
    original_bytes = archive_path.read_bytes()
    pareto_path.write_text(
        json.dumps(
            [
                {"rank": 1, "params": [1, 2, 3, 4, 5], "final_low_corr_gate": {"passed": True}},
            ]
        )
    )

    monkeypatch.setattr(archive_mod, "set_backend", lambda *args, **kwargs: None)
    monkeypatch.setattr(archive_mod, "CryptoDataLoader", lambda **kwargs: object())
    monkeypatch.setattr(
        archive_mod,
        "CryptoFactorProblemV2",
        lambda loader, **kwargs: type(
            "ProblemStub",
            (),
            {
                "loader": loader,
                "period_returns": None,
                "period_tradable_mask": None,
                "_fitness_context_is": None,
                "_fitness_eval_kwargs": {},
                "eval_batch_size": 8,
            },
        )(),
    )
    monkeypatch.setattr(archive_mod, "build_behavior_context", lambda *args, **kwargs: {"ctx": True})
    monkeypatch.setattr(
        archive_mod,
        "reconstruct_archive_features",
        lambda params, problem, **kwargs: (
            np.ones((1, 4), dtype=np.float32),
            np.asarray([[0, 1, 0, 1]], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(archive_mod, "standardize_phenotypes", lambda arr: np.asarray(arr))

    appended = archive_mod.merge_pareto_into_archive(
        archive_path=str(archive_path),
        pareto_path=str(pareto_path),
        h5_path=str(h5_path),
        period_minutes=1440,
        start_date="20220102",
        end_date="20241231",
        run_label="seed242",
    )

    out = capsys.readouterr().out
    assert appended == 0
    assert archive_path.read_bytes() == original_bytes
    assert "dim mismatch" in out
