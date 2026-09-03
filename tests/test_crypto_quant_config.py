from datetime import date
from pathlib import Path

import pytest

from data.crypto_quant.config import PipelineConfig


def test_default_config_uses_repo_data_directory(tmp_path: Path):
    cfg = PipelineConfig.default(tmp_path)
    assert cfg.store_path == tmp_path / "data" / "crypto_quant.h5"
    assert cfg.staging_path == tmp_path / "data" / ".crypto_quant.staging.h5"
    assert cfg.lock_path == tmp_path / "data" / ".crypto_quant.lock"
    assert cfg.universe_start == date(2024, 1, 1)
    assert cfg.warmup_days == 180
    assert cfg.top_n == 50
    assert cfg.cmc_overlap_days == 10
    assert cfg.binance_overlap_days == 7


def test_manual_config_normalizes_path_fields(tmp_path: Path):
    cfg = PipelineConfig(
        repo_root=tmp_path / "repo",
        store_path=tmp_path / "repo" / "data" / "store.h5",
        staging_path=Path("relative-staging.h5"),
        lock_path=Path("relative.lock"),
        rules_path=Path("relative-rules.json"),
    )

    assert cfg.repo_root.is_absolute()
    assert cfg.store_path.is_absolute()
    assert cfg.staging_path.is_absolute()
    assert cfg.lock_path.is_absolute()
    assert cfg.rules_path.is_absolute()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warmup_days", 0),
        ("top_n", 0),
        ("cmc_page_days", 0),
        ("cmc_overlap_days", -1),
        ("binance_overlap_days", -1),
        ("request_timeout_seconds", 0),
        ("max_attempts", 0),
        ("base_backoff_seconds", 0),
        ("max_backoff_seconds", 0),
    ],
)
def test_config_rejects_invalid_positive_or_nonnegative_values(
    tmp_path: Path, field: str, value: int
):
    values = {
        "repo_root": tmp_path,
        "store_path": tmp_path / "store.h5",
        "staging_path": tmp_path / "staging.h5",
        "lock_path": tmp_path / "lock",
        "rules_path": tmp_path / "rules.json",
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        PipelineConfig(**values)


def test_config_rejects_max_backoff_below_base_backoff(tmp_path: Path):
    with pytest.raises(ValueError, match="max_backoff_seconds"):
        PipelineConfig(
            repo_root=tmp_path,
            store_path=tmp_path / "store.h5",
            staging_path=tmp_path / "staging.h5",
            lock_path=tmp_path / "lock",
            rules_path=tmp_path / "rules.json",
            base_backoff_seconds=5.0,
            max_backoff_seconds=4.0,
        )


def test_config_is_immutable(tmp_path: Path):
    cfg = PipelineConfig.default(tmp_path)

    with pytest.raises(AttributeError):
        cfg.top_n = 25
