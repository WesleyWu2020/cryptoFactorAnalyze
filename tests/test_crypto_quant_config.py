from datetime import date
from pathlib import Path

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
