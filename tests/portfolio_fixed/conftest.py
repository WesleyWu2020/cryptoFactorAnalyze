from __future__ import annotations

from pathlib import Path

import pytest

from tests.factor_common.conftest import (
    EXAMPLE_CALENDAR,
    _complete_funding_schedule,
    _example_funding_events,
    write_h5_fixture,
)


@pytest.fixture
def fixed_h5(tmp_path: Path) -> Path:
    """A complete, deterministic HDF5 fixture for fixed-portfolio tests."""
    path = tmp_path / "fixed.h5"
    return write_h5_fixture(
        path,
        calendar=EXAMPLE_CALENDAR,
        funding_schedule=_complete_funding_schedule(),
        funding_events=_example_funding_events(),
    )


@pytest.fixture
def factor_paths(tmp_path: Path) -> list[Path]:
    """Write two tiny loader-compatible factors used by portfolio tests."""
    factor_dir = tmp_path
    common = '''
TYPE = "regular"
META = {
    "factor_name": FACTOR_NAME,
    "author": "portfolio-test",
    "level": "daily",
    "category": "test",
    "description": "Deterministic fixture factor",
}
SETTING = {
    "data_needed": ["close"],
    "universe": "historical_top50",
    "warmup_bars": 0,
    "preprocessing": "none",
    "params": {},
    "factor_direction": DIRECTION,
}

def calc_factor(data_ctx):
    return data_ctx["close"].copy()
'''
    specs = {
        "FixedClose": 1,
        "ReverseClose": -1,
    }
    paths: list[Path] = []
    for name, direction in specs.items():
        source = common.replace("FACTOR_NAME", repr(name))
        source = source.replace("DIRECTION", str(direction))
        path = factor_dir / f"{name}.py"
        path.write_text(source.lstrip(), encoding="utf-8")
        paths.append(path)
    return paths


@pytest.fixture
def config_dict(fixed_h5: Path, factor_paths: list[Path]) -> dict:
    """Build the canonical small fixed-portfolio configuration."""
    from dataclasses import asdict

    from portfolio.fixed_config import FactorAllocation, FixedConfig
    from portfolio.fixed_provenance import code_snapshot

    cfg = FixedConfig(
        h5_path=str(fixed_h5),
        signal_start="2024-01-22",
        signal_end="2024-01-24",
        as_of="2024-01-26",
        factors=(FactorAllocation(str(factor_paths[0]), 1.0),),
        code_hashes=code_snapshot(factor_paths[:1]),
        n_groups=2,
        min_valid_instruments=2,
        gross_limit=1.0,
        long_limit=1.0,
        short_limit=1.0,
        single_limit=1.0,
        net_limit=1.0,
        fee_rate=0.001,
        slippage=0.0,
    )
    return asdict(cfg)
