from dataclasses import asdict, FrozenInstanceError

import pytest

from portfolio.fixed_config import FactorAllocation, FixedConfig, daily_date


def _valid_kwargs(tmp_path):
    return dict(
        h5_path=str(tmp_path / "data.h5"),
        signal_start="2024-01-22",
        signal_end="2024-01-24",
        as_of="2024-01-26",
        factors=(FactorAllocation("factor.py", 1.0),),
        code_hashes={"factor.py": "a" * 64},
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


def test_fixed_config_roundtrip_and_frozen(tmp_path):
    cfg = FixedConfig(**_valid_kwargs(tmp_path))
    restored = FixedConfig.from_dict(asdict(cfg))
    assert restored == cfg
    assert isinstance(restored.factors, tuple)
    assert isinstance(restored.factors[0], FactorAllocation)
    assert restored.to_dict() == asdict(cfg)
    with pytest.raises(FrozenInstanceError):
        cfg.n_groups = 3
    with pytest.raises(TypeError):
        cfg.code_hashes["other.py"] = "b" * 64


def test_from_dict_rejects_unknown_key(tmp_path):
    payload = _valid_kwargs(tmp_path)
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        FixedConfig.from_dict(payload)


def test_daily_date_requires_naive_midnight():
    assert daily_date("2024-01-01") == daily_date("2024-01-01 00:00:00")
    with pytest.raises(ValueError):
        daily_date("2024-01-01 00:00:01")
    with pytest.raises(ValueError):
        daily_date("2024-01-01T00:00:00+00:00")
    with pytest.raises(ValueError):
        daily_date("NaT")


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(h5_path=""),
        lambda d: d.update(factors=({"path": "f.py"},)),
        lambda d: d.update(factors=({"path": "f.py", "allocation": 0},)),
        lambda d: d.update(factors=({"path": "f.py", "allocation": 0.5},)),
        lambda d: d.update(signal_start="2024-01-22T00:00:00"),
        lambda d: d.update(signal_start="2024-01-25"),
        lambda d: d.update(as_of="2024-01-21"),
        lambda d: d.update(rebalance_days=True),
        lambda d: d.update(n_groups=1),
        lambda d: d.update(fee_rate=1.0),
    ],
)
def test_fixed_config_rejects_invalid_changes(tmp_path, change):
    payload = _valid_kwargs(tmp_path)
    change(payload)
    with pytest.raises((TypeError, ValueError)):
        FixedConfig(**payload)
