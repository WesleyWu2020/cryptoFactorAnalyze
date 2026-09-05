from __future__ import annotations

import builtins
import hashlib
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).parents[2]


def _factor_source(
    factor_name: str = "example_momentum",
    *,
    meta_overrides: dict | None = None,
    meta_remove: str | None = None,
    setting_overrides: dict | None = None,
    setting_remove: str | None = None,
    type_value: str = "regular",
    body: str | None = None,
) -> str:
    meta = {
        "factor_name": factor_name,
        "author": "local",
        "level": "daily",
        "category": "momentum",
        "description": "N-day log momentum",
    }
    if meta_overrides:
        meta.update(meta_overrides)
    if meta_remove:
        meta.pop(meta_remove)
    setting = {
        "data_needed": ["close"],
        "universe": "historical_top50",
        "warmup_bars": 20,
        "preprocessing": "mad_rank",
        "params": {"window": 20},
        "factor_direction": 1,
    }
    if setting_overrides:
        setting.update(setting_overrides)
    if setting_remove:
        setting.pop(setting_remove)
    calculation = body or "return data_ctx['close']"
    return f'''\
import numpy as np

TYPE = {type_value!r}
META = {meta!r}
SETTING = {setting!r}

def calc_factor(data_ctx):
    {calculation}
'''


def _write_factor(tmp_path: Path, source: str, name: str = "example_momentum") -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_load_factor_builds_immutable_spec_and_hashes_source(tmp_path):
    from factor_common.definitions import FactorSpec
    from factor_common.loader import load_factor

    source = _factor_source()
    path = _write_factor(tmp_path, source)

    spec = load_factor(path)

    assert isinstance(spec, FactorSpec)
    assert spec.factor_id == "example_momentum"
    assert spec.meta["factor_name"] == "example_momentum"
    assert spec.setting["data_needed"] == ("close",)
    assert spec.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    result = spec.calc_factor({"close": pd.DataFrame([[1.0, 2.0]])})
    assert result.iloc[0, 0] == 1.0


def test_load_factor_executes_hashed_source_despite_stale_pyc(tmp_path):
    from factor_common.loader import load_factor

    source = _factor_source(body="return data_ctx['close'] + 1")
    path = _write_factor(tmp_path, source)
    original_stat = path.stat()
    cached = Path(py_compile.compile(
        str(path), doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    ))
    cached_bytes = cached.read_bytes()
    changed = source.replace("+ 1", "+ 2").encode()
    assert len(changed) == original_stat.st_size
    path.write_bytes(changed)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns

    spec = load_factor(path)

    assert spec.source_sha256 == hashlib.sha256(changed).hexdigest()
    assert spec.calc_factor({"close": 10}) == 12
    namespace = spec.calc_factor.__globals__
    assert namespace["__file__"] == str(path)
    assert namespace["__spec__"].origin == str(path)
    assert namespace["__loader__"] is namespace["__spec__"].loader
    assert cached.read_bytes() == cached_bytes


def test_load_factor_accepts_nested_mapping_params(tmp_path):
    from factor_common.loader import load_factor

    source = _factor_source() + '''
from types import MappingProxyType
SETTING['params'] = MappingProxyType({
    'nested': [MappingProxyType({'windows': [10, 20]})],
})
META = MappingProxyType(META)
SETTING = MappingProxyType(SETTING)
'''
    spec = load_factor(_write_factor(tmp_path, source))

    assert spec.setting['params']['nested'][0]['windows'] == (10, 20)
    with pytest.raises(TypeError):
        spec.setting['params']['nested'][0]['windows'] = ()


@pytest.mark.parametrize("key", ["pool", "rebalance_days", "warmup_bar", 42])
def test_load_factor_rejects_unknown_setting_keys(tmp_path, key):
    from factor_common.loader import load_factor

    path = _write_factor(tmp_path, _factor_source(setting_overrides={key: "ignored"}))
    with pytest.raises(ValueError, match="SETTING.*unsupported.*" + str(key)):
        load_factor(path)


@pytest.mark.parametrize("universe", ["current_top50", "all", "", None, []])
def test_load_factor_rejects_unsupported_universe(tmp_path, universe):
    from factor_common.loader import load_factor

    path = _write_factor(tmp_path, _factor_source(setting_overrides={"universe": universe}))
    with pytest.raises(ValueError, match="SETTING.universe.*historical_top50"):
        load_factor(path)


@pytest.mark.parametrize("frequency", ["daily", "hourly", None])
def test_load_factor_setting_frequency(tmp_path, frequency):
    from factor_common.loader import load_factor

    path = _write_factor(tmp_path, _factor_source(setting_overrides={"frequency": frequency}))
    if frequency == "daily":
        assert load_factor(path).setting["frequency"] == "daily"
    else:
        with pytest.raises(ValueError, match="daily.*SETTING.frequency"):
            load_factor(path)


@pytest.mark.parametrize("missing", ["factor_name", "author", "level", "category", "description"])
def test_load_factor_rejects_missing_meta_fields(tmp_path, missing):
    from factor_common.loader import load_factor

    source = _factor_source(meta_remove=missing)
    path = _write_factor(tmp_path, source)

    with pytest.raises(ValueError, match=missing):
        load_factor(path)


def test_load_factor_rejects_minute_frequency_with_daily_guidance(tmp_path):
    from factor_common.loader import load_factor

    path = _write_factor(tmp_path, _factor_source(meta_overrides={"level": "minutes"}))

    with pytest.raises(ValueError, match="daily"):
        load_factor(path)


@pytest.mark.parametrize("unsafe", ["../escape", "bad-name", "1factor", "_factor", "factor/name"])
def test_load_factor_rejects_unsafe_factor_identifiers(tmp_path, unsafe):
    from factor_common.loader import load_factor

    path = _write_factor(tmp_path, _factor_source(factor_name=unsafe), name="example_momentum")

    with pytest.raises(ValueError, match="factor_name|identifier"):
        load_factor(path)


def test_load_factor_rejects_unsafe_filename_with_path_traversal(tmp_path):
    from factor_common.loader import load_factor

    path = tmp_path / ".." / tmp_path.name / "bad-name.py"
    path.write_text(_factor_source(), encoding="utf-8")

    with pytest.raises(ValueError, match="factor filename"):
        load_factor(path)


def test_load_factor_requires_filename_and_meta_identifier_to_match(tmp_path):
    from factor_common.loader import load_factor

    path = _write_factor(tmp_path, _factor_source(factor_name="other_factor"))

    with pytest.raises(ValueError, match="match"):
        load_factor(path)


def test_load_factor_rejects_unsupported_type(tmp_path):
    from factor_common.loader import load_factor

    path = _write_factor(tmp_path, _factor_source(type_value="super"))

    with pytest.raises(ValueError, match="TYPE.*regular"):
        load_factor(path)


@pytest.mark.parametrize("field", [
    "close", "open", "high", "low", "volume", "quote_volume", "trade_count",
    "taker_buy_base_volume", "taker_buy_quote_volume",
])
def test_load_factor_accepts_fields_declared_by_daily_provider(tmp_path, field):
    from factor_common.loader import load_factor

    path = _write_factor(tmp_path, _factor_source(setting_overrides={"data_needed": [field]}))

    spec = load_factor(path)

    assert spec.setting["data_needed"] == (field,)


def test_load_factor_rejects_nonexistent_data_field(tmp_path):
    from factor_common.loader import load_factor

    path = _write_factor(tmp_path, _factor_source(setting_overrides={"data_needed": ["not_a_field"]}))

    with pytest.raises(ValueError, match="not_a_field"):
        load_factor(path)


@pytest.mark.parametrize("warmup", [-1, 1.5, True, "20"])
def test_load_factor_rejects_invalid_warmup(tmp_path, warmup):
    from factor_common.loader import load_factor

    path = _write_factor(tmp_path, _factor_source(setting_overrides={"warmup_bars": warmup}))

    with pytest.raises(ValueError, match="warmup_bars"):
        load_factor(path)


def test_load_factor_rejects_missing_settings_and_invalid_setting_values(tmp_path):
    from factor_common.loader import load_factor

    for key, value, message in [
        ("params", None, "params"),
        ("preprocessing", "unknown", "preprocessing"),
        ("factor_direction", 0, "factor_direction"),
        ("factor_direction", 1.0, "factor_direction"),
        ("factor_direction", -1.0, "factor_direction"),
        ("factor_direction", True, "factor_direction"),
    ]:
        setting = {key: value}
        source = _factor_source(
            setting_overrides=setting,
            setting_remove=key if value is None else None,
        )
        path = _write_factor(tmp_path, source, name=f"example_momentum_{key}")
        source = source.replace("'factor_name': 'example_momentum'", f"'factor_name': 'example_momentum_{key}'")
        path.write_text(source, encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            load_factor(path)


def test_validate_factor_output_rejects_duplicate_date_and_instrument_axes():
    from factor_common.loader import validate_factor_output

    duplicate_dates = pd.DataFrame(
        [[1.0], [2.0]],
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-01"], name="date"),
        columns=["BTCUSDT"],
    )
    duplicate_instruments = pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.DatetimeIndex(["2024-01-01"], name="date"),
        columns=["BTCUSDT", "BTCUSDT"],
    )

    with pytest.raises(ValueError, match="duplicate date"):
        validate_factor_output(duplicate_dates)
    with pytest.raises(ValueError, match="duplicate instrument"):
        validate_factor_output(duplicate_instruments)


def test_validate_factor_output_accepts_unique_matrix_axes():
    from factor_common.loader import validate_factor_output

    output = pd.DataFrame(
        [[1.0, np.nan]],
        index=pd.DatetimeIndex(["2024-01-01"], name="date"),
        columns=["BTCUSDT", "ETHUSDT"],
    )

    assert validate_factor_output(output) is output


def test_import_loader_does_not_read_market_data_or_write_files(tmp_path):
    script = (
        "import pandas as pd\n"
        "class ForbiddenHDFStore:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise AssertionError('market data read during import')\n"
        "pd.HDFStore = ForbiddenHDFStore\n"
        "import factor_common.loader\n"
    )
    environment = {"PYTHONPATH": str(REPO_ROOT)}

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_load_factor_does_not_read_market_data_or_write_files(monkeypatch, tmp_path):
    from factor_common.loader import load_factor

    path = _write_factor(tmp_path, _factor_source())
    market_reads = []
    writes = []
    real_open = builtins.open

    def forbidden_hdf_store(*args, **kwargs):
        market_reads.append((args, kwargs))
        raise AssertionError("market data read during factor load")

    def spy_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            writes.append((file, mode))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(pd, "HDFStore", forbidden_hdf_store)
    monkeypatch.setattr(builtins, "open", spy_open)

    spec = load_factor(path)

    assert spec.factor_id == "example_momentum"
    assert market_reads == []
    assert writes == []


def test_create_template_has_exact_daily_contract_and_formula(tmp_path):
    from factor_common.loader import create_template, load_factor
    from factor_common.loader import validate_factor_output

    target = create_template("new_factor", tmp_path)

    assert target == tmp_path / "new_factor.py"
    spec = load_factor(target)
    assert dict(spec.meta) == {
        "factor_name": "new_factor",
        "author": "local",
        "level": "daily",
        "category": "momentum",
        "description": "N-day log momentum",
    }
    assert spec.setting["data_needed"] == ("close",)
    assert spec.setting["universe"] == "historical_top50"
    assert spec.setting["warmup_bars"] == 20
    assert spec.setting["preprocessing"] == "mad_rank"
    assert dict(spec.setting["params"]) == {"window": 20}
    assert spec.setting["factor_direction"] == 1

    dates = pd.date_range("2024-01-01", periods=21, freq="D")
    close = pd.DataFrame({"BTCUSDT": np.arange(1.0, 22.0)}, index=dates)
    result = spec.calc_factor({"close": close})
    expected = np.log(close / close.shift(20))
    pd.testing.assert_frame_equal(result, expected)
    assert validate_factor_output(result) is result

    with pytest.raises(FileExistsError):
        create_template("new_factor", tmp_path)
    with pytest.raises(ValueError, match="identifier"):
        create_template("../escape", tmp_path)
