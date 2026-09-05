from __future__ import annotations

import hashlib
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


def test_create_template_validates_name_and_never_overwrites(tmp_path):
    from factor_common.loader import create_template, load_factor

    target = create_template("new_factor", tmp_path)

    assert target == tmp_path / "new_factor.py"
    assert '"factor_name": "new_factor"' in target.read_text(encoding="utf-8")
    assert load_factor(target).factor_id == "new_factor"

    with pytest.raises(FileExistsError):
        create_template("new_factor", tmp_path)
    with pytest.raises(ValueError, match="identifier"):
        create_template("../escape", tmp_path)
