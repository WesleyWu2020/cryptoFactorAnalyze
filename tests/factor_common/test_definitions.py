from dataclasses import FrozenInstanceError, fields
from collections import UserDict
from types import MappingProxyType

import pandas as pd
import pytest

from factor_common.definitions import FactorSpec


def _calc_factor(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame()


class MutableMetadata:
    pass


@pytest.mark.parametrize("mapping", [dict, MappingProxyType, UserDict])
def test_factor_spec_recursively_copies_and_freezes_mappings(mapping):
    windows = [10, 20]
    nested = mapping({"windows": windows})
    meta = mapping({"nested": (nested,)})
    setting = mapping({"params": mapping({"items": [nested]})})

    spec = FactorSpec("mapping", meta, setting, _calc_factor, "sha")
    windows.append(30)

    assert spec.meta["nested"][0]["windows"] == (10, 20)
    assert spec.setting["params"]["items"][0]["windows"] == (10, 20)
    with pytest.raises(TypeError):
        spec.setting["params"]["items"][0]["windows"] = ()
    copied = FactorSpec("copy", spec.meta, spec.setting, _calc_factor, "sha")
    assert copied.meta == spec.meta
    assert copied.setting == spec.setting


def test_factor_spec_has_immutable_contract_and_copies_inputs():
    meta = {"tags": ["momentum"], "nested": {"enabled": True}}
    setting = {"required_fields": ["open"], "nested": {"window": 20}}
    spec = FactorSpec("momentum", meta, setting, _calc_factor, "abc123")

    assert [field.name for field in fields(FactorSpec)] == [
        "factor_id",
        "meta",
        "setting",
        "calc_factor",
        "source_sha256",
    ]
    assert spec.factor_id == "momentum"
    assert spec.calc_factor is _calc_factor
    assert spec.source_sha256 == "abc123"

    meta["tags"].append("reversal")
    meta["nested"]["enabled"] = False
    setting["required_fields"].append("close")
    setting["nested"]["window"] = 60

    assert spec.meta["tags"] == ("momentum",)
    assert spec.meta["nested"]["enabled"] is True
    assert spec.setting["required_fields"] == ("open",)
    assert spec.setting["nested"]["window"] == 20

    with pytest.raises(TypeError):
        spec.meta["new"] = "value"
    with pytest.raises(TypeError):
        spec.setting["nested"]["window"] = 10
    with pytest.raises(FrozenInstanceError):
        spec.factor_id = "other"


def test_factor_spec_freezes_bytearray_values():
    blob = bytearray(b"abc")
    spec = FactorSpec("bytes", {"blob": blob}, {}, _calc_factor, "sha")

    blob[0] = ord("z")

    assert spec.meta["blob"] == b"abc"
    assert isinstance(spec.meta["blob"], bytes)


def test_factor_spec_rejects_unsupported_mutable_values():
    with pytest.raises(TypeError, match="Unsupported mutable value"):
        FactorSpec("custom", {"value": MutableMetadata()}, {}, _calc_factor, "sha")
