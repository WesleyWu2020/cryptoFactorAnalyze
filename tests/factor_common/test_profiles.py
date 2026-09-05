from dataclasses import FrozenInstanceError

import pytest

from factor_common.profiles import resolve_profile


def test_daily_defaults_include_strict_funding():
    p = resolve_profile("perp_1d", {})

    assert (p.signal_delay_days, p.price_field, p.include_funding) == (
        1,
        "open",
        True,
    )
    assert p.funding_price_mode == "strict"


@pytest.mark.parametrize("name", ["perp_4h", "perp_8h"])
def test_intraday_is_rejected(name):
    with pytest.raises(ValueError, match="daily"):
        resolve_profile(name, {})


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unsupported profile"):
        resolve_profile("unknown", {})


@pytest.mark.parametrize("fee_rate", [-0.0001, 1.0, float("inf")])
def test_illegal_fees_are_rejected(fee_rate):
    with pytest.raises(ValueError, match="fee_rate"):
        resolve_profile("perp_1d", {"fee_rate": fee_rate})


def test_unknown_override_is_rejected_before_replacement():
    with pytest.raises(ValueError, match="Unknown override"):
        resolve_profile("perp_1d", {"not_a_field": 1})


def test_mutable_override_input_is_not_retained_or_mutated():
    overrides = {"fee_rate": 0.001}
    original = dict(overrides)

    profile = resolve_profile("perp_1d", overrides)
    overrides["fee_rate"] = 0.2

    assert overrides == {"fee_rate": 0.2}
    assert original == {"fee_rate": 0.001}
    assert profile.fee_rate == 0.001


def test_profile_is_frozen():
    profile = resolve_profile("perp_1d", {})

    with pytest.raises(FrozenInstanceError):
        profile.fee_rate = 0.001


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("periods_per_year", 0, "periods_per_year"),
        ("factor_direction", 0, "factor_direction"),
        ("initial_equity", 0, "initial_equity"),
        ("n_groups", 1, "n_groups"),
        ("gross_exposure", 0, "gross_exposure"),
        ("funding_price_mode", "loose", "funding_price_mode"),
        ("rebalance_days", 2, "daily"),
        ("price_field", "close", "open"),
        ("signal_delay_days", 0, "one-day"),
    ],
)
def test_profile_overrides_are_validated(field, value, message):
    with pytest.raises(ValueError, match=message):
        resolve_profile("perp_1d", {field: value})
