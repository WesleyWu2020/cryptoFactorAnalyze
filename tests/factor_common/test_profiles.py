from dataclasses import FrozenInstanceError, asdict

import pytest

from factor_common.profiles import resolve_profile


def test_daily_defaults_include_strict_funding():
    p = resolve_profile("perp_1d", {})

    assert asdict(p) == {
        "profile_id": "perp_1d",
        "rebalance_days": 1,
        "anchor_date": "2024-01-01",
        "signal_delay_days": 1,
        "price_field": "open",
        "n_groups": 10,
        "factor_direction": 1,
        "initial_equity": 1.0,
        "gross_exposure": 1.0,
        "fee_rate": 0.0003,
        "include_funding": True,
        "funding_price_mode": "strict",
        "out_of_sample_days": 180,
        "split_date": None,
        "periods_per_year": 365,
    }


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
        ("rebalance_days", 0, "rebalance_days"),
        ("rebalance_days", -1, "rebalance_days"),
        ("rebalance_days", True, "rebalance_days"),
        ("rebalance_days", 1.5, "rebalance_days"),
        ("rebalance_days", "2", "rebalance_days"),
        ("price_field", "close", "open"),
        ("signal_delay_days", 0, "one-day"),
    ],
)
def test_profile_overrides_are_validated(field, value, message):
    with pytest.raises(ValueError, match=message):
        resolve_profile("perp_1d", {field: value})


@pytest.mark.parametrize("days", [1, 2, 7, 30, 365])
def test_daily_profile_supports_positive_integer_rebalance_days(days):
    profile = resolve_profile("perp_1d", {"rebalance_days": days})
    assert profile.rebalance_days == days
    assert profile.signal_delay_days == 1
    assert profile.price_field == "open"
