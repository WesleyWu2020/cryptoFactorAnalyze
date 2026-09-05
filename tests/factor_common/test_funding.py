from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_common.funding import check_funding_coverage, funding_cashflow, settle_funding
from factor_common.profiles import resolve_profile


STRICT = resolve_profile("perp_1d", {})
APPROX = resolve_profile("perp_1d", {"funding_price_mode": "daily_open_approx"})


def _make_events(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["funding_time"] = pd.to_datetime(frame["funding_time"], utc=True)
    price = pd.to_numeric(frame["mark_price"], errors="coerce")
    frame["mark_price_valid"] = price.gt(0) & np.isfinite(price)
    return frame


def _make_opens(values: dict[pd.Timestamp, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(values).T
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame.sort_index()


def _make_quality(pairs: list[tuple[str, str, str]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(day), instrument) for day, instrument, _ in pairs],
        names=["date", "instrument"],
    )
    return pd.DataFrame(
        {"funding_coverage_status": [status for _, _, status in pairs]}, index=index
    )


@pytest.mark.parametrize("quantity,rate,expected", [(2., .001, -.2),
                                                  (-2., .001, .2),
                                                  (2., -.001, .2)])
def test_funding_sign(quantity, rate, expected):
    assert funding_cashflow(quantity, 100., rate) == pytest.approx(expected)


@pytest.mark.parametrize("mark_price", [0.0, -1.0, float("nan"), float("inf")])
def test_funding_cashflow_rejects_invalid_price(mark_price):
    with pytest.raises(ValueError, match="mark_price"):
        funding_cashflow(2.0, mark_price, 0.001)


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), -float("inf")])
def test_funding_cashflow_rejects_nonfinite_rate(rate):
    with pytest.raises(ValueError, match="rate"):
        funding_cashflow(2.0, 100.0, rate)


@pytest.mark.parametrize("quantity", [float("nan"), float("inf")])
def test_funding_cashflow_rejects_nonfinite_quantity(quantity):
    with pytest.raises(ValueError, match="quantity"):
        funding_cashflow(quantity, 100.0, 0.001)


def test_settle_funding_signed_cashflows_for_held_symbols():
    events = _make_events([
        {"funding_time": "2024-01-03 00:00:00", "instrument": "AUSDT",
         "funding_rate": 0.001, "mark_price": 100.0, "rate_type": "Regular"},
        {"funding_time": "2024-01-03 08:00:00", "instrument": "BUSDT",
         "funding_rate": 0.001, "mark_price": 100.0, "rate_type": "Regular"},
        {"funding_time": "2024-01-03 16:00:00", "instrument": "AUSDT",
         "funding_rate": -0.001, "mark_price": 100.0, "rate_type": "Regular"},
    ])

    rows, diagnostics = settle_funding(
        {"AUSDT": 2.0, "BUSDT": -2.0}, events, profile=STRICT,
        opens=_make_opens({}),
    )

    assert rows["cashflow"].tolist() == pytest.approx([-0.2, 0.2, 0.2])
    assert rows["resolved"].all()
    assert not rows["price_approximated"].any()
    assert rows["quantity"].tolist() == [2.0, -2.0, 2.0]
    assert diagnostics["status"] == "complete"
    assert diagnostics["events_held"] == 3
    assert diagnostics["events_unresolved"] == 0
    assert diagnostics["funding_total"] == pytest.approx(0.2)


def test_settle_funding_ignores_unheld_symbols_and_preserves_raw_records():
    events = _make_events([
        {"funding_time": "2024-01-03 00:00:00", "instrument": "AUSDT",
         "funding_rate": 0.001, "mark_price": 100.0, "rate_type": "Regular"},
        {"funding_time": "2024-01-03 00:00:00", "instrument": "ZZZUSDT",
         "funding_rate": float("nan"), "mark_price": float("nan"), "rate_type": "Regular"},
        {"funding_time": "2024-01-03 08:00:00", "instrument": "ZEROUSDT",
         "funding_rate": float("nan"), "mark_price": -5.0, "rate_type": "Regular"},
    ])
    original = events.copy(deep=True)

    rows, diagnostics = settle_funding(
        {"AUSDT": 1.0, "ZEROUSDT": 0.0}, events, profile=STRICT,
        opens=_make_opens({}),
    )

    assert rows["instrument"].tolist() == ["AUSDT"]
    assert diagnostics["events_total"] == 3
    assert diagnostics["events_held"] == 1
    assert diagnostics["events_unresolved"] == 0
    assert diagnostics["status"] == "complete"
    pd.testing.assert_frame_equal(events, original)


def test_settle_funding_zero_holdings_with_unrelated_bad_events():
    events = _make_events([
        {"funding_time": "2024-01-03 00:00:00", "instrument": "ZZZUSDT",
         "funding_rate": float("nan"), "mark_price": float("nan"), "rate_type": "Regular"},
    ])

    rows, diagnostics = settle_funding({}, events, profile=STRICT, opens=_make_opens({}))

    assert rows.empty
    assert diagnostics["funding_total"] == 0.0
    assert diagnostics["events_unresolved"] == 0
    assert diagnostics["status"] == "complete"


def test_settle_funding_strict_invalid_mark_is_unresolved_not_zero():
    events = _make_events([
        {"funding_time": "2024-01-03 00:00:00", "instrument": "AUSDT",
         "funding_rate": 0.001, "mark_price": float("nan"), "rate_type": "Regular"},
        {"funding_time": "2024-01-03 08:00:00", "instrument": "AUSDT",
         "funding_rate": 0.001, "mark_price": 100.0, "rate_type": "Regular"},
    ])

    rows, diagnostics = settle_funding(
        {"AUSDT": 2.0}, events, profile=STRICT, opens=_make_opens({}),
    )

    bad = rows.iloc[0]
    assert not bad["resolved"]
    assert np.isnan(bad["cashflow"])
    assert bad["unresolved_reason"] == "invalid_mark_price"
    assert np.isnan(bad["mark_price"])
    assert np.isnan(bad["settlement_price"])
    assert rows.iloc[1]["cashflow"] == pytest.approx(-0.2)
    assert diagnostics["status"] == "incomplete"
    assert diagnostics["events_unresolved"] == 1
    assert diagnostics["unresolved_reasons"] == {"invalid_mark_price": 1}
    assert diagnostics["funding_total"] == pytest.approx(-0.2)


def test_settle_funding_missing_rate_is_never_approximated():
    events = _make_events([
        {"funding_time": "2024-01-03 08:00:00", "instrument": "AUSDT",
         "funding_rate": float("nan"), "mark_price": float("nan"), "rate_type": "Regular"},
        {"funding_time": "2024-01-03 16:00:00", "instrument": "AUSDT",
         "funding_rate": float("nan"), "mark_price": 100.0, "rate_type": "Regular"},
    ])
    opens = _make_opens({pd.Timestamp("2024-01-03"): {"AUSDT": 100.0}})

    rows, diagnostics = settle_funding({"AUSDT": 2.0}, events, profile=APPROX, opens=opens)

    assert not rows["resolved"].any()
    assert rows["unresolved_reason"].tolist() == ["missing_rate", "missing_rate"]
    assert rows["cashflow"].isna().all()
    assert diagnostics["status"] == "incomplete"
    assert diagnostics["approximated_events"] == 0


def test_settle_funding_daily_open_approx_records_replacement():
    events = _make_events([
        {"funding_time": "2024-01-03 00:00:00", "instrument": "AUSDT",
         "funding_rate": 0.001, "mark_price": -3.0, "rate_type": "Regular"},
        {"funding_time": "2024-01-03 08:00:00", "instrument": "AUSDT",
         "funding_rate": 0.001, "mark_price": float("nan"), "rate_type": "Regular"},
    ])
    opens = _make_opens({pd.Timestamp("2024-01-03"): {"AUSDT": 100.0}})

    rows, diagnostics = settle_funding({"AUSDT": 2.0}, events, profile=APPROX, opens=opens)

    assert rows["resolved"].all()
    assert rows["price_approximated"].all()
    assert rows["settlement_price"].tolist() == [100.0, 100.0]
    assert rows["mark_price"].tolist()[0] == -3.0
    assert np.isnan(rows["mark_price"].tolist()[1])
    assert rows["cashflow"].tolist() == pytest.approx([-0.2, -0.2])
    assert diagnostics["status"] == "complete"
    assert diagnostics["approximated_events"] == 2
    assert diagnostics["funding_total"] == pytest.approx(-0.4)


def test_settle_funding_approx_unavailable_without_valid_prior_open():
    events = _make_events([
        {"funding_time": "2024-01-04 08:00:00", "instrument": "AUSDT",
         "funding_rate": 0.001, "mark_price": float("nan"), "rate_type": "Regular"},
        {"funding_time": "2024-01-04 16:00:00", "instrument": "BUSDT",
         "funding_rate": 0.001, "mark_price": float("nan"), "rate_type": "Regular"},
    ])
    opens = _make_opens({
        pd.Timestamp("2024-01-03"): {"AUSDT": 100.0},
        pd.Timestamp("2024-01-04"): {"AUSDT": 110.0, "BUSDT": 0.0},
    })

    rows, diagnostics = settle_funding(
        {"AUSDT": 2.0, "BUSDT": 2.0}, events, profile=APPROX, opens=opens,
    )

    assert rows["resolved"].tolist() == [True, False]
    assert rows.iloc[0]["settlement_price"] == pytest.approx(110.0)
    assert rows.iloc[1]["unresolved_reason"] == "invalid_mark_price"
    assert not rows.iloc[1]["price_approximated"]
    assert np.isnan(rows.iloc[1]["cashflow"])
    assert diagnostics["status"] == "incomplete"


def test_coverage_accepts_complete_and_independently_scheduled_zero_event_days():
    quality = _make_quality([
        ("2024-01-03", "AUSDT", "complete"),
        ("2024-01-04", "AUSDT", "not_applicable"),
    ])

    rows, diagnostics = check_funding_coverage(
        [("AUSDT", "2024-01-03", "2024-01-05")], quality,
    )

    assert rows["accepted"].all()
    assert rows["status"].tolist() == ["complete", "not_applicable"]
    assert not rows["partial_day"].any()
    assert diagnostics["status"] == "complete"
    assert diagnostics["days_total"] == 2
    assert diagnostics["days_unresolved"] == 0


@pytest.mark.parametrize("status", ["missing", "unknown", "no_events"])
def test_coverage_leaves_unverified_statuses_unresolved(status):
    quality = _make_quality([("2024-01-03", "AUSDT", status)])

    rows, diagnostics = check_funding_coverage(
        [("AUSDT", "2024-01-03", "2024-01-04")], quality,
    )

    assert rows["status"].tolist() == [status]
    assert not rows["accepted"].iloc[0]
    assert diagnostics["status"] == "incomplete"
    assert diagnostics["days_unresolved"] == 1


def test_coverage_treats_absent_quality_row_as_unknown():
    quality = _make_quality([("2024-01-03", "BUSDT", "complete")])

    rows, diagnostics = check_funding_coverage(
        [("AUSDT", "2024-01-03", "2024-01-04")], quality,
    )

    assert rows["status"].tolist() == ["unknown"]
    assert rows["observed_status"].isna().all()
    assert not rows["accepted"].iloc[0]
    assert diagnostics["status"] == "incomplete"


def test_coverage_partial_day_does_not_claim_daily_completeness():
    quality = _make_quality([
        ("2024-01-03", "AUSDT", "complete"),
        ("2024-01-04", "AUSDT", "complete"),
    ])

    rows, diagnostics = check_funding_coverage(
        [("AUSDT", "2024-01-03 08:00:00", "2024-01-05")], quality,
    )

    first, second = rows.iloc[0], rows.iloc[1]
    assert first["partial_day"]
    assert first["observed_status"] == "complete"
    assert first["status"] == "unknown"
    assert not first["accepted"]
    assert second["accepted"]
    assert diagnostics["status"] == "incomplete"


def test_coverage_partial_day_accepts_independent_zero_event_schedule():
    quality = _make_quality([("2024-01-03", "AUSDT", "not_applicable")])

    rows, diagnostics = check_funding_coverage(
        [("AUSDT", "2024-01-03 08:00:00", "2024-01-03 16:00:00")], quality,
    )

    assert rows["partial_day"].iloc[0]
    assert rows["accepted"].iloc[0]
    assert diagnostics["status"] == "complete"


def test_approximation_does_not_upgrade_unknown_coverage():
    events = _make_events([
        {"funding_time": "2024-01-03 08:00:00", "instrument": "AUSDT",
         "funding_rate": 0.001, "mark_price": float("nan"), "rate_type": "Regular"},
    ])
    opens = _make_opens({pd.Timestamp("2024-01-03"): {"AUSDT": 100.0}})
    settled_rows, settle_diagnostics = settle_funding(
        {"AUSDT": 2.0}, events, profile=APPROX, opens=opens,
    )
    assert settle_diagnostics["status"] == "complete"
    assert settled_rows["price_approximated"].all()

    quality = _make_quality([("2024-01-03", "AUSDT", "unknown")])
    rows, diagnostics = check_funding_coverage(
        [("AUSDT", "2024-01-03", "2024-01-04")], quality,
    )

    assert rows["status"].tolist() == ["unknown"]
    assert not rows["accepted"].iloc[0]
    assert diagnostics["status"] == "incomplete"


def test_coverage_empty_holdings_are_complete():
    rows, diagnostics = check_funding_coverage([], _make_quality([]))

    assert rows.empty
    assert diagnostics["status"] == "complete"
    assert diagnostics["days_total"] == 0
