import pandas as pd
import pytest

from factor_common.labels import label_dates, make_labels


def test_label_uses_next_open_not_signal_close():
    opens = pd.DataFrame({"A": [90., 100., 110., 121.]},
                         index=pd.date_range("2024-01-01", periods=4))
    out = make_labels(opens, hold_days=1)
    assert abs(out.loc["2024-01-01", "A"] - 0.1) < 1e-12
    assert pd.isna(out.iloc[-1, 0])


def test_hold_days_extends_exit_to_entry_plus_hold():
    opens = pd.DataFrame({"A": [90., 100., 110., 121., 133.1]},
                         index=pd.date_range("2024-01-01", periods=5))
    out = make_labels(opens, hold_days=2)
    # Entry is the next open (t+1); exit is hold_days after entry (t+1+hold_days).
    assert out.loc["2024-01-01", "A"] == pytest.approx(121.0 / 100.0 - 1)
    assert out.loc["2024-01-02", "A"] == pytest.approx(133.1 / 110.0 - 1)
    assert out.iloc[-2:].isna().all().all()


def test_daily_calendar_gaps_are_rejected():
    opens = pd.DataFrame({"A": [90., 100., 110.]},
                         index=pd.date_range("2024-01-01", periods=3))
    gapped = opens.drop(pd.Timestamp("2024-01-02"))
    with pytest.raises(ValueError, match="calendar day"):
        make_labels(gapped, hold_days=1)


def test_invalid_hold_days_are_rejected():
    opens = pd.DataFrame({"A": [90., 100.]}, index=pd.date_range("2024-01-01", periods=2))
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="hold_days"):
            make_labels(opens, hold_days=bad)


def test_label_dates_align_entry_and_exit_with_execution_calendar():
    dates = pd.date_range("2024-01-01", periods=4)
    entry, exit_ = label_dates(dates, hold_days=1)
    assert list(entry) == list(dates + pd.Timedelta(days=1))
    assert list(exit_) == list(dates + pd.Timedelta(days=2))
    # Exit times stay concrete beyond the data tail so boundary purge can
    # compare them against split dates instead of dropping NaT silently.
    assert exit_[-1] == pd.Timestamp("2024-01-06")
    entry2, exit2 = label_dates(dates, hold_days=2)
    assert entry2[0] == pd.Timestamp("2024-01-02")
    assert exit2[0] == pd.Timestamp("2024-01-04")
