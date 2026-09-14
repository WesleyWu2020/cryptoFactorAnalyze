from types import SimpleNamespace

import numpy as np
import pandas as pd

from ML_factor_mining.dataset import rank_panel
from ML_factor_mining.targets import build_targets, partition_masks


def _config(*, holding_days=2, min_pairs=5, target_type="rank_return"):
    return SimpleNamespace(
        holding_days=holding_days,
        min_pairs=min_pairs,
        target_type=target_type,
    )


def _opens(days=8):
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            instrument: np.arange(days, dtype=float) + offset
            for offset, instrument in enumerate(["A", "B", "C", "D", "E"], start=1)
        },
        index=dates,
    )


def test_build_targets_uses_next_open_and_exposes_label_dates():
    opens = _opens()
    membership = pd.DataFrame(True, index=opens.index, columns=opens.columns)

    targets = build_targets(opens, membership, _config(holding_days=2))
    first = targets.loc[(pd.Timestamp("2024-01-01"), "A")]

    assert first["raw_return"] == (opens.loc["2024-01-04", "A"] / opens.loc["2024-01-02", "A"]) - 1
    assert first["entry_date"] == pd.Timestamp("2024-01-02")
    assert first["exit_date"] == pd.Timestamp("2024-01-04")
    assert pd.isna(targets.loc[(pd.Timestamp("2024-01-06"), "A"), "raw_return"])


def test_rank_return_matches_rank_panel_on_eligible_raw_returns():
    opens = _opens()
    membership = pd.DataFrame(True, index=opens.index, columns=opens.columns)

    targets = build_targets(opens, membership, _config(holding_days=1))
    expected = rank_panel(targets["raw_return"])

    pd.testing.assert_series_equal(targets["target"], expected, check_names=False)


def test_future_membership_does_not_admit_earlier_signal_rows():
    opens = _opens()
    membership = pd.DataFrame(False, index=opens.index, columns=opens.columns)
    membership.loc["2024-01-02":, :] = True
    membership.loc["2024-01-01", ["A", "B", "C", "D"]] = True

    targets = build_targets(opens, membership, _config(holding_days=1, min_pairs=5))

    first_day = targets.xs(pd.Timestamp("2024-01-01"), level="date")
    assert first_day["raw_return"].isna().all()
    assert targets.loc[(pd.Timestamp("2024-01-02"), "E"), "raw_return"] > 0


def test_partition_masks_purge_by_concrete_exit_boundary():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "instrument"])
    fold = SimpleNamespace(
        history_start=pd.Timestamp("2024-01-01"),
        validation_start=pd.Timestamp("2024-01-10"),
        retrain_at=pd.Timestamp("2024-01-20"),
    )

    masks = partition_masks(index, fold, holding_days=3)

    assert index.get_level_values("date")[masks["fit"]].tolist() == list(
        pd.date_range("2024-01-01", "2024-01-05")
    )
    assert index.get_level_values("date")[masks["validation"]].tolist() == list(
        pd.date_range("2024-01-10", "2024-01-15")
    )
    assert index.get_level_values("date")[masks["refit"]].tolist() == list(
        pd.date_range("2024-01-01", "2024-01-15")
    )
