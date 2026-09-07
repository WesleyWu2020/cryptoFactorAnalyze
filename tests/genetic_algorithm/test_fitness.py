from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_common.labels import make_labels


def _panel(*, symbols=30, remove_days=()):
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    instruments = [f"S{i:02d}" for i in range(symbols)]
    rank = np.arange(symbols, dtype=float)
    opens = pd.DataFrame(
        100.0 * np.power(1.0 + rank * 0.0001, np.arange(len(dates))[:, None]),
        index=dates,
        columns=instruments,
    )
    labels = make_labels(opens, 1)
    values = pd.DataFrame(
        np.broadcast_to(rank, (len(dates), symbols)),
        index=dates,
        columns=instruments,
    )
    quality = pd.DataFrame(True, index=dates, columns=instruments)
    for date in remove_days:
        values.loc[pd.Timestamp(date)] = np.nan
    return values, labels, quality


def _config(**overrides):
    config = {
        "min_pairs": 20,
        "min_quarter_days": 45,
        "min_day_coverage": 0.8,
        "min_cell_coverage": 0.8,
        "node_count": 3,
    }
    config.update(overrides)
    return config


def test_training_score_uses_four_quarters_and_accepts_ordered_signal():
    from Genetic_Algorithm.fitness import score_training

    values, labels, quality = _panel()
    score = score_training(values, labels, quality, _config())

    assert score.direction == 1
    assert score.valid_days == 364
    assert score.day_coverage == pytest.approx(364 / 366)
    assert score.cell_coverage == pytest.approx(1.0)
    assert len(score.quarter_means) == 4
    assert score.eligible
    assert score.objective_vector == pytest.approx(
        (score.mean_ic, score.worst_quarter_ic, -score.node_count)
    )


def test_reversed_factor_gets_direction_from_training_only():
    from Genetic_Algorithm.fitness import score_training

    values, labels, quality = _panel()
    reversed_values = values.iloc[:, ::-1].copy()
    reversed_values.columns = values.columns
    score = score_training(reversed_values, labels, quality, _config())

    assert score.direction == -1
    assert score.mean_ic > 0.99
    assert all(value > 0.99 for value in score.quarter_means.values())


def test_zero_raw_training_mean_is_ineligible_before_direction_and_quarter_gates():
    from Genetic_Algorithm.fitness import score_training

    values, labels, quality = _panel()
    for row, date in enumerate(values.index):
        if row % 2:
            values.loc[date] = values.loc[date].iloc[::-1].to_numpy()

    score = score_training(values, labels, quality, _config())

    assert score.direction == 0
    assert score.mean_ic is None
    assert score.worst_quarter_ic is None
    assert score.objective_vector == (None, None, -score.node_count)
    assert not score.eligible
    assert any("zero raw training mean" in reason for reason in score.reasons)


def test_fewer_than_minimum_pairs_produce_no_valid_daily_ic():
    from Genetic_Algorithm.fitness import score_training

    values, labels, quality = _panel(symbols=19)
    score = score_training(values, labels, quality, _config())

    assert score.valid_days == 0
    assert score.mean_ic is None
    assert not score.eligible
    assert "minimum pairs" in " ".join(score.reasons)


def test_constant_labels_produce_no_valid_daily_ic():
    from Genetic_Algorithm.fitness import score_training

    values, labels, quality = _panel()
    labels.loc[:, :] = 0.0
    score = score_training(values, labels, quality, _config())

    assert score.valid_days == 0
    assert score.mean_ic is None
    assert score.icir is None


def test_cell_coverage_denominator_keeps_quality_eligible_cells():
    from Genetic_Algorithm.fitness import score_training

    values, labels, quality = _panel(remove_days=["2024-02-01", "2024-05-01"])
    score = score_training(values, labels, quality, _config())

    expected = len(values) * len(values.columns)
    observed = (values.notna() & quality).to_numpy().sum()
    assert score.cell_coverage == pytest.approx(observed / expected)
    assert score.cell_coverage < 1.0


def test_invalid_labels_do_not_shrink_cell_coverage_denominator():
    from Genetic_Algorithm.fitness import score_training

    values, labels, quality = _panel()
    labels.loc[pd.Timestamp("2024-06-01"), "S00"] = np.nan

    score = score_training(values, labels, quality, _config())

    assert score.cell_coverage == pytest.approx(1.0)
    assert score.day_coverage == pytest.approx(364 / 366)


def test_infinite_factor_values_are_excluded_from_daily_ic_pairs():
    from Genetic_Algorithm.fitness import score_training

    values, labels, quality = _panel()
    values.loc[pd.Timestamp("2024-06-01"), values.columns[:1]] = np.inf

    score = score_training(values, labels, quality, _config())

    assert score.valid_days == 364
    assert score.cell_coverage == pytest.approx(1 - 1 / (366 * 30))


def test_daily_ic_with_fewer_than_twenty_finite_pairs_is_invalid():
    from Genetic_Algorithm.fitness import score_training

    values, labels, quality = _panel()
    values.loc[pd.Timestamp("2024-06-01"), values.columns[:11]] = np.inf

    score = score_training(values, labels, quality, _config())

    assert score.valid_days == 363


def test_purged_tail_rows_cannot_change_training_score():
    from Genetic_Algorithm.fitness import score_training

    values, labels, quality = _panel()
    config = _config(train_start="2024-01-01", train_end="2024-12-31")
    baseline = score_training(values, labels, quality, config)
    changed = values.copy()
    changed.loc[pd.Timestamp("2024-12-30")] = 10_000.0
    changed.loc[pd.Timestamp("2024-12-31")] = -10_000.0
    score = score_training(changed, labels, quality, config)

    assert score == baseline


def test_signals_outside_training_range_are_rejected():
    from Genetic_Algorithm.fitness import score_training

    values, labels, quality = _panel()
    config = _config(train_start="2024-01-01", train_end="2024-12-30")

    with pytest.raises(ValueError, match="training range"):
        score_training(values, labels, quality, config)
