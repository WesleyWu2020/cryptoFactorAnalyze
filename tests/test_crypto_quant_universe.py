from datetime import date

import pandas as pd
import pytest

from data.crypto_quant.universe import UniverseBuildError, build_monthly_universe


@pytest.fixture
def synthetic_inputs():
    constituents = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 1),
                "cmc_id": cmc_id,
                "symbol": f"C{cmc_id}",
                "name": f"Coin {cmc_id}",
                "weight": 100 - cmc_id,
            }
            for cmc_id in range(1, 61)
        ]
    )
    # CMC id 50 is in the CMC top 50 but has no mapping; id 51 must enter.
    mapped_ids = list(range(1, 50)) + list(range(51, 57))
    mappings = pd.DataFrame(
        [
            {
                "cmc_id": cmc_id,
                "cmc_symbol": f"C{cmc_id}",
                "binance_symbol": f"C{cmc_id}USDT",
                "valid_from": pd.Timestamp("2020-01-01"),
                "valid_to": pd.NaT,
                "status": "TRADING",
            }
            for cmc_id in mapped_ids
        ]
    )
    symbols = mappings["binance_symbol"].tolist()
    klines = pd.DataFrame(
        [
            {"date": timestamp, "symbol": symbol, "completed": True}
            for timestamp in (pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-31"))
            for symbol in symbols
        ]
    )
    return constituents, mappings, klines


def test_intersects_before_selecting_top50(synthetic_inputs):
    out = build_monthly_universe(*synthetic_inputs, date(2024, 1, 1), date(2024, 1, 31))
    january = out[out["decision_date"] == pd.Timestamp("2024-01-01")]
    assert len(january) == 50
    assert january["binance_symbol"].nunique() == 50
    assert january["market_cap_rank"].tolist() == list(range(1, 51))
    assert "C51USDT" in set(january["binance_symbol"])
    assert pd.Timestamp("2024-01-02") == january["effective_date"].iloc[0]


def test_requires_t_minus_one_kline(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    missing_symbol = mappings.iloc[0]["binance_symbol"]
    klines = klines[~((klines["symbol"] == missing_symbol) & (klines["date"] == pd.Timestamp("2023-12-31")))]
    out = build_monthly_universe(constituents, mappings, klines, date(2024, 1, 1), date(2024, 1, 31))
    assert missing_symbol not in set(out["binance_symbol"])


def test_fails_closed_below_50(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    with pytest.raises(UniverseBuildError, match="eligible contracts: 49"):
        build_monthly_universe(constituents, mappings.head(49), klines, date(2024, 1, 1), date(2024, 1, 31))


def test_tie_breaks_by_cmc_id_and_closes_intervals(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    february = constituents.copy()
    february["date"] = date(2024, 2, 1)
    february.loc[february["cmc_id"].isin([1, 2]), "weight"] = 999
    out = build_monthly_universe(
        pd.concat([constituents, february], ignore_index=True), mappings, klines,
        date(2024, 1, 1), date(2024, 2, 29), top_n=2,
    )
    accepted = out.sort_values(["decision_date", "market_cap_rank"])
    assert accepted.loc[accepted["decision_date"] == pd.Timestamp("2024-02-01"), "cmc_id"].tolist() == [1, 2]
    january = accepted[accepted["decision_date"] == pd.Timestamp("2024-01-01")]
    assert january["effective_end_date"].eq(pd.Timestamp("2024-02-01") - pd.Timedelta(days=1)).all()
    assert accepted[accepted["decision_date"] == pd.Timestamp("2024-02-01")]["effective_end_date"].isna().all()


def test_historical_break_status_is_not_retroactively_filtered(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    mappings.loc[mappings["cmc_id"] == 1, "status"] = "BREAK"
    out = build_monthly_universe(constituents, mappings, klines, date(2024, 1, 1), date(2024, 1, 31))
    assert "C1USDT" in set(out["binance_symbol"])


def test_current_observation_filters_symbols_for_same_day_only(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    current_symbols = set(mappings["binance_symbol"]) - {"C1USDT"}
    out = build_monthly_universe(
        constituents, mappings, klines, date(2024, 1, 1), date(2024, 1, 31),
        current_trading_symbols=current_symbols, current_observed_date=date(2024, 1, 1),
    )
    assert "C1USDT" not in set(out["binance_symbol"])


def test_schema_is_deterministic(synthetic_inputs):
    out = build_monthly_universe(*synthetic_inputs, date(2024, 1, 1), date(2024, 1, 31))
    assert list(out.columns) == [
        "decision_date", "effective_date", "effective_end_date", "cmc_id",
        "cmc_symbol", "binance_symbol", "weight", "market_cap_rank",
    ]


def test_cutoff_replay_matches_full_history_before_cutoff(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    later_constituent = constituents.copy()
    later_constituent["date"] = date(2024, 2, 1)
    full = build_monthly_universe(
        pd.concat([constituents, later_constituent], ignore_index=True), mappings, klines,
        date(2024, 1, 1), date(2024, 2, 29),
    )
    cutoff = build_monthly_universe(
        constituents, mappings, klines[klines["date"] <= pd.Timestamp("2024-01-31")],
        date(2024, 1, 1), date(2024, 1, 31),
    )
    full_january = full[full["decision_date"] == pd.Timestamp("2024-01-01")].reset_index(drop=True)
    comparable = [column for column in cutoff.columns if column != "effective_end_date"]
    pd.testing.assert_frame_equal(full_january[comparable], cutoff[comparable].reset_index(drop=True))


def test_excludes_cmc_ids_with_multiple_current_mappings(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    duplicate = mappings.iloc[[0]].copy()
    duplicate["binance_symbol"] = "C1ALTUSDT"
    mappings = pd.concat([mappings, duplicate], ignore_index=True)
    klines = pd.concat(
        [klines, pd.DataFrame([{"date": pd.Timestamp("2023-12-31"), "symbol": "C1ALTUSDT", "completed": True}])],
        ignore_index=True,
    )
    out = build_monthly_universe(constituents, mappings, klines, date(2024, 1, 1), date(2024, 1, 31))
    assert "C1USDT" not in set(out["binance_symbol"])
    assert "C1ALTUSDT" not in set(out["binance_symbol"])
    assert out["cmc_id"].is_unique


def test_excludes_cross_cmc_symbol_mapping_conflicts(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    mappings.loc[mappings["cmc_id"] == 1, "binance_symbol"] = "C2USDT"
    out = build_monthly_universe(constituents, mappings, klines, date(2024, 1, 1), date(2024, 1, 31))
    assert out["binance_symbol"].is_unique
    assert not out["cmc_id"].isin([1, 2]).any()


def test_completed_requires_strict_boolean_true(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    klines["completed"] = klines["completed"].astype(object)
    klines.loc[(klines["symbol"] == "C1USDT") & (klines["date"] == pd.Timestamp("2023-12-31")), "completed"] = "False"
    out = build_monthly_universe(constituents, mappings, klines, date(2024, 1, 1), date(2024, 1, 31))
    assert "C1USDT" not in set(out["binance_symbol"])


def test_uses_first_available_cmc_snapshot_in_october_and_delays_effective_date(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    october = constituents.copy()
    october["date"] = date(2024, 10, 2)
    october["weight"] = october["weight"] + 1_000
    november = constituents.copy()
    november["date"] = date(2024, 11, 1)
    later_october = constituents.copy()
    later_october["date"] = date(2024, 10, 15)
    later_october["weight"] = later_october["weight"] + 2_000
    klines = pd.concat(
        [
            klines,
            pd.DataFrame(
                [
                    {"date": timestamp, "symbol": symbol, "completed": True}
                    for timestamp in (pd.Timestamp("2024-10-01"), pd.Timestamp("2024-10-31"))
                    for symbol in mappings["binance_symbol"]
                ]
            ),
        ],
        ignore_index=True,
    )

    out = build_monthly_universe(
        pd.concat([constituents, october, later_october, november], ignore_index=True),
        mappings,
        klines,
        date(2024, 10, 1),
        date(2024, 11, 30),
        top_n=2,
    )

    october_rows = out[out["decision_date"] == pd.Timestamp("2024-10-02")]
    assert len(october_rows) == 2
    assert october_rows["decision_date"].eq(pd.Timestamp("2024-10-02")).all()
    assert out["decision_date"].tolist() == [
        pd.Timestamp("2024-10-02"), pd.Timestamp("2024-10-02"),
        pd.Timestamp("2024-11-01"), pd.Timestamp("2024-11-01"),
    ]
    assert october_rows["effective_date"].eq(pd.Timestamp("2024-10-03")).all()
    assert october_rows["effective_end_date"].eq(pd.Timestamp("2024-10-31")).all()
    assert not out["effective_date"].eq(pd.Timestamp("2024-10-02")).any()


def test_uses_september_2025_snapshot_on_september_2(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    september = constituents.copy()
    september["date"] = date(2025, 9, 2)
    september["weight"] = september["weight"] + 1_000
    klines = pd.concat(
        [
            klines,
            pd.DataFrame(
                [
                    {"date": pd.Timestamp("2025-09-01"), "symbol": symbol, "completed": True}
                    for symbol in mappings["binance_symbol"]
                ]
            ),
        ],
        ignore_index=True,
    )

    out = build_monthly_universe(
        september, mappings, klines, date(2025, 9, 1), date(2025, 9, 30), top_n=2
    )

    assert out["decision_date"].eq(pd.Timestamp("2025-09-02")).all()
    assert out["effective_date"].eq(pd.Timestamp("2025-09-03")).all()


def test_october_snapshot_cannot_change_universe_before_its_decision(synthetic_inputs):
    constituents, mappings, klines = synthetic_inputs
    september = constituents.copy()
    september["date"] = date(2024, 9, 1)
    october = constituents.copy()
    october["date"] = date(2024, 10, 2)
    october["weight"] = 0
    october.loc[october["cmc_id"] == 55, "weight"] = 2
    october.loc[october["cmc_id"] == 56, "weight"] = 1
    klines = pd.concat(
        [
            klines,
            pd.DataFrame(
                [
                    {"date": timestamp, "symbol": symbol, "completed": True}
                    for timestamp in (
                        pd.Timestamp("2024-08-31"),
                        pd.Timestamp("2024-09-30"),
                        pd.Timestamp("2024-10-01"),
                    )
                    for symbol in mappings["binance_symbol"]
                ]
            ),
        ],
        ignore_index=True,
    )

    out = build_monthly_universe(
        pd.concat([september, october], ignore_index=True),
        mappings,
        klines,
        date(2024, 9, 1),
        date(2024, 10, 31),
        top_n=2,
    )

    prior = out[out["decision_date"] == pd.Timestamp("2024-09-01")]
    later = out[out["decision_date"] == pd.Timestamp("2024-10-02")]
    assert set(prior["cmc_id"]) == {1, 2}
    assert prior["effective_date"].eq(pd.Timestamp("2024-09-02")).all()
    assert prior["effective_end_date"].eq(pd.Timestamp("2024-10-01")).all()
    assert set(later["cmc_id"]) == {55, 56}
    assert later["effective_date"].eq(pd.Timestamp("2024-10-03")).all()
