import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.crypto_quant.mapping import (
    MappingRules,
    build_contract_mappings,
    load_mapping_rules,
)


@pytest.fixture
def rules():
    return load_mapping_rules(Path("data/crypto_quant_rules.json"))


@pytest.fixture
def constituents():
    return pd.DataFrame(
        [
            {"date": date(2026, 9, 1), "cmc_id": 1, "symbol": "BTC", "name": "Bitcoin", "weight": 0.5},
            {"date": date(2026, 9, 1), "cmc_id": 2, "symbol": "USDT", "name": "Tether", "weight": 0.2},
            {"date": date(2026, 9, 1), "cmc_id": 3, "symbol": "SHIB", "name": "Shiba Inu", "weight": 0.1},
        ]
    )


@pytest.fixture
def exchange_info():
    return pd.DataFrame(
        [
            {"binance_symbol": "BTCUSDT", "base_asset": "BTC", "quote_asset": "USDT", "contract_type": "PERPETUAL", "onboard_date": pd.Timestamp("2020-01-01"), "status": "TRADING", "fetched_at_utc": pd.Timestamp("2026-09-03")},
            {"binance_symbol": "SHIBUSDT", "base_asset": "SHIB", "quote_asset": "USDT", "contract_type": "PERPETUAL", "onboard_date": pd.Timestamp("2021-05-01"), "status": "TRADING", "fetched_at_utc": pd.Timestamp("2026-09-03")},
            {"binance_symbol": "1000SHIBUSDT", "base_asset": "1000SHIB", "quote_asset": "USDT", "contract_type": "PERPETUAL", "onboard_date": pd.Timestamp("2022-01-01"), "status": "TRADING", "fetched_at_utc": pd.Timestamp("2026-09-03")},
        ]
    )


def test_mapping_is_keyed_by_cmc_id_and_excludes_stablecoins(rules, constituents, exchange_info):
    mappings, issues = build_contract_mappings(constituents, exchange_info, rules, lambda *_: False)
    assert 1 in set(mappings["cmc_id"])
    assert "USDT" not in set(mappings["cmc_symbol"])
    assert mappings.set_index("cmc_id").loc[1, "binance_symbol"] == "BTCUSDT"
    assert issues.empty


def test_override_wins_and_records_mapping_source(rules, exchange_info):
    constituents = pd.DataFrame([{"date": date(2026, 9, 1), "cmc_id": 999, "symbol": "SHIB", "name": "Shiba Inu", "weight": 1.0}])
    override = {"cmc_id": 999, "binance_symbol": "1000SHIBUSDT", "valid_from": "2025-01-01"}
    rules_with_override = MappingRules(rules.version, rules.stablecoin_symbols, rules.wrapper_symbols, [override], rules.blocked_cmc_ids)
    mappings, _ = build_contract_mappings(constituents, exchange_info, rules_with_override, lambda *_: False)
    row = mappings.loc[mappings["cmc_id"] == 999].iloc[0]
    assert row["binance_symbol"] == "1000SHIBUSDT"
    assert row["mapping_source"] == "explicit_override"
    assert row["valid_from"] == pd.Timestamp("2025-01-01")


def test_override_missing_from_current_exchange_info_keeps_stable_contract_schema(rules, exchange_info):
    constituents = pd.DataFrame([{"date": date(2026, 9, 1), "cmc_id": 999, "symbol": "SHIB", "name": "Shiba Inu", "weight": 1.0}])
    override = {"cmc_id": 999, "binance_symbol": "ARCHIVEDSHIBUSDT", "valid_from": "2025-01-01"}
    rules_with_override = MappingRules(rules.version, rules.stablecoin_symbols, rules.wrapper_symbols, [override], rules.blocked_cmc_ids)
    mappings, issues = build_contract_mappings(constituents, exchange_info, rules_with_override, lambda *_: False)
    assert issues.empty
    assert list(mappings.columns) == ["cmc_id", "cmc_symbol", "cmc_name", "mapping_source", "valid_from", "valid_to", *exchange_info.columns]
    assert mappings.iloc[0]["binance_symbol"] == "ARCHIVEDSHIBUSDT"
    assert pd.isna(mappings.iloc[0]["onboard_date"])


def test_same_symbol_collision_is_unresolved(rules, exchange_info):
    constituents = pd.DataFrame([{"date": date(2026, 9, 1), "cmc_id": 7, "symbol": "BTC", "name": "Bitcoin", "weight": 1.0}])
    duplicate = exchange_info.iloc[[0]].copy()
    duplicate["binance_symbol"] = "BTCUSDT_2"
    exchange_info = pd.concat([exchange_info, duplicate], ignore_index=True)
    mappings, issues = build_contract_mappings(constituents, exchange_info, rules, lambda *_: False)
    assert mappings.empty
    assert issues.iloc[0]["issue"] == "ambiguous_current_match"


def test_same_symbol_different_cmc_ids_remain_distinct(rules, exchange_info):
    constituents = pd.DataFrame(
        [
            {"date": date(2026, 9, 1), "cmc_id": 10, "symbol": "BTC", "name": "Bitcoin A", "weight": 0.5},
            {"date": date(2026, 9, 1), "cmc_id": 11, "symbol": "BTC", "name": "Bitcoin B", "weight": 0.5},
        ]
    )
    mappings, issues = build_contract_mappings(constituents, exchange_info, rules, lambda *_: False)
    assert list(mappings["cmc_id"]) == [10, 11]
    assert set(mappings["binance_symbol"]) == {"BTCUSDT"}
    assert issues.empty


def test_exclusion_is_exact_not_substring(rules, exchange_info):
    constituents = pd.DataFrame([{"date": date(2026, 9, 1), "cmc_id": 12, "symbol": "USDTX", "name": "USDTX", "weight": 1.0}])
    exchange_info = pd.concat([exchange_info, pd.DataFrame([{
        "binance_symbol": "USDTXUSDT", "base_asset": "USDTX", "quote_asset": "USDT",
        "contract_type": "PERPETUAL", "onboard_date": pd.Timestamp("2026-01-01"),
        "status": "TRADING", "fetched_at_utc": pd.Timestamp("2026-09-03")
    }])], ignore_index=True)
    mappings, issues = build_contract_mappings(constituents, exchange_info, rules, lambda *_: False)
    assert mappings.iloc[0]["cmc_symbol"] == "USDTX"
    assert issues.empty


def test_historical_probe_confirms_contract_absent_from_current_exchange_info(rules, exchange_info):
    old_constituent = pd.DataFrame([{"date": date(2020, 1, 2), "cmc_id": 8, "symbol": "OLD", "name": "Old", "weight": 1.0}])
    mappings, issues = build_contract_mappings(old_constituent, exchange_info, rules, lambda symbol, _: symbol == "OLDUSDT")
    assert mappings.iloc[0]["mapping_source"] == "historical_kline_probe"
    assert mappings.iloc[0]["valid_from"] == pd.Timestamp("2020-01-02")
    assert issues.empty


def test_current_match_is_unresolved_when_first_observed_date_precedes_onboard_date(rules, exchange_info):
    constituents = pd.DataFrame([{"date": date(2019, 12, 31), "cmc_id": 13, "symbol": "BTC", "name": "Bitcoin", "weight": 1.0}])
    mappings, issues = build_contract_mappings(constituents, exchange_info, rules, lambda symbol, _: False)
    assert mappings.empty
    assert issues.iloc[0]["issue"] == "unresolved"


def test_unresolved_issue_is_emitted(rules, exchange_info):
    constituents = pd.DataFrame([{"date": date(2026, 9, 1), "cmc_id": 88, "symbol": "MISSING", "name": "Missing", "weight": 1.0}])
    mappings, issues = build_contract_mappings(constituents, exchange_info, rules, lambda *_: False)
    assert mappings.empty
    assert issues.iloc[0]["cmc_id"] == 88
    assert issues.iloc[0]["issue"] == "unresolved"


def test_empty_cmc_text_fields_remain_unresolved(rules, exchange_info):
    constituents = pd.DataFrame([{"date": date(2024, 1, 1), "cmc_id": 28683, "symbol": "", "name": "", "weight": 1.0}])

    mappings, issues = build_contract_mappings(constituents, exchange_info, rules, lambda *_: False)

    assert mappings.empty
    assert issues.iloc[0]["cmc_id"] == 28683
    assert issues.iloc[0]["cmc_symbol"] == ""
    assert issues.iloc[0]["cmc_name"] == ""
    assert issues.iloc[0]["issue"] == "unresolved"


def test_empty_cmc_symbol_does_not_probe_a_fake_contract(rules, exchange_info):
    constituents = pd.DataFrame([{"date": date(2024, 1, 1), "cmc_id": 28683, "symbol": "", "name": "", "weight": 1.0}])
    probed_symbols = []

    def historical_probe(symbol, _):
        probed_symbols.append(symbol)
        return False

    mappings, issues = build_contract_mappings(constituents, exchange_info, rules, historical_probe)

    assert mappings.empty
    assert issues.iloc[0]["cmc_symbol"] == ""
    assert issues.iloc[0]["cmc_name"] == ""
    assert issues.iloc[0]["issue"] == "unresolved"
    assert probed_symbols == []


def test_rules_are_versioned_and_case_insensitive(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"version": "v", "stablecoin_symbols": ["usdt"], "wrapper_symbols": ["wbtc"], "overrides": [], "blocked_cmc_ids": [9]}))
    rules = load_mapping_rules(path)
    assert rules.version == "v"
    assert rules.stablecoin_symbols == frozenset({"USDT"})
    assert rules.wrapper_symbols == frozenset({"WBTC"})


def test_rules_reject_reverse_override_validity_window(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({
        "version": "v",
        "stablecoin_symbols": [],
        "wrapper_symbols": [],
        "overrides": [{"cmc_id": 1, "binance_symbol": "BTCUSDT", "valid_from": "2026-09-02", "valid_to": "2026-09-01"}],
        "blocked_cmc_ids": [],
    }))
    with pytest.raises(ValueError, match="valid_from.*valid_to"):
        load_mapping_rules(path)
