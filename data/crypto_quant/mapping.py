"""Deterministic CMC constituent to Binance futures contract mapping."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class MappingRules:
    version: str
    stablecoin_symbols: frozenset[str]
    wrapper_symbols: frozenset[str]
    overrides: Sequence[Mapping[str, object]]
    blocked_cmc_ids: frozenset[int]


def _symbols(values: object) -> frozenset[str]:
    if not isinstance(values, list):
        raise ValueError("mapping rule symbols must be a list")
    return frozenset(str(value).strip().upper() for value in values)


def load_mapping_rules(path: Path) -> MappingRules:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("version"), str):
        raise ValueError("mapping rules require a version")
    overrides = payload.get("overrides", [])
    if not isinstance(overrides, list) or not all(isinstance(item, dict) for item in overrides):
        raise ValueError("mapping rule overrides must be a list of objects")
    blocked = payload.get("blocked_cmc_ids", [])
    if not isinstance(blocked, list):
        raise ValueError("blocked_cmc_ids must be a list")
    return MappingRules(
        version=payload["version"],
        stablecoin_symbols=_symbols(payload.get("stablecoin_symbols", [])),
        wrapper_symbols=_symbols(payload.get("wrapper_symbols", [])),
        overrides=tuple(overrides),
        blocked_cmc_ids=frozenset(int(value) for value in blocked),
    )


def is_excluded_asset(cmc_id: int, symbol: str, rules: MappingRules) -> bool:
    normalized = str(symbol).strip().upper()
    return cmc_id in rules.blocked_cmc_ids or normalized in rules.stablecoin_symbols or normalized in rules.wrapper_symbols


def _as_date(value: object) -> date | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date()


def _override_for(cmc_id: int, decision_date: date, rules: MappingRules) -> Mapping[str, object] | None:
    matches = []
    for override in rules.overrides:
        if int(override.get("cmc_id", -1)) != cmc_id:
            continue
        start = _as_date(override.get("valid_from"))
        end = _as_date(override.get("valid_to"))
        if (start is None or decision_date >= start) and (end is None or decision_date <= end):
            matches.append(override)
    if len(matches) > 1:
        raise ValueError(f"multiple valid overrides for cmc_id={cmc_id}")
    return matches[0] if matches else None


def _empty_frames(exchange_info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_columns = ["cmc_id", "cmc_symbol", "cmc_name", "mapping_source", "valid_from", "valid_to"] + list(exchange_info.columns)
    return pd.DataFrame(columns=list(dict.fromkeys(mapping_columns))), pd.DataFrame(columns=["cmc_id", "cmc_symbol", "cmc_name", "decision_date", "issue"])


def _blank_contract(exchange_info: pd.DataFrame) -> dict[str, object]:
    base = {column: pd.NA for column in exchange_info.columns}
    for column in ("onboard_date", "fetched_at_utc"):
        if column in base:
            base[column] = pd.NaT
    return base


def build_contract_mappings(
    constituents: pd.DataFrame,
    exchange_info: pd.DataFrame,
    rules: MappingRules,
    historical_probe: Callable[[str, date], bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mappings, issues = _empty_frames(exchange_info)
    if constituents.empty:
        return mappings, issues

    rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    ordered = constituents.sort_values(["date", "cmc_id"], kind="stable")
    for cmc_id, group in ordered.groupby("cmc_id", sort=True):
        first = group.iloc[0]
        symbol = str(first["symbol"]).strip().upper()
        decision_date = pd.Timestamp(first["date"]).date()
        issue_base = {"cmc_id": int(cmc_id), "cmc_symbol": symbol, "cmc_name": first.get("name", ""), "decision_date": decision_date}
        if is_excluded_asset(int(cmc_id), symbol, rules):
            continue

        override = _override_for(int(cmc_id), decision_date, rules)
        if override is not None:
            contract = exchange_info.loc[exchange_info["binance_symbol"].str.upper() == str(override["binance_symbol"]).upper()]
            base = contract.iloc[0].to_dict() if len(contract) == 1 else _blank_contract(exchange_info)
            if len(contract) != 1:
                base.update({"binance_symbol": override["binance_symbol"], "base_asset": str(override["binance_symbol"]).removesuffix("USDT"), "quote_asset": "USDT", "contract_type": "PERPETUAL", "status": "UNKNOWN"})
            for key, value in override.items():
                if key in exchange_info.columns:
                    base[key] = value
            base.update({"cmc_id": int(cmc_id), "cmc_symbol": symbol, "cmc_name": first.get("name", ""), "mapping_source": "explicit_override", "valid_from": pd.Timestamp(override.get("valid_from", decision_date)), "valid_to": pd.to_datetime(override.get("valid_to"), errors="coerce")})
            rows.append(base)
            continue

        candidates = exchange_info.loc[exchange_info["base_asset"].astype(str).str.upper() == symbol]
        if len(candidates) > 1:
            issue_rows.append({**issue_base, "issue": "ambiguous_current_match"})
            continue
        if len(candidates) == 1:
            onboard_date = _as_date(candidates.iloc[0].get("onboard_date"))
            if onboard_date is not None and decision_date >= onboard_date:
                base = candidates.iloc[0].to_dict()
                base.update({"cmc_id": int(cmc_id), "cmc_symbol": symbol, "cmc_name": first.get("name", ""), "mapping_source": "current_exchange_info", "valid_from": base.get("onboard_date", pd.NaT), "valid_to": pd.NaT})
                rows.append(base)
                continue

        contract_symbol = f"{symbol}USDT"
        if historical_probe(contract_symbol, decision_date):
            base = {"binance_symbol": contract_symbol, "base_asset": symbol, "quote_asset": "USDT", "contract_type": "PERPETUAL", "onboard_date": pd.NaT, "status": "UNKNOWN", "fetched_at_utc": pd.NaT}
            base.update({"cmc_id": int(cmc_id), "cmc_symbol": symbol, "cmc_name": first.get("name", ""), "mapping_source": "historical_kline_probe", "valid_from": pd.Timestamp(decision_date), "valid_to": pd.NaT})
            rows.append(base)
        else:
            issue_rows.append({**issue_base, "issue": "unresolved"})

    if rows:
        mappings = pd.DataFrame(rows)
        mappings = mappings[[*dict.fromkeys(["cmc_id", "cmc_symbol", "cmc_name", "mapping_source", "valid_from", "valid_to", *exchange_info.columns])]]
        mappings = mappings.sort_values("cmc_id").reset_index(drop=True)
    if issue_rows:
        issues = pd.DataFrame(issue_rows, columns=issues.columns)
    return mappings, issues
