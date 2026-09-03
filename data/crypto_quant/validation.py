"""Storage and point-in-time invariant validation for the crypto-quant store."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import pandas as pd

from .schemas import TABLE_SPECS
from .store import CryptoQuantStore


class StoreValidationError(ValueError):
    """Raised when a store contains one or more error-level issues."""


@dataclass(frozen=True)
class ValidationIssue:
    level: Literal["error", "warning"]
    code: str
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    issues: Sequence[ValidationIssue]

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def raise_for_errors(self) -> None:
        errors = [issue for issue in self.issues if issue.level == "error"]
        if errors:
            raise StoreValidationError(
                "; ".join(f"{issue.code}: {issue.detail}" for issue in errors)
            )


def _timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _symbol_column(frame: pd.DataFrame) -> str | None:
    if "binance_symbol" in frame:
        return "binance_symbol"
    if "symbol" in frame:
        return "symbol"
    return None


def _issue(issues: list[ValidationIssue], code: str, detail: object, level: Literal["error", "warning"] = "error") -> None:
    issues.append(ValidationIssue(level, code, str(detail)))


def _stablecoins(metadata: Mapping[str, object]) -> set[str]:
    values = metadata.get("stablecoin_symbols", [])
    if isinstance(metadata.get("mapping_rules"), Mapping):
        values = list(values) + list(metadata["mapping_rules"].get("stablecoin_symbols", []))
    return {str(value).upper() for value in values}


def _validate_duplicate_keys(frames: Mapping[str, pd.DataFrame], issues: list[ValidationIssue]) -> None:
    for name, frame in frames.items():
        spec = TABLE_SPECS.get(name)
        if spec is None or frame.empty or not set(spec.key).issubset(frame.columns):
            continue
        duplicated = frame[frame.duplicated(list(spec.key), keep=False)]
        if not duplicated.empty:
            key = tuple(duplicated.iloc[0][column] for column in spec.key)
            _issue(issues, "duplicate_key", f"{name} key={key!r}")


def _validate_klines(klines: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    if klines.empty:
        return
    for column in ("volume", "quote_volume", "quote_asset_volume", "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"):
        if column in klines:
            bad = pd.to_numeric(klines[column], errors="coerce") < 0
            if bad.any():
                row = klines.loc[bad].iloc[0]
                _issue(issues, "negative_volume", f"date={row.get('date')!s}, symbol={row.get('symbol', row.get('binance_symbol'))!s}, column={column}")
                break
    required = [column for column in ("open", "high", "low", "close") if column in klines]
    if len(required) == 4:
        values = klines[required].apply(pd.to_numeric, errors="coerce")
        bad = (values.high < values[["open", "close"]].max(axis=1)) | (values.low > values[["open", "close"]].min(axis=1))
        if bad.any():
            row = klines.loc[bad].iloc[0]
            _issue(issues, "invalid_ohlc", f"date={row.get('date')!s}, symbol={row.get('symbol', row.get('binance_symbol'))!s}")


def _validate_universe(frames: Mapping[str, pd.DataFrame], metadata: Mapping[str, object], issues: list[ValidationIssue]) -> None:
    universe = frames.get("universe_monthly", pd.DataFrame())
    if universe.empty:
        return
    date_column = "decision_date" if "decision_date" in universe else "date"
    sizes = universe.groupby(date_column, dropna=False).size()
    for decision_date, size in sizes.items():
        if size != 50:
            _issue(issues, "universe_size", f"decision_date={decision_date!s}, rows={size}")
    symbols = _symbol_column(universe)
    excluded = _stablecoins(metadata)
    if symbols:
        symbol_values = universe[symbols].astype(str).str.upper()
        cmc_values = universe.get("cmc_symbol", pd.Series(index=universe.index, dtype=object)).astype(str).str.upper()
        bad = symbol_values.isin(excluded) | cmc_values.isin(excluded)
        if bad.any():
            row = universe.loc[bad].iloc[0]
            _issue(issues, "excluded_asset", f"decision_date={row.get('decision_date')!s}, symbol={row[symbols]!s}")
        if "effective_date" in universe and "decision_date" in universe:
            early = pd.to_datetime(universe["effective_date"], errors="coerce").dt.normalize() < pd.to_datetime(universe["decision_date"], errors="coerce").dt.normalize() + pd.Timedelta(days=1)
            if early.any():
                row = universe.loc[early].iloc[0]
                _issue(issues, "early_effective_date", f"decision_date={row['decision_date']!s}, effective_date={row['effective_date']!s}, symbol={row[symbols]!s}")

        constituents = frames.get("cmc100_constituents", pd.DataFrame())
        if not constituents.empty and {"date", "cmc_id"}.issubset(constituents.columns) and "cmc_id" in universe:
            available = set(zip(pd.to_datetime(constituents["date"]).dt.normalize(), constituents["cmc_id"]))
            missing = [(row.get("decision_date"), row.get("cmc_id")) for _, row in universe.iterrows() if (_timestamp(row["decision_date"]), row["cmc_id"]) not in available]
            if missing:
                _issue(issues, "missing_cmc_snapshot", f"decision_date={missing[0][0]!s}, cmc_id={missing[0][1]!s}")

        klines = frames.get("klines_daily", pd.DataFrame())
        if not klines.empty and {"date", "symbol"}.issubset(klines.columns):
            completed = klines.get("completed", pd.Series(True, index=klines.index)).map(lambda value: type(value) is bool and value) if "completed" in klines else pd.Series(True, index=klines.index)
            keys = set(zip(pd.to_datetime(klines["date"]).dt.normalize(), klines["symbol"], completed))
            for _, row in universe.iterrows():
                expected = (_timestamp(row["decision_date"]) - pd.Timedelta(days=1), row[symbols])
                if not any(key[0] == expected[0] and key[1] == expected[1] and key[2] for key in keys):
                    _issue(issues, "missing_t_minus_one", f"decision_date={row['decision_date']!s}, symbol={row[symbols]!s}")
                    break


def _validate_funding(funding: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    if funding.empty or "funding_time" not in funding:
        return
    times = pd.to_datetime(funding["funding_time"], errors="coerce", utc=True)
    symbol_column = _symbol_column(funding)
    groups = funding.assign(_funding_time=times).groupby(symbol_column, sort=False) if symbol_column else [(None, funding.assign(_funding_time=times))]
    for symbol, group in groups:
        group_times = group["_funding_time"]
        if not group_times.is_monotonic_increasing:
            _issue(issues, "funding_not_monotonic", f"symbol={symbol!s}, first out-of-order funding_time near index={group.index[0]!s}")
            break


def _validate_panel(panel: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    if panel.empty:
        return
    if {"date", "binance_symbol"}.issubset(panel.columns):
        for timestamp, group in panel.groupby("date"):
            count = group["binance_symbol"].nunique()
            if count != 50:
                _issue(issues, "panel_universe_size", f"date={timestamp!s}, memberships={count}")
    for column, code in (("has_complete_kline", "incomplete_kline"), ("has_complete_funding", "incomplete_funding")):
        if column in panel and (~panel[column].fillna(False).astype(bool)).any():
            first = pd.to_datetime(panel.loc[~panel[column].fillna(False).astype(bool), "date"]).min()
            _issue(issues, code, f"first incomplete date={first!s}", "warning")


def validate_frames(frames: Mapping[str, pd.DataFrame], metadata: Mapping[str, object]) -> ValidationReport:
    """Validate normalized or builder-level frames without mutating them."""
    issues: list[ValidationIssue] = []
    _validate_duplicate_keys(frames, issues)
    _validate_klines(frames.get("klines_daily", pd.DataFrame()), issues)
    _validate_universe(frames, metadata, issues)
    _validate_funding(frames.get("funding_events", pd.DataFrame()), issues)
    _validate_panel(frames.get("research_panel_daily", pd.DataFrame()), issues)
    return ValidationReport(tuple(issues))


def validate_store(path: Path) -> ValidationReport:
    """Read a HDF5 store and validate all available tables and metadata."""
    store = CryptoQuantStore(Path(path))
    frames = {name: store.read(name) for name in TABLE_SPECS}
    return validate_frames(frames, store.read_metadata())


__all__ = ["StoreValidationError", "ValidationIssue", "ValidationReport", "validate_frames", "validate_store"]
