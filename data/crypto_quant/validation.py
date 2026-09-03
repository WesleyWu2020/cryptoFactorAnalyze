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


def _parse_universe_timestamp(
    universe: pd.DataFrame,
    column: str,
    issues: list[ValidationIssue],
    *,
    allow_null: bool = False,
) -> pd.Series:
    parsed = pd.to_datetime(universe[column], errors="coerce", utc=True, format="mixed").dt.tz_localize(None).dt.normalize()
    invalid = parsed.isna() if not allow_null else parsed.isna() & universe[column].notna()
    for index in universe.index[invalid]:
        _issue(issues, "invalid_timestamp", f"column={column}, index={index!s}, value={universe.at[index, column]!s}")
    return parsed


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
    parsed_dates = {
        column: _parse_universe_timestamp(
            universe, column, issues, allow_null=column == "effective_end_date"
        )
        for column in ("decision_date", "effective_date", "effective_end_date")
        if column in universe
    }
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
            early = parsed_dates["effective_date"] < parsed_dates["decision_date"] + pd.Timedelta(days=1)
            if early.any():
                row = universe.loc[early].iloc[0]
                _issue(issues, "early_effective_date", f"decision_date={row['decision_date']!s}, effective_date={row['effective_date']!s}, symbol={row[symbols]!s}")
        if {"effective_date", "effective_end_date"}.issubset(universe.columns):
            starts = parsed_dates["effective_date"]
            ends = parsed_dates["effective_end_date"]
            invalid = starts.notna() & ends.notna() & (ends < starts)
            if invalid.any():
                row = universe.loc[invalid].iloc[0]
                _issue(issues, "invalid_effective_interval", f"effective_date={row['effective_date']!s}, effective_end_date={row['effective_end_date']!s}, symbol={row[symbols]!s}")
            intervals = universe.assign(_start=starts, _end=ends).sort_values([symbols, "_start"], kind="stable")
            for symbol, group in intervals.groupby(symbols, sort=False):
                previous_end = pd.NaT
                for _, row in group.iterrows():
                    if pd.notna(previous_end) and pd.notna(row["_start"]) and row["_start"] <= previous_end:
                        _issue(issues, "overlapping_universe", f"symbol={symbol!s}, effective_date={row['effective_date']!s}")
                        break
                    if pd.notna(row["_end"]):
                        previous_end = row["_end"] if pd.isna(previous_end) else max(previous_end, row["_end"])

        constituents = frames.get("cmc100_constituents", pd.DataFrame())
        if not constituents.empty and {"date", "cmc_id"}.issubset(constituents.columns) and "cmc_id" in universe:
            available = set(zip(pd.to_datetime(constituents["date"]).dt.normalize(), constituents["cmc_id"]))
            missing = [(row.get("decision_date"), row.get("cmc_id")) for index, row in universe.iterrows() if pd.notna(parsed_dates["decision_date"].loc[index]) and (parsed_dates["decision_date"].loc[index], row["cmc_id"]) not in available]
            if missing:
                _issue(issues, "missing_cmc_snapshot", f"decision_date={missing[0][0]!s}, cmc_id={missing[0][1]!s}")

        klines = frames.get("klines_daily", pd.DataFrame())
        if not klines.empty and {"date", "symbol"}.issubset(klines.columns):
            completed = klines.get("completed", pd.Series(True, index=klines.index)).map(lambda value: type(value) is bool and value) if "completed" in klines else pd.Series(True, index=klines.index)
            keys = set(zip(pd.to_datetime(klines["date"]).dt.normalize(), klines["symbol"], completed))
            for _, row in universe.iterrows():
                decision = parsed_dates["decision_date"].loc[row.name]
                if pd.isna(decision):
                    continue
                expected = (decision - pd.Timedelta(days=1), row[symbols])
                if not any(key[0] == expected[0] and key[1] == expected[1] and key[2] for key in keys):
                    _issue(issues, "missing_t_minus_one", f"decision_date={row['decision_date']!s}, symbol={row[symbols]!s}")
                    break


def _validate_funding(funding: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    if funding.empty or "funding_time" not in funding:
        return
    times = pd.to_datetime(funding["funding_time"], errors="coerce", utc=True, format="mixed")
    invalid = times.isna()
    if invalid.any():
        _issue(issues, "invalid_funding_timestamp", f"index={funding.index[invalid].tolist()[0]!s}, value={funding.loc[invalid].iloc[0]['funding_time']!s}")
        times = times[~invalid]
    if times.empty:
        return
    symbol_column = _symbol_column(funding)
    groups = funding.assign(_funding_time=times).groupby(symbol_column, sort=False) if symbol_column else [(None, funding.assign(_funding_time=times))]
    for symbol, group in groups:
        group_times = group["_funding_time"]
        if not group_times.is_monotonic_increasing:
            _issue(issues, "funding_not_monotonic", f"symbol={symbol!s}, first out-of-order funding_time near index={group.index[0]!s}")
            break


def _validate_panel(panel: pd.DataFrame, universe: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    if panel.empty:
        return
    if {"date", "binance_symbol"}.issubset(panel.columns):
        for timestamp, group in panel.groupby("date"):
            count = group["binance_symbol"].nunique()
            if count != 50:
                _issue(issues, "panel_universe_size", f"date={timestamp!s}, memberships={count}")
        if not universe.empty and {"effective_date", "effective_end_date"}.issubset(universe.columns):
            panel_dates = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
            universe_starts = pd.to_datetime(universe["effective_date"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None).dt.normalize()
            universe_ends = pd.to_datetime(universe["effective_end_date"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None).dt.normalize()
            universe_symbol = _symbol_column(universe)
            if universe_symbol:
                for index, row in panel.iterrows():
                    active = (universe[universe_symbol] == row["binance_symbol"]) & (universe_starts <= panel_dates.loc[index]) & (universe_ends.isna() | (universe_ends >= panel_dates.loc[index]))
                    if not active.any():
                        _issue(issues, "panel_outside_universe", f"date={row['date']!s}, symbol={row['binance_symbol']!s}")
                        break
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
    _validate_panel(
        frames.get("research_panel_daily", pd.DataFrame()),
        frames.get("universe_monthly", pd.DataFrame()),
        issues,
    )
    return ValidationReport(tuple(issues))


def validate_store(path: Path) -> ValidationReport:
    """Read a HDF5 store and validate all available tables and metadata."""
    path = Path(path)
    if not path.exists():
        return ValidationReport((ValidationIssue("error", "missing_store", f"store path does not exist: {path}"),))
    store = CryptoQuantStore(path)
    keys = store.keys()
    issues: list[ValidationIssue] = []
    if not keys:
        issues.append(ValidationIssue("error", "empty_store", f"store has no tables: {path}"))
    frames = {name: store.read(name) for name in TABLE_SPECS}
    for name in TABLE_SPECS:
        if name not in keys:
            issues.append(ValidationIssue("error", "missing_table", name))
    report = validate_frames(frames, store.read_metadata())
    return ValidationReport(tuple(issues) + tuple(report.issues))


__all__ = ["StoreValidationError", "ValidationIssue", "ValidationReport", "validate_frames", "validate_store"]
