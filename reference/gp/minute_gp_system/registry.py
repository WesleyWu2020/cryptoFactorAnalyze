from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


COMMON_ROOT = Path("/root/crypto-research/common")
SCRIPTS_ROOT = COMMON_ROOT / "scripts"
GP_ROOT = COMMON_ROOT / "gp"
SYSTEM_ROOT = GP_ROOT / "minute_gp_system"
ENGINES_ROOT = SYSTEM_ROOT / "engines"
UNIFIED_V2_ROOT = ENGINES_ROOT / "unified_v2"
RUNS_ROOT = SYSTEM_ROOT / "runs"
SUPERVISOR_RUN_ROOT = RUNS_ROOT / "supervisor"
CPU_RUNS_ROOT = SYSTEM_ROOT / "cpu_runs"
RAW_DATA_ROOT = COMMON_ROOT / "data" / "raw"
RAW_PARQUET_ROOT = RAW_DATA_ROOT / "parquet" / "binance_perp_1m"
RAW_CHECKPOINT_PATH = RAW_DATA_ROOT / "checkpoint" / "binance_perp_1m.json"
RAW_H5_PATH = RAW_DATA_ROOT / "h5" / "binance_perp_1m.h5"
RAW_DOWNLOAD_SCRIPT = SCRIPTS_ROOT / "download_perp_1m.py"
RAW_DAILY_UPDATE_SCRIPT = SCRIPTS_ROOT / "daily_update.py"
RAW_BUILD_H5_SCRIPT = SCRIPTS_ROOT / "build_h5_tensor.py"

BYBIT_PARQUET_ROOT = RAW_DATA_ROOT / "parquet"
BYBIT_CHECKPOINT_ROOT = RAW_DATA_ROOT / "checkpoint"
BYBIT_H5_ROOT = RAW_DATA_ROOT / "h5"
SERVER_SCRIPT_ROOT = SYSTEM_ROOT.parent.parent / "scripts"

BYBIT_CORE_PARQUET_ROOT = BYBIT_PARQUET_ROOT / "bybit_linear_1m_core"
BYBIT_CORE_CHECKPOINT_PATH = BYBIT_CHECKPOINT_ROOT / "bybit_linear_1m_core.json"
BYBIT_CORE_STATE_PATH = BYBIT_CHECKPOINT_ROOT / "bybit_linear_1m_core_state.json"
BYBIT_CORE_H5_PATH = BYBIT_H5_ROOT / "bybit_linear_1m_core.h5"
BYBIT_CORE_MANIFEST_PATH = BYBIT_H5_ROOT / "bybit_linear_1m_core.manifest.json"
BYBIT_CORE_SYNC_SCRIPT = SERVER_SCRIPT_ROOT / "run_bybit_linear_1m_core_sync.sh"
BYBIT_CORE_H5_BUILD_SCRIPT = SERVER_SCRIPT_ROOT / "run_bybit_linear_1m_h5_build.sh"

BYBIT_PRICE_STATE_PARQUET_ROOT = BYBIT_PARQUET_ROOT / "bybit_price_state_5m"
BYBIT_PRICE_STATE_CHECKPOINT_PATH = BYBIT_CHECKPOINT_ROOT / "bybit_price_state_5m.json"
BYBIT_PRICE_STATE_STATE_PATH = BYBIT_CHECKPOINT_ROOT / "bybit_price_state_5m_state.json"
BYBIT_PRICE_STATE_SYNC_SCRIPT = SERVER_SCRIPT_ROOT / "run_bybit_price_state_sync.sh"

BYBIT_OPEN_INTEREST_PARQUET_ROOT = BYBIT_PARQUET_ROOT / "bybit_open_interest_1h"
BYBIT_OPEN_INTEREST_CHECKPOINT_PATH = BYBIT_CHECKPOINT_ROOT / "bybit_open_interest_1h.json"
BYBIT_OPEN_INTEREST_STATE_PATH = BYBIT_CHECKPOINT_ROOT / "bybit_open_interest_1h_state.json"

BYBIT_FUNDING_PARQUET_ROOT = BYBIT_PARQUET_ROOT / "bybit_funding_8h"
BYBIT_FUNDING_CHECKPOINT_PATH = BYBIT_CHECKPOINT_ROOT / "bybit_funding_8h.json"
BYBIT_FUNDING_STATE_PATH = BYBIT_CHECKPOINT_ROOT / "bybit_funding_8h_state.json"
BYBIT_STATE_FIELDS_SYNC_SCRIPT = SERVER_SCRIPT_ROOT / "run_bybit_state_fields_sync.sh"

BYBIT_ORDER_FLOW_PARQUET_ROOT = BYBIT_PARQUET_ROOT / "bybit_order_flow_1m"
BYBIT_ORDER_FLOW_CHECKPOINT_PATH = BYBIT_CHECKPOINT_ROOT / "bybit_order_flow_1m.json"
BYBIT_ORDER_FLOW_SYNC_SCRIPT = SERVER_SCRIPT_ROOT / "run_bybit_order_flow_archive_sync.sh"

BYBIT_UNIFIED_H5_PATH = BYBIT_H5_ROOT / "bybit_linear_1m_unified.h5"
BYBIT_UNIFIED_MANIFEST_PATH = BYBIT_H5_ROOT / "bybit_linear_1m_unified.manifest.json"
BYBIT_UNIFIED_H5_BUILD_SCRIPT = SERVER_SCRIPT_ROOT / "run_bybit_linear_unified_h5_build.sh"

ACTIVE_MARKET_DATA_ALIAS = "bybit_linear_core"
ACTIVE_MINING_DATA_ALIAS = "bybit_linear_unified"
FALLBACK_MINING_DATA_ALIAS = "bybit_linear_core"


@dataclass(frozen=True)
class SourceSpec:
    alias: str
    role: str
    description: str
    module: str
    discovery_roots: tuple[str, ...] = ()
    default_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataSourceSpec:
    alias: str
    description: str
    exchange: str
    production_enabled: bool = True
    raw_frequency: str = "1m"
    parquet_root: str | None = None
    checkpoint_path: str | None = None
    state_path: str | None = None
    h5_path: str | None = None
    manifest_path: str | None = None
    sync_script: str | None = None
    h5_build_script: str | None = None
    component_aliases: tuple[str, ...] = ()


def get_sources() -> dict[str, SourceSpec]:
    return {
        "unified_v2": SourceSpec(
            alias="unified_v2",
            role="generator",
            description="Production minute-GP engine aligned to live perp_4h trading mouth.",
            module="gp.minute_gp_system.engines.unified_v2.run",
            discovery_roots=(str(SYSTEM_ROOT / "engines" / "unified_v2"), str(RUNS_ROOT)),
            default_args={
                "start_date": "20230101",
                "end_date": "20260131",
                "train_end_date": "20241231",
                "period_minutes": 240,
                "population": 500,
                "generations": 15,
                "backend": "gpu",
                "chunk_periods": 300,
                "eval_batch_size": 48,
                "cache_dtype": "float16",
                "top_quantile": 0.1,
                "trading_cost": 0.0005,
                "result_target_min": 32,
                "result_target_max": 48,
                "candidate_pool_size": 96,
            },
        ),
        "framework_evaluator": SourceSpec(
            alias="framework_evaluator",
            role="evaluator",
            description="Formal replay evaluator for top GP candidates.",
            module="gp.minute_gp_system.engines.unified_v2.evaluate_with_framework",
            default_args={
                "profile": "perp_1d",
                "cost": 0.0015,
            },
        ),
    }


def get_data_sources() -> dict[str, DataSourceSpec]:
    return {
        "legacy_binance_raw": DataSourceSpec(
            alias="legacy_binance_raw",
            description="Frozen Binance raw research store kept only for legacy backfills and compatibility.",
            exchange="binance",
            production_enabled=False,
            raw_frequency="1m",
            parquet_root=str(RAW_PARQUET_ROOT),
            checkpoint_path=str(RAW_CHECKPOINT_PATH),
            h5_path=str(RAW_H5_PATH),
            sync_script=str(RAW_DAILY_UPDATE_SCRIPT),
            h5_build_script=str(RAW_BUILD_H5_SCRIPT),
        ),
        "bybit_linear_core": DataSourceSpec(
            alias="bybit_linear_core",
            description="Bybit linear core market data (1m OHLCVT) with aligned H5 snapshot.",
            exchange="bybit",
            raw_frequency="1m",
            parquet_root=str(BYBIT_CORE_PARQUET_ROOT),
            checkpoint_path=str(BYBIT_CORE_CHECKPOINT_PATH),
            state_path=str(BYBIT_CORE_STATE_PATH),
            h5_path=str(BYBIT_CORE_H5_PATH),
            manifest_path=str(BYBIT_CORE_MANIFEST_PATH),
            sync_script=str(BYBIT_CORE_SYNC_SCRIPT),
            h5_build_script=str(BYBIT_CORE_H5_BUILD_SCRIPT),
        ),
        "bybit_price_state_5m": DataSourceSpec(
            alias="bybit_price_state_5m",
            description="Bybit 5m mark/index/premium price-state store for high-liquidity universe.",
            exchange="bybit",
            raw_frequency="5m",
            parquet_root=str(BYBIT_PRICE_STATE_PARQUET_ROOT),
            checkpoint_path=str(BYBIT_PRICE_STATE_CHECKPOINT_PATH),
            state_path=str(BYBIT_PRICE_STATE_STATE_PATH),
            sync_script=str(BYBIT_PRICE_STATE_SYNC_SCRIPT),
        ),
        "bybit_open_interest_1h": DataSourceSpec(
            alias="bybit_open_interest_1h",
            description="Bybit hourly open-interest state store.",
            exchange="bybit",
            raw_frequency="1h",
            parquet_root=str(BYBIT_OPEN_INTEREST_PARQUET_ROOT),
            checkpoint_path=str(BYBIT_OPEN_INTEREST_CHECKPOINT_PATH),
            state_path=str(BYBIT_OPEN_INTEREST_STATE_PATH),
            sync_script=str(BYBIT_STATE_FIELDS_SYNC_SCRIPT),
        ),
        "bybit_funding_8h": DataSourceSpec(
            alias="bybit_funding_8h",
            description="Bybit funding-history state store.",
            exchange="bybit",
            raw_frequency="8h",
            parquet_root=str(BYBIT_FUNDING_PARQUET_ROOT),
            checkpoint_path=str(BYBIT_FUNDING_CHECKPOINT_PATH),
            state_path=str(BYBIT_FUNDING_STATE_PATH),
            sync_script=str(BYBIT_STATE_FIELDS_SYNC_SCRIPT),
        ),
        "bybit_order_flow_1m": DataSourceSpec(
            alias="bybit_order_flow_1m",
            description="Bybit historical public trades aggregated to 1m taker/order-flow fields.",
            exchange="bybit",
            raw_frequency="1m",
            parquet_root=str(BYBIT_ORDER_FLOW_PARQUET_ROOT),
            checkpoint_path=str(BYBIT_ORDER_FLOW_CHECKPOINT_PATH),
            sync_script=str(BYBIT_ORDER_FLOW_SYNC_SCRIPT),
        ),
        "bybit_linear_unified": DataSourceSpec(
            alias="bybit_linear_unified",
            description="Future unified Bybit mining snapshot (core + price_state + OI + funding on one master axis).",
            exchange="bybit",
            raw_frequency="1m_master",
            h5_path=str(BYBIT_UNIFIED_H5_PATH),
            manifest_path=str(BYBIT_UNIFIED_MANIFEST_PATH),
            h5_build_script=str(BYBIT_UNIFIED_H5_BUILD_SCRIPT),
            component_aliases=(
                "bybit_linear_core",
                "bybit_price_state_5m",
                "bybit_open_interest_1h",
                "bybit_funding_8h",
                "bybit_order_flow_1m",
            ),
        ),
    }


def get_data_source(alias: str) -> DataSourceSpec:
    sources = get_data_sources()
    if alias not in sources:
        raise ValueError(f"Unknown data source alias: {alias}")
    return sources[alias]


def resolve_data_source(alias: str | None = None) -> DataSourceSpec:
    target_alias = alias or ACTIVE_MINING_DATA_ALIAS
    target = get_data_source(target_alias)
    if target.h5_path and Path(target.h5_path).exists():
        return target
    if target_alias == ACTIVE_MINING_DATA_ALIAS:
        fallback = get_data_source(FALLBACK_MINING_DATA_ALIAS)
        if fallback.h5_path and Path(fallback.h5_path).exists():
            return fallback
    return target


def resolve_default_h5_path(alias: str | None = None) -> str:
    source = resolve_data_source(alias)
    if not source.h5_path:
        raise ValueError(f"Data source has no H5 path: {source.alias}")
    return str(source.h5_path)


def resolve_default_manifest_path(alias: str | None = None) -> str | None:
    source = resolve_data_source(alias)
    return source.manifest_path


def list_pareto_results(
    *,
    source_alias: str | None = None,
    limit_per_source: int = 20,
    modified_since_epoch: int | None = None,
) -> dict[str, Any]:
    if limit_per_source <= 0:
        raise ValueError("limit_per_source must be positive")

    sources = get_sources()
    aliases = [source_alias] if source_alias else [
        alias
        for alias, spec in sources.items()
        if spec.role in {"generator", "supervisor"}
    ]
    payload: list[dict[str, Any]] = []
    for alias in aliases:
        if alias not in sources:
            raise ValueError(f"Unknown source alias: {alias}")
        spec = sources[alias]
        files: list[dict[str, Any]] = []
        for root in spec.discovery_roots:
            path = Path(root)
            if not path.exists():
                continue
            for pareto_path in sorted(
                path.glob("**/pareto_results.json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )[:limit_per_source]:
                stat = pareto_path.stat()
                if modified_since_epoch is not None and int(stat.st_mtime) < int(modified_since_epoch):
                    continue
                files.append(
                    {
                        "remote_path": str(pareto_path),
                        "size_bytes": stat.st_size,
                        "modified_at_epoch": int(stat.st_mtime),
                    }
                )
        payload.append(
            {
                "source_alias": alias,
                "role": spec.role,
                "description": spec.description,
                "files": files[:limit_per_source],
            }
        )
    return {
        "system_root": str(SYSTEM_ROOT),
        "sources": payload,
    }


def get_source_summary() -> dict[str, Any]:
    active_data = resolve_data_source()
    return {
        "system_root": str(SYSTEM_ROOT),
        "active_market_data_alias": ACTIVE_MARKET_DATA_ALIAS,
        "active_mining_data_alias": ACTIVE_MINING_DATA_ALIAS,
        "resolved_mining_data_alias": active_data.alias,
        "resolved_default_h5_path": active_data.h5_path,
        "sources": [asdict(spec) for spec in get_sources().values()],
        "data_sources": [asdict(spec) for spec in get_data_sources().values()],
    }
