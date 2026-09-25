from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


COMMON_ROOT = Path("/root/crypto-research/common")
DEFAULT_FAMILY_FEEDBACK_PATH = (
    COMMON_ROOT
    / "data"
    / "factor_platform"
    / "strategy_factory_artifacts"
    / "minute_gp_system"
    / "family_feedback"
    / "latest.json"
)

_PRICE_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "mid_price",
    "typical_price",
    "vwap_proxy",
}
_PRICE_STATE_FIELDS = {
    "body",
    "upper_shadow",
    "lower_shadow",
    "close_position",
}
_TREND_FIELDS = {"returns", "cum_return", "am_return", "pm_return", "am_pm_return_diff"}
_VOLATILITY_FIELDS = {
    "return_abs",
    "price_range_pct",
    "spread",
    "rv_5d",
    "rv_20d",
    "rv_ratio_5_20",
    "vol_of_vol_20d",
    "idio_vol_btc_20d",
}
_LIQUIDITY_FIELDS = {
    "volume",
    "log_volume",
    "turnover",
    "dollar_volume_rank",
    "volume_intensity",
    "late_volume_share",
    "volume_burst_share",
    "amihud_illiq",
    "volume_zscore_60d",
}
_FLOW_FIELDS = {
    "money_flow",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "taker_buy_ratio",
    "taker_quote_ratio",
    "avg_trade_size",
    "taker_pressure",
    "xbinance_quote_volume",
    "xbinance_trade_count",
    "xbinance_taker_buy_volume",
    "xbinance_taker_buy_quote_volume",
    "xbinance_taker_quote_ratio",
    "xbinance_avg_trade_size",
    "xbinance_taker_pressure",
}
_LEVERAGE_FIELDS = {
    "funding",
    "funding_abs",
    "open_interest",
    "oi_change_intraday",
    "oi_price_alignment",
    "funding_z20",
    "funding_persistence_20d",
    "funding_ma_diff_3_30",
    "funding_vol_30d",
    "oi_z20",
    "oi_change_5d",
    "oi_change_decay_20d",
    "funding_oi_joint_5d",
}
_DISLOCATION_FIELDS = {
    "premium_close",
    "premium_change_intraday",
    "mark_close",
    "index_close",
    "mark_index_spread",
    "mark_index_spread_change",
}
_PATH_SHAPE_FIELDS = {
    "path_efficiency",
    "high_time_frac",
    "low_time_frac",
    "high_before_low",
    "late_range_share",
    "intraday_reversal",
}
_REGIME_FIELDS = {
    "market_cum_return",
    "market_vol_intensity",
    "beta_btc_60d",
    "corr_btc_20d",
    "resid_return_btc_20d",
    "relative_strength_btc_5d",
}
_FUNDAMENTAL_FIELDS = {
    "cm_log_mcap_lag1",
    "cm_mcap_pct_lag1",
    "cm_mcap_z20_lag1",
    "cm_turnover_to_mcap_lag1",
}

_SOURCE_FAMILY_MAP: dict[str, str] = {}
for _name in _PRICE_FIELDS:
    _SOURCE_FAMILY_MAP[_name] = "price"
for _name in _PRICE_STATE_FIELDS:
    _SOURCE_FAMILY_MAP[_name] = "price_state"
for _name in _TREND_FIELDS:
    _SOURCE_FAMILY_MAP[_name] = "trend"
for _name in _VOLATILITY_FIELDS:
    _SOURCE_FAMILY_MAP[_name] = "volatility"
for _name in _LIQUIDITY_FIELDS:
    _SOURCE_FAMILY_MAP[_name] = "liquidity"
for _name in _FLOW_FIELDS:
    _SOURCE_FAMILY_MAP[_name] = "flow"
for _name in _LEVERAGE_FIELDS:
    _SOURCE_FAMILY_MAP[_name] = "leverage"
for _name in _DISLOCATION_FIELDS:
    _SOURCE_FAMILY_MAP[_name] = "dislocation"
for _name in _PATH_SHAPE_FIELDS:
    _SOURCE_FAMILY_MAP[_name] = "path_shape"
for _name in _REGIME_FIELDS:
    _SOURCE_FAMILY_MAP[_name] = "regime"
for _name in _FUNDAMENTAL_FIELDS:
    _SOURCE_FAMILY_MAP[_name] = "fundamental"

_MODE1_FAMILY = {
    "mean": "level",
    "median": "level",
    "sum": "level",
    "max": "level",
    "min": "level",
    "std": "dispersion",
    "skew": "dispersion",
    "kurt": "dispersion",
    "zscore": "dispersion",
    "momentum": "trend",
    "range_position": "state",
}
_MODE2_FAMILY = {
    "corr": "co_movement",
    "cov": "co_movement",
    "rank_corr": "co_movement",
    "cos_sim": "co_movement",
    "slope": "mapping",
    "beta": "mapping",
    "intercept": "mapping",
    "r2": "fit_quality",
    "euc_dist": "distance",
    "wmean": "weighted_level",
    "residual_std": "spread_relation",
    "diff_mean": "spread_relation",
    "diff_std": "spread_relation",
    "ratio_mean": "spread_relation",
    "ratio_std": "spread_relation",
}
_MODE3_FAMILY = {
    "mean_ratio": "cross_horizon",
    "std_ratio": "cross_horizon",
    "mean_diff": "cross_horizon",
    "zscore_anchor": "cross_horizon",
    "std_normalized_diff": "cross_horizon",
}
_MODE4_FAMILY = {
    "group_slope": "ordered_profile",
    "group_early_late_diff": "ordered_profile",
    "group_path_length": "ordered_profile",
    "group_first_last_diff": "ordered_profile",
    "group_dispersion": "bucket_dispersion",
    "group_skew": "bucket_dispersion",
    "group_jump_ratio": "bucket_dispersion",
    "group_top_share": "bucket_concentration",
    "group_corr": "bucket_relation",
    "group_beta": "bucket_relation",
    "group_dispersion_vclock": "volume_clock_dispersion",
    "group_slope_vclock": "volume_clock_profile",
    "group_path_length_vclock": "volume_clock_profile",
    "group_early_late_diff_vclock": "volume_clock_profile",
}
_TS_FAMILY = {
    "none": "none",
    "delta": "delta",
    "pct_change": "delta",
    "ts_accel": "delta",
    "ts_zscore": "normalization",
    "ts_rank": "normalization",
    "ts_decay": "smoothing",
}
_CS_FAMILY = {
    "none": "none",
    "cs_rank": "ranking",
    "cs_zscore": "normalization",
    "cs_demean": "normalization",
    "cs_scale": "normalization",
}


def _utcnow_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _median(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(median(clean))


def _unique_list(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def resolve_family_feedback_path(path: str | Path | None = None) -> Path:
    override = str(path).strip() if path is not None else os.environ.get("GP_FAMILY_FEEDBACK_PATH", "").strip()
    return Path(override).resolve() if override else DEFAULT_FAMILY_FEEDBACK_PATH.resolve()


def load_family_feedback_artifact(path: str | Path | None = None) -> dict[str, Any] | None:
    resolved = resolve_family_feedback_path(path)
    if not resolved.exists():
        return None
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    payload.setdefault("artifact_path", str(resolved))
    return payload


def indicator_source_family(name: str | None) -> str:
    if not name:
        return "other"
    return _SOURCE_FAMILY_MAP.get(str(name), "other")


def mode1_operator_family(name: str | None) -> str:
    return _MODE1_FAMILY.get(str(name), "other")


def mode2_operator_family(name: str | None) -> str:
    return _MODE2_FAMILY.get(str(name), "other")


def mode3_operator_family(name: str | None) -> str:
    return _MODE3_FAMILY.get(str(name), "other")


def mode4_operator_family(name: str | None) -> str:
    return _MODE4_FAMILY.get(str(name), "other")


def ts_operator_family(name: str | None) -> str:
    return _TS_FAMILY.get(str(name), "other")


def cs_operator_family(name: str | None) -> str:
    return _CS_FAMILY.get(str(name), "other")


def window_bucket(window: Any) -> str:
    value = _safe_float(window)
    if value is None:
        return "unknown"
    if value <= 45:
        return "short"
    if value <= 120:
        return "mid"
    return "long"


def turnover_bucket(turnover: Any) -> str:
    value = _safe_float(turnover)
    if value is None:
        return "unknown"
    if value < 0.40:
        return "low"
    if value < 0.90:
        return "mid"
    return "high"


def sharpe_bucket(sharpe: Any) -> str:
    value = _safe_float(sharpe)
    if value is None:
        return "unknown"
    if value < 0.80:
        return "weak"
    if value < 1.00:
        return "ok"
    if value < 1.25:
        return "strong"
    return "elite"


def mask_side(rule: str | None) -> str:
    if not rule or rule == "none":
        return "none"
    if str(rule).startswith("high_"):
        return "high"
    if str(rule).startswith("low_"):
        return "low"
    return "other"


def extract_quality_metrics(
    metadata: dict[str, Any] | None = None,
    screening: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    screening = screening or {}
    validation = validation or {}
    fullsample = metadata.get("fullsample_metrics")
    fullsample = fullsample if isinstance(fullsample, dict) else {}
    return {
        "ls_net_sharpe": (
            _safe_float(fullsample.get("ls_net_sharpe"))
            or _safe_float(screening.get("ls_net_sharpe"))
            or _safe_float(screening.get("ls_sharpe"))
        ),
        "ls_netret": (
            _safe_float(fullsample.get("ls_netret"))
            or _safe_float(screening.get("ls_netret"))
            or _safe_float(screening.get("ls_ret"))
        ),
        "ls_turnover": (
            _safe_float(fullsample.get("ls_turnover"))
            or _safe_float(screening.get("ls_turnover"))
            or _safe_float(screening.get("long_turnover"))
        ),
        "direction_consistent": fullsample.get("direction_consistent"),
        "oos_positive_rate": _safe_float(validation.get("oob_monthly_positive_rate")),
        "replacement_win": bool(metadata.get("replaced_factor")),
        "blocked_by_pool": bool(metadata.get("blocker")),
    }


def build_candidate_feedback_payload(
    decoded: dict[str, Any] | None,
    *,
    metadata: dict[str, Any] | None = None,
    screening: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    lifecycle_state: str | None = None,
) -> dict[str, Any]:
    decoded = decoded or {}
    metrics = extract_quality_metrics(metadata=metadata, screening=screening, validation=validation)
    mode = int(decoded.get("mode") or 0)
    source_a = indicator_source_family(decoded.get("A"))
    source_b = indicator_source_family(decoded.get("B"))
    mode1_op = decoded.get("mode1_op")
    mode2_op = decoded.get("mode2_op")
    mode3_op = decoded.get("mode3_op")
    mode4_op = decoded.get("mode4_op")
    operator_name = mode1_op or mode2_op or mode3_op or mode4_op or "none"
    if mode == 1:
        operator_family = mode1_operator_family(mode1_op)
    elif mode == 2:
        operator_family = mode2_operator_family(mode2_op)
    elif mode == 3:
        operator_family = mode3_operator_family(mode3_op)
    elif mode == 4:
        operator_family = mode4_operator_family(mode4_op)
    else:
        operator_family = "other"
    ts_op = str(decoded.get("ts_comp_op") or "none")
    cs_op = str(decoded.get("cs_comp_op") or "none")
    ts_family = ts_operator_family(ts_op)
    cs_family = cs_operator_family(cs_op)
    rule = str(decoded.get("mask_rule") or "none")
    mask_field_family = indicator_source_family(decoded.get("mask_field"))
    mask_family = "none" if rule == "none" else f"{mask_side(rule)}:{mask_field_family}"
    win_bucket = window_bucket(decoded.get("window"))
    source_pair = f"{source_a}->{source_b}" if mode in {2, 4} else source_a
    composition_signature = f"{ts_op}|{cs_op}"
    family_key = "|".join(
        (
            f"m{mode}",
            source_pair,
            operator_family,
            win_bucket,
            ts_family,
            cs_family,
            mask_family,
        )
    )
    return {
        "family_key": family_key,
        "behavior_signature": "|".join(
            (
                family_key,
                turnover_bucket(metrics.get("ls_turnover")),
                sharpe_bucket(metrics.get("ls_net_sharpe")),
            )
        ),
        "mode": mode,
        "A": decoded.get("A"),
        "B": decoded.get("B"),
        "source_family_a": source_a,
        "source_family_b": source_b,
        "source_pair": source_pair,
        "operator_name": operator_name,
        "operator_family": operator_family,
        "window": decoded.get("window"),
        "window_bucket": win_bucket,
        "mask_rule": rule,
        "mask_field": decoded.get("mask_field"),
        "mask_field_family": mask_field_family,
        "mask_family": mask_family,
        "mask_side": mask_side(rule),
        "lag": decoded.get("B_shift_lag"),
        "ts_comp_op": ts_op,
        "ts_comp_window": decoded.get("ts_comp_window"),
        "ts_family": ts_family,
        "cs_comp_op": cs_op,
        "cs_family": cs_family,
        "composition_signature": composition_signature,
        "turnover_bucket": turnover_bucket(metrics.get("ls_turnover")),
        "sharpe_bucket": sharpe_bucket(metrics.get("ls_net_sharpe")),
        "lifecycle_state": lifecycle_state,
    }


def _summarize_samples(samples: list[dict[str, Any]], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    count = len(samples)
    approved_count = sum(1 for sample in samples if sample["approved"])
    approved_rate = approved_count / count if count else 0.0
    direction_values = [
        float(bool(sample["direction_consistent"]))
        for sample in samples
        if sample["direction_consistent"] is not None
    ]
    direction_rate = sum(direction_values) / len(direction_values) if direction_values else None
    replacement_win_rate = (
        sum(1 for sample in samples if sample["replacement_win"]) / count if count else 0.0
    )
    blocked_by_pool_rate = (
        sum(1 for sample in samples if sample["blocked_by_pool"]) / count if count else 0.0
    )
    stats = {
        "count": count,
        "approved_count": approved_count,
        "archived_count": count - approved_count,
        "approved_rate": round(approved_rate, 6),
        "median_sharpe": _median(sample["ls_net_sharpe"] for sample in samples),
        "approved_median_sharpe": _median(
            sample["ls_net_sharpe"] for sample in samples if sample["approved"]
        ),
        "median_netret": _median(sample["ls_netret"] for sample in samples),
        "approved_median_netret": _median(
            sample["ls_netret"] for sample in samples if sample["approved"]
        ),
        "direction_consistency_rate": (
            round(direction_rate, 6) if direction_rate is not None else None
        ),
        "replacement_win_rate": round(replacement_win_rate, 6),
        "blocked_by_pool_rate": round(blocked_by_pool_rate, 6),
    }
    if baseline is None or count == 0:
        stats["sampler_weight"] = 1.0
        return stats

    baseline_approved = float(baseline.get("approved_rate") or 0.0)
    baseline_direction = baseline.get("direction_consistency_rate")
    baseline_direction = 0.5 if baseline_direction is None else float(baseline_direction)
    baseline_sharpe = baseline.get("approved_median_sharpe")
    if baseline_sharpe is None:
        baseline_sharpe = baseline.get("median_sharpe")
    baseline_sharpe = 0.0 if baseline_sharpe is None else float(baseline_sharpe)
    target_sharpe = stats["approved_median_sharpe"]
    if target_sharpe is None:
        target_sharpe = stats["median_sharpe"]
    target_sharpe = baseline_sharpe if target_sharpe is None else float(target_sharpe)
    target_direction = stats["direction_consistency_rate"]
    target_direction = baseline_direction if target_direction is None else float(target_direction)
    raw = (
        1.0
        + 0.90 * (float(stats["approved_rate"]) - baseline_approved)
        + 0.35 * (target_sharpe - baseline_sharpe)
        + 0.10 * (target_direction - baseline_direction)
        + 0.10 * float(stats["replacement_win_rate"])
        - 0.35 * float(stats["blocked_by_pool_rate"])
    )
    raw = min(1.80, max(0.35, raw))
    shrink = count / (count + 4.0)
    stats["sampler_weight"] = round(1.0 + (raw - 1.0) * shrink, 6)
    return stats


def _sorted_stats_map(
    groups: dict[str, list[dict[str, Any]]],
    baseline: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    items: list[dict[str, Any]] = []
    weights: dict[str, float] = {}
    for value, samples in groups.items():
        if not value:
            continue
        stats = _summarize_samples(samples, baseline=baseline)
        item = {"value": value, **stats}
        items.append(item)
        weights[value] = float(stats["sampler_weight"])
    items.sort(key=lambda item: (-item["sampler_weight"], -item["approved_rate"], -item["count"], item["value"]))
    return items, weights


def aggregate_family_feedback(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    state_counts: dict[str, int] = defaultdict(int)
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "mode": defaultdict(list),
        "indicator": defaultdict(list),
        "source_family": defaultdict(list),
        "mode1_op": defaultdict(list),
        "mode2_op": defaultdict(list),
        "mode3_op": defaultdict(list),
        "mode4_op": defaultdict(list),
        "operator_family": defaultdict(list),
        "window": defaultdict(list),
        "window_bucket": defaultdict(list),
        "mask_rule": defaultdict(list),
        "mask_field_family": defaultdict(list),
        "lag": defaultdict(list),
        "ts_comp_op": defaultdict(list),
        "ts_family": defaultdict(list),
        "ts_comp_window": defaultdict(list),
        "cs_comp_op": defaultdict(list),
        "cs_family": defaultdict(list),
        "behavior_key": defaultdict(list),
        "dominant_regime": defaultdict(list),
    }

    for row in rows:
        decoded = row.get("decoded") or {}
        if not isinstance(decoded, dict) or not decoded:
            continue
        metadata = row.get("metadata") or {}
        screening = row.get("screening_metrics") or {}
        validation = row.get("validation_metrics") or {}
        behavior_feedback = metadata.get("behavior_feedback")
        behavior_feedback = behavior_feedback if isinstance(behavior_feedback, dict) else {}
        lifecycle_state = str(row.get("lifecycle_state") or "")
        feedback = build_candidate_feedback_payload(
            decoded,
            metadata=metadata,
            screening=screening,
            validation=validation,
            lifecycle_state=lifecycle_state,
        )
        quality = extract_quality_metrics(metadata=metadata, screening=screening, validation=validation)
        sample = {
            "display_name": row.get("display_name"),
            "lifecycle_state": lifecycle_state,
            "approved": lifecycle_state == "approved",
            "ls_net_sharpe": quality.get("ls_net_sharpe"),
            "ls_netret": quality.get("ls_netret"),
            "ls_turnover": quality.get("ls_turnover"),
            "direction_consistent": quality.get("direction_consistent"),
            "replacement_win": quality.get("replacement_win"),
            "blocked_by_pool": quality.get("blocked_by_pool"),
            "feedback": feedback,
            "behavior_feedback": behavior_feedback,
        }
        samples.append(sample)
        state_counts[lifecycle_state] += 1
        family_groups[feedback["family_key"]].append(sample)
        component_groups["mode"][str(feedback["mode"])].append(sample)
        for name in _unique_list((feedback.get("A"), feedback.get("B"))):
            component_groups["indicator"][name].append(sample)
        for name in _unique_list((feedback.get("source_family_a"), feedback.get("source_family_b"))):
            component_groups["source_family"][name].append(sample)
        if feedback.get("mode") == 1 and feedback.get("operator_name"):
            component_groups["mode1_op"][str(feedback["operator_name"])].append(sample)
        if feedback.get("mode") == 2 and feedback.get("operator_name"):
            component_groups["mode2_op"][str(feedback["operator_name"])].append(sample)
        if feedback.get("mode") == 3 and feedback.get("operator_name"):
            component_groups["mode3_op"][str(feedback["operator_name"])].append(sample)
        if feedback.get("mode") == 4 and feedback.get("operator_name"):
            component_groups["mode4_op"][str(feedback["operator_name"])].append(sample)
        component_groups["operator_family"][str(feedback["operator_family"])].append(sample)
        if feedback.get("window") is not None:
            component_groups["window"][str(feedback["window"])].append(sample)
        component_groups["window_bucket"][str(feedback["window_bucket"])].append(sample)
        component_groups["mask_rule"][str(feedback["mask_rule"])].append(sample)
        component_groups["mask_field_family"][str(feedback["mask_field_family"])].append(sample)
        if feedback.get("lag") is not None:
            component_groups["lag"][str(feedback["lag"])].append(sample)
        component_groups["ts_comp_op"][str(feedback["ts_comp_op"])].append(sample)
        component_groups["ts_family"][str(feedback["ts_family"])].append(sample)
        if feedback.get("ts_comp_window") is not None:
            component_groups["ts_comp_window"][str(feedback["ts_comp_window"])].append(sample)
        component_groups["cs_comp_op"][str(feedback["cs_comp_op"])].append(sample)
        component_groups["cs_family"][str(feedback["cs_family"])].append(sample)
        if behavior_feedback.get("behavior_key"):
            component_groups["behavior_key"][str(behavior_feedback["behavior_key"])].append(sample)
        if behavior_feedback.get("dominant_regime"):
            component_groups["dominant_regime"][str(behavior_feedback["dominant_regime"])].append(sample)

    baseline = _summarize_samples(samples)
    families, _ = _sorted_stats_map(family_groups, baseline=baseline)
    component_stats: dict[str, list[dict[str, Any]]] = {}
    sampler_weights: dict[str, dict[str, float]] = {}
    for key, groups in component_groups.items():
        stats, weights = _sorted_stats_map(groups, baseline=baseline)
        component_stats[key] = stats
        sampler_weights[key] = weights

    top_families = [item for item in families if item["count"] >= 2][:10]
    top_behaviors = [item for item in component_stats["behavior_key"] if item["count"] >= 2][:10]
    return {
        "feedback_version": "v1",
        "generated_at": _utcnow_text(),
        "candidate_count": len(samples),
        "state_counts": dict(sorted(state_counts.items())),
        "baselines": baseline,
        "top_families": top_families,
        "top_behaviors": top_behaviors,
        "families": families,
        "component_stats": component_stats,
        "sampler_weights": sampler_weights,
    }


def write_family_feedback_artifact(
    payload: dict[str, Any],
    *,
    path: str | Path | None = None,
) -> Path:
    resolved = resolve_family_feedback_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved
