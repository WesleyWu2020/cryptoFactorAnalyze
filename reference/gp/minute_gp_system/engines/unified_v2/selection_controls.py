"""Low-correlation final selection helpers for GP candidates."""
from __future__ import annotations

import os
from collections import Counter
from typing import Callable, Iterable

import numpy as np

STYLE_EXPOSURE_FIELDS = [
    # Core size/liquidity/style dimensions.
    "mean_beta_cm_log_mcap_lag1",
    "mean_beta_dollar_volume_rank",
    "mean_beta_amihud_illiq",
    # BTC-relative market behavior.
    "mean_beta_beta_btc_60d",
    "mean_beta_corr_btc_20d",
    "mean_beta_relative_strength_btc_5d",
    # Volatility regime descriptors.
    "mean_beta_rv_20d",
    "mean_beta_rv_ratio_5_20",
    # Crypto-specific funding and positioning dimensions.
    "mean_beta_funding",
    "mean_beta_funding_z20",
    "mean_beta_oi_change_5d",
    "mean_beta_funding_oi_joint_5d",
]
LOW_CORR_USE_STYLE_EXPOSURE = os.environ.get("GP_LOW_CORR_USE_STYLE_EXPOSURE", "1").lower() in {
    "1", "true", "yes", "on",
}
LOW_CORR_MAX_STYLE_EXPOSURE_COSINE = float(
    os.environ.get("GP_LOW_CORR_MAX_STYLE_EXPOSURE_COSINE", "0.80")
)
TAIL_OVERLAP_HARD = float(os.environ.get("GP_TAIL_OVERLAP_HARD", "0.70"))
SPEARMAN_CORR_HARD = float(os.environ.get("GP_SPEARMAN_CORR_HARD", "0.75"))


def _candidate_index(item: dict) -> int | None:
    try:
        return int(item.get("_candidate_index"))
    except (TypeError, ValueError):
        return None


def _unit_rows(values) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - arr.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    return arr / norm


def _max_abs_corr(idx: int | None, selected_indices: Iterable[int], unit_values: np.ndarray | None) -> float:
    if idx is None or unit_values is None:
        return 0.0
    if idx < 0 or idx >= unit_values.shape[0]:
        return 0.0
    selected = [int(i) for i in selected_indices if 0 <= int(i) < unit_values.shape[0]]
    if not selected:
        return 0.0
    corr = np.abs(unit_values[idx] @ unit_values[np.asarray(selected, dtype=np.intp)].T)
    return float(np.max(corr)) if corr.size else 0.0


def _compute_tail_overlap(
    idx: int | None,
    selected_indices: Iterable[int],
    factor_values: np.ndarray | None,
    top_frac: float = 0.20,
) -> float:
    if idx is None or factor_values is None:
        return 0.0
    values = np.asarray(factor_values, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        return 0.0
    if idx < 0 or idx >= values.shape[0]:
        return 0.0
    selected = [int(i) for i in selected_indices if 0 <= int(i) < values.shape[0]]
    if not selected:
        return 0.0

    frac = float(top_frac)
    if not np.isfinite(frac) or frac <= 0.0:
        frac = 0.20

    candidate = values[idx]
    best_overlap = 0.0
    for selected_idx in selected:
        selected_values = values[selected_idx]
        period_scores: list[float] = []
        for period_idx in range(values.shape[1]):
            candidate_period = candidate[period_idx]
            selected_period = selected_values[period_idx]
            valid = np.isfinite(candidate_period) & np.isfinite(selected_period)
            n_valid = int(valid.sum())
            if n_valid <= 0:
                continue
            k = max(1, int(np.ceil(n_valid * frac)))
            if n_valid < 2 * k:
                continue

            candidate_valid = candidate_period[valid]
            selected_valid = selected_period[valid]
            top_candidate = np.argpartition(candidate_valid, n_valid - k)[-k:]
            bottom_candidate = np.argpartition(candidate_valid, k - 1)[:k]
            top_selected = np.argpartition(selected_valid, n_valid - k)[-k:]
            bottom_selected = np.argpartition(selected_valid, k - 1)[:k]

            top_overlap = np.intersect1d(top_candidate, top_selected).size / float(k)
            bottom_overlap = np.intersect1d(bottom_candidate, bottom_selected).size / float(k)
            period_scores.append(0.5 * (top_overlap + bottom_overlap))

        if period_scores:
            best_overlap = max(best_overlap, float(np.mean(period_scores)))
    return float(best_overlap)


def _max_abs_spearman(
    idx: int | None,
    selected_indices: Iterable[int],
    unit_values: np.ndarray | None,
) -> float:
    if idx is None or unit_values is None:
        return 0.0
    values = np.asarray(unit_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        return 0.0
    if idx < 0 or idx >= values.shape[0]:
        return 0.0
    selected = [int(i) for i in selected_indices if 0 <= int(i) < values.shape[0]]
    if not selected:
        return 0.0

    indices = np.asarray([idx, *selected], dtype=np.intp)
    subset = np.nan_to_num(values[indices], nan=0.0, posinf=0.0, neginf=0.0)
    rank_unit = np.empty_like(subset, dtype=np.float32)
    for row_idx, row in enumerate(subset):
        order = np.argsort(row, kind="mergesort")
        sorted_row = row[order]
        row_ranks = np.empty(row.shape[0], dtype=np.float32)
        start = 0
        while start < row.shape[0]:
            end = start + 1
            while end < row.shape[0] and sorted_row[end] == sorted_row[start]:
                end += 1
            row_ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        row_ranks = row_ranks - row_ranks.mean()
        norm = float(np.linalg.norm(row_ranks))
        if not np.isfinite(norm) or norm <= 1e-12:
            rank_unit[row_idx] = 0.0
        else:
            rank_unit[row_idx] = row_ranks / norm

    corr = np.abs(rank_unit[0] @ rank_unit[1:].T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.max(corr)) if corr.size else 0.0


def _decoded_field(item: dict, key: str) -> str | None:
    decoded = item.get("decoded") if isinstance(item.get("decoded"), dict) else {}
    value = decoded.get(key)
    return str(value) if value else None


def _ab_pair_key(item: dict) -> tuple | None:
    decoded = item.get("decoded") if isinstance(item.get("decoded"), dict) else {}
    try:
        mode = int(decoded.get("mode") or 0)
    except (TypeError, ValueError):
        mode = 0
    if mode == 0:
        return None
    a = decoded.get("A")
    b = decoded.get("B")
    if not a or not b:
        return None
    return tuple(sorted((str(a), str(b))))


def _candidate_value(candidate, field: str) -> float:
    if isinstance(candidate, dict):
        value = candidate.get(field, 0.0)
    else:
        value = getattr(candidate, field, 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _candidate_label(candidate):
    for field in ("label", "formula", "raw_formula", "rank", "_candidate_index"):
        if isinstance(candidate, dict):
            value = candidate.get(field)
        else:
            value = getattr(candidate, field, None)
        if value is not None:
            return str(value)
    return None


def _candidate_style_exposure_vector(candidate) -> np.ndarray:
    return np.asarray(
        [_candidate_value(candidate, field) for field in STYLE_EXPOSURE_FIELDS],
        dtype=np.float32,
    )


def _style_exposure_cosine(a, b) -> float:
    left = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    right = np.nan_to_num(np.asarray(b, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm < 1e-8 or right_norm < 1e-8:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _passes_style_overlap_gate(candidate, selected, max_abs_cosine=0.80):
    candidate_vec = _candidate_style_exposure_vector(candidate)
    max_abs = 0.0
    max_signed = 0.0
    overlap_with = None
    for selected_item in selected:
        cosine = _style_exposure_cosine(candidate_vec, _candidate_style_exposure_vector(selected_item))
        abs_cosine = abs(cosine)
        if abs_cosine > max_abs:
            max_abs = float(abs_cosine)
            max_signed = float(cosine)
            overlap_with = _candidate_label(selected_item)
    if max_abs >= float(max_abs_cosine):
        detail = {
            "rejected_reason": "style_overlap",
            "style_exposure_cosine": float(max_abs),
            "style_exposure_signed_cosine": float(max_signed),
        }
        if overlap_with is not None:
            detail["style_overlap_with"] = overlap_with
        return False, detail
    return True, {}


def _reject(summary: dict, item: dict, reason: str) -> None:
    summary["rejected"][reason] = int(summary["rejected"].get(reason, 0)) + 1
    item["final_low_corr_gate"] = {"passed": False, "reason": reason}


def select_low_corr_results(
    candidates: list[dict],
    *,
    target_max: int,
    phenotypes=None,
    behavior_signatures=None,
    factor_values=None,
    pool_corr_hard: float = 0.60,
    spearman_corr_hard: float = 0.75,
    behavior_corr_hard: float = 0.55,
    external_corr_hard: float = 0.55,
    field_family_func: Callable[[str], str] | None = None,
    field_family_max_share: float = 0.30,
    ab_pair_max: int = 1,
    use_style_exposure: bool | None = None,
    max_style_exposure_cosine: float | None = None,
    tail_overlap_hard: float = 0.70,
) -> tuple[list[dict], dict]:
    """Greedy final selector that admits high-score candidates only if they add diversity.

    Candidates are considered in descending composite order. A candidate is
    rejected when it is too correlated to the selected set, too correlated to
    the external archive, or would exceed simple field-family / A:B quotas.
    """
    target = max(0, int(target_max))
    pheno = _unit_rows(phenotypes)
    behavior = _unit_rows(behavior_signatures)
    ordered = sorted(
        candidates,
        key=lambda item: float(item.get("composite", -1e18) or -1e18),
        reverse=True,
    )
    selected: list[dict] = []
    selected_indices: list[int] = []
    family_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple] = Counter()
    family_cap = max(1, int(np.ceil(max(target, 1) * float(field_family_max_share))))
    pair_cap = max(1, int(ab_pair_max))
    use_style = LOW_CORR_USE_STYLE_EXPOSURE if use_style_exposure is None else bool(use_style_exposure)
    style_limit = (
        LOW_CORR_MAX_STYLE_EXPOSURE_COSINE
        if max_style_exposure_cosine is None
        else float(max_style_exposure_cosine)
    )
    summary = {
        "enabled": True,
        "target_max": target,
        "pool_corr_hard": float(pool_corr_hard),
        "spearman_corr_hard": float(spearman_corr_hard),
        "behavior_corr_hard": float(behavior_corr_hard),
        "external_corr_hard": float(external_corr_hard),
        "tail_overlap_hard": float(tail_overlap_hard),
        "use_style_exposure": bool(use_style),
        "max_style_exposure_cosine": float(style_limit),
        "field_family_max_share": float(field_family_max_share),
        "field_family_cap": int(family_cap),
        "ab_pair_max": int(pair_cap),
        "rejected": {},
    }

    for item in ordered:
        if len(selected) >= target:
            _reject(summary, item, "target_full")
            continue

        external_corr = item.get("external_behavior_corr_max")
        if external_corr is not None and np.isfinite(float(external_corr)) and float(external_corr) > external_corr_hard:
            _reject(summary, item, "external_archive_corr")
            continue

        idx = _candidate_index(item)
        pool_corr = _max_abs_corr(idx, selected_indices, pheno)
        if pool_corr > pool_corr_hard:
            item["selected_pool_corr_max"] = float(pool_corr)
            _reject(summary, item, "selected_pool_corr")
            continue

        spearman_corr = _max_abs_spearman(idx, selected_indices, pheno)
        if spearman_corr_hard < 1.0 and spearman_corr > spearman_corr_hard:
            item["selected_spearman_corr_max"] = float(spearman_corr)
            _reject(summary, item, "selected_spearman_corr")
            continue

        behavior_corr = _max_abs_corr(idx, selected_indices, behavior)
        if behavior_corr > behavior_corr_hard:
            item["selected_behavior_corr_max"] = float(behavior_corr)
            _reject(summary, item, "selected_behavior_corr")
            continue

        tail_overlap = _compute_tail_overlap(idx, selected_indices, factor_values)
        if tail_overlap > tail_overlap_hard:
            item["selected_tail_overlap"] = float(tail_overlap)
            _reject(summary, item, "selected_tail_overlap")
            continue

        if use_style:
            passed_style, style_detail = _passes_style_overlap_gate(
                item,
                selected,
                max_abs_cosine=style_limit,
            )
            if not passed_style:
                item.update(style_detail)
                _reject(summary, item, "style_overlap")
                continue

        family = None
        a_field = _decoded_field(item, "A")
        if field_family_func is not None and a_field is not None:
            family = str(field_family_func(a_field))
            if family_counts[family] >= family_cap:
                _reject(summary, item, "field_family_quota")
                continue

        pair = _ab_pair_key(item)
        if pair is not None and pair_counts[pair] >= pair_cap:
            _reject(summary, item, "ab_pair_quota")
            continue

        item["selected_pool_corr_max"] = float(pool_corr)
        item["selected_spearman_corr_max"] = float(spearman_corr)
        item["selected_behavior_corr_max"] = float(behavior_corr)
        item["selected_tail_overlap"] = float(tail_overlap)
        item["final_low_corr_gate"] = {"passed": True, "reason": None}
        selected.append(item)
        if idx is not None:
            selected_indices.append(idx)
        if family is not None:
            family_counts[family] += 1
        if pair is not None:
            pair_counts[pair] += 1

    summary["selected"] = int(len(selected))
    summary["input_candidates"] = int(len(candidates))
    summary["field_family_counts"] = dict(family_counts)
    summary["ab_pair_counts"] = {str(k): int(v) for k, v in pair_counts.items()}
    return selected, summary
