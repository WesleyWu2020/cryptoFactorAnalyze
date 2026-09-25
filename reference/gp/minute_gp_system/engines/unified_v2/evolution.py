"""Main evolution loop v2 aligned to full-sample net-performance search."""
import os
import json
import time
import gc
import numpy as np
import pandas as pd

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.crossover import Crossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.core.mutation import Mutation
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.core.problem import Problem

from .config import (
    POPULATION_SIZE, N_GENERATIONS, N_OBJECTIVES,
    PARAM_BOUNDS_LOWER, PARAM_BOUNDS_UPPER, N_PARAMS,
    DEFAULT_H5_PATH, CHUNK_PERIODS, MINUTES_PER_PERIOD,
    LAG_PERIODS, PERIODS_PER_YEAR, TOP_QUANTILE,
    EVAL_BATCH_SIZE, CACHE_DTYPE,
    MODE1_OPS, MODE2_OPS, MODE3_OPS, MODE4_OPS, WINDOW_CHOICES,
    INTRADAY_GROUP_CHOICES,
    N_INDICATORS, NICHE_SHARING_ALPHA,
    RESULT_TARGET_MIN, RESULT_TARGET_MAX, FINAL_CANDIDATE_POOL_SIZE,
    DIVERSITY_MAX_PERIODS,
    PERF_PROFILE, GPU_GROUP_TARGET_INDIVIDUALS,
    INDICATOR_INDEX, INDICATOR_NAMES, SLICE_CHOICES, MASK_RULES, B_SHIFT_CHOICES,
    TS_COMP_OPS, TS_COMP_WINDOWS, CS_COMP_OPS, CS_COMP_DEPENDENCY_FIELDS, decode_individual, formula_string,
    DEFAULT_SEARCH_SPACE_PROFILE, build_search_space_state, set_search_space_overrides,
    project_population_to_search_space, search_space_indicator_indices,
    FUNDAMENTAL_INDICATOR_NAMES, fundamental_required_enabled,
    ORDER_FLOW_INDICATOR_NAMES, XBINANCE_ORDER_FLOW_INDICATOR_NAMES,
    CROSS_PERIOD_INDICATOR_NAMES, MODE4_A_INTRADAY_FIELD_NAMES,
    PAIR_A_FAMILY_DEFAULT_MAX_SHARE, pair_a_family_of_field,
)
from .backend import set_backend, xp, free_gpu, to_numpy
from .data_loader import CryptoDataLoader
from .evaluator import evaluate_population
from .fitness import (
    build_lagged_returns, prepare_fitness_context,
    compute_five_objectives_prepared,
    sample_diversity_features, standardize_phenotypes,
    gpu_fitness_enabled, materialize_fitness_context,
    prepare_residual_basis_context, compute_residual_rankic, compute_rankic_novelty,
)
from .behavior_signature import build_behavior_context, build_behavior_signatures
from .selection_controls import select_low_corr_results
from gp.minute_gp_system.family_feedback import (
    cs_operator_family,
    indicator_source_family,
    load_family_feedback_artifact,
    mode1_operator_family,
    mode2_operator_family,
    mode3_operator_family,
    mode4_operator_family,
    ts_operator_family,
    window_bucket,
)


# ---------------------------------------------------------------------------
# Exploration-biased initial sampling
# ---------------------------------------------------------------------------
def _sample_from_pool(rng, choices, probs=None):
    if probs is None:
        return int(choices[int(rng.integers(0, len(choices)))])
    idx = int(rng.choice(len(choices), p=probs))
    return int(choices[idx])


def _sample_mask_pair(rng, choices, probs=None):
    idx = int(rng.integers(0, len(choices))) if probs is None else int(rng.choice(len(choices), p=probs))
    return int(choices[idx][0]), int(choices[idx][1])


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _json_env_dict(name, default):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return dict(default)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return dict(value)


_COMPOUND_SHARE = 0.20
_PRIMARY_FEEDBACK_BLEND = 0.70
_EXPLORE_FEEDBACK_BLEND = 0.35
_PATH_SHAPE_TEMPLATE_FIELDS = tuple(
    name for name in (
        "path_efficiency",
        "high_time_frac",
        "low_time_frac",
        "high_before_low",
        "am_pm_return_diff",
        "late_volume_share",
        "late_range_share",
        "volume_burst_share",
        "intraday_reversal",
    )
    if name in INDICATOR_INDEX
)
_FUNDAMENTAL_FIELD_INDICES = tuple(
    INDICATOR_INDEX[name]
    for name in FUNDAMENTAL_INDICATOR_NAMES
    if name in INDICATOR_INDEX
)
_FUNDAMENTAL_CAP_FIELDS = tuple(
    name for name in (
        "cm_mcap_pct_lag1",
        "cm_mcap_z20_lag1",
        "cm_log_mcap_lag1",
        "cm_turnover_to_mcap_lag1",
    )
    if name in INDICATOR_INDEX
)
_FUNDAMENTAL_BASE_FIELDS = tuple(
    name for name in (
        "price_range_pct",
        "return_abs",
        "money_flow",
        "turnover",
        "late_range_share",
        "path_efficiency",
        "intraday_reversal",
    )
    if name in INDICATOR_INDEX
)
FORCE_MODE4_ONLY = os.environ.get("GP_FORCE_MODE4_ONLY", "0").lower() in {
    "1", "true", "yes", "on",
}
FINAL_MIN_LS_NETRET = float(os.environ.get("GP_FINAL_MIN_LS_NETRET", "0.0"))
FINAL_MIN_LS_NET_SHARPE = float(os.environ.get("GP_FINAL_MIN_LS_NET_SHARPE", "0.80"))
FINAL_MAX_LS_TURNOVER = float(os.environ.get("GP_FINAL_MAX_LS_TURNOVER", "0.80"))
FINAL_MIN_LS_POSITIVE_RATE = float(os.environ.get("GP_FINAL_MIN_LS_POSITIVE_RATE", "0.50"))
FINAL_MIN_COVERAGE = float(os.environ.get("GP_FINAL_MIN_COVERAGE", "0.95"))
FINAL_MIN_FACTOR_FINITE_PERIOD_RATIO = float(os.environ.get("GP_FINAL_MIN_FACTOR_FINITE_PERIOD_RATIO", "0.50"))
FINAL_MIN_FACTOR_NONZERO_RATIO = float(os.environ.get("GP_FINAL_MIN_FACTOR_NONZERO_RATIO", "0.10"))
FINAL_MIN_FACTOR_XS_STD_MEDIAN = float(os.environ.get("GP_FINAL_MIN_FACTOR_XS_STD_MEDIAN", "1e-8"))
FINAL_MIN_FACTOR_UNIQUE_MEDIAN = float(os.environ.get("GP_FINAL_MIN_FACTOR_UNIQUE_MEDIAN", "5"))
FINAL_FACTOR_NONZERO_EPS = float(os.environ.get("GP_FINAL_FACTOR_NONZERO_EPS", "1e-12"))
FINAL_YEARLY_STABILITY_WINDOWS = (
    ("2022", "20220102", "20230101"),
    ("2023", "20230101", "20240101"),
    ("2024", "20240101", "20250101"),
    ("2025", "20250101", "20260101"),
)
FINAL_MIN_YEARLY_LS_NETRET = float(os.environ.get("GP_FINAL_MIN_YEARLY_LS_NETRET", "0.0"))
FINAL_MIN_YEARLY_LS_NET_SHARPE = float(os.environ.get("GP_FINAL_MIN_YEARLY_LS_NET_SHARPE", "0.0"))
FINAL_MAX_YEARLY_TURNOVER = float(os.environ.get("GP_FINAL_MAX_YEARLY_TURNOVER", "0.85"))
FINAL_MIN_YEARLY_COVERAGE = float(os.environ.get("GP_FINAL_MIN_YEARLY_COVERAGE", "0.90"))
FINAL_MIN_LATEST_YEAR_LS_NETRET = float(os.environ.get("GP_FINAL_MIN_LATEST_YEAR_LS_NETRET", "-1000000"))
FINAL_MIN_LATEST_YEAR_LS_NET_SHARPE = float(os.environ.get("GP_FINAL_MIN_LATEST_YEAR_LS_NET_SHARPE", "-1000000"))
FINAL_MIN_LATEST_TO_MEAN_SHARPE_RATIO = float(os.environ.get("GP_FINAL_MIN_LATEST_TO_MEAN_SHARPE_RATIO", "-1000000"))
FINAL_MIN_YEARLY_RANKICIR = float(os.environ.get("GP_FINAL_MIN_YEARLY_RANKICIR", "-1000000"))
FINAL_MIN_LATEST_YEAR_RANKICIR = float(os.environ.get("GP_FINAL_MIN_LATEST_YEAR_RANKICIR", "-1000000"))
FINAL_REQUIRE_LATEST_RANKIC_SIGN = os.environ.get(
    "GP_FINAL_REQUIRE_LATEST_RANKIC_SIGN", "0"
).lower() in {"1", "true", "yes", "on"}
FINAL_PAIR_A_FAMILY_GATE_ENABLED = _env_flag("GP_FINAL_PAIR_A_FAMILY_GATE", "1")
FINAL_PAIR_A_FAMILY_MAX_SHARE = _json_env_dict(
    "GP_FINAL_PAIR_A_FAMILY_MAX_SHARE",
    PAIR_A_FAMILY_DEFAULT_MAX_SHARE,
)
FINAL_PAIR_A_FAMILY_DEFAULT_CAP = float(os.environ.get("GP_FINAL_PAIR_A_FAMILY_DEFAULT_CAP", "0.50"))
FINAL_TS_COMP_OP_MAX_SHARE = float(os.environ.get("GP_FINAL_TS_COMP_OP_MAX_SHARE", "0.50"))
FINAL_DIVERSITY_MAX_PER_OPERATOR = 4
FINAL_DIVERSITY_MAX_PER_A_FIELD = 3
FINAL_DIVERSITY_MAX_PER_SOURCE_FAMILY = 5
FINAL_DIVERSITY_MAX_PER_AB_PAIR = 2
FINAL_DIVERSITY_MAX_PER_TEMPLATE = 2
CANDIDATE_SLATE_DIVERSITY_LIMITS = {
    "operator": 20,
    "a_field": 7,
    "source_family": 24,
    "ab_pair": 5,
    "template": 2,
}
CANDIDATE_SLATE_FILL_LIMIT_MULT = float(
    os.environ.get("GP_SLATE_FILL_LIMIT_MULT", "4.0")
)
CANDIDATE_SLATE_MAX_COMPOSITION_SHARE = float(
    os.environ.get("GP_SLATE_MAX_COMPOSITION_SHARE", "0.15")
)
CANDIDATE_SLATE_MIN_ORDER_FLOW = max(
    0,
    int(os.environ.get("GP_SLATE_MIN_ORDER_FLOW", "8")),
)
CANDIDATE_SLATE_MIN_FIELD_SCOUT = max(
    0,
    int(os.environ.get("GP_SLATE_MIN_FIELD_SCOUT", "12")),
)
CANDIDATE_SLATE_ORDER_FLOW_SUPPLEMENT_MULT = max(
    1,
    int(os.environ.get("GP_SLATE_ORDER_FLOW_SUPPLEMENT_MULT", "16")),
)
CANDIDATE_SLATE_FIELD_SCOUT_SUPPLEMENT_MULT = max(
    1,
    int(os.environ.get("GP_SLATE_FIELD_SCOUT_SUPPLEMENT_MULT", "16")),
)
CANDIDATE_SLATE_RESERVE_XBINANCE_ORDER_FLOW = os.environ.get(
    "GP_SLATE_RESERVE_XBINANCE_ORDER_FLOW", "1"
).lower() in {"1", "true", "yes", "on"}
PREGATE_DIVERSITY_ORDER_ENABLED = os.environ.get(
    "GP_PREGATE_DIVERSITY_ORDER", "1"
).lower() in {"1", "true", "yes", "on"}
DISABLE_ARCHIVE_REPLAY = os.environ.get("GP_DISABLE_ARCHIVE_REPLAY", "0").lower() in {
    "1", "true", "yes", "on",
}
ARCHIVE_SOURCE = os.environ.get("GP_ARCHIVE_SOURCE", "active_approved").strip().lower()
CROWDED_AB_PAIRS_RAW = os.environ.get("GP_CROWDED_AB_PAIRS", "").strip()
CROWDED_AB_PAIR_MAX_SHARE = float(os.environ.get("GP_CROWDED_AB_PAIR_MAX_SHARE", "0.0"))
CROWDED_AB_PAIR_PENALTY = float(os.environ.get("GP_CROWDED_AB_PAIR_PENALTY", "0.35"))
BANNED_AB_PAIRS_RAW = os.environ.get("GP_BANNED_AB_PAIRS", "").strip()
AB_PAIR_SYMMETRIC = os.environ.get("GP_AB_PAIR_SYMMETRIC", "1").lower() in {
    "1", "true", "yes", "on",
}
EXTERNAL_BEHAVIOR_ARCHIVE_PATH = os.environ.get("GP_EXTERNAL_BEHAVIOR_ARCHIVE", "").strip()
AUTO_MERGE_ARCHIVE_AFTER_RUN = os.environ.get('GP_AUTO_MERGE_ARCHIVE_AFTER_RUN', '1').lower() in { '1', 'true', 'yes', 'on' }
AUTO_MERGE_ARCHIVE_TARGET = os.environ.get('GP_AUTO_MERGE_ARCHIVE_TARGET', '').strip()
EXTERNAL_CORR_SOFT_ENABLED = os.environ.get("GP_EXTERNAL_CORR_SOFT", "1").lower() in {
    "1", "true", "yes", "on",
}
EXTERNAL_CORR_TARGET = float(os.environ.get("GP_EXTERNAL_CORR_TARGET", "0.35"))
EXTERNAL_CORR_HARD = float(os.environ.get("GP_EXTERNAL_CORR_HARD", "0.55"))
EXTERNAL_CORR_STRENGTH = float(os.environ.get("GP_EXTERNAL_CORR_STRENGTH", "1.0"))
EXTERNAL_CORR_MIN_MULT = float(os.environ.get("GP_EXTERNAL_CORR_MIN_MULT", "0.05"))
EXTERNAL_CORR_FINAL_HARD = float(os.environ.get("GP_EXTERNAL_CORR_FINAL_HARD", str(EXTERNAL_CORR_HARD)))
FINAL_POOL_CORR_HARD = float(os.environ.get("GP_FINAL_POOL_CORR_HARD", "0.60"))
FINAL_BEHAVIOR_CORR_HARD = float(os.environ.get("GP_FINAL_BEHAVIOR_CORR_HARD", "0.55"))
SPEARMAN_CORR_HARD = float(os.environ.get("GP_SPEARMAN_CORR_HARD", "0.75"))
TAIL_OVERLAP_HARD = float(os.environ.get("GP_TAIL_OVERLAP_HARD", "0.70"))
FINAL_FIELD_FAMILY_MAX_SHARE = float(os.environ.get("GP_FINAL_FIELD_FAMILY_MAX_SHARE", "0.30"))
FINAL_AB_PAIR_MAX = max(1, int(os.environ.get("GP_FINAL_AB_PAIR_MAX", "1")))
ELITE_INIT_ENABLED = os.environ.get("GP_ELITE_INIT_ENABLED", "0").lower() in {
    "1", "true", "yes", "on",
}
ELITE_INIT_MULT = max(1.0, float(os.environ.get("GP_ELITE_INIT_MULT", "4.0")))
ELITE_INIT_MIN_NETRET = float(os.environ.get("GP_ELITE_INIT_MIN_NETRET", "-1000000"))
ELITE_INIT_MIN_SHARPE = float(os.environ.get("GP_ELITE_INIT_MIN_SHARPE", "0.05"))
ELITE_INIT_MIN_RANKICIR = float(os.environ.get("GP_ELITE_INIT_MIN_RANKICIR", "0.10"))
ELITE_INIT_MAX_TURNOVER = float(os.environ.get("GP_ELITE_INIT_MAX_TURNOVER", "1.20"))
ELITE_INIT_MIN_COVERAGE = float(os.environ.get("GP_ELITE_INIT_MIN_COVERAGE", "0.85"))
ELITE_INIT_POOL_CORR_HARD = float(os.environ.get("GP_ELITE_INIT_POOL_CORR_HARD", "0.70"))
ELITE_INIT_FIELD_FAMILY_MAX_SHARE = float(os.environ.get("GP_ELITE_INIT_FIELD_FAMILY_MAX_SHARE", "0.25"))
ELITE_INIT_AB_PAIR_MAX = max(1, int(os.environ.get("GP_ELITE_INIT_AB_PAIR_MAX", "3")))
ELITE_INIT_MAX_SCOUT = max(0, int(os.environ.get("GP_ELITE_INIT_MAX_SCOUT", "5000")))
SUBMIT_PRIOR_SHARE = float(np.clip(float(os.environ.get("GP_SUBMIT_PRIOR_SHARE", "0.0")), 0.0, 0.80))

_PROVEN_RELATION_TEMPLATES = (
    {
        "a": ("price_range_pct", "return_abs"),
        "b": ("turnover", "money_flow", "volume_intensity"),
        "ops": ("slope", "ratio_std", "cov"),
        "windows": (90, 120, 180, 240),
        "slices": (None, 0.3, 0.5),
        "masks": (
            ("turnover", "high_0.8"),
            ("return_abs", "high_0.8"),
            ("price_range_pct", "high_0.7"),
            ("open", "none"),
        ),
        "lags": (0, 1, 2, 3),
    },
    {
        "a": ("money_flow", "turnover", "volume_intensity"),
        "b": ("returns",),
        "ops": ("ratio_std", "cov", "slope"),
        "windows": (120, 180, 240),
        "slices": (None, 0.3, 0.5),
        "masks": (
            ("funding", "high_0.8"),
            ("funding", "high_0.5"),
            ("open_interest", "high_0.8"),
            ("turnover", "high_0.8"),
        ),
        "lags": (0, 1, 2, 3),
    },
    {
        "a": ("turnover", "money_flow", "typical_price"),
        "b": ("premium_close", "mark_close", "index_close"),
        "ops": ("ratio_mean", "cov", "ratio_std"),
        "windows": (120, 180, 240),
        "slices": (None, 0.3),
        "masks": (
            ("funding", "high_0.8"),
            ("funding", "low_0.7"),
            ("open_interest", "high_0.8"),
            ("open_interest", "low_0.5"),
        ),
        "lags": (0, 1, 2),
    },
    {
        "a": ("premium_close", "mark_close", "index_close"),
        "b": ("turnover", "money_flow", "returns"),
        "ops": ("cov", "slope", "ratio_mean"),
        "windows": (120, 180, 240),
        "slices": (None, 0.3),
        "masks": (
            ("funding", "high_0.5"),
            ("funding", "low_0.7"),
            ("turnover", "high_0.6"),
            ("open", "none"),
        ),
        "lags": (0, 1, 2, 3),
    },
    {
        "a": (
            "mark_index_spread",
            "mark_index_spread_change",
            "premium_change_intraday",
            "oi_change_intraday",
            "oi_price_alignment",
            "funding_abs",
        ),
        "b": ("turnover", "money_flow", "return_abs", "price_range_pct"),
        "ops": ("cov", "slope", "ratio_mean", "ratio_std"),
        "windows": (90, 120, 180, 240),
        "slices": (None, 0.3, 0.5),
        "masks": (
            ("market_vol_intensity", "high_0.8"),
            ("market_vol_intensity", "low_0.3"),
            ("market_cum_return", "high_0.8"),
            ("market_cum_return", "low_0.3"),
            ("funding", "high_0.8"),
            ("open_interest", "high_0.8"),
        ),
        "lags": (0, 1, 2, 3),
    },
)

_ORDER_FLOW_TEMPLATE_FIELDS = tuple(
    name for name in (
        "xbinance_taker_pressure",
        "xbinance_taker_quote_ratio",
        "xbinance_avg_trade_size",
        "taker_pressure",
        "taker_buy_ratio",
        "taker_quote_ratio",
        "avg_trade_size",
    )
    if name in INDICATOR_INDEX
)
_ORDER_FLOW_CONTEXT_FIELDS = tuple(
    name for name in (
        "returns",
        "return_abs",
        "price_range_pct",
        "money_flow",
        "turnover",
        "dollar_volume_rank",
        "volume_zscore_60d",
        "rv_5d",
        "rv_20d",
        "rv_ratio_5_20",
        "funding",
        "funding_z20",
        "funding_oi_joint_5d",
        "open_interest",
        "oi_change_5d",
        "market_vol_intensity",
        "relative_strength_btc_5d",
    )
    if name in INDICATOR_INDEX
)
_ORDER_FLOW_MASK_PAIRS = (
    ("open", "none"),
    ("turnover", "high_0.8"),
    ("dollar_volume_rank", "high_0.8"),
    ("dollar_volume_rank", "low_0.3"),
    ("volume_zscore_60d", "high_0.8"),
    ("rv_5d", "high_0.8"),
    ("market_vol_intensity", "high_0.8"),
    ("market_vol_intensity", "low_0.3"),
    ("funding", "high_0.8"),
    ("open_interest", "high_0.8"),
)
_FIELD_SCOUT_TEMPLATE_FIELDS = tuple(
    name for name in (
        "amihud_illiq",
        "volume_zscore_60d",
        "dollar_volume_rank",
        "path_efficiency",
        "high_time_frac",
        "low_time_frac",
        "volume_burst_share",
        "intraday_reversal",
        "funding_z20",
        "funding_vol_30d",
        "oi_z20",
        "oi_change_5d",
        "funding_oi_joint_5d",
        "beta_btc_60d",
        "resid_return_btc_20d",
        "corr_btc_20d",
        "relative_strength_btc_5d",
        "rv_5d",
        "rv_20d",
        "rv_ratio_5_20",
    )
    if name in INDICATOR_INDEX
)
_FIELD_SCOUT_CONTEXT_FIELDS = tuple(
    name for name in (
        "returns",
        "return_abs",
        "price_range_pct",
        "money_flow",
        "turnover",
        "rv_5d",
        "rv_20d",
        "funding",
        "open_interest",
        "mark_close",
        "index_close",
        "funding_z20",
        "funding_oi_joint_5d",
        "beta_btc_60d",
        "relative_strength_btc_5d",
        "xbinance_taker_pressure",
        "xbinance_taker_quote_ratio",
    )
    if name in INDICATOR_INDEX
)
_FIELD_SCOUT_MASK_PAIRS = (
    ("open", "none"),
    ("amihud_illiq", "high_0.8"),
    ("volume_zscore_60d", "high_0.8"),
    ("dollar_volume_rank", "high_0.8"),
    ("dollar_volume_rank", "low_0.3"),
    ("path_efficiency", "high_0.8"),
    ("path_efficiency", "low_0.3"),
    ("intraday_reversal", "high_0.8"),
    ("volume_burst_share", "high_0.8"),
    ("funding_z20", "high_0.8"),
    ("funding_z20", "low_0.3"),
    ("funding_oi_joint_5d", "high_0.8"),
    ("funding_oi_joint_5d", "low_0.3"),
    ("oi_z20", "high_0.8"),
    ("beta_btc_60d", "high_0.8"),
    ("relative_strength_btc_5d", "high_0.8"),
    ("relative_strength_btc_5d", "low_0.3"),
    ("corr_btc_20d", "high_0.8"),
    ("corr_btc_20d", "low_0.3"),
    ("rv_ratio_5_20", "high_0.8"),
    ("turnover", "high_0.8"),
    ("return_abs", "high_0.8"),
    ("price_range_pct", "high_0.7"),
    ("funding", "high_0.8"),
    ("open_interest", "high_0.8"),
)


def _normalize_probs(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    arr = np.where(np.isfinite(arr) & (arr >= 0.0), arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0 or not np.isfinite(total):
        return None
    return arr / total


def _blend_probs_with_uniform(probs, alpha):
    if probs is None:
        return None
    alpha = float(np.clip(alpha, 0.0, 1.0))
    uniform = np.full(len(probs), 1.0 / len(probs), dtype=np.float64)
    mixed = uniform * (1.0 - alpha) + np.asarray(probs, dtype=np.float64) * alpha
    return mixed / mixed.sum()


def _mode_base_probs(state):
    single = float(np.clip(state.get("single_share", 0.5), 0.0, 1.0))
    compound = float(np.clip(state.get("compound_share", _COMPOUND_SHARE), 0.0, 1.0))
    intraday = float(np.clip(state.get("intraday_share", 0.0), 0.0, 1.0))
    total = single + compound + intraday
    if total > 1.0:
        scale = 1.0 / total
        single *= scale
        compound *= scale
        intraday *= scale
    pair = max(1.0 - single - compound - intraday, 0.0)
    return _normalize_probs([single, pair, compound, intraday])


def _weight_lookup(weight_map, key):
    if not isinstance(weight_map, dict):
        return 1.0
    try:
        value = float(weight_map.get(str(key), 1.0))
    except (TypeError, ValueError):
        return 1.0
    return value if np.isfinite(value) and value > 0.0 else 1.0


def _count_lookup(count_map, key):
    if not isinstance(count_map, dict):
        return 0
    try:
        return int(count_map.get(str(key), 0))
    except (TypeError, ValueError):
        return 0


def _component_approved_counts(component_stats, key):
    items = component_stats.get(key, []) if isinstance(component_stats, dict) else []
    counts = {}
    for item in items:
        try:
            counts[str(item["value"])] = int(item.get("approved_count", 0))
        except (KeyError, TypeError, ValueError):
            continue
    return counts


def _crowd_penalty(count):
    try:
        value = int(count)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        return 1.10
    penalty = 1.0 / np.sqrt(1.0 + 0.35 * float(value))
    return float(np.clip(penalty, 0.55, 1.05))


def _build_pool_probs(
    choices,
    *,
    exact_weights=None,
    exact_key=None,
    family_weights=None,
    family_key=None,
    exact_crowd=None,
    family_crowd=None,
):
    multipliers = []
    for choice in choices:
        exact_value = exact_key(choice) if exact_key is not None else choice
        weight = _weight_lookup(exact_weights, exact_value)
        weight *= _crowd_penalty(_count_lookup(exact_crowd, exact_value)) if exact_crowd is not None else 1.0
        if family_key is not None:
            family_value = family_key(choice)
            weight *= _weight_lookup(family_weights, family_value)
            weight *= _crowd_penalty(_count_lookup(family_crowd, family_value)) if family_crowd is not None else 1.0
        multipliers.append(max(weight, 0.05))
    return _normalize_probs(multipliers)


def _build_mask_pair_probs(
    choices,
    *,
    rule_weights=None,
    family_weights=None,
    rule_crowd=None,
    family_crowd=None,
):
    multipliers = []
    for field_idx, rule_idx in choices:
        field_name = INDICATOR_NAMES[int(field_idx)]
        rule_name = MASK_RULES[int(rule_idx)]
        weight = _weight_lookup(rule_weights, rule_name)
        if rule_crowd is not None:
            weight *= _crowd_penalty(_count_lookup(rule_crowd, rule_name))
        weight *= _weight_lookup(family_weights, indicator_source_family(field_name))
        if family_crowd is not None:
            weight *= _crowd_penalty(_count_lookup(family_crowd, indicator_source_family(field_name)))
        multipliers.append(max(weight, 0.05))
    return _normalize_probs(multipliers)


def _mode_probabilities(state, feedback, *, primary):
    base = _mode_base_probs(state)
    if feedback is None:
        return base
    probs = feedback.get("mode_primary" if primary else "mode_explore")
    return probs if probs is not None else base


def _build_sampling_feedback(state, feedback_artifact):
    if not isinstance(feedback_artifact, dict):
        return None
    sampler_weights = feedback_artifact.get("sampler_weights")
    if not isinstance(sampler_weights, dict) or not sampler_weights:
        return None

    indicator_weights = sampler_weights.get("indicator", {})
    source_family_weights = sampler_weights.get("source_family", {})
    operator_family_weights = sampler_weights.get("operator_family", {})
    window_weights = sampler_weights.get("window", {})
    window_bucket_weights = sampler_weights.get("window_bucket", {})
    mode_weights = sampler_weights.get("mode", {})
    mode1_weights = sampler_weights.get("mode1_op", {})
    mode2_weights = sampler_weights.get("mode2_op", {})
    mode3_weights = sampler_weights.get("mode3_op", {})
    mask_rule_weights = sampler_weights.get("mask_rule", {})
    mask_field_family_weights = sampler_weights.get("mask_field_family", {})
    lag_weights = sampler_weights.get("lag", {})
    ts_op_weights = sampler_weights.get("ts_comp_op", {})
    ts_window_weights = sampler_weights.get("ts_comp_window", {})
    cs_op_weights = sampler_weights.get("cs_comp_op", {})
    component_stats = feedback_artifact.get("component_stats", {})
    if bool(state.get("use_feedback_crowd_penalty", True)):
        indicator_crowd = _component_approved_counts(component_stats, "indicator")
        source_family_crowd = _component_approved_counts(component_stats, "source_family")
        operator_family_crowd = _component_approved_counts(component_stats, "operator_family")
        window_crowd = _component_approved_counts(component_stats, "window")
        window_bucket_crowd = _component_approved_counts(component_stats, "window_bucket")
        mode_crowd = _component_approved_counts(component_stats, "mode")
        mode1_crowd = _component_approved_counts(component_stats, "mode1_op")
        mode2_crowd = _component_approved_counts(component_stats, "mode2_op")
        mode3_crowd = _component_approved_counts(component_stats, "mode3_op")
        mask_rule_crowd = _component_approved_counts(component_stats, "mask_rule")
        mask_family_crowd = _component_approved_counts(component_stats, "mask_field_family")
        lag_crowd = _component_approved_counts(component_stats, "lag")
        ts_op_crowd = _component_approved_counts(component_stats, "ts_comp_op")
        ts_family_crowd = _component_approved_counts(component_stats, "ts_family")
        ts_window_crowd = _component_approved_counts(component_stats, "ts_comp_window")
        cs_op_crowd = _component_approved_counts(component_stats, "cs_comp_op")
        cs_family_crowd = _component_approved_counts(component_stats, "cs_family")
    else:
        indicator_crowd = source_family_crowd = operator_family_crowd = None
        window_crowd = window_bucket_crowd = mode_crowd = None
        mode1_crowd = mode2_crowd = mode3_crowd = None
        mask_rule_crowd = mask_family_crowd = lag_crowd = None
        ts_op_crowd = ts_family_crowd = ts_window_crowd = None
        cs_op_crowd = cs_family_crowd = None

    def stage_probs(primary_builder, all_builder):
        return (
            _blend_probs_with_uniform(
                primary_builder(),
                float(state.get("feedback_primary_blend", _PRIMARY_FEEDBACK_BLEND)),
            ),
            _blend_probs_with_uniform(
                all_builder(),
                float(state.get("feedback_explore_blend", _EXPLORE_FEEDBACK_BLEND)),
            ),
        )

    single_field_primary, single_field_all = stage_probs(
        lambda: _build_pool_probs(
            state["single_field_primary"],
            exact_weights=indicator_weights,
            exact_key=lambda idx: INDICATOR_NAMES[int(idx)],
            family_weights=source_family_weights,
            family_key=lambda idx: indicator_source_family(INDICATOR_NAMES[int(idx)]),
            exact_crowd=indicator_crowd,
            family_crowd=source_family_crowd,
        ),
        lambda: _build_pool_probs(
            state["single_field_all"],
            exact_weights=indicator_weights,
            exact_key=lambda idx: INDICATOR_NAMES[int(idx)],
            family_weights=source_family_weights,
            family_key=lambda idx: indicator_source_family(INDICATOR_NAMES[int(idx)]),
            exact_crowd=indicator_crowd,
            family_crowd=source_family_crowd,
        ),
    )
    pair_a_field_primary, pair_a_field_all = stage_probs(
        lambda: _build_pool_probs(
            state["pair_a_field_primary"],
            exact_weights=indicator_weights,
            exact_key=lambda idx: INDICATOR_NAMES[int(idx)],
            family_weights=source_family_weights,
            family_key=lambda idx: indicator_source_family(INDICATOR_NAMES[int(idx)]),
            exact_crowd=indicator_crowd,
            family_crowd=source_family_crowd,
        ),
        lambda: _build_pool_probs(
            state["pair_a_field_all"],
            exact_weights=indicator_weights,
            exact_key=lambda idx: INDICATOR_NAMES[int(idx)],
            family_weights=source_family_weights,
            family_key=lambda idx: indicator_source_family(INDICATOR_NAMES[int(idx)]),
            exact_crowd=indicator_crowd,
            family_crowd=source_family_crowd,
        ),
    )
    pair_b_field_primary, pair_b_field_all = stage_probs(
        lambda: _build_pool_probs(
            state["pair_b_field_primary"],
            exact_weights=indicator_weights,
            exact_key=lambda idx: INDICATOR_NAMES[int(idx)],
            family_weights=source_family_weights,
            family_key=lambda idx: indicator_source_family(INDICATOR_NAMES[int(idx)]),
            exact_crowd=indicator_crowd,
            family_crowd=source_family_crowd,
        ),
        lambda: _build_pool_probs(
            state["pair_b_field_all"],
            exact_weights=indicator_weights,
            exact_key=lambda idx: INDICATOR_NAMES[int(idx)],
            family_weights=source_family_weights,
            family_key=lambda idx: indicator_source_family(INDICATOR_NAMES[int(idx)]),
            exact_crowd=indicator_crowd,
            family_crowd=source_family_crowd,
        ),
    )
    single_window_primary, single_window_all = stage_probs(
        lambda: _build_pool_probs(
            state["single_window_primary"],
            exact_weights=window_weights,
            exact_key=lambda idx: WINDOW_CHOICES[int(idx)],
            family_weights=window_bucket_weights,
            family_key=lambda idx: window_bucket(WINDOW_CHOICES[int(idx)]),
            exact_crowd=window_crowd,
            family_crowd=window_bucket_crowd,
        ),
        lambda: _build_pool_probs(
            state["single_window_all"],
            exact_weights=window_weights,
            exact_key=lambda idx: WINDOW_CHOICES[int(idx)],
            family_weights=window_bucket_weights,
            family_key=lambda idx: window_bucket(WINDOW_CHOICES[int(idx)]),
            exact_crowd=window_crowd,
            family_crowd=window_bucket_crowd,
        ),
    )
    pair_window_primary, pair_window_all = stage_probs(
        lambda: _build_pool_probs(
            state["pair_window_primary"],
            exact_weights=window_weights,
            exact_key=lambda idx: WINDOW_CHOICES[int(idx)],
            family_weights=window_bucket_weights,
            family_key=lambda idx: window_bucket(WINDOW_CHOICES[int(idx)]),
            exact_crowd=window_crowd,
            family_crowd=window_bucket_crowd,
        ),
        lambda: _build_pool_probs(
            state["pair_window_all"],
            exact_weights=window_weights,
            exact_key=lambda idx: WINDOW_CHOICES[int(idx)],
            family_weights=window_bucket_weights,
            family_key=lambda idx: window_bucket(WINDOW_CHOICES[int(idx)]),
            exact_crowd=window_crowd,
            family_crowd=window_bucket_crowd,
        ),
    )
    mask_pairs_primary, mask_pairs_all = stage_probs(
        lambda: _build_mask_pair_probs(
            state["mask_pairs_primary"],
            rule_weights=mask_rule_weights,
            family_weights=mask_field_family_weights,
            rule_crowd=mask_rule_crowd,
            family_crowd=mask_family_crowd,
        ),
        lambda: _build_mask_pair_probs(
            state["mask_pairs_all"],
            rule_weights=mask_rule_weights,
            family_weights=mask_field_family_weights,
            rule_crowd=mask_rule_crowd,
            family_crowd=mask_family_crowd,
        ),
    )
    mode1_op_primary, mode1_op_all = stage_probs(
        lambda: _build_pool_probs(
            state["mode1_op_primary"],
            exact_weights=mode1_weights,
            exact_key=lambda idx: MODE1_OPS[int(idx)],
            family_weights=operator_family_weights,
            family_key=lambda idx: mode1_operator_family(MODE1_OPS[int(idx)]),
            exact_crowd=mode1_crowd,
            family_crowd=operator_family_crowd,
        ),
        lambda: _build_pool_probs(
            state["mode1_op_all"],
            exact_weights=mode1_weights,
            exact_key=lambda idx: MODE1_OPS[int(idx)],
            family_weights=operator_family_weights,
            family_key=lambda idx: mode1_operator_family(MODE1_OPS[int(idx)]),
            exact_crowd=mode1_crowd,
            family_crowd=operator_family_crowd,
        ),
    )
    mode2_op_primary, mode2_op_all = stage_probs(
        lambda: _build_pool_probs(
            state["mode2_op_primary"],
            exact_weights=mode2_weights,
            exact_key=lambda idx: MODE2_OPS[int(idx)],
            family_weights=operator_family_weights,
            family_key=lambda idx: mode2_operator_family(MODE2_OPS[int(idx)]),
            exact_crowd=mode2_crowd,
            family_crowd=operator_family_crowd,
        ),
        lambda: _build_pool_probs(
            state["mode2_op_all"],
            exact_weights=mode2_weights,
            exact_key=lambda idx: MODE2_OPS[int(idx)],
            family_weights=operator_family_weights,
            family_key=lambda idx: mode2_operator_family(MODE2_OPS[int(idx)]),
            exact_crowd=mode2_crowd,
            family_crowd=operator_family_crowd,
        ),
    )
    mode3_op_primary, mode3_op_all = stage_probs(
        lambda: _build_pool_probs(
            state["mode3_op_primary"],
            exact_weights=mode3_weights,
            exact_key=lambda idx: MODE3_OPS[int(idx)],
            family_weights=operator_family_weights,
            family_key=lambda idx: mode3_operator_family(MODE3_OPS[int(idx)]),
            exact_crowd=mode3_crowd,
            family_crowd=operator_family_crowd,
        ),
        lambda: _build_pool_probs(
            state["mode3_op_all"],
            exact_weights=mode3_weights,
            exact_key=lambda idx: MODE3_OPS[int(idx)],
            family_weights=operator_family_weights,
            family_key=lambda idx: mode3_operator_family(MODE3_OPS[int(idx)]),
            exact_crowd=mode3_crowd,
            family_crowd=operator_family_crowd,
        ),
    )
    mode4_op_primary, mode4_op_all = stage_probs(
        lambda: _build_pool_probs(
            state["mode4_op_primary"],
            exact_weights=sampler_weights.get("mode4_op", {}),
            exact_key=lambda idx: MODE4_OPS[int(idx)],
            family_weights=operator_family_weights,
            family_key=lambda idx: mode4_operator_family(MODE4_OPS[int(idx)]),
            family_crowd=operator_family_crowd,
        ),
        lambda: _build_pool_probs(
            state["mode4_op_all"],
            exact_weights=sampler_weights.get("mode4_op", {}),
            exact_key=lambda idx: MODE4_OPS[int(idx)],
            family_weights=operator_family_weights,
            family_key=lambda idx: mode4_operator_family(MODE4_OPS[int(idx)]),
            family_crowd=operator_family_crowd,
        ),
    )
    lag_primary, lag_all = stage_probs(
        lambda: _build_pool_probs(
            state["lag_primary"],
            exact_weights=lag_weights,
            exact_key=lambda idx: B_SHIFT_CHOICES[int(idx)],
            exact_crowd=lag_crowd,
        ),
        lambda: _build_pool_probs(
            state["lag_all"],
            exact_weights=lag_weights,
            exact_key=lambda idx: B_SHIFT_CHOICES[int(idx)],
            exact_crowd=lag_crowd,
        ),
    )
    ts_comp_op_primary, ts_comp_op_all = stage_probs(
        lambda: _build_pool_probs(
            state["ts_comp_op_all"],
            exact_weights=ts_op_weights,
            exact_key=lambda idx: TS_COMP_OPS[int(idx)],
            family_weights=sampler_weights.get("ts_family", {}),
            family_key=lambda idx: ts_operator_family(TS_COMP_OPS[int(idx)]),
            exact_crowd=ts_op_crowd,
            family_crowd=ts_family_crowd,
        ),
        lambda: _build_pool_probs(
            state["ts_comp_op_all"],
            exact_weights=ts_op_weights,
            exact_key=lambda idx: TS_COMP_OPS[int(idx)],
            family_weights=sampler_weights.get("ts_family", {}),
            family_key=lambda idx: ts_operator_family(TS_COMP_OPS[int(idx)]),
            exact_crowd=ts_op_crowd,
            family_crowd=ts_family_crowd,
        ),
    )
    ts_comp_window_primary, ts_comp_window_all = stage_probs(
        lambda: _build_pool_probs(
            state["ts_comp_window_all"],
            exact_weights=ts_window_weights,
            exact_key=lambda idx: TS_COMP_WINDOWS[int(idx)],
            exact_crowd=ts_window_crowd,
        ),
        lambda: _build_pool_probs(
            state["ts_comp_window_all"],
            exact_weights=ts_window_weights,
            exact_key=lambda idx: TS_COMP_WINDOWS[int(idx)],
            exact_crowd=ts_window_crowd,
        ),
    )
    cs_comp_op_primary, cs_comp_op_all = stage_probs(
        lambda: _build_pool_probs(
            state["cs_comp_op_all"],
            exact_weights=cs_op_weights,
            exact_key=lambda idx: CS_COMP_OPS[int(idx)],
            family_weights=sampler_weights.get("cs_family", {}),
            family_key=lambda idx: cs_operator_family(CS_COMP_OPS[int(idx)]),
            exact_crowd=cs_op_crowd,
            family_crowd=cs_family_crowd,
        ),
        lambda: _build_pool_probs(
            state["cs_comp_op_all"],
            exact_weights=cs_op_weights,
            exact_key=lambda idx: CS_COMP_OPS[int(idx)],
            family_weights=sampler_weights.get("cs_family", {}),
            family_key=lambda idx: cs_operator_family(CS_COMP_OPS[int(idx)]),
            exact_crowd=cs_op_crowd,
            family_crowd=cs_family_crowd,
        ),
    )

    base_mode = _mode_base_probs(state)
    mode_multiplier = np.asarray(
        [
            _weight_lookup(mode_weights, 1) * _crowd_penalty(_count_lookup(mode_crowd, 1)),
            _weight_lookup(mode_weights, 2) * _crowd_penalty(_count_lookup(mode_crowd, 2)),
            _weight_lookup(mode_weights, 3) * _crowd_penalty(_count_lookup(mode_crowd, 3)),
            _weight_lookup(mode_weights, 4) * _crowd_penalty(_count_lookup(mode_crowd, 4)),
        ],
        dtype=np.float64,
    )
    weighted_mode = _normalize_probs(base_mode * mode_multiplier)
    mode_primary = _blend_probs_with_uniform(
        weighted_mode if weighted_mode is not None else base_mode,
        float(state.get("feedback_primary_blend", _PRIMARY_FEEDBACK_BLEND)),
    )
    mode_explore = _blend_probs_with_uniform(
        weighted_mode if weighted_mode is not None else base_mode,
        float(state.get("feedback_explore_blend", _EXPLORE_FEEDBACK_BLEND)),
    )

    meta = {
        "artifact_path": feedback_artifact.get("artifact_path"),
        "generated_at": feedback_artifact.get("generated_at"),
        "candidate_count": int(feedback_artifact.get("candidate_count", 0)),
        "top_family": ((feedback_artifact.get("top_families") or [{}])[0]).get("value"),
        "top_behavior": ((feedback_artifact.get("top_behaviors") or [{}])[0]).get("value"),
    }
    return {
        "mode_primary": mode_primary,
        "mode_explore": mode_explore,
        "single_field_primary": single_field_primary,
        "single_field_all": single_field_all,
        "pair_a_field_primary": pair_a_field_primary,
        "pair_a_field_all": pair_a_field_all,
        "pair_b_field_primary": pair_b_field_primary,
        "pair_b_field_all": pair_b_field_all,
        "single_window_primary": single_window_primary,
        "single_window_all": single_window_all,
        "pair_window_primary": pair_window_primary,
        "pair_window_all": pair_window_all,
        "mask_pairs_primary": mask_pairs_primary,
        "mask_pairs_all": mask_pairs_all,
        "mode1_op_primary": mode1_op_primary,
        "mode1_op_all": mode1_op_all,
        "mode2_op_primary": mode2_op_primary,
        "mode2_op_all": mode2_op_all,
        "mode3_op_primary": mode3_op_primary,
        "mode3_op_all": mode3_op_all,
        "lag_primary": lag_primary,
        "lag_all": lag_all,
        "ts_comp_op_primary": ts_comp_op_primary,
        "ts_comp_op_all": ts_comp_op_all,
        "ts_comp_window_primary": ts_comp_window_primary,
        "ts_comp_window_all": ts_comp_window_all,
        "cs_comp_op_primary": cs_comp_op_primary,
        "cs_comp_op_all": cs_comp_op_all,
        "meta": meta,
    }


def _generate_policy_individual(rng, state, *, explore=False, mode_override=None):
    row = np.zeros(N_PARAMS, dtype=np.int32)
    primary = not explore
    feedback = state.get("sampling_feedback")
    mode4_key = "mode4_op_primary" if primary else "mode4_op_all"
    mode4_choices = tuple(state.get(mode4_key, ()))
    if mode_override is not None:
        mode_choice = int(mode_override)
    else:
        mode_probs = np.asarray(_mode_probabilities(state, feedback, primary=primary), dtype=np.float64)
        if not mode4_choices:
            mode_probs[3] = 0.0
            mode_probs = _normalize_probs(mode_probs)
            if mode_probs is None:
                mode_probs = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float64)
        mode_choice = int(rng.choice(4, p=mode_probs))
    if mode_choice == 3 and not mode4_choices:
        # Strict profiles can disable mode4; structural exploration may still
        # request it explicitly, so fall back to ordinary pair mode.
        mode_choice = 1
    row[6] = mode_choice

    if mode_choice == 0:
        field_pool = state["single_field_primary"] if primary else state["single_field_all"]
        op_pool = state["mode1_op_primary"] if primary else state["mode1_op_all"]
        window_pool = state["single_window_primary"] if primary else state["single_window_all"]
        slice_pool = state["single_slice_primary"] if primary else state["single_slice_all"]
        mask_pool = state["mask_pairs_primary"] if primary else state["mask_pairs_all"]
        field_probs = feedback.get("single_field_primary" if primary else "single_field_all") if feedback else None
        op_probs = feedback.get("mode1_op_primary" if primary else "mode1_op_all") if feedback else None
        window_probs = feedback.get("single_window_primary" if primary else "single_window_all") if feedback else None
        mask_probs = feedback.get("mask_pairs_primary" if primary else "mask_pairs_all") if feedback else None
        row[0] = _sample_from_pool(rng, field_pool, field_probs)
        row[1] = row[0]
        row[7] = _sample_from_pool(rng, op_pool, op_probs)
        row[8] = 0
        row[9] = B_SHIFT_CHOICES.index(0)
        row[2] = _sample_from_pool(rng, window_pool, window_probs)
        if WINDOW_CHOICES[int(row[2])] >= MINUTES_PER_PERIOD:
            row[3] = SLICE_CHOICES.index(None)
        else:
            row[3] = _sample_from_pool(rng, slice_pool)
        row[4], row[5] = _sample_mask_pair(rng, mask_pool, mask_probs)
    elif mode_choice == 1:
        a_pool = state["pair_a_field_primary"] if primary else state["pair_a_field_all"]
        b_pool = state["pair_b_field_primary"] if primary else state["pair_b_field_all"]
        op_pool = state["mode2_op_primary"] if primary else state["mode2_op_all"]
        window_pool = state["pair_window_primary"] if primary else state["pair_window_all"]
        slice_pool = state["pair_slice_primary"] if primary else state["pair_slice_all"]
        lag_pool = state["lag_primary"] if primary else state["lag_all"]
        mask_pool = state["mask_pairs_primary"] if primary else state["mask_pairs_all"]
        a_probs = feedback.get("pair_a_field_primary" if primary else "pair_a_field_all") if feedback else None
        b_probs = feedback.get("pair_b_field_primary" if primary else "pair_b_field_all") if feedback else None
        op_probs = feedback.get("mode2_op_primary" if primary else "mode2_op_all") if feedback else None
        window_probs = feedback.get("pair_window_primary" if primary else "pair_window_all") if feedback else None
        lag_probs = feedback.get("lag_primary" if primary else "lag_all") if feedback else None
        mask_probs = feedback.get("mask_pairs_primary" if primary else "mask_pairs_all") if feedback else None
        row[0] = _sample_from_pool(rng, a_pool, a_probs)
        row[1] = _sample_from_pool(rng, b_pool, b_probs)
        row[7] = 0
        row[8] = _sample_from_pool(rng, op_pool, op_probs)
        row[2] = _sample_from_pool(rng, window_pool, window_probs)
        if WINDOW_CHOICES[int(row[2])] >= MINUTES_PER_PERIOD:
            row[3] = SLICE_CHOICES.index(None)
        else:
            row[3] = _sample_from_pool(rng, slice_pool)
        row[9] = _sample_from_pool(rng, lag_pool, lag_probs)
        row[4], row[5] = _sample_mask_pair(rng, mask_pool, mask_probs)
    elif mode_choice == 2:
        field_pool = state["single_field_primary"] if primary else state["single_field_all"]
        window_pool = state["single_window_primary"] if primary else state["single_window_all"]
        slice_pool = state["single_slice_primary"] if primary else state["single_slice_all"]
        mask_pool = state["mask_pairs_primary"] if primary else state["mask_pairs_all"]
        mode3_pool = state["mode3_op_primary"] if primary else state["mode3_op_all"]
        field_probs = feedback.get("single_field_primary" if primary else "single_field_all") if feedback else None
        window_probs = feedback.get("single_window_primary" if primary else "single_window_all") if feedback else None
        mask_probs = feedback.get("mask_pairs_primary" if primary else "mask_pairs_all") if feedback else None
        mode3_probs = feedback.get("mode3_op_primary" if primary else "mode3_op_all") if feedback else None
        row[0] = _sample_from_pool(rng, field_pool, field_probs)
        row[1] = row[0]
        row[7] = 0
        row[8] = 0
        row[9] = B_SHIFT_CHOICES.index(0)
        row[13] = _sample_from_pool(rng, mode3_pool, mode3_probs)
        short_idx = _sample_from_pool(rng, window_pool, window_probs)
        row[2] = short_idx
        if WINDOW_CHOICES[int(short_idx)] >= MINUTES_PER_PERIOD:
            row[3] = SLICE_CHOICES.index(None)
        else:
            row[3] = _sample_from_pool(rng, slice_pool)
        long_mult_pool = [i for i in range(len(TS_COMP_WINDOWS)) if TS_COMP_WINDOWS[i] >= 2]
        row[11] = int(rng.choice(long_mult_pool)) if long_mult_pool else 0
        row[4], row[5] = _sample_mask_pair(rng, mask_pool, mask_probs)
    else:
        a_pool = state["mode4_a_field_primary"] if primary else state["mode4_a_field_all"]
        b_pool = state["pair_b_field_primary"] if primary else state["pair_b_field_all"]
        window_pool = state["pair_window_primary"] if primary else state["pair_window_all"]
        slice_pool = state["pair_slice_primary"] if primary else state["pair_slice_all"]
        mask_pool = state["mask_pairs_primary"] if primary else state["mask_pairs_all"]
        a_probs = None
        b_probs = feedback.get("pair_b_field_primary" if primary else "pair_b_field_all") if feedback else None
        window_probs = feedback.get("pair_window_primary" if primary else "pair_window_all") if feedback else None
        mask_probs = feedback.get("mask_pairs_primary" if primary else "mask_pairs_all") if feedback else None
        mode4_probs = feedback.get("mode4_op_primary" if primary else "mode4_op_all") if feedback else None
        row[0] = _sample_from_pool(rng, a_pool, a_probs)
        row[1] = _sample_from_pool(rng, b_pool, b_probs)
        row[7] = 0
        row[8] = 0
        row[9] = B_SHIFT_CHOICES.index(0)
        mode4_pool = state["mode4_op_primary"] if primary else state.get("mode4_op_all", tuple(range(len(MODE4_OPS))))
        row[13] = _sample_from_pool(rng, mode4_pool, mode4_probs)
        row[14] = int(rng.integers(0, len(INTRADAY_GROUP_CHOICES)))
        row[2] = _sample_from_pool(rng, window_pool, window_probs)
        if WINDOW_CHOICES[int(row[2])] >= MINUTES_PER_PERIOD:
            row[3] = SLICE_CHOICES.index(None)
        else:
            row[3] = _sample_from_pool(rng, slice_pool)
        row[4], row[5] = _sample_mask_pair(rng, mask_pool, mask_probs)

    none_ts_idx = TS_COMP_OPS.index('none')
    none_cs_idx = CS_COMP_OPS.index('none')
    active_ts = tuple(idx for idx in state["ts_comp_op_all"] if TS_COMP_OPS[int(idx)] != "none")
    active_cs = tuple(idx for idx in state["cs_comp_op_all"] if CS_COMP_OPS[int(idx)] != "none")
    active_ts_windows = tuple(
        idx for idx in state["ts_comp_window_all"]
        if TS_COMP_WINDOWS[int(idx)] > 1
    ) or tuple(state["ts_comp_window_all"])
    use_comp = rng.random() < float(np.clip(state.get("composition_share", 0.0), 0.0, 1.0))
    use_cs = rng.random() < float(np.clip(state.get("composition_cs_share", 0.5), 0.0, 1.0))
    row[10] = none_ts_idx
    row[12] = none_cs_idx
    if use_comp and use_cs and active_cs:
        row[12] = _sample_from_pool(rng, active_cs)
    elif use_comp and mode_choice != 2 and active_ts:
        row[10] = _sample_from_pool(rng, active_ts)
        row[11] = _sample_from_pool(rng, active_ts_windows)
    elif mode_choice != 2:
        row[11] = 0
    return row


def _generate_path_shape_individual(rng, state):
    row = np.zeros(N_PARAMS, dtype=np.int32)
    path_field = _sample_template_value(rng, _PATH_SHAPE_TEMPLATE_FIELDS or ("path_efficiency",))
    path_idx = INDICATOR_INDEX[path_field]
    row[0] = path_idx
    row[1] = path_idx
    row[2] = WINDOW_CHOICES.index(240) if 240 in WINDOW_CHOICES else int(state["single_window_all"][-1])
    row[3] = SLICE_CHOICES.index(None)
    row[4] = INDICATOR_INDEX.get("open", 0)
    row[5] = MASK_RULES.index("none")
    row[6] = 0
    row[7] = MODE1_OPS.index("mean")
    row[8] = 0
    row[9] = B_SHIFT_CHOICES.index(0)
    cross_day_op = "ts_zscore" if rng.random() < 0.70 else ("delta" if rng.random() < 0.65 else "ts_decay")
    row[10] = TS_COMP_OPS.index(cross_day_op if cross_day_op in TS_COMP_OPS else "ts_decay")
    path_windows = tuple(state.get("path_shape_ts_window") or ())
    if not path_windows:
        path_windows = (TS_COMP_WINDOWS.index(20),) if 20 in TS_COMP_WINDOWS else tuple(state["ts_comp_window_all"])
    row[11] = _sample_from_pool(rng, path_windows)
    row[12] = CS_COMP_OPS.index("none")
    row[13] = 0
    return row


def _sample_template_value(rng, values):
    return values[int(rng.integers(0, len(values)))]


def _allowed_indicator_indices(state, names, key):
    allowed = set(int(x) for x in state.get(key, ()))
    out = tuple(
        INDICATOR_INDEX[name]
        for name in names
        if name in INDICATOR_INDEX and (not allowed or INDICATOR_INDEX[name] in allowed)
    )
    return out


def _allowed_value_indices(names, registry, allowed):
    allowed_set = set(int(x) for x in allowed)
    return tuple(
        registry.index(name)
        for name in names
        if name in registry and (not allowed_set or registry.index(name) in allowed_set)
    )


def _choose_index(rng, values, fallback):
    pool = tuple(values) or tuple(fallback)
    if not pool:
        return None
    return int(pool[int(rng.integers(0, len(pool)))])


def _allowed_mask_pairs(state, pairs):
    allowed = set(tuple(map(int, item)) for item in state.get("mask_pairs_all", ()))
    out = []
    for field, rule in pairs:
        if field not in INDICATOR_INDEX or rule not in MASK_RULES:
            continue
        pair = (INDICATOR_INDEX[field], MASK_RULES.index(rule))
        if not allowed or pair in allowed:
            out.append(pair)
    if not out and "open" in INDICATOR_INDEX:
        pair = (INDICATOR_INDEX["open"], MASK_RULES.index("none"))
        if not allowed or pair in allowed:
            out.append(pair)
    return tuple(out)


def _state_has_any_fields(state, names):
    flow_indices = {
        INDICATOR_INDEX[name]
        for name in names
        if name in INDICATOR_INDEX
    }
    if not flow_indices:
        return False
    active = set()
    for key in ("single_field_all", "pair_a_field_all", "pair_b_field_all"):
        active.update(int(x) for x in state.get(key, ()))
    active.update(int(field) for field, _ in state.get("mask_pairs_all", ()))
    return bool(active.intersection(flow_indices))


def _state_has_native_order_flow_fields(state):
    return _state_has_any_fields(state, ORDER_FLOW_INDICATOR_NAMES)


def _state_has_xbinance_order_flow_fields(state):
    return _state_has_any_fields(state, XBINANCE_ORDER_FLOW_INDICATOR_NAMES)


def _state_has_order_flow_fields(state):
    return (
        _state_has_native_order_flow_fields(state)
        or _state_has_xbinance_order_flow_fields(state)
    )


def _uses_fundamental(row):
    if not _FUNDAMENTAL_FIELD_INDICES:
        return False
    fund = set(_FUNDAMENTAL_FIELD_INDICES)
    mode = int(row[6]) % 4
    if int(row[0]) in fund:
        return True
    if mode in {1, 3} and int(row[1]) in fund:
        return True
    return int(row[5]) != MASK_RULES.index("none") and int(row[4]) in fund


def _enforce_fundamental_required(pop, rng=None, state=None):
    if not fundamental_required_enabled():
        return pop
    if pop is None:
        return pop
    arr = np.asarray(pop, dtype=np.int32).copy()
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if state is None:
        state = build_search_space_state()
    rng = rng or np.random.default_rng(0)

    cap_single = _allowed_indicator_indices(state, _FUNDAMENTAL_CAP_FIELDS, "single_field_all")
    cap_b = _allowed_indicator_indices(state, _FUNDAMENTAL_CAP_FIELDS, "pair_b_field_all")
    base_single = _allowed_indicator_indices(state, _FUNDAMENTAL_BASE_FIELDS, "single_field_all")
    base_a = _allowed_indicator_indices(state, _FUNDAMENTAL_BASE_FIELDS, "pair_a_field_all")
    mode1_ops = _allowed_value_indices(
        ("mean", "std", "zscore", "momentum", "range_position"),
        MODE1_OPS,
        state.get("mode1_op_all", ()),
    )
    mode2_ops = _allowed_value_indices(
        ("ratio_mean", "ratio_std", "slope", "cov", "diff_mean"),
        MODE2_OPS,
        state.get("mode2_op_all", ()),
    )
    mode3_ops = _allowed_value_indices(
        ("zscore_anchor", "std_normalized_diff", "std_ratio"),
        MODE3_OPS,
        state.get("mode3_op_all", ()),
    )
    cap_masks = tuple(
        (field, rule)
        for field, rule in state.get("mask_pairs_all", ())
        if int(field) in _FUNDAMENTAL_FIELD_INDICES and MASK_RULES[int(rule)] != "none"
    )
    open_none = (
        (INDICATOR_INDEX.get("open", 0), MASK_RULES.index("none"))
        if "open" in INDICATOR_INDEX else (0, MASK_RULES.index("none"))
    )
    windows_single = tuple(state.get("single_window_all", ())) or (0,)
    windows_pair = tuple(state.get("pair_window_all", ())) or windows_single
    ts_active = tuple(
        idx for idx in state.get("ts_comp_op_all", ())
        if TS_COMP_OPS[int(idx)] in {"none", "ts_decay", "ts_zscore", "delta"}
    ) or (TS_COMP_OPS.index("none"),)
    ts_windows = tuple(state.get("ts_comp_window_all", ())) or (0,)
    cs_ops = tuple(state.get("cs_comp_op_all", ())) or (CS_COMP_OPS.index("none"),)
    zero_lag = B_SHIFT_CHOICES.index(0)
    lag_pool = tuple(state.get("lag_all", (zero_lag,))) or (zero_lag,)
    none_slice = SLICE_CHOICES.index(None)

    def choose(values, fallback):
        values = tuple(values) or tuple(fallback)
        return int(values[int(rng.integers(0, len(values)))])

    for i, row in enumerate(arr):
        if _uses_fundamental(row):
            continue
        template = int((int(row.sum()) + i) % 4)
        row[:] = 0
        if template == 0:
            row[6] = 0
            row[0] = choose(cap_single, _FUNDAMENTAL_FIELD_INDICES)
            row[1] = row[0]
            row[7] = choose(mode1_ops, state.get("mode1_op_all", (0,)))
            row[2] = choose(windows_single, (0,))
            row[3] = none_slice if WINDOW_CHOICES[int(row[2])] >= MINUTES_PER_PERIOD else SLICE_CHOICES.index(0.3)
            row[4], row[5] = open_none
            row[9] = zero_lag
        elif template == 1:
            row[6] = 1
            row[0] = choose(base_a, state.get("pair_a_field_all", _FUNDAMENTAL_FIELD_INDICES))
            row[1] = choose(cap_b, _FUNDAMENTAL_FIELD_INDICES)
            row[8] = choose(mode2_ops, state.get("mode2_op_all", (0,)))
            row[2] = choose(windows_pair, (0,))
            row[3] = none_slice if WINDOW_CHOICES[int(row[2])] >= MINUTES_PER_PERIOD else SLICE_CHOICES.index(0.3)
            row[4], row[5] = open_none if rng.random() < 0.55 or not cap_masks else cap_masks[int(rng.integers(0, len(cap_masks)))]
            row[9] = choose(lag_pool, (zero_lag,))
        elif template == 2 and cap_masks:
            row[6] = int(rng.choice([0, 1]))
            if row[6] == 0:
                row[0] = choose(base_single, state.get("single_field_all", _FUNDAMENTAL_FIELD_INDICES))
                row[1] = row[0]
                row[7] = choose(mode1_ops, state.get("mode1_op_all", (0,)))
                row[2] = choose(windows_single, (0,))
                row[9] = zero_lag
            else:
                row[0] = choose(base_a, state.get("pair_a_field_all", _FUNDAMENTAL_FIELD_INDICES))
                row[1] = choose(state.get("pair_b_field_all", ()), _FUNDAMENTAL_FIELD_INDICES)
                row[8] = choose(mode2_ops, state.get("mode2_op_all", (0,)))
                row[2] = choose(windows_pair, (0,))
                row[9] = choose(lag_pool, (zero_lag,))
            row[3] = none_slice if WINDOW_CHOICES[int(row[2])] >= MINUTES_PER_PERIOD else SLICE_CHOICES.index(0.3)
            row[4], row[5] = cap_masks[int(rng.integers(0, len(cap_masks)))]
        else:
            row[6] = 2
            row[0] = choose(cap_single, _FUNDAMENTAL_FIELD_INDICES)
            row[1] = row[0]
            row[13] = choose(mode3_ops, state.get("mode3_op_all", (0,)))
            row[2] = choose(windows_single, (0,))
            row[3] = none_slice if WINDOW_CHOICES[int(row[2])] >= MINUTES_PER_PERIOD else SLICE_CHOICES.index(0.3)
            row[4], row[5] = open_none
            row[9] = zero_lag
            long_windows = [idx for idx in state.get("ts_comp_window_all", ()) if TS_COMP_WINDOWS[int(idx)] >= 2]
            row[11] = choose(long_windows, (TS_COMP_WINDOWS.index(5) if 5 in TS_COMP_WINDOWS else 0,))
        row[10] = choose(ts_active, (TS_COMP_OPS.index("none"),)) if row[6] != 2 else TS_COMP_OPS.index("none")
        if row[10] == TS_COMP_OPS.index("none") and row[6] != 2:
            row[11] = 0
        elif row[6] != 2:
            row[11] = choose(ts_windows, (0,))
        row[12] = choose(cs_ops, (CS_COMP_OPS.index("none"),))
    return arr


def _generate_order_flow_individual(rng, state):
    flow_single = _allowed_indicator_indices(
        state,
        _ORDER_FLOW_TEMPLATE_FIELDS,
        "single_field_all",
    )
    flow_a = _allowed_indicator_indices(
        state,
        _ORDER_FLOW_TEMPLATE_FIELDS,
        "pair_a_field_all",
    )
    context_b = _allowed_indicator_indices(
        state,
        _ORDER_FLOW_CONTEXT_FIELDS,
        "pair_b_field_all",
    )
    if not flow_single and not (flow_a and context_b):
        return _generate_policy_individual(rng, state, explore=True, mode_override=1)

    row = np.zeros(N_PARAMS, dtype=np.int32)
    single_possible = bool(flow_single)
    pair_possible = bool(flow_a and context_b)
    use_single = single_possible and (not pair_possible or rng.random() < 0.35)
    windows = _allowed_value_indices(
        (60, 90, 120, 180, 240),
        WINDOW_CHOICES,
        state.get("single_window_all" if use_single else "pair_window_all", ()),
    )
    slices = state.get("single_slice_all" if use_single else "pair_slice_all", ())
    masks = _allowed_mask_pairs(state, _ORDER_FLOW_MASK_PAIRS)
    ts_ops = _allowed_value_indices(
        ("ts_zscore", "delta", "ts_decay", "none"),
        TS_COMP_OPS,
        state.get("ts_comp_op_all", ()),
    )
    ts_windows = _allowed_value_indices(
        (5, 10, 20),
        TS_COMP_WINDOWS,
        state.get("ts_comp_window_all", ()),
    )
    cs_ops = _allowed_value_indices(
        ("cs_zscore", "none"),
        CS_COMP_OPS,
        state.get("cs_comp_op_all", ()),
    )
    none_slice = SLICE_CHOICES.index(None)
    zero_lag = B_SHIFT_CHOICES.index(0)
    none_ts = TS_COMP_OPS.index("none")
    none_cs = CS_COMP_OPS.index("none")

    if use_single:
        mode1_ops = _allowed_value_indices(
            ("mean", "last", "zscore", "std"),
            MODE1_OPS,
            state.get("mode1_op_all", ()),
        )
        row[6] = 0
        row[0] = _choose_index(rng, flow_single, state.get("single_field_all", (0,)))
        row[1] = row[0]
        row[7] = _choose_index(rng, mode1_ops, state.get("mode1_op_all", (0,)))
        row[8] = 0
        row[9] = zero_lag
    else:
        mode2_ops = _allowed_value_indices(
            ("slope", "cov", "corr", "diff_mean", "residual_std", "ratio_std"),
            MODE2_OPS,
            state.get("mode2_op_all", ()),
        )
        lags = _allowed_value_indices(
            (0, 1, 2, 3),
            B_SHIFT_CHOICES,
            state.get("lag_all", ()),
        )
        row[6] = 1
        row[0] = _choose_index(rng, flow_a, state.get("pair_a_field_all", (0,)))
        row[1] = _choose_index(rng, context_b, state.get("pair_b_field_all", (0,)))
        row[7] = 0
        row[8] = _choose_index(rng, mode2_ops, state.get("mode2_op_all", (0,)))
        row[9] = _choose_index(rng, lags, (zero_lag,))

    row[2] = _choose_index(
        rng,
        windows,
        state.get("single_window_all" if use_single else "pair_window_all", (0,)),
    )
    row[3] = none_slice if WINDOW_CHOICES[int(row[2])] >= MINUTES_PER_PERIOD else _choose_index(rng, slices, (none_slice,))
    row[4], row[5] = masks[int(rng.integers(0, len(masks)))] if masks else (
        INDICATOR_INDEX.get("open", 0),
        MASK_RULES.index("none"),
    )
    if rng.random() < 0.70:
        row[10] = _choose_index(rng, ts_ops, (none_ts,))
        if row[10] == none_ts:
            row[11] = 0
        else:
            row[11] = _choose_index(rng, ts_windows, state.get("ts_comp_window_all", (0,)))
    else:
        row[10] = none_ts
        row[11] = 0
    row[12] = _choose_index(rng, cs_ops, (none_cs,))
    row[13] = 0
    return row


def _state_has_field_scout_fields(state):
    return _state_has_any_fields(state, _FIELD_SCOUT_TEMPLATE_FIELDS)


def _generate_field_scout_individual(rng, state):
    scout_single = _allowed_indicator_indices(
        state,
        _FIELD_SCOUT_TEMPLATE_FIELDS,
        "single_field_all",
    )
    scout_a = _allowed_indicator_indices(
        state,
        _FIELD_SCOUT_TEMPLATE_FIELDS,
        "pair_a_field_all",
    )
    context_b = _allowed_indicator_indices(
        state,
        _FIELD_SCOUT_CONTEXT_FIELDS,
        "pair_b_field_all",
    )
    if not scout_single and not (scout_a and context_b):
        return _generate_policy_individual(rng, state, explore=True, mode_override=1)

    row = np.zeros(N_PARAMS, dtype=np.int32)
    single_possible = bool(scout_single)
    pair_possible = bool(scout_a and context_b)
    use_single = single_possible and (not pair_possible or rng.random() < 0.30)
    windows = _allowed_value_indices(
        (60, 90, 120, 180, 240),
        WINDOW_CHOICES,
        state.get("single_window_all" if use_single else "pair_window_all", ()),
    )
    slices = state.get("single_slice_all" if use_single else "pair_slice_all", ())
    masks = _allowed_mask_pairs(state, _FIELD_SCOUT_MASK_PAIRS)
    ts_ops = _allowed_value_indices(
        ("ts_zscore", "delta", "ts_decay", "none"),
        TS_COMP_OPS,
        state.get("ts_comp_op_all", ()),
    )
    ts_windows = _allowed_value_indices(
        (5, 10, 20),
        TS_COMP_WINDOWS,
        state.get("ts_comp_window_all", ()),
    )
    cs_ops = _allowed_value_indices(
        ("cs_zscore", "none"),
        CS_COMP_OPS,
        state.get("cs_comp_op_all", ()),
    )
    none_slice = SLICE_CHOICES.index(None)
    zero_lag = B_SHIFT_CHOICES.index(0)
    none_ts = TS_COMP_OPS.index("none")
    none_cs = CS_COMP_OPS.index("none")

    if use_single:
        mode1_ops = _allowed_value_indices(
            ("zscore", "mean", "std", "momentum", "range_position"),
            MODE1_OPS,
            state.get("mode1_op_all", ()),
        )
        row[6] = 0
        row[0] = _choose_index(rng, scout_single, state.get("single_field_all", (0,)))
        row[1] = row[0]
        row[7] = _choose_index(rng, mode1_ops, state.get("mode1_op_all", (0,)))
        row[8] = 0
        row[9] = zero_lag
    else:
        mode2_ops = _allowed_value_indices(
            ("ratio_mean", "ratio_std", "slope", "cov", "corr", "diff_mean"),
            MODE2_OPS,
            state.get("mode2_op_all", ()),
        )
        lags = _allowed_value_indices(
            (0, 1, 2, 3),
            B_SHIFT_CHOICES,
            state.get("lag_all", ()),
        )
        row[6] = 1
        row[0] = _choose_index(rng, scout_a, state.get("pair_a_field_all", (0,)))
        row[1] = _choose_index(rng, context_b, state.get("pair_b_field_all", (0,)))
        row[7] = 0
        row[8] = _choose_index(rng, mode2_ops, state.get("mode2_op_all", (0,)))
        row[9] = _choose_index(rng, lags, (zero_lag,))

    row[2] = _choose_index(
        rng,
        windows,
        state.get("single_window_all" if use_single else "pair_window_all", (0,)),
    )
    row[3] = none_slice if WINDOW_CHOICES[int(row[2])] >= MINUTES_PER_PERIOD else _choose_index(rng, slices, (none_slice,))
    row[4], row[5] = masks[int(rng.integers(0, len(masks)))] if masks else (
        INDICATOR_INDEX.get("open", 0),
        MASK_RULES.index("none"),
    )
    row[10] = _choose_index(rng, ts_ops, (none_ts,))
    if row[10] == none_ts:
        row[11] = 0
    else:
        row[11] = _choose_index(rng, ts_windows, state.get("ts_comp_window_all", (0,)))
    row[12] = _choose_index(rng, cs_ops, (none_cs,))
    row[13] = 0
    return row


_SUBMIT_PRIOR_TEMPLATES = (
    # submit/20260514: log-volume / volume stability.
    {
        "mode": 0,
        "fields": ("log_volume", "volume_intensity", "volume_zscore_60d", "turnover"),
        "ops": ("std", "skew", "zscore", "mean"),
        "windows": (90, 120, 180, 240),
        "slices": (None, 0.3, 0.5),
        "masks": (("open", "none"), ("turnover", "high_0.8"), ("dollar_volume_rank", "high_0.8")),
        "ts_ops": ("ts_decay", "ts_zscore", "none"),
        "cs_ops": ("cs_zscore", "cs_rank", "none"),
    },
    # submit/20260521: dollar-volume convergence / liquidity stability.
    {
        "mode": 1,
        "a_fields": ("turnover", "money_flow", "volume_intensity", "dollar_volume_rank"),
        "b_fields": ("cm_turnover_to_mcap_lag1", "volume_zscore_60d", "rv_20d", "return_abs"),
        "ops": ("ratio_mean", "ratio_std", "slope", "cov", "diff_mean"),
        "windows": (90, 120, 180, 240),
        "slices": (None, 0.3, 0.5),
        "lags": (0, 1, 2, 3),
        "masks": (("open", "none"), ("dollar_volume_rank", "high_0.8"), ("turnover", "high_0.8")),
        "ts_ops": ("ts_decay", "ts_zscore", "delta", "none"),
        "cs_ops": ("cs_zscore", "none"),
    },
    # submit/20260528: realized skew / tail risk with liquidity confirmation.
    {
        "mode": 1,
        "a_fields": ("returns", "return_abs", "price_range_pct", "rv_5d", "rv_20d", "vol_of_vol_20d"),
        "b_fields": ("turnover", "money_flow", "volume_zscore_60d", "dollar_volume_rank"),
        "ops": ("cov", "corr", "ratio_std", "diff_mean", "residual_std"),
        "windows": (90, 120, 180, 240),
        "slices": (None, 0.3, 0.5),
        "lags": (0, 1, 2, 3),
        "masks": (("open", "none"), ("rv_5d", "high_0.8"), ("market_vol_intensity", "high_0.8")),
        "ts_ops": ("ts_decay", "ts_zscore", "none"),
        "cs_ops": ("cs_zscore", "cs_rank", "none"),
    },
    # Trajectory / illiquidity: keep path fields paired with liquidity context, not naked.
    {
        "mode": 1,
        "a_fields": ("amihud_illiq", "price_range_pct", "late_range_share", "volume_burst_share", "intraday_reversal"),
        "b_fields": ("turnover", "money_flow", "volume_zscore_60d", "return_abs", "rv_20d"),
        "ops": ("ratio_mean", "ratio_std", "slope", "cov", "diff_mean"),
        "windows": (90, 120, 180, 240),
        "slices": (None, 0.3, 0.5),
        "lags": (0, 1, 2, 3),
        "masks": (("open", "none"), ("turnover", "high_0.8"), ("money_flow", "high_0.6")),
        "ts_ops": ("ts_decay", "ts_zscore", "none"),
        "cs_ops": ("cs_zscore", "none"),
    },
    # BTC-relative / regime confirmation templates from the existing indicator set.
    {
        "mode": 1,
        "a_fields": ("relative_strength_btc_5d", "resid_return_btc_20d", "idio_vol_btc_20d", "corr_btc_20d"),
        "b_fields": ("funding_z20", "oi_z20", "funding_oi_joint_5d", "market_vol_intensity"),
        "ops": ("diff_mean", "ratio_std", "cov", "corr"),
        "windows": (90, 120, 180, 240),
        "slices": (None, 0.3),
        "lags": (0, 1, 2),
        "masks": (("open", "none"), ("funding", "high_0.8"), ("open_interest", "high_0.8")),
        "ts_ops": ("ts_decay", "ts_zscore", "none"),
        "cs_ops": ("cs_zscore", "none"),
    },
)


def _choose_allowed_registry_value(rng, values, registry, allowed, fallback):
    choices = _allowed_value_indices(values, registry, allowed)
    if not choices:
        choices = tuple(fallback) or tuple(allowed) or (0,)
    return _choose_index(rng, choices, choices)


def _generate_submit_prior_individual(rng, state):
    templates = list(_SUBMIT_PRIOR_TEMPLATES)
    rng.shuffle(templates)
    none_slice = SLICE_CHOICES.index(None)
    none_ts = TS_COMP_OPS.index("none")
    none_cs = CS_COMP_OPS.index("none")
    zero_lag = B_SHIFT_CHOICES.index(0)

    for template in templates:
        mode = int(template["mode"])
        row = np.zeros(N_PARAMS, dtype=np.int32)
        if mode == 0:
            fields = _allowed_indicator_indices(state, template["fields"], "single_field_all")
            ops = _allowed_value_indices(template["ops"], MODE1_OPS, state.get("mode1_op_all", ()))
            if not fields or not ops:
                continue
            row[6] = 0
            row[0] = _choose_index(rng, fields, state.get("single_field_all", (0,)))
            row[1] = row[0]
            row[7] = _choose_index(rng, ops, state.get("mode1_op_all", (0,)))
            window_allowed = state.get("single_window_all", ())
            slice_allowed = state.get("single_slice_all", ())
            row[9] = zero_lag
        else:
            a_fields = _allowed_indicator_indices(state, template["a_fields"], "pair_a_field_all")
            b_fields = _allowed_indicator_indices(state, template["b_fields"], "pair_b_field_all")
            ops = _allowed_value_indices(template["ops"], MODE2_OPS, state.get("mode2_op_all", ()))
            if not a_fields or not b_fields or not ops:
                continue
            row[6] = 1
            row[0] = _choose_index(rng, a_fields, state.get("pair_a_field_all", (0,)))
            row[1] = _choose_index(rng, b_fields, state.get("pair_b_field_all", (0,)))
            row[8] = _choose_index(rng, ops, state.get("mode2_op_all", (0,)))
            row[9] = _choose_allowed_registry_value(
                rng, template.get("lags", (0,)), B_SHIFT_CHOICES, state.get("lag_all", ()), (zero_lag,)
            )
            window_allowed = state.get("pair_window_all", ())
            slice_allowed = state.get("pair_slice_all", ())

        row[2] = _choose_allowed_registry_value(
            rng, template["windows"], WINDOW_CHOICES, window_allowed, window_allowed
        )
        if WINDOW_CHOICES[int(row[2])] >= MINUTES_PER_PERIOD:
            row[3] = none_slice
        else:
            row[3] = _choose_allowed_registry_value(
                rng, template["slices"], SLICE_CHOICES, slice_allowed, (none_slice,)
            )
        masks = _allowed_mask_pairs(state, template["masks"])
        row[4], row[5] = masks[int(rng.integers(0, len(masks)))] if masks else (
            INDICATOR_INDEX.get("open", 0), MASK_RULES.index("none")
        )
        row[10] = _choose_allowed_registry_value(
            rng, template["ts_ops"], TS_COMP_OPS, state.get("ts_comp_op_all", ()), (none_ts,)
        )
        if row[10] == none_ts:
            row[11] = 0
        else:
            row[11] = _choose_allowed_registry_value(
                rng, (5, 10, 20), TS_COMP_WINDOWS, state.get("ts_comp_window_all", ()), (0,)
            )
        row[12] = _choose_allowed_registry_value(
            rng, template["cs_ops"], CS_COMP_OPS, state.get("cs_comp_op_all", ()), (none_cs,)
        )
        row[13] = 0
        return row

    return _generate_field_scout_individual(rng, state)


def _generate_proven_relation_individual(rng, state):
    template = _sample_template_value(rng, _PROVEN_RELATION_TEMPLATES)
    row = np.zeros(N_PARAMS, dtype=np.int32)
    row[0] = INDICATOR_INDEX[_sample_template_value(rng, template["a"])]
    row[1] = INDICATOR_INDEX[_sample_template_value(rng, template["b"])]
    row[2] = WINDOW_CHOICES.index(_sample_template_value(rng, template["windows"]))
    row[3] = SLICE_CHOICES.index(_sample_template_value(rng, template["slices"]))
    mask_field, mask_rule = _sample_template_value(rng, template["masks"])
    row[4] = INDICATOR_INDEX[mask_field]
    row[5] = MASK_RULES.index(mask_rule)
    row[6] = 1
    row[7] = 0
    row[8] = MODE2_OPS.index(_sample_template_value(rng, template["ops"]))
    row[9] = B_SHIFT_CHOICES.index(_sample_template_value(rng, template["lags"]))
    use_zscore = rng.random() < float(np.clip(state.get("proven_relation_ts_zscore_share", 0.0), 0.0, 1.0))
    row[10] = TS_COMP_OPS.index("ts_zscore" if use_zscore else "ts_decay")
    row[11] = TS_COMP_WINDOWS.index(20 if rng.random() < 0.65 else 10)
    use_cs = rng.random() < float(np.clip(state.get("proven_relation_cs_zscore_share", 0.0), 0.0, 1.0))
    row[12] = CS_COMP_OPS.index("cs_zscore" if use_cs else "none")
    row[13] = 0
    return row


def _generate_exploration_pop(
    n,
    rng=None,
    *,
    feedback_artifact=None,
):
    """Generate n individuals from a constrained high-quality search space.

    The main budget stays inside a high-quality production pool; a smaller
    explore budget samples adjacent but still semantically meaningful regions.
    """
    if rng is None:
        rng = np.random.default_rng()
    state = build_search_space_state()
    state["sampling_feedback"] = _build_sampling_feedback(state, feedback_artifact)
    X = np.zeros((n, N_PARAMS), dtype=np.int32)
    adjacent_share = float(np.clip(state.get("adjacent_explore_share", 0.0), 0.0, 1.0))
    structural_share = float(np.clip(state.get("structural_explore_share", 0.0), 0.0, 1.0))
    path_shape_share = float(np.clip(state.get("path_shape_share", 0.0), 0.0, 1.0))
    proven_relation_share = float(np.clip(state.get("proven_relation_share", 0.0), 0.0, 1.0))
    order_flow_share = float(np.clip(state.get("order_flow_share", 0.0), 0.0, 1.0))
    submit_prior_share = SUBMIT_PRIOR_SHARE
    if not _state_has_order_flow_fields(state):
        order_flow_share = 0.0
    n_submit_prior = min(n, int(round(n * submit_prior_share)))
    n_path_shape = min(n - n_submit_prior, int(round(n * path_shape_share)))
    n_proven_relation = min(n - n_submit_prior - n_path_shape, int(round(n * proven_relation_share)))
    n_order_flow = min(n - n_submit_prior - n_path_shape - n_proven_relation, int(round(n * order_flow_share)))
    n_structural = min(n - n_submit_prior - n_path_shape - n_proven_relation - n_order_flow, int(round(n * structural_share)))
    n_adjacent = min(n - n_submit_prior - n_path_shape - n_proven_relation - n_order_flow - n_structural, int(round(n * adjacent_share)))
    n_primary = n - n_submit_prior - n_path_shape - n_proven_relation - n_order_flow - n_adjacent - n_structural
    lanes = (
        ["primary"] * n_primary
        + ["adjacent"] * n_adjacent
        + ["structural"] * n_structural
        + ["order_flow"] * n_order_flow
        + ["proven_relation"] * n_proven_relation
        + ["submit_prior"] * n_submit_prior
        + ["path_shape"] * n_path_shape
    )
    rng.shuffle(lanes)
    mode3_share = float(np.clip(state.get("structural_mode3_share", 0.5), 0.0, 1.0))
    for i, lane in enumerate(lanes):
        if lane == "primary":
            X[i] = _generate_policy_individual(rng, state, explore=False)
        elif lane == "adjacent":
            X[i] = _generate_policy_individual(rng, state, explore=True, mode_override=1)
        elif lane == "structural":
            mode_override = 3 if rng.random() < 0.15 else (2 if rng.random() < mode3_share else 0)
            X[i] = _generate_policy_individual(rng, state, explore=True, mode_override=mode_override)
        elif lane == "order_flow":
            X[i] = _generate_order_flow_individual(rng, state)
        elif lane == "proven_relation":
            X[i] = _generate_proven_relation_individual(rng, state)
        elif lane == "submit_prior":
            X[i] = _generate_submit_prior_individual(rng, state)
        else:
            X[i] = _generate_path_shape_individual(rng, state)
    X = project_population_to_search_space(X)
    return _enforce_fundamental_required(X, rng=rng, state=state)


class ExplorationSampling(Sampling):
    """pymoo Sampling that uses _generate_exploration_pop."""

    def __init__(self, *, feedback_artifact=None):
        super().__init__()
        self.feedback_artifact = feedback_artifact

    def _do(self, problem, n_samples, **kwargs):
        return _generate_exploration_pop(
            n_samples,
            feedback_artifact=self.feedback_artifact,
        )


_DISCRETE_LOCI = (0, 1, 4, 5, 6, 7, 8, 10, 12, 13)
_ORDERED_LOCI = (2, 3, 9, 11, 14)


class HybridCrossover(Crossover):
    def __init__(self, prob=0.9, sbx_eta=3.0, rng=None):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self._sbx = SBX(prob=1.0, eta=sbx_eta, vtype=float)
        self._rng = rng

    def _resolve_rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(int(np.random.randint(0, 2 ** 32)))
        return self._rng

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X, dtype=float)
        offspring = X.copy()
        n_matings = X.shape[1]
        state = build_search_space_state()
        rng = self._resolve_rng()

        if _DISCRETE_LOCI:
            swap = rng.random((n_matings, len(_DISCRETE_LOCI))) < 0.5
            for idx, col in enumerate(_DISCRETE_LOCI):
                mask = swap[:, idx]
                if np.any(mask):
                    offspring[0, mask, col] = X[1, mask, col]
                    offspring[1, mask, col] = X[0, mask, col]

        if _ORDERED_LOCI:
            ordered_idx = np.asarray(_ORDERED_LOCI, dtype=np.int32)
            xl = np.asarray(problem.xl, dtype=float)[ordered_idx]
            xu = np.asarray(problem.xu, dtype=float)[ordered_idx]
            ordered_problem = Problem(n_var=len(_ORDERED_LOCI), n_obj=1, xl=xl, xu=xu)
            ordered = self._sbx._do(problem=ordered_problem, X=X[:, :, ordered_idx], **kwargs)
            offspring[:, :, ordered_idx] = ordered

        projected = project_population_to_search_space(offspring.reshape(-1, offspring.shape[-1]))
        projected = _force_mode4_only_population(projected)
        projected = _enforce_fundamental_required(projected, rng=rng, state=state)
        return np.asarray(projected, dtype=np.int32).reshape(offspring.shape)


class DiversityMutation(Mutation):
    """Custom mutation that injects genuine discrete diversity into the op /
    mode fields — pymoo's PM operator treats ints as continuous and after
    round(·) rarely flips op/mode (residual_std basin self-reinforcing).

    Pipeline per offspring row:
      1. Base PM mutation on all genes (inherits the tuned eta/prob).
      2. Op columns (7=mode1_op, 8=mode2_op, 13=mode3_op/mode4_op) are
         independently re-sampled uniformly within bounds with probability
         OP_FLIP_PROB — a real discrete jump, not a continuous perturbation.
      3. Mode column (6) with probability MODE_FLIP_PROB jumps uniformly to
         {0,1,2,3} (MODE1/MODE2/MODE3/MODE4) — forces crossings between modes
         that crossover alone can't reach.
      4. Diversity drifts cover window, field_a family, and high/low mask rules
         before search-space projection repairs any incompatible combinations.
    Post-mutation the SearchSpaceRepair re-projects, so invalid combos
    (mode vs op-column coherence) are corrected.
    """

    OP_FLIP_PROB = 0.30
    MODE_FLIP_PROB = 0.15
    WINDOW_DRIFT_PROB = 0.12
    FIELD_SWAP_PROB = 0.08
    MASK_FLIP_PROB = 0.05

    def __init__(self, eta=3.0, prob=0.15, vtype=float):
        super().__init__()
        self._pm = PM(prob=prob, eta=eta, vtype=vtype)
        self._rng = None

    def _resolve_rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(int(np.random.randint(0, 2 ** 32)))
        return self._rng
    @staticmethod
    def _window_pool_for_mode(state, mode):
        if int(mode) in (0, 2):
            return tuple(int(v) for v in state.get("single_window_all", ()))
        return tuple(int(v) for v in state.get("pair_window_all", ()))
    @staticmethod
    def _field_pool_for_mode(state, mode):
        if int(mode) in (0, 2):
            return tuple(int(v) for v in state.get("single_field_all", ()))
        if int(mode) == 1:
            return tuple(int(v) for v in state.get("pair_a_field_all", ()))
        return tuple(int(v) for v in state.get("mode4_a_field_all", ()))
    @staticmethod
    def _adjacent_window_indices(current_idx, active_pool):
        pool = tuple(sorted(set(int(v) for v in active_pool)))
        if not pool:
            return ()
        try:
            pos = pool.index(int(current_idx))
        except ValueError:
            return pool
        out = []
        if pos > 0:
            out.append(pool[pos - 1])
        if pos + 1 < len(pool):
            out.append(pool[pos + 1])
        return tuple(out)
    @staticmethod
    def _paired_mask_rule_index(rule_idx):
        rule_name = MASK_RULES[int(rule_idx)]
        if rule_name.startswith("high_"):
            paired = "low_" + rule_name.split("_", 1)[1]
        elif rule_name.startswith("low_"):
            paired = "high_" + rule_name.split("_", 1)[1]
        else:
            return None
        return MASK_RULES.index(paired) if paired in MASK_RULES else None

    def _do(self, problem, X, **kwargs):
        X_pm = self._pm._do(problem, X, **kwargs)
        X_out = np.asarray(X_pm, dtype=float).copy()
        n = X_out.shape[0]
        state = build_search_space_state()
        rng = self._resolve_rng()
        allowed_ops = {
            7: tuple(state["mode1_op_all"]),
            8: tuple(state["mode2_op_all"]),
            13: tuple(state["mode3_op_all"]),
        }
        # Op columns 7, 8, 13 → profile-aware re-sample on flip.
        # Bounds are registry-compatible; the profile is the active interface.
        for col, choices in allowed_ops.items():
            flip = rng.random(n) < self.OP_FLIP_PROB
            if flip.any() and choices:
                X_out[flip, col] = rng.choice(choices, size=int(flip.sum()))
        # Mode column 6 → profile-aware re-sample on flip
        flip_m = rng.random(n) < self.MODE_FLIP_PROB
        if flip_m.any():
            mode_probs = _mode_base_probs(state)
            if mode_probs is not None:
                allowed_modes = [i for i, p in enumerate(mode_probs) if p > 0]
                if len(allowed_modes) == 1:
                    X_out[flip_m, 6] = allowed_modes[0]
                elif allowed_modes:
                    p = np.asarray([mode_probs[i] for i in allowed_modes], dtype=np.float64)
                    p = p / p.sum()
                    X_out[flip_m, 6] = rng.choice(allowed_modes, size=int(flip_m.sum()), p=p)
                else:
                    X_out[flip_m, 6] = rng.integers(0, 4, size=int(flip_m.sum()))
            else:
                X_out[flip_m, 6] = rng.integers(0, 4, size=int(flip_m.sum()))

        for i in range(n):
            mode = int(np.rint(X_out[i, 6]))

            if rng.random() < self.WINDOW_DRIFT_PROB:
                window_pool = self._window_pool_for_mode(state, mode)
                if window_pool:
                    current_window = int(np.rint(X_out[i, 2]))
                    candidates = self._adjacent_window_indices(current_window, window_pool)
                    if candidates:
                        X_out[i, 2] = candidates[int(rng.integers(0, len(candidates)))]

            if rng.random() < self.FIELD_SWAP_PROB:
                current_field = int(np.rint(X_out[i, 0]))
                field_pool = self._field_pool_for_mode(state, mode)
                if 0 <= current_field < len(INDICATOR_NAMES) and field_pool:
                    family = _family_of_field(INDICATOR_NAMES[current_field])
                    same_family = tuple(
                        idx for idx in field_pool
                        if int(idx) != current_field
                        and 0 <= int(idx) < len(INDICATOR_NAMES)
                        and _family_of_field(INDICATOR_NAMES[int(idx)]) == family
                    )
                    if same_family:
                        X_out[i, 0] = same_family[int(rng.integers(0, len(same_family)))]

            if rng.random() < self.MASK_FLIP_PROB:
                flipped_rule = self._paired_mask_rule_index(int(np.rint(X_out[i, 5])))
                if flipped_rule is not None:
                    X_out[i, 5] = flipped_rule

        projected = project_population_to_search_space(X_out)
        projected = _force_mode4_only_population(projected)
        return _enforce_fundamental_required(projected, rng=rng, state=state)


class SearchSpaceRepair(Repair):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        rounded = np.rint(np.asarray(X)).astype(np.int32)
        projected = project_population_to_search_space(rounded)
        projected = _force_mode4_only_population(projected)
        return _enforce_fundamental_required(projected)


def _force_mode4_only_population(pop):
    projected = np.asarray(pop, dtype=np.int32).copy()
    if FORCE_MODE4_ONLY and projected.size:
        projected[:, 6] = 3
        state = build_search_space_state()
        mode4_pool = tuple(state.get("mode4_op_all", tuple(range(len(MODE4_OPS)))))
        if mode4_pool:
            projected[:, 13] = [mode4_pool[int(v) % len(mode4_pool)] for v in projected[:, 13]]
        a_pool = tuple(state.get("mode4_a_field_all", ()))
        b_pool = tuple(state.get("pair_b_field_all", ()))
        if a_pool:
            projected[:, 0] = [a_pool[int(v) % len(a_pool)] if int(v) not in a_pool else int(v) for v in projected[:, 0]]
        if b_pool:
            projected[:, 1] = [b_pool[int(v) % len(b_pool)] if int(v) not in b_pool else int(v) for v in projected[:, 1]]
    return projected


def _is_allowed_by_force_mode4(params):
    return (not FORCE_MODE4_ONLY) or int(np.asarray(params, dtype=np.int32)[6]) == 3


def _normalize_population(pop):
    projected = project_population_to_search_space(pop)
    return _enforce_fundamental_required(projected)


def _project_population_to_cached_fields(pop, field_indices):
    if field_indices is None:
        return pop
    fields = np.asarray(field_indices, dtype=np.int32)
    if fields.ndim != 1 or fields.size == 0:
        return pop
    fields = np.unique(fields)
    projected = np.asarray(pop, dtype=np.int32).copy()
    none_mask_idx = MASK_RULES.index("none")
    open_idx = INDICATOR_INDEX.get("open", int(fields[0]))
    fallback_mask_field = open_idx if np.any(fields == open_idx) else int(fields[0])
    for col in (0, 1):
        values = projected[:, col]
        pos = np.searchsorted(fields, values)
        ok = (pos < fields.size)
        ok[ok] = fields[pos[ok]] == values[ok]
        if np.any(~ok):
            projected[~ok, col] = fields[np.mod(values[~ok], fields.size)]
    values = projected[:, 4]
    pos = np.searchsorted(fields, values)
    ok = (pos < fields.size)
    ok[ok] = fields[pos[ok]] == values[ok]
    if np.any(~ok):
        projected[~ok, 4] = fallback_mask_field
        projected[~ok, 5] = none_mask_idx
    return projected


# ---------------------------------------------------------------------------
# Indicator niche cap — prevent population collapse onto a few indicators
#
# Two layers work together:
#   1. _apply_indicator_cap_penalty() — runs inside _evaluate(), affects
#      offspring fitness so NSGA-III's non-dominated sorting sees the cap.
#   2. IndicatorNicheCallback — runs after each generation, modifies the
#      surviving population's F so overrepresented *parents* lose fitness
#      and get replaced next round.  Safety net for accumulated parents.
# ---------------------------------------------------------------------------
_RESULT_OBJECTIVE_NAMES = ['rankicir', 'ls_net_sharpe', 'neg_turnover', 'novelty', 'ls_1-maxdd']
_BASE_SEARCH_OBJECTIVE_PROFILE = 'annual_worst_v1'
_BASE_SEARCH_OBJECTIVE_NAMES = [
    'rankicir',
    'ls_net_sharpe',
    'min_yearly_ls_netret',
    'min_yearly_ls_net_sharpe',
    'neg_turnover',
]
_BASE_SEARCH_SCORE_WEIGHTS = np.array([1.0, 2.0, 4.0, 4.0, 1.0], dtype=np.float32)
_SUBMIT_SHARPE_SEARCH_OBJECTIVE_PROFILE = 'submit_sharpe_v1'
_SUBMIT_SHARPE_SEARCH_OBJECTIVE_NAMES = [
    'ls_net_sharpe',
    'min_yearly_ls_net_sharpe',
    'min_yearly_ls_netret',
    'positive_rate_quality',
    'neg_turnover',
]
_SUBMIT_SHARPE_SEARCH_SCORE_WEIGHTS = np.array([3.0, 4.0, 4.0, 1.5, 1.0], dtype=np.float32)
_BARRA_SEARCH_OBJECTIVE_NAMES = [
    'barra_resid_ls_net_sharpe',
    'min_yearly_barra_resid_ls_net_sharpe',
    'barra_resid_retention',
    'neg_barra_style_r2',
]
_BARRA_SEARCH_SCORE_WEIGHTS = np.array([3.0, 4.0, 1.0, 1.0], dtype=np.float32)


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _json_env_dict(name, default):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return dict(default)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return dict(value)


# ---------------------------------------------------------------------------
# Search-time novelty objective (idea: make low-correlation a real NSGA-III
# objective instead of only a post-hoc soft multiplier). When enabled, the
# search objective vector gains a 6th column = production RankIC novelty
# (1 - max absolute Pearson correlation on IS), with existing pool/behavior
# novelty as a fallback when the production RankIC library is unavailable.
# NOTE: must be defined before _search_objective_config() runs at import.
# ---------------------------------------------------------------------------
SEARCH_NOVELTY_OBJECTIVE_ENABLED = os.environ.get(
    "GP_SEARCH_NOVELTY_OBJECTIVE", "1"
).lower() in {"1", "true", "yes", "on"}
SEARCH_NOVELTY_WEIGHT = float(os.environ.get("GP_SEARCH_NOVELTY_WEIGHT", "1.5"))
RANKIC_NOVELTY_OBJECTIVE_ENABLED = _env_flag("GP_RANKIC_NOVELTY_OBJECTIVE", True)
RANKIC_NOVELTY_LIBRARY_PATH = os.environ.get(
    "GP_RANKIC_LIBRARY_PATH",
    "/root/crypto-research/common/factor_base/rankics_cache_dir/rankics_perp_1d_abe1143e48a8.parquet",
).strip()
RANKIC_NOVELTY_MIN_VALID = max(1, int(os.environ.get("GP_RANKIC_NOVELTY_MIN_VALID", "60")))
RANKIC_NOVELTY_LIBRARY_CHUNK = max(1, int(os.environ.get("GP_RANKIC_NOVELTY_LIBRARY_CHUNK", "128")))
SEARCH_N_OBJECTIVES = N_OBJECTIVES + (1 if SEARCH_NOVELTY_OBJECTIVE_ENABLED else 0)

FINAL_SIGNATURE_DEDUP_ENABLED = _env_flag("GP_FINAL_SIGNATURE_DEDUP", "1")
FORMULA_BLACKLIST_ENABLED = _env_flag("GP_FORMULA_BLACKLIST_ENABLED", "1")
FORMULA_BLACKLIST_PATHS = os.environ.get("GP_FORMULA_BLACKLIST_PATHS", "")


def _search_objective_config(use_barra_objectives=None, profile=None):
    """Return candidate search objective metadata.

    The active NSGA-III objective vector is fixed by config.N_OBJECTIVES.
    Profiles define how the raw five fitness objectives are recomposed for
    search and final candidate ranking.
    """
    requested = (
        os.environ.get("GP_SEARCH_OBJECTIVE_PROFILE", _BASE_SEARCH_OBJECTIVE_PROFILE)
        if profile is None
        else str(profile)
    ).strip()
    if requested == _SUBMIT_SHARPE_SEARCH_OBJECTIVE_PROFILE:
        weights = np.array(_SUBMIT_SHARPE_SEARCH_SCORE_WEIGHTS, dtype=np.float32, copy=True)
        env_weights = os.environ.get("GP_SCORE_WEIGHTS", "").strip()
        if env_weights:
            parsed = [float(x) for x in env_weights.split(",")]
            if len(parsed) == len(weights):
                weights[:] = parsed
                print(f"[GP_SCORE_WEIGHTS] override: {parsed}", flush=True)
            else:
                print(f"[GP_SCORE_WEIGHTS] expected {len(weights)} values, got {len(parsed)}; ignoring", flush=True)
        names = list(_SUBMIT_SHARPE_SEARCH_OBJECTIVE_NAMES)
        if SEARCH_NOVELTY_OBJECTIVE_ENABLED:
            names.append('novelty')
            weights = np.concatenate(
                [weights, np.array([SEARCH_NOVELTY_WEIGHT], dtype=np.float32)]
            ).astype(np.float32, copy=False)
        return (
            _SUBMIT_SHARPE_SEARCH_OBJECTIVE_PROFILE,
            names,
            weights,
        )
    if requested and requested != _BASE_SEARCH_OBJECTIVE_PROFILE:
        raise ValueError(f"unknown GP_SEARCH_OBJECTIVE_PROFILE: {requested}")

    use_barra = (
        _env_flag("GP_SEARCH_USE_BARRA_OBJECTIVES")
        if use_barra_objectives is None
        else bool(use_barra_objectives)
    )
    names = list(_BASE_SEARCH_OBJECTIVE_NAMES)
    weights = np.array(_BASE_SEARCH_SCORE_WEIGHTS, dtype=np.float32, copy=True)
    active_profile = _BASE_SEARCH_OBJECTIVE_PROFILE
    if use_barra:
        names.extend(_BARRA_SEARCH_OBJECTIVE_NAMES)
        weights = np.concatenate([weights, _BARRA_SEARCH_SCORE_WEIGHTS]).astype(
            np.float32,
            copy=False,
        )
        active_profile = 'annual_worst_barra_candidate_v1'
    if SEARCH_NOVELTY_OBJECTIVE_ENABLED:
        names.append('novelty')
        weights = np.concatenate(
            [weights, np.array([SEARCH_NOVELTY_WEIGHT], dtype=np.float32)]
        ).astype(np.float32, copy=False)
    return active_profile, names, weights


SEARCH_OBJECTIVE_PROFILE, SEARCH_OBJECTIVE_NAMES, SEARCH_SCORE_WEIGHTS = _search_objective_config()


def _barra_residual_sentinel_metrics(n_obs):
    return {
        "barra_resid_ls_netret": -1e6,
        "barra_resid_ls_net_sharpe": -1e6,
        "barra_resid_retention": -1e6,
        "barra_style_r2": 1.0,
        "n_obs": int(n_obs),
    }


def _compute_barra_residual_metrics(
        factor_returns, style_returns, train_mask, periods_per_year=365):
    y = np.asarray(factor_returns, dtype=np.float64).reshape(-1)
    styles = np.asarray(style_returns, dtype=np.float64)
    if styles.ndim == 1:
        styles = styles.reshape(-1, 1)
    elif styles.ndim != 2:
        styles = styles.reshape(styles.shape[0], -1)
    mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    if y.shape[0] != styles.shape[0] or y.shape[0] != mask.shape[0]:
        raise ValueError("factor_returns, style_returns, and train_mask must align")

    finite = mask & np.isfinite(y)
    if styles.shape[1] > 0:
        finite &= np.all(np.isfinite(styles), axis=1)
    y_train = y[finite]
    styles_train = styles[finite]
    n_obs = int(y_train.shape[0])
    if n_obs < 20:
        return _barra_residual_sentinel_metrics(n_obs)

    x = np.column_stack([np.ones(n_obs, dtype=np.float64), styles_train])
    ridge = 1e-6
    xtx = x.T @ x
    rhs = x.T @ y_train
    ridge_eye = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    try:
        coef = np.linalg.solve(xtx + ridge_eye, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx + ridge_eye, rhs, rcond=None)[0]
    pred = x @ coef
    residual = y_train - pred

    netret = float(np.sum(residual))
    resid_mean = float(np.mean(residual))
    resid_std = float(np.std(residual, ddof=1)) if n_obs > 1 else np.nan
    if np.isfinite(resid_std) and resid_std > 1e-12:
        sharpe = float(resid_mean / resid_std * np.sqrt(float(periods_per_year)))
    else:
        sharpe = -1e6

    raw_netret = float(np.sum(y_train))
    if np.isfinite(raw_netret) and abs(raw_netret) > 1e-8:
        retention = float(netret / raw_netret)
    else:
        retention = -1e6

    ss_res = float(np.sum(residual * residual))
    centered = y_train - float(np.mean(y_train))
    ss_tot = float(np.sum(centered * centered))
    if np.isfinite(ss_tot) and ss_tot > 1e-12:
        r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
    else:
        r2 = 0.0
    return {
        "barra_resid_ls_netret": float(netret),
        "barra_resid_ls_net_sharpe": float(sharpe),
        "barra_resid_retention": float(retention),
        "barra_style_r2": float(r2),
        "n_obs": n_obs,
    }


POSITIVE_RATE_SOFT_ENABLED = os.environ.get("GP_POSITIVE_RATE_SOFT", "1").lower() in {
    "1", "true", "yes", "on",
}
POSITIVE_RATE_TARGET = float(os.environ.get("GP_POSITIVE_RATE_TARGET", "0.515"))
POSITIVE_RATE_MARGIN = float(os.environ.get("GP_POSITIVE_RATE_MARGIN", "0.050"))
YEARLY_POSITIVE_RATE_TARGET = float(os.environ.get("GP_YEARLY_POSITIVE_RATE_TARGET", "0.470"))
YEARLY_POSITIVE_RATE_MARGIN = float(os.environ.get("GP_YEARLY_POSITIVE_RATE_MARGIN", "0.050"))
POSITIVE_RATE_SOFT_STRENGTH = float(os.environ.get("GP_POSITIVE_RATE_SOFT_STRENGTH", "0.20"))
POSITIVE_RATE_SOFT_MIN_MULT = float(os.environ.get("GP_POSITIVE_RATE_SOFT_MIN_MULT", "0.80"))
POSITIVE_RATE_SOFT_MAX_MULT = float(os.environ.get("GP_POSITIVE_RATE_SOFT_MAX_MULT", "1.12"))
RANKICIR_SOFT_ENABLED = os.environ.get("GP_RANKICIR_SOFT", "1").lower() in {
    "1", "true", "yes", "on",
}
RANKICIR_SOFT_TARGET = float(os.environ.get("GP_RANKICIR_SOFT_TARGET", "0.0"))
RANKICIR_SOFT_MARGIN = float(os.environ.get("GP_RANKICIR_SOFT_MARGIN", "1.0"))
RANKICIR_SOFT_STRENGTH = float(os.environ.get("GP_RANKICIR_SOFT_STRENGTH", "0.25"))
RANKICIR_SOFT_MIN_MULT = float(os.environ.get("GP_RANKICIR_SOFT_MIN_MULT", "0.70"))
POOL_NOVELTY_SOFT_ENABLED = os.environ.get("GP_POOL_NOVELTY_SOFT", "1").lower() in {
    "1", "true", "yes", "on",
}
POOL_NOVELTY_TARGET = float(os.environ.get("GP_POOL_NOVELTY_TARGET", "0.25"))
POOL_NOVELTY_STRENGTH = float(os.environ.get("GP_POOL_NOVELTY_STRENGTH", "0.50"))
POOL_NOVELTY_MIN_MULT = float(os.environ.get("GP_POOL_NOVELTY_MIN_MULT", "0.50"))
BEHAVIOR_OVERLAP_SOFT_ENABLED = os.environ.get("GP_BEHAVIOR_OVERLAP_SOFT", "1").lower() in {
    "1", "true", "yes", "on",
}
BEHAVIOR_NOVELTY_TARGET = float(os.environ.get("GP_BEHAVIOR_NOVELTY_TARGET", "0.08"))
BEHAVIOR_OVERLAP_STRENGTH = float(os.environ.get("GP_BEHAVIOR_OVERLAP_STRENGTH", "0.35"))
BEHAVIOR_OVERLAP_MIN_MULT = float(os.environ.get("GP_BEHAVIOR_OVERLAP_MIN_MULT", "0.65"))

# ---------------------------------------------------------------------------
# Residual-IC pressure: cross-sectionally regress each candidate on a small
# basis of representative production factors and penalize candidates whose
# residual RankICIR is low (alpha fully explained by the existing base).
# ---------------------------------------------------------------------------
RESIDUAL_BASIS_ARCHIVE_PATH = os.environ.get("GP_RESIDUAL_BASIS_ARCHIVE", "").strip()
RESIDUAL_IC_SOFT_ENABLED = os.environ.get("GP_RESIDUAL_IC_SOFT", "1").lower() in {
    "1", "true", "yes", "on",
}
RESIDUAL_IC_TARGET = float(os.environ.get("GP_RESIDUAL_IC_TARGET", "1.0"))
RESIDUAL_IC_STRENGTH = float(os.environ.get("GP_RESIDUAL_IC_STRENGTH", "0.75"))
RESIDUAL_IC_MIN_MULT = float(os.environ.get("GP_RESIDUAL_IC_MIN_MULT", "0.30"))
RESIDUAL_IC_MAX_PERIODS = int(os.environ.get("GP_RESIDUAL_IC_MAX_PERIODS", "128"))
RESIDUAL_IC_MIN_PERIODS = int(os.environ.get("GP_RESIDUAL_IC_MIN_PERIODS", "30"))


def _perf_add(perf_stats, key, value):
    if perf_stats is not None:
        perf_stats[key] = perf_stats.get(key, 0.0) + float(value)


def _directed_formula_string(formula, direction):
    return formula if float(direction) >= 0.0 else f"-({formula})"


def _objective_dict(row, aux_dict=None, idx=None):
    out = {name: float(row[i]) for i, name in enumerate(_RESULT_OBJECTIVE_NAMES)}
    if aux_dict is not None and idx is not None:
        out["ls_netret"] = float(aux_dict["ls_netret_ann"][idx])
        out["rankicir"] = float(aux_dict["rankicir"][idx])
    return out


def _compute_search_scores(objectives):
    scores = np.full(len(objectives), -1e6, dtype=np.float32)
    if objectives is None or len(objectives) == 0:
        return scores
    valid = objectives[:, 1] > -1e5
    if np.any(valid):
        weights = SEARCH_SCORE_WEIGHTS
        n_cols = objectives.shape[1]
        if n_cols != weights.shape[0]:
            # Replayed/archive runs can carry a different objective width
            # (e.g. 5-wide vectors from before the novelty objective). Score
            # on the shared prefix instead of crashing.
            n = min(n_cols, weights.shape[0])
            scores[valid] = objectives[valid, :n] @ weights[:n]
        else:
            scores[valid] = objectives[valid] @ weights
    return scores


def _pool_novelty_soft_config():
    return {
        "enabled": bool(POOL_NOVELTY_SOFT_ENABLED),
        "target": float(np.clip(POOL_NOVELTY_TARGET, 1e-6, 1.0)),
        "strength": float(np.clip(POOL_NOVELTY_STRENGTH, 0.0, 1.0)),
        "min_multiplier": float(np.clip(POOL_NOVELTY_MIN_MULT, 1e-6, 1.0)),
    }


def _behavior_overlap_soft_config():
    return {
        "enabled": bool(BEHAVIOR_OVERLAP_SOFT_ENABLED),
        "target": float(np.clip(BEHAVIOR_NOVELTY_TARGET, 1e-6, 1.0)),
        "strength": float(np.clip(BEHAVIOR_OVERLAP_STRENGTH, 0.0, 1.0)),
        "min_multiplier": float(np.clip(BEHAVIOR_OVERLAP_MIN_MULT, 1e-6, 1.0)),
    }


def _external_corr_config():
    hard = float(np.clip(EXTERNAL_CORR_HARD, 1e-6, 1.0))
    target = float(np.clip(EXTERNAL_CORR_TARGET, 1e-6, hard - 1e-6 if hard > 1e-6 else hard))
    return {
        "archive_path": EXTERNAL_BEHAVIOR_ARCHIVE_PATH,
        "enabled": bool(EXTERNAL_CORR_SOFT_ENABLED and EXTERNAL_BEHAVIOR_ARCHIVE_PATH),
        "target": target,
        "hard": hard,
        "final_hard": float(np.clip(EXTERNAL_CORR_FINAL_HARD, 1e-6, 1.0)),
        "strength": float(np.clip(EXTERNAL_CORR_STRENGTH, 0.0, 1.0)),
        "min_multiplier": float(np.clip(EXTERNAL_CORR_MIN_MULT, 1e-6, 1.0)),
    }


def _load_external_behavior_archive(path):
    path = str(path or "").strip()
    if not path:
        return None, {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"GP_EXTERNAL_BEHAVIOR_ARCHIVE not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        key = "behavior_signatures" if "behavior_signatures" in data.files else "archive"
        if key not in data.files:
            raise ValueError(f"external behavior archive missing behavior_signatures/archive: {path}")
        archive = data[key].astype(np.float32, copy=False)
        meta = {}
        if "metadata_json" in data.files:
            try:
                meta = json.loads(str(data["metadata_json"]))
            except Exception:
                meta = {}
    if archive.ndim != 2 or archive.shape[0] == 0 or archive.shape[1] == 0:
        raise ValueError(f"external behavior archive is empty or invalid: {path}")
    return standardize_phenotypes(archive), meta


def _compute_external_behavior_corr(behavior_signatures, valid_mask, external_archive):
    corr = np.full(len(valid_mask), np.nan, dtype=np.float32)
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return corr
    if (
        behavior_signatures is None
        or external_archive is None
        or not isinstance(external_archive, np.ndarray)
        or external_archive.size == 0
    ):
        return corr
    if behavior_signatures.ndim != 2 or external_archive.ndim != 2:
        if not getattr(_compute_external_behavior_corr, "_warned_dim_mismatch", False):
            print(
                f"[EXTERNAL_ARCHIVE] dimension mismatch: behavior_signatures.shape="
                f"{behavior_signatures.shape} != archive.shape={external_archive.shape} "
                f"-> external_corr is NaN; will not penalize. "
                f"Check archive period_minutes / start_date / end_date / coin universe alignment.",
                flush=True,
            )
            _compute_external_behavior_corr._warned_dim_mismatch = True
        return corr
    if behavior_signatures.shape[1] != external_archive.shape[1]:
        if not getattr(_compute_external_behavior_corr, "_warned_dim_mismatch", False):
            print(
                f"[EXTERNAL_ARCHIVE] dimension mismatch: behavior_signatures.shape[1]="
                f"{behavior_signatures.shape[1]} != archive.shape[1]={external_archive.shape[1]} "
                f"-> external_corr is NaN; will not penalize. "
                f"Check archive period_minutes / start_date / end_date / coin universe alignment.",
                flush=True,
            )
            _compute_external_behavior_corr._warned_dim_mismatch = True
        return corr
    sig = standardize_phenotypes(behavior_signatures[valid_idx])
    corr[valid_idx] = np.abs(sig @ external_archive.T).max(axis=1).astype(np.float32, copy=False)
    return corr


def _external_corr_multipliers(external_corr, config=None):
    config = _external_corr_config() if config is None else config
    corr = np.asarray(external_corr, dtype=np.float32)
    mult = np.ones(len(corr), dtype=np.float32)
    if not config.get("enabled"):
        return mult
    valid = np.isfinite(corr)
    if not np.any(valid):
        return mult
    target = np.float32(config["target"])
    hard = np.float32(config["hard"])
    denom = np.maximum(hard - target, np.float32(1e-6))
    severity = np.clip((corr[valid] - target) / denom, 0.0, 1.0)
    raw = np.float32(1.0) - np.float32(config["strength"]) * np.square(severity)
    mult[valid] = np.clip(raw, np.float32(config["min_multiplier"]), 1.0)
    return mult


def _apply_external_behavior_corr_penalty(search_objectives, external_corr, config=None):
    config = _external_corr_config() if config is None else config
    multipliers = _external_corr_multipliers(external_corr, config=config)
    if not config.get("enabled"):
        return search_objectives, multipliers, 0
    out = np.array(search_objectives, dtype=np.float32, copy=True)
    corr = np.asarray(external_corr, dtype=np.float32)
    valid = (out[:, 1] > -1e5) & np.isfinite(corr)
    hard = valid & (corr >= np.float32(config["hard"]))
    if np.any(hard):
        out[hard] = -1e6
    soft = valid & ~hard & np.isfinite(multipliers)
    if np.any(soft):
        safe_mult = np.maximum(multipliers[soft], np.float32(1e-6))
        for col in (0, 1, 2, 3):
            values = out[soft, col]
            out[soft, col] = np.where(values >= 0.0, values * safe_mult, values / safe_mult)
    return out, multipliers, int(np.sum(hard))


def _positive_rate_soft_config():
    return {
        "enabled": bool(POSITIVE_RATE_SOFT_ENABLED),
        "target": float(np.clip(POSITIVE_RATE_TARGET, 0.0, 1.0)),
        "margin": float(np.clip(POSITIVE_RATE_MARGIN, 1e-6, 1.0)),
        "yearly_target": float(np.clip(YEARLY_POSITIVE_RATE_TARGET, 0.0, 1.0)),
        "yearly_margin": float(np.clip(YEARLY_POSITIVE_RATE_MARGIN, 1e-6, 1.0)),
        "strength": float(np.clip(POSITIVE_RATE_SOFT_STRENGTH, 0.0, 1.0)),
        "min_multiplier": float(np.clip(POSITIVE_RATE_SOFT_MIN_MULT, 1e-6, 1.0)),
        "max_multiplier": float(np.clip(POSITIVE_RATE_SOFT_MAX_MULT, 1.0, 2.0)),
    }


def _positive_rate_multipliers(is_positive_rate, min_yearly_positive_rate, config=None):
    config = _positive_rate_soft_config() if config is None else config
    n = len(is_positive_rate) if is_positive_rate is not None else len(min_yearly_positive_rate)
    mult = np.ones(n, dtype=np.float32)
    if not config.get("enabled"):
        return mult
    is_pr = np.asarray(is_positive_rate, dtype=np.float32)
    yearly_pr = np.asarray(min_yearly_positive_rate, dtype=np.float32)
    valid = np.isfinite(is_pr) & np.isfinite(yearly_pr)
    if not np.any(valid):
        return mult
    is_edge = np.clip(
        (is_pr[valid] - np.float32(config["target"])) / np.float32(config["margin"]),
        -1.0,
        1.0,
    )
    yearly_edge = np.clip(
        (yearly_pr[valid] - np.float32(config["yearly_target"])) / np.float32(config["yearly_margin"]),
        -1.0,
        1.0,
    )
    quality = np.float32(0.65) * is_edge + np.float32(0.35) * yearly_edge
    raw = np.float32(1.0) + np.float32(config["strength"]) * quality
    mult[valid] = np.clip(
        raw,
        np.float32(config["min_multiplier"]),
        np.float32(config["max_multiplier"]),
    ).astype(np.float32, copy=False)
    return mult


def _apply_positive_rate_soft_multiplier(
    search_objectives,
    is_positive_rate,
    min_yearly_positive_rate,
    config=None,
):
    config = _positive_rate_soft_config() if config is None else config
    multipliers = _positive_rate_multipliers(
        is_positive_rate,
        min_yearly_positive_rate,
        config=config,
    )
    if not config.get("enabled"):
        return search_objectives, multipliers
    out = np.array(search_objectives, dtype=np.float32, copy=True)
    valid = (out[:, 1] > -1e5) & np.isfinite(multipliers)
    if not np.any(valid):
        return out, multipliers
    safe_mult = np.maximum(multipliers[valid], np.float32(1e-6))
    for col in (0, 1, 2, 3):
        values = out[valid, col]
        out[valid, col] = np.where(values >= 0.0, values * safe_mult, values / safe_mult)
    return out, multipliers


def _rankicir_soft_config():
    return {
        "enabled": bool(
            RANKICIR_SOFT_ENABLED
            and SEARCH_OBJECTIVE_PROFILE == _SUBMIT_SHARPE_SEARCH_OBJECTIVE_PROFILE
        ),
        "target": float(RANKICIR_SOFT_TARGET),
        "margin": float(np.clip(RANKICIR_SOFT_MARGIN, 1e-6, 1e6)),
        "strength": float(np.clip(RANKICIR_SOFT_STRENGTH, 0.0, 1.0)),
        "min_multiplier": float(np.clip(RANKICIR_SOFT_MIN_MULT, 1e-6, 1.0)),
    }


def _rankicir_quality_multipliers(rankicir, min_yearly_rankicir=None, config=None):
    config = _rankicir_soft_config() if config is None else config
    rankicir = np.asarray(rankicir, dtype=np.float32)
    mult = np.ones(len(rankicir), dtype=np.float32)
    if not config.get("enabled"):
        return mult
    quality = rankicir.copy()
    if min_yearly_rankicir is not None:
        yearly = np.asarray(min_yearly_rankicir, dtype=np.float32)
        valid_yearly = np.isfinite(yearly)
        quality[valid_yearly] = np.minimum(quality[valid_yearly], yearly[valid_yearly])
    valid = np.isfinite(quality)
    if not np.any(valid):
        return mult
    deficit = np.clip(
        (np.float32(config["target"]) - quality[valid]) / np.float32(config["margin"]),
        0.0,
        1.0,
    )
    raw = np.float32(1.0) - np.float32(config["strength"]) * np.square(deficit)
    mult[valid] = np.clip(raw, np.float32(config["min_multiplier"]), 1.0)
    return mult


def _apply_rankicir_soft_multiplier(
    search_objectives,
    rankicir,
    min_yearly_rankicir=None,
    config=None,
):
    config = _rankicir_soft_config() if config is None else config
    multipliers = _rankicir_quality_multipliers(
        rankicir,
        min_yearly_rankicir=min_yearly_rankicir,
        config=config,
    )
    if not config.get("enabled"):
        return search_objectives, multipliers
    out = np.array(search_objectives, dtype=np.float32, copy=True)
    valid = (out[:, 1] > -1e5) & np.isfinite(multipliers)
    if not np.any(valid):
        return out, multipliers
    safe_mult = np.maximum(multipliers[valid], np.float32(1e-6))
    for col in (0, 1, 2, 3):
        values = out[valid, col]
        out[valid, col] = np.where(values >= 0.0, values * safe_mult, values / safe_mult)
    return out, multipliers


def _compute_archive_novelty(phenotypes, valid_mask, archive_phenotypes=None):
    novelty = np.full(len(valid_mask), -1e6, dtype=np.float32)
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return novelty
    if phenotypes is None or phenotypes.shape[0] != len(valid_mask) or phenotypes.shape[1] == 0:
        novelty[valid_idx] = 1.0
        return novelty
    if (
        archive_phenotypes is None
        or not isinstance(archive_phenotypes, np.ndarray)
        or archive_phenotypes.size == 0
    ):
        novelty[valid_idx] = 1.0
        return novelty
    pheno = standardize_phenotypes(phenotypes[valid_idx])
    if archive_phenotypes.shape[1] != pheno.shape[1]:
        novelty[valid_idx] = 1.0
        return novelty
    max_corr = np.abs(pheno @ archive_phenotypes.T).max(axis=1)
    novelty[valid_idx] = 1.0 - np.clip(max_corr.astype(np.float32, copy=False), 0.0, 1.0)
    return novelty


def _pool_novelty_multipliers(pool_novelty, config=None):
    config = _pool_novelty_soft_config() if config is None else config
    mult = np.ones(len(pool_novelty), dtype=np.float32)
    if not config.get("enabled"):
        return mult
    target = float(config["target"])
    strength = float(config["strength"])
    min_mult = float(config["min_multiplier"])
    valid = np.isfinite(pool_novelty) & (pool_novelty > -1e5)
    if not np.any(valid):
        return mult
    deficit = np.clip((target - pool_novelty[valid]) / target, 0.0, 1.0)
    penalty = 1.0 - strength * np.square(deficit)
    mult[valid] = np.clip(penalty, min_mult, 1.0).astype(np.float32, copy=False)
    return mult


def _apply_pool_novelty_soft_penalty(search_objectives, pool_novelty, config=None):
    config = _pool_novelty_soft_config() if config is None else config
    multipliers = _pool_novelty_multipliers(pool_novelty, config=config)
    if not config.get("enabled"):
        return search_objectives, multipliers
    out = np.array(search_objectives, dtype=np.float32, copy=True)
    valid = (out[:, 1] > -1e5) & np.isfinite(multipliers)
    if not np.any(valid):
        return out, multipliers
    safe_mult = np.maximum(multipliers[valid], np.float32(1e-6))
    # Penalize return/sharpe dimensions only. Turnover remains a pure cost objective.
    for col in (0, 1, 2, 3):
        values = out[valid, col]
        out[valid, col] = np.where(values >= 0.0, values * safe_mult, values / safe_mult)
    return out, multipliers


def _combine_search_novelty(pool_novelty, external_corr, valid_mask, rankic_novelty=None):
    """Per-individual novelty for the 6th search objective.

    Production RankIC novelty is primary when supplied. Existing pool/external
    behavior novelty remains the fallback for compatibility and ablations.
    Individuals without any reference get 1.0 (neutral-high).
    Rows outside valid_mask stay at -1e6 so they never win the objective.
    """
    out = np.full(len(valid_mask), -1e6, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid_mask):
        return out
    pool = np.asarray(pool_novelty, dtype=np.float32)
    pool_ok = np.isfinite(pool) & (pool > -1e5)
    ext = np.asarray(external_corr, dtype=np.float32)
    ext_ok = np.isfinite(ext)
    nov = np.where(pool_ok, pool, np.nan)
    ext_nov = np.where(ext_ok, 1.0 - np.clip(ext, 0.0, 1.0), np.nan)
    both = np.isfinite(nov) & np.isfinite(ext_nov)
    only_pool = np.isfinite(nov) & ~np.isfinite(ext_nov)
    only_ext = ~np.isfinite(nov) & np.isfinite(ext_nov)
    combined = np.full(len(valid_mask), 1.0, dtype=np.float32)
    combined[both] = np.minimum(nov[both], ext_nov[both])
    combined[only_pool] = nov[only_pool]
    combined[only_ext] = ext_nov[only_ext]
    if rankic_novelty is not None:
        rankic = np.asarray(rankic_novelty, dtype=np.float32)
        rankic_ok = np.isfinite(rankic) & (rankic > -1e5)
        combined[rankic_ok] = np.clip(rankic[rankic_ok], 0.0, 1.0)
    out[valid_mask] = combined[valid_mask]
    return out


def _residual_ic_soft_config():
    return {
        "enabled": bool(RESIDUAL_IC_SOFT_ENABLED),
        "target": float(max(RESIDUAL_IC_TARGET, 1e-6)),
        "strength": float(np.clip(RESIDUAL_IC_STRENGTH, 0.0, 1.0)),
        "min_multiplier": float(np.clip(RESIDUAL_IC_MIN_MULT, 1e-6, 1.0)),
    }


def _residual_ic_multipliers(resid_rankicir, config=None):
    config = _residual_ic_soft_config() if config is None else config
    mult = np.ones(len(resid_rankicir), dtype=np.float32)
    if not config.get("enabled"):
        return mult
    resid = np.asarray(resid_rankicir, dtype=np.float32)
    valid = np.isfinite(resid)
    if not np.any(valid):
        return mult
    deficit = np.clip(
        (np.float32(config["target"]) - resid[valid]) / np.float32(config["target"]),
        0.0,
        1.0,
    )
    raw = np.float32(1.0) - np.float32(config["strength"]) * np.square(deficit)
    mult[valid] = np.clip(raw, np.float32(config["min_multiplier"]), 1.0)
    return mult


def _apply_residual_ic_soft_penalty(search_objectives, resid_rankicir, config=None):
    config = _residual_ic_soft_config() if config is None else config
    multipliers = _residual_ic_multipliers(resid_rankicir, config=config)
    if not config.get("enabled"):
        return search_objectives, multipliers
    out = np.array(search_objectives, dtype=np.float32, copy=True)
    valid = (out[:, 1] > -1e5) & np.isfinite(multipliers)
    if not np.any(valid):
        return out, multipliers
    safe_mult = np.maximum(multipliers[valid], np.float32(1e-6))
    for col in (0, 1, 2, 3):
        values = out[valid, col]
        out[valid, col] = np.where(values >= 0.0, values * safe_mult, values / safe_mult)
    return out, multipliers


def _load_rankic_novelty_basis(path, problem):
    """Load the production daily RankIC library on the current run IS dates."""
    path = str(path or "").strip()
    if not path:
        return None, {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"GP_RANKIC_LIBRARY_PATH not found: {path}")

    rankics = pd.read_parquet(path)
    if not isinstance(rankics, pd.DataFrame) or rankics.ndim != 2:
        raise ValueError(f"RankIC library must be a 2D DataFrame: {path}")
    if rankics.shape[0] == 0 or rankics.shape[1] == 0:
        raise ValueError(f"RankIC library is empty: {path}")
    dates = pd.to_datetime(rankics.index, errors="coerce").normalize()
    if bool(pd.isna(dates).any()):
        raise ValueError(f"RankIC library contains invalid dates: {path}")
    if dates.has_duplicates:
        raise ValueError(f"RankIC library contains duplicate dates: {path}")
    rankics = rankics.copy()
    rankics.index = dates

    active_dates = pd.to_datetime(
        np.asarray(problem.loader.dates_flat)[
            np.asarray(problem.loader.active_period_indices, dtype=np.int64)
        ].astype(str),
        format="%Y%m%d",
        errors="coerce",
    ).normalize()
    if bool(pd.isna(active_dates).any()):
        raise ValueError("run loader contains invalid active period dates")
    is_context = problem._fitness_context_is
    if is_context is None:
        raise ValueError("run has no usable IS fitness context")
    is_positions = np.asarray(is_context[0], dtype=np.int64)
    if is_positions.size == 0:
        raise ValueError("run IS context is empty")
    if np.any(is_positions < 0) or np.any(is_positions >= active_dates.size):
        raise ValueError("run IS positions exceed active period date range")
    is_dates = active_dates[is_positions]
    aligned = rankics.reindex(is_dates)
    aligned_values = aligned.to_numpy(dtype=np.float32, copy=False)
    finite_counts = np.isfinite(aligned_values).sum(axis=0)
    keep = finite_counts >= int(RANKIC_NOVELTY_MIN_VALID)
    if not np.any(keep):
        raise ValueError(
            f"no production RankIC columns have {RANKIC_NOVELTY_MIN_VALID} "
            f"finite IS observations: {path}"
        )
    basis = aligned_values[:, keep].astype(np.float32, copy=False)
    meta = {
        "path": path,
        "n_library": int(basis.shape[1]),
        "n_is_periods": int(basis.shape[0]),
        "date_start": str(is_dates[0].date()),
        "date_end": str(is_dates[-1].date()),
        "min_valid": int(RANKIC_NOVELTY_MIN_VALID),
    }
    print(
        f"[RANKIC_NOVELTY] basis loaded: K={basis.shape[1]} "
        f"T={basis.shape[0]} IS={meta['date_start']}..{meta['date_end']} "
        f"from {path}",
        flush=True,
    )
    return basis, meta


def _load_residual_basis_context(path, problem):
    """Load a residual-IC basis archive and bind it to the problem's IS context.

    The npz must carry basis_panels (K, P, S), plus symbols / n_active_periods
    metadata matching the current run; otherwise the pressure is disabled.
    """
    path = str(path or "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"GP_RESIDUAL_BASIS_ARCHIVE not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "basis_panels" not in data.files:
            raise ValueError(f"residual basis archive missing basis_panels: {path}")
        panels = data["basis_panels"].astype(np.float32)
        meta = {}
        if "metadata_json" in data.files:
            try:
                meta = json.loads(str(data["metadata_json"]))
            except Exception:
                meta = {}
        symbols = (
            [str(s) for s in data["symbols"].tolist()]
            if "symbols" in data.files
            else None
        )
    n_periods_meta = int(meta.get("n_active_periods", panels.shape[1]))
    if panels.shape[1] != problem.n_active_periods or n_periods_meta != problem.n_active_periods:
        raise ValueError(
            f"residual basis period mismatch: panels P={panels.shape[1]} "
            f"meta P={n_periods_meta} vs run P={problem.n_active_periods}"
        )
    if panels.shape[2] != problem.n_coins:
        raise ValueError(
            f"residual basis symbol mismatch: panels S={panels.shape[2]} "
            f"vs run S={problem.n_coins}"
        )
    if symbols is not None:
        run_symbols = [str(s) for s in problem.loader.symbols[problem.loader.coin_indices]]
        if symbols != run_symbols:
            raise ValueError("residual basis symbols do not match run universe order")
    ctx = prepare_residual_basis_context(
        panels,
        problem._fitness_context_is,
        max_periods=RESIDUAL_IC_MAX_PERIODS,
    )
    if ctx is None:
        raise ValueError("residual basis has too few usable IS periods")
    print(
        f"[RESIDUAL_IC] basis loaded: K={panels.shape[0]} "
        f"sampled_periods={ctx['Bc'].shape[0]} from {path}",
        flush=True,
    )
    return ctx


def _compose_annual_worst_search_objectives(is_objectives, min_yearly_netret, min_yearly_sharpe):
    out = np.full((len(is_objectives), SEARCH_N_OBJECTIVES), -1e6, dtype=np.float32)
    if is_objectives is None or len(is_objectives) == 0:
        return out
    min_yearly_netret = np.asarray(min_yearly_netret, dtype=np.float32)
    min_yearly_sharpe = np.asarray(min_yearly_sharpe, dtype=np.float32)
    valid = (
        (is_objectives[:, 1] > -1e5)
        & np.isfinite(min_yearly_netret)
        & np.isfinite(min_yearly_sharpe)
    )
    if np.any(valid):
        out[valid, 0] = is_objectives[valid, 0]
        out[valid, 1] = is_objectives[valid, 1]
        out[valid, 2] = min_yearly_netret[valid]
        out[valid, 3] = min_yearly_sharpe[valid]
        out[valid, 4] = is_objectives[valid, 2]
        if SEARCH_NOVELTY_OBJECTIVE_ENABLED:
            # Neutral placeholder; overwritten with real novelty downstream.
            out[valid, 5] = 0.0
    return out


def _positive_rate_quality(is_positive_rate, min_yearly_positive_rate):
    is_pr = np.asarray(is_positive_rate, dtype=np.float32)
    yearly_pr = np.asarray(min_yearly_positive_rate, dtype=np.float32)
    quality = np.full(len(is_pr), -1e6, dtype=np.float32)
    valid = np.isfinite(is_pr) & np.isfinite(yearly_pr)
    if np.any(valid):
        quality[valid] = (
            np.float32(0.65) * (is_pr[valid] - np.float32(POSITIVE_RATE_TARGET))
            + np.float32(0.35) * (yearly_pr[valid] - np.float32(YEARLY_POSITIVE_RATE_TARGET))
        )
    return quality


def _compose_submit_sharpe_search_objectives(
    is_objectives,
    min_yearly_netret,
    min_yearly_sharpe,
    is_positive_rate,
    min_yearly_positive_rate,
):
    out = np.full((len(is_objectives), SEARCH_N_OBJECTIVES), -1e6, dtype=np.float32)
    if is_objectives is None or len(is_objectives) == 0:
        return out
    min_yearly_netret = np.asarray(min_yearly_netret, dtype=np.float32)
    min_yearly_sharpe = np.asarray(min_yearly_sharpe, dtype=np.float32)
    positive_quality = _positive_rate_quality(is_positive_rate, min_yearly_positive_rate)
    valid = (
        (is_objectives[:, 1] > -1e5)
        & np.isfinite(min_yearly_netret)
        & np.isfinite(min_yearly_sharpe)
        & np.isfinite(positive_quality)
    )
    if np.any(valid):
        out[valid, 0] = is_objectives[valid, 1]
        out[valid, 1] = min_yearly_sharpe[valid]
        out[valid, 2] = min_yearly_netret[valid]
        out[valid, 3] = positive_quality[valid]
        out[valid, 4] = is_objectives[valid, 2]
        if SEARCH_NOVELTY_OBJECTIVE_ENABLED:
            # Neutral placeholder; overwritten with real novelty downstream.
            out[valid, 5] = 0.0
    return out


def _compose_search_objectives(
    is_objectives,
    min_yearly_netret,
    min_yearly_sharpe,
    is_positive_rate=None,
    min_yearly_positive_rate=None,
):
    if SEARCH_OBJECTIVE_PROFILE == _SUBMIT_SHARPE_SEARCH_OBJECTIVE_PROFILE:
        return _compose_submit_sharpe_search_objectives(
            is_objectives,
            min_yearly_netret,
            min_yearly_sharpe,
            is_positive_rate,
            min_yearly_positive_rate,
        )
    return _compose_annual_worst_search_objectives(
        is_objectives,
        min_yearly_netret,
        min_yearly_sharpe,
    )


def _yearly_min_from_store(yearly_store, n_rows):
    min_netret = np.full(n_rows, np.inf, dtype=np.float32)
    min_sharpe = np.full(n_rows, np.inf, dtype=np.float32)
    min_positive_rate = np.full(n_rows, np.inf, dtype=np.float32)
    min_rankicir = np.full(n_rows, np.inf, dtype=np.float32)
    valid_all = np.ones(n_rows, dtype=bool)
    has_window = False
    for store in yearly_store.values():
        objectives = store.get("objective")
        aux = store.get("aux") or {}
        netret = aux.get("ls_netret_ann")
        positive_rate = aux.get("ls_positive_rate")
        if objectives is None or netret is None or positive_rate is None:
            valid_all &= False
            continue
        has_window = True
        valid = (objectives[:, 1] > -1e5) & np.isfinite(netret) & np.isfinite(positive_rate)
        valid_all &= valid
        min_netret = np.minimum(min_netret, netret)
        min_sharpe = np.minimum(min_sharpe, objectives[:, 1])
        min_positive_rate = np.minimum(min_positive_rate, positive_rate)
        min_rankicir = np.minimum(min_rankicir, objectives[:, 0])
    if not has_window:
        valid_all &= False
    min_netret = np.where(valid_all, min_netret, -1e6).astype(np.float32, copy=False)
    min_sharpe = np.where(valid_all, min_sharpe, -1e6).astype(np.float32, copy=False)
    min_positive_rate = np.where(valid_all, min_positive_rate, np.nan).astype(np.float32, copy=False)
    min_rankicir = np.where(valid_all, min_rankicir, np.nan).astype(np.float32, copy=False)
    return min_netret, min_sharpe, min_positive_rate, min_rankicir


def _is_finite_number(value):
    return value is not None and isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def _finite_values(values):
    return [
        float(value) for value in values
        if _is_finite_number(value)
    ]


def _safe_min(values):
    values = _finite_values(values)
    return min(values) if values else np.nan


def _safe_max(values):
    values = _finite_values(values)
    return max(values) if values else np.nan


def _safe_mean(values):
    values = _finite_values(values)
    return float(np.mean(values)) if values else np.nan


def _resolve_yearly_stability_windows(active_dates):
    active_dates = np.asarray(active_dates)
    if active_dates.size == 0:
        return []

    windows = []
    for label, start, end in FINAL_YEARLY_STABILITY_WINDOWS:
        mask = (active_dates >= start) & (active_dates < end)
        if np.any(mask):
            windows.append((label, start, end, mask))
    return windows


def _build_yearly_fitness_contexts(problem, scope="active"):
    active_dates = np.asarray(problem.loader.get_active_dates())
    if str(scope).lower() in {"train", "is", "in_sample"}:
        base_mask = np.asarray(problem.loader.is_mask, dtype=bool)
    else:
        base_mask = np.ones(len(active_dates), dtype=bool)
    out = []
    for label, start, end, mask in _resolve_yearly_stability_windows(active_dates):
        scoped_mask = mask & base_mask
        if not np.any(scoped_mask):
            continue
        ctx = prepare_fitness_context(
            problem._lagged_period_returns,
            is_mask=scoped_mask,
            tradable_mask=problem.period_tradable_mask,
        )
        if ctx is not None and problem._gpu_fitness:
            ctx = materialize_fitness_context(ctx)
        out.append(
            {
                "label": label,
                "start": start,
                "end": end,
                "n_periods": int(scoped_mask.sum()),
                "scope": str(scope),
                "context": ctx,
            }
        )
    return out


def _empty_yearly_metric_store(n_rows, yearly_contexts):
    return {
        item["label"]: {
            "objective": np.full((n_rows, N_OBJECTIVES), -1e6, dtype=np.float32),
            "aux": _empty_aux_store(n_rows),
            "start": item["start"],
            "end": item["end"],
            "n_periods": item["n_periods"],
            "has_context": item["context"] is not None,
        }
        for item in yearly_contexts
    }


def _yearly_window_metric(objectives, aux_store, idx):
    return {
        "ls_netret": float(aux_store["ls_netret_ann"][idx]),
        "ls_net_sharpe": float(objectives[idx, 1]),
        "turnover": (
            float(aux_store["ls_turnover"][idx])
            if np.isfinite(aux_store["ls_turnover"][idx]) else None
        ),
        "long_turnover": (
            float(aux_store["long_turnover"][idx])
            if np.isfinite(aux_store["long_turnover"][idx]) else None
        ),
        "coverage": float(aux_store["coverage"][idx]),
        "ls_positive_rate": float(aux_store["ls_positive_rate"][idx]),
        "rankicir": float(aux_store["rankicir"][idx]),
        "effective_bars": int(round(float(aux_store["effective_bars"][idx]))),
    }



def _safe_metric_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _sample_tail_factor_values(batch_factors, ic_directions, tradable_mask=None, max_periods=DIVERSITY_MAX_PERIODS):
    max_periods = max(1, int(max_periods))
    tail_step = max(1, batch_factors.shape[1] // max_periods)
    tail_sampled = batch_factors[:, ::tail_step, :]

    directions = np.asarray(ic_directions, dtype=np.float32).reshape(-1) if ic_directions is not None else None
    if directions is None or directions.size != tail_sampled.shape[0]:
        directions = np.ones(tail_sampled.shape[0], dtype=np.float32)
    tail_sampled = tail_sampled * directions[:, None, None]

    sampled_np = to_numpy(tail_sampled.astype(np.float32, copy=False))
    if tradable_mask is None:
        return sampled_np

    sampled_mask = np.asarray(to_numpy(tradable_mask[::tail_step]), dtype=bool)
    if sampled_mask.shape != sampled_np.shape[1:]:
        return sampled_np
    return np.where(sampled_mask[None, :, :], sampled_np, np.nan).astype(np.float32, copy=False)


def _compute_factor_shape_metrics(factors, tradable_mask=None):
    arr = to_numpy(factors).astype(np.float32, copy=False)
    if arr.ndim != 3:
        raise ValueError("factors must have shape (n_factors, n_periods, n_assets)")
    n_factors, n_periods, n_assets = arr.shape
    if tradable_mask is None:
        tradable = np.ones((n_periods, n_assets), dtype=bool)
    else:
        tradable = np.asarray(tradable_mask, dtype=bool)
        if tradable.shape != (n_periods, n_assets):
            raise ValueError("tradable_mask must have shape (n_periods, n_assets)")
    tradable_counts = tradable.sum(axis=1).astype(np.float32)
    period_has_universe = tradable_counts > 0
    out = []
    for i in range(n_factors):
        factor = arr[i]
        valid = np.isfinite(factor) & tradable
        valid_counts = valid.sum(axis=1).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            period_valid_ratio = np.divide(
                valid_counts,
                tradable_counts,
                out=np.zeros_like(valid_counts, dtype=np.float32),
                where=tradable_counts > 0,
            )
        usable_period = period_has_universe & (period_valid_ratio >= 0.50)
        finite_period_ratio = (
            float(np.mean(usable_period[period_has_universe]))
            if np.any(period_has_universe)
            else 0.0
        )
        finite_values = factor[valid]
        finite_count = int(finite_values.size)
        nonzero_ratio = (
            float(np.mean(np.abs(finite_values) > FINAL_FACTOR_NONZERO_EPS))
            if finite_count > 0
            else 0.0
        )
        xs_stds = []
        unique_counts = []
        for t in np.flatnonzero(usable_period):
            row = factor[t, valid[t]]
            if row.size == 0:
                continue
            xs_stds.append(float(np.nanstd(row)))
            unique_counts.append(float(np.unique(np.round(row.astype(np.float64), 8)).size))
        xs_std_median = float(np.median(xs_stds)) if xs_stds else 0.0
        unique_median = float(np.median(unique_counts)) if unique_counts else 0.0
        out.append({
            "finite_period_ratio": finite_period_ratio,
            "cell_finite_ratio": float(valid.sum() / max(1, int(tradable.sum()))),
            "nonzero_ratio": nonzero_ratio,
            "xs_std_median": xs_std_median,
            "unique_median": unique_median,
            "finite_count": finite_count,
            "usable_periods": int(np.sum(usable_period)),
        })
    return out


def _factor_shape_reasons(shape):
    if not isinstance(shape, dict):
        return ["factor_shape_missing"]
    reasons = []
    finite_period_ratio = _safe_metric_float(shape.get("finite_period_ratio"))
    nonzero_ratio = _safe_metric_float(shape.get("nonzero_ratio"))
    xs_std_median = _safe_metric_float(shape.get("xs_std_median"))
    unique_median = _safe_metric_float(shape.get("unique_median"))
    if finite_period_ratio is None or finite_period_ratio < FINAL_MIN_FACTOR_FINITE_PERIOD_RATIO:
        reasons.append("factor_low_finite_period")
    if nonzero_ratio is None or nonzero_ratio < FINAL_MIN_FACTOR_NONZERO_RATIO:
        reasons.append("factor_low_nonzero")
    if xs_std_median is None or xs_std_median < FINAL_MIN_FACTOR_XS_STD_MEDIAN:
        reasons.append("factor_low_xs_std")
    if unique_median is None or unique_median < FINAL_MIN_FACTOR_UNIQUE_MEDIAN:
        reasons.append("factor_low_unique")
    return reasons

def _yearly_stability_reasons(summary):
    reasons = []
    if not isinstance(summary, dict):
        return ["yearly_missing"]
    min_netret = summary.get("min_ls_netret")
    min_sharpe = summary.get("min_ls_net_sharpe")
    max_turnover = summary.get("max_turnover")
    min_coverage = summary.get("min_coverage")
    latest_netret = summary.get("latest_ls_netret")
    latest_sharpe = summary.get("latest_ls_net_sharpe")
    latest_to_mean = summary.get("latest_to_mean_sharpe_ratio")
    min_rankicir = summary.get("min_rankicir")
    latest_rankicir = summary.get("latest_rankicir")
    if not _is_finite_number(min_netret) or min_netret <= FINAL_MIN_YEARLY_LS_NETRET:
        reasons.append("yearly_netret")
    if not _is_finite_number(min_sharpe) or min_sharpe <= FINAL_MIN_YEARLY_LS_NET_SHARPE:
        reasons.append("yearly_sharpe")
    if not _is_finite_number(latest_netret) or latest_netret < FINAL_MIN_LATEST_YEAR_LS_NETRET:
        reasons.append("latest_year_netret")
    if not _is_finite_number(latest_sharpe) or latest_sharpe < FINAL_MIN_LATEST_YEAR_LS_NET_SHARPE:
        reasons.append("latest_year_sharpe")
    if not _is_finite_number(latest_to_mean) or latest_to_mean < FINAL_MIN_LATEST_TO_MEAN_SHARPE_RATIO:
        reasons.append("latest_year_sharpe_decay")
    if not _is_finite_number(min_rankicir) or min_rankicir < FINAL_MIN_YEARLY_RANKICIR:
        reasons.append("yearly_rankicir")
    if not _is_finite_number(latest_rankicir) or latest_rankicir < FINAL_MIN_LATEST_YEAR_RANKICIR:
        reasons.append("latest_year_rankicir")
    if FINAL_REQUIRE_LATEST_RANKIC_SIGN and (
        not _is_finite_number(latest_rankicir) or latest_rankicir <= 0.0
    ):
        reasons.append("latest_year_rankic_sign")
    if not _is_finite_number(max_turnover) or max_turnover > FINAL_MAX_YEARLY_TURNOVER:
        reasons.append("yearly_turnover")
    if not _is_finite_number(min_coverage) or min_coverage < FINAL_MIN_YEARLY_COVERAGE:
        reasons.append("yearly_coverage")
    return reasons


def _build_yearly_stability_entry(yearly_store, idx):
    windows = {}
    for label, store in yearly_store.items():
        windows[label] = {
            "start": store["start"],
            "end": store["end"],
            "n_periods": store["n_periods"],
            "has_context": bool(store["has_context"]),
            **_yearly_window_metric(store["objective"], store["aux"], idx),
        }
    ordered_windows = list(windows.items())
    latest_label, latest_metrics = ordered_windows[-1] if ordered_windows else (None, {})
    mean_sharpe = _safe_mean([m["ls_net_sharpe"] for m in windows.values()])
    latest_sharpe = latest_metrics.get("ls_net_sharpe")
    latest_to_mean = (
        float(latest_sharpe) / mean_sharpe
        if _is_finite_number(latest_sharpe) and _is_finite_number(mean_sharpe) and abs(mean_sharpe) > 1e-12
        else np.nan
    )
    summary = {
        "windows": list(windows.keys()),
        "min_ls_netret": _safe_min([m["ls_netret"] for m in windows.values()]),
        "min_ls_net_sharpe": _safe_min([m["ls_net_sharpe"] for m in windows.values()]),
        "mean_ls_net_sharpe": mean_sharpe,
        "latest_window": latest_label,
        "latest_ls_netret": latest_metrics.get("ls_netret"),
        "latest_ls_net_sharpe": latest_sharpe,
        "latest_to_mean_sharpe_ratio": latest_to_mean,
        "min_rankicir": _safe_min([m["rankicir"] for m in windows.values()]),
        "latest_rankicir": latest_metrics.get("rankicir"),
        "max_turnover": _safe_max([m["turnover"] for m in windows.values()]),
        "min_coverage": _safe_min([m["coverage"] for m in windows.values()]),
        "min_ls_positive_rate": _safe_min([m["ls_positive_rate"] for m in windows.values()]),
    }
    reasons = _yearly_stability_reasons(summary)
    summary["passed"] = not reasons
    summary["reasons"] = reasons
    summary["min_required"] = {
        "min_ls_netret": FINAL_MIN_YEARLY_LS_NETRET,
        "min_ls_net_sharpe": FINAL_MIN_YEARLY_LS_NET_SHARPE,
        "min_latest_year_ls_netret": FINAL_MIN_LATEST_YEAR_LS_NETRET,
        "min_latest_year_ls_net_sharpe": FINAL_MIN_LATEST_YEAR_LS_NET_SHARPE,
        "min_latest_to_mean_sharpe_ratio": FINAL_MIN_LATEST_TO_MEAN_SHARPE_RATIO,
        "min_yearly_rankicir": FINAL_MIN_YEARLY_RANKICIR,
        "min_latest_year_rankicir": FINAL_MIN_LATEST_YEAR_RANKICIR,
        "require_latest_rankic_sign": FINAL_REQUIRE_LATEST_RANKIC_SIGN,
        "max_turnover": FINAL_MAX_YEARLY_TURNOVER,
        "min_coverage": FINAL_MIN_YEARLY_COVERAGE,
    }
    return {"summary": summary, "windows": windows}


def _entry_hits_banned_ab_pair(entry):
    if not BANNED_AB_PAIRS:
        return False
    decoded = entry.get("decoded") or {}
    if int(decoded.get("mode") or 0) == 0:
        return False
    a_name = decoded.get("A")
    b_name = decoded.get("B")
    if a_name not in INDICATOR_INDEX or b_name not in INDICATOR_INDEX:
        return False
    pair = _ab_pair_key(INDICATOR_INDEX[a_name], INDICATOR_INDEX[b_name])
    banned = set(_ab_pair_key(a, b) for a, b in BANNED_AB_PAIRS)
    return pair in banned


def _final_quality_reasons(entry):
    obj = entry.get("is_objectives") or {}
    screening = entry.get("is_screening") or {}
    yearly = entry.get("yearly_stability") or {}
    reasons = []
    if _entry_hits_banned_ab_pair(entry):
        reasons.append("banned_ab_pair")
    external_corr = entry.get("external_behavior_corr_max")
    if _is_finite_number(external_corr) and external_corr >= EXTERNAL_CORR_FINAL_HARD:
        reasons.append("external_behavior_corr")
    netret = obj.get("ls_netret")
    sharpe = obj.get("ls_net_sharpe")
    turnover = screening.get("ls_turnover")
    long_turnover = screening.get("long_turnover")
    positive_rate = screening.get("ls_positive_rate")
    coverage = screening.get("coverage")
    if not _is_finite_number(netret) or netret <= FINAL_MIN_LS_NETRET:
        reasons.append("is_netret")
    if not _is_finite_number(sharpe) or sharpe < FINAL_MIN_LS_NET_SHARPE:
        reasons.append("is_sharpe")
    turnover_values = [
        float(value) for value in (turnover, long_turnover)
        if _is_finite_number(value)
    ]
    turnover_pressure = max(turnover_values) if turnover_values else np.nan
    if not np.isfinite(turnover_pressure) or turnover_pressure > FINAL_MAX_LS_TURNOVER:
        reasons.append("turnover")
    if not _is_finite_number(positive_rate) or positive_rate < FINAL_MIN_LS_POSITIVE_RATE:
        reasons.append("positive_rate")
    if not _is_finite_number(coverage) or coverage < FINAL_MIN_COVERAGE:
        reasons.append("coverage")
    reasons.extend(_factor_shape_reasons(entry.get("factor_shape")))
    reasons.extend(_yearly_stability_reasons(yearly.get("summary")))
    return reasons


def _apply_final_quality_gate(results):
    gated = []
    failed = {}
    for item in results:
        reasons = _final_quality_reasons(item)
        yearly_summary = ((item.get("yearly_stability") or {}).get("summary") or {})
        item["final_quality_gate"] = {
            "passed": not reasons,
            "reasons": reasons,
            "min_is_netret": FINAL_MIN_LS_NETRET,
            "min_is_net_sharpe": FINAL_MIN_LS_NET_SHARPE,
            "max_turnover": FINAL_MAX_LS_TURNOVER,
            "min_positive_rate": FINAL_MIN_LS_POSITIVE_RATE,
            "min_coverage": FINAL_MIN_COVERAGE,
            "factor_shape": {
                "min_finite_period_ratio": FINAL_MIN_FACTOR_FINITE_PERIOD_RATIO,
                "min_nonzero_ratio": FINAL_MIN_FACTOR_NONZERO_RATIO,
                "min_xs_std_median": FINAL_MIN_FACTOR_XS_STD_MEDIAN,
                "min_unique_median": FINAL_MIN_FACTOR_UNIQUE_MEDIAN,
                "nonzero_eps": FINAL_FACTOR_NONZERO_EPS,
            },
            "external_behavior_corr": {
                "archive_path": EXTERNAL_BEHAVIOR_ARCHIVE_PATH,
                "target": EXTERNAL_CORR_TARGET,
                "hard": EXTERNAL_CORR_HARD,
                "final_hard": EXTERNAL_CORR_FINAL_HARD,
                "strength": EXTERNAL_CORR_STRENGTH,
                "min_multiplier": EXTERNAL_CORR_MIN_MULT,
            },
            "search_novelty_objective": {
                "enabled": SEARCH_NOVELTY_OBJECTIVE_ENABLED,
                "weight": SEARCH_NOVELTY_WEIGHT,
            },
            "rankic_novelty": {
                "enabled": bool(RANKIC_NOVELTY_OBJECTIVE_ENABLED),
                "library_path": RANKIC_NOVELTY_LIBRARY_PATH,
                "min_valid": RANKIC_NOVELTY_MIN_VALID,
                "library_chunk": RANKIC_NOVELTY_LIBRARY_CHUNK,
            },
            "residual_ic": {
                "archive_path": RESIDUAL_BASIS_ARCHIVE_PATH,
                "enabled": RESIDUAL_IC_SOFT_ENABLED,
                "target": RESIDUAL_IC_TARGET,
                "strength": RESIDUAL_IC_STRENGTH,
                "min_multiplier": RESIDUAL_IC_MIN_MULT,
                "max_periods": RESIDUAL_IC_MAX_PERIODS,
                "min_periods": RESIDUAL_IC_MIN_PERIODS,
            },
            "yearly_stability": {
                "windows": list(yearly_summary.get("windows") or []),
                "min_yearly_ls_netret": FINAL_MIN_YEARLY_LS_NETRET,
                "min_yearly_ls_net_sharpe": FINAL_MIN_YEARLY_LS_NET_SHARPE,
                "min_latest_year_ls_netret": FINAL_MIN_LATEST_YEAR_LS_NETRET,
                "min_latest_year_ls_net_sharpe": FINAL_MIN_LATEST_YEAR_LS_NET_SHARPE,
                "min_latest_to_mean_sharpe_ratio": FINAL_MIN_LATEST_TO_MEAN_SHARPE_RATIO,
                "min_yearly_rankicir": FINAL_MIN_YEARLY_RANKICIR,
                "min_latest_year_rankicir": FINAL_MIN_LATEST_YEAR_RANKICIR,
                "require_latest_rankic_sign": FINAL_REQUIRE_LATEST_RANKIC_SIGN,
                "max_yearly_turnover": FINAL_MAX_YEARLY_TURNOVER,
                "min_yearly_coverage": FINAL_MIN_YEARLY_COVERAGE,
            },
        }
        if reasons:
            for reason in reasons:
                failed[reason] = failed.get(reason, 0) + 1
            continue
        gated.append(item)
    return gated, failed


def _dedup_by_formula(results):
    deduped = []
    seen = set()
    n_removed = 0
    for item in results:
        formula = str(item.get("formula") or item.get("raw_formula") or "")
        if formula in seen:
            item["final_formula_dedup_gate"] = {
                "passed": False,
                "reason": "duplicate_formula",
            }
            n_removed += 1
            continue
        seen.add(formula)
        item["final_formula_dedup_gate"] = {
            "passed": True,
            "reason": None,
        }
        deduped.append(item)
    return deduped, n_removed


def _dedup_by_signature(results):
    deduped = []
    seen = set()
    n_removed = 0
    for item in results:
        sig = _signature_of(item)
        if sig is not None and sig in seen:
            item["final_signature_dedup_gate"] = {
                "passed": False,
                "reason": "duplicate_signature",
            }
            n_removed += 1
            continue
        if sig is not None:
            seen.add(sig)
        item["final_signature_dedup_gate"] = {
            "passed": True,
            "reason": None,
        }
        deduped.append(item)
    return deduped, n_removed


def _load_signature_blacklist(paths):
    blacklist = set()
    for raw_path in str(paths or "").split(":"):
        path = raw_path.strip()
        if not path or not os.path.exists(path):
            continue
        try:
            if os.path.getsize(path) <= 0:
                continue
            with open(path) as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                raise ValueError("blacklist JSON must be a list")
            for row in rows:
                sig = _signature_of(row)
                if sig is not None:
                    blacklist.add(sig)
        except Exception as exc:
            print(f"[BLACKLIST] skip {path}: {exc}", flush=True)
    return blacklist


def _dedup_by_signature_blacklist(results, blacklist):
    if not blacklist:
        for item in results:
            item["final_signature_blacklist_gate"] = {
                "passed": True,
                "reason": None,
            }
        return results, 0
    filtered = []
    n_removed = 0
    for item in results:
        sig = _signature_of(item)
        if sig is not None and sig in blacklist:
            item["final_signature_blacklist_gate"] = {
                "passed": False,
                "reason": "external_signature_match",
            }
            n_removed += 1
            continue
        item["final_signature_blacklist_gate"] = {
            "passed": True,
            "reason": None,
        }
        filtered.append(item)
    return filtered, n_removed


def _decoded_operator(decoded):
    try:
        mode = int(decoded.get("mode") or 0)
    except Exception:
        mode = 0
    return str(decoded.get(f"mode{mode}_op") or decoded.get("mode2_op") or "")


def _family_of_field(field_name):
    return pair_a_family_of_field(field_name)


def _safe_bucket(value, scale, default=None):
    try:
        return int(round(float(value) * scale))
    except Exception:
        return default


def _signature_of(decoded_or_item):
    if decoded_or_item is None:
        return None
    if not isinstance(decoded_or_item, dict):
        return None
    decoded = decoded_or_item.get("decoded")
    if decoded is None and ("mode" in decoded_or_item or "mode4_op" in decoded_or_item):
        decoded = decoded_or_item
    if not isinstance(decoded, dict):
        return None
    op = _decoded_operator(decoded)
    if not op:
        return None
    window_bucket = _safe_bucket(decoded.get("window"), 1.0 / 60.0)
    slice_value = decoded.get("slice")
    slice_bucket = -1 if slice_value is None else _safe_bucket(slice_value, 10.0, default=-1)
    return (
        op,
        _family_of_field(decoded.get("A")),
        _family_of_field(decoded.get("B")) if decoded.get("B") else None,
        _family_of_field(decoded.get("mask_field")) if decoded.get("mask_field") else None,
        window_bucket,
        slice_bucket,
        decoded.get("ts_comp_op"),
    )


def _final_diversity_keys(item):
    decoded = item.get("decoded") if isinstance(item.get("decoded"), dict) else {}
    op = _decoded_operator(decoded)
    mode = decoded.get("mode")
    a_field = decoded.get("A")
    b_field = decoded.get("B") if mode == 2 else None
    window = decoded.get("window")
    mask_field = decoded.get("mask_field")
    mask_rule = decoded.get("mask_rule")
    lag = decoded.get("B_shift_lag") if mode == 2 else None
    return {
        "operator": (mode, op),
        "a_field": a_field,
        "source_family": indicator_source_family(a_field),
        "pair_a_family": _family_of_field(a_field),
        "ts_comp_op": decoded.get("ts_comp_op"),
        "ab_pair": (mode, op, a_field, b_field),
        "template": (mode, op, a_field, b_field, window, mask_field, mask_rule, lag),
    }


def _apply_final_diversity_constraints(results):
    if not results:
        return results, {}
    n_total = max(1, len(results))
    limits = {
        "operator": FINAL_DIVERSITY_MAX_PER_OPERATOR,
        "a_field": FINAL_DIVERSITY_MAX_PER_A_FIELD,
        "source_family": FINAL_DIVERSITY_MAX_PER_SOURCE_FAMILY,
        "ab_pair": FINAL_DIVERSITY_MAX_PER_AB_PAIR,
        "template": FINAL_DIVERSITY_MAX_PER_TEMPLATE,
    }
    counts = {name: {} for name in limits}
    counts["pair_a_family"] = {}
    counts["ts_comp_op"] = {}
    pair_a_family_caps = {}
    pair_a_family_counts = {}
    ts_comp_op_counts = {}

    def _pair_a_family_cap(family):
        if not FINAL_PAIR_A_FAMILY_GATE_ENABLED or family is None:
            return 0
        if family in pair_a_family_caps:
            return pair_a_family_caps[family]
        raw_share = FINAL_PAIR_A_FAMILY_MAX_SHARE.get(family, FINAL_PAIR_A_FAMILY_DEFAULT_CAP)
        try:
            share = float(raw_share)
        except (TypeError, ValueError):
            share = float(FINAL_PAIR_A_FAMILY_DEFAULT_CAP)
        cap = max(1, int(np.ceil(n_total * share)))
        pair_a_family_caps[family] = cap
        return cap

    ts_comp_op_cap = 0 if FINAL_TS_COMP_OP_MAX_SHARE <= 0.0 else max(1, int(np.ceil(n_total * FINAL_TS_COMP_OP_MAX_SHARE)))
    filtered = []
    removed = {}
    for item in results:
        keys = _final_diversity_keys(item)
        failed = None
        for name, limit in limits.items():
            key = keys[name]
            if counts[name].get(key, 0) >= limit:
                failed = f"diversity_{name}"
                break
        if failed is None and FINAL_PAIR_A_FAMILY_GATE_ENABLED:
            family = keys.get("pair_a_family")
            family_cap = _pair_a_family_cap(family)
            if family_cap > 0 and pair_a_family_counts.get(family, 0) >= family_cap:
                failed = "diversity_pair_a_family"
        if failed is None and ts_comp_op_cap > 0:
            ts_comp_op = keys.get("ts_comp_op")
            if ts_comp_op not in (None, "none") and ts_comp_op_counts.get(ts_comp_op, 0) >= ts_comp_op_cap:
                failed = "diversity_ts_comp_op"
        item["final_diversity_gate"] = {
            "passed": failed is None,
            "reason": failed,
            "limits": limits,
            "pair_a_family_gate_enabled": FINAL_PAIR_A_FAMILY_GATE_ENABLED,
            "pair_a_family_max_share": FINAL_PAIR_A_FAMILY_MAX_SHARE,
            "pair_a_family_default_cap": FINAL_PAIR_A_FAMILY_DEFAULT_CAP,
            "pair_a_family_caps": dict(pair_a_family_caps),
            "ts_comp_op_max_share": FINAL_TS_COMP_OP_MAX_SHARE,
            "ts_comp_op_cap": ts_comp_op_cap,
        }
        if failed is not None:
            removed[failed] = removed.get(failed, 0) + 1
            continue
        filtered.append(item)
        for name, key in keys.items():
            counts[name][key] = counts[name].get(key, 0) + 1
        family = keys.get("pair_a_family")
        if FINAL_PAIR_A_FAMILY_GATE_ENABLED and family is not None:
            pair_a_family_counts[family] = pair_a_family_counts.get(family, 0) + 1
        ts_comp_op = keys.get("ts_comp_op")
        if ts_comp_op_cap > 0 and ts_comp_op not in (None, "none"):
            ts_comp_op_counts[ts_comp_op] = ts_comp_op_counts.get(ts_comp_op, 0) + 1
    return filtered, removed


def _json_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _counter_inc(counter, key, amount=1):
    text = str(key)
    counter[text] = int(counter.get(text, 0)) + int(amount)


def _nested_counter_inc(counter, key1, key2, amount=1):
    text1 = str(key1)
    text2 = str(key2)
    bucket = counter.setdefault(text1, {})
    bucket[text2] = int(bucket.get(text2, 0)) + int(amount)


def _candidate_field_names(decoded):
    fields = []
    for key in ("A", "B", "mask_field"):
        value = decoded.get(key)
        if value and value in INDICATOR_INDEX:
            fields.append(str(value))
    return tuple(dict.fromkeys(fields))


def _diagnostic_field_group(name):
    if name in XBINANCE_ORDER_FLOW_INDICATOR_NAMES:
        return "xbinance_order_flow"
    if name in ORDER_FLOW_INDICATOR_NAMES:
        return "bybit_order_flow"
    if name in FUNDAMENTAL_INDICATOR_NAMES:
        return "fundamental"
    if name in CROSS_PERIOD_INDICATOR_NAMES:
        return "advanced_period"
    return indicator_source_family(name)


def _candidate_stage(item, retained_ids):
    quality = item.get("final_quality_gate") or {}
    if quality and not bool(quality.get("passed")):
        return "rejected_quality"
    dedup = item.get("final_formula_dedup_gate") or {}
    if dedup and not bool(dedup.get("passed")):
        return "rejected_duplicate_formula"
    signature_dedup = item.get("final_signature_dedup_gate") or {}
    if signature_dedup and not bool(signature_dedup.get("passed")):
        return "rejected_duplicate_signature"
    signature_blacklist = item.get("final_signature_blacklist_gate") or {}
    if signature_blacklist and not bool(signature_blacklist.get("passed")):
        return "rejected_external_signature"
    diversity = item.get("final_diversity_gate") or {}
    if diversity and not bool(diversity.get("passed")):
        return "rejected_diversity"
    if id(item) in retained_ids:
        return "retained"
    return "truncated_or_unclassified"


def _candidate_metric_snapshot(item):
    obj = item.get("is_objectives") or {}
    screening = item.get("is_screening") or {}
    yearly = ((item.get("yearly_stability") or {}).get("summary") or {})
    shape = item.get("factor_shape") or {}
    return {
        "ls_netret": _json_float(obj.get("ls_netret")),
        "rankicir": _json_float(obj.get("rankicir")),
        "ls_net_sharpe": _json_float(obj.get("ls_net_sharpe")),
        "ls_1_minus_maxdd": _json_float(obj.get("ls_1-maxdd")),
        "turnover": _json_float(screening.get("ls_turnover")),
        "long_turnover": _json_float(screening.get("long_turnover")),
        "positive_rate": _json_float(screening.get("ls_positive_rate")),
        "coverage": _json_float(screening.get("coverage")),
        "min_yearly_ls_netret": _json_float(yearly.get("min_ls_netret")),
        "min_yearly_ls_net_sharpe": _json_float(yearly.get("min_ls_net_sharpe")),
        "min_yearly_rankicir": _json_float(yearly.get("min_rankicir")),
        "min_yearly_coverage": _json_float(yearly.get("min_coverage")),
        "max_yearly_turnover": _json_float(yearly.get("max_turnover")),
        "factor_finite_period_ratio": _json_float(shape.get("finite_period_ratio")),
        "factor_cell_finite_ratio": _json_float(shape.get("cell_finite_ratio")),
        "factor_nonzero_ratio": _json_float(shape.get("nonzero_ratio")),
        "factor_xs_std_median": _json_float(shape.get("xs_std_median")),
        "factor_unique_median": _json_float(shape.get("unique_median")),
        "pool_corr_max": _json_float(item.get("pool_corr_max")),
        "behavior_corr_max": _json_float(item.get("behavior_corr_max")),
        "composite": _json_float(item.get("composite")),
    }


def _build_candidate_diagnostics(all_candidates, retained_results):
    retained_ids = {id(item) for item in retained_results}
    by_stage = {}
    by_a_field = {}
    by_a_family = {}
    by_a_group = {}
    by_any_field = {}
    by_any_family = {}
    by_any_group = {}
    failed_by_a_family = {}
    failed_by_a_group = {}
    failed_by_any_family = {}
    failed_by_any_group = {}
    rows = []

    for idx, item in enumerate(all_candidates, start=1):
        decoded = item.get("decoded") if isinstance(item.get("decoded"), dict) else {}
        a_field = str(decoded.get("A") or "")
        a_family = indicator_source_family(a_field) if a_field else "unknown"
        a_group = _diagnostic_field_group(a_field) if a_field else "unknown"
        any_fields = _candidate_field_names(decoded)
        any_families = tuple(dict.fromkeys(indicator_source_family(name) for name in any_fields))
        any_groups = tuple(dict.fromkeys(_diagnostic_field_group(name) for name in any_fields))
        stage = _candidate_stage(item, retained_ids)
        quality_reasons = tuple((item.get("final_quality_gate") or {}).get("reasons") or ())
        diversity_reason = (item.get("final_diversity_gate") or {}).get("reason")
        dedup_reason = (item.get("final_formula_dedup_gate") or {}).get("reason")
        signature_dedup_reason = (
            item.get("final_signature_dedup_gate") or {}
        ).get("reason")
        signature_blacklist_reason = (
            item.get("final_signature_blacklist_gate") or {}
        ).get("reason")

        _counter_inc(by_stage, stage)
        _counter_inc(by_a_field, a_field or "unknown")
        _counter_inc(by_a_family, a_family)
        _counter_inc(by_a_group, a_group)
        for field in any_fields or ("unknown",):
            _counter_inc(by_any_field, field)
        for family in any_families or ("unknown",):
            _counter_inc(by_any_family, family)
        for group in any_groups or ("unknown",):
            _counter_inc(by_any_group, group)
        for reason in quality_reasons:
            _nested_counter_inc(failed_by_a_family, a_family, reason)
            _nested_counter_inc(failed_by_a_group, a_group, reason)
            for family in any_families or ("unknown",):
                _nested_counter_inc(failed_by_any_family, family, reason)
            for group in any_groups or ("unknown",):
                _nested_counter_inc(failed_by_any_group, group, reason)

        rows.append({
            "candidate_rank": int(idx),
            "result_rank": item.get("rank"),
            "stage": stage,
            "formula": item.get("formula"),
            "raw_formula": item.get("raw_formula"),
            "selection_source": item.get("selection_source"),
            "selection_search_rank": item.get("selection_search_rank"),
            "selection_diversity_pass": item.get("selection_diversity_pass"),
            "uses_composition": bool(item.get("selection_uses_composition")),
            "a_field": a_field or None,
            "b_field": decoded.get("B"),
            "mask_field": decoded.get("mask_field"),
            "operator": _decoded_operator(decoded),
            "source_family": a_family,
            "field_group": a_group,
            "any_fields": list(any_fields),
            "any_families": list(any_families),
            "any_groups": list(any_groups),
            "quality_reasons": list(quality_reasons),
            "dedup_reason": dedup_reason,
            "signature_dedup_reason": signature_dedup_reason,
            "signature_blacklist_reason": signature_blacklist_reason,
            "diversity_reason": diversity_reason,
            "metrics": _candidate_metric_snapshot(item),
        })

    return {
        "n_candidates": int(len(all_candidates)),
        "n_retained": int(len(retained_results)),
        "summary": {
            "by_stage": dict(sorted(by_stage.items())),
            "by_a_field": dict(sorted(by_a_field.items())),
            "by_a_family": dict(sorted(by_a_family.items())),
            "by_a_group": dict(sorted(by_a_group.items())),
            "by_any_field": dict(sorted(by_any_field.items())),
            "by_any_family": dict(sorted(by_any_family.items())),
            "by_any_group": dict(sorted(by_any_group.items())),
            "quality_failed_by_a_family": {
                key: dict(sorted(value.items()))
                for key, value in sorted(failed_by_a_family.items())
            },
            "quality_failed_by_a_group": {
                key: dict(sorted(value.items()))
                for key, value in sorted(failed_by_a_group.items())
            },
            "quality_failed_by_any_family": {
                key: dict(sorted(value.items()))
                for key, value in sorted(failed_by_any_family.items())
            },
            "quality_failed_by_any_group": {
                key: dict(sorted(value.items()))
                for key, value in sorted(failed_by_any_group.items())
            },
        },
        "candidates": rows,
    }


def _compute_population_novelty(phenotypes, valid_mask, archive_phenotypes=None):
    """Phenotype-space novelty. Considers both in-population peers and an
    optional persistent archive of past-published factors (cross-run memory).

    For each valid individual, novelty = 1 - max(|corr|) across both pools.
    """
    novelty = np.full(len(valid_mask), -1e6, dtype=np.float32)
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return novelty
    if phenotypes is None or phenotypes.shape[0] != len(valid_mask) or phenotypes.shape[1] == 0:
        novelty[valid_idx] = 1.0
        return novelty
    pheno = standardize_phenotypes(phenotypes[valid_idx])
    # pop-internal correlations
    corr_pop = np.abs(pheno @ pheno.T)
    np.fill_diagonal(corr_pop, 0.0)
    max_corr = (np.max(corr_pop, axis=1) if corr_pop.shape[0] > 1
                else np.zeros(1, dtype=np.float32))
    # archive correlations (if available)
    if (archive_phenotypes is not None
            and isinstance(archive_phenotypes, np.ndarray)
            and archive_phenotypes.size > 0
            and archive_phenotypes.shape[1] == pheno.shape[1]):
        corr_arch = np.abs(pheno @ archive_phenotypes.T)  # (n_valid, n_arch)
        max_arch = np.max(corr_arch, axis=1)
        max_corr = np.maximum(max_corr, max_arch)
    novelty[valid_idx] = 1.0 - np.clip(max_corr.astype(np.float32, copy=False), 0.0, 1.0)
    return novelty


def _screening_dict(aux_dict, idx):
    return {
        'factor_direction': int(-1 if float(aux_dict['ic_direction'][idx]) < 0.0 else 1),
        'ic_mean': float(aux_dict['ic_mean'][idx]),
        'rankicir': float(aux_dict['rankicir'][idx]),
        'positive_rate': float(aux_dict['positive_rate'][idx]),
        'coverage': float(aux_dict['coverage'][idx]),
        'n_valid': int(round(float(aux_dict['n_valid'][idx]))),
        'effective_bars': int(round(float(aux_dict['effective_bars'][idx]))),
        'ls_turnover': float(aux_dict['ls_turnover'][idx]),
        'long_turnover': float(aux_dict['long_turnover'][idx]),
        'ls_positive_rate': float(aux_dict['ls_positive_rate'][idx]),
    }


def _empty_aux_store(n_rows):
    return {
        'ic_direction': np.ones(n_rows, dtype=np.float32),
        'ic_mean': np.full(n_rows, np.nan, dtype=np.float32),
        'rankicir': np.full(n_rows, np.nan, dtype=np.float32),
        'ls_netret_ann': np.full(n_rows, np.nan, dtype=np.float32),
        'positive_rate': np.full(n_rows, np.nan, dtype=np.float32),
        'coverage': np.full(n_rows, np.nan, dtype=np.float32),
        'n_valid': np.zeros(n_rows, dtype=np.float32),
        'effective_bars': np.zeros(n_rows, dtype=np.float32),
        'ls_turnover': np.full(n_rows, np.nan, dtype=np.float32),
        'long_turnover': np.full(n_rows, np.nan, dtype=np.float32),
        'ls_positive_rate': np.full(n_rows, np.nan, dtype=np.float32),
    }


def _store_aux_slice(store, start, end, batch_aux):
    for key, values in batch_aux.items():
        store[key][start:end] = values


def _population_field_indices(pop):
    if pop.size == 0:
        return np.empty(0, dtype=np.int32)
    return np.unique(pop[:, (0, 1, 4)].astype(np.int32, copy=False).ravel())


def _project_population_fields(pop, field_indices):
    fields = np.asarray(field_indices, dtype=np.int32)
    projected = np.asarray(pop, dtype=np.int32).copy()
    for col in (0, 1, 4):
        projected[:, col] = np.searchsorted(fields, projected[:, col])
    return projected


def _field_locality_order(pop):
    if pop.shape[0] <= 1:
        return np.arange(pop.shape[0], dtype=np.intp)
    return np.lexsort((pop[:, 4], pop[:, 1], pop[:, 0])).astype(np.intp, copy=False)


def _new_perf_stats(gen_idx, pop_size, batch_size):
    return {
        'gen': int(gen_idx),
        'pop_size': int(pop_size),
        'batch_size': int(batch_size),
        'chunk_upload_s': 0.0,
        'chunk_cast_s': 0.0,
        'chunk_uploads': 0,
        'chunk_upload_bytes': 0,
        'chunk_field_uploads': 0,
        'chunk_field_full_uploads': 0,
        'eval_group_s': 0.0,
        'eval_mask_s': 0.0,
        'eval_task_build_s': 0.0,
        'eval_task_exec_s': 0.0,
        'eval_d2h_s': 0.0,
        'eval_h2d_s': 0.0,
        'eval_base_s': 0.0,
        'eval_comp_s': 0.0,
        'fit_total_s': 0.0,
        'fit_gpu_io_s': 0.0,
        'fit_gpu_postprocess_s': 0.0,
        'fit_gpu_ic_s': 0.0,
        'fit_gpu_ls_s': 0.0,
        'fit_gpu_dd_s': 0.0,
        'fit_gpu_total_s': 0.0,
        'yearly_fitness_s': 0.0,
        'diversity_s': 0.0,
        'penalties_s': 0.0,
        'eval_task_count': 0,
        'eval_group_count': 0,
        'eval_mask_count': 0,
        'group_count': 0,
    }


def _as_param_matrix(values):
    if values is None:
        return np.empty((0, N_PARAMS), dtype=int)
    arr = np.asarray(values)
    if arr.size == 0:
        return np.empty((0, N_PARAMS), dtype=int)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(int, copy=True)


def _param_operator(decoded):
    try:
        mode = int(decoded.get("mode") or 0)
    except Exception:
        mode = 0
    return str(decoded.get(f"mode{mode}_op") or decoded.get("mode2_op") or "")


def _slate_diversity_keys(params):
    decoded = decode_individual(params)
    mode = decoded.get("mode")
    op = _param_operator(decoded)
    a_field = decoded.get("A")
    b_field = decoded.get("B") if mode == 2 else None
    window = decoded.get("window")
    mask_field = decoded.get("mask_field")
    mask_rule = decoded.get("mask_rule")
    lag = decoded.get("B_shift_lag") if mode == 2 else None
    return {
        "operator": (mode, op),
        "a_field": a_field,
        "source_family": indicator_source_family(a_field),
        "ab_pair": (mode, op, a_field, b_field),
        "template": (
            mode,
            op,
            a_field,
            b_field,
            window,
            mask_field,
            mask_rule,
            lag,
            decoded.get("ts_comp_op"),
            decoded.get("cs_comp_op"),
        ),
    }


def _uses_composition(params):
    decoded = decode_individual(params)
    return (
        decoded.get("ts_comp_op") != "none"
        or decoded.get("cs_comp_op") != "none"
    )


def _has_order_flow_group(params):
    decoded = decode_individual(params)
    return any(
        _diagnostic_field_group(field) in {"xbinance_order_flow", "bybit_order_flow"}
        for field in _candidate_field_names(decoded)
    )


def _has_field_scout_group(params):
    decoded = decode_individual(params)
    return any(
        field in _FIELD_SCOUT_TEMPLATE_FIELDS
        for field in _candidate_field_names(decoded)
    )


def _within_diversity_limits(params, counts, limits):
    keys = _slate_diversity_keys(params)
    for name, limit in limits.items():
        if counts[name].get(keys[name], 0) >= int(limit):
            return False, name, keys
    return True, None, keys


def _build_candidate_slate(
    result,
    *,
    target_min,
    target_max,
    candidate_pool_size,
):
    pareto_x = _normalize_population(
        _as_param_matrix(getattr(result, "X", None))
    )
    n_pareto_raw = len(pareto_x)
    pareto_f = getattr(result, "F", None)
    pareto_composite = None
    if pareto_f is not None:
        try:
            pareto_f = np.asarray(pareto_f, dtype=np.float32)
            if pareto_f.ndim == 2 and pareto_f.shape[0] == n_pareto_raw:
                pareto_objectives = -pareto_f
                pareto_composite = _compute_search_scores(pareto_objectives)
        except Exception:
            pareto_composite = None

    final_pop = getattr(result, "pop", None)
    final_x = np.empty((0, N_PARAMS), dtype=int)
    final_f = None
    if final_pop is not None:
        try:
            final_x = _normalize_population(
                _as_param_matrix(final_pop.get("X"))
            )
            raw_f = final_pop.get("F")
            if raw_f is not None:
                final_f = np.asarray(raw_f, dtype=np.float32)
        except Exception:
            final_x = np.empty((0, N_PARAMS), dtype=int)
            final_f = None

    if final_f is None or final_f.ndim != 2 or final_f.shape[0] != len(final_x):
        final_x = np.empty((0, N_PARAMS), dtype=int)
        final_f = None

    target_pool = max(int(candidate_pool_size), int(target_max), int(target_min))
    comp_share = float(np.clip(CANDIDATE_SLATE_MAX_COMPOSITION_SHARE, 0.0, 1.0))
    composition_cap = int(np.floor(target_pool * comp_share))
    if comp_share > 0.0 and composition_cap < 1:
        composition_cap = 1
    selected = []
    selected_meta = []
    seen = set()
    diversity_counts = {name: {} for name in CANDIDATE_SLATE_DIVERSITY_LIMITS}
    diversity_skipped = {}
    diversity_pass_counts = {"strict": 0, "relaxed": 0, "fill": 0}
    composition_selected = 0
    composition_skipped = 0

    def add_candidate(
        params,
        *,
        source,
        search_rank=None,
        search_composite=None,
        diversity_pass="strict",
        limits=None,
        enforce_diversity=True,
    ):
        nonlocal composition_selected, composition_skipped
        if not _is_allowed_by_force_mode4(params):
            return False
        key = tuple(int(x) for x in params.tolist())
        if key in seen:
            return False
        uses_comp = _uses_composition(params)
        if uses_comp and composition_selected >= composition_cap:
            composition_skipped += 1
            return False
        keys = None
        if enforce_diversity:
            ok, failed_name, keys = _within_diversity_limits(
                params,
                diversity_counts,
                limits or CANDIDATE_SLATE_DIVERSITY_LIMITS,
            )
            if not ok:
                reason = f"slate_diversity_{failed_name}"
                diversity_skipped[reason] = diversity_skipped.get(reason, 0) + 1
                return False
        elif keys is None:
            keys = _slate_diversity_keys(params)
        seen.add(key)
        selected.append(params.copy())
        selected_meta.append(
            {
                "selection_source": source,
                "selection_search_rank": (
                    None if search_rank is None else int(search_rank)
                ),
                "selection_search_composite": (
                    None if search_composite is None else float(search_composite)
                ),
                "selection_diversity_pass": diversity_pass,
                "selection_uses_composition": bool(uses_comp),
            }
        )
        if uses_comp:
            composition_selected += 1
        for name, value in keys.items():
            if name in diversity_counts:
                diversity_counts[name][value] = diversity_counts[name].get(value, 0) + 1
        diversity_pass_counts[diversity_pass] = diversity_pass_counts.get(diversity_pass, 0) + 1
        return True

    candidate_items = []
    for pos, params in enumerate(pareto_x, start=1):
        comp = None
        if pareto_composite is not None and pos - 1 < len(pareto_composite):
            comp = pareto_composite[pos - 1]
        candidate_items.append(
            {
                "params": params,
                "source": "pareto",
                "search_rank": pos,
                "search_composite": comp,
            }
        )

    if len(final_x):
        if final_f.shape[1] == 1:
            search_composite = (-final_f[:, 0]).astype(np.float32, copy=False)
        else:
            search_objectives = -final_f
            search_composite = _compute_search_scores(search_objectives)
        ranked = np.argsort(search_composite)[::-1]
        for pos, idx in enumerate(ranked, start=1):
            candidate_items.append(
                {
                    "params": final_x[idx],
                    "source": "final_population",
                    "search_rank": pos,
                    "search_composite": search_composite[idx],
                }
            )

    state = None

    order_flow_reserve = min(int(CANDIDATE_SLATE_MIN_ORDER_FLOW), target_pool)
    order_flow_supplement_generated = 0
    if order_flow_reserve > 0:
        state = build_search_space_state()
        reserve_order_flow = _state_has_native_order_flow_fields(state) or (
            CANDIDATE_SLATE_RESERVE_XBINANCE_ORDER_FLOW
            and _state_has_xbinance_order_flow_fields(state)
        )
        if reserve_order_flow:
            rng = np.random.default_rng(10_000 + int(n_pareto_raw) + int(len(final_x)))
            n_supplement = order_flow_reserve * CANDIDATE_SLATE_ORDER_FLOW_SUPPLEMENT_MULT
            for pos in range(n_supplement):
                params = _generate_order_flow_individual(rng, state)
                candidate_items.append(
                    {
                        "params": params,
                        "source": "order_flow_supplement",
                        "search_rank": pos + 1,
                        "search_composite": -1e8 + float(pos) * 1e-3,
                    }
                )
            order_flow_supplement_generated = int(n_supplement)
        else:
            order_flow_reserve = 0

    field_scout_reserve = min(
        int(CANDIDATE_SLATE_MIN_FIELD_SCOUT),
        max(0, target_pool - order_flow_reserve),
    )
    field_scout_supplement_generated = 0
    if field_scout_reserve > 0:
        if state is None:
            state = build_search_space_state()
        if _state_has_field_scout_fields(state):
            rng = np.random.default_rng(20_000 + int(n_pareto_raw) + int(len(final_x)))
            n_supplement = field_scout_reserve * CANDIDATE_SLATE_FIELD_SCOUT_SUPPLEMENT_MULT
            for pos in range(n_supplement):
                params = _generate_field_scout_individual(rng, state)
                candidate_items.append(
                    {
                        "params": params,
                        "source": "field_scout_supplement",
                        "search_rank": pos + 1,
                        "search_composite": -1e8 + float(pos) * 1e-3,
                    }
                )
            field_scout_supplement_generated = int(n_supplement)
        else:
            field_scout_reserve = 0

    candidate_items.sort(
        key=lambda item: (
            float(item["search_composite"])
            if item.get("search_composite") is not None
            and np.isfinite(float(item["search_composite"]))
            else -1e9
        ),
        reverse=True,
    )

    order_flow_reserved = 0
    if order_flow_reserve > 0:
        for item in candidate_items:
            if len(selected) >= target_pool or order_flow_reserved >= order_flow_reserve:
                break
            if not _has_order_flow_group(item["params"]):
                continue
            if add_candidate(**item, diversity_pass="order_flow_reserve"):
                order_flow_reserved += 1

    field_scout_reserved = 0
    if field_scout_reserve > 0:
        for item in candidate_items:
            if len(selected) >= target_pool or field_scout_reserved >= field_scout_reserve:
                break
            if not _has_field_scout_group(item["params"]):
                continue
            if add_candidate(**item, diversity_pass="field_scout_reserve"):
                field_scout_reserved += 1

    for item in candidate_items:
        if len(selected) >= target_pool:
            break
        add_candidate(**item, diversity_pass="strict")

    relaxed_limits = {
        name: max(int(limit) + 1, int(limit) * 2)
        for name, limit in CANDIDATE_SLATE_DIVERSITY_LIMITS.items()
    }
    for item in candidate_items:
        if len(selected) >= target_pool:
            break
        add_candidate(
            **item,
            diversity_pass="relaxed",
            limits=relaxed_limits,
        )

    fill_limit_mult = float(np.clip(CANDIDATE_SLATE_FILL_LIMIT_MULT, 1.0, 20.0))
    fill_limits = {
        name: max(int(limit) + 2, int(np.ceil(int(limit) * fill_limit_mult)))
        for name, limit in CANDIDATE_SLATE_DIVERSITY_LIMITS.items()
    }
    for item in candidate_items:
        if len(selected) >= target_pool:
            break
        add_candidate(
            **item,
            diversity_pass="fill_soft",
            limits=fill_limits,
        )

    for item in candidate_items:
        if len(selected) >= min(target_pool, int(target_min)):
            break
        add_candidate(
            **item,
            diversity_pass="emergency_fill",
            enforce_diversity=False,
        )

    if selected:
        selected_x = np.vstack(selected)
    elif FORCE_MODE4_ONLY:
        selected_x = np.empty((0, N_PARAMS), dtype=int)
    else:
        selected_x = pareto_x

    return selected_x, selected_meta, {
        "n_pareto_raw": int(n_pareto_raw),
        "n_candidate_pool": int(len(selected_x)),
        "force_mode4_only": bool(FORCE_MODE4_ONLY),
        "target_min": int(target_min),
        "target_max": int(target_max),
        "candidate_pool_size": int(target_pool),
        "n_selected_from_pareto": int(sum(
            1 for item in selected_meta if item["selection_source"] == "pareto"
        )),
        "n_selected_from_final_population": int(sum(
            1 for item in selected_meta if item["selection_source"] == "final_population"
        )),
        "candidate_slate_diversity": {
            "strict_limits": CANDIDATE_SLATE_DIVERSITY_LIMITS,
            "relaxed_limits": relaxed_limits,
            "fill_limits": fill_limits,
            "selected_by_pass": diversity_pass_counts,
            "skipped": diversity_skipped,
            "composition_cap": int(composition_cap),
            "composition_max_share": float(comp_share),
            "composition_selected": int(composition_selected),
            "composition_skipped": int(composition_skipped),
            "order_flow_reserve": int(order_flow_reserve),
            "order_flow_reserved": int(order_flow_reserved),
            "order_flow_supplement_generated": int(order_flow_supplement_generated),
            "field_scout_reserve": int(field_scout_reserve),
            "field_scout_reserved": int(field_scout_reserved),
            "field_scout_supplement_generated": int(field_scout_supplement_generated),
            "final_counts": {
                name: {
                    str(key): int(value)
                    for key, value in sorted(bucket.items(), key=lambda kv: str(kv[0]))
                }
                for name, bucket in diversity_counts.items()
            },
        },
    }


def _behavior_overlap_band(novelty_value):
    if not _is_finite_number(novelty_value) or float(novelty_value) < -1e5:
        return "unknown"
    corr = 1.0 - float(novelty_value)
    if corr >= 0.98:
        return "corr_ge_0.98"
    if corr >= 0.96:
        return "corr_ge_0.96"
    if corr >= 0.94:
        return "corr_ge_0.94"
    return "corr_lt_0.94"


def _pre_gate_diversity_order(candidate_x, composite, behavior_novelty=None):
    n = len(candidate_x)
    if n == 0:
        return np.array([], dtype=int), [], {}
    base_score = np.asarray(composite, dtype=np.float32).copy()
    base_score = np.where(np.isfinite(base_score), base_score, -1e9)
    if behavior_novelty is not None:
        bn = np.asarray(behavior_novelty, dtype=np.float32)
        if bn.shape[0] == n:
            valid_bn = np.isfinite(bn) & (bn > -1e5)
            base_score[valid_bn] += np.float32(0.75) * bn[valid_bn]
    base_order = np.argsort(base_score)[::-1]
    if not PREGATE_DIVERSITY_ORDER_ENABLED:
        meta = [{"pass": "score_only", "score": float(base_score[i])} for i in range(n)]
        return base_order, meta, {"enabled": False, "selected_by_pass": {"score_only": int(n)}}

    strict_limits = {
        "operator": FINAL_DIVERSITY_MAX_PER_OPERATOR,
        "a_field": FINAL_DIVERSITY_MAX_PER_A_FIELD,
        "source_family": FINAL_DIVERSITY_MAX_PER_SOURCE_FAMILY,
        "ab_pair": FINAL_DIVERSITY_MAX_PER_AB_PAIR,
        "template": FINAL_DIVERSITY_MAX_PER_TEMPLATE,
    }
    relaxed_limits = {
        name: max(int(limit) + 1, int(limit) * 2)
        for name, limit in strict_limits.items()
    }
    passes = (
        ("strict", strict_limits, {"corr_ge_0.98": 2, "corr_ge_0.96": 8}),
        ("relaxed", relaxed_limits, {"corr_ge_0.98": 6, "corr_ge_0.96": 20}),
        ("fill", None, None),
    )
    selected = []
    seen = set()
    meta = [None] * n
    counts = {name: {} for name in strict_limits}
    behavior_counts = {}
    selected_by_pass = {}
    skipped_by_pass = {}

    def try_add(idx, pass_name, limits, behavior_limits):
        if int(idx) in seen:
            return False
        keys = _final_diversity_keys({"decoded": decode_individual(candidate_x[idx])})
        if limits is not None:
            for name, limit in limits.items():
                key = keys[name]
                if counts[name].get(key, 0) >= int(limit):
                    skipped_by_pass[pass_name] = skipped_by_pass.get(pass_name, 0) + 1
                    return False
        band = _behavior_overlap_band(
            behavior_novelty[idx] if behavior_novelty is not None and len(behavior_novelty) > idx else None
        )
        if behavior_limits is not None and band in behavior_limits:
            if behavior_counts.get(band, 0) >= int(behavior_limits[band]):
                skipped_by_pass[pass_name] = skipped_by_pass.get(pass_name, 0) + 1
                return False
        seen.add(int(idx))
        selected.append(int(idx))
        for name, key in keys.items():
            if name in counts:
                counts[name][key] = counts[name].get(key, 0) + 1
        behavior_counts[band] = behavior_counts.get(band, 0) + 1
        selected_by_pass[pass_name] = selected_by_pass.get(pass_name, 0) + 1
        meta[idx] = {
            "pass": pass_name,
            "score": float(base_score[idx]),
            "behavior_band": band,
        }
        return True

    for pass_name, limits, behavior_limits in passes:
        for idx in base_order:
            try_add(int(idx), pass_name, limits, behavior_limits)

    for idx in range(n):
        if meta[idx] is None:
            meta[idx] = {"pass": "unselected", "score": float(base_score[idx])}
    summary = {
        "enabled": True,
        "strict_limits": strict_limits,
        "relaxed_limits": relaxed_limits,
        "selected_by_pass": {k: int(v) for k, v in selected_by_pass.items()},
        "skipped_by_pass": {k: int(v) for k, v in skipped_by_pass.items()},
        "behavior_counts": {str(k): int(v) for k, v in behavior_counts.items()},
        "final_counts": {
            name: {
                str(key): int(value)
                for key, value in sorted(bucket.items(), key=lambda kv: str(kv[0]))
            }
            for name, bucket in counts.items()
        },
    }
    return np.asarray(selected, dtype=int), meta, summary


# ---------------------------------------------------------------------------
# In-evaluate penalties (affect NSGA-III non-dominated sorting directly)
# ---------------------------------------------------------------------------
# Composite weighting used only for ranking within niche/diversity penalties.
# Order: [rankicir, ls_net_sharpe, neg_turnover, novelty, ls_1-maxdd]
_NICHE_WEIGHTS = np.array([100.0, 5.0, 30.0, 20.0, 20.0], dtype=np.float32)


def _niche_cull_excess(indicator_a, composite, max_per, sentinel_value):
    """Shared: find overrepresented indicators, return lower-composite excess indices."""
    counts = np.bincount(indicator_a, minlength=N_INDICATORS)
    overrep = np.where(counts > max_per)[0]
    if not overrep.size:
        return np.array([], dtype=int), overrep
    excess_all = []
    for ind_a in overrep:
        indices = np.flatnonzero(indicator_a == ind_a)
        if len(indices) <= max_per:
            continue
        order = np.argsort(composite[indices])  # ascending = best first when lower=better
        excess_all.append(indices[order[max_per:]])
    if excess_all:
        return np.concatenate(excess_all), overrep
    return np.array([], dtype=int), overrep


def _ab_pair_key(a_idx, b_idx):
    a_idx = int(a_idx)
    b_idx = int(b_idx)
    if AB_PAIR_SYMMETRIC and a_idx > b_idx:
        return (b_idx, a_idx)
    return (a_idx, b_idx)


def _parse_crowded_ab_pairs(raw):
    pairs = []
    if not raw:
        return pairs
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            a_name, b_name = item.split(":", 1)
        elif "/" in item:
            a_name, b_name = item.split("/", 1)
        else:
            continue
        a_name = a_name.strip()
        b_name = b_name.strip()
        if a_name in INDICATOR_INDEX and b_name in INDICATOR_INDEX:
            pairs.append(_ab_pair_key(INDICATOR_INDEX[a_name], INDICATOR_INDEX[b_name]))
    return tuple(dict.fromkeys(pairs))


CROWDED_AB_PAIRS = _parse_crowded_ab_pairs(CROWDED_AB_PAIRS_RAW)
BANNED_AB_PAIRS = _parse_crowded_ab_pairs(BANNED_AB_PAIRS_RAW)


def _apply_factor_shape_penalty(is_objectives, factor_shapes):
    """Hard-sentinel degenerate factor values during NSGA search.

    Final quality gates already reject constant/all-zero factors before saving,
    but applying the same shape contract here prevents evolution from spending
    generations on tie-break artifacts.
    """
    if factor_shapes is None:
        return is_objectives, 0
    out = is_objectives
    if isinstance(factor_shapes, np.ndarray) and factor_shapes.dtype == bool:
        bad = np.asarray(factor_shapes, dtype=bool)
    else:
        bad = np.asarray([bool(_factor_shape_reasons(shape)) for shape in factor_shapes], dtype=bool)
    if bad.size == 0:
        return out, 0
    if bad.shape[0] != out.shape[0]:
        raise ValueError("factor_shapes must match objective row count")
    if bad.any():
        out[bad] = -1e6
    return out, int(bad.sum())


def _apply_banned_ab_pair_penalty(is_objectives, pop, banned_pairs=None):
    """Hard-ban configured A/B basins by sentineling their objectives."""
    pairs = BANNED_AB_PAIRS if banned_pairs is None else tuple(banned_pairs)
    if len(pop) == 0 or not pairs:
        return is_objectives, 0
    pair_set = set(_ab_pair_key(a, b) for a, b in pairs)
    banned = np.zeros(len(pop), dtype=bool)
    a_col = pop[:, 0].astype(int)
    b_col = pop[:, 1].astype(int)
    mode_col = pop[:, 6].astype(int)
    candidate_keys = np.array([_ab_pair_key(a, b) for a, b in zip(a_col, b_col)], dtype=np.int32)
    for a_idx, b_idx in pair_set:
        banned |= (
            (candidate_keys[:, 0] == a_idx)
            & (candidate_keys[:, 1] == b_idx)
            & (mode_col != 0)
        )
    if banned.any():
        is_objectives[banned] = -1e6
    return is_objectives, int(banned.sum())


def _apply_crowded_ab_pair_penalty(
    is_objectives,
    pop,
    crowded_pairs=None,
    max_share=CROWDED_AB_PAIR_MAX_SHARE,
    penalty=CROWDED_AB_PAIR_PENALTY,
):
    """Down-weight crowded exact A/B basins during NSGA search.

    This keeps the best few members of proven basins while preventing variants
    of the same A/B pair from consuming most Pareto slots. Only positive
    performance objectives are scaled, so bad candidates are not rescued.
    """
    pairs = CROWDED_AB_PAIRS if crowded_pairs is None else tuple(crowded_pairs)
    pop_size = int(len(pop))
    if pop_size == 0 or not pairs or max_share <= 0.0:
        return is_objectives, 0
    max_per_pair = max(1, int(np.floor(pop_size * float(max_share))))
    mult = float(np.clip(penalty, 0.0, 1.0))
    if mult >= 1.0:
        return is_objectives, 0

    out = is_objectives
    valid = out[:, 0] > -1e5
    composite = np.full(pop_size, -1e8, dtype=np.float32)
    composite[valid] = out[valid] @ _NICHE_WEIGHTS
    penalized = np.zeros(pop_size, dtype=bool)
    pair_set = set(_ab_pair_key(a, b) for a, b in pairs)
    a_col = pop[:, 0].astype(int)
    b_col = pop[:, 1].astype(int)
    mode_col = pop[:, 6].astype(int)
    candidate_keys = np.array([_ab_pair_key(a, b) for a, b in zip(a_col, b_col)], dtype=np.int32)

    for pair in pair_set:
        pair_mask = (
            (candidate_keys[:, 0] == pair[0])
            & (candidate_keys[:, 1] == pair[1])
            & (mode_col != 0)
        )
        indices = np.flatnonzero(pair_mask & valid)
        if indices.size <= max_per_pair:
            continue
        order = np.argsort(composite[indices])[::-1]
        excess = indices[order[max_per_pair:]]
        penalized[excess] = True

    if penalized.any():
        for col in (0, 1):
            mask = penalized & (out[:, col] > 0.0)
            out[mask, col] = out[mask, col] * mult
    return out, int(penalized.sum())


def _apply_indicator_cap_penalty(is_objectives, pop, cap_pct=0.15):
    """Sentinel excess offspring from overrepresented indicators (in-evaluate)."""
    pop_size = len(pop)
    max_per = max(int(pop_size * cap_pct), 5)

    valid = is_objectives[:, 0] > -1e5
    composite = np.full(pop_size, 1e8, dtype=np.float32)  # high = bad
    composite[valid] = -(is_objectives[valid] @ _NICHE_WEIGHTS)  # negate so lower=better

    excess, _ = _niche_cull_excess(pop[:, 0].astype(int), composite, max_per, -1e6)
    # Only cull those that are valid (don't re-sentinel already dead ones)
    excess = excess[valid[excess]]
    is_objectives[excess] = -1e6
    return is_objectives, int(excess.size)


def _apply_diversity_penalty(is_objectives, phenotypes, max_corr=0.95,
                             archive_phenotypes=None):
    """Sentinel near-duplicate factors (|phenotype_corr| > max_corr).

    Checks:
      1. pop-internal: duplicates of any higher-composite pop member
      2. archive: duplicates of any past-published factor (if archive provided)

    Both mechanisms send offender's is_objectives to -1e6 → dominated by NSGA-III.
    """
    pop = len(is_objectives)
    if phenotypes is None or pop < 2 or phenotypes.shape[1] == 0:
        return is_objectives, 0

    valid = is_objectives[:, 0] > -1e5
    composite = np.full(pop, -1e8, dtype=np.float32)
    composite[valid] = is_objectives[valid] @ _NICHE_WEIGHTS
    rank_order = np.argsort(composite)[::-1]

    pheno = standardize_phenotypes(phenotypes)

    # Archive taboo: any valid candidate whose max |corr| with archive > max_corr dies.
    n_arch_killed = 0
    if (archive_phenotypes is not None
            and isinstance(archive_phenotypes, np.ndarray)
            and archive_phenotypes.size > 0
            and archive_phenotypes.shape[1] == pheno.shape[1]):
        # compute max corr per pop row vs archive (only for currently valid)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size:
            max_corr_arr = np.abs(pheno[valid_idx] @ archive_phenotypes.T).max(axis=1)
            kill = valid_idx[max_corr_arr > max_corr]
            is_objectives[kill] = -1e6
            valid[kill] = False  # keep the rest of logic consistent
            n_arch_killed = int(kill.size)

    # Pop-internal: top-300 by composite, kill lower-ranked that mirror a higher one
    n_pop_killed = 0
    for rank_pos in range(1, min(pop, 300)):
        idx = rank_order[rank_pos]
        if not valid[idx]:
            continue
        for higher_pos in range(rank_pos):
            higher_idx = rank_order[higher_pos]
            if not valid[higher_idx]:
                continue
            if abs(float(np.dot(pheno[idx], pheno[higher_idx]))) > max_corr:
                is_objectives[idx] = -1e6
                valid[idx] = False
                n_pop_killed += 1
                break
    return is_objectives, n_arch_killed + n_pop_killed


# ---------------------------------------------------------------------------
# Niche-based fitness sharing (Goldberg 1987, Deb 1989)
#
# First-principles: NSGA-III IS fitness has natural basins (residual_std,
# kurt). Without pressure, pop collapses to one basin — seed31 showed 42/48
# pareto were residual_std. Fitness sharing divides each positive objective
# by a niche-density factor so crowded niches' mean fitness sinks, leaving
# sparse niches' strong individuals room to dominate. Pure formal diversity
# pressure — no hand-tuned priors, no behavior-specific rules.
#
# niche = (mode, op_in_mode, A_source_family). Source family is the same
# taxonomy used by slate/final diversity, so newly appended path/regime/
# positioning fields do not get lumped into one "crypto_native" bucket.
# ---------------------------------------------------------------------------
_A_FAMILY_CODE = {
    "price": 0,
    "price_state": 1,
    "trend": 2,
    "volatility": 3,
    "liquidity": 4,
    "flow": 5,
    "leverage": 6,
    "dislocation": 7,
    "path_shape": 8,
    "regime": 9,
    "other": 10,
}
_A_FAMILY_LOOKUP = np.asarray(
    [_A_FAMILY_CODE.get(indicator_source_family(name), _A_FAMILY_CODE["other"]) for name in INDICATOR_NAMES],
    dtype=np.int32,
)


def _niche_ids(pop):
    """Pack (mode, op_in_mode, A_family) into a single int32 niche id."""
    modes = pop[:, 6].astype(np.int32)
    op = np.where(modes == 0, pop[:, 7],
         np.where(modes == 1, pop[:, 8], pop[:, 13])).astype(np.int32)
    a = pop[:, 0].astype(np.int32)
    a_safe = np.clip(a, 0, len(_A_FAMILY_LOOKUP) - 1)
    a_family = _A_FAMILY_LOOKUP[a_safe]
    return modes * 10000 + op * 100 + a_family


def _apply_niche_sharing(is_objectives, pop, alpha=NICHE_SHARING_ALPHA):
    """Dilute positive obj[0]=rankicir and obj[1]=sharpe by log(niche density).

    share(n) = 1 / (1 + alpha * log1p(n - 1))
      n=1  → 1.00  (solo niche — no dilution)
      n=10 → 0.60
      n=100→ 0.35
      n=500→ 0.26
    So a niche with 500 members keeps only ~26% of its raw fitness mass,
    while a niche of 1 keeps 100% — NSGA-III is forced to allocate pareto
    slots to sparse niches once their rescaled fitness beats the crowded
    niche's best.

    Only positive raw fitness is diluted — negative values already lose
    in NSGA-III sorting, no need to rescue them. Sentinels (-1e6 set by
    cap/diversity penalties above) are skipped via the valid mask.
    """
    pop_size = len(pop)
    if alpha <= 0.0 or pop_size < 2:
        return is_objectives, np.zeros(0, dtype=np.int64)
    niches = _niche_ids(pop)
    _uniq, inverse, counts = np.unique(
        niches, return_inverse=True, return_counts=True)
    niche_count = counts[inverse].astype(np.float32)
    share = 1.0 / (1.0 + alpha * np.log1p(niche_count - 1.0))

    valid = is_objectives[:, 0] > -1e5
    for col in (0, 1):
        mask = valid & (is_objectives[:, col] > 0.0)
        if mask.any():
            is_objectives[mask, col] = is_objectives[mask, col] * share[mask]
    return is_objectives, counts


class CryptoFactorProblemV2(Problem):
    """pymoo Problem v2: cached indicators, exploration-driven alpha search."""

    def __init__(self, loader, backend='cpu', gpu_id=0,
                 eval_batch_size=EVAL_BATCH_SIZE,
                 perf_profile=PERF_PROFILE,
                 **fitness_kwargs):
        xl = np.array(PARAM_BOUNDS_LOWER, dtype=int)
        xu = np.array(PARAM_BOUNDS_UPPER, dtype=int)
        super().__init__(
            n_var=N_PARAMS, n_obj=SEARCH_N_OBJECTIVES, n_constr=0,
            xl=xl,
            xu=xu,
            vtype=int,
        )
        self.loader = loader
        self._indicator_field_indices = getattr(loader, 'indicator_field_indices', None)
        self.backend = backend
        self.gpu_id = gpu_id
        self.fitness_kwargs = fitness_kwargs
        self._lag = self.fitness_kwargs.get('lag', LAG_PERIODS)
        self._fitness_eval_kwargs = dict(self.fitness_kwargs)
        self._fitness_eval_kwargs.pop('lag', None)
        self.eval_batch_size = eval_batch_size
        self.perf_profile = bool(perf_profile)
        self._perf_history = []

        self.period_returns = loader.get_period_returns()
        self.period_tradable_mask = loader.get_period_tradable_mask()
        self.n_active_periods = loader.n_active_periods
        self.n_coins = loader.n_tradable_coins
        self.is_mask = loader.is_mask
        self._lagged_period_returns = build_lagged_returns(
            self.period_returns, lag=self._lag)
        self._fitness_context_is = prepare_fitness_context(
            self._lagged_period_returns, is_mask=self.is_mask,
            tradable_mask=self.period_tradable_mask)
        self._gpu_fitness = backend == 'gpu' and gpu_fitness_enabled()
        self._gpu_resident_factors = False
        if backend == 'gpu':
            if self._gpu_resident_factors:
                self._chunk_reuse_batches = 4
            else:
                default_reuse = max(
                    1,
                    int(GPU_GROUP_TARGET_INDIVIDUALS) // max(1, int(self.eval_batch_size)),
                )
                default_reuse = int(os.environ.get("GP_CHUNK_REUSE_BATCHES", str(default_reuse)))
                self._chunk_reuse_batches = max(1, default_reuse)
        else:
            self._chunk_reuse_batches = 1
        self._fitness_context_is_eval = self._fitness_context_is
        if self._gpu_fitness:
            set_backend(self.backend, self.gpu_id)
            self._fitness_context_is_eval = materialize_fitness_context(self._fitness_context_is)
        self._yearly_search_contexts = _build_yearly_fitness_contexts(self, scope="train")

        self._rankic_novelty_basis = None
        self._rankic_novelty_meta = {}
        if SEARCH_NOVELTY_OBJECTIVE_ENABLED and RANKIC_NOVELTY_OBJECTIVE_ENABLED:
            try:
                self._rankic_novelty_basis, self._rankic_novelty_meta = _load_rankic_novelty_basis(
                    RANKIC_NOVELTY_LIBRARY_PATH, self
                )
            except Exception as exc:
                print(
                    f"[RANKIC_NOVELTY] basis load failed ({exc}); "
                    f"falling back to existing novelty references",
                    flush=True,
                )
                self._rankic_novelty_basis = None
                self._rankic_novelty_meta = {}

        # Precompute and cache indicators
        self._chunk_sizes = loader.precompute_indicators()
        self._gen_count = 0

        # Reconstruct novelty reference features. Production uses only the
        # current active approved pool; the historical publish archive is a
        # legacy fallback for controlled ablations.
        self._archive_phenotypes = None
        self._archive_behavior_signatures = None
        self._external_behavior_archive = None
        self._external_behavior_archive_meta = {}
        self._behavior_context = None
        try:
            if EXTERNAL_BEHAVIOR_ARCHIVE_PATH:
                self._external_behavior_archive, self._external_behavior_archive_meta = _load_external_behavior_archive(
                    EXTERNAL_BEHAVIOR_ARCHIVE_PATH
                )
                print(
                    f"[EXTERNAL_ARCHIVE] loaded behavior signatures: "
                    f"n={self._external_behavior_archive.shape[0]} "
                    f"dim={self._external_behavior_archive.shape[1]} "
                    f"from {EXTERNAL_BEHAVIOR_ARCHIVE_PATH}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[EXTERNAL_ARCHIVE] load failed ({exc}); proceeding without external corr gate", flush=True)
            self._external_behavior_archive = None
            self._external_behavior_archive_meta = {}
        try:
            from .archive import (
                load_active_approved_params,
                load_archive_params,
                reconstruct_archive_features,
            )
            source_meta = {"source": "disabled"}
            if DISABLE_ARCHIVE_REPLAY or ARCHIVE_SOURCE in {"", "0", "false", "none", "off", "disabled"}:
                arch_params = []
                source_meta = {"source": "disabled"}
            elif ARCHIVE_SOURCE in {"active", "approved", "active_approved"}:
                arch_params, source_meta = load_active_approved_params()
            elif ARCHIVE_SOURCE in {"legacy", "legacy_json", "all_published"}:
                arch_params = load_archive_params()
                source_meta = {
                    "source": "legacy_json",
                    "valid_param_versions": len(arch_params),
                    "params_digest": None,
                }
            else:
                raise ValueError(f"unknown GP_ARCHIVE_SOURCE={ARCHIVE_SOURCE!r}")
            if arch_params:
                if BEHAVIOR_OVERLAP_SOFT_ENABLED or self._external_behavior_archive is not None:
                    self._behavior_context = build_behavior_context(
                        self.loader,
                        self.period_returns,
                        self.period_tradable_mask,
                    )
                skipped = int(source_meta.get("skipped_invalid_params", 0) or 0)
                print(
                    f"[ARCHIVE] loading {len(arch_params)} {source_meta.get('source')} "
                    f"factors for novelty pressure"
                    + (f" (skipped_invalid={skipped})" if skipped else "")
                    + "...",
                    flush=True,
                )
                t0 = time.time()
                self._archive_phenotypes, self._archive_behavior_signatures = reconstruct_archive_features(
                    arch_params,
                    self,
                    behavior_context=self._behavior_context,
                    source_meta=source_meta,
                )
                n = 0 if self._archive_phenotypes is None else self._archive_phenotypes.shape[0]
                nb = 0 if self._archive_behavior_signatures is None else self._archive_behavior_signatures.shape[0]
                print(
                    f'[ARCHIVE] reconstructed {n} phenotypes, {nb} behavior signatures '
                    f'in {time.time()-t0:.1f}s',
                    flush=True,
                )
            else:
                if self._external_behavior_archive is not None and self._behavior_context is None:
                    self._behavior_context = build_behavior_context(
                        self.loader,
                        self.period_returns,
                        self.period_tradable_mask,
                    )
                if DISABLE_ARCHIVE_REPLAY:
                    reason = "disabled by GP_DISABLE_ARCHIVE_REPLAY"
                elif source_meta.get("source") == "active_approved":
                    reason = "no active approved GP-param factors"
                else:
                    reason = "first run or no prior publish"
                print(f'[ARCHIVE] empty — {reason}', flush=True)
        except Exception as exc:
            print(f'[ARCHIVE] load failed ({exc}); proceeding without archive novelty', flush=True)
            self._archive_phenotypes = None
            self._archive_behavior_signatures = None
            if self._external_behavior_archive is not None and self._behavior_context is None:
                try:
                    self._behavior_context = build_behavior_context(
                        self.loader,
                        self.period_returns,
                        self.period_tradable_mask,
                    )
                except Exception as ctx_exc:
                    print(f'[EXTERNAL_ARCHIVE] behavior context failed ({ctx_exc}); disabling external corr gate', flush=True)
                    self._external_behavior_archive = None
                    self._behavior_context = None
            elif self._external_behavior_archive is None:
                self._behavior_context = None

        # Residual-IC basis (idea 2): representative production factor panels
        # used to measure incremental RankIC after removing the base's span.
        self._residual_basis_ctx = None
        if RESIDUAL_BASIS_ARCHIVE_PATH:
            try:
                self._residual_basis_ctx = _load_residual_basis_context(
                    RESIDUAL_BASIS_ARCHIVE_PATH, self
                )
            except Exception as exc:
                print(
                    f"[RESIDUAL_IC] basis load failed ({exc}); "
                    f"proceeding without residual-IC pressure",
                    flush=True,
                )
                self._residual_basis_ctx = None

    def _allocate_batch_factors(self, batch_n):
        output_backend = 'xp' if self._gpu_resident_factors else 'cpu'
        if output_backend == 'xp':
            return xp.empty((batch_n, self.n_active_periods, self.n_coins), dtype=xp.float32)
        return np.empty((batch_n, self.n_active_periods, self.n_coins), dtype=np.float32)

    def _build_batch_factor_group(self, batch_pops, perf_stats=None):
        if not batch_pops:
            return []

        output_backend = 'xp' if self._gpu_resident_factors else 'cpu'
        batch_factors_list = [
            self._allocate_batch_factors(batch_pop.shape[0]) for batch_pop in batch_pops
        ]
        group_pop_raw = np.vstack(batch_pops)
        field_indices = _population_field_indices(group_pop_raw)
        dependency_names = set()
        for op_idx in np.unique(group_pop_raw[:, 12].astype(np.intp, copy=False)):
            op_name = CS_COMP_OPS[int(op_idx)]
            dependency_names.update(CS_COMP_DEPENDENCY_FIELDS.get(op_name, ()))
        dependency_indices = [INDICATOR_INDEX[name] for name in dependency_names if name in INDICATOR_INDEX]
        if dependency_indices:
            field_indices = np.unique(np.concatenate([field_indices, np.asarray(dependency_indices, dtype=np.int32)]))
        batch_pops_projected = [
            _project_population_fields(batch_pop, field_indices) for batch_pop in batch_pops
        ]
        batch_lengths = [int(batch_pop.shape[0]) for batch_pop in batch_pops_projected]
        group_pop = np.vstack(batch_pops_projected)
        day_offset = 0
        for ci in range(len(self._chunk_sizes)):
            ind = self.loader.get_cached_chunk_fields(ci, field_indices, perf_stats=perf_stats)
            chunk_p = self._chunk_sizes[ci]
            group_chunk_factors = evaluate_population(
                group_pop, ind,
                output_backend=output_backend,
                indicator_field_indices=field_indices,
                perf_stats=perf_stats,
            )
            offset = 0
            for batch_factors, batch_len in zip(batch_factors_list, batch_lengths):
                next_offset = offset + batch_len
                batch_factors[:, day_offset:day_offset + chunk_p, :] = (
                    group_chunk_factors[offset:next_offset]
                )
                offset = next_offset
            day_offset += chunk_p
            del ind, group_chunk_factors
        if self.backend == 'gpu' and not self._gpu_resident_factors:
            free_gpu()
        return batch_factors_list

    def _build_batch_factors(self, batch_pop, perf_stats=None):
        return self._build_batch_factor_group([batch_pop], perf_stats=perf_stats)[0]

    def _compute_batch_yearly_min_objectives(self, batch_factors, ic_directions, perf_stats=None):
        n_rows = batch_factors.shape[0]
        min_netret = np.full(n_rows, np.inf, dtype=np.float32)
        min_sharpe = np.full(n_rows, np.inf, dtype=np.float32)
        min_positive_rate = np.full(n_rows, np.inf, dtype=np.float32)
        min_rankicir = np.full(n_rows, np.inf, dtype=np.float32)
        valid_all = np.ones(n_rows, dtype=bool)
        has_context = False
        for yearly_context in self._yearly_search_contexts:
            context = yearly_context.get("context")
            if context is None:
                valid_all &= False
                continue
            has_context = True
            t0 = time.perf_counter()
            yearly_obj, yearly_aux = compute_five_objectives_prepared(
                batch_factors,
                context,
                return_aux=True,
                ic_directions=ic_directions,
                **self._fitness_eval_kwargs,
            )
            _perf_add(perf_stats, 'yearly_fitness_s', time.perf_counter() - t0)
            yearly_netret = yearly_aux["ls_netret_ann"]
            yearly_positive = yearly_aux["ls_positive_rate"]
            valid = (
                (yearly_obj[:, 1] > -1e5)
                & np.isfinite(yearly_netret)
                & np.isfinite(yearly_positive)
            )
            valid_all &= valid
            min_netret = np.minimum(min_netret, yearly_netret)
            min_sharpe = np.minimum(min_sharpe, yearly_obj[:, 1])
            min_positive_rate = np.minimum(min_positive_rate, yearly_positive)
            min_rankicir = np.minimum(min_rankicir, yearly_obj[:, 0])
        if not has_context:
            valid_all &= False
        min_netret = np.where(valid_all, min_netret, -1e6).astype(np.float32, copy=False)
        min_sharpe = np.where(valid_all, min_sharpe, -1e6).astype(np.float32, copy=False)
        min_positive_rate = np.where(valid_all, min_positive_rate, np.nan).astype(np.float32, copy=False)
        min_rankicir = np.where(valid_all, min_rankicir, np.nan).astype(np.float32, copy=False)
        return min_netret, min_sharpe, min_positive_rate, min_rankicir

    def _iter_batch_factor_groups(self, pop, batch_size, perf_stats=None):
        pop_size = pop.shape[0]
        group_batches = max(1, int(self._chunk_reuse_batches))
        group_span = batch_size * group_batches
        eval_order = _field_locality_order(pop)
        for group_start in range(0, pop_size, group_span):
            group_end = min(group_start + group_span, pop_size)
            batch_indices = []
            batch_pops = []
            for batch_start in range(group_start, group_end, batch_size):
                batch_end = min(batch_start + batch_size, pop_size)
                idx = eval_order[batch_start:batch_end]
                batch_indices.append(idx)
                batch_pops.append(pop[idx])
            if perf_stats is not None:
                perf_stats['group_count'] = perf_stats.get('group_count', 0) + 1
            yield batch_indices, self._build_batch_factor_group(batch_pops, perf_stats=perf_stats)

    def _evaluate(self, X, out, *args, **kwargs):
        t0 = time.time()
        self._gen_count += 1
        pop = _normalize_population(
            np.asarray(X, dtype=np.int32).copy()
        )
        pop = _project_population_to_cached_fields(pop, self._indicator_field_indices)
        pop_size = pop.shape[0]

        set_backend(self.backend, self.gpu_id)

        batch_size = min(self.eval_batch_size, pop_size)
        is_objectives = np.full((pop_size, N_OBJECTIVES), -1e6, dtype=np.float32)
        yearly_min_netret = np.full(pop_size, -1e6, dtype=np.float32)
        yearly_min_sharpe = np.full(pop_size, -1e6, dtype=np.float32)
        is_positive_rate = np.full(pop_size, np.nan, dtype=np.float32)
        yearly_min_positive_rate = np.full(pop_size, np.nan, dtype=np.float32)
        yearly_min_rankicir = np.full(pop_size, np.nan, dtype=np.float32)
        resid_rankicir = np.full(pop_size, np.nan, dtype=np.float32)
        resid_ic_mean = np.full(pop_size, np.nan, dtype=np.float32)
        rankic_novelty = np.full(pop_size, np.nan, dtype=np.float32)
        factor_shape_bad = np.zeros(pop_size, dtype=bool)
        perf_stats = _new_perf_stats(self._gen_count, pop_size, batch_size) if self.perf_profile else None

        # Evaluate all individuals + collect phenotypes/behavior signatures for diversity.
        collect_external_behavior = (
            self._external_behavior_archive is not None
            and self._behavior_context is not None
        )
        _pheno_list = []
        _pheno_indices = []
        _external_behavior_list = []
        _external_behavior_indices = []
        for batch_indices, factor_batches in self._iter_batch_factor_groups(
                pop, batch_size, perf_stats=perf_stats):
            for batch_idx, batch_factors in zip(batch_indices, factor_batches):
                shape_t0 = time.perf_counter()
                batch_shapes = _compute_factor_shape_metrics(
                    batch_factors, tradable_mask=self.period_tradable_mask)
                factor_shape_bad[batch_idx] = [bool(_factor_shape_reasons(shape)) for shape in batch_shapes]
                _perf_add(perf_stats, 'factor_shape_s', time.perf_counter() - shape_t0)

                is_t0 = time.perf_counter()
                want_rankic_series = self._rankic_novelty_basis is not None
                fitness_result = compute_five_objectives_prepared(
                    batch_factors, self._fitness_context_is_eval,
                    perf_stats=perf_stats,
                    return_aux=True,
                    return_rankic_series=want_rankic_series,
                    **self._fitness_eval_kwargs)
                if want_rankic_series:
                    batch_is, batch_is_aux, batch_rankic_series = fitness_result
                    rankic_novelty[batch_idx] = compute_rankic_novelty(
                        batch_rankic_series,
                        self._rankic_novelty_basis,
                        min_valid=RANKIC_NOVELTY_MIN_VALID,
                        library_chunk=RANKIC_NOVELTY_LIBRARY_CHUNK,
                    )
                else:
                    batch_is, batch_is_aux = fitness_result
                is_objectives[batch_idx] = batch_is
                is_positive_rate[batch_idx] = batch_is_aux["ls_positive_rate"]
                _perf_add(perf_stats, 'is_fitness_s', time.perf_counter() - is_t0)

                y_netret, y_sharpe, y_positive_rate, y_rankicir = self._compute_batch_yearly_min_objectives(
                    batch_factors,
                    batch_is_aux["ic_direction"],
                    perf_stats=perf_stats,
                )
                yearly_min_netret[batch_idx] = y_netret
                yearly_min_sharpe[batch_idx] = y_sharpe
                yearly_min_positive_rate[batch_idx] = y_positive_rate
                yearly_min_rankicir[batch_idx] = y_rankicir

                if self._residual_basis_ctx is not None:
                    ri_t0 = time.perf_counter()
                    valid_rows = batch_is[:, 1] > -1e5
                    if np.any(valid_rows):
                        ri_rankicir, ri_mean = compute_residual_rankic(
                            batch_factors[valid_rows],
                            self._residual_basis_ctx,
                            ic_directions=batch_is_aux["ic_direction"][valid_rows],
                            periods_per_year=self._fitness_eval_kwargs.get(
                                "periods_per_year", PERIODS_PER_YEAR),
                            min_periods=RESIDUAL_IC_MIN_PERIODS,
                        )
                        local_idx = np.asarray(batch_idx, dtype=np.int64)
                        resid_rankicir[local_idx[valid_rows]] = ri_rankicir
                        resid_ic_mean[local_idx[valid_rows]] = ri_mean
                    _perf_add(perf_stats, 'residual_ic_s', time.perf_counter() - ri_t0)

                div_t0 = time.perf_counter()
                _pheno_list.append(sample_diversity_features(
                    batch_factors, tradable_mask=self.period_tradable_mask,
                    max_periods=DIVERSITY_MAX_PERIODS))
                _pheno_indices.append(batch_idx)
                if collect_external_behavior:
                    batch_behavior, _ = build_behavior_signatures(
                        batch_factors,
                        self._behavior_context,
                        ic_directions=batch_is_aux["ic_direction"],
                        standardize=False,
                    )
                    _external_behavior_list.append(batch_behavior)
                    _external_behavior_indices.append(batch_idx)
                _perf_add(perf_stats, 'diversity_s', time.perf_counter() - div_t0)
                del batch_factors

        if _pheno_list:
            phenotypes = np.empty((pop_size, _pheno_list[0].shape[1]), dtype=_pheno_list[0].dtype)
            for batch_idx, batch_pheno in zip(_pheno_indices, _pheno_list):
                phenotypes[batch_idx] = batch_pheno
        else:
            phenotypes = None
        if _external_behavior_list:
            external_behavior_signatures = np.empty(
                (pop_size, _external_behavior_list[0].shape[1]),
                dtype=_external_behavior_list[0].dtype,
            )
            for batch_idx, batch_behavior in zip(_external_behavior_indices, _external_behavior_list):
                external_behavior_signatures[batch_idx] = batch_behavior
        else:
            external_behavior_signatures = None
        del _pheno_list, _pheno_indices, _external_behavior_list, _external_behavior_indices

        # Hard dominance penalties (re-enabled 2026-04-21):
        #   - indicator cap: same A field > 15% of pop → excess sentineled
        #   - diversity: phenotype |rho|>0.95 with better-composite pop OR archive → sentineled
        # Niche-based fitness sharing (added 2026-04-22):
        #   - positive obj[0]/obj[1] scaled by 1/(1+alpha*log(niche_count))
        #   - niche = (mode, op_in_mode, A_family); forces NSGA-III to allocate
        #     pareto slots across niches instead of collapsing to residual_std basin
        is_objectives, n_shape = _apply_factor_shape_penalty(is_objectives, factor_shape_bad)
        is_objectives, n_cap = _apply_indicator_cap_penalty(is_objectives, pop)
        is_objectives, n_div = _apply_diversity_penalty(
            is_objectives, phenotypes,
            max_corr=0.95,
            archive_phenotypes=self._archive_phenotypes,
        )
        is_objectives, niche_counts = _apply_niche_sharing(is_objectives, pop)
        is_objectives, n_ab_ban = _apply_banned_ab_pair_penalty(is_objectives, pop)
        is_objectives, n_ab_pen = _apply_crowded_ab_pair_penalty(is_objectives, pop)

        search_objectives = _compose_search_objectives(
            is_objectives,
            yearly_min_netret,
            yearly_min_sharpe,
            is_positive_rate,
            yearly_min_positive_rate,
        )
        search_objectives, positive_rate_mult = _apply_positive_rate_soft_multiplier(
            search_objectives,
            is_positive_rate,
            yearly_min_positive_rate,
        )
        search_objectives, rankicir_mult = _apply_rankicir_soft_multiplier(
            search_objectives,
            is_objectives[:, 0],
            min_yearly_rankicir=yearly_min_rankicir,
        )
        pool_novelty = _compute_archive_novelty(
            phenotypes,
            search_objectives[:, 1] > -1e5,
            archive_phenotypes=self._archive_phenotypes,
        )
        search_objectives, novelty_mult = _apply_pool_novelty_soft_penalty(
            search_objectives,
            pool_novelty,
        )
        external_behavior_corr = _compute_external_behavior_corr(
            external_behavior_signatures,
            search_objectives[:, 1] > -1e5,
            self._external_behavior_archive,
        )
        search_objectives, external_corr_mult, n_external_hard = _apply_external_behavior_corr_penalty(
            search_objectives,
            external_behavior_corr,
        )
        search_objectives, resid_ic_mult = _apply_residual_ic_soft_penalty(
            search_objectives,
            resid_rankicir,
        )
        if SEARCH_NOVELTY_OBJECTIVE_ENABLED:
            # Fill the 6th objective after all penalties so hard-vetoed rows
            # keep the -1e6 sentinel on every column.
            search_objectives[:, 5] = _combine_search_novelty(
                pool_novelty,
                external_behavior_corr,
                search_objectives[:, 1] > -1e5,
                rankic_novelty=rankic_novelty,
            )
        valid = search_objectives[:, 1] > -1e5
        del phenotypes, external_behavior_signatures

        # Log
        nv = valid.sum()
        netret_med = np.median(search_objectives[valid, 0]) if nv > 0 else 0
        shp_max = np.max(search_objectives[valid, 1]) if nv > 0 else 0
        yret_med = np.median(search_objectives[valid, 2]) if nv > 0 else 0
        yshp_max = np.max(search_objectives[valid, 3]) if nv > 0 else 0
        turn_best = -np.max(search_objectives[valid, 4]) if nv > 0 else 0
        profile_log = f" MinYRet_med={yret_med:.2f} MinYShp_max={yshp_max:.2f} ShapeBad={n_shape}"
        if BANNED_AB_PAIRS:
            profile_log += f" ABBan={n_ab_ban}"
        if CROWDED_AB_PAIRS:
            profile_log += f" ABPen={n_ab_pen}"
        if POSITIVE_RATE_SOFT_ENABLED and nv > 0:
            is_pos_med = np.nanmedian(is_positive_rate[valid])
            y_pos_med = np.nanmedian(yearly_min_positive_rate[valid])
            pos_boost = int(np.sum(positive_rate_mult[valid] > 1.001))
            pos_pen = int(np.sum(positive_rate_mult[valid] < 0.999))
            profile_log += (
                f" PosRate_med={is_pos_med:.3f} MinYPos_med={y_pos_med:.3f} "
                f"PosBoost={pos_boost} PosPen={pos_pen}"
            )
        if POOL_NOVELTY_SOFT_ENABLED and nv > 0:
            pool_nov_med = np.median(pool_novelty[valid])
            pool_pen = int(np.sum(novelty_mult[valid] < 0.999))
            profile_log += f" PoolNov_med={pool_nov_med:.3f} PoolPen={pool_pen}"
        if self._external_behavior_archive is not None and nv > 0:
            ext_valid = valid & np.isfinite(external_behavior_corr)
            ext_corr_med = np.median(external_behavior_corr[ext_valid]) if np.any(ext_valid) else np.nan
            ext_corr_max = np.max(external_behavior_corr[ext_valid]) if np.any(ext_valid) else np.nan
            ext_pen = int(np.sum(external_corr_mult[ext_valid] < 0.999)) if np.any(ext_valid) else 0
            profile_log += (
                f" ExtCorr_med={ext_corr_med:.3f} ExtCorr_max={ext_corr_max:.3f} "
                f"ExtPen={ext_pen} ExtHard={n_external_hard}"
            )
        if SEARCH_NOVELTY_OBJECTIVE_ENABLED and nv > 0:
            nov_med = np.median(search_objectives[valid, 5])
            profile_log += f" SearchNov_med={nov_med:.3f}"
        if self._rankic_novelty_basis is not None and nv > 0:
            rankic_nov_valid = valid & np.isfinite(rankic_novelty)
            if np.any(rankic_nov_valid):
                profile_log += f" RankICNov_med={np.median(rankic_novelty[rankic_nov_valid]):.3f}"
        if self._residual_basis_ctx is not None and nv > 0:
            ri_valid = valid & np.isfinite(resid_rankicir)
            if np.any(ri_valid):
                ri_med = np.median(resid_rankicir[ri_valid])
                ri_pen = int(np.sum(resid_ic_mult[ri_valid] < 0.999))
                profile_log += f" ResidICIR_med={ri_med:.2f} ResidPen={ri_pen}"
        n_niches = int(niche_counts.size)
        max_niche = int(niche_counts.max()) if n_niches else 0

        print(f"  Gen {self._gen_count}: {time.time()-t0:.0f}s "
              f"valid={nv}/{pop_size} "
              f"NetRet_med={netret_med:.2f} NetShp_max={shp_max:.2f} "
              f"Turn_best={turn_best:.2f}{profile_log} "
              f"niches={n_niches} max_n={max_niche}", flush=True)
        if perf_stats is not None:
            perf_stats['gen_total_s'] = time.time() - t0
            self._perf_history.append(perf_stats)
            upload_gb = perf_stats['chunk_upload_bytes'] / 1e9
            print(
                f"  [PERF] up={perf_stats['chunk_upload_s']:.1f}s({upload_gb:.1f}GB) "
                f"fields={perf_stats.get('chunk_field_uploads', 0)}/"
                f"{perf_stats.get('chunk_field_full_uploads', 0)} "
                f"cast={perf_stats.get('chunk_cast_s', 0.0):.1f}s "
                f"base={perf_stats.get('eval_base_s', 0.0):.1f}s "
                f"comp={perf_stats.get('eval_comp_s', 0.0):.1f}s "
                f"is={perf_stats.get('is_fitness_s', 0.0):.1f}s "
                f"yearly={perf_stats.get('yearly_fitness_s', 0.0):.1f}s "
                f"div={perf_stats.get('diversity_s', 0.0):.1f}s "
                f"shape={perf_stats.get('factor_shape_s', 0.0):.1f}s "
                f"tasks={perf_stats.get('eval_task_count', 0)} "
                f"groups={perf_stats.get('group_count', 0)}",
                flush=True,
            )

        out["F"] = -search_objectives
        gc.collect()

    def perf_summary(self):
        if not self._perf_history:
            return None
        keys = (
            'gen_total_s', 'chunk_upload_s', 'eval_base_s', 'eval_comp_s',
            'chunk_cast_s',
            'is_fitness_s', 'yearly_fitness_s', 'diversity_s', 'factor_shape_s', 'penalties_s',
            'fit_gpu_postprocess_s', 'fit_gpu_ic_s', 'fit_gpu_ls_s', 'fit_gpu_dd_s',
        )
        summary = {'n_gens': len(self._perf_history)}
        for key in keys:
            vals = [row.get(key, 0.0) for row in self._perf_history]
            summary[key] = float(np.mean(vals))
        summary['chunk_upload_gb'] = float(np.mean([
            row.get('chunk_upload_bytes', 0) / 1e9 for row in self._perf_history
        ]))
        summary['chunk_field_ratio'] = float(np.mean([
            row.get('chunk_field_uploads', 0) / max(1, row.get('chunk_field_full_uploads', 0))
            for row in self._perf_history
        ]))
        summary['eval_task_count'] = float(np.mean([
            row.get('eval_task_count', 0) for row in self._perf_history
        ]))
        summary['group_count'] = float(np.mean([
            row.get('group_count', 0) for row in self._perf_history
        ]))
        return summary


def _elite_init_score(objectives, aux):
    rankicir = np.asarray(aux.get("rankicir"), dtype=np.float32)
    turnover = np.asarray(aux.get("ls_turnover"), dtype=np.float32)
    positive = np.asarray(aux.get("ls_positive_rate"), dtype=np.float32)
    dd_quality = objectives[:, 4] if objectives.shape[1] > 4 else 0.0
    return (
        1.5 * objectives[:, 0]
        + 3.0 * objectives[:, 1]
        + 0.20 * np.nan_to_num(rankicir, nan=-10.0)
        + 0.50 * np.nan_to_num(dd_quality, nan=-10.0)
        + 0.25 * np.nan_to_num(positive, nan=0.0)
        - 0.50 * np.nan_to_num(turnover, nan=10.0)
    ).astype(np.float32, copy=False)


def _elite_init_valid_mask(objectives, aux):
    rankicir = np.asarray(aux.get("rankicir"), dtype=np.float32)
    netret = np.asarray(aux.get("ls_netret_ann"), dtype=np.float32)
    turnover = np.asarray(aux.get("ls_turnover"), dtype=np.float32)
    coverage = np.asarray(aux.get("coverage"), dtype=np.float32)
    return (
        np.isfinite(rankicir)
        & np.isfinite(objectives[:, 1])
        & np.isfinite(netret)
        & (netret >= ELITE_INIT_MIN_NETRET)
        & (objectives[:, 1] >= ELITE_INIT_MIN_SHARPE)
        & (rankicir >= ELITE_INIT_MIN_RANKICIR)
        & np.isfinite(turnover)
        & (turnover <= ELITE_INIT_MAX_TURNOVER)
        & np.isfinite(coverage)
        & (coverage >= ELITE_INIT_MIN_COVERAGE)
    )


def _elite_init_candidates(pop, objectives, aux, scores, valid_mask):
    candidates = []
    for idx in np.flatnonzero(valid_mask):
        decoded = decode_individual(pop[int(idx)])
        candidates.append({
            "rank": int(idx) + 1,
            "composite": float(scores[int(idx)]),
            "_candidate_index": int(idx),
            "params": pop[int(idx)].tolist(),
            "decoded": decoded,
            "is_objectives": {
                "ls_netret": float(aux["ls_netret_ann"][int(idx)]),
                "rankicir": float(aux["rankicir"][int(idx)]),
                "ls_net_sharpe": float(objectives[int(idx), 1]),
                "neg_turnover": float(objectives[int(idx), 2]),
                "ls_1-maxdd": float(objectives[int(idx), 4]) if objectives.shape[1] > 4 else None,
            },
            "is_screening": {
                "rankicir": float(aux["rankicir"][int(idx)]),
                "coverage": float(aux["coverage"][int(idx)]),
                "ls_turnover": float(aux["ls_turnover"][int(idx)]),
                "ls_positive_rate": float(aux["ls_positive_rate"][int(idx)]),
            },
        })
    candidates.sort(key=lambda item: item.get("composite", -1e9), reverse=True)
    return candidates


def _build_elite_initial_population(problem, pop_size, seed, feedback_artifact=None):
    if not ELITE_INIT_ENABLED:
        return None, None
    rng = np.random.default_rng(int(seed) + 7919)
    scout_n = max(pop_size, int(round(pop_size * ELITE_INIT_MULT)))
    if ELITE_INIT_MAX_SCOUT > 0:
        scout_n = min(scout_n, ELITE_INIT_MAX_SCOUT)
    scout_X = _generate_exploration_pop(
        scout_n,
        rng=rng,
        feedback_artifact=feedback_artifact,
    )
    scout_X = _normalize_population(scout_X)
    scout_X = np.unique(scout_X, axis=0)
    scout_n = int(scout_X.shape[0])
    print(
        f"[ELITE_INIT] scouting {scout_n} formulas for {pop_size} initial slots "
        f"(mult={ELITE_INIT_MULT:.2f}, submit_prior={SUBMIT_PRIOR_SHARE:.2f})",
        flush=True,
    )

    objectives = np.full((scout_n, N_OBJECTIVES), -1e6, dtype=np.float32)
    aux_store = _empty_aux_store(scout_n)
    phenotype_chunks = []
    set_backend(problem.backend, problem.gpu_id)
    batch_size = min(max(1, int(problem.eval_batch_size)), scout_n)
    for batch_start in range(0, scout_n, batch_size):
        batch_end = min(batch_start + batch_size, scout_n)
        batch_pop = scout_X[batch_start:batch_end]
        batch_factors = problem._build_batch_factors(batch_pop)
        batch_obj, batch_aux = compute_five_objectives_prepared(
            batch_factors,
            problem._fitness_context_is_eval,
            return_aux=True,
            **problem._fitness_eval_kwargs,
        )
        objectives[batch_start:batch_end] = batch_obj
        _store_aux_slice(aux_store, batch_start, batch_end, batch_aux)
        phenotype_chunks.append(sample_diversity_features(
            batch_factors,
            tradable_mask=problem.period_tradable_mask,
            max_periods=DIVERSITY_MAX_PERIODS,
        ))
        del batch_factors
    if problem.backend == 'gpu':
        free_gpu()

    phenotypes = standardize_phenotypes(np.vstack(phenotype_chunks)) if phenotype_chunks else None
    scores = _elite_init_score(objectives, aux_store)
    valid_mask = _elite_init_valid_mask(objectives, aux_store)
    candidates = _elite_init_candidates(scout_X, objectives, aux_store, scores, valid_mask)
    selected, summary = select_low_corr_results(
        candidates,
        target_max=pop_size,
        phenotypes=phenotypes,
        behavior_signatures=None,
        pool_corr_hard=ELITE_INIT_POOL_CORR_HARD,
        spearman_corr_hard=1.0,
        behavior_corr_hard=1.0,
        external_corr_hard=1.0,
        field_family_max_share=ELITE_INIT_FIELD_FAMILY_MAX_SHARE,
        ab_pair_max=ELITE_INIT_AB_PAIR_MAX,
    )
    selected_indices = [int(item["_candidate_index"]) for item in selected]
    used = set(selected_indices)
    if len(selected_indices) < pop_size:
        for idx in np.argsort(-scores):
            idx = int(idx)
            if idx in used or not np.isfinite(scores[idx]) or objectives[idx, 1] <= -1e5:
                continue
            selected_indices.append(idx)
            used.add(idx)
            if len(selected_indices) >= pop_size:
                break
    if len(selected_indices) < pop_size:
        fill = _generate_exploration_pop(
            pop_size - len(selected_indices),
            rng=rng,
            feedback_artifact=feedback_artifact,
        )
        X0 = np.vstack([scout_X[selected_indices], _normalize_population(fill)])
    else:
        X0 = scout_X[selected_indices[:pop_size]]
    X0 = _normalize_population(X0)
    print(
        f"[ELITE_INIT] valid={int(valid_mask.sum())}/{scout_n} "
        f"low_corr_selected={len(selected_indices[:pop_size])} "
        f"median_sharpe={float(np.nanmedian(objectives[:, 1])):.3f} "
        f"best_sharpe={float(np.nanmax(objectives[:, 1])):.3f} "
        f"rejected={summary.get('rejected', {})}",
        flush=True,
    )
    meta = {
        "enabled": True,
        "scout_n": scout_n,
        "valid": int(valid_mask.sum()),
        "selected": int(min(len(selected_indices), pop_size)),
        "min_ls_netret": ELITE_INIT_MIN_NETRET,
        "min_sharpe": ELITE_INIT_MIN_SHARPE,
        "min_rankicir": ELITE_INIT_MIN_RANKICIR,
        "max_turnover": ELITE_INIT_MAX_TURNOVER,
        "min_coverage": ELITE_INIT_MIN_COVERAGE,
        "pool_corr_hard": ELITE_INIT_POOL_CORR_HARD,
        "field_family_max_share": ELITE_INIT_FIELD_FAMILY_MAX_SHARE,
        "ab_pair_max": ELITE_INIT_AB_PAIR_MAX,
        "submit_prior_share": SUBMIT_PRIOR_SHARE,
        "low_corr_summary": summary,
    }
    return X0, meta


def _load_seed_population(
    seed_files,
    pop_size,
    feedback_artifact=None,
):
    """Load Pareto individuals from prior runs (v1 or v2) as seed population.

    Handles both v1 (10-param) and v2 (13-param) formats.
    """
    seeds = []
    for path in seed_files:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            results = json.load(f)
        for r in results:
            params = np.array(r['params'], dtype=int)
            if len(params) == 10:
                # v1 format: pad with [0, 0, 0] (none, 1, none) for composition
                params = np.concatenate([params, [0, 0, 0]])
            params = _normalize_population(params.reshape(1, -1))[0]
            seeds.append(params)

    if not seeds:
        return None

    seed_X = np.unique(np.vstack(seeds), axis=0)
    n_seed = len(seed_X)
    if n_seed >= pop_size:
        return seed_X[:pop_size]

    # Fill remaining with exploration-biased random
    rng = np.random.default_rng(123)
    n_fill = pop_size - n_seed
    random_X = _generate_exploration_pop(
        n_fill,
        rng,
        feedback_artifact=feedback_artifact,
    )
    random_X = _normalize_population(random_X)

    X0 = np.vstack([seed_X, random_X])
    print(f"[SEED] {n_seed} Pareto individuals from prior runs + {n_fill} random", flush=True)
    return X0


def run_evolution(config=None):
    cfg = config or {}

    h5_path = cfg.get('h5_path', DEFAULT_H5_PATH)
    backend = cfg.get('backend', 'cpu')
    gpu_id = cfg.get('gpu_id', 0)
    minutes_per_period = cfg.get('minutes_per_period', MINUTES_PER_PERIOD)
    chunk_periods = cfg.get('chunk_periods', CHUNK_PERIODS)
    cache_dtype = cfg.get('cache_dtype', CACHE_DTYPE)
    start_date = cfg.get('start_date', None)
    end_date = cfg.get('end_date', None)
    train_end_date = cfg.get('train_end_date', None)
    tradable_mask_overlay_path = cfg.get('tradable_mask_overlay_path', None)
    fundamental_panel_path = cfg.get('fundamental_panel_path', None)
    pop_size = cfg.get('population_size', POPULATION_SIZE)
    n_gen = cfg.get('n_generations', N_GENERATIONS)
    lag = cfg.get('lag_periods', LAG_PERIODS)
    top_frac = cfg.get('top_quantile', TOP_QUANTILE)
    output_dir = cfg.get('output_dir', 'crypto_parametric_v2_output')
    seed = cfg.get('seed', 42)
    seed_from = cfg.get('seed_from', None)
    eval_batch_size = cfg.get('eval_batch_size', EVAL_BATCH_SIZE)
    perf_profile = cfg.get('perf_profile', PERF_PROFILE)
    result_target_min = int(cfg.get('result_target_min', RESULT_TARGET_MIN))
    result_target_max = int(cfg.get('result_target_max', RESULT_TARGET_MAX))
    candidate_pool_size = int(cfg.get('candidate_pool_size', FINAL_CANDIDATE_POOL_SIZE))
    family_feedback_path = cfg.get('family_feedback_path')
    if result_target_min <= 0:
        raise ValueError('result_target_min must be positive')
    if result_target_max < result_target_min:
        raise ValueError('result_target_max must be >= result_target_min')
    if candidate_pool_size < result_target_max:
        candidate_pool_size = result_target_max
    os.makedirs(output_dir, exist_ok=True)
    feedback_artifact = load_family_feedback_artifact(family_feedback_path)
    feedback_summary = None
    if feedback_artifact is not None:
        feedback_summary = {
            'artifact_path': feedback_artifact.get('artifact_path'),
            'generated_at': feedback_artifact.get('generated_at'),
            'candidate_count': int(feedback_artifact.get('candidate_count', 0)),
            'top_family': ((feedback_artifact.get('top_families') or [{}])[0]).get('value'),
        }

    # --- apply runtime search-space overrides from config JSON ----------------
    _OVERRIDE_KEYS = {
        'single_share', 'compound_share', 'intraday_share',
        'structural_explore_share', 'path_shape_share', 'proven_relation_share',
        'order_flow_share', 'adjacent_explore_share', 'primary_share',
        'composition_share', 'composition_cs_share',
        'ts_comp_ops', 'cs_comp_ops',
        'mode1_ops_primary', 'mode1_ops_explore',
        'mode2_ops_primary', 'mode2_ops_explore',
        'mode3_ops_primary', 'mode3_ops_explore',
        'mode4_ops_primary', 'mode4_ops_explore',
        'single_fields_primary', 'single_fields_explore',
        'pair_a_fields_primary', 'pair_a_fields_explore',
        'pair_b_fields_primary', 'pair_b_fields_explore',
    }
    overrides = {k: v for k, v in cfg.items() if k in _OVERRIDE_KEYS and v is not None}
    if overrides:
        set_search_space_overrides(overrides)
        print(f"[CONFIG] Search-space overrides applied: {overrides}", flush=True)
    # --------------------------------------------------------------------------

    search_space_state = build_search_space_state()
    indicator_field_indices = search_space_indicator_indices()
    sampling_lanes = {
        "primary": float(search_space_state.get("primary_share", 0.0)),
        "adjacent_explore": float(search_space_state.get("adjacent_explore_share", 0.0)),
        "structural_explore": float(search_space_state.get("structural_explore_share", 0.0)),
        "path_shape": float(search_space_state.get("path_shape_share", 0.0)),
        "proven_relation": float(search_space_state.get("proven_relation_share", 0.0)),
        "submit_prior": float(SUBMIT_PRIOR_SHARE),
        "order_flow": float(search_space_state.get("order_flow_share", 0.0)),
        "structural_mode3_share": float(search_space_state.get("structural_mode3_share", 0.0)),
    }

    loader = CryptoDataLoader(
        h5_path=h5_path,
        minutes_per_period=minutes_per_period,
        chunk_periods=chunk_periods,
        start_date=start_date,
        end_date=end_date,
        train_end_date=train_end_date,
        backend=backend,
        gpu_id=gpu_id,
        cache_dtype=cache_dtype,
        tradable_mask_overlay_path=tradable_mask_overlay_path,
        fundamental_panel_path=fundamental_panel_path,
        indicator_field_indices=indicator_field_indices,
    )
    loader.print_info()

    fitness_kwargs = dict(lag=lag, top_frac=top_frac,
                          periods_per_year=cfg.get('periods_per_year', PERIODS_PER_YEAR),
                          trading_cost=cfg.get('trading_cost', 0.0005))

    problem = CryptoFactorProblemV2(
        loader, backend=backend, gpu_id=gpu_id,
        eval_batch_size=eval_batch_size,
        perf_profile=perf_profile,
        **fitness_kwargs)

    # Build initial population
    X0 = None
    elite_init_meta = {"enabled": False}
    if seed_from:
        X0 = _load_seed_population(
            seed_from,
            pop_size,
            feedback_artifact=feedback_artifact,
        )
    elif ELITE_INIT_ENABLED:
        X0, elite_init_meta = _build_elite_initial_population(
            problem,
            pop_size,
            seed,
            feedback_artifact=feedback_artifact,
        )
        if X0 is None:
            elite_init_meta = {"enabled": False, "reason": "elite_init_returned_none"}
    sampling = X0 if X0 is not None else ExplorationSampling(
        feedback_artifact=feedback_artifact,
    )
    # das-dennis with 7 partitions gives C(11,4)=330 dirs for 5 objectives;
    # for 6 objectives 7 partitions would give C(12,5)=792 > pop_size, so use
    # 5 partitions (C(10,5)=252 dirs) to keep niching discriminative.
    _ref_partitions = 7 if SEARCH_N_OBJECTIVES <= 5 else 5
    ref_dirs = get_reference_directions(
        "das-dennis", SEARCH_N_OBJECTIVES, n_partitions=_ref_partitions)
    # DiversityMutation wraps PM and additionally re-samples the discrete
    # op/mode columns with non-trivial probability, so offspring can actually
    # cross op/mode boundaries rather than being trapped in the residual_std
    # basin that continuous-round mutation can never exit.
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        sampling=sampling,
        crossover=HybridCrossover(prob=0.9, sbx_eta=3.0),
        mutation=DiversityMutation(prob=0.15, eta=3.0, vtype=float),
    )

    seed_tag = f", seeded from {len(seed_from)} runs" if seed_from else ""
    print(f"[EVOLVE] Pop={pop_size}, Gen={n_gen}, Backend={backend}, "
          f"EvalBatch={eval_batch_size}, CacheDtype={cache_dtype}, "
          f"SearchProfile={DEFAULT_SEARCH_SPACE_PROFILE}, "
          f"SearchObjective={SEARCH_OBJECTIVE_PROFILE}, "
          f"GPUFitness={'on' if problem._gpu_fitness else 'off'}{seed_tag}", flush=True)
    print(
        f"[EVOLVE] Search objectives: "
        f"{', '.join(SEARCH_OBJECTIVE_NAMES)}",
        flush=True,
    )
    print(
        "[EVOLVE] Annual objective windows: "
        + ", ".join(
            f"{item['label']}({item['n_periods']}p)"
            for item in problem._yearly_search_contexts
        ),
        flush=True,
    )
    print(f"[EVOLVE] ChunkReuseBatches={problem._chunk_reuse_batches}, "
          f"PerfProfile={'on' if problem.perf_profile else 'off'}", flush=True)
    print(f"[EVOLVE] Search space: {N_PARAMS} params "
          f"(10 base + 3 composition + mode3)", flush=True)
    slate_names = tuple(str(x) for x in search_space_state.get("field_slate_names", ()))
    print(
        f"[EVOLVE] Field slate: {len(indicator_field_indices)}/{N_INDICATORS} fields "
        f"enabled={search_space_state.get('field_slate_enabled')} "
        f"seed={search_space_state.get('field_slate_seed')} "
        f"target={search_space_state.get('field_slate_target_size')} "
        f"advanced={search_space_state.get('advanced_period_enabled')} "
        f"fields={list(slate_names)}",
        flush=True,
    )
    print(
        f"[EVOLVE] Sampling lanes: primary={sampling_lanes['primary']:.2f}, "
        f"adjacent={sampling_lanes['adjacent_explore']:.2f}, "
        f"structural={sampling_lanes['structural_explore']:.2f} "
        f"(mode3={sampling_lanes['structural_mode3_share']:.2f}), "
        f"order_flow={sampling_lanes['order_flow']:.2f}, "
        f"proven_relation={sampling_lanes['proven_relation']:.2f}, "
        f"submit_prior={sampling_lanes['submit_prior']:.2f}, "
        f"path_shape={sampling_lanes['path_shape']:.2f}",
        flush=True,
    )
    print(f"[EVOLVE] Output slate target: {result_target_min}-{result_target_max} "
          f"(candidate_pool={candidate_pool_size})", flush=True)
    if feedback_summary is not None:
        print(
            f"[FEEDBACK] path={feedback_summary['artifact_path']}  "
            f"candidates={feedback_summary['candidate_count']}  "
            f"top_family={feedback_summary.get('top_family')}",
            flush=True,
        )
    t0 = time.time()

    result = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                       seed=seed, verbose=True, save_history=False)

    elapsed = time.time() - t0
    print(f"[DONE] {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)
    perf_summary = problem.perf_summary()
    if perf_summary is not None:
        print(
            f"[PERF] avg/gen={perf_summary['gen_total_s']:.1f}s "
            f"upload={perf_summary['chunk_upload_s']:.1f}s({perf_summary['chunk_upload_gb']:.1f}GB) "
            f"field_ratio={perf_summary.get('chunk_field_ratio', 1.0):.2f} "
            f"cast={perf_summary['chunk_cast_s']:.1f}s "
            f"base={perf_summary['eval_base_s']:.1f}s comp={perf_summary['eval_comp_s']:.1f}s "
            f"is={perf_summary['is_fitness_s']:.1f}s "
            f"yearly={perf_summary['yearly_fitness_s']:.1f}s "
            f"div={perf_summary['diversity_s']:.1f}s shape={perf_summary.get('factor_shape_s', 0.0):.1f}s pen={perf_summary['penalties_s']:.1f}s "
            f"fit_post={perf_summary['fit_gpu_postprocess_s']:.1f}s "
            f"fit_ic={perf_summary['fit_gpu_ic_s']:.1f}s "
            f"fit_ls={perf_summary['fit_gpu_ls_s']:.1f}s "
            f"fit_dd={perf_summary['fit_gpu_dd_s']:.1f}s "
            f"tasks={perf_summary['eval_task_count']:.0f} "
            f"groups={perf_summary['group_count']:.1f}",
            flush=True,
        )

    candidate_X, selection_meta, slate_meta = _build_candidate_slate(
        result,
        target_min=result_target_min,
        target_max=result_target_max,
        candidate_pool_size=candidate_pool_size,
    )
    n_pareto = slate_meta['n_pareto_raw']
    n_candidates = len(candidate_X)
    print(f"[RESULT] Pareto front: {n_pareto} factors", flush=True)
    print(f"[RESULT] Candidate slate: {n_candidates} factors "
          f"({slate_meta['n_selected_from_pareto']} pareto + "
          f"{slate_meta['n_selected_from_final_population']} near-front)", flush=True)
    slate_div = slate_meta.get("candidate_slate_diversity", {})
    print(
        f"[SLATE] composition selected={slate_div.get('composition_selected')} "
        f"cap={slate_div.get('composition_cap')} "
        f"skipped={slate_div.get('composition_skipped')} "
        f"max_share={slate_div.get('composition_max_share')}",
        flush=True,
    )

    print(f"[FINAL] Evaluating {n_candidates} candidate factors + saving DataFrames...", flush=True)
    set_backend(backend, gpu_id)

    pareto_F_is = np.full((n_candidates, N_OBJECTIVES), -1e6, dtype=np.float32)
    pareto_aux_is = _empty_aux_store(n_candidates)
    resid_rankicir_final = np.full(n_candidates, np.nan, dtype=np.float32)
    resid_ic_mean_final = np.full(n_candidates, np.nan, dtype=np.float32)
    rankic_novelty_final = np.full(n_candidates, np.nan, dtype=np.float32)
    yearly_contexts = _build_yearly_fitness_contexts(problem)
    yearly_store = _empty_yearly_metric_store(n_candidates, yearly_contexts)

    period_tradable_mask = loader.get_period_tradable_mask()
    _pheno_chunks = []
    _factor_chunks: list[np.ndarray] = []
    _factor_shape_metrics = []
    _behavior_chunks = []
    _behavior_feedback = []
    behavior_context = getattr(problem, "_behavior_context", None)
    archive_behavior = getattr(problem, "_archive_behavior_signatures", None)
    external_behavior_archive = getattr(problem, "_external_behavior_archive", None)
    final_behavior_corr_enabled = FINAL_BEHAVIOR_CORR_HARD < 1.0
    if behavior_context is None and final_behavior_corr_enabled:
        try:
            behavior_context = build_behavior_context(
                problem.loader,
                problem.period_returns,
                problem.period_tradable_mask,
            )
        except Exception as exc:
            print(f"[LOW_CORR] behavior context unavailable ({exc}); selected behavior corr disabled", flush=True)
            behavior_context = None
    use_behavior_overlap = (
        behavior_context is not None
        and (
            final_behavior_corr_enabled
            or (
                BEHAVIOR_OVERLAP_SOFT_ENABLED
                and isinstance(archive_behavior, np.ndarray)
                and archive_behavior.size > 0
            )
            or (
                isinstance(external_behavior_archive, np.ndarray)
                and external_behavior_archive.size > 0
            )
        )
    )
    print(
        "[FINAL] Yearly stability gate: "
        + ", ".join(
            f"{item['label']}({item['n_periods']}p)"
            for item in yearly_contexts
        ),
        flush=True,
    )

    for batch_start in range(0, n_candidates, problem.eval_batch_size):
        batch_end = min(batch_start + problem.eval_batch_size, n_candidates)
        batch_pop = candidate_X[batch_start:batch_end]
        batch_factors = problem._build_batch_factors(batch_pop)

        want_rankic_series = getattr(problem, "_rankic_novelty_basis", None) is not None
        fitness_result = compute_five_objectives_prepared(
            batch_factors, problem._fitness_context_is_eval,
            return_aux=True,
            return_rankic_series=want_rankic_series,
            **problem._fitness_eval_kwargs)
        if want_rankic_series:
            batch_is, batch_is_aux, batch_rankic_series = fitness_result
            rankic_novelty_final[batch_start:batch_end] = compute_rankic_novelty(
                batch_rankic_series,
                problem._rankic_novelty_basis,
                min_valid=RANKIC_NOVELTY_MIN_VALID,
                library_chunk=RANKIC_NOVELTY_LIBRARY_CHUNK,
            )
        else:
            batch_is, batch_is_aux = fitness_result
        pareto_F_is[batch_start:batch_end] = batch_is
        _store_aux_slice(pareto_aux_is, batch_start, batch_end, batch_is_aux)

        # Calendar-year stability is evaluated with the full-sample IC
        # direction fixed. Re-resolving direction per year would hide sign
        # instability and leak selection freedom into the gate.
        batch_ic_direction = batch_is_aux["ic_direction"]
        for yearly_context in yearly_contexts:
            label = yearly_context["label"]
            if yearly_context["context"] is None:
                continue
            yearly_obj, yearly_aux = compute_five_objectives_prepared(
                batch_factors,
                yearly_context["context"],
                return_aux=True,
                ic_directions=batch_ic_direction,
                **problem._fitness_eval_kwargs,
            )
            yearly_store[label]["objective"][batch_start:batch_end] = yearly_obj
            _store_aux_slice(
                yearly_store[label]["aux"],
                batch_start,
                batch_end,
                yearly_aux,
            )

        if problem._residual_basis_ctx is not None:
            valid_rows = batch_is[:, 1] > -1e5
            if np.any(valid_rows):
                ri_rankicir, ri_mean = compute_residual_rankic(
                    batch_factors[valid_rows],
                    problem._residual_basis_ctx,
                    ic_directions=batch_ic_direction[valid_rows],
                    periods_per_year=problem._fitness_eval_kwargs.get(
                        "periods_per_year", PERIODS_PER_YEAR),
                    min_periods=RESIDUAL_IC_MIN_PERIODS,
                )
                rows = np.arange(batch_start, batch_end, dtype=np.int64)
                resid_rankicir_final[rows[valid_rows]] = ri_rankicir
                resid_ic_mean_final[rows[valid_rows]] = ri_mean

        _factor_shape_metrics.extend(_compute_factor_shape_metrics(
            batch_factors,
            tradable_mask=period_tradable_mask,
        ))
        _pheno_chunks.append(sample_diversity_features(
            batch_factors,
            tradable_mask=period_tradable_mask,
            max_periods=DIVERSITY_MAX_PERIODS,
        ))
        _factor_chunks.append(_sample_tail_factor_values(
            batch_factors,
            batch_ic_direction,
            tradable_mask=period_tradable_mask,
            max_periods=DIVERSITY_MAX_PERIODS,
        ))
        if use_behavior_overlap:
            batch_behavior, batch_behavior_feedback = build_behavior_signatures(
                batch_factors,
                behavior_context,
                ic_directions=batch_is_aux["ic_direction"],
                standardize=False,
            )
            _behavior_chunks.append(batch_behavior)
            _behavior_feedback.extend(batch_behavior_feedback)

        del batch_factors

    full_phenotypes = np.vstack(_pheno_chunks) if _pheno_chunks else None
    del _pheno_chunks
    full_factor_values = np.vstack(_factor_chunks) if _factor_chunks else None
    del _factor_chunks
    behavior_signatures = np.vstack(_behavior_chunks) if _behavior_chunks else None
    del _behavior_chunks

    gc.collect()

    # Format results
    formulas = [formula_string(candidate_X[i]) for i in range(n_candidates)]
    result_objectives = pareto_F_is.copy()
    result_objectives[:, 3] = _compute_population_novelty(
        full_phenotypes,
        result_objectives[:, 1] > -1e5,
        archive_phenotypes=getattr(problem, '_archive_phenotypes', None),
    )
    min_yearly_netret, min_yearly_sharpe, min_yearly_positive_rate, min_yearly_rankicir = _yearly_min_from_store(yearly_store, n_candidates)
    ranking_objectives = _compose_search_objectives(
        result_objectives,
        min_yearly_netret,
        min_yearly_sharpe,
        pareto_aux_is["ls_positive_rate"],
        min_yearly_positive_rate,
    )
    ranking_objectives, positive_rate_multiplier = _apply_positive_rate_soft_multiplier(
        ranking_objectives,
        pareto_aux_is["ls_positive_rate"],
        min_yearly_positive_rate,
    )
    ranking_objectives, rankicir_multiplier = _apply_rankicir_soft_multiplier(
        ranking_objectives,
        result_objectives[:, 0],
        min_yearly_rankicir=min_yearly_rankicir,
    )
    pool_novelty = _compute_archive_novelty(
        full_phenotypes,
        ranking_objectives[:, 1] > -1e5,
        archive_phenotypes=getattr(problem, '_archive_phenotypes', None),
    )
    ranking_objectives, pool_novelty_multiplier = _apply_pool_novelty_soft_penalty(
        ranking_objectives,
        pool_novelty,
    )
    behavior_novelty = _compute_archive_novelty(
        behavior_signatures,
        ranking_objectives[:, 1] > -1e5,
        archive_phenotypes=archive_behavior,
    )
    ranking_objectives, behavior_overlap_multiplier = _apply_pool_novelty_soft_penalty(
        ranking_objectives,
        behavior_novelty,
        config=_behavior_overlap_soft_config(),
    )
    external_behavior_corr = _compute_external_behavior_corr(
        behavior_signatures,
        ranking_objectives[:, 1] > -1e5,
        external_behavior_archive,
    )
    ranking_objectives, external_corr_multiplier, n_external_final_hard = _apply_external_behavior_corr_penalty(
        ranking_objectives,
        external_behavior_corr,
    )
    ranking_objectives, resid_ic_multiplier = _apply_residual_ic_soft_penalty(
        ranking_objectives,
        resid_rankicir_final,
    )
    if SEARCH_NOVELTY_OBJECTIVE_ENABLED:
        ranking_objectives[:, 5] = _combine_search_novelty(
            pool_novelty,
            external_behavior_corr,
            ranking_objectives[:, 1] > -1e5,
            rankic_novelty=rankic_novelty_final,
        )
    if isinstance(external_behavior_archive, np.ndarray) and external_behavior_archive.size > 0:
        finite_ext = np.isfinite(external_behavior_corr)
        print(
            f"[EXTERNAL_ARCHIVE] final corr: valid={int(finite_ext.sum())}/{len(external_behavior_corr)} "
            f"hard={n_external_final_hard} "
            f"median={float(np.nanmedian(external_behavior_corr)) if np.any(finite_ext) else np.nan:.3f} "
            f"max={float(np.nanmax(external_behavior_corr)) if np.any(finite_ext) else np.nan:.3f}",
            flush=True,
        )
    composite = _compute_search_scores(ranking_objectives)

    order, pregate_order_meta, pregate_order_summary = _pre_gate_diversity_order(
        candidate_X,
        composite,
        behavior_novelty=behavior_novelty,
    )
    if pregate_order_summary.get("enabled"):
        print(
            f"[PREGATE_ORDER] selected_by_pass={pregate_order_summary.get('selected_by_pass')} "
            f"behavior_counts={pregate_order_summary.get('behavior_counts')}",
            flush=True,
        )

    results = []
    for rank, idx in enumerate(order):
        direction = -1 if float(pareto_aux_is['ic_direction'][idx]) < 0.0 else 1
        raw_formula = formulas[idx]
        entry = {
            'rank': rank + 1,
            'formula': _directed_formula_string(raw_formula, direction),
            'raw_formula': raw_formula,
            'factor_direction': direction,
            'ic_sign': direction,
            'params': candidate_X[idx].tolist(),
            'decoded': decode_individual(candidate_X[idx]),
            'is_objectives': _objective_dict(result_objectives[idx], pareto_aux_is, idx),
            'is_screening': _screening_dict(pareto_aux_is, idx),
            'yearly_stability': _build_yearly_stability_entry(yearly_store, idx),
            'factor_shape': (
                _factor_shape_metrics[idx]
                if idx < len(_factor_shape_metrics)
                else None
            ),
            'composite': float(composite[idx]),
            'pool_novelty': float(pool_novelty[idx]),
            'pool_corr_max': float(1.0 - pool_novelty[idx]) if pool_novelty[idx] > -1e5 else None,
            'pool_novelty_multiplier': float(pool_novelty_multiplier[idx]),
            'positive_rate_multiplier': float(positive_rate_multiplier[idx]),
            'rankicir_multiplier': float(rankicir_multiplier[idx]),
            'behavior_novelty': float(behavior_novelty[idx]) if behavior_novelty[idx] > -1e5 else None,
            'behavior_corr_max': float(1.0 - behavior_novelty[idx]) if behavior_novelty[idx] > -1e5 else None,
            'behavior_overlap_multiplier': float(behavior_overlap_multiplier[idx]),
            'external_behavior_corr_max': (
                float(external_behavior_corr[idx])
                if np.isfinite(external_behavior_corr[idx])
                else None
            ),
            'external_behavior_corr_multiplier': float(external_corr_multiplier[idx]),
            'rankic_novelty': (
                float(rankic_novelty_final[idx])
                if np.isfinite(rankic_novelty_final[idx])
                else None
            ),
            'rankic_corr_max': (
                float(1.0 - rankic_novelty_final[idx])
                if np.isfinite(rankic_novelty_final[idx]) and rankic_novelty_final[idx] > -1e5
                else None
            ),
            'resid_rankicir': (
                float(resid_rankicir_final[idx])
                if np.isfinite(resid_rankicir_final[idx])
                else None
            ),
            'resid_ic_mean': (
                float(resid_ic_mean_final[idx])
                if np.isfinite(resid_ic_mean_final[idx])
                else None
            ),
            'resid_ic_multiplier': float(resid_ic_multiplier[idx]),
            'behavior_feedback': (
                _behavior_feedback[idx]
                if idx < len(_behavior_feedback)
                else None
            ),
            'pregate_order': (
                pregate_order_meta[idx]
                if idx < len(pregate_order_meta)
                else None
            ),
            '_candidate_index': int(idx),
        }
        entry.update(selection_meta[idx])
        results.append(entry)

    all_candidate_results = results
    n_before_quality_gate = len(results)
    results, quality_failed_counts = _apply_final_quality_gate(results)
    n_after_quality_gate = len(results)
    results, n_formula_duplicates = _dedup_by_formula(results)
    if n_formula_duplicates:
        quality_failed_counts["duplicate_formula"] = (
            quality_failed_counts.get("duplicate_formula", 0) + n_formula_duplicates
        )
    n_after_formula_dedup = len(results)
    if FINAL_SIGNATURE_DEDUP_ENABLED:
        results, n_signature_duplicates = _dedup_by_signature(results)
    else:
        n_signature_duplicates = 0
    if n_signature_duplicates:
        quality_failed_counts["duplicate_signature"] = (
            quality_failed_counts.get("duplicate_signature", 0) + n_signature_duplicates
        )
    n_after_signature_dedup = len(results)
    signature_blacklist = (
        _load_signature_blacklist(FORMULA_BLACKLIST_PATHS)
        if FORMULA_BLACKLIST_ENABLED
        else set()
    )
    results, n_signature_blacklist = _dedup_by_signature_blacklist(
        results, signature_blacklist
    )
    if n_signature_blacklist:
        quality_failed_counts["external_signature_match"] = (
            quality_failed_counts.get("external_signature_match", 0) + n_signature_blacklist
        )
    n_after_signature_blacklist = len(results)
    results, diversity_failed_counts = _apply_final_diversity_constraints(results)
    n_after_diversity_gate = len(results)
    for reason, count in diversity_failed_counts.items():
        quality_failed_counts[reason] = quality_failed_counts.get(reason, 0) + count
    results, low_corr_summary = select_low_corr_results(
        results,
        target_max=result_target_max,
        phenotypes=full_phenotypes,
        behavior_signatures=behavior_signatures,
        factor_values=full_factor_values,
        pool_corr_hard=FINAL_POOL_CORR_HARD,
        spearman_corr_hard=SPEARMAN_CORR_HARD,
        behavior_corr_hard=FINAL_BEHAVIOR_CORR_HARD,
        external_corr_hard=EXTERNAL_CORR_FINAL_HARD,
        field_family_func=indicator_source_family,
        field_family_max_share=FINAL_FIELD_FAMILY_MAX_SHARE,
        ab_pair_max=FINAL_AB_PAIR_MAX,
        tail_overlap_hard=TAIL_OVERLAP_HARD,
    )
    n_after_low_corr_gate = len(results)
    for reason, count in low_corr_summary.get("rejected", {}).items():
        if reason == "target_full":
            continue
        quality_failed_counts[f"low_corr_{reason}"] = (
            quality_failed_counts.get(f"low_corr_{reason}", 0) + int(count)
        )
    print(
        f"[LOW_CORR] retained {n_after_low_corr_gate}/{n_after_diversity_gate} "
        f"pool_hard={FINAL_POOL_CORR_HARD:.2f} "
        f"behavior_hard={FINAL_BEHAVIOR_CORR_HARD:.2f} "
        f"external_hard={EXTERNAL_CORR_FINAL_HARD:.2f} "
        f"tail_hard={TAIL_OVERLAP_HARD:.2f} "
        f"rejected={low_corr_summary.get('rejected')}",
        flush=True,
    )
    candidate_diagnostics = _build_candidate_diagnostics(
        all_candidate_results,
        results[:result_target_max],
    )
    candidate_diagnostics.setdefault("summary", {})["final_low_corr_gate"] = low_corr_summary
    results = [dict(item) for item in results[:result_target_max]]
    for item in results:
        item.pop("_candidate_index", None)
    for idx, item in enumerate(results, start=1):
        item['rank'] = idx

    if len(results) < result_target_min:
        print(f"[WARN] Final quality gate retained {len(results)} factors "
              f"(target_min={result_target_min}, before_gate={n_before_quality_gate}, "
              f"after_quality={n_after_quality_gate}, after_dedup={n_after_formula_dedup}, "
              f"after_signature_dedup={n_after_signature_dedup}, "
              f"after_signature_blacklist={n_after_signature_blacklist}, "
              f"after_diversity={n_after_diversity_gate}, "
              f"after_low_corr={n_after_low_corr_gate}, "
              f"failed={quality_failed_counts})", flush=True)
    else:
        print(f"[QUALITY] Final gate retained {n_after_quality_gate}/{n_before_quality_gate} "
              f"before formula dedup, {n_after_formula_dedup} after dedup, "
              f"{n_after_signature_dedup} after signature dedup, "
              f"{n_after_signature_blacklist} after signature blacklist, "
              f"{n_after_diversity_gate} after diversity, "
              f"{n_after_low_corr_gate} after low-corr; "
              f"failed={quality_failed_counts}", flush=True)

    # Print top results
    print(f"\n{'Rank':>4} | {'Dir':>3} | {'NetRet':>9} | {'RankICIR':>8} | {'NetShp':>8} | {'Turn':>6} | {'Nov':>5} | {'DD':>5} | Formula")
    for r in results[:25]:
        i = r['is_objectives']
        print(f"{r['rank']:4d} | {r['factor_direction']:3d} | {i['ls_netret']:9.4f} | "
              f"{i['rankicir']:8.3f} | {i['ls_net_sharpe']:8.3f} | {-i['neg_turnover']:6.3f} | "
              f"{i['novelty']:5.3f} | {i['ls_1-maxdd']:5.3f} | {r['formula']}")

    unique_base_a = {r['decoded']['A'] for r in results[:20]}
    print(f"\n[SUMMARY] Saved {len(results)}/{n_candidates} candidates "
          f"(Pareto: {n_pareto}, candidate_pool: {n_candidates})", flush=True)
    print(f"[DIVERSITY] Top-20 uses {len(unique_base_a)} distinct base indicators: "
          f"{sorted(unique_base_a)}", flush=True)

    comp_used = sum(
        1 for r in results[:20]
        if r['decoded']['ts_comp_op'] != 'none' or r['decoded']['cs_comp_op'] != 'none'
    )
    print(f"[COMPOSITION] {comp_used}/20 top factors use composition layer", flush=True)
    if POSITIVE_RATE_SOFT_ENABLED:
        pos_mults = [float(r.get('positive_rate_multiplier', 1.0)) for r in results[:20]]
        print(
            f"[POSITIVE_RATE] soft=on target={POSITIVE_RATE_TARGET:.3f} "
            f"yearly_target={YEARLY_POSITIVE_RATE_TARGET:.3f} "
            f"strength={POSITIVE_RATE_SOFT_STRENGTH:.3f} "
            f"top20_boosted={sum(1 for v in pos_mults if v > 1.001)} "
            f"top20_penalized={sum(1 for v in pos_mults if v < 0.999)}",
            flush=True,
        )
    if POOL_NOVELTY_SOFT_ENABLED:
        valid_pool_corr = [
            r['pool_corr_max'] for r in results[:20]
            if r.get('pool_corr_max') is not None
        ]
        max_pool_corr = max(valid_pool_corr) if valid_pool_corr else None
        print(
            f"[POOL_NOVELTY] soft=on target={POOL_NOVELTY_TARGET:.3f} "
            f"strength={POOL_NOVELTY_STRENGTH:.3f} min_mult={POOL_NOVELTY_MIN_MULT:.3f} "
            f"top20_max_corr={max_pool_corr}",
            flush=True,
        )
    if BEHAVIOR_OVERLAP_SOFT_ENABLED:
        valid_behavior_corr = [
            r['behavior_corr_max'] for r in results[:20]
            if r.get('behavior_corr_max') is not None
        ]
        max_behavior_corr = max(valid_behavior_corr) if valid_behavior_corr else None
        behavior_pen = sum(
            1 for r in results[:20]
            if float(r.get('behavior_overlap_multiplier', 1.0)) < 0.999
        )
        print(
            f"[BEHAVIOR_OVERLAP] soft=on target={BEHAVIOR_NOVELTY_TARGET:.3f} "
            f"strength={BEHAVIOR_OVERLAP_STRENGTH:.3f} "
            f"min_mult={BEHAVIOR_OVERLAP_MIN_MULT:.3f} "
            f"top20_max_corr={max_behavior_corr} penalized_top20={behavior_pen}",
            flush=True,
        )

    # Save
    with open(os.path.join(output_dir, 'pareto_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[SAVED] {output_dir}/pareto_results.json", flush=True)
    diagnostics_path = os.path.join(output_dir, "candidate_diagnostics.json")
    with open(diagnostics_path, "w") as f:
        json.dump(candidate_diagnostics, f, indent=2, default=str)
    print(f"[SAVED] {diagnostics_path}", flush=True)

    auto_merge_summary = {'enabled': bool(AUTO_MERGE_ARCHIVE_AFTER_RUN), 'target': None, 'appended': 0, 'error': None}
    auto_merge_target = AUTO_MERGE_ARCHIVE_TARGET or os.environ.get('GP_EXTERNAL_BEHAVIOR_ARCHIVE', '').strip()
    auto_merge_summary['target'] = auto_merge_target or None
    if AUTO_MERGE_ARCHIVE_AFTER_RUN and auto_merge_target:
        try:
            from .build_external_behavior_archive import merge_pareto_into_archive
            archive_meta = getattr(problem, '_external_behavior_archive_meta', {}) or {}
            appended = merge_pareto_into_archive(
                archive_path=auto_merge_target,
                pareto_path=os.path.join(output_dir, 'pareto_results.json'),
                h5_path=str(config.get('h5_path') or archive_meta.get('h5_path') or ''),
                period_minutes=int(archive_meta.get('period_minutes', config.get('minutes_per_period', 1440))),
                start_date=str(archive_meta.get('start_date', start_date)),
                end_date=str(archive_meta.get('end_date', end_date)),
                run_label=os.path.basename(os.path.dirname(output_dir)) or os.path.basename(output_dir),
            )
            auto_merge_summary['appended'] = int(appended)
        except Exception as exc:
            auto_merge_summary['error'] = str(exc)
            print(f'[AUTO_MERGE_ARCHIVE] failed: {exc}; archive unchanged', flush=True)

    # Save evolution metadata
    meta = {
        'version': 'v2',
        'elapsed_seconds': elapsed,
        'population_size': pop_size,
        'n_generations': n_gen,
        'n_params': N_PARAMS,
        'backend': backend,
        'gpu_fitness': bool(problem._gpu_fitness),
        'n_pareto': n_pareto,
        'n_candidate_pool': n_candidates,
        'eval_batch_size': eval_batch_size,
        'chunk_reuse_batches': int(problem._chunk_reuse_batches),
        'cache_dtype': str(cache_dtype),
        'fundamental_panel_path': str(fundamental_panel_path) if fundamental_panel_path else None,
        'perf_profile': bool(problem.perf_profile),
        'search_eval_scope': 'train_until' if loader.n_is < loader.n_active_periods else 'full_sample',
        'start_date': start_date,
        'end_date': end_date,
        'train_end_date': train_end_date,
        'n_is_periods': int(loader.n_is),
        'n_active_periods': int(loader.n_active_periods),
        'search_space_profile': DEFAULT_SEARCH_SPACE_PROFILE,
        'field_slate': {
            'enabled': bool(search_space_state.get('field_slate_enabled')),
            'seed': int(search_space_state.get('field_slate_seed', 0)),
            'target_size': int(search_space_state.get('field_slate_target_size', 0)),
            'advanced_period_enabled': bool(search_space_state.get('advanced_period_enabled')),
            'field_count': int(len(indicator_field_indices)),
            'fields': [str(x) for x in search_space_state.get('field_slate_names', ())],
        },
        'sampling_lanes': sampling_lanes,
        'elite_init': elite_init_meta,
        'search_objective_profile': SEARCH_OBJECTIVE_PROFILE,
        'search_objective_names': SEARCH_OBJECTIVE_NAMES,
        'search_score_weights': SEARCH_SCORE_WEIGHTS.tolist(),
        'positive_rate_soft': _positive_rate_soft_config(),
        'external_behavior_corr': _external_corr_config(),
        'external_behavior_archive_meta': getattr(problem, '_external_behavior_archive_meta', {}),
        'rankic_novelty': getattr(problem, '_rankic_novelty_meta', {}),
        'pool_novelty_soft': _pool_novelty_soft_config(),
        'behavior_overlap_soft': _behavior_overlap_soft_config(),
        'pregate_diversity_order': pregate_order_summary,
        'objective_names': _RESULT_OBJECTIVE_NAMES,
        'factor_direction_source': 'is_rankic_sign',
        'n_pareto_raw': n_pareto,
        'n_results_saved': len(results),
        'n_before_quality_gate': n_before_quality_gate,
        'n_after_quality_gate': n_after_quality_gate,
        'n_after_formula_dedup': n_after_formula_dedup,
        'n_after_signature_dedup': n_after_signature_dedup,
        'n_after_signature_blacklist': n_after_signature_blacklist,
        'n_after_diversity_gate': n_after_diversity_gate,
        'n_after_low_corr_gate': n_after_low_corr_gate,
        'final_signature_dedup_enabled': FINAL_SIGNATURE_DEDUP_ENABLED,
        'final_signature_blacklist': {
            'enabled': FORMULA_BLACKLIST_ENABLED,
            'paths': [
                path.strip() for path in str(FORMULA_BLACKLIST_PATHS or '').split(':')
                if path.strip()
            ],
            'n_loaded_signatures': len(signature_blacklist),
        },
        'final_low_corr_gate': low_corr_summary,
        'final_quality_gate_failed_counts': quality_failed_counts,
        'candidate_diagnostics': {
            'path': diagnostics_path,
            'summary': candidate_diagnostics.get('summary'),
        },
        'final_quality_gate': {
            'min_is_netret': FINAL_MIN_LS_NETRET,
            'min_is_net_sharpe': FINAL_MIN_LS_NET_SHARPE,
            'max_turnover': FINAL_MAX_LS_TURNOVER,
            'min_positive_rate': FINAL_MIN_LS_POSITIVE_RATE,
            'min_coverage': FINAL_MIN_COVERAGE,
            'yearly_stability_windows': [
                tuple(item[:3]) for item in _resolve_yearly_stability_windows(loader.get_active_dates())
            ],
            'min_yearly_ls_netret': FINAL_MIN_YEARLY_LS_NETRET,
            'min_yearly_ls_net_sharpe': FINAL_MIN_YEARLY_LS_NET_SHARPE,
            'min_yearly_rankicir': FINAL_MIN_YEARLY_RANKICIR,
            'min_latest_year_rankicir': FINAL_MIN_LATEST_YEAR_RANKICIR,
            'require_latest_rankic_sign': FINAL_REQUIRE_LATEST_RANKIC_SIGN,
            'max_yearly_turnover': FINAL_MAX_YEARLY_TURNOVER,
            'min_yearly_coverage': FINAL_MIN_YEARLY_COVERAGE,
            'ab_pair_symmetric': AB_PAIR_SYMMETRIC,
            'banned_ab_pairs': [list(pair) for pair in BANNED_AB_PAIRS],
            'crowded_ab_pairs': [list(pair) for pair in CROWDED_AB_PAIRS],
            'diversity': {
                'max_per_operator': FINAL_DIVERSITY_MAX_PER_OPERATOR,
                'max_per_a_field': FINAL_DIVERSITY_MAX_PER_A_FIELD,
                'max_per_source_family': FINAL_DIVERSITY_MAX_PER_SOURCE_FAMILY,
                'max_per_ab_pair': FINAL_DIVERSITY_MAX_PER_AB_PAIR,
                'max_per_template': FINAL_DIVERSITY_MAX_PER_TEMPLATE,
                'pair_a_family_gate_enabled': FINAL_PAIR_A_FAMILY_GATE_ENABLED,
                'pair_a_family_max_share': FINAL_PAIR_A_FAMILY_MAX_SHARE,
                'pair_a_family_default_cap': FINAL_PAIR_A_FAMILY_DEFAULT_CAP,
                'ts_comp_op_max_share': FINAL_TS_COMP_OP_MAX_SHARE,
            },
            'low_corr': {
                'pool_corr_hard': FINAL_POOL_CORR_HARD,
                'behavior_corr_hard': FINAL_BEHAVIOR_CORR_HARD,
                'external_corr_hard': EXTERNAL_CORR_FINAL_HARD,
                'tail_overlap_hard': TAIL_OVERLAP_HARD,
                'field_family_max_share': FINAL_FIELD_FAMILY_MAX_SHARE,
                'ab_pair_max': FINAL_AB_PAIR_MAX,
            },
        },
        'result_target_min': result_target_min,
        'result_target_max': result_target_max,
        'candidate_pool_size': slate_meta['candidate_pool_size'],
        'n_selected_from_pareto': slate_meta['n_selected_from_pareto'],
        'n_selected_from_final_population': slate_meta['n_selected_from_final_population'],
        'candidate_slate_diversity': slate_meta.get('candidate_slate_diversity'),
        'family_feedback': feedback_summary,
        'auto_merge_archive': auto_merge_summary,
    }
    with open(os.path.join(output_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return {'results': results, 'pareto_F_is': pareto_F_is}
