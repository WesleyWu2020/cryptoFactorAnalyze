"""Persistent phenotype archive for cross-run novelty pressure.

The production reference set is the active approved factor pool. Older
``phenotype_archive.json`` support remains for legacy experiments, but the
default search should not replay every historical publish ever seen.

For runtime cost control, replayed phenotype / behavior signatures are cached
as small compressed arrays keyed by active approved params and data shape.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from factor_platform.config import DEFAULT_DB_PATH, DEFAULT_GP_PHENOTYPE_ARCHIVE_PATH


ARCHIVE_PATH = DEFAULT_GP_PHENOTYPE_ARCHIVE_PATH
ACTIVE_FEATURE_CACHE_PATH = ARCHIVE_PATH.with_name("active_approved_archive_features.npz")
ARCHIVE_CAP = 2000                # max entries; older ones auto-trimmed
ARCHIVE_MIN_PARAM_LEN = 13        # sanity check
ARCHIVE_FEATURE_CACHE_VERSION = 1


def _load_raw() -> dict:
    if not ARCHIVE_PATH.exists():
        return {'entries': []}
    try:
        with open(ARCHIVE_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'entries' not in data:
            return {'entries': []}
        return data
    except Exception:
        return {'entries': []}


def load_archive_params() -> list[np.ndarray]:
    from .config import N_PARAMS
    data = _load_raw()
    out = []
    for e in data.get('entries', []):
        p = e.get('params')
        if not isinstance(p, list) or len(p) < ARCHIVE_MIN_PARAM_LEN:
            continue
        try:
            arr = np.asarray(p, dtype=np.int32)
        except (ValueError, TypeError):
            continue
        if arr.size < N_PARAMS:
            pad = np.zeros(N_PARAMS - arr.size, dtype=np.int32)
            arr = np.concatenate([arr, pad])
        out.append(arr)
    return out


def _normalize_param_vector(params) -> np.ndarray | None:
    from .config import N_PARAMS

    if not isinstance(params, list) or len(params) < ARCHIVE_MIN_PARAM_LEN:
        return None
    try:
        arr = np.asarray(params, dtype=np.int32)
    except (ValueError, TypeError):
        return None
    if arr.size < N_PARAMS:
        arr = np.concatenate([arr, np.zeros(N_PARAMS - arr.size, dtype=np.int32)])
    return arr[:N_PARAMS].astype(np.int32, copy=False)


def load_active_approved_params(
    db_path: str | Path | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Load replay params from current active approved factor versions.

    Only rows with valid GP params are returned; manual/hypothesis factors with
    no param vector remain in the approved pool but cannot be formula-replayed
    by GP.
    """
    db = Path(db_path or DEFAULT_DB_PATH)
    if not db.exists():
        return [], {"source": "active_approved", "db_path": str(db), "missing_db": True}

    rows = []
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute(
            """
            SELECT
                fv.id AS version_id,
                fv.params_json AS params_json,
                fv.formula AS formula,
                fv.updated_at AS version_updated_at,
                fd.id AS definition_id,
                fd.source_kind AS source_kind,
                fd.lifecycle_state AS definition_state
            FROM factor_versions fv
            JOIN factor_definitions fd ON fd.id = fv.factor_definition_id
            WHERE fv.lifecycle_state = 'approved'
              AND fd.lifecycle_state = 'approved'
            ORDER BY fv.created_at, fv.id
            """
        ))

    params: list[np.ndarray] = []
    digest_rows = []
    skipped_invalid = 0
    source_counts: dict[str, int] = {}
    for row in rows:
        source_kind = str(row["source_kind"] or "unknown")
        source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
        try:
            raw_params = json.loads(row["params_json"] or "[]")
        except json.JSONDecodeError:
            raw_params = []
        arr = _normalize_param_vector(raw_params)
        if arr is None:
            skipped_invalid += 1
            continue
        params.append(arr)
        digest_rows.append({
            "version_id": row["version_id"],
            "definition_id": row["definition_id"],
            "source_kind": source_kind,
            "params": arr.tolist(),
            "version_updated_at": row["version_updated_at"],
        })

    digest = hashlib.sha1(
        json.dumps(digest_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return params, {
        "source": "active_approved",
        "db_path": str(db),
        "approved_versions": len(rows),
        "valid_param_versions": len(params),
        "skipped_invalid_params": skipped_invalid,
        "source_counts": source_counts,
        "params_digest": digest,
    }


def append_archive(entries: Iterable[dict]) -> int:
    """Append new entries to archive. Each entry must have `params` + `candidate_id`.

    Deduplicates by candidate_id. Trims to ARCHIVE_CAP most-recent.
    Returns number of NEW rows added.
    """
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _load_raw()
    existing = data.get('entries', [])
    existing_ids = {e.get('candidate_id') for e in existing}

    new = []
    now = datetime.now(timezone.utc).isoformat()
    for e in entries:
        cid = e.get('candidate_id')
        p = e.get('params')
        if not cid or not isinstance(p, list) or len(p) < ARCHIVE_MIN_PARAM_LEN:
            continue
        if cid in existing_ids:
            continue
        new.append({
            'candidate_id': cid,
            'params': list(p),
            'formula': e.get('formula'),
            'experiment': e.get('experiment'),
            'added_at': e.get('added_at') or now,
        })
        existing_ids.add(cid)

    if not new:
        return 0

    combined = existing + new
    if len(combined) > ARCHIVE_CAP:
        combined = combined[-ARCHIVE_CAP:]
    tmp = ARCHIVE_PATH.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump({'entries': combined}, f, indent=2, default=str)
    os.replace(tmp, ARCHIVE_PATH)
    return len(new)


def _archive_cache_key(problem, source_meta: dict, behavior_enabled: bool) -> dict:
    from .config import DIVERSITY_MAX_PERIODS, N_PARAMS

    loader = problem.loader
    fields = getattr(loader, "indicator_field_indices", None)
    if fields is None:
        field_list = None
    else:
        field_list = [int(x) for x in np.asarray(fields).ravel()]
    symbols = getattr(loader, "symbols", np.asarray([], dtype=object))
    coin_indices = getattr(loader, "coin_indices", np.asarray([], dtype=np.intp))
    selected_symbols = [str(symbols[int(i)]) for i in np.asarray(coin_indices).ravel()] if len(symbols) else []
    symbol_digest = hashlib.sha1(
        json.dumps(selected_symbols, separators=(",", ":")).encode()
    ).hexdigest()
    active_periods = getattr(loader, "active_period_indices", np.asarray([], dtype=np.intp))
    active_periods = np.asarray(active_periods, dtype=np.int64)
    period_digest = hashlib.sha1(active_periods.tobytes()).hexdigest()
    return {
        "version": ARCHIVE_FEATURE_CACHE_VERSION,
        "source": source_meta.get("source"),
        "params_digest": source_meta.get("params_digest"),
        "n_params": int(N_PARAMS),
        "h5_path": str(getattr(loader, "h5_path", "")),
        "minutes_per_period": int(getattr(loader, "minutes_per_period", 0)),
        "n_active_periods": int(getattr(loader, "n_active_periods", 0)),
        "n_tradable_coins": int(getattr(loader, "n_tradable_coins", 0)),
        "active_period_digest": period_digest,
        "symbol_digest": symbol_digest,
        "indicator_field_indices": field_list,
        "diversity_max_periods": int(DIVERSITY_MAX_PERIODS),
        "behavior_enabled": bool(behavior_enabled),
    }


def _cache_enabled() -> bool:
    return os.environ.get("GP_ARCHIVE_FEATURE_CACHE", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _cache_path() -> Path:
    raw = os.environ.get("GP_ARCHIVE_FEATURE_CACHE_PATH", "").strip()
    return Path(raw) if raw else ACTIVE_FEATURE_CACHE_PATH


def _candidate_cache_paths() -> list[Path]:
    primary = _cache_path()
    paths = [primary]
    if primary != ACTIVE_FEATURE_CACHE_PATH:
        paths.append(ACTIVE_FEATURE_CACHE_PATH)
    return paths


def _feature_cache_key_matches(meta: dict, expected: dict) -> bool:
    for key, value in expected.items():
        if meta.get(key) != value:
            return False
    return True


def _try_load_feature_cache(problem, source_meta: dict, behavior_enabled: bool):
    if not _cache_enabled():
        return None, None
    expected = _archive_cache_key(problem, source_meta, behavior_enabled)
    loaded_path = None
    loaded = None
    for path in _candidate_cache_paths():
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata_json"]))
                if not _feature_cache_key_matches(meta, expected):
                    continue
                loaded = (
                    data["phenotypes"].astype(np.float32, copy=False),
                    data["behavior_signatures"].astype(np.float32, copy=False)
                    if "behavior_signatures" in data.files
                    else None,
                )
                loaded_path = path
                break
        except Exception:
            continue
    if loaded is None:
        return None, None
    try:
        phenotypes, behavior = loaded
        if behavior is not None and behavior.size == 0:
            behavior = None
    except Exception:
        return None, None

    from .fitness import standardize_phenotypes

    phenotypes = standardize_phenotypes(phenotypes)
    if behavior is not None:
        behavior = standardize_phenotypes(behavior)
    print(
        f"[ARCHIVE] loaded compressed active-approved features: "
        f"phenotypes={phenotypes.shape[0]}, behavior={0 if behavior is None else behavior.shape[0]} "
        f"from {loaded_path}",
        flush=True,
    )
    return phenotypes, behavior


def _save_feature_cache(problem, source_meta: dict, behavior_enabled: bool, phenotypes, behavior):
    if not _cache_enabled() or phenotypes is None:
        return
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = _archive_cache_key(problem, source_meta, behavior_enabled)
    tmp = path.with_suffix(path.suffix + ".tmp")
    behavior_arr = (
        np.asarray(behavior, dtype=np.float16)
        if behavior is not None
        else np.zeros((0, 0), dtype=np.float16)
    )
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            metadata_json=np.asarray(json.dumps(meta, sort_keys=True), dtype=np.str_),
            phenotypes=np.asarray(phenotypes, dtype=np.float16),
            behavior_signatures=behavior_arr,
        )
    os.replace(tmp, path)
    print(f"[ARCHIVE] saved compressed active-approved features: {path}", flush=True)


def reconstruct_archive_features(
    archive_params,
    problem,
    *,
    behavior_context: dict | None = None,
    source_meta: dict | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Replay archived formulas once and derive pool-comparison features."""
    if not archive_params:
        return None, None
    source_meta = source_meta or {"source": "legacy_json"}
    behavior_enabled = behavior_context is not None
    if source_meta.get("source") == "active_approved":
        cached = _try_load_feature_cache(problem, source_meta, behavior_enabled)
        if cached[0] is not None:
            return cached
    from .fitness import sample_diversity_features, standardize_phenotypes
    from .fitness import compute_five_objectives_prepared
    from .config import DIVERSITY_MAX_PERIODS
    from .behavior_signature import build_behavior_signatures

    pop_arr = np.stack(archive_params, axis=0)
    pop_arr = _sanitize_replay_params(pop_arr)
    available_fields = getattr(problem.loader, "indicator_field_indices", None)
    if available_fields is not None:
        available = set(int(x) for x in np.asarray(available_fields).ravel())

        def _required_fields(row) -> set[int]:
            return {int(row[0]), int(row[1]), int(row[4])}

        keep = np.fromiter(
            (_required_fields(row).issubset(available) for row in pop_arr),
            dtype=bool,
            count=pop_arr.shape[0],
        )
        skipped = int((~keep).sum())
        if skipped:
            print(
                f"[ARCHIVE] skipped {skipped}/{pop_arr.shape[0]} factors outside active indicator cache",
                flush=True,
            )
        pop_arr = pop_arr[keep]
        if pop_arr.size == 0:
            return None, None
    phenos = []
    behaviors = []
    bs = max(8, int(problem.eval_batch_size))
    for start in range(0, pop_arr.shape[0], bs):
        end = min(start + bs, pop_arr.shape[0])
        batch = pop_arr[start:end]
        factors = problem._build_batch_factors(batch)
        phenos.append(sample_diversity_features(
            factors,
            tradable_mask=problem.period_tradable_mask,
            max_periods=DIVERSITY_MAX_PERIODS,
        ))
        if behavior_context is not None:
            _, aux = compute_five_objectives_prepared(
                factors,
                problem._fitness_context_is,
                return_aux=True,
                **problem._fitness_eval_kwargs,
            )
            batch_behavior, _ = build_behavior_signatures(
                factors,
                behavior_context,
                ic_directions=aux.get("ic_direction"),
                standardize=False,
            )
            behaviors.append(batch_behavior)
        del factors
    phen_all = np.vstack(phenos)
    behavior_all = standardize_phenotypes(np.vstack(behaviors)) if behaviors else None
    phen_all = standardize_phenotypes(phen_all)
    if source_meta.get("source") == "active_approved":
        _save_feature_cache(problem, source_meta, behavior_enabled, phen_all, behavior_all)
    return phen_all, behavior_all


def reconstruct_archive_phenotypes(archive_params, problem) -> np.ndarray | None:
    """Evaluate archived param vectors on the current training data."""
    phenotypes, _ = reconstruct_archive_features(archive_params, problem)
    return phenotypes


def _sanitize_replay_params(pop_arr: np.ndarray) -> np.ndarray:
    """Drop stale indices for fields that are not semantically used."""
    from .config import MASK_RULES

    out = np.asarray(pop_arr, dtype=np.int32).copy()
    if out.ndim != 2 or out.shape[1] < 6:
        return out
    # raw mode 0 is mode1, where B is ignored by the evaluator.
    mode1 = out[:, 6] == 0 if out.shape[1] > 6 else np.zeros(out.shape[0], dtype=bool)
    out[mode1, 1] = out[mode1, 0]
    none_mask = out[:, 5] == MASK_RULES.index("none")
    out[none_mask, 4] = out[none_mask, 0]
    return out
