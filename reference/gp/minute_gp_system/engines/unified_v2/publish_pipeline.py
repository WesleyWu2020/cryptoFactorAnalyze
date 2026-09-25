"""Thin publish layer: filter pareto+formal_replay by "IS AND OOS netret > 0" and push to review queue.

Usage:
    python -m gp.minute_gp_system.engines.unified_v2.publish_pipeline \
        --pareto /path/to/pareto_results.json \
        --formal /path/to/formal_replay.json \
        --experiment exp_20260420_q20_15bp

Outputs: appends pending candidates to the central review queue JSON.
"""
import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from factor_platform.config import DEFAULT_GP_REVIEW_QUEUE_PATH
except Exception:
    DEFAULT_GP_REVIEW_QUEUE_PATH = os.environ.get(
        "GP_REVIEW_QUEUE_PATH", "/tmp/gp_review_queue.json"
    )


DEFAULT_QUEUE = str(DEFAULT_GP_REVIEW_QUEUE_PATH)
MIN_IS_NETRET = float(os.environ.get("GP_PUBLISH_MIN_IS_NETRET", "0.0"))
MIN_IS_SHARPE = float(os.environ.get("GP_PUBLISH_MIN_IS_NET_SHARPE", "0.8"))
MIN_OOS_NETRET = float(os.environ.get("GP_PUBLISH_MIN_OOS_NETRET", "0.08"))
MIN_OOS_SHARPE = float(os.environ.get("GP_PUBLISH_MIN_OOS_NET_SHARPE", "0.8"))
MAX_OOS_TURNOVER = float(os.environ.get("GP_PUBLISH_MAX_OOS_TURNOVER", "0.70"))
MIN_OOS_SEG_POSITIVE_RATIO = float(os.environ.get("GP_PUBLISH_MIN_OOS_SEG_POSITIVE_RATIO", "0.50"))
MIN_OOS_SEG_MIN = float(os.environ.get("GP_PUBLISH_MIN_OOS_SEG_MIN", "-0.10"))
REQUIRE_YEARLY_STABILITY = os.environ.get("GP_PUBLISH_REQUIRE_YEARLY_STABILITY", "1").lower() in {
    "1", "true", "yes", "on",
}
MIN_YEARLY_NETRET = float(os.environ.get("GP_PUBLISH_MIN_YEARLY_NETRET", "0.0"))
MIN_YEARLY_SHARPE = float(os.environ.get("GP_PUBLISH_MIN_YEARLY_SHARPE", "0.5"))
MAX_YEARLY_TURNOVER = float(os.environ.get("GP_PUBLISH_MAX_YEARLY_TURNOVER", "0.85"))
MIN_YEARLY_COVERAGE = float(os.environ.get("GP_PUBLISH_MIN_YEARLY_COVERAGE", "0.70"))


def _pos(v):
    return v is not None and isinstance(v, (int, float)) and v > 0.0


def _finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _yearly_gate_reasons(yearly_stability):
    if not REQUIRE_YEARLY_STABILITY:
        return []
    if not isinstance(yearly_stability, dict):
        return ['yearly_missing']
    summary = yearly_stability.get('summary')
    if not isinstance(summary, dict):
        return ['yearly_missing']
    reasons = []
    min_netret = summary.get('min_ls_netret')
    min_sharpe = summary.get('min_ls_net_sharpe')
    max_turnover = summary.get('max_turnover')
    min_coverage = summary.get('min_coverage')
    if not _finite(min_netret) or min_netret < MIN_YEARLY_NETRET:
        reasons.append('yearly_netret')
    if not _finite(min_sharpe) or min_sharpe < MIN_YEARLY_SHARPE:
        reasons.append('yearly_sharpe')
    if not _finite(max_turnover) or max_turnover > MAX_YEARLY_TURNOVER:
        reasons.append('yearly_turnover')
    if not _finite(min_coverage) or min_coverage < MIN_YEARLY_COVERAGE:
        reasons.append('yearly_coverage')
    return reasons


def _gate_reasons(is_metrics, oos_metrics, yearly_stability=None):
    reasons = []
    is_netret = is_metrics.get('ls_netret')
    is_sharpe = is_metrics.get('ls_net_sharpe')
    oos_netret = oos_metrics.get('ls_netret')
    oos_sharpe = oos_metrics.get('ls_net_sharpe')
    oos_turnover = oos_metrics.get('ls_turnover')
    oos_seg_pos = oos_metrics.get('ls_seg_positive_ratio')
    oos_seg_min = oos_metrics.get('ls_seg_min')
    if not _finite(is_netret) or is_netret < MIN_IS_NETRET:
        reasons.append('is_netret')
    if not _finite(is_sharpe) or is_sharpe < MIN_IS_SHARPE:
        reasons.append('is_sharpe')
    if not _finite(oos_netret) or oos_netret < MIN_OOS_NETRET:
        reasons.append('oos_netret')
    if not _finite(oos_sharpe) or oos_sharpe < MIN_OOS_SHARPE:
        reasons.append('oos_sharpe')
    if not _finite(oos_turnover) or oos_turnover > MAX_OOS_TURNOVER:
        reasons.append('oos_turnover')
    if not _finite(oos_seg_pos) or oos_seg_pos < MIN_OOS_SEG_POSITIVE_RATIO:
        reasons.append('oos_segment')
    if not _finite(oos_seg_min) or oos_seg_min < MIN_OOS_SEG_MIN:
        reasons.append('oos_segment_min')
    reasons.extend(_yearly_gate_reasons(yearly_stability))
    return reasons


def _candidate_id(experiment, formula):
    raw = f"{experiment}::{formula}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def _load_queue(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _save_queue(path, queue):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(queue, f, indent=2, default=str, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Publish GP factors passing IS+OOS>0 gate to review queue.')
    parser.add_argument('--pareto', required=True, help='pareto_results.json from unified_v2.run')
    parser.add_argument('--formal', required=True, help='formal_replay.json from evaluate_with_framework')
    parser.add_argument('--experiment', required=True, help='Experiment name (used for candidate_id namespacing)')
    parser.add_argument('--queue', default=DEFAULT_QUEUE, help='Review queue JSON path')
    args = parser.parse_args()

    with open(args.pareto) as f:
        pareto = json.load(f)
    with open(args.formal) as f:
        formal = json.load(f)

    formal_by_rank = {r['rank']: r for r in formal.get('results', [])}

    pending = []
    failed_counts = {}
    for c in pareto:
        rk = c.get('rank')
        is_obj = c.get('is_objectives') or {}
        f_entry = formal_by_rank.get(rk) or {}

        is_netret = is_obj.get('ls_netret')
        is_sharpe = is_obj.get('ls_net_sharpe')
        oos_netret = f_entry.get('ls_netret')
        oos_sharpe = f_entry.get('ls_net_sharpe')

        yearly_stability = c.get('yearly_stability')
        gate_reasons = _gate_reasons(is_obj, f_entry, yearly_stability)
        if gate_reasons:
            for reason in gate_reasons:
                failed_counts[reason] = failed_counts.get(reason, 0) + 1
            continue

        formula = c.get('formula') or c.get('raw_formula')
        pending.append({
            'candidate_id': _candidate_id(args.experiment, formula),
            'experiment': args.experiment,
            'formula': formula,
            'raw_formula': c.get('raw_formula'),
            'factor_direction': c.get('factor_direction'),
            'params': c.get('params'),
            'decoded': c.get('decoded'),
            'source_rank': rk,
            'is_metrics': {
                'ls_netret': is_netret,
                'ls_net_sharpe': is_sharpe,
                'ls_turnover': -(is_obj.get('neg_turnover') or 0),
                'novelty': is_obj.get('novelty'),
                'ls_1-maxdd': is_obj.get('ls_1-maxdd'),
            },
            'oos_metrics': {
                'ls_netret': oos_netret,
                'ls_net_sharpe': oos_sharpe,
                'ls_turnover': f_entry.get('ls_turnover'),
                'ls_seg_positive_ratio': f_entry.get('ls_seg_positive_ratio'),
                'ls_seg_min': f_entry.get('ls_seg_min'),
                'coverage': f_entry.get('coverage'),
                'ls_maxdd': f_entry.get('ls_maxdd'),
                'ls_netmaxdd': f_entry.get('ls_netmaxdd'),
            },
            'yearly_stability': yearly_stability,
            'publish_gate': {
                'min_is_netret': MIN_IS_NETRET,
                'min_is_sharpe': MIN_IS_SHARPE,
                'min_oos_netret': MIN_OOS_NETRET,
                'min_oos_sharpe': MIN_OOS_SHARPE,
                'max_oos_turnover': MAX_OOS_TURNOVER,
                'min_oos_seg_positive_ratio': MIN_OOS_SEG_POSITIVE_RATIO,
                'min_oos_seg_min': MIN_OOS_SEG_MIN,
                'require_yearly_stability': REQUIRE_YEARLY_STABILITY,
                'min_yearly_netret': MIN_YEARLY_NETRET,
                'min_yearly_sharpe': MIN_YEARLY_SHARPE,
                'max_yearly_turnover': MAX_YEARLY_TURNOVER,
                'min_yearly_coverage': MIN_YEARLY_COVERAGE,
            },
            'oos_window': {
                'start': formal.get('start'),
                'end': formal.get('end'),
                'cost': formal.get('cost'),
                'profile': formal.get('profile'),
            },
            'pareto_source': os.path.abspath(args.pareto),
            'formal_source': os.path.abspath(args.formal),
            'published_at': datetime.now(timezone.utc).isoformat(),
            'status': 'pending',
        })

    queue = _load_queue(args.queue)
    existing_ids = {e['candidate_id'] for e in queue}
    seen_ids = set(existing_ids)
    new = []
    for item in pending:
        candidate_id = item['candidate_id']
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        new.append(item)
    dup_in_batch = len(pending) - len(new)

    queue.extend(new)
    _save_queue(args.queue, queue)

    # Persist new entries to phenotype archive so next GP run has cross-run
    # novelty pressure (knows about past published basins).
    try:
        from .archive import append_archive
        n_added = append_archive([
            {
                'candidate_id': p['candidate_id'],
                'params': p.get('params'),
                'formula': p.get('formula'),
                'experiment': p.get('experiment'),
                'added_at': p.get('published_at'),
            }
            for p in new
        ])
    except Exception as exc:
        print(f'[PUBLISH] archive append failed (non-fatal): {exc}')
        n_added = 0

    print(f'[PUBLISH] experiment={args.experiment}')
    print(f'  pareto candidates:                          {len(pareto)}')
    print(f'  passed strict quality gate:                 {len(pending)}')
    print(f'  failed gate counts:                         {failed_counts}')
    print(f'  already in queue (dedup):                   {dup_in_batch}')
    print(f'  new appended to queue:                      {len(new)}')
    print(f'  queue total:                                {len(queue)}  ({sum(1 for q in queue if q.get("status")=="pending")} pending)')
    print(f'  queue path:                                 {args.queue}')
    print(f'  archive appended:                           {n_added}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
