"""Multi-phase GP pipeline: train -> filter -> (optional) refine."""
import argparse, csv, json, os, pickle, sys
from datetime import datetime
from pathlib import Path
import numpy as np


def _load_json(p): return json.loads(Path(p).read_text('utf-8'))
def _dump_json(o, p): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(o, ensure_ascii=False, indent=2), 'utf-8')
def _dump_pkl(o, p): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).open('wb').write(pickle.dumps(o, pickle.HIGHEST_PROTOCOL))
def _write_lines(ls, p): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text('\n'.join(str(x).strip() for x in ls if str(x).strip()) + '\n', 'utf-8')
def _blend(is_v, oob_v, w_oob=0.7, w_is=0.3):
    try: return w_oob * float(oob_v) + w_is * float(is_v)
    except Exception: return -1e18


def _set_threads():
    for k in ['NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[k] = '1'


def parse_formula(formula_str, feature_names):
    """Parse a formula string into a program token list (for seed injection)."""
    import re
    from .ops import OP_MAP, FINANCIAL_CYCLES
    from copy import deepcopy

    tokens = []
    i = 0
    s = formula_str.strip()
    feat_map = {n: idx for idx, n in enumerate(feature_names)} if feature_names else {}

    while i < len(s):
        if s[i] in ' \t\n,':
            i += 1; continue
        if s[i] == ')':
            i += 1; continue

        m = re.match(r'(-?\d+\.\d+)', s[i:])
        if m:
            tokens.append(float(m.group(1)))
            i += m.end(); continue

        m = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', s[i:])
        if m:
            name = m.group(1)
            i += m.end()
            if name in OP_MAP:
                fn = deepcopy(OP_MAP[name])
                if fn.isRandom and i < len(s):
                    rest = s[i:]
                    pm = re.match(r'\s*\(', rest)
                    if pm: i += pm.end()
                tokens.append(fn)
            elif name in feat_map:
                tokens.append(feat_map[name])
            else:
                try: tokens.append(float(name))
                except ValueError: tokens.append(0)
            continue

        m = re.match(r'(\d+)', s[i:])
        if m:
            v = int(m.group(1))
            if tokens and hasattr(tokens[-1], 'isRandom') and tokens[-1].isRandom:
                tokens[-1].baseConst = v
            else:
                if v in range(len(feature_names) if feature_names else 0):
                    tokens.append(v)
                else:
                    tokens.append(float(v))
            i += m.end(); continue

        if s[i] == '(':
            i += 1; continue
        i += 1

    return tokens if tokens else None


def run_pipeline(cfg: dict, root_dir: Path, run_id: str):
    arts = (root_dir / cfg['run']['artifacts_dir'] / run_id).resolve()
    arts.mkdir(parents=True, exist_ok=True)
    _dump_json(cfg, arts / 'config.json')
    _set_threads()

    sys.path.insert(0, str(root_dir))
    sys.path.insert(0, str(root_dir / 'vendors'))
    from main.globalManager import gm
    from main.comEnum import GMKeyConfig
    from envConfig.configDev import ConfigDev
    if gm.get(GMKeyConfig) is None:
        gm.set(GMKeyConfig, ConfigDev())

    from .trainer import GPTrainer, GPConfig
    from .fitness import score_from_factor, compute_factor_score

    data = cfg['data']
    split = cfg.get('split', {})
    is_s, is_e = split.get('is', [None, None])
    oob_s, oob_e = split.get('oob', [None, None])
    if not (is_s and is_e and oob_s and oob_e):
        raise ValueError("config.split must include is/oob")

    gp = cfg.get('gp', {})
    stream = cfg.get('stream', {})
    final = cfg.get('final', {})
    refine = cfg.get('refine_window', {})
    universe = data['universe']
    lag = int(data.get('lag_periods', 2))
    quantile = float(data.get('quantile', 0.2))
    n_td = int(data.get('n_trading_days', 243))

    stream_dir = Path(stream.get('dir') or str(arts / 'stream')).resolve()
    stream_dir.mkdir(parents=True, exist_ok=True)

    dates = sorted([d for d in [is_s, is_e, oob_s, oob_e] if d])
    start_date, end_date = dates[0], dates[-1]

    # Phase 1: GP Training
    gp_pkl = arts / 'gp.pkl'
    if gp_pkl.exists():
        blob = pickle.loads(gp_pkl.read_bytes())
        formulas = list(blob.get('factor_formula', []))
        is_cagr = list(blob.get('is_cagr', []))
        oob_cagr = list(blob.get('oob_cagr', []))
        print(f"[RESUME] Using existing gp.pkl")
    else:
        params = {
            'start_date': start_date, 'end_date': end_date,
            'train_start_date': str(is_s), 'train_end_date': str(is_e),
            'oob_start_date': str(oob_s), 'oob_end_date': str(oob_e),
            'universe': universe, 'returns_field': data.get('returns_field', 'ctc_returns'),
            'lag_periods': lag, 'quantile': quantile, 'n_trading_days': n_td,
            'fitness_align_backtest': bool(gp.get('fitness_align_backtest', True)),
            'fitness_use_quality_guard': gp.get('fitness_use_quality_guard'),
            'population_size': int(gp.get('population', 250)),
            'generations': int(gp.get('generations', 20)),
            'hall_of_fame': int(gp.get('hall_of_fame', 100)),
            'n_components': int(gp.get('n_components', 60)),
            'gp_n_jobs': int(gp.get('gp_n_jobs', 6)),
            'oob_weight': float(gp.get('oob_weight', 0.25)),
            'elite_size': int(gp.get('elite_size', 0)),
            'diversity_weight': float(gp.get('diversity_weight', 0.0)),
            'secondary_decorrelation': True,
            'decorr_max_points': int(gp.get('decorr_max_points', 200000)),
            'normalize_inputs': bool(gp.get('normalize_inputs', False)),
            'init_depth': gp.get('init_depth', (1, 4)),
            'p_crossover': float(gp.get('p_crossover', 0.25)),
            'p_subtree_mutation': float(gp.get('p_subtree_mutation', 0.20)),
            'p_hoist_mutation': float(gp.get('p_hoist_mutation', 0.10)),
            'p_point_mutation': float(gp.get('p_point_mutation', 0.40)),
            'random_state': int(cfg.get('run', {}).get('seed', 0)),
            'stream_log_enable': bool(stream.get('enable', True)),
            'stream_log_dir': str(stream_dir),
            'stream_min_is': float(stream.get('min_is', 0.15)),
            'stream_min_oob': float(stream.get('min_oob', 0.08)),
            'stream_min_oob_is_ratio': float(stream.get('min_oob_is_ratio', 0.40)),
            'exclude_features': data.get('exclude_features', []),
        }
        for k in ['quality_min_valid_frac', 'quality_min_valid_day_frac', 'quality_min_top_frac',
                   'quality_min_top_day_frac', 'quality_min_nonzero_frac', 'quality_min_nonzero_day_frac',
                   'quality_nonzero_epsilon', 'quality_min_unique_bins', 'quality_min_unique_day_frac']:
            if k in gp: params[k] = gp[k]

        trainer = GPTrainer(params=params)
        trainer.train()
        formulas = list(trainer.factor_formulas)
        is_cagr = list(trainer.ir_is)
        oob_cagr = list(trainer.ir_oob)
        _dump_pkl({'factor_formula': formulas, 'is_cagr': is_cagr, 'oob_cagr': oob_cagr}, gp_pkl)

    # Phase 2: Filter candidates
    w_oob = float(final.get('score_weights', {}).get('oob', 0.7))
    w_is = float(final.get('score_weights', {}).get('is', 0.3))
    min_oob = float(final.get('min_oob_cagr', 0.0))
    min_is = float(final.get('min_is_cagr', stream.get('min_is', 0.15)))
    min_ratio = float(final.get('min_oob_is_ratio', stream.get('min_oob_is_ratio', 0.0)))
    top_n = int(final.get('top_n', 30))

    cand = {}
    for i, f in enumerate(formulas):
        try:
            iv, ov = float(is_cagr[i]), float(oob_cagr[i])
            if iv != iv or ov != ov: continue
            cand[str(f)] = {'formula': str(f), 'is_cagr': iv, 'oob_cagr': ov,
                            'score': _blend(iv, ov, w_oob, w_is), 'source': 'hof'}
        except Exception: continue

    spath = stream_dir / 'evolution_stream.csv'
    if spath.exists():
        try:
            with spath.open('r', encoding='utf-8') as fp:
                for row in csv.DictReader(fp):
                    ff = str(row.get('formula', '')).strip()
                    if not ff: continue
                    try:
                        iv, ov = float(row.get('ann_long_is', 'nan')), float(row.get('ann_long_oob', 'nan'))
                    except Exception: continue
                    if iv != iv or ov != ov: continue
                    sc = _blend(iv, ov, w_oob, w_is)
                    prev = cand.get(ff)
                    if prev is None or sc > prev.get('score', -1e18):
                        cand[ff] = {'formula': ff, 'is_cagr': iv, 'oob_cagr': ov, 'score': sc, 'source': 'stream'}
        except Exception: pass

    items = []
    for x in cand.values():
        try:
            iv, ov = float(x['is_cagr']), float(x['oob_cagr'])
        except Exception: continue
        if iv < min_is or ov < min_oob: continue
        if min_ratio > 0 and iv > 0 and (ov / (iv + 1e-12)) < min_ratio: continue
        items.append({**x, 'score': _blend(iv, ov, w_oob, w_is)})
    items.sort(key=lambda x: x.get('score', -1e18), reverse=True)
    items = items[:top_n]

    _write_lines([f"{x['oob_cagr']:+.4f} | {x['formula']}" for x in items], arts / 'final_formulas_pre_refine.txt')
    _dump_json(items, arts / 'oob_top.json')

    # Phase 3: Optional refine
    refine_kept = None
    rk_path = arts / 'refine_kept.json'
    if rk_path.exists():
        try: refine_kept = _load_json(rk_path); print("[RESUME] Using existing refine_kept.json")
        except Exception: refine_kept = None

    if refine_kept is None and bool(refine.get('enable', False)) and items:
        try:
            win = refine.get('window', [])
            rs, re_ = (str(win[0]), str(win[1])) if len(win) >= 2 else (None, None)
            if rs and re_:
                from .refiner import AdvancedEvolution
                from vendors.quant_lib.formula_backtest import FormulaBacktester
                bt_ref = FormulaBacktester(start_date=rs, end_date=re_, universe=universe)
                bt_ref.load_data()

                seeds = [{'formula': str(x['formula']).strip()} for x in items[:int(refine.get('seed_top_k', len(items)))] if str(x.get('formula', '')).strip()]
                ev = AdvancedEvolution(seeds[0]['formula'], bt_ref, lag=lag)

                import zlib, random as _rnd
                base_seed = int(cfg.get('run', {}).get('seed', 0))
                origin, all_cand = {}, []
                cap = int(refine.get('candidate_cap', 3000))
                per_cap = int(refine.get('per_seed_cap', 300))

                for sd in seeds:
                    sf = str(sd['formula'])
                    sseed = (base_seed * 1000003 + (zlib.adler32(sf.encode()) & 0xFFFFFFFF)) & 0xFFFFFFFF
                    ev.rng = _rnd.Random(int(sseed))
                    one = [sf]
                    one += ev.mutate_param(sf, n_variants=int(refine.get('mutate_param_variants', 6)))
                    one += ev.mutate_leaf(sf, n_per_pos=int(refine.get('mutate_leaf_per_pos', 2)))
                    ts = int(refine.get('mutate_structure_take', 40))
                    if ts != 0:
                        st = ev.mutate_structure(sf)
                        one += st[:ts] if ts > 0 else st
                    seen = set()
                    one = [x for x in one if x not in seen and not seen.add(x)]
                    if per_cap > 0: one = one[:per_cap]
                    for f in one:
                        if f not in origin: origin[f] = sf; all_cand.append(f)
                if cap > 0: all_cand = all_cand[:cap]

                y_ref = np.roll(bt_ref.returns.values, -lag, axis=0)
                if lag > 0: y_ref[-lag:] = np.nan
                uni = getattr(bt_ref, 'universe', None)
                uni_mask = (uni.reindex(index=bt_ref.index_list, columns=bt_ref.col_list).fillna(0).values > 0.5) if uni is not None else None
                top_frac = float(refine.get('eval_top_frac', quantile))
                min_top_cnt = int(refine.get('eval_min_top_cnt', 200))

                from vendors.quant_lib.formula_backtest import execute_formula
                from joblib import Parallel, delayed
                n_jobs = int(refine.get('n_jobs', int(gp.get('gp_n_jobs', 6))))

                def _ev(f):
                    try:
                        fac = execute_formula(f, bt_ref.X, bt_ref.feature_names)
                        if float(np.isfinite(fac).mean()) < 0.5: return None
                        r = score_from_factor(f, fac, y_ref, uni_mask=uni_mask, top_frac=top_frac, min_top_cnt=min_top_cnt, lag=lag)
                        if r: r['score'] = compute_factor_score(r)
                        return r
                    except Exception: return None

                rslt = Parallel(n_jobs=n_jobs, prefer='threads')(delayed(_ev)(f) for f in all_cand) if n_jobs > 1 else [_ev(f) for f in all_cand]
                rslt = [r for r in rslt if r and 'score' in r]

                per_seed = {}
                for r in rslt:
                    fml = str(r.get('formula', '')).strip()
                    s0 = origin.get(fml)
                    if fml and s0: per_seed.setdefault(s0, []).append(r)

                keep_by = str(refine.get('keep_by', 'cagr')).lower()
                min_improve = float(refine.get('min_improve', 0.0))
                _obj = lambda r: float(r.get('score' if keep_by == 'score' else 'cagr', -1e18))

                picked = []
                for sd in seeds:
                    sf = str(sd['formula'])
                    xs = sorted(per_seed.get(sf, []), key=_obj, reverse=True)
                    if not xs:
                        picked.append({'seed_formula': sf, 'kept_formula': sf, 'improved': False}); continue
                    best, base = xs[0], next((r for r in xs if r.get('formula') == sf), xs[0])
                    imp = (_obj(best) - _obj(base)) >= min_improve and best.get('formula') != sf
                    picked.append({'seed_formula': sf, 'kept_formula': best['formula'] if imp else sf, 'improved': imp,
                                   'base': {'formula': sf, 'cagr': base.get('cagr'), 'score': base.get('score')},
                                   'best': {'formula': best['formula'], 'cagr': best.get('cagr'), 'score': best.get('score')}})

                ktn = int(refine.get('keep_top_n', len(seeds)))
                picked.sort(key=lambda x: float((x.get('best') or {}).get(keep_by, -1e18) or -1e18), reverse=True)
                if ktn > 0: picked = picked[:ktn]
                refine_kept = {'window': [rs, re_], 'n_candidates': len(all_cand), 'n_valid': len(rslt), 'picked': picked}
                _dump_json(refine_kept, rk_path)
        except Exception:
            import traceback; _dump_json({'error': traceback.format_exc()}, arts / 'refine_error.json')

    # Write final output
    if refine_kept and refine_kept.get('picked'):
        final_lines = [f"{float((x.get('best') or {}).get('cagr', 0)):+.4f} | {x['kept_formula']}" for x in refine_kept['picked']]
    else:
        final_lines = [f"{x['oob_cagr']:+.4f} | {x['formula']}" for x in items]
    _write_lines(final_lines, arts / 'final_formulas.txt')

    print(f"[DONE] artifacts: {arts}")
    if items: print(f"[TOP1] OOB={items[0]['oob_cagr']:.4f}  {items[0]['formula']}")
    return {'artifacts': str(arts), 'oob_top': items, 'refine_kept': refine_kept}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gp/configs/super_factor_pipeline.json')
    parser.add_argument('--run_id', default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg_path = (root / args.config).resolve()
    if not cfg_path.exists():
        alt = root / 'gp' / args.config
        if alt.exists(): cfg_path = alt
    cfg = _load_json(cfg_path)
    rid = args.run_id or cfg.get('run', {}).get('run_id') or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_pipeline(cfg=cfg, root_dir=root, run_id=rid)


if __name__ == '__main__':
    main()
