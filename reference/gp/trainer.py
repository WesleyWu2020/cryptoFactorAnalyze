"""GPTrainer: load data, train GP, evaluate and decorrelate factors."""
import os, time, gc
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

from .features import build_features
from .engine import SymbolicTransformer
from .fitness import make_fitness_long_return, _dense_minmax_rank_2d, score_from_factor, compute_factor_score


@dataclass
class GPConfig:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data_needed: List[str] = field(default_factory=lambda: [
        'adj_closes', 'adj_opens', 'adj_highs', 'adj_lows',
        'amounts', 'volumes', 'turnovers', 'returns', 'float_market_caps', 'vwaps'])
    returns_field: str = 'returns'
    lag_periods: int = 2
    quantile: float = 0.1
    n_trading_days: int = 243
    universe: Optional[str] = None
    benchmark: Optional[str] = None
    sector_neutral: bool = False
    size_neutral: bool = False
    amount_neutral: bool = False
    normalize_inputs: bool = True
    winsor_quantile: float = 0.01
    normalize_features: List[str] = field(default_factory=lambda: ['volume', 'turnover', 'float_mkt_cap'])
    exclude_features: List[str] = field(default_factory=list)
    exclude_operators: List[str] = field(default_factory=list)
    const_range: tuple = (-0.5, 0.5)

    population_size: int = 100
    generations: int = 10
    hall_of_fame: int = 30
    n_components: int = 10
    tournament_size: int = 5
    init_depth: tuple = (1, 5)
    p_crossover: float = 0.3
    p_subtree_mutation: float = 0.35
    p_hoist_mutation: float = 0.2
    p_point_mutation: float = 0.1
    max_samples: float = 1.0
    oob_weight: float = 0.2
    parsimony_coefficient: float = 0.01
    stopping_criteria: float = float('inf')
    elite_size: int = 0
    monotonic_best: bool = False
    diversity_weight: float = 0.0
    diversity_decay_start: float = 0.5
    diversity_decay_min_ratio: float = 0.1
    diversity_compare_top: int = 10
    structure_penalty_weight: float = 1.0
    gp_n_jobs: int = -1
    gp_verbose: int = 1
    random_state: Optional[int] = None

    fitness_align_backtest: bool = True
    fitness_use_quality_guard: Optional[bool] = None
    fitness_use_benchmark: bool = False
    quality_min_valid_frac: float = 0.3
    quality_min_valid_day_frac: float = 0.7
    quality_min_top_frac: float = 0.6
    quality_min_top_day_frac: float = 0.8
    quality_min_valid_cnt: Optional[int] = None
    quality_min_top_cnt: Optional[int] = None
    quality_min_nonzero_frac: float = 0.0
    quality_min_nonzero_day_frac: float = 0.0
    quality_nonzero_epsilon: Optional[float] = None
    quality_min_unique_bins: int = 0
    quality_min_unique_day_frac: float = 0.0

    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    oob_start_date: Optional[str] = None
    oob_end_date: Optional[str] = None
    holdout_start_date: Optional[str] = None
    holdout_end_date: Optional[str] = None
    train_ratio: float = 0.8
    train_horizons: Optional[list] = None
    train_horizon_weights: Optional[list] = None

    decorr_threshold: float = 0.85
    min_decorr_factors: int = 3
    secondary_decorrelation: bool = True
    secondary_decorrelation_top_n: int = 0
    decorr_max_points: int = 0
    decorr_auto_max_points: int = 200000

    seed_formulas: List[str] = field(default_factory=list)
    seed_population_ratio: float = 0.4
    seed_mutation_prob: float = 0.6

    stream_log_enable: bool = True
    stream_log_dir: Optional[str] = None
    stream_min_is: float = 0.15
    stream_min_oob: float = 0.08
    stream_min_oob_is_ratio: float = 0.40

    data_dir: str = ''
    factor_dir: str = ''

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


class GPTrainer:
    def __init__(self, config: GPConfig = None, params: dict = None):
        if config is not None:
            self.cfg = config
        elif params is not None:
            self.cfg = GPConfig(**{k: v for k, v in params.items() if hasattr(GPConfig, k)})
        else:
            self.cfg = GPConfig()

        self.data = {}
        self.returns = None
        self.universe_mask = None
        self.index_returns = None
        self.sectors = None
        self.float_mkt_caps = None
        self.sample_weight = None
        self.feature_names = None
        self.best_programs = []
        self.factor_formulas = []
        self.fits = []
        self.ir_is = []
        self.ir_oob = []
        self.factors = []

    def _read_pkl(self, fld):
        path = os.path.join(self.cfg.data_dir, fld + '.pkl')
        df = pd.read_pickle(path)
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        s = pd.to_datetime(self.cfg.start_date) if self.cfg.start_date else None
        e = pd.to_datetime(self.cfg.end_date) if self.cfg.end_date else None
        if s: df = df[df.index >= s]
        if e: df = df[df.index <= e]
        try:
            if hasattr(df, 'dtypes') and all(np.issubdtype(t, np.number) for t in df.dtypes):
                df = df.astype(np.float32, copy=False)
        except Exception: pass
        return df

    def load_data(self):
        c = self.cfg
        for fld in c.data_needed:
            self.data[fld] = self._read_pkl(fld)
        self.returns = self.data.get(c.returns_field) or self._read_pkl(c.returns_field)
        if c.universe:
            self.universe_mask = (self._read_pkl(c.universe) > 0.5).astype(np.uint8)
        if c.benchmark:
            idx = self._read_pkl('index_data')
            for sfx in ['s_returns', '_returns', '_pct']:
                col = c.benchmark + sfx
                if col in idx.columns:
                    self.index_returns = idx[col]; break
            else:
                raise KeyError(f"Cannot find benchmark column: {c.benchmark}")
        if c.sector_neutral:
            self.sectors = self._read_pkl('industrys')
        if c.size_neutral:
            self.float_mkt_caps = self.data.get('float_market_caps') or self._read_pkl('float_market_caps')

        s = pd.to_datetime(c.start_date) if c.start_date else self.returns.index[0]
        e = pd.to_datetime(c.end_date) if c.end_date else self.returns.index[-1]
        mask = (self.returns.index >= s) & (self.returns.index <= e)
        self.returns = self.returns[mask]
        for d in c.data_needed:
            self.data[d] = self.data[d][(self.data[d].index >= s) & (self.data[d].index <= e)]
        if self.universe_mask is not None:
            self.universe_mask = self.universe_mask[(self.universe_mask.index >= s) & (self.universe_mask.index <= e)]
        if self.index_returns is not None:
            self.index_returns = self.index_returns[(self.index_returns.index >= s) & (self.index_returns.index <= e)]
        if self.sectors is not None:
            self.sectors = self.sectors[(self.sectors.index >= s) & (self.sectors.index <= e)]

    def _build_sample_weight(self, index):
        c = self.cfg
        n = len(index)
        sw = np.full(n, -1.0, dtype=np.float32)
        _ts = lambda d: pd.to_datetime(d) if d else None
        ts, te = _ts(c.train_start_date), _ts(c.train_end_date)
        os_, oe = _ts(c.oob_start_date), _ts(c.oob_end_date)
        hs, he = _ts(c.holdout_start_date), _ts(c.holdout_end_date)

        def _rm(s, e):
            m = np.ones(n, dtype=bool)
            if s: m &= (index >= s)
            if e: m &= (index <= e)
            return m

        ho_mask = _rm(hs, he) if (hs or he) else np.zeros(n, dtype=bool)
        pool = ~ho_mask

        is_m = _rm(ts, te) & ~ho_mask if (ts or te) else None
        oo_m = _rm(os_, oe) & ~ho_mask if (os_ or oe) else None

        if is_m is None and oo_m is None:
            pi = np.where(pool)[0]
            if pi.size > 0:
                cut = int(pi.size * max(0.1, min(0.9, c.train_ratio)))
                cut = max(1, min(pi.size - 1, cut))
                is_m = np.zeros(n, dtype=bool); oo_m = np.zeros(n, dtype=bool)
                is_m[pi[:cut]] = True; oo_m[pi[cut:]] = True
            else:
                is_m = oo_m = np.zeros(n, dtype=bool)
        elif is_m is None: is_m = pool & ~oo_m
        elif oo_m is None: oo_m = pool & ~is_m
        else: oo_m = oo_m & ~is_m

        sw[is_m] = 1.0; sw[oo_m] = 0.0; sw[ho_mask] = -1.0
        print(f"Split: IS={int(is_m.sum())}d, OOB={int(oo_m.sum())}d, Holdout={int(ho_mask.sum())}d")
        return sw

    def _build_neutralize_fn(self, index_list, col_list):
        c = self.cfg
        uni = None
        if self.universe_mask is not None:
            try: uni = (self.universe_mask.reindex(index=index_list, columns=col_list) == 1).values
            except Exception: pass

        do_sec = c.sector_neutral
        sec_codes, n_sec = None, 0
        if do_sec and self.sectors is not None:
            try:
                sd = self.sectors.reindex(index=index_list, columns=col_list).values
                uq = [s for s in pd.unique(sd.ravel()) if pd.notna(s)]
                m = {s: i for i, s in enumerate(uq)}; n_sec = len(uq)
                sec_codes = np.full(sd.shape, -1, dtype=np.int32)
                for s, code in m.items(): sec_codes[sd == s] = code
            except Exception: do_sec = False

        do_sz = c.size_neutral
        log_mc = None
        if do_sz and self.float_mkt_caps is not None:
            try: log_mc = np.log(np.maximum(self.float_mkt_caps.reindex(index=index_list, columns=col_list).values, 1e-10))
            except Exception: do_sz = False

        do_amt = c.amount_neutral
        log_amt = None
        if do_amt:
            amt = self.data.get('amounts')
            if amt is not None:
                try: log_amt = np.log(np.maximum(amt.reindex(index=index_list, columns=col_list).values, 1e-10))
                except Exception: do_amt = False

        if not do_sec and not do_sz and not do_amt:
            if uni is None: return None
            def _fn(y_pred, time_mask=None):
                Y = np.asarray(y_pred, dtype=np.float32)
                um = uni[time_mask] if time_mask is not None else uni
                return np.where(np.isfinite(Y) & um, Y, np.nan)
            return _fn

        def _sec_neut(Y, codes, ns):
            r = Y.copy()
            for t in range(Y.shape[0]):
                row, cr = r[t], codes[t]
                v = np.isfinite(row) & (cr >= 0)
                for ci in range(ns):
                    m = v & (cr == ci)
                    if m.sum() > 0: row[m] -= np.mean(row[m])
            return r

        def _reg_neut(Y, lx):
            r = Y.copy()
            for t in range(Y.shape[0]):
                y, x = Y[t], lx[t]
                m = np.isfinite(y) & np.isfinite(x)
                if m.sum() > 2:
                    ym, xm = y[m], x[m]
                    xd = xm - xm.mean()
                    b = np.dot(xd, ym) / (np.dot(xd, xd) + 1e-10)
                    r[t, m] = ym - b * xd
            return r

        def _neut(y_pred, time_mask=None):
            Y = np.asarray(y_pred, dtype=np.float32).copy()
            if uni is not None:
                um = uni[time_mask] if time_mask is not None else uni
                Y = np.where(um, Y, np.nan)
            if do_sec:
                sc = sec_codes[time_mask] if time_mask is not None else sec_codes
                Y = _sec_neut(Y, sc, n_sec)
            if do_sz:
                lm = log_mc[time_mask] if time_mask is not None else log_mc
                Y = _reg_neut(Y, lm)
            if do_amt:
                la = log_amt[time_mask] if time_mask is not None else log_amt
                Y = _reg_neut(Y, la)
            return np.where(np.isfinite(Y), Y, np.nan)
        return _neut

    def train(self):
        t0 = time.perf_counter()
        self.load_data()
        c = self.cfg
        idx_list = list(self.returns.index)
        col_list = list(self.returns.columns)

        x, feat_names, idx_list, col_list = build_features(self.data, self.returns, c.to_dict())
        self.feature_names = feat_names

        sw = self._build_sample_weight(pd.DatetimeIndex(idx_list))
        self.sample_weight = sw
        y = self.returns.values.astype(np.float32)

        neutralize_fn = self._build_neutralize_fn(idx_list, col_list)

        uni_np = None
        if self.universe_mask is not None:
            try: uni_np = (self.universe_mask.reindex(index=idx_list, columns=col_list) == 1).values
            except Exception: pass

        uq = c.fitness_use_quality_guard
        use_qg = (not c.fitness_align_backtest) if uq is None else bool(uq)

        bm = None
        if c.fitness_use_benchmark and self.index_returns is not None:
            bm = self.index_returns.reindex(idx_list).values.astype(np.float32)

        metric = make_fitness_long_return(
            quantile=c.quantile, n_trading_days=c.n_trading_days,
            horizons=c.train_horizons, horizon_weights=c.train_horizon_weights,
            neutralize_fn=neutralize_fn, benchmark=bm, lag_periods=c.lag_periods,
            use_quality_guard=use_qg, align_backtest=c.fitness_align_backtest,
            universe_mask=uni_np,
            min_valid_frac=c.quality_min_valid_frac, min_valid_day_frac=c.quality_min_valid_day_frac,
            min_top_frac=c.quality_min_top_frac, min_top_day_frac=c.quality_min_top_day_frac,
            min_valid_cnt=c.quality_min_valid_cnt, min_top_cnt=c.quality_min_top_cnt,
            min_nonzero_frac=c.quality_min_nonzero_frac or None,
            min_nonzero_day_frac=c.quality_min_nonzero_day_frac,
            nonzero_epsilon=c.quality_nonzero_epsilon,
            min_unique_bins=c.quality_min_unique_bins or None,
            min_unique_day_frac=c.quality_min_unique_day_frac)

        from .ops import DEFAULT_FUNCTION_SET, OP_MAP
        exclude_ops = set(c.exclude_operators)
        func_set = [OP_MAP[n] for n in DEFAULT_FUNCTION_SET if n not in exclude_ops]

        init_depth = tuple(c.init_depth) if isinstance(c.init_depth, list) else c.init_depth

        est = SymbolicTransformer(
            feature_names=feat_names, function_set=func_set,
            generations=c.generations, population_size=c.population_size,
            tournament_size=c.tournament_size, metric=metric,
            hall_of_fame=c.hall_of_fame, n_components=c.n_components,
            init_method='half and half', init_depth=init_depth,
            stopping_criteria=c.stopping_criteria, const_range=c.const_range,
            p_crossover=c.p_crossover, p_subtree_mutation=c.p_subtree_mutation,
            p_hoist_mutation=c.p_hoist_mutation, p_point_mutation=c.p_point_mutation,
            p_point_replace=0.1, max_samples=c.max_samples, oob_weight=c.oob_weight,
            diversity_weight=c.diversity_weight, diversity_decay_start=c.diversity_decay_start,
            diversity_decay_min_ratio=c.diversity_decay_min_ratio,
            diversity_compare_top=c.diversity_compare_top,
            monotonic_best=c.monotonic_best, elite_size=c.elite_size,
            structure_penalty_weight=c.structure_penalty_weight,
            parsimony_coefficient=c.parsimony_coefficient,
            n_jobs=c.gp_n_jobs, verbose=c.gp_verbose, random_state=c.random_state)

        est.stream_log_enable = c.stream_log_enable
        est.stream_log_dir = c.stream_log_dir
        est.stream_min_is = c.stream_min_is
        est.stream_min_oob = c.stream_min_oob
        est.stream_min_oob_is_ratio = c.stream_min_oob_is_ratio
        if c.seed_formulas:
            est.seed_formulas = c.seed_formulas
            est.seed_population_ratio = c.seed_population_ratio
            est.seed_mutation_prob = c.seed_mutation_prob

        self.data.clear()
        gc.collect()

        print(f"Data loaded in {time.perf_counter()-t0:.2f}s, shape=({x.shape[0]},{x.shape[1]},{x.shape[2]})")
        t1 = time.perf_counter()
        est.fit_3D(x, y, sample_weight=sw)
        print(f"GP training done in {time.perf_counter()-t1:.2f}s")

        programs = list(est._best_programs)
        formulas = [str(p) for p in programs]
        fits = [float(getattr(p, 'raw_fitness_', 0.0)) for p in programs]
        self.ir_is = [float(getattr(p, 'raw_fitness_', np.nan)) for p in programs]
        self.ir_oob = [float(getattr(p, 'oob_fitness_', np.nan)) for p in programs]
        self.best_programs = list(programs)

        if c.secondary_decorrelation and len(programs) > 1:
            programs, formulas, fits = self._decorrelate(programs, formulas, fits, x)
            self.best_programs = list(programs)

        self.factor_formulas = formulas
        self.fits = fits
        self._x_array = x
        self._idx_list = idx_list
        self._col_list = col_list
        del est; gc.collect()
        return self

    def _decorrelate(self, programs, formulas, fits, x_array):
        c = self.cfg
        top_n = c.secondary_decorrelation_top_n
        if top_n > 0 and len(programs) > top_n:
            order = np.argsort(np.asarray(fits))[::-1][:top_n].tolist()
            programs = [programs[i] for i in order]
            formulas = [formulas[i] for i in order]
            fits = [fits[i] for i in order]

        oob_mask = None
        if self.sample_weight is not None:
            oob_mask = (self.sample_weight == 0).astype(bool)

        nd, ns = x_array.shape[0], x_array.shape[2]
        nd_eff = int(oob_mask.sum()) if (oob_mask is not None and oob_mask.any()) else nd
        n_pts = nd_eff * ns

        max_pts = c.decorr_max_points
        if max_pts <= 0 and n_pts > c.decorr_auto_max_points > 0:
            max_pts = c.decorr_auto_max_points

        sample_idx = None
        if 0 < max_pts < n_pts:
            sample_idx = np.random.RandomState(c.random_state or 0).choice(n_pts, size=max_pts, replace=False)

        stacked = []
        for p in programs:
            arr = p.execute_3D(x_array)
            if oob_mask is not None and oob_mask.any(): arr = arr[oob_mask]
            r = _dense_minmax_rank_2d(np.where(np.isfinite(arr), arr, np.nan)).reshape(-1)
            stacked.append((r[sample_idx] if sample_idx is not None else r).astype(np.float32))

        fs = np.stack(stacked, axis=1)
        vm = ~np.isnan(fs).any(axis=1)
        corr = np.corrcoef(fs[vm], rowvar=False) if vm.sum() > 0 else np.eye(len(programs))
        np.fill_diagonal(corr, 0)

        keep = [0]
        for i in range(1, len(programs)):
            mc = max(abs(corr[i, j]) for j in keep)
            th = c.decorr_threshold if len(keep) >= c.min_decorr_factors else max(c.decorr_threshold, 0.85)
            if mc < th: keep.append(i)

        print(f"Decorrelation: {len(programs)} -> {len(keep)} factors")
        return [programs[i] for i in keep], [formulas[i] for i in keep], [fits[i] for i in keep]

    def evaluate(self, top_k=None):
        if not self.best_programs: return []
        x = self._x_array
        idx, col = self._idx_list, self._col_list
        c = self.cfg

        sw = self.sample_weight
        y = self.returns.reindex(index=idx, columns=col).values.astype(np.float32) if self.returns is not None else None
        uni = None
        if self.universe_mask is not None:
            try: uni = (self.universe_mask.reindex(index=idx, columns=col) == 1).values
            except Exception: pass

        results = []
        n = min(len(self.best_programs), top_k or len(self.best_programs))
        for i in range(n):
            p = self.best_programs[i]
            f = p.execute_3D(x)
            formula = self.factor_formulas[i] if i < len(self.factor_formulas) else str(p)
            r = score_from_factor(formula, f, y, uni_mask=uni, top_frac=c.quantile,
                                  lag=c.lag_periods, n_trading_days=c.n_trading_days)
            if r: r['score'] = compute_factor_score(r)
            results.append(r)
        return results

    def get_factor_df(self, idx):
        if not self.best_programs or idx >= len(self.best_programs): return None
        x = self._x_array
        arr = self.best_programs[idx].execute_3D(x)
        return pd.DataFrame(arr, index=self._idx_list, columns=self._col_list)
