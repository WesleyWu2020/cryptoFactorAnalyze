"""Evolution engine: population management, selection, and generational loop."""
import copy, itertools, os, re, gc
from collections import Counter
from time import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from .program import _Program
from .ops import _Function, OP_MAP, FINANCIAL_CYCLES

MAX_INT = np.iinfo(np.int32).max

# ---------------------------------------------------------------------------
# Feature group mapping (for diversity penalty)
# ---------------------------------------------------------------------------
_GROUP_RULES = [
    (re.compile(r'^(amount|volume|turnover|illiquidity|liquidity|vol_ratio|volume_ratio|amount_ratio)'), 'LIQUIDITY'),
    (re.compile(r'^(ret|return|returns|ctc_returns)'), 'RETURN'),
    (re.compile(r'^(trend|momentum|mom)'), 'TREND'),
    (re.compile(r'^(vol_|volatility|realized_vol|downside_vol|vol_regime)'), 'VOLATILITY'),
    (re.compile(r'^(gap|upper_shadow|lower_shadow|intraday_range)'), 'KLINE'),
    (re.compile(r'^(adj_|open|high|low|close|vwap|price)'), 'PRICE'),
    (re.compile(r'^(beta|corr_|skew|kurt)'), 'RISK'),
    (re.compile(r'^(mkt_|market)'), 'MARKET'),
]

def _feat_group(name):
    if not isinstance(name, str): return name
    for pat, grp in _GROUP_RULES:
        if pat.match(name.lower()): return grp
    return name

def _get_groups(features):
    return {_feat_group(f) for f in features}


# ---------------------------------------------------------------------------
# EliteHall: compact global elite management
# ---------------------------------------------------------------------------
class EliteHall:
    def __init__(self, max_size=60, oob_weight=0.0):
        self.hall = []
        self.max_size = max_size
        self.oob_w = oob_weight

    def _blend(self, prog):
        is_v = getattr(prog, 'raw_fitness_', -np.inf) or -np.inf
        oob_v = getattr(prog, 'oob_fitness_', None)
        oob_v = float(oob_v) if oob_v is not None and np.isfinite(float(oob_v)) else float(is_v)
        return (1.0 - self.oob_w) * float(is_v) + self.oob_w * oob_v

    def update(self, population, fitness_arr, min_oob=0.0):
        top_idx = np.argsort(fitness_arr)[::-1][:len(population) // 3 + 1]
        existing = {str(p) for _, p in self.hall}
        for idx in top_idx:
            prog = population[idx]
            formula = str(prog)
            if formula in existing: continue
            raw = prog.raw_fitness_
            if raw is None or raw < -1e5: continue
            oob = getattr(prog, 'oob_fitness_', None)
            if oob is None or not np.isfinite(float(oob)) or float(oob) < min_oob: continue
            self.hall.append((raw, prog))
            existing.add(formula)
        self.hall.sort(key=lambda x: -self._blend(x[1]))
        if len(self.hall) > self.max_size:
            self.hall = self.hall[:self.max_size]

    def inject_into(self, parents, n):
        n = min(n, len(self.hall), len(parents))
        indices = []
        for i in range(n):
            parents[i] = copy.deepcopy(self.hall[i][1])
            indices.append(i)
        return indices

    def inject_population(self, population, n):
        n = min(n, len(self.hall), len(population))
        for i in range(n):
            ec = copy.deepcopy(self.hall[i][1])
            ec.parents = {'method': 'GlobalElite', 'parent_nodes': []}
            population[-(i + 1)] = ec


# ---------------------------------------------------------------------------
# Parallel evolution worker
# ---------------------------------------------------------------------------
def _check_random_state(seed):
    if isinstance(seed, np.random.RandomState): return seed
    return np.random.RandomState(seed)

def _parallel_evolve(n_programs, parents, X, y, sample_weight, seeds, params):
    n_dates, n_features, n_stocks = X.shape
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = int(params['max_samples'] * n_dates)
    feature_names = params['feature_names']
    seed_programs = params.get('seed_programs', [])
    seed_ratio = float(params.get('seed_population_ratio', 0.0) or 0.0)
    seed_mut_prob = float(params.get('seed_mutation_prob', 0.0) or 0.0)
    elite_indices = params.get('elite_indices', [])
    elite_xover_prob = params.get('elite_crossover_prob', 0.5)

    fixed_split = False
    base_sw = base_oob = None
    if sample_weight is not None:
        sw0 = np.asarray(sample_weight, dtype=np.float32)
        if np.any(sw0 == 0) and np.any(sw0 > 0):
            fixed_split = True
            base_sw = sw0
            base_oob = (sw0 == 0).astype(np.float32)

    programs = []
    for i in range(n_programs):
        random_state = _check_random_state(seeds[i])

        if parents is None:
            prog_tokens = None
            if seed_programs and seed_ratio > 0 and random_state.uniform() < seed_ratio:
                base = seed_programs[random_state.randint(len(seed_programs))]
                if seed_mut_prob > 0 and random_state.uniform() < seed_mut_prob:
                    try:
                        tmp = _Program(function_set=function_set, arities=arities, init_depth=init_depth,
                                       init_method=init_method, n_features=n_features, metric=metric,
                                       const_range=const_range, p_point_replace=p_point_replace,
                                       parsimony_coefficient=parsimony_coefficient, feature_names=feature_names,
                                       random_state=random_state, program=base)
                        prog_tokens, _ = tmp.point_mutation(random_state)
                    except Exception:
                        prog_tokens = base
                else:
                    prog_tokens = base
            program = _Program(function_set=function_set, arities=arities, init_depth=init_depth,
                               init_method=init_method, n_features=n_features, metric=metric,
                               const_range=const_range, p_point_replace=p_point_replace,
                               parsimony_coefficient=parsimony_coefficient, feature_names=feature_names,
                               random_state=random_state, program=prog_tokens)
        else:
            def _tournament():
                contenders = random_state.randint(0, len(parents), tournament_size)
                scored = []
                for pi in contenders:
                    f = getattr(parents[pi], 'fitness_', None) if parents[pi] else None
                    scored.append((float(f) if f is not None and np.isfinite(f) else -np.inf, pi))
                return parents[max(scored)[1]], max(scored)[1]

            def _elite_or_tournament():
                if elite_indices and random_state.uniform() < elite_xover_prob:
                    ei = elite_indices[random_state.randint(len(elite_indices))]
                    return parents[ei], ei
                return _tournament()

            valid_program = None
            for _ in range(20):
                method = random_state.uniform()
                parent, pidx = _elite_or_tournament()
                is_simple = (parent.depth_ if hasattr(parent, 'depth_') else 99) <= 2

                if is_simple:
                    if method < 0.55:
                        tokens, mutated = parent.point_mutation(random_state)
                        genome = {'method': 'Point Mutation', 'parent_idx': pidx, 'parent_nodes': mutated}
                    elif method < 0.80:
                        tokens = parent.reproduce()
                        genome = {'method': 'Reproduction', 'parent_idx': pidx, 'parent_nodes': []}
                    else:
                        donor, didx = _elite_or_tournament()
                        tokens, rm, dn = donor.crossover(parent.program, random_state)
                        genome = {'method': 'Crossover', 'parent_idx': didx, 'parent_nodes': rm}
                elif method < method_probs[0]:
                    donor, didx = _elite_or_tournament()
                    tokens, rm, dn = parent.crossover(donor.program, random_state)
                    genome = {'method': 'Crossover', 'parent_idx': pidx, 'parent_nodes': rm}
                elif method < method_probs[1]:
                    tokens, rm, _ = parent.subtree_mutation(random_state)
                    genome = {'method': 'Subtree Mutation', 'parent_idx': pidx, 'parent_nodes': rm}
                elif method < method_probs[2]:
                    tokens, rm = parent.hoist_mutation(random_state)
                    genome = {'method': 'Hoist Mutation', 'parent_idx': pidx, 'parent_nodes': rm}
                elif method < method_probs[3]:
                    tokens, mutated = parent.point_mutation(random_state)
                    genome = {'method': 'Point Mutation', 'parent_idx': pidx, 'parent_nodes': mutated}
                else:
                    tokens = parent.reproduce()
                    genome = {'method': 'Reproduction', 'parent_idx': pidx, 'parent_nodes': []}

                try:
                    cand = _Program(function_set=function_set, arities=arities, init_depth=init_depth,
                                    init_method=init_method, n_features=n_features, metric=metric,
                                    const_range=const_range, p_point_replace=p_point_replace,
                                    parsimony_coefficient=parsimony_coefficient, feature_names=feature_names,
                                    random_state=random_state, program=tokens, strict_grammar=True)
                    if cand.depth_ > 20 or cand.length_ > 200: continue
                    valid_program = cand
                    valid_program.parents = genome
                    break
                except ValueError:
                    continue

            if valid_program is None:
                parent, pidx = _elite_or_tournament()
                tokens = parent.reproduce()
                valid_program = _Program(function_set=function_set, arities=arities, init_depth=init_depth,
                                         init_method=init_method, n_features=n_features, metric=metric,
                                         const_range=const_range, p_point_replace=p_point_replace,
                                         parsimony_coefficient=parsimony_coefficient, feature_names=feature_names,
                                         random_state=random_state, program=tokens)
                valid_program.parents = {'method': 'Reproduction(Fallback)', 'parent_idx': pidx, 'parent_nodes': []}
            program = valid_program

        if fixed_split:
            sw, oob_sw = base_sw, base_oob
        else:
            sw = np.asarray(sample_weight, dtype=np.float32).copy() if sample_weight is not None else np.ones(n_dates, dtype=np.float32)
            oob_sw = sw.copy()
            indices, not_indices = program.get_all_indices(n_dates, max_samples, random_state)
            sw[not_indices] = 0
            oob_sw[indices] = 0

        program.raw_fitness_ = program.raw_fitness_3D(X, y, sw)
        if np.any(oob_sw > 0):
            program.oob_fitness_ = program.raw_fitness_3D(X, y, oob_sw)
        programs.append(program)
    return programs


def _partition(n_estimators, n_jobs):
    n_jobs = min(n_jobs, n_estimators) if n_jobs > 0 else n_estimators
    n_per = n_estimators // n_jobs
    starts = [0]
    for i in range(n_jobs):
        starts.append(starts[-1] + n_per + (1 if i < n_estimators % n_jobs else 0))
    return n_jobs, [starts[i + 1] - starts[i] for i in range(n_jobs)], starts


# ---------------------------------------------------------------------------
# SymbolicTransformer
# ---------------------------------------------------------------------------
class SymbolicTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *, population_size=1000, hall_of_fame=None, n_components=None,
                 generations=20, tournament_size=20, stopping_criteria=0.0,
                 const_range=(-1., 1.), init_depth=(2, 6), init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'), metric='pearson',
                 parsimony_coefficient=0.001, p_crossover=0.9, p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01, p_point_mutation=0.01, p_point_replace=0.05,
                 max_samples=1.0, oob_weight=0.0, diversity_weight=0.0,
                 diversity_decay_start=0.5, diversity_decay_min_ratio=0.1, diversity_compare_top=10,
                 elite_size=0, monotonic_best=False, structure_penalty_weight=1.0,
                 feature_names=None, n_jobs=1, verbose=0, random_state=None):
        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.max_samples = max_samples
        self.oob_weight = oob_weight
        self.elite_size = elite_size
        self.diversity_weight = diversity_weight
        self.diversity_decay_start = diversity_decay_start
        self.diversity_decay_min_ratio = diversity_decay_min_ratio
        self.diversity_compare_top = diversity_compare_top
        self.monotonic_best = monotonic_best
        self.structure_penalty_weight = structure_penalty_weight
        self.feature_names = feature_names
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _get_feature_set(self, prog):
        features = set()
        for node in prog.program:
            if isinstance(node, int) and node >= 0:
                features.add(prog.feature_names[node] if prog.feature_names and node < len(prog.feature_names) else node)
        return features

    def _structure_penalty(self, prog):
        try:
            counts = Counter()
            total = 0
            for node in prog.program:
                if isinstance(node, int) and 0 <= node < prog.n_features:
                    counts[node] += 1; total += 1
            if total == 0: return 0.0
            dominance = max(counts.values()) / float(total)
            pen = 0.0
            if dominance > 0.5: pen += 2.0 * (dominance - 0.5)
            if len(counts) < 2: pen += 0.5
            return float(pen) if np.isfinite(pen) else 0.0
        except Exception:
            return 0.0

    def fit_3D(self, X, y, baseline=None, sample_weight=None, need_parallel=True):
        random_state = _check_random_state(self.random_state)
        hof = self.hall_of_fame or self.population_size
        n_comp = self.n_components or hof

        func_set = []
        for f in self.function_set:
            if isinstance(f, str):
                func_set.append(OP_MAP[f])
            elif isinstance(f, _Function):
                func_set.append(f)
        arities = {}
        for f in func_set:
            arities.setdefault(f.arity, []).append(f)

        from .fitness import _Fitness
        if not isinstance(self.metric, _Fitness):
            raise ValueError('metric must be a _Fitness object')
        _metric = self.metric

        method_probs = np.cumsum([self.p_crossover, self.p_subtree_mutation, self.p_hoist_mutation, self.p_point_mutation])

        params = self.get_params()
        params.update({'_metric': _metric, 'function_set': func_set, 'arities': arities,
                       'method_probs': method_probs, 'seed_programs': [],
                       'seed_population_ratio': float(getattr(self, 'seed_population_ratio', 0.0) or 0.0),
                       'seed_mutation_prob': float(getattr(self, 'seed_mutation_prob', 0.0) or 0.0)})

        self._programs = []
        self.run_details_ = {k: [] for k in ('generation', 'average_length', 'average_fitness',
                                              'best_length', 'best_fitness', 'best_oob_fitness', 'generation_time')}

        elite_hall = EliteHall(max_size=max(60, self.elite_size * 3), oob_weight=self.oob_weight)
        best_raw_fitness, best_raw_program = -np.inf, None
        bad_raw = -np.inf

        # Seed injection
        seed_formulas = getattr(self, 'seed_formulas', None)
        seed_programs = []
        if seed_formulas:
            from .pipeline import parse_formula
            for i, fs in enumerate(seed_formulas):
                try:
                    tokens = parse_formula(fs, self.feature_names)
                    sp = _Program(function_set=func_set, arities=arities, init_depth=self.init_depth,
                                  init_method=self.init_method, n_features=X.shape[1], metric=_metric,
                                  const_range=self.const_range, p_point_replace=self.p_point_replace,
                                  parsimony_coefficient=self.parsimony_coefficient, feature_names=self.feature_names,
                                  random_state=_check_random_state(i), program=tokens)
                    sw = sample_weight.copy() if sample_weight is not None else np.ones(X.shape[0])
                    oob = (sw == 0).astype(np.float32) if np.any(sw == 0) else sw.copy()
                    sp.raw_fitness_ = sp.raw_fitness_3D(X, y, sw)
                    if np.any(oob > 0): sp.oob_fitness_ = sp.raw_fitness_3D(X, y, oob)
                    if sp.raw_fitness_ is not None and np.isfinite(sp.raw_fitness_):
                        elite_hall.hall.append((sp.raw_fitness_, sp))
                        if sp.grammar_valid_ and sp.raw_fitness_ > -1e5:
                            seed_programs.append(sp.program)
                except Exception:
                    pass
            if elite_hall.hall:
                elite_hall.hall.sort(key=lambda x: -x[0])
        params['seed_programs'] = seed_programs

        # CSV stream accumulator
        stream_rows = []
        stream_enable = bool(getattr(self, 'stream_log_enable', True))
        stream_min_is = float(getattr(self, 'stream_min_is', 0.15))
        stream_min_oob = float(getattr(self, 'stream_min_oob', 0.08))
        stream_min_ratio = float(getattr(self, 'stream_min_oob_is_ratio', 0.40))

        for gen in range(self.generations):
            t0 = time()
            parents = None if gen == 0 else self._programs[gen - 1]
            elite_indices = []

            if parents is not None and self.elite_size > 0 and elite_hall.hall:
                elite_indices = elite_hall.inject_into(parents, self.elite_size)
            params['elite_indices'] = elite_indices

            n_jobs, n_progs, starts = _partition(self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            if need_parallel:
                population = list(itertools.chain.from_iterable(
                    Parallel(n_jobs=n_jobs, prefer="threads")(
                        delayed(_parallel_evolve)(n_progs[i], parents, X, y, sample_weight,
                                                  seeds[starts[i]:starts[i + 1]], params)
                        for i in range(n_jobs))))
            else:
                population = list(itertools.chain.from_iterable(
                    [_parallel_evolve(n_progs[i], parents, X, y, sample_weight,
                                      seeds[starts[i]:starts[i + 1]], params) for i in range(n_jobs)]))

            if self.elite_size > 0 and gen > 0:
                elite_hall.inject_population(population, self.elite_size)

            raw_arr = np.array([float(p.raw_fitness_) if p and np.isfinite(getattr(p, 'raw_fitness_', bad_raw)) else bad_raw
                                for p in population], dtype=np.float32)

            if self.monotonic_best:
                ci = int(np.argmax(raw_arr))
                if raw_arr[ci] > best_raw_fitness:
                    best_raw_fitness = raw_arr[ci]
                    best_raw_program = copy.deepcopy(population[ci])
                if best_raw_program is not None and np.isfinite(best_raw_fitness):
                    wi = int(np.argmin(raw_arr))
                    population[wi] = copy.deepcopy(best_raw_program)
                    raw_arr[wi] = best_raw_fitness

            # Compute effective fitness
            use_oob = float(self.oob_weight) > 0
            dw = self.diversity_weight
            if dw > 0 and gen >= int(self.generations * self.diversity_decay_start):
                prog_ratio = (gen - int(self.generations * self.diversity_decay_start)) / max(1, self.generations - int(self.generations * self.diversity_decay_start))
                dw *= max(self.diversity_decay_min_ratio, 1.0 - (1.0 - self.diversity_decay_min_ratio) * prog_ratio)

            feature_sets = [self._get_feature_set(p) for p in population]
            fitness = []
            for i, prog in enumerate(population):
                is_fit = prog.raw_fitness_ if prog.raw_fitness_ is not None else 0.0
                oob_fit = getattr(prog, 'oob_fitness_', None)
                oob_valid = oob_fit is not None and np.isfinite(oob_fit) and oob_fit > -1e5
                if not oob_valid: oob_fit = is_fit
                wf = is_fit if not use_oob else ((1 - self.oob_weight) * is_fit + self.oob_weight * oob_fit)

                sp = self._structure_penalty(prog) * self.structure_penalty_weight
                prog.structure_penalty_ = sp
                n_st = int(X.shape[2]) if X.ndim >= 3 else 0
                adf = float(getattr(prog, 'pred_active_day_frac_', 1.0))
                mvc = float(getattr(prog, 'pred_median_valid_cnt_', float(n_st)))
                k_sparse = min(1.0, max(0.0, adf / 0.7)) * (min(1.0, max(0.0, (mvc / n_st) / 0.3)) if n_st > 0 else 1.0)
                prog.sparse_penalty_ = 1.0 - k_sparse

                base = (wf - sp) * k_sparse

                div_pen = 0.0
                if dw > 0:
                    nf, ng = len(feature_sets[i]), len(_get_groups(feature_sets[i]))
                    if nf == 1: div_pen += dw * 1.5
                    elif nf == 2: div_pen += dw * 0.3
                    if ng == 1 and nf > 1: div_pen += dw * 0.8

                prog.fitness_ = base - div_pen
                prog.diversity_penalty_ = div_pen
                prog.feature_set_ = feature_sets[i]
                prog.feature_groups_ = _get_groups(feature_sets[i])
                fitness.append(prog.fitness_)

            # Stream accumulation
            if stream_enable:
                for p in population:
                    raw = p.raw_fitness_
                    oob = getattr(p, 'oob_fitness_', None)
                    if raw is None or not np.isfinite(raw) or raw < stream_min_is: continue
                    if oob is None or not np.isfinite(oob) or oob < stream_min_oob: continue
                    ratio = float(oob) / (float(raw) + 1e-12) if raw > 0 else 0.0
                    if ratio < stream_min_ratio: continue
                    stream_rows.append((gen, float(raw), float(oob), ratio, str(p)))

            self._programs.append(population)
            if gen > 0: self._programs[gen - 1] = None

            if self.elite_size > 0:
                elite_hall.update(population, np.array(fitness), min_oob=stream_min_oob)

            best_prog = population[np.argmax(fitness)]
            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean([p.length_ for p in population]))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_prog.length_)
            self.run_details_['best_fitness'].append(best_prog.raw_fitness_)
            self.run_details_['best_oob_fitness'].append(getattr(best_prog, 'oob_fitness_', np.nan))
            self.run_details_['generation_time'].append(time() - t0)

            if gen % 5 == 0: gc.collect()

            if self.verbose:
                bf = best_prog.raw_fitness_
                bo = getattr(best_prog, 'oob_fitness_', np.nan)
                uq = len(set(str(p) for p in population))
                print(f"Gen {gen:3d} | Best IS={bf:.3f} OOB={bo:.3f} | AvgFit={np.mean(fitness):.3f} | "
                      f"Unique={uq}/{len(population)} | Elite={len(elite_hall.hall)} | {time()-t0:.1f}s")

            if max(fitness) >= self.stopping_criteria:
                break

        # Write accumulated stream
        if stream_enable and stream_rows:
            log_dir = getattr(self, 'stream_log_dir', None) or os.path.join(os.getcwd(), 'factor_output')
            os.makedirs(log_dir, exist_ok=True)
            import csv
            path = os.path.join(log_dir, 'evolution_stream.csv')
            write_header = not os.path.exists(path)
            with open(path, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(['gen', 'ann_long_is', 'ann_long_oob', 'oob_is_ratio', 'formula'])
                for row in stream_rows:
                    w.writerow([row[0], f'{row[1]:.6f}', f'{row[2]:.6f}', f'{row[3]:.6f}', row[4]])

        # Hall of fame selection
        fitness_arr = np.array(fitness)
        sorted_idx = fitness_arr.argsort()[::-1]
        seen = set()
        unique_idx = []
        for idx in sorted_idx:
            formula = str(self._programs[-1][idx])
            if formula not in seen:
                seen.add(formula)
                unique_idx.append(idx)
            if len(unique_idx) >= hof: break
        hall_indices = np.array(unique_idx)

        eval_progs = [self._programs[-1][i] for i in hall_indices]
        n_pts = X.shape[0] * X.shape[2]
        max_pts = min(n_pts, int(os.environ.get("GP_HOF_CORR_MAX_POINTS", "200000") or 200000))
        sample_idx = None
        if max_pts > 0 and max_pts < n_pts:
            sample_idx = _check_random_state(self.random_state).choice(n_pts, size=max_pts, replace=False)

        if sample_idx is None:
            evaluation = np.array([gp.execute_3D(X).reshape(-1) for gp in eval_progs], dtype=np.float32)
        else:
            evaluation = np.array([gp.execute_3D(X).reshape(-1)[sample_idx] for gp in eval_progs], dtype=np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            corr = evaluation - np.nanmean(evaluation, axis=1).reshape(-1, 1)
            corr = np.abs(np.corrcoef(np.nan_to_num(corr, nan=0.)))
        np.fill_diagonal(corr, 0.)

        components = list(range(len(hall_indices)))
        indices = list(range(len(hall_indices)))
        while len(components) > n_comp:
            most = np.unravel_index(np.argmax(corr), corr.shape)
            worst = max(most)
            components.pop(worst)
            indices.remove(worst)
            corr = corr[:, indices][indices, :]
            indices = list(range(len(components)))

        self._best_programs = [self._programs[-1][i] for i in hall_indices[components]]
        return self
