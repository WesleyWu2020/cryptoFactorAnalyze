"""Post-GP refinement: parameter/leaf/structure mutation + evolutionary search."""
import os, re, ast, random
import numpy as np
from .fitness import score_from_factor, compute_factor_score

SEMANTIC_GROUPS = {
    'TIMING': ['mkt_trend_20d', 'mkt_breadth', 'mkt_vol_regime', 'mkt_amt_ratio'],
    'VOLATILITY': ['mkt_vol_20d', 'vol_regime', 'realized_vol', 'downside_vol'],
    'MOMENTUM': ['ret_10d', 'ret_20d', 'price_acceleration'],
    'LIQUIDITY': ['turnover', 'volume_ratio', 'vol_ratio_10d'],
    'RELATIVE': ['beta_20d'],
    'PATTERN': ['gap', 'intraday_range', 'upper_shadow', 'lower_shadow'],
    'QUALITY': ['turnover_stability', 'trend_strength'],
}
FEATURE_TO_GROUP = {f: g for g, fs in SEMANTIC_GROUPS.items() for f in fs}
ALL_FEATURES = list(FEATURE_TO_GROUP.keys())
_LEAF_RE = re.compile(r'\b(' + '|'.join(map(re.escape, ALL_FEATURES)) + r')\b')

TIMING_CONDITIONS = [
    'mkt_trend_20d', 'mkt_breadth', 'mkt_vol_20d', 'mkt_vol_regime', 'mkt_amt_ratio',
    'mul(step(mkt_trend_20d), step(neg(mkt_vol_regime)))',
    'mul(step(mkt_breadth), step(mkt_trend_20d))',
    'mul(step(mkt_amt_ratio), step(mkt_trend_20d))',
    'step(ret_20d)', 'mul(step(ret_10d), step(ret_20d))',
]
DEFENSIVE_CONDITIONS = [
    'step(neg(mkt_vol_regime))', 'step(neg(mkt_vol_20d))', 'step(mkt_breadth)',
    'mul(step(mkt_trend_20d), step(neg(mkt_vol_regime)))',
]

PARAM_RANGES = {
    'tiny': [3, 5, 7, 10],
    'short': [10, 12, 15, 18, 20],
    'medium': [20, 22, 25, 28, 30, 35, 40, 42, 45],
    'long': [40, 45, 50, 55, 60, 65, 70],
}

def _param_range(val):
    if val <= 10: return PARAM_RANGES['tiny'] + PARAM_RANGES['short']
    if val <= 25: return PARAM_RANGES['short'] + PARAM_RANGES['medium']
    if val <= 45: return PARAM_RANGES['medium'] + PARAM_RANGES['long']
    return PARAM_RANGES['long']


def generate_structure_variants(base, timing=None, defensive=None, scale_soft=0.3, scale_def=0.5):
    timing = timing or TIMING_CONDITIONS
    defensive = defensive or DEFENSIVE_CONDITIONS
    vs = []
    for c in timing:
        vs.append(f'if_else({c}, {base}, 0)')
        vs.append(f'if_else({c}, {base}, mul({base}, {scale_soft:.3f}))')
        vs.append(f'mul({base}, tanh({c}))')
    for c in defensive:
        vs.append(f'if_else({c}, {base}, mul({base}, {scale_def:.3f}))')
    for c1 in timing[:3]:
        for c2 in defensive[:2]:
            inner = f'if_else({c2}, {base}, mul({base}, {scale_def:.3f}))'
            vs.append(f'if_else({c1}, {inner}, 0)')
    return vs


def evaluate(formula, bt, y, lag=2, uni_mask=None, top_frac=0.2, min_top_cnt=50, min_years=1):
    try:
        from vendors.quant_lib.formula_backtest import execute_formula
        factor = execute_formula(formula, bt.X, bt.feature_names)
        if np.isfinite(factor).mean() < 0.5: return None
        return score_from_factor(formula, factor, y, uni_mask=uni_mask, top_frac=top_frac,
                                 min_top_cnt=min_top_cnt, min_years=min_years, lag=lag)
    except Exception:
        return None


class AdvancedEvolution:
    def __init__(self, base_formula, bt, lag=2):
        self.base = base_formula
        self.bt = bt
        self.lag = lag
        self.rng = random.Random(int(os.environ.get('EVO_SEED', '0')))
        self.population = []
        self.cache = {}
        y = np.roll(bt.returns.values, -lag, axis=0)
        if lag > 0: y[-lag:] = np.nan
        self.y = y
        uni = getattr(bt, 'universe', None)
        self.uni_mask = None
        if uni is not None:
            try: self.uni_mask = (uni.reindex(index=bt.index_list, columns=bt.col_list).fillna(0).values > 0.5)
            except Exception: pass
        self.top_frac = float(os.environ.get('EVO_TOP_FRAC', '0.2'))
        self.min_top_cnt = int(os.environ.get('EVO_MIN_TOP_CNT', '50'))

    def mutate_param(self, formula, n_variants=5):
        vs = []
        for m in re.finditer(r',\s*(\d+)\s*\)', formula):
            val = int(m.group(1))
            rng = _param_range(val)
            for nv in self.rng.sample(rng, min(n_variants, len(rng))):
                if nv != val:
                    vs.append(formula[:m.start(1)] + str(nv) + formula[m.end(1):])
        return vs

    def mutate_leaf(self, formula, n_per_pos=3):
        vs = []
        for m in _LEAF_RE.finditer(formula):
            leaf = m.group()
            grp = FEATURE_TO_GROUP.get(leaf)
            if grp and grp in SEMANTIC_GROUPS:
                cands = [f for f in SEMANTIC_GROUPS[grp] if f != leaf]
                for nl in self.rng.sample(cands, min(n_per_pos, len(cands))):
                    vs.append(formula[:m.start()] + nl + formula[m.end():])
        return vs

    def mutate_structure(self, formula):
        return generate_structure_variants(formula)

    def _ast_nodes(self, formula):
        try:
            tree = ast.parse(formula, mode='eval')
            nodes = []
            def visit(n):
                if isinstance(n, ast.Call):
                    nodes.append(n)
                    for a in n.args: visit(a)
                elif isinstance(n, ast.Name):
                    nodes.append(n)
            visit(tree.body if isinstance(tree, ast.Expression) else tree)
            return nodes
        except Exception: return []

    def crossover(self, f1, f2):
        try:
            n1, n2 = self._ast_nodes(f1), self._ast_nodes(f2)
            if not n1 or not n2: return None
            a, b = self.rng.choice(n1), self.rng.choice(n2)
            frag = f2[b.col_offset:b.end_col_offset]
            return f1[:a.col_offset] + frag + f1[a.end_col_offset:]
        except Exception: return None

    def _eval(self, formula):
        if formula in self.cache: return self.cache[formula]
        r = evaluate(formula, self.bt, self.y, self.lag, self.uni_mask, self.top_frac, self.min_top_cnt)
        if r: r['score'] = compute_factor_score(r)
        self.cache[formula] = r
        return r

    def evolve(self, n_generations=15, pop_size=80, elite_size=15, early_stop=4, n_jobs=1):
        init = [self.base]
        init += self.mutate_param(self.base, 8)
        init += self.mutate_leaf(self.base, 5)
        init += self.mutate_structure(self.base)
        seen = set()
        self.population = [x for x in init if x not in seen and not seen.add(x)]

        best_ever, best_score, no_improve = None, -999, 0

        for gen in range(n_generations):
            to_eval = [f for f in self.population if f not in self.cache]
            if n_jobs > 1:
                from joblib import Parallel, delayed
                pairs = Parallel(n_jobs=n_jobs, prefer='threads')(
                    delayed(lambda f: (f, evaluate(f, self.bt, self.y, self.lag, self.uni_mask, self.top_frac, self.min_top_cnt)))(f) for f in to_eval)
                for f, r in pairs:
                    if r: r['score'] = compute_factor_score(r)
                    self.cache[f] = r
            else:
                for f in to_eval: self._eval(f)

            results = sorted([self.cache[f] for f in self.population if self.cache.get(f)], key=lambda x: -x['score'])
            if not results: break

            top = results[0]
            if self.bt and hasattr(self.bt, 'start_date'):
                print(f"Gen {gen+1}/{n_generations} | Top: score={top['score']:.1f} mean={top['mean']*100:.1f}% recent={top['recent']*100:.1f}% | pop={len(self.population)}")

            if top['score'] > best_score:
                best_score, best_ever, no_improve = top['score'], top.copy(), 0
            else:
                no_improve += 1

            if no_improve >= early_stop: break

            elites = [r['formula'] for r in results[:elite_size]]
            new_pop = list(elites)
            seen = set(new_pop)

            pool = [r['formula'] for r in results[:min(len(results), elite_size * 4)]]
            scores = np.array([r['score'] for r in results[:len(pool)]])
            w = scores - scores.min()
            weights = (w + 1e-6).tolist() if w.max() > 1e-9 else None

            def _pick():
                return self.rng.choices(pool, weights=weights, k=1)[0] if weights else pool[self.rng.randrange(len(pool))]

            for _ in range(pop_size * 20):
                if len(new_pop) >= pop_size: break
                op = self.rng.random()
                p = _pick()
                if op < 0.25:
                    cs = self.mutate_param(p, 4)
                    child = cs[self.rng.randrange(len(cs))] if cs else None
                elif op < 0.40:
                    cs = self.mutate_leaf(p, 2)
                    child = cs[self.rng.randrange(len(cs))] if cs else None
                elif op < 0.50:
                    cs = self.mutate_structure(p)
                    child = cs[self.rng.randrange(len(cs))] if cs else None
                else:
                    child = self.crossover(_pick(), _pick())
                if child and child not in seen:
                    new_pop.append(child); seen.add(child)

            self.population = new_pop

        return best_ever, results[:20] if results else []
