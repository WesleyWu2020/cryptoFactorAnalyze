"""Program AST: grammar-checked expression tree with 3D stack-VM execution."""
from copy import copy, deepcopy
import numpy as np
from sklearn.utils.random import sample_without_replacement
from .ops import _Function, FINANCIAL_CYCLES


class _Program:
    __slots__ = (
        'function_set', 'arities', 'init_depth', 'init_method', 'n_features',
        'const_range', 'metric', 'p_point_replace', 'parsimony_coefficient',
        'feature_names', 'program', '_feature_types', 'grammar_valid_', 'age_',
        'raw_fitness_', 'fitness_', 'oob_fitness_', 'parents',
        '_n_samples', '_max_samples', '_indices_state',
        'structure_penalty_', 'sparse_penalty_', 'diversity_penalty_',
        'feature_set_', 'feature_groups_',
        'pred_active_day_frac_', 'pred_median_valid_cnt_', 'pred_finite_ratio_',
    )

    # Feature type classification
    _TYPE_MAP = {
        'returns': 'RET', 'gap': 'RET', 'ret_5d': 'RET', 'ret_10d': 'RET', 'ret_20d': 'RET',
        'close_position': 'RATIO', 'intraday_range': 'RATIO', 'upper_shadow': 'RATIO',
        'lower_shadow': 'RATIO', 'price_to_ma20': 'RATIO', 'price_to_high250': 'RATIO',
        'turnover': 'TURNOVER',
        'volume_ratio': 'VOL_RATIO', 'amount_ratio': 'VOL_RATIO', 'vol_ratio_5d': 'VOL_RATIO', 'vol_ratio_10d': 'VOL_RATIO',
        'realized_vol': 'VOL', 'vol_regime': 'VOL',
        'adj_close': 'PRICE', 'adj_open': 'PRICE', 'adj_high': 'PRICE', 'adj_low': 'PRICE', 'vwap': 'PRICE',
        'amount': 'RAW_LIQ', 'volume': 'RAW_LIQ',
        'float_mkt_cap': 'RAW_SIZE',
    }

    def __init__(self, function_set, arities, init_depth, init_method, n_features,
                 const_range, metric, p_point_replace, parsimony_coefficient,
                 random_state, feature_names=None, program=None, strict_grammar=False):
        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.feature_names = feature_names
        self.program = program

        self._feature_types = None
        if self.feature_names:
            self._feature_types = [self._TYPE_MAP.get(n, 'OTHER') for n in self.feature_names]

        if self.program is not None:
            if (not self.validate_program()) or (not self._grammar_ok(self.program)):
                if strict_grammar:
                    raise ValueError("Program failed grammar check")
                self.program = None

        if self.program is None:
            for _ in range(10):
                last = self.build_program(random_state)
                if self._grammar_ok(last):
                    self.program = last
                    break
            if self.program is None:
                self.program = last

        self.grammar_valid_ = bool(self.program is not None and self._grammar_ok(self.program))
        self.age_ = 0
        self.raw_fitness_ = None
        self.fitness_ = None
        self.oob_fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None
        self.structure_penalty_ = 0.0
        self.sparse_penalty_ = 0.0
        self.diversity_penalty_ = 0.0
        self.feature_set_ = set()
        self.feature_groups_ = set()
        self.pred_active_day_frac_ = 1.0
        self.pred_median_valid_cnt_ = 0.0
        self.pred_finite_ratio_ = 0.0

    def build_program(self, random_state):
        method = ('full' if random_state.randint(2) else 'grow') if self.init_method == 'half and half' else self.init_method
        max_depth = random_state.randint(*self.init_depth)

        def _pick_cycle(fn):
            valid = [c for c in FINANCIAL_CYCLES if fn.RandRange[0] <= c <= fn.RandRange[1]]
            return valid[random_state.randint(len(valid))] if valid else random_state.randint(fn.RandRange[0], fn.RandRange[1])

        function = deepcopy(self.function_set[random_state.randint(len(self.function_set))])
        if function.isRandom:
            function.baseConst = _pick_cycle(function)

        program = [function]
        terminal_stack = [function.arity]
        ts_depth_stack = [1 if function.is_ts_op else 0]
        in_ts_subtree = [function.is_ts_op]
        seen_terminals = []

        while terminal_stack:
            depth = len(terminal_stack)
            choice = random_state.randint(self.n_features + len(self.function_set))
            curr_ts = ts_depth_stack[-1]
            is_in_ts = in_ts_subtree[-1]

            if (depth < max_depth) and (method == 'full' or choice <= len(self.function_set)):
                candidates = self.function_set
                if curr_ts >= 1:
                    candidates = [f for f in candidates if not f.is_ts_op]
                if is_in_ts:
                    allowed = {'delta', 'delay', 'pos', 'abs', 'tanh', 'add', 'sub', 'mul', 'safe_div', 'neg', 'step', 'if_else'}
                    candidates = [f for f in candidates if f.name in allowed]
                if not candidates:
                    candidates = self.function_set

                function = deepcopy(candidates[random_state.randint(len(candidates))])
                if function.isRandom:
                    function.baseConst = _pick_cycle(function)
                program.append(function)
                terminal_stack.append(function.arity)
                ts_depth_stack.append(curr_ts + (1 if function.is_ts_op else 0))
                in_ts_subtree.append(function.is_ts_op)
            else:
                in_ts_ctx = curr_ts > 0
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features if in_ts_ctx else self.n_features + 1)
                    while terminal in seen_terminals and terminal != self.n_features:
                        terminal = random_state.randint(self.n_features if in_ts_ctx else self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)

                if (not in_ts_ctx) and terminal == self.n_features:
                    terminal = round(random_state.uniform(*self.const_range), 3)
                    while terminal == 0:
                        terminal = random_state.uniform(*self.const_range)

                program.append(terminal)
                terminal_stack[-1] -= 1
                if terminal_stack[-1] > 0:
                    seen_terminals.append(terminal)
                while terminal_stack[-1] == 0:
                    seen_terminals = []
                    terminal_stack.pop()
                    if ts_depth_stack: ts_depth_stack.pop()
                    if in_ts_subtree: in_ts_subtree.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
        return None

    # ------------------------------------------------------------------
    # Grammar check (dimensional type system)
    # ------------------------------------------------------------------
    _ADD_OPS = {'add', 'sub'}
    _MUL_OPS = {'mul'}
    _DIV_OPS = {'div', 'safe_div'}
    _ABS_DIMS = {'PRICE', 'RAW_LIQ', 'RAW_SIZE'}
    _REL_DIMS = {'RET', 'RATIO', 'TURNOVER', 'VOL_RATIO', 'VOL', 'CORR'}
    _NORM_TYPES = {'NORM', 'NORM_ABS', 'NORM_REL'}
    _ADD_FORBIDDEN = {'PRICE', 'RAW_LIQ', 'RAW_SIZE', 'NORM_ABS'}
    _CS_NORM = {'cs_rank', 'cs_zscore', 'rank', 'scale'}
    _TS_NORM = {'ts_zscore', 'ts_rank', 'cs_rank', 'cs_zscore', 'rank', 'scale',
                'ts_ir', 'dynamic_ts_rank', 'dynamic_ts_skew', 'dynamic_ts_kurt'}
    _TS_PRESERVE = {'dynamic_ts_mean', 'ts_ema', 'ts_range', 'dynamic_ts_max',
                    'dynamic_ts_min', 'dynamic_ts_std', 'dynamic_ts_sum', 'ts_std'}
    _TS_ALLOWED_ABS = {'ts_zscore', 'ts_rank', 'dynamic_ts_sum', 'dynamic_ts_mean',
                       'dynamic_ts_max', 'dynamic_ts_min', 'dynamic_ts_std', 'ts_ema',
                       'ts_range', 'delta', 'delay', 'ts_ir', 'ts_return',
                       'dynamic_ts_rank', 'cs_zscore', 'rank', 'scale'}
    _TS_INNER = {'delta', 'delay', 'pos', 'abs', 'tanh', 'safe_div', 'add', 'sub', 'mul', 'neg'}
    _NO_CHAIN = {'abs', 'neg', 'delta'}
    _LOGIC_OPS = {'step', 'if_else'}
    _ADD_COMPAT = {frozenset({'RET', 'RATIO'}), frozenset({'VOL', 'CORR'}), frozenset({'VOL', 'VOL_RATIO'})}

    def _grammar_ok(self, program):
        if program is None:
            return False
        ft = self._feature_types
        stack = []
        feat_used = set()
        n_logic = n_ts = 0

        def _dim(t): return t[0] if isinstance(t, tuple) else t
        def _has_abs(children): return any(_dim(c) in self._ABS_DIMS or _dim(c) == 'NORM_ABS' for c in children)
        def _pass_dim(children):
            for c in children:
                d = _dim(c)
                if d is not None: return d
            return 'OTHER'

        def _infer(func, cdims):
            name = func.name if func else ''

            if name in self._CS_NORM:
                if any(d in self._ABS_DIMS for d in cdims): return None
                if any(d in self._NORM_TYPES for d in cdims): return None
                return 'NORM_REL'
            if name in self._TS_NORM:
                return 'NORM_REL'
            if name in {'max', 'min'}:
                real = [d for d in cdims if d not in {'CONST', 'OTHER', None}]
                if any(d in self._ABS_DIMS or d == 'NORM_ABS' for d in real) and any(d in self._REL_DIMS or d == 'NORM_REL' for d in real):
                    return None
                if not real: return _pass_dim([(d,) for d in cdims])
                if len(real) == 1: return real[0]
                t1, t2 = real[0], real[1]
                if t1 == t2: return t1
                if 'NORM_REL' in (t1, t2) and (t1 in self._REL_DIMS or t2 in self._REL_DIMS): return 'NORM_REL'
                if frozenset({t1, t2}) in self._ADD_COMPAT: return 'NORM_REL'
                return None
            if name in self._ADD_OPS:
                real = [d for d in cdims if d not in {'CONST', 'OTHER', None}]
                if any(d in self._ADD_FORBIDDEN for d in real): return None
                if not real: return _pass_dim([(d,) for d in cdims])
                if len(real) == 1: return real[0]
                t1, t2 = real[0], real[1]
                if t1 == t2: return t1
                if 'NORM_REL' in (t1, t2) and (t1 in self._REL_DIMS or t2 in self._REL_DIMS): return 'NORM_REL'
                if frozenset({t1, t2}) in self._ADD_COMPAT: return 'NORM_REL'
                return None
            if name in self._MUL_OPS:
                real = [d for d in cdims if d not in {'CONST', 'OTHER', None}]
                if any(d in {'RAW_LIQ', 'RAW_SIZE', 'NORM_ABS'} for d in real): return None
                if not real: return _pass_dim([(d,) for d in cdims])
                if len(real) == 1: return real[0]
                t1, t2 = real[0], real[1]
                if t1 == t2: return t1
                safe = {'NORM', 'NORM_REL'} | self._REL_DIMS
                if t1 in safe and t2 in safe: return 'NORM_REL'
                if 'PRICE' in (t1, t2) and ('NORM' in (t1, t2) or 'NORM_REL' in (t1, t2)): return 'NORM_ABS'
                return None
            if name in self._DIV_OPS:
                real = [d for d in cdims if d not in {'CONST', 'OTHER', None}]
                if any(d in {'RAW_LIQ', 'RAW_SIZE', 'NORM_ABS'} for d in real): return None
                if not real: return _pass_dim([(d,) for d in cdims])
                if len(real) == 1: return real[0]
                t1, t2 = real[0], real[1]
                if t1 == t2: return 'NORM_REL'
                safe = {'NORM', 'NORM_REL'} | self._REL_DIMS
                if t1 in safe and t2 in safe: return 'NORM_REL'
                if 'PRICE' in (t1, t2) and (t1 == t2): return 'RATIO'
                return None
            if name == 'neg': return cdims[0] if cdims else 'OTHER'
            if name == 'step': return 'NORM_REL'
            if name == 'if_else':
                if len(cdims) != 3: return None
                dt, df = cdims[1], cdims[2]
                is_t_abs = dt in self._ABS_DIMS or dt == 'NORM_ABS'
                is_f_abs = df in self._ABS_DIMS or df == 'NORM_ABS'
                is_t_rel = dt in self._REL_DIMS or dt == 'NORM_REL'
                is_f_rel = df in self._REL_DIMS or df == 'NORM_REL'
                if (is_t_abs and is_f_rel) or (is_t_rel and is_f_abs): return None
                if dt == 'CONST': return df
                if df == 'CONST': return dt
                if dt == df: return dt
                if 'NORM_REL' in (dt, df) and (dt in self._REL_DIMS or df in self._REL_DIMS): return 'NORM_REL'
                if frozenset({dt, df}) in self._ADD_COMPAT: return 'NORM_REL'
                return None
            if name in {'abs', 'pos'}: return cdims[0] if cdims else 'OTHER'
            if name == 'tanh':
                d = cdims[0] if cdims else 'OTHER'
                if d in self._ABS_DIMS or d == 'NORM_ABS': return None
                return d
            if name == 'delay':
                return 'NORM_ABS' if (cdims and cdims[0] == 'PRICE') else (cdims[0] if cdims else 'OTHER')
            if name == 'delta':
                return 'NORM_ABS' if _has_abs([(d,) for d in cdims]) else 'RET'
            if name == 'ts_return': return 'RET'
            if name in self._TS_PRESERVE:
                d = cdims[0] if cdims else 'OTHER'
                return 'NORM_ABS' if _has_abs([(d,) for d in cdims]) else d
            return cdims[0] if cdims else 'OTHER'

        try:
            for node in program:
                if isinstance(node, _Function):
                    name = node.name
                    is_ts = node.is_ts_op

                    if stack:
                        p = stack[-1]
                        p_ts_depth, p_ts_parent = p['ts_depth'], p['ts_parent']
                        p_delay = p['delay_chain']
                        p_depth = p['depth']
                        p_name = p['func'].name if p['func'] else None
                    else:
                        p_ts_depth = p_ts_parent = p_delay = p_depth = 0
                        p_name = None

                    if name in self._LOGIC_OPS:
                        n_logic += 1
                        if n_logic > 2: return False
                    if is_ts:
                        n_ts += 1
                        if n_ts > 6: return False

                    curr_ts = p_ts_depth + (1 if is_ts else 0)
                    if curr_ts > 3: return False
                    curr_delay = p_delay + 1 if name == 'delay' else 0
                    if curr_delay > 8: return False
                    curr_depth = p_depth + 1

                    if stack:
                        pf = stack[-1]['func']
                        if pf and pf.arity == 1 and node.arity == 1 and pf.name in self._NO_CHAIN and name == pf.name:
                            return False
                    if p_ts_parent and not is_ts and name not in self._TS_INNER:
                        return False
                    if name == 'if_else' and curr_depth > 2: return False
                    if p_ts_parent and name in {'step', 'if_else'}: return False

                    ts_par = p_ts_parent if p_ts_parent else (name if is_ts else None)
                    stack.append({'pending': node.arity, 'ts_depth': curr_ts, 'ts_parent': ts_par,
                                  'delay_chain': curr_delay, 'func': node, 'child_dims': [],
                                  'has_leaf': False, 'depth': curr_depth})
                else:
                    if not stack: continue
                    ctx = stack[-1]
                    ts_par = ctx['ts_parent']
                    dim = 'CONST'
                    is_leaf = False

                    if isinstance(node, int) and 0 <= node < self.n_features:
                        is_leaf = True
                        feat_used.add(node)
                        if ft and node < len(ft):
                            dim = ft[node]
                            pname = ctx['func'].name if ctx['func'] else None

                            if self.feature_names and node < len(self.feature_names):
                                fn = self.feature_names[node]
                                if isinstance(fn, str) and fn.startswith('mkt_'):
                                    if pname != 'if_else' and (ts_par != 'if_else' if ts_par else True):
                                        return False

                            if dim in self._ABS_DIMS:
                                if not (pname in self._TS_ALLOWED_ABS or (ts_par and ts_par in self._TS_ALLOWED_ABS)):
                                    return False
                                if ts_par in self._CS_NORM or pname in self._CS_NORM:
                                    return False
                            if ts_par == 'dynamic_ts_sum' and dim in {'PRICE', 'CONST', 'OTHER'}:
                                return False
                    elif isinstance(node, int):
                        dim = 'OTHER'

                    ctx['pending'] -= 1
                    ctx['child_dims'].append(dim)
                    if is_leaf: ctx['has_leaf'] = True

                    while stack and stack[-1]['pending'] == 0:
                        done = stack.pop()
                        fn, cdims = done['func'], done['child_dims']
                        out_dim = _infer(fn, cdims)
                        if out_dim is None: return False

                        if isinstance(fn, _Function) and fn.is_ts_op and not done['has_leaf']:
                            return False
                        if isinstance(fn, _Function) and fn.arity == 1 and cdims and cdims[0] == 'CONST':
                            return False

                        fname = fn.name if fn else ''
                        if fname == 'safe_div' and len(cdims) == 2:
                            if cdims[1] in {'RET', 'RATIO'}: return False
                            if cdims[0] in self._ABS_DIMS: return False
                        if fname == 'ts_ir' and len(cdims) == 1:
                            ok_dims = {'RET', 'RATIO', 'TURNOVER', 'VOL_RATIO', 'VOL', 'CORR', 'NORM_REL'}
                            if cdims[0] not in ok_dims: return False

                        if out_dim in {'RAW_LIQ', 'RAW_SIZE', 'PRICE', 'NORM_ABS'}:
                            return False

                        if stack:
                            stack[-1]['pending'] -= 1
                            stack[-1]['child_dims'].append(out_dim)
                            if done['has_leaf']: stack[-1]['has_leaf'] = True

            return len(feat_used) >= 2
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Execution engine (stack VM)
    # ------------------------------------------------------------------
    def execute_3D(self, X):
        program = self.program
        node = program[0]
        n_date, n_feat, n_stock = X.shape
        shape = (n_date, n_stock)

        if isinstance(node, float): return np.full(shape, node, dtype=np.float32)
        if isinstance(node, int): return X[:, node, :].astype(np.float32, copy=False)

        stack = []
        for i in range(len(program) - 1, -1, -1):
            node = program[i]
            if isinstance(node, _Function):
                arity = node.arity
                is_ts = node.is_ts_op
                if arity == 1:
                    if not stack: return None
                    a = stack.pop()
                    inp = X[:, a, :] if isinstance(a, int) else a
                    if is_ts and isinstance(inp, float): inp = np.full(shape, inp, dtype=np.float32)
                    stack.append(node(inp))
                elif arity == 2:
                    if len(stack) < 2: return None
                    a1, a2 = stack.pop(), stack.pop()
                    v1 = X[:, a1, :] if isinstance(a1, int) else a1
                    v2 = X[:, a2, :] if isinstance(a2, int) else a2
                    if is_ts and isinstance(v1, float): v1 = np.full(shape, v1, dtype=np.float32)
                    if is_ts and isinstance(v2, float): v2 = np.full(shape, v2, dtype=np.float32)
                    stack.append(node(v1, v2))
                else:
                    if len(stack) < arity: return None
                    args = []
                    for _ in range(arity):
                        a = stack.pop()
                        val = X[:, a, :] if isinstance(a, int) else a
                        if is_ts and isinstance(val, float): val = np.full(shape, val, dtype=np.float32)
                        args.append(val)
                    stack.append(node(*args))
            else:
                stack.append(node)

        if not stack: return None
        result = stack.pop()
        if isinstance(result, int): return X[:, result, :].astype(np.float32, copy=False)
        if isinstance(result, float): return np.full(shape, result, dtype=np.float32)
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def validate_program(self):
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    @property
    def depth_(self):
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    @property
    def length_(self):
        return len(self.program)

    def __str__(self):
        terminals = [0]
        output = ''
        rf_stack = []
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                rf_stack.append(deepcopy(node))
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    output += self.feature_names[node] if self.feature_names else f'X{node}'
                else:
                    output += f'{node:.3f}'
                terminals[-1] -= 1
                if rf_stack:
                    rf_stack[-1].arity -= 1
                    if rf_stack[-1].isRandom and rf_stack[-1].arity == 0:
                        output += ',' + str(rf_stack[-1].baseConst)
                while terminals[-1] == 0:
                    rf_stack.pop()
                    terminals.pop()
                    terminals[-1] -= 1
                    if rf_stack:
                        rf_stack[-1].arity -= 1
                        output += ')'
                        if rf_stack and rf_stack[-1].isRandom:
                            output += ',' + str(rf_stack[-1].baseConst)
                    else:
                        output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def get_subtree(self, random_state, program=None):
        if program is None: program = self.program
        probs = np.array([0.9 if isinstance(n, _Function) else 0.1 for n in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())
        stack, end = 1, start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function): stack += node.arity
            end += 1
        return start, end

    def reproduce(self):
        return copy(self.program)

    def crossover(self, donor, random_state):
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        ds, de = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) - set(range(ds, de)))
        return self.program[:start] + donor[ds:de] + self.program[end:], removed, donor_removed

    def subtree_mutation(self, random_state):
        return self.crossover(self.build_program(random_state), random_state)

    def hoist_mutation(self, random_state):
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        ss, se = self.get_subtree(random_state, subtree)
        hoist = subtree[ss:se]
        removed = list(set(range(start, end)) - set(range(start + ss, start + se)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        program = copy(self.program)
        mutate = np.where(random_state.uniform(size=len(program)) < self.p_point_replace)[0]
        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                replacement = self.arities[arity][random_state.randint(len(self.arities[arity]))]
                replacement.baseConst = program[node].baseConst
                program[node] = replacement
            else:
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                program[node] = terminal
        return program, list(mutate)

    def get_all_indices(self, n_samples=None, max_samples=None, random_state=None):
        if self._indices_state is None and random_state is None:
            raise ValueError('Program not evaluated, indices not available.')
        if n_samples is not None and self._n_samples is None: self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None: self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()
        from sklearn.utils.random import sample_without_replacement
        state = np.random.RandomState()
        state.set_state(self._indices_state)
        not_indices = sample_without_replacement(self._n_samples, self._n_samples - self._max_samples, random_state=state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        return np.where(sample_counts == 0)[0], not_indices

    def raw_fitness_3D(self, X, y, sample_weight):
        BAD = -1e6 if self.metric.greater_is_better else np.inf
        try:
            if not self.grammar_valid_: return BAD
            y_pred = self.execute_3D(X)
            if y_pred is None: return BAD

            sample = y_pred.ravel()[::max(1, y_pred.size // 100)]
            if not np.isfinite(sample).any(): return BAD

            finite = np.isfinite(y_pred)
            n_fin = np.count_nonzero(finite)
            if n_fin == 0 or n_fin < y_pred.size * 0.1: return BAD

            per_day = finite.sum(axis=1).astype(np.int32, copy=False)
            self.pred_finite_ratio_ = float(n_fin) / float(y_pred.size)
            self.pred_active_day_frac_ = float(np.mean(per_day > 0))
            self.pred_median_valid_cnt_ = float(np.median(per_day[per_day > 0])) if np.any(per_day > 0) else 0.0

            scan = y_pred.ravel()[:min(y_pred.size, 5000)]
            valid = scan[np.isfinite(scan)]
            if valid.size > 10:
                if np.std(valid) < 1e-10 or np.allclose(valid, 0, atol=1e-10): return BAD

            if n_fin < y_pred.size:
                if not y_pred.flags.owndata: y_pred = y_pred.copy()
                y_pred[~finite] = np.nan

            raw = self.metric(y, y_pred, sample_weight)
            return raw if np.isfinite(raw) else BAD
        except Exception:
            return BAD
