"""Single-source feature construction for GP training and evaluation."""
import numpy as np
import pandas as pd


def build_features(data: dict, returns: pd.DataFrame, params: dict):
    """Build 3D feature array: (n_dates, n_features, n_stocks).

    Returns (x_array, feature_names, index_list, col_list).
    """
    index_list = list(returns.index)
    col_list = list(returns.columns)
    nd, ns = len(index_list), len(col_list)

    def _get(key, field_map=None):
        fm = field_map or {}
        k = fm.get(key, key)
        df = data.get(k)
        return df.reindex(index=index_list, columns=col_list) if df is not None else None

    adj_close = _get('adj_closes')
    adj_high = _get('adj_highs')
    adj_low = _get('adj_lows')
    adj_open = _get('adj_opens')
    amount = _get('amounts')
    volumes = _get('volumes')
    turnovers = _get('turnovers')

    if params.get('normalize_inputs', True):
        clip_q = float(params.get('winsor_quantile', 0.01))
        norm_feats = set(params.get('normalize_features', []))

        def _cs_wz(df, do_log=False):
            src = np.log1p(df) if do_log else df.copy()
            lo, hi = src.quantile(clip_q, axis=1), src.quantile(1 - clip_q, axis=1)
            c = src.clip(lower=lo, upper=hi, axis=0)
            m, s = c.mean(axis=1), c.std(axis=1)
            return c.sub(m, axis=0).divide(s.replace(0, np.nan) + 1e-9, axis=0)

        if 'volume' in norm_feats and volumes is not None: volumes = _cs_wz(volumes, True)
        if 'turnover' in norm_feats and turnovers is not None: turnovers = _cs_wz(turnovers, True)
        if 'amount' in norm_feats and amount is not None: amount = _cs_wz(amount, True)

    daily_returns = adj_close.pct_change()
    intraday_range = (adj_high - adj_low) / (adj_close + 1e-8)
    gap = adj_open / adj_close.shift(1) - 1
    close_position = (adj_close - adj_low) / (adj_high - adj_low + 1e-8)
    upper_shadow = (adj_high - np.maximum(adj_open, adj_close)) / (adj_close + 1e-8)
    lower_shadow = (np.minimum(adj_open, adj_close) - adj_low) / (adj_close + 1e-8)

    volume_ratio = None
    if volumes is not None:
        volume_ratio = volumes / (volumes.rolling(20, min_periods=5).mean() + 1e-8)

    ret_10d = adj_close.pct_change(10)
    ret_20d = adj_close.pct_change(20)
    ret_5d = adj_close.pct_change(5)

    vol_ratio_10d = None
    if volumes is not None:
        vol_ratio_10d = volumes / (volumes.rolling(10, min_periods=3).mean() + 1e-8)

    realized_vol = daily_returns.rolling(20, min_periods=5).std()
    vol_ma60 = realized_vol.rolling(60, min_periods=20).mean()
    vol_regime = realized_vol / (vol_ma60 + 1e-8)

    exclude = set(params.get('exclude_features', []))

    names = []
    def _plan(n, ok=True):
        if ok and n not in exclude: names.append(n)

    _plan('returns')
    _plan('intraday_range')
    _plan('gap')
    _plan('close_position')
    _plan('upper_shadow')
    _plan('lower_shadow')
    _plan('volume_ratio', volumes is not None)
    _plan('turnover', turnovers is not None)
    _plan('ret_10d')
    _plan('ret_20d')
    _plan('vol_ratio_10d', volumes is not None)
    _plan('realized_vol')
    _plan('vol_regime')
    _plan('mkt_trend_20d')
    _plan('mkt_vol_20d')
    _plan('mkt_amt_ratio', amount is not None)
    _plan('mkt_breadth')
    _plan('mkt_vol_regime')
    _plan('beta_20d')
    _plan('turnover_stability', turnovers is not None)
    _plan('trend_strength')
    _plan('price_acceleration')
    _plan('downside_vol')

    if not names:
        raise ValueError("No features available")

    x = np.empty((nd, len(names), ns), dtype=np.float32)
    fi = {n: i for i, n in enumerate(names)}

    def _put(name, df):
        i = fi.get(name)
        if i is None: return
        if df is None:
            x[:, i, :] = np.nan; return
        try:
            df = df.reindex(index=index_list, columns=col_list) if hasattr(df, 'reindex') else df
            arr = df.to_numpy(dtype=np.float32, copy=False) if hasattr(df, 'to_numpy') else np.asarray(df, dtype=np.float32)
            x[:, i, :] = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        except Exception:
            x[:, i, :] = np.nan

    _put('returns', daily_returns)
    _put('intraday_range', intraday_range)
    _put('gap', gap)
    _put('close_position', close_position)
    _put('upper_shadow', upper_shadow)
    _put('lower_shadow', lower_shadow)
    _put('volume_ratio', volume_ratio)
    _put('turnover', turnovers)
    _put('ret_10d', ret_10d)
    _put('ret_20d', ret_20d)
    _put('vol_ratio_10d', vol_ratio_10d)
    _put('realized_vol', realized_vol)
    _put('vol_regime', vol_regime)

    mkt_ret = daily_returns.mean(axis=1)
    mkt_ret_20d = mkt_ret.rolling(20, min_periods=5).sum()
    _put('mkt_trend_20d', mkt_ret_20d)

    mkt_vol = realized_vol.mean(axis=1)
    _put('mkt_vol_20d', mkt_vol)

    if amount is not None and 'mkt_amt_ratio' in fi:
        ms = amount.sum(axis=1)
        _put('mkt_amt_ratio', ms / (ms.rolling(20, min_periods=5).mean() + 1e-8))

    breadth = (daily_returns > 0).sum(axis=1) / (daily_returns.notna().sum(axis=1) + 1e-8)
    _put('mkt_breadth', breadth.rolling(5, min_periods=2).mean())

    mvr = mkt_vol / (mkt_vol.rolling(60, min_periods=20).mean() + 1e-8)
    _put('mkt_vol_regime', mvr)

    if 'beta_20d' in fi:
        cov = daily_returns.rolling(20, min_periods=10).cov(mkt_ret)
        var_m = mkt_ret.rolling(20, min_periods=10).var()
        _put('beta_20d', cov.div(var_m + 1e-8, axis=0))

    if turnovers is not None and 'turnover_stability' in fi:
        ts = turnovers.rolling(20, min_periods=5).std()
        tm = turnovers.rolling(20, min_periods=5).mean()
        _put('turnover_stability', tm / (ts + 1e-8))

    if 'trend_strength' in fi:
        h20 = adj_high.rolling(20, min_periods=5).max()
        l20 = adj_low.rolling(20, min_periods=5).min()
        _put('trend_strength', (adj_close - l20) / (h20 - l20 + 1e-8))

    _put('price_acceleration', ret_5d - ret_5d.shift(5))
    _put('downside_vol', daily_returns.clip(upper=0).rolling(20, min_periods=5).std())

    return x, names, index_list, col_list
