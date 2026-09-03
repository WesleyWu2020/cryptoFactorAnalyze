# factor_analyse/Alpha101/Alpha39_Factor.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 路径以便导入 util_factor
current_dir = os.path.dirname(os.path.abspath(__file__))
factor_mining_dir = os.path.join(current_dir, '..', 'factor_mining')
import sys
sys.path.insert(0, factor_mining_dir)

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

def create_alpha39_factor(delta_window=7, adv_window=20, decay_window=9, long_ret_window=250,
                          rebalance_period=7, availability_lookback_days=90):
    """
    Alpha 39
    ((-1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume/adv20, 9))))) * (1 + rank(sum(returns, 250))))

    思路:
    - 短期动量: delta(close, 7)（用相对价格基准）
    - 交易量确认: volume/adv20 后做线性衰减均值，再做横截面rank并取(1 - rank)
    - 长期收益: sum(returns, 250) 的横截面rank，作为(1 + rank)权重
    - 交互: 短期反向 × 长期权重
    """
    print(f"开始构建 Alpha 39 因子...")
    print(f"参数: delta_window={delta_window}, adv_window={adv_window}, decay_window={decay_window}, "
          f"long_ret_window={long_ret_window}, rebalance_period={rebalance_period}")

    # 可用性池
    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(
        historical_df, lookback_days=availability_lookback_days, mode="window"
    )
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # K线数据
    df = load_kline_df()

    # 线性衰减 rolling（最近权重最大）
    w = np.arange(1, decay_window + 1, dtype=float)
    w = w / w.sum()

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < max(delta_window, adv_window, decay_window, long_ret_window) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # 相对价格基准（缓解量纲差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['norm_open'] = gp['open'] / (gp['price_base'] + 1e-8)
        gp['norm_close'] = gp['close'] / (gp['price_base'] + 1e-8)

        # 短期动量: delta(norm_close, 7)
        gp['delta_short'] = gp['norm_close'] - gp['norm_close'].shift(delta_window)

        # 交易量确认: volume/adv20 → 线性衰减均值
        gp['adv20'] = gp['volume'].rolling(window=adv_window).mean()
        gp['vol_ratio'] = gp['volume'] / (gp['adv20'] + 1e-8)

        def lin_decay(a: pd.Series) -> float:
            if a.isna().any():
                return np.nan
            # a为长度=decay_window的window
            return float(np.dot(a.values, w))

        gp['vol_decay'] = gp['vol_ratio'].rolling(window=decay_window).apply(lin_decay, raw=False)

        # 长期收益: sum(returns, 250) — returns 用 norm_close 的 pct_change
        gp['returns'] = gp['norm_close'].pct_change()
        gp['sum_ret_long'] = gp['returns'].rolling(window=long_ret_window).sum()

        # 未来收益（评估用）
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[['date', 'symbol', 'delta_short', 'vol_decay', 'sum_ret_long', 'future_ret']].dropna()

    print("计算 Alpha 39 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 横截面组合
    def combine_by_date(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        # rank(decay) 与 rank(sum_ret)
        decay_rank = d['vol_decay'].rank(pct=True)
        sumret_rank = d['sum_ret_long'].rank(pct=True)

        # 内部组合: delta_short * (1 - rank(decay))
        inner = d['delta_short'] * (1.0 - decay_rank)

        # -1 * rank(inner)
        short_component = -1.0 * inner.rank(pct=True)

        # (1 + rank(sum_ret_long))
        long_component = 1.0 + sumret_rank

        d['alpha39_raw'] = short_component * long_component
        return d

    factor_df = factor_df.groupby('date', group_keys=False).apply(combine_by_date)
    factor_df = factor_df.dropna(subset=['alpha39_raw'])

    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha39_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha39_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha39_multi_horizon_reversal_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 推荐：短中期结合，略偏中期持有
    create_alpha39_factor(delta_window=10, adv_window=10, decay_window=10, long_ret_window=90, rebalance_period=10)
