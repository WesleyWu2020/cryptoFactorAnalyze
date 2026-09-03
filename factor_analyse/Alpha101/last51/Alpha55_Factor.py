# factor_analyse/Alpha101/last51/Alpha55_Factor.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

# 路径以便导入 util_factor
current_dir = os.path.dirname(os.path.abspath(__file__))
factor_mining_dir = os.path.join(current_dir, '..', '..', 'factor_mining')
import sys
sys.path.insert(0, factor_mining_dir)

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

def create_alpha55_factor(pos_window=12, corr_window=6, rebalance_period=5, availability_lookback_days=90):
    """
    Alpha 55 因子 (币圈7×24h优化版)
    原始定义:
      (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))),
                        rank(volume), 6))
    步骤:
      - 计算12日相对位置 pos = (close - minLow12) / (maxHigh12 - minLow12)
      - 按日期横截面 rank(pos) 与 rank(volume)
      - 对于每个symbol，计算6日滚动相关性，并取负值
    """
    print(f"开始构建 Alpha 55 因子...")
    print(f"参数: pos_window={pos_window}, corr_window={corr_window}, rebalance_period={rebalance_period}")

    # 可用性池（避免未来函数）
    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(
        historical_df, lookback_days=availability_lookback_days, mode="window"
    )
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # K线数据
    df = load_kline_df()

    # 先在单symbol内计算 pos 与 future_ret，再做可用性过滤
    def compute_base(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < max(pos_window, corr_window) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # 12日区间位置
        min_low = gp['low'].rolling(window=pos_window, min_periods=pos_window).min()
        max_high = gp['high'].rolling(window=pos_window, min_periods=pos_window).max()
        gp['pos'] = (gp['close'] - min_low) / (max_high - min_low + 1e-8)

        # 未来收益（百分比）
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[['date', 'symbol', 'pos', 'volume', 'close', 'future_ret']].dropna()

    print("计算基础组件（pos、volume、future_ret）并进行可用性过滤...")
    base_parts = group_apply_with_progress(df, 'symbol', compute_base)
    if not base_parts:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    base_df = pd.concat(base_parts, ignore_index=True)

    # 横截面排名（按日期）
    print("按日期进行横截面排名...")
    def rank_by_date(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g['pos_rank'] = g['pos'].rank(pct=True)
        g['vol_rank'] = g['volume'].rank(pct=True)
        return g

    ranked_df = base_df.groupby('date', group_keys=False).apply(rank_by_date)

    # 每个symbol内计算滚动相关性并取负
    print("按symbol计算滚动相关性并取负...")
    def calc_corr(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) < corr_window:
            return pd.DataFrame()
        g = group.sort_values('date').copy()
        g['corr'] = g['pos_rank'].rolling(window=corr_window, min_periods=corr_window).corr(g['vol_rank'])
        g['alpha55_raw'] = -1.0 * g['corr']
        return g[['date', 'symbol', 'alpha55_raw', 'future_ret']].dropna()

    corr_parts = group_apply_with_progress(ranked_df, 'symbol', calc_corr)
    if not corr_parts:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
    factor_df = pd.concat(corr_parts, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col='alpha55_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha55_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha55_pos_vol_corr_{pos_window}d_{corr_window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    create_alpha55_factor(pos_window=12, corr_window=6, rebalance_period=5)
