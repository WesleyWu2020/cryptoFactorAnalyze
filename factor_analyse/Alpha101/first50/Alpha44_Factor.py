# factor_analyse/Alpha101/Alpha44_Factor.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
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

def create_alpha44_factor(corr_window=5, rebalance_period=3, availability_lookback_days=90):
    """
    Alpha 44 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha#44 = (-1 * correlation(high, rank(volume), 5))
    
    思路:
    - 价量关系: 分析最高价与交易量排名的相关性
    - 反向操作: 通过取负值，当正相关性强时给出看跌信号
    - 短期分析: 使用5天窗口捕捉短期关系
    - 信号解读: 当最高价与交易量排名正相关性强时，可能表明追涨情绪过热，预期回调
    - 平均持有期: 2-5天
    """
    print(f"开始构建 Alpha 44 因子...")
    print(f"参数: corr_window={corr_window}, rebalance_period={rebalance_period}")

    # 可用性池
    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(
        historical_df, lookback_days=availability_lookback_days, mode="window"
    )
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # K线数据
    df = load_kline_df()

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < corr_window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['norm_high'] = gp['high'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算交易量的横截面排名
        # 注意：这里我们使用相对于自身历史的交易量排名，而不是横截面排名
        # 因为我们关注的是单个token内部的价量关系
        gp['vol_rank'] = gp['volume'].rolling(window=20, min_periods=5).rank(pct=True)
        
        # 3. 计算最高价与交易量排名的相关性
        # 使用rolling.corr计算滚动相关性
        gp['high_vol_corr'] = gp['norm_high'].rolling(window=corr_window, min_periods=corr_window).corr(gp['vol_rank'])
        
        # 4. 计算因子值：取负相关性
        gp['alpha44_raw'] = -1 * gp['high_vol_corr']
        
        # 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[['date', 'symbol', 'alpha44_raw', 'future_ret']].dropna()

    print("计算 Alpha 44 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha44_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha44_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha44_high_volume_reversal_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数 (短期价量反转因子)
    create_alpha44_factor(corr_window=20, rebalance_period=10)
    
    # 可选：更长的相关性窗口，捕捉更稳定的价量关系
    # create_alpha44_factor(corr_window=10, rebalance_period=5)
