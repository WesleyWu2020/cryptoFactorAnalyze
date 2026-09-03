# factor_analyse/Alpha101/Alpha42_Factor.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

# 路径以便导入 util_factor
current_dir = os.path.dirname(os.path.abspath(__file__))
factor_mining_dir = os.path.join(current_dir, '../../', 'factor_mining')
import sys
sys.path.insert(0, factor_mining_dir)

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

def create_alpha42_factor(vwap_window=20, rebalance_period=3, availability_lookback_days=90):
    """
    Alpha 42 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha#42 = (rank((vwap - close)) / rank((vwap + close)))
    
    思路:
    - 价格偏离: VWAP-close衡量收盘价相对于平均交易价格的偏离
    - 价格水平: VWAP+close衡量整体价格水平
    - 相对分析: 通过排名比值进行相对分析
    - 信号解读: 当收盘价显著低于VWAP（分子大）且整体价格水平不高（分母小）时，因子值较大，预期价格上涨
    - 平均持有期: 1-3天
    """
    print(f"开始构建 Alpha 42 因子...")
    print(f"参数: vwap_window={vwap_window}, rebalance_period={rebalance_period}")

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
        if len(group) < vwap_window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['norm_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算VWAP (简化版，使用加权移动平均)
        # 注意：真实VWAP需要分钟级数据，这里使用成交量加权的收盘价作为近似
        gp['vol_price'] = gp['close'] * gp['volume']
        gp['vol_sum'] = gp['volume'].rolling(window=vwap_window).sum()
        gp['vwap'] = gp['vol_price'].rolling(window=vwap_window).sum() / (gp['vol_sum'] + 1e-8)
        gp['norm_vwap'] = gp['vwap'] / (gp['price_base'] + 1e-8)
        
        # 3. 计算因子组件
        gp['vwap_minus_close'] = gp['norm_vwap'] - gp['norm_close']  # 价格偏离
        gp['vwap_plus_close'] = gp['norm_vwap'] + gp['norm_close']   # 价格水平
        
        # 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[['date', 'symbol', 'vwap_minus_close', 'vwap_plus_close', 'future_ret']].dropna()

    print("计算 Alpha 42 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    # 按日期计算排名比值
    def calculate_alpha42_by_date(df_date):
        df_date = df_date.copy()
        
        # 对组件进行排名
        df_date['rank_vwap_minus_close'] = df_date['vwap_minus_close'].rank(pct=True)
        df_date['rank_vwap_plus_close'] = df_date['vwap_plus_close'].rank(pct=True)
        
        # 计算排名比值 (避免除零)
        df_date['alpha42_raw'] = df_date['rank_vwap_minus_close'] / (df_date['rank_vwap_plus_close'] + 1e-8)
        
        return df_date

    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_alpha42_by_date)
    factor_df = factor_df.dropna(subset=['alpha42_raw'])
    
    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha42_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha42_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha42_vwap_ratio_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数 (短期价格偏离因子)
    create_alpha42_factor(vwap_window=20, rebalance_period=10)
    
    # 可选：更长的VWAP窗口，更稳定的价格偏离信号
    # create_alpha42_factor(vwap_window=30, rebalance_period=5)
