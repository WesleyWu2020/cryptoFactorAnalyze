# factor_analyse/Alpha101/Alpha50_Factor.py
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

def create_alpha50_factor(corr_window=5, ts_max_window=5, rebalance_period=3, availability_lookback_days=90):
    """
    Alpha 50 因子 (币圈7×24h优化版)
    
    原始定义:
    (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    
    思路:
    1. 排名相关性: 分析交易量排名与VWAP排名的关系
    2. 峰值识别: 取5天内相关性排名的最大值
    3. 反向操作: 当相关性排名达到峰值时给出看跌信号
    
    当交易量排名与VWAP排名的相关性在市场中达到相对高点时，可能表明市场过热。
    
    - 平均持有期: 2-5天
    """
    print(f"开始构建 Alpha 50 因子...")
    print(f"参数: corr_window={corr_window}, ts_max_window={ts_max_window}, rebalance_period={rebalance_period}")

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
        # 降低数据长度要求，适应加密货币数据特点
        min_required = max(corr_window, ts_max_window) + rebalance_period + 2
        if len(group) < min_required:
            return pd.DataFrame()
        
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['norm_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算VWAP (如果没有，用(high+low+close)/3近似)
        if 'vwap' not in gp.columns:
            gp['vwap'] = (gp['high'] + gp['low'] + gp['close']) / 3
        gp['norm_vwap'] = gp['vwap'] / (gp['price_base'] + 1e-8)
        
        # 3. 计算因子组件
        
        # 3.1 计算交易量和VWAP的排名
        # 这里先用NaN占位，后面按日期分组计算横截面排名
        gp['volume_rank'] = np.nan
        gp['vwap_rank'] = np.nan
        
        # 3.2 计算相关性
        gp['volume_vwap_corr'] = np.nan
        
        # 3.3 计算相关性排名
        gp['corr_rank'] = np.nan
        
        # 3.4 计算时间序列最大值
        gp['ts_max_corr_rank'] = np.nan
        
        # 4. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")
        
        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)
        
        return gp[['date', 'symbol', 'volume', 'norm_vwap', 'volume_rank', 'vwap_rank', 
                   'volume_vwap_corr', 'corr_rank', 'ts_max_corr_rank', 'future_ret']].dropna()

    print("计算 Alpha 50 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    # 合并所有token的数据
    all_data = pd.concat(result_dfs, ignore_index=True)
    
    # 按日期分组计算横截面排名和相关性
    print("计算横截面排名和相关性组件...")
    all_data_by_date = all_data.groupby('date')
    
    processed_dfs = []
    for date, group in tqdm(all_data_by_date):
        if len(group) < 2:  # 至少需要2个token才能计算有意义的排名
            continue
            
        # 计算横截面排名
        group = group.copy()
        group['volume_rank'] = group['volume'].rank(pct=True)
        group['vwap_rank'] = group['norm_vwap'].rank(pct=True)
        
        # 计算相关性排名
        group['corr_rank'] = group['volume_vwap_corr'].rank(pct=True)
        
        processed_dfs.append(group)
    
    if not processed_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
    
    # 重新按symbol分组计算时间序列最大值
    print("计算时间序列最大值...")
    all_data_processed = pd.concat(processed_dfs, ignore_index=True)
    
    def compute_ts_max(group):
        if len(group) < ts_max_window:
            return group
        
        group = group.copy()
        # 计算相关性排名的时间序列最大值
        group['ts_max_corr_rank'] = group['corr_rank'].rolling(window=ts_max_window, min_periods=ts_max_window).max()
        
        # 计算因子值: -1 * ts_max_corr_rank
        group['alpha50_raw'] = -1 * group['ts_max_corr_rank']
        
        return group
    
    final_dfs = group_apply_with_progress(all_data_processed, 'symbol', compute_ts_max)
    if not final_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
    
    factor_df = pd.concat(final_dfs, ignore_index=True)
    
    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha50_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha50_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha50_volume_vwap_corr_peak_{corr_window}d_{ts_max_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 建议使用更小的参数，适应加密货币数据特点
    create_alpha50_factor(corr_window=10, ts_max_window=10, rebalance_period=5)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_alpha50_factor(corr_window=5, ts_max_window=5, rebalance_period=3)
