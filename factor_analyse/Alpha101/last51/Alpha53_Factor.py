# factor_analyse/Alpha101/last51/Alpha53_Factor.py
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

def create_alpha53_factor(delta_window=9, rebalance_period=5, availability_lookback_days=90):
    """
    Alpha 53 因子 (币圈7×24h优化版)
    
    原始定义:
    (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    
    思路:
    1. 日内位置: 衡量收盘价在日内价格区间的相对位置
       - (close - low): 收盘价与最低价的差异，衡量从最低点的涨幅
       - (high - close): 最高价与收盘价的差异，衡量从最高点的跌幅
       - ((close - low) - (high - close)) / (close - low): 标准化处理
    2. 位置变化: 分析该位置的9天变化趋势
    3. 反向操作: 当位置改善时给出看跌信号
    
    当股票收盘价在日内区间的相对位置持续改善时，可能存在过度乐观，预期回调。
    
    - 平均持有期: 3-7天
    """
    print(f"开始构建 Alpha 53 因子...")
    print(f"参数: delta_window={delta_window}, rebalance_period={rebalance_period}")

    # 可用性池
    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(
        historical_df, lookback_days=availability_lookback_days, mode="window"
    )
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # K线数据
    df = load_kline_df()

    # 单symbol计算
    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < delta_window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['norm_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        gp['norm_high'] = gp['high'] / (gp['price_base'] + 1e-8)
        gp['norm_low'] = gp['low'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算因子组件
        
        # 2.1 计算日内价格位置指标
        # (close - low): 收盘价与最低价的差异
        gp['close_low_diff'] = gp['norm_close'] - gp['norm_low']
        
        # (high - close): 最高价与收盘价的差异
        gp['high_close_diff'] = gp['norm_high'] - gp['norm_close']
        
        # ((close - low) - (high - close)) / (close - low): 标准化处理
        # 这个指标衡量收盘价在日内区间的相对位置
        gp['intraday_position'] = (gp['close_low_diff'] - gp['high_close_diff']) / (gp['close_low_diff'] + 1e-8)
        
        # 2.2 计算9天变化
        gp['position_change'] = gp['intraday_position'].diff(delta_window)
        
        # 2.3 计算因子值: -1 * 位置变化
        gp['alpha53_raw'] = -1 * gp['position_change']
        
        # 3. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "alpha53_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha53_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha53_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha53_intraday_position_reversal_{delta_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_alpha53_factor(delta_window=9, rebalance_period=5)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_alpha53_factor(delta_window=5, rebalance_period=3)
