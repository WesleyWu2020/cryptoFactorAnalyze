# factor_analyse/Alpha101/first50/Alpha6_Factor.py
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

def create_alpha6_factor(correlation_window=10, rebalance_period=3, availability_lookback_days=90):
    """
    Alpha 6 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha 6 = (-1 * correlation(high, taker_buy_quote, 10))
    
    思路:
    1. 价量关系：分析最高价与主动买入金额的相关性
    2. 信号反转：负相关表示价量背离，可能存在反转机会
    3. 主动买入：使用taker_buy_quote反映真实的买入压力
    
    - 平均持有期: 3-7天
    """
    print(f"开始构建 Alpha 6 因子...")
    print(f"参数: correlation_window={correlation_window}, rebalance_period={rebalance_period}")

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
        if len(group) < correlation_window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 计算因子组件
        # 计算 (high - close) 与 taker_buy_quote 的相关性
        gp['high_close_diff'] = gp['high'] - gp['close']  # 最高价与收盘价的差异
        gp['taker_buy_quote'] = gp['taker_buy_quote']
        
        # 2. 计算滚动相关性
        def rolling_correlation(x, y, window):
            """计算滚动相关性"""
            if len(x) < window:
                return pd.Series([np.nan] * len(x), index=x.index)
            
            corr_values = []
            for i in range(len(x)):
                if i < window - 1:
                    corr_values.append(np.nan)
                else:
                    # 取过去window天的数据
                    x_window = x.iloc[i-window+1:i+1]
                    y_window = y.iloc[i-window+1:i+1]
                    
                    # 计算相关系数
                    if len(x_window) == window and len(y_window) == window:
                        # 检查数据是否有变化（避免常数列）
                        if x_window.std() > 1e-10 and y_window.std() > 1e-10:
                            corr = x_window.corr(y_window)
                            corr_values.append(corr if not np.isnan(corr) else 0)
                        else:
                            corr_values.append(0)  # 如果数据无变化，相关性为0
                    else:
                        corr_values.append(np.nan)
            
            return pd.Series(corr_values, index=x.index)
        
        # 计算过去correlation_window天的相关性
        correlation = rolling_correlation(
            gp['high_close_diff'],  # (high - close)
            gp['taker_buy_quote'], 
            correlation_window
        )
        
        # 3. 计算Alpha 6因子值: (-1 * correlation)
        gp['alpha6_raw'] = -1 * correlation
        
        # 4. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "alpha6_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha6_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha6_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha6_high_close_taker_buy_corr_{correlation_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_alpha6_factor(correlation_window=20, rebalance_period=5)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_alpha6_factor(correlation_window=5, rebalance_period=2)
