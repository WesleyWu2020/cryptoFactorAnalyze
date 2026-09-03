# factor_analyse/Alpha101/first50/Alpha35_Factor.py
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

def create_alpha35_factor(vol_window=32, price_window=16, ret_window=32, rebalance_period=7, availability_lookback_days=90):
    """
    Alpha 35 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha 35 = ((Ts_Rank(taker_buy_quote, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
    
    思路:
    1. 主动买入金额时序排名：Ts_Rank(taker_buy_quote, 32)
    2. 价格强度时序排名：Ts_Rank(((close + high) - low), 16)
    3. 收益率时序排名：Ts_Rank(returns, 32)
    4. 组合逻辑：买入强度 × (1 - 价格强度排名) × (1 - 收益率排名)
    
    - 平均持有期: 7-15天
    """
    print(f"开始构建 Alpha 35 因子...")
    print(f"参数: vol_window={vol_window}, price_window={price_window}, ret_window={ret_window}, rebalance_period={rebalance_period}")

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
        if len(group) < max(vol_window, price_window, ret_window) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === Alpha 35 因子计算 ===
        
        # 1. 收益率
        gp['returns'] = gp['close'].pct_change()
        
        # 2. 价格强度（无量纲化）
        # ((close + high) - low) / close
        gp['price_strength_rel'] = ((gp['close'] + gp['high']) - gp['low']) / (gp['close'] + 1e-8)
        gp['price_strength_rel'] = gp['price_strength_rel'].replace([np.inf, -np.inf], np.nan)
        
        # 3. 时序排名计算
        def ts_rank(series):
            if len(series) < vol_window:
                return np.nan
            # 计算当前值在过去窗口中的排名百分位
            return series.rank(pct=True).iloc[-1]
        
        # 主动买入金额时序排名
        gp['tsrank_taker_buy'] = gp['taker_buy_quote'].rolling(window=vol_window, min_periods=vol_window).apply(
            ts_rank, raw=False
        )
        
        # 价格强度时序排名
        gp['tsrank_price'] = gp['price_strength_rel'].rolling(window=price_window, min_periods=price_window).apply(
            ts_rank, raw=False
        )
        
        # 收益率时序排名
        gp['tsrank_ret'] = gp['returns'].rolling(window=ret_window, min_periods=ret_window).apply(
            ts_rank, raw=False
        )
        
        # 4. 组合因子值
        gp['alpha35_raw'] = (gp['tsrank_taker_buy'] * (1 - gp['tsrank_price'])) * (1 - gp['tsrank_ret'])
        
        # 5. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "alpha35_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha35_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha35_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha35_taker_buy_tsrank_vol{vol_window}d_price{price_window}d_ret{ret_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_alpha35_factor(vol_window=30, price_window=10, ret_window=10, rebalance_period=7)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_alpha35_factor(vol_window=20, price_window=7, ret_window=7, rebalance_period=5)
