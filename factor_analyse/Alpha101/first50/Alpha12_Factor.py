# factor_analyse/Alpha101/first50/Alpha12_Factor.py
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

def create_alpha12_factor(delta_window=1, rebalance_period=2, availability_lookback_days=90):
    """
    Alpha 12 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha 12 = (sign(delta(taker_buy_quote, 1)) * (-1 * delta(close, 1)))
    
    核心步骤:
    1. 主动买入金额变化方向：计算主动买入金额1日变化的符号
    2. 价格变化：计算收盘价1日变化
    3. 混合策略：
       - 主动买入增加时：采用反转策略（价格涨则看跌，价格跌则看涨）
       - 主动买入减少时：采用动量策略（价格涨则看涨，价格跌则看跌）
    
    - 平均持有期: 2-5天
    """
    print(f"开始构建 Alpha 12 因子...")
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

        # === Alpha 12 因子计算 ===
        
        # 1. 计算主动买入金额变化 (delta(taker_buy_quote, 1))
        gp['taker_buy_delta'] = gp['taker_buy_quote'] - gp['taker_buy_quote'].shift(delta_window)
        
        # 2. 计算价格变化 (delta(close, 1))
        gp['price_delta'] = gp['close'] - gp['close'].shift(delta_window)
        
        # 3. 计算主动买入金额变化符号 (sign(delta(taker_buy_quote, 1)))
        gp['taker_buy_sign'] = np.sign(gp['taker_buy_delta'])
        
        # 4. 计算 Alpha 12 因子: (sign(delta(taker_buy_quote, 1)) * (-1 * delta(close, 1)))
        gp['alpha12_raw'] = gp['taker_buy_sign'] * (-1 * gp['price_delta'])
        
        # 5. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "alpha12_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha12_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha12_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha12_taker_buy_price_mixed_delta{delta_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_alpha12_factor(delta_window=30, rebalance_period=10)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_alpha12_factor(delta_window=1, rebalance_period=1)
