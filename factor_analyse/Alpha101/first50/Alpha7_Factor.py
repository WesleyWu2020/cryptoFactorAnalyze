# factor_analyse/Alpha101/first50/Alpha7_Factor.py
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

def create_alpha7_factor(adv_window=20, delta_window=7, ts_rank_window=60, rebalance_period=5, availability_lookback_days=90):
    """
    Alpha 7 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha 7 = ((adv20 < taker_buy_quote) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
    
    思路:
    1. 主动买入突破判断：识别当前主动买入金额是否突破20日均值
    2. 放量时的反转信号：结合价格波动幅度和方向，构建反转信号
    3. 缩量时的默认信号：返回-1（默认看跌）
    
    - 平均持有期: 5-10天
    """
    print(f"开始构建 Alpha 7 因子...")
    print(f"参数: adv_window={adv_window}, delta_window={delta_window}, ts_rank_window={ts_rank_window}, rebalance_period={rebalance_period}")

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
        if len(group) < max(adv_window, delta_window, ts_rank_window) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === Alpha 7 因子计算 ===
        
        # 1. 计算平均主动买入金额 (adv20)
        gp['adv'] = gp['taker_buy_quote'].rolling(window=adv_window, min_periods=adv_window).mean()
        
        # 2. 计算价格变化 (delta(close, 7))
        gp['price_delta'] = gp['close'].diff(delta_window)
        
        # 3. 计算价格变化的绝对值
        gp['price_delta_abs'] = np.abs(gp['price_delta'])
        
        # 4. 计算价格变化的符号
        gp['price_delta_sign'] = np.sign(gp['price_delta'])
        
        # 5. 计算时序排名
        def ts_rank(series):
            if len(series) < ts_rank_window:
                return np.nan
            # 计算当前值在过去ts_rank_window天中的排名百分位
            return series.rank(pct=True).iloc[-1]
        
        gp['price_delta_abs_ts_rank'] = gp['price_delta_abs'].rolling(window=ts_rank_window, min_periods=ts_rank_window).apply(
            ts_rank, raw=False
        )
        
        # 6. 实现Alpha 7的条件逻辑
        # ((adv20 < taker_buy_quote) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        
        # 条件1: 主动买入突破时 (adv < taker_buy_quote)
        condition_buy_spike = gp['adv'] < gp['taker_buy_quote']
        value_buy_spike = (-1 * gp['price_delta_abs_ts_rank']) * gp['price_delta_sign']
        
        # 条件2: 正常主动买入时 (adv >= taker_buy_quote)
        condition_normal_buy = gp['adv'] >= gp['taker_buy_quote']
        value_normal_buy = -1
        
        # 使用numpy.where实现条件逻辑
        gp['alpha7_raw'] = np.where(
            condition_buy_spike,
            value_buy_spike,
            value_normal_buy
        )
        
        # 7. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "alpha7_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha7_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha7_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha7_taker_buy_adv{adv_window}d_delta{delta_window}d_tsrank{ts_rank_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_alpha7_factor(adv_window=14, delta_window=7, ts_rank_window=49, rebalance_period=10)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_alpha7_factor(adv_window=15, delta_window=5, ts_rank_window=40, rebalance_period=3)
