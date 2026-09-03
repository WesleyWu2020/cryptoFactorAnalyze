# factor_analyse/Alpha101/first50/Alpha5_Factor.py
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

def create_alpha5_factor(vwap_window=10, rebalance_period=3, availability_lookback_days=90):
    """
    Alpha 5 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha 5 = (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    
    思路:
    1. 中期趋势偏离：开盘价相对于过去N天VWAP均值的偏离程度
    2. 日内行为异常：收盘价与当日VWAP的偏离程度
    3. 复合信号：结合趋势偏离和异常行为，识别潜在的反转或延续机会
    
    - 平均持有期: 3-7天
    """
    print(f"开始构建 Alpha 5 因子...")
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

    # 单symbol计算
    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < vwap_window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 计算VWAP (成交量加权平均价格)
        gp['vwap'] = gp['taker_buy_quote'] / (gp['taker_buy_base'] + 1e-8)
        
        # 2. 计算中期VWAP均值
        gp['vwap_ma'] = gp['vwap'].rolling(window=vwap_window, min_periods=vwap_window).mean()
        
        # 3. 计算因子组件
        # 第一部分：开盘价相对于中期趋势的偏离
        gp['trend_deviation'] = gp['open'] - gp['vwap_ma']
        
        # 第二部分：收盘价与当日VWAP的偏离
        gp['intraday_deviation'] = gp['close'] - gp['vwap']
        
        # 4. 计算Alpha 5因子值
        # 这里先计算基础组件，后续进行横截面排名
        gp['alpha5_raw'] = gp['trend_deviation'] * gp['intraday_deviation']
        
        # 5. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "alpha5_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha5_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha5_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha5_relative_price_vwap{vwap_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_alpha5_factor(vwap_window=10, rebalance_period=3)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_alpha5_factor(vwap_window=5, rebalance_period=2)
