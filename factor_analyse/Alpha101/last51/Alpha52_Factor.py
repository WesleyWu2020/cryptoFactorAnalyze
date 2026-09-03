# factor_analyse/Alpha101/last51/Alpha52_Factor.py
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

def create_alpha52_factor(ts_min_window=5, delay_days=5, long_window=120, short_window=20, 
                         volume_ts_rank_window=5, rebalance_period=15, availability_lookback_days=90):
    """
    Alpha 52 因子 (币圈7×24h优化版)
    
    原始定义:
    ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * 
     ts_rank(volume, 5))
    
    思路:
    1. 技术面反转信号: (-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)
       - 当前5日最低价与5天前5日最低价的差异
       - 正值表示支撑位下移，负值表示支撑位上移
    2. 基本面分析: rank(((sum(returns, 120) - sum(returns, 20)) / 100))
       - 长期与短期收益率的差异排名
       - 反映长期收益表现相对强度
    3. 交易量确认: ts_rank(volume, 5)
       - 过去5天交易量的时间序列排名
    4. 综合逻辑: 三部分相乘，寻找技术面反转+基本面良好+交易量支持的组合
    
    - 平均持有期: 10-20天
    """
    print(f"开始构建 Alpha 52 因子...")
    print(f"参数: ts_min_window={ts_min_window}, delay_days={delay_days}, long_window={long_window}, "
          f"short_window={short_window}, volume_ts_rank_window={volume_ts_rank_window}, rebalance_period={rebalance_period}")

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
        if len(group) < max(ts_min_window + delay_days, long_window, volume_ts_rank_window) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['norm_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        gp['norm_low'] = gp['low'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算因子组件
        
        # 2.1 技术面反转信号: (-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)
        gp['ts_min_low_5'] = gp['norm_low'].rolling(window=ts_min_window, min_periods=ts_min_window).min()
        gp['delay_ts_min_low_5'] = gp['ts_min_low_5'].shift(delay_days)
        gp['technical_signal'] = (-1 * gp['ts_min_low_5']) + gp['delay_ts_min_low_5']
        
        # 2.2 基本面分析: (sum(returns, 120) - sum(returns, 20)) / 100
        gp['returns'] = gp['norm_close'].pct_change()
        gp['long_sum_returns'] = gp['returns'].rolling(window=long_window, min_periods=long_window).sum()
        gp['short_sum_returns'] = gp['returns'].rolling(window=short_window, min_periods=short_window).sum()
        gp['fundamental_signal'] = (gp['long_sum_returns'] - gp['short_sum_returns']) / (long_window - short_window)
        
        # 2.3 交易量时间序列排名
        def ts_rank(series):
            if len(series) < volume_ts_rank_window:
                return np.nan
            # 对序列进行排名，返回最后一个元素的排名（百分比）
            return series.rank(pct=True).iloc[-1]
        
        gp['volume_ts_rank'] = gp['volume'].rolling(window=volume_ts_rank_window, min_periods=volume_ts_rank_window).apply(
            ts_rank, raw=False
        )
        
        # 2.4 计算因子值: 三部分相乘
        gp['alpha52_raw'] = gp['technical_signal'] * gp['fundamental_signal'] * gp['volume_ts_rank']
        
        # 3. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "alpha52_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha52_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha52_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha52_technical_fundamental_volume_{ts_min_window}d_{long_window}d_{short_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数 (调整窗口大小以适应加密货币数据)
    create_alpha52_factor(ts_min_window=7, delay_days=7, long_window=140, short_window=21, 
                         volume_ts_rank_window=14, rebalance_period=7)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_alpha52_factor(ts_min_window=3, delay_days=3, long_window=60, short_window=10, 
    #                      volume_ts_rank_window=3, rebalance_period=10)
