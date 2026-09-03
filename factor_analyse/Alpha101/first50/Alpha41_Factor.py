# factor_analyse/Alpha101/first50/Alpha41_Factor.py
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

def create_alpha41_factor(vwap_window=20, rebalance_period=3, availability_lookback_days=90):
    """
    Alpha 41 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha#41 = (((high * low)^0.5) - vwap)
    
    思路:
    - 几何平均价格: (high * low)^0.5，代表理论中点价格
    - 实际交易价格: VWAP代表实际交易的平均价格
    - 偏离分析: 当几何平均价格高于VWAP时，可能表明交易偏向低价区间
    - 平均持有期: 1-3天
    """
    print(f"开始构建 Alpha 41 因子...")
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
        gp['norm_high'] = gp['high'] / (gp['price_base'] + 1e-8)
        gp['norm_low'] = gp['low'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算几何平均价格 (high * low)^0.5
        gp['geo_mean'] = np.sqrt(gp['norm_high'] * gp['norm_low'])
        
        # 3. 计算主买价 ( taker_buy_quote / taker_buy_base)
        # 主买量/主买金额 = 主买价，反映主动买入的平均价格
        gp['taker_buy_price'] = gp['taker_buy_quote'] / (gp['taker_buy_base'] + 1e-8) 
        
        # 4. 计算主买价的移动平均（替代原来的VWAP）
        gp['taker_buy_ma'] = gp['taker_buy_price'].rolling(window=vwap_window, min_periods=1).mean()
        gp['norm_taker_buy_ma'] = gp['taker_buy_ma'] / (gp['price_base'] + 1e-8)
        
        # 5. 计算因子值: (几何平均 - 主买价移动平均)
        gp['alpha41_raw'] = gp['geo_mean'] - gp['norm_taker_buy_ma']
        
        # 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[['date', 'symbol', 'alpha41_raw', 'future_ret']].dropna()

    print("计算 Alpha 41 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha41_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha41_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha41_geometric_price_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数 (短期价格偏离因子)
    create_alpha41_factor(vwap_window=20, rebalance_period=1)
    
    # 可选：更短的VWAP窗口，更敏感地捕捉价格偏离
    # create_alpha41_factor(vwap_window=5, rebalance_period=1)
