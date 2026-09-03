# factor_analyse/Alpha101/Alpha48_Factor.py
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

def create_alpha48_factor(corr_window=250, rebalance_period=2, availability_lookback_days=90):
    """
    Alpha 48 因子 (币圈7×24h优化版)
    
    原始定义:
    (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), 
    IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    
    币圈修改版 (移除行业中性化):
    ((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close) / 
    sum(((delta(close, 1) / delay(close, 1))^2), 250)
    
    思路:
    1. 序列相关性: 计算今日收益与昨日收益的250天长期自相关性
    2. 收益率调整: 使用相关性调整今日收益率，并除以价格进行标准化
    3. 波动率标准化: 使用收益率平方和进行标准化
    4. 加密货币特殊处理: 移除行业中性化步骤
    
    - 平均持有期: 1-3天
    """
    print(f"开始构建 Alpha 48 因子...")
    print(f"参数: corr_window={corr_window}, rebalance_period={rebalance_period}")

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
        if len(group) < corr_window + rebalance_period + 5:  # 需要额外的数据来计算滞后项
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['norm_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算因子组件
        
        # 2.1 计算收益率
        gp['delta_close'] = gp['norm_close'].diff()  # 今日收益
        gp['delay_close'] = gp['norm_close'].shift(1)  # 昨日价格
        gp['delta_delay_close'] = gp['delay_close'].diff()  # 昨日收益
        gp['pct_change'] = gp['delta_close'] / (gp['delay_close'] + 1e-8)  # 收益率
        
        # 2.2 计算收益率平方和
        gp['pct_change_squared'] = gp['pct_change'] ** 2
        gp['sum_pct_change_squared'] = gp['pct_change_squared'].rolling(window=corr_window, min_periods=corr_window).sum()
        
        # 2.3 计算收益率序列相关性
        def rolling_correlation(x):
            if len(x) < corr_window or np.isnan(x).any():
                return np.nan
            # 计算两个时间序列的相关性
            return np.corrcoef(x[:, 0], x[:, 1])[0, 1]
        
        # 准备数据用于rolling correlation
        corr_data = pd.DataFrame({
            'delta_close': gp['delta_close'],
            'delta_delay_close': gp['delta_delay_close']
        }).dropna()
        
        if len(corr_data) >= corr_window:
            # 使用rolling apply计算相关性
            corr_values = []
            for i in range(corr_window, len(corr_data) + 1):
                window_data = corr_data.iloc[i-corr_window:i].values
                if len(window_data) == corr_window:
                    try:
                        corr = rolling_correlation(window_data)
                        corr_values.append(corr)
                    except:
                        corr_values.append(np.nan)
                else:
                    corr_values.append(np.nan)
            
            # 将相关性值添加到原始数据中
            corr_series = pd.Series(corr_values, index=corr_data.index[corr_window-1:])
            gp.loc[corr_series.index, 'return_correlation'] = corr_series
        
        # 2.4 计算因子值
        # ((correlation * delta_close) / close) / sum_pct_change_squared
        gp['numerator'] = gp['return_correlation'] * gp['delta_close'] / (gp['norm_close'] + 1e-8)
        gp['alpha48_raw'] = gp['numerator'] / (gp['sum_pct_change_squared'] + 1e-8)
        
        # 3. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")
        
        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)
        
        return gp[['date', 'symbol', 'alpha48_raw', 'future_ret']].dropna()

    print("计算 Alpha 48 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha48_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha48_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha48_return_autocorr_{corr_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_alpha48_factor(corr_window=20, rebalance_period=5)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_alpha48_factor(corr_window=120, rebalance_period=1)
