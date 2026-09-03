# factor_analyse/Alpha101/Alpha45_Factor.py
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

def create_alpha45_factor(delay_days=5, mid_window=20, short_corr_window=2, 
                         short_sum_window=5, rebalance_period=5, availability_lookback_days=90):
    """
    Alpha 45 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha#45 = (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * 
               rank(correlation(sum(close, 5), sum(close, 20), 2))))
    
    思路:
    1. 中期价格趋势基准: 从5天前开始的20天收盘价均值，横截面排名
    2. 短期量价相关性: 过去2天收盘价与交易量的相关性
    3. 多周期价格相关性: 5天与20天收盘价总和的相关性，横截面排名
    4. 信号合成: 三者相乘，取负值
    
    - 平均持有期: 3-7天
    """
    print(f"开始构建 Alpha 45 因子...")
    print(f"参数: delay_days={delay_days}, mid_window={mid_window}, short_corr_window={short_corr_window}, "
          f"short_sum_window={short_sum_window}, rebalance_period={rebalance_period}")

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
        if len(group) < max(delay_days + mid_window, short_corr_window, short_sum_window + mid_window) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['norm_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算因子组件
        
        # 2.1 中期价格趋势基准: 从5天前开始的20天收盘价均值
        gp['delay_close'] = gp['norm_close'].shift(delay_days)
        gp['mid_term_avg'] = gp['delay_close'].rolling(window=mid_window, min_periods=mid_window).mean()
        
        # 2.2 短期量价相关性: 过去2天收盘价与交易量的相关性
        gp['vol_norm'] = gp['volume'] / gp['volume'].rolling(window=20, min_periods=1).mean()
        gp['price_vol_corr'] = gp['norm_close'].rolling(window=short_corr_window, min_periods=short_corr_window).corr(gp['vol_norm'])
        
        # 2.3 多周期价格相关性: 5天与20天收盘价总和的相关性
        # 计算短期和中期收盘价总和
        gp['short_sum'] = gp['norm_close'].rolling(window=short_sum_window, min_periods=short_sum_window).sum()
        gp['mid_sum'] = gp['norm_close'].rolling(window=mid_window, min_periods=mid_window).sum()
        
        # 计算两个周期总和的相关性
        # 这里需要特殊处理，因为我们需要计算两个时间序列的相关性
        # 使用rolling apply来实现
        def rolling_corr(x):
            if len(x) < short_corr_window or x.isna().any():
                return np.nan
            # 取最后short_corr_window个值计算相关性
            return np.corrcoef(x[-short_corr_window:, 0], x[-short_corr_window:, 1])[0, 1]
        
        # 准备数据用于rolling apply
        corr_data = pd.DataFrame({
            'short': gp['short_sum'],
            'mid': gp['mid_sum']
        }).dropna()
        
        if len(corr_data) >= short_corr_window:
            # 计算相关性
            corr_values = []
            for i in range(short_corr_window, len(corr_data) + 1):
                window_data = corr_data.iloc[i-short_corr_window:i]
                if len(window_data) == short_corr_window:
                    try:
                        corr = np.corrcoef(window_data['short'], window_data['mid'])[0, 1]
                        corr_values.append(corr)
                    except:
                        corr_values.append(np.nan)
                else:
                    corr_values.append(np.nan)
            
            # 将相关性结果添加回原始数据
            corr_index = corr_data.index[short_corr_window-1:]
            if len(corr_index) == len(corr_values):
                corr_series = pd.Series(corr_values, index=corr_index)
                gp.loc[corr_series.index, 'period_corr'] = corr_series
        
        # 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[['date', 'symbol', 'mid_term_avg', 'price_vol_corr', 'period_corr', 'future_ret']].dropna()

    print("计算 Alpha 45 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    # 按日期计算横截面排名和因子值
    def calculate_alpha45_by_date(df_date):
        df_date = df_date.copy()
        
        # 对中期价格趋势和多周期相关性进行横截面排名
        df_date['rank_mid_term_avg'] = df_date['mid_term_avg'].rank(pct=True)
        df_date['rank_period_corr'] = df_date['period_corr'].rank(pct=True)
        
        # 计算因子值：三者相乘，取负值
        df_date['alpha45_raw'] = -1 * (
            df_date['rank_mid_term_avg'] * 
            df_date['price_vol_corr'] * 
            df_date['rank_period_corr']
        )
        
        return df_date

    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_alpha45_by_date)
    factor_df = factor_df.dropna(subset=['alpha45_raw'])
    
    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha45_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha45_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha45_multi_dimension_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数 (多维度因子)
    create_alpha45_factor(delay_days=5, mid_window=20, short_corr_window=2, 
                         short_sum_window=5, rebalance_period=5)
    
    # 可选：更短的窗口，适应加密货币高波动特性
    # create_alpha45_factor(delay_days=3, mid_window=10, short_corr_window=2, 
    #                      short_sum_window=3, rebalance_period=3)
