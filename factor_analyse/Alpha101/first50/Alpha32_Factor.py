# factor_analyse/Alpha101/first50/Alpha32_Factor.py
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

def create_alpha32_factor(short_mean_window=7, corr_window=30, delay_lag=5, rebalance_period=7, availability_lookback_days=90):
    """
    Alpha 32 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha#32 = scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(taker_buy_price, delay(close, 5), 30)))
    
    核心步骤:
    1. 短期均值回归：当收盘价偏离7天均价时，预期回归
    2. 长期价格关系：分析主买价与历史价格(5天滞后)的长期关系
    3. 权重分配：给长期关系分析更高权重（20倍）
    4. 标准化处理：通过cross-sectional scale确保两部分可比
    
    - 平均持有期: 7-15天
    """
    print(f"开始构建 Alpha 32 因子...")
    print(f"参数: short_mean_window={short_mean_window}, corr_window={corr_window}, delay_lag={delay_lag}, rebalance_period={rebalance_period}")

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
        if len(group) < max(short_mean_window, corr_window + delay_lag) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === Alpha 32 因子计算 ===
        
        # 1. 计算主买价 (taker_buy_quote / taker_buy_base)
        gp['taker_buy_price'] = gp['taker_buy_quote'] / (gp['taker_buy_base'] + 1e-8)
        
        # 2. 均值回归项: (MA(close, N) - close)
        gp['close_ma'] = gp['close'].rolling(window=short_mean_window, min_periods=short_mean_window).mean()
        gp['mean_revert_raw'] = gp['close_ma'] - gp['close']
        
        # 3. 长期相关性: corr(taker_buy_price, delay(close, D), W)
        gp['close_delay'] = gp['close'].shift(delay_lag)
        
        def rolling_correlation(x, y, window):
            """计算滚动相关性"""
            if len(x) < window:
                return pd.Series([np.nan] * len(x), index=x.index)
            
            corr_values = []
            for i in range(len(x)):
                if i < window - 1:
                    corr_values.append(np.nan)
                else:
                    x_window = x.iloc[i-window+1:i+1]
                    y_window = y.iloc[i-window+1:i+1]
                    
                    if len(x_window) == window and len(y_window) == window:
                        if x_window.std() > 1e-10 and y_window.std() > 1e-10:
                            corr = x_window.corr(y_window)
                            corr_values.append(corr if not np.isnan(corr) else 0)
                        else:
                            corr_values.append(0)
                    else:
                        corr_values.append(np.nan)
            
            return pd.Series(corr_values, index=x.index)
        
        gp['corr_long_raw'] = rolling_correlation(
            gp['taker_buy_price'], 
            gp['close_delay'], 
            corr_window
        )
        
        # 4. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "mean_revert_raw", "corr_long_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    # 合并所有token的数据
    all_data = pd.concat(result_dfs, ignore_index=True)
    
    # 按日期分组计算横截面scale与组合
    print("进行横截面scale与组合...")
    def combine_per_date(df_date):
        df_date = df_date.copy()
        
        # cross-sectional scale: x / sum(|x|)
        def scale_cross_section(s):
            s = s.replace([np.inf, -np.inf], np.nan)
            denom = np.nansum(np.abs(s.values))
            if denom is None or np.isnan(denom) or denom == 0:
                return pd.Series(0.0, index=s.index)
            return s / denom
        
        df_date['s1'] = scale_cross_section(df_date['mean_revert_raw'])
        df_date['s2'] = scale_cross_section(df_date['corr_long_raw'])
        df_date['alpha32_raw'] = df_date['s1'] + 20.0 * df_date['s2']
        
        return df_date
    
    factor_df = all_data.groupby('date', group_keys=False).apply(combine_per_date)
    factor_df = factor_df.dropna(subset=['alpha32_raw'])
    
    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha32_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha32_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha32_taker_buy_corr{corr_window}d_delay{delay_lag}_mean{short_mean_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 推荐默认：7日均值、30日相关性窗口、5日滞后、7日调仓
    create_alpha32_factor(short_mean_window=7, corr_window=30, delay_lag=8, rebalance_period=10)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_alpha32_factor(short_mean_window=5, corr_window=20, delay_lag=5, rebalance_period=5)
