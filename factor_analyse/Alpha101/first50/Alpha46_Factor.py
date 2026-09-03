# factor_analyse/Alpha101/Alpha46_Factor.py
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

def create_alpha46_factor(long_delay=20, mid_delay=10, rebalance_period=7, availability_lookback_days=90):
    """
    Alpha 46 因子 (币圈7×24h优化版)
    
    原始定义:
    ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? 
      (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 
      1 : ((-1 * 1) * (close - delay(close, 1)))))
    
    思路:
    1. 计算两个时间段的日均价格变化率
       - 远期变化率: (20天前到10天前的价格变化)/10天
       - 近期变化率: (10天前到现在的价格变化)/10天
    2. 计算趋势变化 = 远期变化率 - 近期变化率
    3. 三层条件判断:
       - 如果趋势变化 > 0.25 (加速下跌)，返回-1
       - 如果趋势变化 < 0 (加速上涨)，返回1
       - 否则返回昨日收益的负值
    
    - 平均持有期: 5-10天
    """
    print(f"开始构建 Alpha 46 因子...")
    print(f"参数: long_delay={long_delay}, mid_delay={mid_delay}, rebalance_period={rebalance_period}")

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
        if len(group) < long_delay + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['norm_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算因子组件
        
        # 2.1 远期价格变化率: (20天前到10天前的价格变化)/10天
        gp['delay_close_20'] = gp['norm_close'].shift(long_delay)
        gp['delay_close_10'] = gp['norm_close'].shift(mid_delay)
        gp['far_change_rate'] = (gp['delay_close_20'] - gp['delay_close_10']) / (long_delay - mid_delay)
        
        # 2.2 近期价格变化率: (10天前到现在的价格变化)/10天
        gp['recent_change_rate'] = (gp['delay_close_10'] - gp['norm_close']) / mid_delay
        
        # 2.3 趋势变化 = 远期变化率 - 近期变化率
        gp['trend_change'] = gp['far_change_rate'] - gp['recent_change_rate']
        
        # 2.4 昨日收益
        gp['daily_return'] = gp['norm_close'] - gp['norm_close'].shift(1)
        
        # 2.5 条件判断
        gp['alpha46_raw'] = np.nan
        
        # 如果趋势变化 > 0.25 (加速下跌)，返回-1
        gp.loc[gp['trend_change'] > 0.25, 'alpha46_raw'] = -1
        
        # 如果趋势变化 < 0 (加速上涨)，返回1
        gp.loc[gp['trend_change'] < 0, 'alpha46_raw'] = 1
        
        # 否则返回昨日收益的负值
        gp.loc[(gp['trend_change'] <= 0.25) & (gp['trend_change'] >= 0), 'alpha46_raw'] = -1 * gp['daily_return']
        
        # 未来收益（百分比）
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="pct")
        
        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)
        
        return gp[["date", "symbol", "alpha46_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
    
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha46_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha46_raw", out_col="factor")
    
    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha46_trend_accel_{long_delay}d_{mid_delay}d_rebalance{rebalance_period}d_")
    
    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 默认参数
    create_alpha46_factor(long_delay=20, mid_delay=10, rebalance_period=10)
