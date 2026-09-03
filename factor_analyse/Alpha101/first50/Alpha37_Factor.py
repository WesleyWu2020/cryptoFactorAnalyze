# factor_analyse/Alpha101/Alpha37_Factor.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加 factor_mining 目录到 Python 路径
factor_mining_dir = os.path.join(current_dir, '..', 'factor_mining')
import sys
sys.path.insert(0, factor_mining_dir)

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample
)

def create_alpha37_factor(corr_window=200, delay_lag=1, rebalance_period=5, availability_lookback_days=90):
    """
    Alpha 37 因子 (币圈7×24h优化版)
    严格避免未来函数，只使用上一期已出现的token
    
    原始定义:
    (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    
    核心步骤:
    1. 历史模式分析：通过200天相关性分析历史日内模式与价格的关系
    2. 当前表现：直接评估当日日内表现
    3. 模式延续：假设历史模式对当前有预测价值
    4. 综合评估：结合历史模式和当前表现
    
    币圈7×24h特殊设计:
    1. 相对价格归一化：适应币圈巨大的价格差异
    2. 日内模式分析：适应7×24h连续交易特性
    3. 历史相关性：捕捉长期日内模式与价格关系
    4. 🔥 关键：结合历史市值排名，确保只使用上一期已出现的token
    
    该因子适用于捕捉日内模式的延续，平均持有期约3-7天
    """
    print(f"开始构建 Alpha 37 因子 (币圈7×24h优化版)...")
    print(f"参数: corr_window={corr_window}, delay_lag={delay_lag}, rebalance_period={rebalance_period}")

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
        if len(group) < corr_window + delay_lag + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['normalized_open'] = gp['open'] / gp['price_base']
        gp['normalized_close'] = gp['close'] / gp['price_base']
        
        # 2. 计算因子组件
        
        # 组件1: 历史日内模式与价格相关性 (权重: 1.0)
        # correlation(delay((open - close), 1), close, 200)
        gp['intraday_ret'] = gp['normalized_open'] - gp['normalized_close']
        gp['intraday_ret_delay'] = gp['intraday_ret'].shift(delay_lag)
        gp['corr_historical_intraday'] = gp['intraday_ret_delay'].rolling(window=corr_window).corr(gp['normalized_close'])
        
        # 组件2: 当前日内表现 (权重: 1.0)
        # (open - close)
        gp['current_intraday'] = gp['normalized_open'] - gp['normalized_close']
        
        # 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[['date', 'symbol', 'corr_historical_intraday', 'current_intraday', 'future_ret']].dropna()

    print("计算 Alpha 37 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    print("计算横截面排名和因子值...")
    
    # 按日期计算Alpha 37因子
    def calculate_alpha37_by_date(df_date):
        df_date = df_date.copy()
        
        # 对各个组件进行排名
        df_date['corr_historical_rank'] = df_date['corr_historical_intraday'].rank(pct=True)
        df_date['current_intraday_rank'] = df_date['current_intraday'].rank(pct=True)
        
        # 按权重组合因子 (两个组件权重相等，各为1.0)
        df_date['alpha37_raw'] = (
            1.0 * df_date['corr_historical_rank'] +
            1.0 * df_date['current_intraday_rank']
        )
        
        return df_date

    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_alpha37_by_date)
    factor_df = factor_df.dropna(subset=['alpha37_raw'])

    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha37_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha37_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha37_intraday_pattern_")

    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n因子统计信息:")
    print(factor_df['factor'].describe())
    return factor_df

if __name__ == "__main__":
    # === Alpha 37 因子测试 (币圈7×24h优化版) ===
    
    # 策略1: 标准参数 (推荐)
    create_alpha37_factor(corr_window=50, delay_lag=5, rebalance_period=10)
    
    # 策略2: 缩短相关性窗口 (适应币圈高频特性)
    # create_alpha37_factor(corr_window=100, delay_lag=1, rebalance_period=3)
    
    # 策略3: 极短窗口 (适应7×24h快速变化)
    # create_alpha37_factor(corr_window=50, delay_lag=1, rebalance_period=2)
