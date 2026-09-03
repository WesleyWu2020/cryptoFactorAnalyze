# factor_analyse/Alpha101/Alpha38_Factor.py
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
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,print_factor_summary
)

def create_alpha38_factor(ts_rank_window=10, rebalance_period=2, availability_lookback_days=90):
    """
    Alpha 38 因子 (币圈7×24h优化版)
    严格避免未来函数，只使用上一期已出现的token
    
    原始定义:
    ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    
    核心步骤:
    1. 价格强度反转：收盘价时间序列排名高的股票给出负权重
    2. 日内动量：日内表现好的股票给出正权重
    3. 交互效应：两者相乘产生交互效应
    
    币圈7×24h特殊设计:
    1. 相对价格归一化：适应币圈巨大的价格差异
    2. 日内动量分析：适应7×24h连续交易特性
    3. 价格强度反转：捕捉低位反弹机会
    4. 🔥 关键：结合历史市值排名，确保只使用上一期已出现的token
    
    该因子适用于捕捉"低位反弹"机会，平均持有期约1-3天
    """
    print(f"开始构建 Alpha 38 因子 (币圈7×24h优化版)...")
    print(f"参数: ts_rank_window={ts_rank_window}, rebalance_period={rebalance_period}")

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
        if len(group) < ts_rank_window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['normalized_open'] = gp['open'] / (gp['price_base'] + 1e-8)
        gp['normalized_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算因子组件
        
        # 组件1: 价格强度反转 (权重: -1)
        # Ts_Rank(close, 10) - 收盘价在10天内的时序排名
        def ts_rank(series, window):
            if len(series) < window:
                return np.nan
            recent_data = series.iloc[-window:]
            rank_pct = recent_data.rank(pct=True).iloc[-1]
            return rank_pct
        
        gp['ts_rank_close'] = gp['normalized_close'].rolling(window=ts_rank_window).apply(
            lambda x: ts_rank(x, ts_rank_window), raw=False
        )
        
        # 组件2: 日内动量 (权重: 1)
        # (close / open) - 日内价格比率
        gp['intraday_ratio'] = gp['normalized_close'] / (gp['normalized_open'] + 1e-8)
        
        # 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[['date', 'symbol', 'ts_rank_close', 'intraday_ratio', 'future_ret']].dropna()

    print("计算 Alpha 38 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    print("计算横截面排名和因子值...")
    
    # 按日期计算Alpha 38因子
    def calculate_alpha38_by_date(df_date):
        df_date = df_date.copy()
        
        # 对各个组件进行排名
        df_date['ts_rank_close_rank'] = df_date['ts_rank_close'].rank(pct=True)
        df_date['intraday_ratio_rank'] = df_date['intraday_ratio'].rank(pct=True)
        
        # 按公式组合因子：(-1 * rank(Ts_Rank(close, 10))) * rank((close / open))
        df_date['alpha38_raw'] = (
            (-1 * df_date['ts_rank_close_rank']) * df_date['intraday_ratio_rank']
        )
        
        return df_date

    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_alpha38_by_date)
    factor_df = factor_df.dropna(subset=['alpha38_raw'])

    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha38_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha38_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha38_intraday_reversal_")

    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    
    # 显示因子数据统计信息
    print("\n因子统计信息:")
    print(factor_df['factor'].describe())
    
    # 显示某个token的前5日因子值和收益率
    print("\n=== 某个Token的前5日因子值分析 ===")
    
    print_factor_summary(factor_df, out_path)
    
    return factor_df

if __name__ == "__main__":
    # === Alpha 38 因子测试 (币圈7×24h优化版) ===
    
    # 策略1: 标准参数 (推荐)
    create_alpha38_factor(ts_rank_window=10, rebalance_period=10)
    
    # 策略2: 缩短时序排名窗口 (适应币圈高频特性)
    # create_alpha38_factor(ts_rank_window=5, rebalance_period=1)
    
    # 策略3: 延长时序排名窗口 (更稳定的价格强度判断)
    # create_alpha38_factor(ts_rank_window=15, rebalance_period=3)
