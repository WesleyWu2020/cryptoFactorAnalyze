# factor_analyse/Alpha101/last51/Alpha54_Factor.py
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

def create_alpha54_factor(power_exponent=5, rebalance_period=2, availability_lookback_days=90):
    """
    Alpha 54 因子 (币圈7×24h优化版)
    
    原始定义:
    ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    
    思路:
    1. 分子: -1 × (low - close) × (open^5)
       - (low - close): 最低价与收盘价的差异
       - (open^5): 开盘价的5次方，引入高度非线性
    2. 分母: (low - high) × (close^5)
       - (low - high): 最低价与最高价的差异
       - (close^5): 收盘价的5次方，引入高度非线性
    3. 非线性特性: 通过5次方运算和价格差乘积引入非线性
    4. 价格敏感性: 对价格水平高度敏感，可能针对特定价格区间优化
    
    这是一个高度非线性的价格关系因子，解释性较低但可能捕捉特定的价格模式。
    
    - 平均持有期: 1-3天
    """
    print(f"开始构建 Alpha 54 因子...")
    print(f"参数: power_exponent={power_exponent}, rebalance_period={rebalance_period}")

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
        if len(group) < rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # # 1. 价格归一化处理（适应币圈价格差异）
        # gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        # gp['norm_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        # gp['norm_open'] = gp['open'] / (gp['price_base'] + 1e-8)
        # gp['norm_high'] = gp['high'] / (gp['price_base'] + 1e-8)
        # gp['norm_low'] = gp['low'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算因子组件
        
        # 2.1 分子: -1 × (low - close) × (open^5)
        gp['low_close_diff'] = gp['low'] - gp['close']
        gp['open_power'] = gp['open'] ** power_exponent
        gp['numerator'] = -1 * gp['low_close_diff'] * gp['open_power']
        
        # 2.2 分母: (low - high) × (close^5)
        gp['low_high_diff'] = gp['low'] - gp['high']
        gp['close_power'] = gp['close'] ** power_exponent
        gp['denominator'] = gp['low_high_diff'] * gp['close_power']
        
        # 2.3 计算因子值: 分子/分母
        # 添加小量避免除零错误
        gp['alpha54_raw'] = gp['numerator'] / (gp['denominator'] + 1e-8)
        
        # 3. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "alpha54_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha54_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha54_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha54_nonlinear_price_relation_{power_exponent}th_power_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_alpha54_factor(power_exponent=3, rebalance_period=2)
    
    # 可选：降低幂次，减少非线性程度
    # create_alpha54_factor(power_exponent=3, rebalance_period=2)
    
    # 可选：更短的调仓周期，适应加密货币市场
    # create_alpha54_factor(power_exponent=5, rebalance_period=1)
