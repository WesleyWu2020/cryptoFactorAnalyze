# factor_analyse/Alpha101/Alpha40_Factor.py
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
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    analyze_token_performance
)

def create_alpha40_factor(std_window=10, corr_window=10, rebalance_period=5, availability_lookback_days=90):
    """
    Alpha 40 因子 (币圈7×24h优化版)
    严格避免未来函数，只使用上一期已出现的token
    
    原始定义:
    Alpha#40 = (-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)
    
    核心思路:
    - 波动性偏好: 偏好低波动性股票（通过负号实现）
    - 价量关系: 分析最高价与交易量的相关性
    - 交互效应: 低波动性且价量关系良好的股票得到正评分
    - 平均持有期: 3-7天
    """
    print("开始构建 Alpha 40 因子 (7x24 加密优化)...")
    print(f"参数: std_window={std_window}, corr_window={corr_window}, rebalance_period={rebalance_period}")
    
    # 加载历史市值数据
    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(
        historical_df, lookback_days=availability_lookback_days
    )
    
    # 加载K线数据
    kline_df = load_kline_df()
    
    # 过滤数据：只保留有历史市值的token
    # 按symbol分组处理可用性过滤
    filtered_parts = []
    for symbol, group in kline_df.groupby('symbol'):
        filtered_group = filter_group_by_availability(group, symbol, available_tokens_by_date)
        if not filtered_group.empty:
            filtered_parts.append(filtered_group)
    
    filtered_df = pd.concat(filtered_parts, ignore_index=True) if filtered_parts else pd.DataFrame()
    
    print(f"过滤后数据: {len(filtered_df)} 条记录")
    print_availability_sample(available_tokens_by_date)
    
    # 单symbol计算
    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < max(std_window, corr_window) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()
        
        # === 币圈价格归一化处理 ===
        # 使用相对价格变化，避免不同币种价格差异
        gp['price_normalized'] = gp['close'] / gp['close'].rolling(window=20, min_periods=1).mean()
        gp['high_normalized'] = gp['high'] / gp['close'].rolling(window=20, min_periods=1).mean()
        gp['volume_normalized'] = gp['volume'] / gp['volume'].rolling(window=20, min_periods=1).mean()
        
        # === Alpha 40 因子计算 ===
        
        # 1. 计算最高价的波动率 (stddev(high, 10))
        gp['high_std'] = gp['high_normalized'].rolling(window=std_window, min_periods=std_window).std()
        
        # 2. 计算最高价与交易量的相关性 (correlation(high, volume, 10))
        gp['high_vol_corr'] = gp['high_normalized'].rolling(window=corr_window, min_periods=corr_window).corr(gp['volume_normalized'])
        
        # 3. 计算因子值
        # 注意：这里需要处理NaN值，避免计算错误
        gp['factor'] = np.nan
        
        # 只在有足够数据时计算
        valid_mask = (gp['high_std'].notna()) & (gp['high_vol_corr'].notna())
        if valid_mask.sum() > 0:
            # 对波动率进行反向排名（低波动性更好）
            high_std_rank = gp.loc[valid_mask, 'high_std'].rank(ascending=True, pct=True)
            # 波动率排名取负号（低波动性得高分）
            volatility_score = -1 * high_std_rank
            # 与价量相关性相乘
            gp.loc[valid_mask, 'factor'] = volatility_score * gp.loc[valid_mask, 'high_vol_corr']
        
        # 计算未来收益率
        gp['future_ret'] = future_return(gp['close'], rebalance_period)
        
        # 只返回需要的列
        result = gp[['date', 'factor', 'future_ret']].copy()
        result['instrument'] = symbol
        result = result.dropna(subset=['factor', 'future_ret'])
        
        return result
    
    # 分组计算
    print("开始计算因子值...")
    result_dfs = group_apply_with_progress(filtered_df, 'symbol', compute_one)
    
    if not result_dfs:
        print("❌ 没有生成任何因子数据")
        return pd.DataFrame()
        
    factor_results = pd.concat(result_dfs, ignore_index=True)
    
    # 数据后处理
    print("开始数据后处理...")
    
    # 去极值处理
    factor_results = winsorize_by_date(factor_results, 'factor', n_std=3.0)
    
    # 横截面排名归一化到[-1, 1]
    factor_results = rank_to_unit_by_date(factor_results, 'factor')
    
    # 保存因子数据
    out_path = save_factor_df(factor_results, f"alpha40_volatility_volume")
    
    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_results)} 条因子记录")
    
    # 显示因子统计信息
    print("\n因子统计信息:")
    print(factor_results['factor'].describe())
    
    # 分析某个token的前5日表现
    analyze_token_performance(factor_results, n_days=5)
    
    return factor_results

if __name__ == "__main__":
    # 测试运行
    create_alpha40_factor(std_window=10, corr_window=20, rebalance_period=3)
