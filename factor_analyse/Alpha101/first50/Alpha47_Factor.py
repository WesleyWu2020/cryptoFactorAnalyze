# factor_analyse/Alpha101/Alpha47_Factor.py
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

def create_alpha47_factor(adv_window=20, high_sum_window=5, vwap_delay=5, 
                         rebalance_period=5, availability_lookback_days=90):
    """
    Alpha 47 因子 (币圈7×24h优化版)
    
    原始定义:
    ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - 
     rank((vwap - delay(vwap, 5))))
    
    思路:
    1. 低价股与成交量异常识别:
       - rank(1/close) 聚焦低价股
       - volume/adv20 识别放量
    2. 日内价格强度与趋势分析:
       - high * rank(high - close) 识别冲高回落
       - 除以5日最高价均值进行标准化
    3. 减去VWAP变化排名:
       - rank(vwap - delay(vwap, 5)) 衡量中期资金成本变化
    4. 综合逻辑:
       - 若低价放量 * 冲高回落 > VWAP趋势排名，可能预示反转
    
    - 平均持有期: 3-7天
    """
    print(f"开始构建 Alpha 47 因子...")
    print(f"参数: adv_window={adv_window}, high_sum_window={high_sum_window}, "
          f"vwap_delay={vwap_delay}, rebalance_period={rebalance_period}")

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
        if len(group) < max(adv_window, high_sum_window, vwap_delay) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['norm_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        gp['norm_high'] = gp['high'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算VWAP (如果没有，用(high+low+close)/3近似)
        if 'vwap' not in gp.columns:
            gp['vwap'] = (gp['high'] + gp['low'] + gp['close']) / 3
        gp['norm_vwap'] = gp['vwap'] / (gp['price_base'] + 1e-8)
        
        # 3. 计算因子组件
        
        # 3.1 低价股与成交量异常识别
        gp['inv_close'] = 1 / (gp['norm_close'] + 1e-8)  # 收盘价倒数
        
        # 3.2 成交量比率
        gp['adv20'] = gp['volume'].rolling(window=adv_window, min_periods=adv_window).mean()
        gp['vol_ratio'] = gp['volume'] / (gp['adv20'] + 1e-8)
        
        # 3.3 日内价格强度与趋势分析
        gp['high_close_diff'] = gp['norm_high'] - gp['norm_close']  # 冲高回落
        gp['high_sum_5'] = gp['norm_high'].rolling(window=high_sum_window, min_periods=high_sum_window).sum()
        gp['high_avg_5'] = gp['high_sum_5'] / high_sum_window
        
        # 3.4 VWAP变化
        gp['vwap_change'] = gp['norm_vwap'] - gp['norm_vwap'].shift(vwap_delay)
        
        # 4. 横截面排名组件 (这里先用NaN占位，后面按日期分组计算)
        gp['rank_inv_close'] = np.nan
        gp['rank_high_close_diff'] = np.nan
        gp['rank_vwap_change'] = np.nan
        
        # 5. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")
        
        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)
        
        return gp[['date', 'symbol', 'inv_close', 'vol_ratio', 'norm_high', 'high_close_diff', 
                   'high_avg_5', 'vwap_change', 'future_ret']].dropna()

    print("计算 Alpha 47 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    # 合并所有token的数据
    all_data = pd.concat(result_dfs, ignore_index=True)
    
    # 按日期分组计算横截面排名
    print("计算横截面排名组件...")
    all_data_by_date = all_data.groupby('date')
    
    rank_dfs = []
    for date, group in tqdm(all_data_by_date):
        if len(group) < 2:  # 至少需要2个token才能计算有意义的排名
            continue
            
        # 计算横截面排名
        group = group.copy()
        group['rank_inv_close'] = group['inv_close'].rank(pct=True)
        group['rank_high_close_diff'] = group['high_close_diff'].rank(pct=True)
        group['rank_vwap_change'] = group['vwap_change'].rank(pct=True)
        
        # 计算因子值
        # ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
        group['alpha47_raw'] = ((group['rank_inv_close'] * group['vol_ratio']) * 
                              ((group['norm_high'] * group['rank_high_close_diff']) / (group['high_avg_5'] + 1e-8))) - \
                              group['rank_vwap_change']
        
        rank_dfs.append(group)
    
    if not rank_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
    
    factor_df = pd.concat(rank_dfs, ignore_index=True)
    
    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha47_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha47_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha47_low_price_volume_reversal_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_alpha47_factor(adv_window=20, high_sum_window=5, vwap_delay=5, rebalance_period=5)
    
    # 可选：更短的窗口，更敏感地捕捉短期反转
    # create_alpha47_factor(adv_window=10, high_sum_window=3, vwap_delay=3, rebalance_period=3)
