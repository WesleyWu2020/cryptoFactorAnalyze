# factor_analyse/Alpha101/first50/Alpha21_Factor.py
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

def create_alpha21_factor(ma_window=8, std_window=8, short_ma_window=2, 
                         adv_window=20, vol_threshold=1.0, rebalance_period=7, 
                         availability_lookback_days=90):
    """
    Alpha 21 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha 21 = 
        if (8日均价+8日std < 2日均价): -1
        elif (2日均价 < 8日均价-8日std): 1
        elif (volume/adv20 >= 1): 1
        else: -1
    
    思路:
    1. 价格通道突破：基于移动平均线和标准差构建价格通道
    2. 趋势确认：通过短期均线与通道的关系判断趋势方向
    3. 成交量确认：当价格信号不明确时，用成交量活跃度作为辅助判断
    4. 自适应信号：结合价格位置和成交量，生成-1到1的离散信号
    
    核心逻辑:
    - 当短期均线突破通道上轨时，给出看跌信号(-1)
    - 当短期均线跌破通道下轨时，给出看涨信号(1)
    - 当价格在通道内时，用成交量活跃度判断方向
    
    - 平均持有期: 5-10天
    """
    print(f"开始构建 Alpha 21 因子...")
    print(f"参数: ma_window={ma_window}, std_window={std_window}, short_ma_window={short_ma_window}")
    print(f"      adv_window={adv_window}, vol_threshold={vol_threshold}, rebalance_period={rebalance_period}")

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
        min_required = max(ma_window, std_window, adv_window) + rebalance_period + 5
        
        if len(group) < min_required:
            return pd.DataFrame()
        
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 计算移动平均线
        gp['ma_long'] = gp['close'].rolling(window=ma_window, min_periods=ma_window).mean()
        gp['ma_short'] = gp['close'].rolling(window=short_ma_window, min_periods=short_ma_window).mean()
        
        # 2. 计算标准差（价格波动性）
        gp['std_long'] = gp['close'].rolling(window=std_window, min_periods=std_window).std()
        
        # 3. 计算平均成交量
        gp['adv'] = gp['volume'].rolling(window=adv_window, min_periods=adv_window).mean()
        
        # 4. 计算成交量比率
        gp['vol_ratio'] = gp['volume'] / (gp['adv'] + 1e-8)
        
        # 5. 计算价格通道
        gp['channel_upper'] = gp['ma_long'] + gp['std_long']
        gp['channel_lower'] = gp['ma_long'] - gp['std_long']
        
        # 6. Alpha 21 核心逻辑
        def alpha21_logic(row):
            """Alpha 21 因子计算逻辑"""
            # 检查数据完整性
            if (pd.isna(row['ma_long']) or pd.isna(row['std_long']) or 
                pd.isna(row['ma_short']) or pd.isna(row['adv']) or 
                pd.isna(row['vol_ratio'])):
                return np.nan
            
            # 逻辑1: 短期均线突破通道上轨 -> 看跌信号
            if row['channel_upper'] < row['ma_short']:
                return -1
            
            # 逻辑2: 短期均线跌破通道下轨 -> 看涨信号  
            elif row['ma_short'] < row['channel_lower']:
                return 1
            
            # 逻辑3: 价格在通道内，用成交量判断
            elif row['vol_ratio'] >= vol_threshold:
                return 1
            
            # 逻辑4: 成交量不活跃 -> 看跌信号
            else:
                return -1
        
        gp['alpha21_raw'] = gp.apply(alpha21_logic, axis=1)
        
        # 7. 计算额外的技术指标（用于分析）
        gp['price_position'] = (gp['close'] - gp['channel_lower']) / (gp['channel_upper'] - gp['channel_lower'] + 1e-8)
        gp['trend_strength'] = (gp['ma_short'] - gp['ma_long']) / (gp['ma_long'] + 1e-8)
        
        # 8. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "alpha21_raw", "price_position", "trend_strength", "future_ret"]].dropna()

    print("计算 Alpha 21 因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 由于Alpha21是离散信号(-1, 1)，不需要去极值处理
    # 但可以进行横截面标准化，增强信号的相对强度
    def normalize_discrete_signal(df):
        """对离散信号进行横截面标准化"""
        df = df.copy()
        
        # 计算当日信号分布
        signal_counts = df['alpha21_raw'].value_counts()
        total_signals = len(df)
        
        # 如果信号分布过于不均衡，进行微调
        if len(signal_counts) == 2:  # 只有-1和1
            pos_ratio = signal_counts.get(1, 0) / total_signals
            neg_ratio = signal_counts.get(-1, 0) / total_signals
            
            # 如果信号过于偏向某一方向，进行适度调整
            if pos_ratio > 0.8:  # 过于看涨
                # 对部分看涨信号降级为中性
                pos_mask = df['alpha21_raw'] == 1
                pos_indices = df[pos_mask].index
                if len(pos_indices) > 0:
                    # 随机选择部分看涨信号降级
                    downgrade_count = int(len(pos_indices) * 0.2)
                    downgrade_indices = np.random.choice(pos_indices, 
                                                       size=min(downgrade_count, len(pos_indices)), 
                                                       replace=False)
                    df.loc[downgrade_indices, 'alpha21_raw'] = 0
                    
            elif neg_ratio > 0.8:  # 过于看跌
                # 对部分看跌信号降级为中性
                neg_mask = df['alpha21_raw'] == -1
                neg_indices = df[neg_mask].index
                if len(neg_indices) > 0:
                    # 随机选择部分看跌信号降级
                    downgrade_count = int(len(neg_indices) * 0.2)
                    downgrade_indices = np.random.choice(neg_indices, 
                                                       size=min(downgrade_count, len(neg_indices)), 
                                                       replace=False)
                    df.loc[downgrade_indices, 'alpha21_raw'] = 0
        
        return df

    print("对离散信号进行横截面标准化...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_discrete_signal)

    # 将离散信号转换为连续因子值（保持-1到1的范围）
    def discrete_to_continuous(df):
        """将离散信号转换为连续因子值"""
        df = df.copy()
        
        # 基于价格位置和趋势强度调整信号强度
        df['signal_strength'] = np.abs(df['price_position'] - 0.5) + np.abs(df['trend_strength'])
        
        # 将离散信号与强度结合
        df['factor'] = df['alpha21_raw'] * (0.5 + df['signal_strength'] * 0.5)
        
        # 确保因子值在[-1, 1]范围内
        df['factor'] = df['factor'].clip(-1, 1)
        
        return df

    print("将离散信号转换为连续因子值...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(discrete_to_continuous)

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})
    factor_df = factor_df[["date", "instrument", "factor", "future_ret"]]
    
    out_path = save_factor_df(factor_df, 
                            file_prefix=f"alpha21_price_channel_volume_ma{ma_window}d_std{std_window}d_")

    print_factor_summary(factor_df, out_path)
    
    # 额外的Alpha21特定分析
    print("\n📊 Alpha 21 因子特殊分析:")
    print("="*50)
    
    # 信号分布分析
    signal_dist = factor_df['factor'].apply(lambda x: '看涨' if x > 0.1 else '看跌' if x < -0.1 else '中性')
    signal_counts = signal_dist.value_counts()
    print(f"信号分布: {dict(signal_counts)}")
    
    # 信号强度分析
    strong_signals = factor_df[np.abs(factor_df['factor']) > 0.5]
    if not strong_signals.empty:
        strong_ratio = len(strong_signals) / len(factor_df) * 100
        print(f"强信号比例: {strong_ratio:.1f}%")
    
    return factor_df

if __name__ == "__main__":
    # 标准参数（参考原始Alpha21定义）
    create_alpha21_factor(
        ma_window=8, 
        std_window=8, 
        short_ma_window=2, 
        adv_window=20, 
        vol_threshold=1.0, 
        rebalance_period=7
    )
    
    # 可选：适应加密货币市场的更快节奏
    # create_alpha21_factor(
    #     ma_window=5, 
    #     std_window=5, 
    #     short_ma_window=2, 
    #     adv_window=10, 
    #     vol_threshold=1.2, 
    #     rebalance_period=5
    # )
    
    # 可选：更保守的参数设置
    # create_alpha21_factor(
    #     ma_window=12, 
    #     std_window=12, 
    #     short_ma_window=3, 
    #     adv_window=30, 
    #     vol_threshold=0.8, 
    #     rebalance_period=10
    # )
