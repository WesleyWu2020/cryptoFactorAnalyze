import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha20_factor(rebalance_period=3, normalization_method='relative_price'):
    """
    创建 Alpha 20 因子 (币圈7×24h优化版)
    
    参数:
    rebalance_period: 调仓周期，默认为3天
    normalization_method: 归一化方法，可选: 'relative_price', 'market_cap', 'price_change'
    
    原理:
    Alpha 20 = (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
    """
    print(f"开始构建 Alpha 20 因子 (币圈7×24h优化版)...")
    print(f"参数: rebalance_period={rebalance_period}, normalization_method={normalization_method}")
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))  # 回到Crypto目录
    
    # 数据文件路径
    data_dir = os.path.join(base_dir, "data", "kline_data")
    
    # 查找最新的K线数据文件
    kline_files = [f for f in os.listdir(data_dir) if f.startswith("binance_daily_klines_")]
    if not kline_files:
        raise FileNotFoundError("未找到K线数据文件")
    
    latest_file = sorted(kline_files)[-1]
    data_path = os.path.join(data_dir, latest_file)
    
    print(f"读取数据文件: {data_path}")
    
    # 读取K线数据
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])
    
    def calculate_alpha20_factor(group):
        if len(group) < 3 + rebalance_period + 20:
            return pd.DataFrame()
        
        # 归一化处理
        if normalization_method == 'relative_price':
            group['normalized_open'] = group['open'] / group['open'].rolling(window=20, min_periods=20).mean()
            group['normalized_high'] = group['high'] / group['high'].rolling(window=20, min_periods=20).mean()
            group['normalized_low'] = group['low'] / group['low'].rolling(window=20, min_periods=20).mean()
            group['normalized_close'] = group['close'] / group['close'].rolling(window=20, min_periods=20).mean()
        elif normalization_method == 'market_cap':
            group['normalized_open'] = group['open'] * group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
            group['normalized_high'] = group['high'] * group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
            group['normalized_low'] = group['low'] * group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
            group['normalized_close'] = group['close'] * group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
        elif normalization_method == 'price_change':
            group['normalized_open'] = group['open'].pct_change(periods=5)
            group['normalized_high'] = group['high'].pct_change(periods=5)
            group['normalized_low'] = group['low'].pct_change(periods=5)
            group['normalized_close'] = group['close'].pct_change(periods=5)
        else:
            group['normalized_open'] = group['open'] / group['open'].rolling(window=20, min_periods=20).mean()
            group['normalized_high'] = group['high'] / group['high'].rolling(window=20, min_periods=20).mean()
            group['normalized_low'] = group['low'] / group['low'].rolling(window=20, min_periods=20).mean()
            group['normalized_close'] = group['close'] / group['close'].rolling(window=20, min_periods=20).mean()
        
        # 计算未来对数收益率
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'normalized_open', 'normalized_high', 'normalized_low', 'normalized_close', 'open', 'high', 'low', 'close', 'future_ret']].copy()
        result = result.dropna()
        return result

    print("计算 Alpha 20 因子基础数据...")
    result_dfs = []
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha20_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
    factor_df = pd.concat(result_dfs, ignore_index=True)

    print("计算横截面排名和复合因子...")
    def calculate_final_factor(df):
        df = df.copy()
        # 计算各项差分
        df['open_high_diff'] = df['open'] - df['high'].shift(7)
        df['open_low_diff'] = df['open'] - df['low'].shift(7)
        # 横截面排名
        df['rank_open_high'] = df['open_high_diff'].rank(pct=True)
        df['rank_open_low'] = df['open_low_diff'].rank(pct=True)
        # 复合信号（去掉open-close项）
        df['alpha20_raw'] = (-1 * df['rank_open_high']) * df['rank_open_low']
        return df

    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_final_factor)
    factor_df = factor_df.dropna(subset=['alpha20_raw'])

    print("处理极端值和归一化...")
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha20_raw'].mean()
        std = df['alpha20_raw'].std()
        df['alpha20_raw'] = df['alpha20_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)

    def normalize_by_date(df):
        df = df.copy()
        rank_pct = df['alpha20_raw'].rank(pct=True)
        df['factor'] = 2 * (rank_pct - 0.5)
        return df
    print("排序归一化因子值到-1到1之间...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)

    factor_df = factor_df.rename(columns={'symbol': 'instrument'})
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]

    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha20_open_position_composite_{normalization_method}_rebalance{rebalance_period}d_{today}.csv")
    factor_df.to_csv(output_path, index=False)

    print(f"✅ 因子数据已保存至: {output_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n因子统计信息:")
    print(factor_df['factor'].describe())
    print("\n数据预览:")
    print(factor_df.head())
    return factor_df

if __name__ == "__main__":
    # 推荐参数
    create_alpha20_factor(rebalance_period=7, normalization_method='market_cap')
