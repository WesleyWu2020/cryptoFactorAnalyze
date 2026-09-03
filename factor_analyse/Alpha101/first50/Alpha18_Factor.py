import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha18_factor(std_window=5, corr_window=10, rebalance_period=3, normalization_method='relative_price'):
    """
    创建 Alpha 18 因子 (币圈7×24h优化版)
    
    参数:
    std_window: 波动性窗口，默认为5天
    corr_window: 相关性窗口，默认为10天
    rebalance_period: 调仓周期，默认为3天
    normalization_method: 归一化方法，可选: 'relative_price', 'market_cap', 'price_change'
    
    原理:
    Alpha 18 = -1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))
    """
    print(f"开始构建 Alpha 18 因子 (币圈7×24h优化版)...")
    print(f"参数: std_window={std_window}, corr_window={corr_window}, rebalance_period={rebalance_period}, normalization_method={normalization_method}")
    
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
    
    def calculate_alpha18_factor(group):
        if len(group) < max(std_window, corr_window) + rebalance_period + 20:
            return pd.DataFrame()
        
        # 归一化处理
        if normalization_method == 'relative_price':
            group['normalized_close'] = group['close'] / group['close'].rolling(window=20, min_periods=20).mean()
            group['normalized_open'] = group['open'] / group['open'].rolling(window=20, min_periods=20).mean()
        elif normalization_method == 'market_cap':
            group['normalized_close'] = group['close'] * group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
            group['normalized_open'] = group['open'] * group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
        elif normalization_method == 'price_change':
            group['normalized_close'] = group['close'].pct_change(periods=5)
            group['normalized_open'] = group['open'].pct_change(periods=5)
        else:
            group['normalized_close'] = group['close'] / group['close'].rolling(window=20, min_periods=20).mean()
            group['normalized_open'] = group['open'] / group['open'].rolling(window=20, min_periods=20).mean()
        
        # 计算未来对数收益率
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'normalized_close', 'normalized_open', 'close', 'open', 'future_ret']].copy()
        result = result.dropna()
        return result

    print("计算 Alpha 18 因子基础数据...")
    result_dfs = []
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha18_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
    factor_df = pd.concat(result_dfs, ignore_index=True)

    print("计算时序波动性、日内收益和相关性...")
    def calculate_time_series_metrics(group):
        if len(group) < max(std_window, corr_window) + 1:
            return pd.DataFrame()
        # 日内波动性
        group['abs_co'] = np.abs(group['close'] - group['open'])
        group['std_abs_co'] = group['abs_co'].rolling(window=std_window, min_periods=std_window).std()
        # 日内收益
        group['co_diff'] = group['close'] - group['open']
        # 收盘-开盘相关性
        group['corr_co'] = group['close'].rolling(window=corr_window, min_periods=corr_window).corr(group['open'])
        return group

    result_dfs = []
    for symbol, group in tqdm(factor_df.groupby('symbol')):
        ts_result = calculate_time_series_metrics(group)
        if not ts_result.empty:
            result_dfs.append(ts_result)
    if not result_dfs:
        print("警告: 没有足够的数据计算时序指标")
        return pd.DataFrame()
    factor_df = pd.concat(result_dfs, ignore_index=True)
    factor_df = factor_df.dropna(subset=['std_abs_co', 'co_diff', 'corr_co'])

    print("计算横截面排名和复合因子...")
    def calculate_final_factor(df):
        df = df.copy()
        # 复合信号
        df['alpha18_raw'] = (df['std_abs_co'] + df['co_diff']) + df['corr_co']
        # 横截面排名
        df['alpha18_rank'] = df['alpha18_raw'].rank(pct=True)
        # 取反
        df['alpha18_factor'] = -1 * df['alpha18_rank']
        return df

    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_final_factor)

    print("处理极端值和归一化...")
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha18_factor'].mean()
        std = df['alpha18_factor'].std()
        df['alpha18_factor'] = df['alpha18_factor'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)

    def normalize_by_date(df):
        df = df.copy()
        rank_pct = df['alpha18_factor'].rank(pct=True)
        df['factor'] = 2 * (rank_pct - 0.5)
        return df
    print("排序归一化因子值到-1到1之间...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)

    factor_df = factor_df.rename(columns={'symbol': 'instrument'})
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]

    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha18_intraday_vol_corr_std{std_window}d_corr{corr_window}d_{normalization_method}_rebalance{rebalance_period}d_{today}.csv")
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
    create_alpha18_factor(std_window=5, corr_window=5, rebalance_period=5, normalization_method='relative_price')
