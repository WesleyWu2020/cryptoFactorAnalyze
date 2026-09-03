import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

def create_alpha11_factor(ts_window=3, volume_delta_window=3, rebalance_period=3):
    """
    创建 Alpha 11 因子 (币圈7×24h优化版)
    严格避免未来函数，只使用上一期已出现的token
    
    参数:
    ts_window: 时序统计窗口，默认为3天
    volume_delta_window: 交易量变化窗口，默认为3天
    rebalance_period: 调仓周期，默认为3天
    
    原理:
    Alpha 11 = ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    
    核心步骤:
    1. 价格偏离计算：计算VWAP与收盘价的差值
    2. 时序极值排名：对过去3天价格偏离的最大值和最小值分别排名
    3. 交易量变化排名：对交易量3天变化进行排名
    4. 复合信号：将价格偏离排名和交易量变化排名相乘
    5. 🔥 关键：结合历史市值排名，确保只使用上一期已出现的token
    
    该因子寻找日内价格行为异常且交易量显著变化的币种，适合捕捉市场情绪变化
    """
    print(f"开始构建 Alpha 11 因子 (币圈7×24h优化版)...")
    print(f"参数: ts_window={ts_window}, volume_delta_window={volume_delta_window}, rebalance_period={rebalance_period}")
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))  # 回到Crypto目录
    
    #  关键：加载历史市值排名数据，避免未来函数
    historical_csv_path = os.path.join(base_dir, "data", "cryptocompare_top50_marketcap_historical.csv")
    if not os.path.exists(historical_csv_path):
        raise FileNotFoundError(f"未找到历史市值排名文件: {historical_csv_path}")
    
    print("加载历史市值排名数据...")
    historical_df = pd.read_csv(historical_csv_path)
    historical_df['Date'] = pd.to_datetime(historical_df['Date'], format='mixed', errors='coerce')
    historical_df = historical_df.sort_values('Date').reset_index(drop=True)
    
    # 创建每个token的可用性映射
    def get_available_tokens_by_date(historical_df, lookback_period=30):
        """获取每个日期可用的token列表（基于lookback_period天前的排名）"""
        available_tokens_by_date = {}
        
        for date in historical_df['Date'].unique():
            # 找到lookback_period天前的日期
            lookback_date = date - timedelta(days=lookback_period)
            
            # 获取lookback_date之前的所有token
            available_tokens = historical_df[historical_df['Date'] <= lookback_date]['Trading_Pair'].unique().tolist()
            
            if available_tokens:
                available_tokens_by_date[date] = available_tokens
        
        return available_tokens_by_date
    
    # 获取每个日期可用的token（基于90天前的排名）
    available_tokens_by_date = get_available_tokens_by_date(historical_df, lookback_period=90)
    print(f"已创建 {len(available_tokens_by_date)} 个日期的token可用性映射")
    
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
    
    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date'])
    
    # 按symbol和日期排序
    df = df.sort_values(['symbol', 'date'])
    
    # 创建因子计算函数（避免未来函数版本）
    def calculate_alpha11_factor_with_availability(group, symbol, available_tokens_by_date):
        if len(group) < ts_window + volume_delta_window + rebalance_period + 5:
            # 数据不足，返回空DataFrame
            return pd.DataFrame()
        
        # 🔥 关键：只保留在对应日期可用的token的数据
        valid_rows = []
        for idx, row in group.iterrows():
            current_date = row['date'].date()
            
            # 找到最接近的可用日期
            available_date = None
            for date_key in available_tokens_by_date.keys():
                if date_key.date() <= current_date:
                    available_date = date_key
                else:
                    break
            
            # 检查token在该日期是否可用
            if available_date and symbol in available_tokens_by_date[available_date]:
                valid_rows.append(idx)
        
        if not valid_rows:
            return pd.DataFrame()
        
        # 只保留有效行
        group = group.loc[valid_rows].copy()
        
        # 计算VWAP (Volume Weighted Average Price)
        # 由于没有VWAP数据，使用(high + low + close) / 3作为近似
        group['vwap'] = (group['high'] + group['low'] + group['close']) / 3
        
        # 计算价格偏离 (vwap - close)
        group['price_deviation'] = group['vwap'] - group['close']
        
        # 计算过去N天价格偏离的时序最大值和最小值
        group['ts_max_deviation'] = group['price_deviation'].rolling(window=ts_window, min_periods=ts_window).max()
        group['ts_min_deviation'] = group['price_deviation'].rolling(window=ts_window, min_periods=ts_window).min()
        
        # 计算交易量变化 (delta(volume, 3))
        group['volume_delta'] = group['volume'] - group['volume'].shift(volume_delta_window)
        
        # 计算未来对数收益率 (为因子分析提供)，使用调仓周期
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'ts_max_deviation', 'ts_min_deviation', 
                      'volume_delta', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 11 因子基础数据（避免未来函数）...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha11_factor_with_availability(group, symbol, available_tokens_by_date)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    
    # 合并结果
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
        
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    print("计算横截面排名...")
    
    # 步骤1: 横截面排名标准化
    def calculate_cross_sectional_rank(df):
        """对每个日期的数据进行横截面排名"""
        df = df.copy()
        
        # 对时序最大值进行排名
        df['ts_max_rank'] = df['ts_max_deviation'].rank(pct=True)
        
        # 对时序最小值进行排名
        df['ts_min_rank'] = df['ts_min_deviation'].rank(pct=True)
        
        # 对交易量变化进行排名
        df['volume_delta_rank'] = df['volume_delta'].rank(pct=True)
        
        # 计算复合因子：((ts_max_rank + ts_min_rank) * volume_delta_rank)
        df['alpha11_raw'] = (df['ts_max_rank'] + df['ts_min_rank']) * df['volume_delta_rank']
        
        return df
    
    # 按日期分组进行横截面排名
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_cross_sectional_rank)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['alpha11_raw'])
    
    print("处理极端值和归一化...")
    
    # 检查数据结构
    print(f"数据列: {factor_df.columns.tolist()}")
    print(f"数据形状: {factor_df.shape}")
    
    # 处理极端值
    # 计算每日因子值的上下限(3个标准差)
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha11_raw'].mean()
        std = df['alpha11_raw'].std()
        df['alpha11_raw'] = df['alpha11_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha11_raw'].rank(pct=True)
        df['alpha11_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha11_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha11_vwap_deviation_ts{ts_window}d_vol{volume_delta_window}d_rebalance{rebalance_period}d_{today}.csv")
    factor_df.to_csv(output_path, index=False)
    
    print(f"✅ 因子数据已保存至: {output_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    
    # 显示因子数据统计信息
    print("\n因子统计信息:")
    stats = factor_df['factor'].describe()
    print(stats)
    
    # 显示前几行数据
    print("\n数据预览:")
    print(factor_df.head())
    
    return factor_df

if __name__ == "__main__":
    # === Alpha 11 因子参数优化测试 ===
    
    # 策略1: 标准参数 (推荐)
    create_alpha11_factor(ts_window=20, volume_delta_window=20, rebalance_period=5)
