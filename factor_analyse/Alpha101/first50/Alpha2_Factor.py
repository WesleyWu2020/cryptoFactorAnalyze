import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha2_factor(delta_window=2, correlation_window=6, rebalance_period=3):
    """
    创建 Alpha 2 因子
    
    参数:
    delta_window: 交易量对数变化窗口，默认为2天
    correlation_window: 相关性计算窗口，默认为6天
    rebalance_period: 调仓周期，默认为3天
    
    原理:
    Alpha 2 = (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    
    核心步骤:
    1. 交易量变化率：通过delta(log(volume), 2)计算交易量对数的2日变化
    2. 日内收益率：利用(close - open)/open计算开盘至收盘的收益率
    3. 横截面排名标准化：对交易量变化率和收益率分别进行横截面排名
    4. 相关性计算：计算过去6天两排名序列的相关性
    5. 信号方向反转：对相关系数取反，捕捉量价背离的反转机会
    
    该因子是一个基于价量关系的短期反转型因子，特别适用于短期交易
    """
    print(f"开始构建 Alpha 2 因子...")
    print(f"参数: delta_window={delta_window}, correlation_window={correlation_window}, rebalance_period={rebalance_period}")
    
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
    
    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date'])
    
    # 按symbol和日期排序
    df = df.sort_values(['symbol', 'date'])
    
    # 创建因子计算函数
    def calculate_alpha2_factor(group):
        if len(group) < max(delta_window, correlation_window) + rebalance_period:
            # 数据不足，返回空DataFrame
            return pd.DataFrame()
        
        # 步骤1: 交易量变化率 - delta(log(volume), 2)
        group['log_volume'] = np.log(group['volume'])
        group['volume_delta'] = group['log_volume'].diff(delta_window)
        
        # 步骤2: 日内收益率 - (close - open) / open
        group['intraday_return'] = (group['close'] - group['open']) / group['open']
        
        # 计算未来对数收益率 (为因子分析提供)，使用调仓周期
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'volume_delta', 'intraday_return', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 2 因子基础数据...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha2_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    
    # 合并结果
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
        
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    print("计算横截面排名...")
    
    # 步骤3: 横截面排名标准化
    def calculate_cross_sectional_rank(df):
        """对每个日期的数据进行横截面排名"""
        df = df.copy()
        df['volume_delta_rank'] = df['volume_delta'].rank(pct=True)
        df['intraday_return_rank'] = df['intraday_return'].rank(pct=True)
        return df
    
    # 按日期分组进行横截面排名
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_cross_sectional_rank)
    
    print("计算时序相关性...")
    
    # 步骤4: 计算每个交易对的时序相关性
    def calculate_temporal_correlation(group):
        """计算每个交易对的时序相关性"""
        group = group.copy()
        if len(group) < correlation_window:
            group['alpha2_factor'] = np.nan
            return group
        
        # 计算滚动相关性的函数
        def rolling_correlation(x, y, window):
            """计算滚动相关性"""
            corr_values = []
            for i in range(len(x)):
                if i < window - 1:
                    corr_values.append(np.nan)
                else:
                    # 取过去window天的数据
                    x_window = x.iloc[i-window+1:i+1]
                    y_window = y.iloc[i-window+1:i+1]
                    
                    # 计算相关系数
                    if len(x_window) == window and len(y_window) == window:
                        corr = x_window.corr(y_window)
                        corr_values.append(corr)
                    else:
                        corr_values.append(np.nan)
            
            return pd.Series(corr_values, index=x.index)
        
        # 计算过去correlation_window天的相关性
        correlation = rolling_correlation(
            group['volume_delta_rank'], 
            group['intraday_return_rank'], 
            correlation_window
        )
        
        # 步骤5: 信号方向反转 (-1 * correlation)
        group['alpha2_factor'] = -1 * correlation
        
        return group
    
    # 按交易对分组计算时序相关性
    factor_df = factor_df.groupby('symbol', group_keys=False).apply(calculate_temporal_correlation)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['alpha2_factor'])
    
    print("处理极端值和归一化...")
    
    # 检查数据结构
    print(f"数据列: {factor_df.columns.tolist()}")
    print(f"数据形状: {factor_df.shape}")
    
    # 处理极端值
    # 计算每日因子值的上下限(3个标准差)
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha2_factor'].mean()
        std = df['alpha2_factor'].std()
        df['alpha2_factor'] = df['alpha2_factor'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha2_factor'].rank(pct=True)
        df['alpha2_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha2_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha2_delta{delta_window}d_corr{correlation_window}d_rebalance{rebalance_period}d_{today}.csv")
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
    # 默认使用2天交易量变化窗口，6天相关性窗口，3天调仓
    create_alpha2_factor(delta_window=3, correlation_window=7, rebalance_period=5)