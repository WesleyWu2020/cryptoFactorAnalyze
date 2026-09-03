import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha14_factor(delta_window=3, correlation_window=10, rebalance_period=5, normalization_method='relative_price'):
    """
    创建 Alpha 14 因子 (币圈7×24h优化版)
    
    参数:
    delta_window: 收益率变化计算窗口，默认为3天
    correlation_window: 相关性计算窗口，默认为10天
    rebalance_period: 调仓周期，默认为5天
    normalization_method: 归一化方法，可选: 'relative_price', 'market_cap', 'price_change'
    
    原理:
    Alpha 14 = ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    
    核心步骤:
    1. 收益率趋势：计算收益率3日变化并进行排名
    2. 价量相关性：计算开盘价与交易量10日相关性
    3. 复合信号：两部分相乘，捕捉趋势转变和价量关系
    4. 归一化处理：适应币圈价格和交易量差异
    
    该因子结合收益率趋势和价量关系，识别市场情绪变化和交易模式转变
    """
    print(f"开始构建 Alpha 14 因子 (币圈7×24h优化版)...")
    print(f"参数: delta_window={delta_window}, correlation_window={correlation_window}, rebalance_period={rebalance_period}, normalization_method={normalization_method}")
    
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
    def calculate_alpha14_factor(group):
        if len(group) < max(delta_window, correlation_window) + rebalance_period + 20:
            # 数据不足，返回空DataFrame
            return pd.DataFrame()
        
        # 计算对数收益率
        group['returns'] = np.log(group['close'] / group['close'].shift(1))
        
        # 计算收益率变化 (delta(returns, 3))
        group['returns_delta'] = group['returns'] - group['returns'].shift(delta_window)
        
        # 根据归一化方法处理开盘价和交易量
        if normalization_method == 'relative_price':
            # 方法1: 使用相对价格
            group['normalized_open'] = group['open'] / group['open'].rolling(window=20, min_periods=20).mean()
            group['normalized_volume'] = group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
            
        elif normalization_method == 'market_cap':
            # 方法2: 使用市值加权
            group['normalized_open'] = group['open'] * group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
            group['normalized_volume'] = group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
            
        elif normalization_method == 'price_change':
            # 方法3: 使用价格变化率
            group['normalized_open'] = group['open'].pct_change(periods=5)
            group['normalized_volume'] = group['volume'].pct_change(periods=5)
            
        else:
            # 默认使用相对价格方法
            group['normalized_open'] = group['open'] / group['open'].rolling(window=20, min_periods=20).mean()
            group['normalized_volume'] = group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
        
        # 计算未来对数收益率 (为因子分析提供)，使用调仓周期
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'returns_delta', 'normalized_open', 'normalized_volume', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 14 因子基础数据...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha14_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    
    # 合并结果
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
        
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    print("计算收益率变化排名...")
    
    # 步骤1: 收益率变化排名
    def calculate_returns_rank(df):
        """对每个日期的收益率变化进行排名"""
        df = df.copy()
        
        # 对收益率变化进行排名并取反
        df['returns_delta_rank'] = df['returns_delta'].rank(pct=True)
        df['returns_delta_rank_inv'] = -1 * df['returns_delta_rank']
        
        return df
    
    # 按日期分组进行收益率变化排名
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_returns_rank)
    
    print("计算价量相关性...")
    
    # 步骤2: 计算价量相关性
    def calculate_correlation(group):
        """计算每个交易对的价量相关性"""
        if len(group) < correlation_window:
            return pd.DataFrame()
        
        # 计算过去N天开盘价与交易量的相关性
        group['correlation'] = group['normalized_open'].rolling(window=correlation_window, min_periods=correlation_window).corr(group['normalized_volume'])
        
        return group
    
    # 按交易对分组计算相关性
    result_dfs = []
    for symbol, group in tqdm(factor_df.groupby('symbol')):
        corr_result = calculate_correlation(group)
        if not corr_result.empty:
            result_dfs.append(corr_result)
    
    if not result_dfs:
        print("警告: 没有足够的数据计算相关性")
        return pd.DataFrame()
    
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['correlation'])
    
    print("计算复合因子...")
    
    # 步骤3: 计算复合因子
    def calculate_final_factor(df):
        """计算最终因子值"""
        df = df.copy()
        
        # 复合因子：((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        df['alpha14_raw'] = df['returns_delta_rank_inv'] * df['correlation']
        
        return df
    
    # 按日期分组计算最终因子
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_final_factor)
    
    print("处理极端值和归一化...")
    
    # 检查数据结构
    print(f"数据列: {factor_df.columns.tolist()}")
    print(f"数据形状: {factor_df.shape}")
    
    # 处理极端值
    # 计算每日因子值的上下限(3个标准差)
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha14_raw'].mean()
        std = df['alpha14_raw'].std()
        df['alpha14_raw'] = df['alpha14_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha14_raw'].rank(pct=True)
        df['alpha14_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha14_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha14_returns_correlation_delta{delta_window}d_corr{correlation_window}d_{normalization_method}_{today}.csv")
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
    # === Alpha 14 因子参数优化测试 ===
    
    # 策略1: 标准参数 + 相对价格归一化 (推荐)
    # create_alpha14_factor(delta_window=20, correlation_window=20, rebalance_period=10, normalization_method='relative_price')
    
    # 策略2: 短期策略 + 价格变化率归一化
    # create_alpha14_factor(delta_window=2, correlation_window=5, rebalance_period=3, normalization_method='price_change')
    
    # 策略3: 中期策略 + 市值加权归一化
    # create_alpha14_factor(delta_window=5, correlation_window=15, rebalance_period=7, normalization_method='market_cap')
    
    # 策略4: 超短期策略 + 相对价格归一化
    # create_alpha14_factor(delta_window=1, correlation_window=3, rebalance_period=2, normalization_method='relative_price')
    
    # 策略5: 长期策略 + 相对价格归一化
    create_alpha14_factor(delta_window=5, correlation_window=10, rebalance_period=2, normalization_method='relative_price')
