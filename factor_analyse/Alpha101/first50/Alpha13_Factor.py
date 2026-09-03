import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha13_factor(covariance_window=5, rebalance_period=3, normalization_method='relative_price'):
    """
    创建 Alpha 13 因子 (币圈7×24h优化版)
    
    参数:
    covariance_window: 协方差计算窗口，默认为5天
    rebalance_period: 调仓周期，默认为3天
    normalization_method: 归一化方法，可选: 'relative_price', 'market_cap', 'price_change'
    
    原理:
    Alpha 13 = (-1 * rank(covariance(rank(normalized_close), rank(normalized_volume), 5)))
    
    修复未来函数问题:
    1. 先按交易对分组计算时序协方差
    2. 再按日期分组进行横截面排名
    3. 确保每个时间点的计算只使用历史信息
    
    该因子通过分析归一化后收盘价排名与交易量排名之间的协方差，寻找可能的价格反转机会
    """
    print(f"开始构建 Alpha 13 因子 (币圈7×24h优化版)...")
    print(f"参数: covariance_window={covariance_window}, rebalance_period={rebalance_period}, normalization_method={normalization_method}")
    
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
    def calculate_alpha13_factor(group):
        if len(group) < covariance_window + rebalance_period + 20:  # 增加窗口用于归一化
            # 数据不足，返回空DataFrame
            return pd.DataFrame()
        
        # 根据归一化方法处理价格和交易量
        if normalization_method == 'relative_price':
            # 方法1: 使用相对价格 (当前价格 / 过去20天平均价格)
            group['normalized_close'] = group['close'] / group['close'].rolling(window=20, min_periods=20).mean()
            group['normalized_volume'] = group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
            
        elif normalization_method == 'market_cap':
            # 方法2: 使用市值加权 (如果有市值数据)
            # 这里使用交易量作为市值的近似
            group['normalized_close'] = group['close'] * group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
            group['normalized_volume'] = group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
            
        elif normalization_method == 'price_change':
            # 方法3: 使用价格变化率
            group['normalized_close'] = group['close'].pct_change(periods=5)  # 5日价格变化率
            group['normalized_volume'] = group['volume'].pct_change(periods=5)  # 5日交易量变化率
            
        else:
            # 默认使用相对价格方法
            group['normalized_close'] = group['close'] / group['close'].rolling(window=20, min_periods=20).mean()
            group['normalized_volume'] = group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
        
        # 计算未来对数收益率 (为因子分析提供)，使用调仓周期
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'normalized_close', 'normalized_volume', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 13 因子基础数据...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha13_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    
    # 合并结果
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
        
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    print("计算时序协方差...")
    
    # 步骤1: 先按交易对分组计算时序协方差（修复：使用原始归一化数据）
    def calculate_covariance(group):
        """计算每个交易对的时序协方差"""
        if len(group) < covariance_window:
            return pd.DataFrame()
        
        # 计算过去N天归一化价格与归一化交易量的协方差
        group['covariance'] = group['normalized_close'].rolling(window=covariance_window, min_periods=covariance_window).cov(group['normalized_volume'])
        
        return group
    
    # 按交易对分组计算协方差
    result_dfs = []
    for symbol, group in tqdm(factor_df.groupby('symbol')):
        cov_result = calculate_covariance(group)
        if not cov_result.empty:
            result_dfs.append(cov_result)
    
    if not result_dfs:
        print("警告: 没有足够的数据计算协方差")
        return pd.DataFrame()
    
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['covariance'])
    
    print("计算协方差排名和信号反转...")
    
    # 步骤2: 再按日期分组进行横截面排名（修复：对协方差进行排名）
    def calculate_final_factor(df):
        """计算最终因子值"""
        df = df.copy()
        
        # 对协方差进行横截面排名
        df['covariance_rank'] = df['covariance'].rank(pct=True)
        
        # 信号反转：(-1 * rank(covariance))
        df['alpha13_raw'] = -1 * df['covariance_rank']
        
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
        mean = df['alpha13_raw'].mean()
        std = df['alpha13_raw'].std()
        df['alpha13_raw'] = df['alpha13_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha13_raw'].rank(pct=True)
        df['alpha13_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha13_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha13_covariance_rank_cov{covariance_window}d_{normalization_method}_{today}.csv")
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
    # === Alpha 13 因子参数优化测试 ===
    
    # 策略1: 标准参数 + 相对价格归一化 (推荐) - 修复未来函数版本
    create_alpha13_factor(covariance_window=20, rebalance_period=10, normalization_method='relative_price')
    
    # 策略2: 短期策略 + 价格变化率归一化
    # create_alpha13_factor(covariance_window=3, rebalance_period=2, normalization_method='price_change')
    
    # 策略3: 中期策略 + 市值加权归一化
    # create_alpha13_factor(covariance_window=7, rebalance_period=5, normalization_method='market_cap')
    
    # 策略4: 超短期策略 + 相对价格归一化
    # create_alpha13_factor(covariance_window=2, rebalance_period=1, normalization_method='relative_price')
    
    # 策略5: 长期策略 + 相对价格归一化
    # create_alpha13_factor(covariance_window=10, rebalance_period=7, normalization_method='relative_price')
