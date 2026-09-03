import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha10_factor(trend_window=4, rebalance_period=2):
    """
    创建 Alpha 10 因子 (币圈7×24h优化版)
    
    参数:
    trend_window: 趋势判断窗口，默认为4天
    rebalance_period: 调仓周期，默认为2天
    
    原理:
    Alpha 10 = rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : 
                    ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
    
    核心步骤:
    1. 趋势判断：通过过去4天的收益率判断市场趋势状态
    2. 自适应策略：根据趋势状态选择动量或反转策略
    3. 横截面排名：对策略结果进行排名，强调相对表现
    4. 归一化：将因子值归一化到-1到1之间
    
    该因子是Alpha9的改进版本，通过排名操作更适合构建多空组合和相对价值策略
    """
    print(f"开始构建 Alpha 10 因子 (币圈7×24h优化版)...")
    print(f"参数: trend_window={trend_window}, rebalance_period={rebalance_period}")
    
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
    def calculate_alpha10_factor(group):
        if len(group) < trend_window + rebalance_period + 5:
            # 数据不足，返回空DataFrame
            return pd.DataFrame()
        
        # 计算日收益率
        group['returns'] = group['close'].pct_change()
        
        # 计算过去N天收益率的最小值和最大值（时序统计）
        group['ts_min_returns'] = group['returns'].rolling(window=trend_window, min_periods=trend_window).min()
        group['ts_max_returns'] = group['returns'].rolling(window=trend_window, min_periods=trend_window).max()
        
        # 计算未来对数收益率 (为因子分析提供)，使用调仓周期
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 实现自适应策略逻辑
        def adaptive_strategy(row):
            if pd.isna(row['ts_min_returns']) or pd.isna(row['ts_max_returns']) or pd.isna(row['returns']):
                return np.nan
            
            # 强上涨趋势：过去N天所有收益都为正
            if row['ts_min_returns'] > 0:
                return row['returns']  # 动量策略
            
            # 强下跌趋势：过去N天所有收益都为负
            elif row['ts_max_returns'] < 0:
                return row['returns']  # 动量策略
            
            # 混合/横盘市场：既有正收益又有负收益
            else:
                return -1 * row['returns']  # 反转策略
        
        # 应用自适应策略
        group['alpha10_raw'] = group.apply(adaptive_strategy, axis=1)
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'alpha10_raw', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 10 因子基础数据...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha10_factor(group)
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
        
        # 对自适应策略结果进行排名
        df['alpha10_rank'] = df['alpha10_raw'].rank(pct=True)
        
        return df
    
    # 按日期分组进行横截面排名
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_cross_sectional_rank)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['alpha10_rank'])
    
    print("处理极端值和归一化...")
    
    # 检查数据结构
    print(f"数据列: {factor_df.columns.tolist()}")
    print(f"数据形状: {factor_df.shape}")
    
    # 处理极端值
    # 计算每日因子值的上下限(3个标准差)
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha10_rank'].mean()
        std = df['alpha10_rank'].std()
        df['alpha10_rank'] = df['alpha10_rank'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha10_rank'].rank(pct=True)
        df['alpha10_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha10_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha10_adaptive_trend{trend_window}d_{today}.csv")
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
    # === Alpha 10 因子参数优化测试 ===
    
    # 策略1: 标准参数 (推荐)
    # create_alpha10_factor(trend_window=20, rebalance_period=10)
    
    # 策略2: 缩短趋势窗口 (适应币圈高频变化)
    # create_alpha10_factor(trend_window=3, rebalance_period=1)
    
    # 策略3: 延长趋势窗口 (更稳定的趋势判断)
    # create_alpha10_factor(trend_window=6, rebalance_period=3)
    
    # 策略4: 超短期策略 (日内交易)
    # create_alpha10_factor(trend_window=2, rebalance_period=1)
    
    # 策略5: 中期策略 (波段交易)
    create_alpha10_factor(trend_window=7, rebalance_period=5)

