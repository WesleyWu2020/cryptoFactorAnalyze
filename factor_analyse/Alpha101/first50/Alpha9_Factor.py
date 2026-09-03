import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha9_factor(trend_window=5, rebalance_period=2):
    """
    创建 Alpha 9 因子 (币圈7×24h优化版)
    
    参数:
    trend_window: 趋势判断窗口，默认为5天
    rebalance_period: 调仓周期，默认为2天
    
    原理:
    Alpha 9 = ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : 
               ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    
    币圈7×24h特殊设计:
    1. 趋势判断：通过过去N天的收益率判断市场趋势状态
    2. 自适应策略：
       - 强上涨趋势：动量策略（跟随趋势）
       - 强下跌趋势：动量策略（跟随趋势）
       - 混合/横盘：反转策略（逆向操作）
    3. 连续交易适应：适应7×24h连续交易的快速变化
    
    该因子根据市场条件自适应选择策略，在强趋势中跟随，在震荡中反转
    """
    print(f"开始构建 Alpha 9 因子 (币圈7×24h优化版)...")
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
    def calculate_alpha9_factor(group):
        if len(group) < trend_window + rebalance_period + 1:
            # 数据不足，返回空DataFrame
            return pd.DataFrame()
        
        # === Alpha 9 因子计算 ===
        
        # 1. 计算日收益率 delta(close, 1)
        group['daily_return'] = group['close'].pct_change()
        
        # 2. 计算过去trend_window天收益率的最小值和最大值
        # ts_min(delta(close, 1), trend_window): 过去N天收益率的最小值
        # ts_max(delta(close, 1), trend_window): 过去N天收益率的最大值
        group['ts_min_return'] = group['daily_return'].rolling(window=trend_window).min()
        group['ts_max_return'] = group['daily_return'].rolling(window=trend_window).max()
        
        # 3. 实现Alpha 9的条件逻辑
        # 条件1: 强上涨趋势 (0 < ts_min(delta(close, 1), 5))
        # 意思是过去5天的收益率最小值都大于0，即所有天都是正收益
        condition_strong_uptrend = group['ts_min_return'] > 0
        
        # 条件2: 强下跌趋势 (ts_max(delta(close, 1), 5) < 0)
        # 意思是过去5天的收益率最大值都小于0，即所有天都是负收益
        condition_strong_downtrend = group['ts_max_return'] < 0
        
        # 条件3: 混合/横盘趋势 (既不是强上涨也不是强下跌)
        condition_mixed_trend = ~(condition_strong_uptrend | condition_strong_downtrend)
        
        # 根据条件分配因子值
        group['alpha9_factor'] = np.where(
            condition_strong_uptrend,
            group['daily_return'],  # 强上涨：跟随趋势
            np.where(
                condition_strong_downtrend,
                group['daily_return'],  # 强下跌：跟随趋势
                -1 * group['daily_return']  # 混合：反转策略
            )
        )
        
        # 记录市场状态（用于分析）
        group['market_state'] = np.where(
            condition_strong_uptrend, 'strong_up',
            np.where(condition_strong_downtrend, 'strong_down', 'mixed')
        )
        
        # 计算未来对数收益率 (为因子分析提供)
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'alpha9_factor', 'daily_return', 'market_state', 
                       'ts_min_return', 'ts_max_return', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 9 因子基础数据...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha9_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    
    # 合并结果
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
        
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    print("分析市场状态分布...")
    
    # 分析市场状态分布
    state_distribution = factor_df['market_state'].value_counts()
    total_observations = len(factor_df)
    
    print(f"市场状态分布:")
    for state, count in state_distribution.items():
        percentage = count / total_observations * 100
        print(f"  {state}: {count} ({percentage:.1f}%)")
    
    print("处理极端值和归一化...")
    
    # 检查数据结构
    print(f"数据列: {factor_df.columns.tolist()}")
    print(f"数据形状: {factor_df.shape}")
    
    # 处理极端值
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha9_factor'].mean()
        std = df['alpha9_factor'].std()
        df['alpha9_factor'] = df['alpha9_factor'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha9_factor'].rank(pct=True)
        df['alpha9_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha9_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha9_adaptive_trend{trend_window}d_{today}.csv")
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
    # === Alpha 9 因子测试 (币圈7×24h优化版) ===
    
    # 策略1: 标准参数 (推荐)
    # create_alpha9_factor(trend_window=5, rebalance_period=2)
    
    # 策略2: 缩短趋势窗口 (适应币圈高频变化)
    create_alpha9_factor(trend_window=10, rebalance_period=10)
    
    # 策略3: 延长趋势窗口 (更稳定的趋势判断)
    # create_alpha9_factor(trend_window=7, rebalance_period=3)
    
    # 策略4: 超短期策略 (日内交易)
    # create_alpha9_factor(trend_window=2, rebalance_period=1)
