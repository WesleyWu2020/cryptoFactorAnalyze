import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha8_factor(sum_window=5, delay_window=10, rebalance_period=7, price_normalization='relative_price'):
    """
    创建 Alpha 8 因子 (币圈7×24h优化版)
    
    参数:
    sum_window: 求和窗口，默认为5天
    delay_window: 延迟对比窗口，默认为10天
    rebalance_period: 调仓周期，默认为7天
    price_normalization: 价格归一化方法，默认为'relative_price'
    
    原理:
    Alpha 8 = (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
    
    币圈7×24h特殊设计:
    1. 价格-动量组合：结合短期价格水平与动量方向
    2. 历史对比分析：与历史同期对比，识别信号变化
    3. 反转逻辑：信号显著强于历史时看空，弱于历史时看多
    4. 价格归一化：适应币圈巨大的价格差异
    
    该因子适用于捕捉市场情绪转变和过度反应后的反转机会
    """
    print(f"开始构建 Alpha 8 因子 (币圈7×24h优化版)...")
    print(f"参数: sum_window={sum_window}, delay_window={delay_window}, rebalance_period={rebalance_period}")
    print(f"价格归一化方法: {price_normalization}")
    
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
    def calculate_alpha8_factor(group):
        if len(group) < sum_window + delay_window + rebalance_period + 20:  # +20是为了计算相对价格的历史均值
            # 数据不足，返回空DataFrame
            return pd.DataFrame()
        
        # === 计算收益率 ===
        group['returns'] = group['close'].pct_change()
        
        # === 价格归一化处理（适应币圈价格差异）===
        # 计算相对价格基准
        group['price_base'] = group['close'].rolling(window=20, min_periods=1).mean()
        
        # 归一化开盘价
        if price_normalization == 'relative_price':
            # 相对于自身历史均值的比例
            group['normalized_open'] = group['open'] / group['price_base']
        elif price_normalization == 'log_price':
            # 对数变换处理极端价格差异
            group['normalized_open'] = np.log(group['open'])
        elif price_normalization == 'z_score':
            # 标准化处理
            rolling_mean = group['open'].rolling(window=20, min_periods=1).mean()
            rolling_std = group['open'].rolling(window=20, min_periods=1).std()
            group['normalized_open'] = (group['open'] - rolling_mean) / (rolling_std + 1e-10)
        else:  # 默认使用原始价格
            group['normalized_open'] = group['open']
        
        # === Alpha 8 因子计算 ===
        
        # 1. 计算5日开盘价总和
        group['sum_open'] = group['normalized_open'].rolling(window=sum_window).sum()
        
        # 2. 计算5日收益率总和
        group['sum_returns'] = group['returns'].rolling(window=sum_window).sum()
        
        # 3. 计算价格-动量组合信号
        group['price_momentum_signal'] = group['sum_open'] * group['sum_returns']
        
        # 4. 计算延迟的价格-动量组合信号（10天前）
        group['delayed_signal'] = group['price_momentum_signal'].shift(delay_window)
        
        # 5. 计算当前信号与历史信号的差异
        group['signal_diff'] = group['price_momentum_signal'] - group['delayed_signal']
        
        # 计算未来对数收益率 (为因子分析提供)
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'signal_diff', 'price_momentum_signal', 'delayed_signal', 'future_ret']].copy()
        
        # 删除NaN值和无穷值
        result = result.replace([np.inf, -np.inf], np.nan).dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 8 因子基础数据...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha8_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    
    # 合并结果
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
        
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    print("计算横截面排名和因子值...")
    
    # 步骤1: 计算Alpha 8因子
    def calculate_alpha8_by_date(df):
        """按日期计算Alpha 8因子值"""
        df = df.copy()
        
        # Alpha 8: (-1 * rank(signal_diff))
        # 对信号差异进行横截面排名，然后取反
        df['signal_diff_rank'] = df['signal_diff'].rank(pct=True)
        df['alpha8_factor'] = -1 * df['signal_diff_rank']
        
        return df
    
    # 按日期分组计算因子值
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_alpha8_by_date)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['alpha8_factor'])
    
    print("处理极端值和归一化...")
    
    # 检查数据结构
    print(f"数据列: {factor_df.columns.tolist()}")
    print(f"数据形状: {factor_df.shape}")
    
    # 分析信号差异的分布
    print(f"信号差异统计:")
    print(f"  正差异比例: {(factor_df['signal_diff'] > 0).mean():.2%}")
    print(f"  负差异比例: {(factor_df['signal_diff'] < 0).mean():.2%}")
    print(f"  零差异比例: {(factor_df['signal_diff'] == 0).mean():.2%}")
    
    # 处理极端值
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha8_factor'].mean()
        std = df['alpha8_factor'].std()
        df['alpha8_factor'] = df['alpha8_factor'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha8_factor'].rank(pct=True)
        df['alpha8_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha8_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha8_{price_normalization}_sum{sum_window}d_delay{delay_window}d_{today}.csv")
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
    # === Alpha 8 因子测试 (币圈7×24h优化版) ===
    
    # 策略1: 标准参数 (推荐)
    # create_alpha8_factor(sum_window=20, delay_window=10, rebalance_period=10, price_normalization='relative_price')
    
    # 策略2: 缩短窗口 (适应币圈高频特性)
    create_alpha8_factor(sum_window=3, delay_window=7, rebalance_period=5, price_normalization='relative_price')
    
    # 策略3: 使用对数价格 (处理极端价格差异)
    # create_alpha8_factor(sum_window=5, delay_window=10, rebalance_period=7, price_normalization='log_price')
    
    # 策略4: 使用Z-score标准化
    # create_alpha8_factor(sum_window=5, delay_window=10, rebalance_period=7, price_normalization='z_score')
    
    # 策略5: 长期策略
    # create_alpha8_factor(sum_window=7, delay_window=14, rebalance_period=10, price_normalization='relative_price')
