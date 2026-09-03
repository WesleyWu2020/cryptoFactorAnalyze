import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha19_factor(delay_window=7, returns_sum_window=250, rebalance_period=7, normalization_method='relative_price'):
    """
    创建 Alpha 19 因子 (币圈7×24h优化版)
    
    参数:
    delay_window: 延迟窗口，默认为7天
    returns_sum_window: 收益率累积窗口，默认为250天
    rebalance_period: 调仓周期，默认为7天
    normalization_method: 归一化方法，可选: 'relative_price', 'market_cap', 'price_change'
    
    原理:
    Alpha 19 = ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
    
    核心逻辑:
    1. -1 * sign(((close - delay(close, 7)) + delta(close, 7))): 短期价格方向反转信号
    2. (1 + rank((1 + sum(returns, 250)))): 长期累积收益强度调整
    3. 两部分相乘，形成复合信号
    
    该因子结合短期价格方向和长期累积收益，对短期价格采取反转策略，并根据长期累积收益调整信号强度
    """
    print(f"开始构建 Alpha 19 因子 (币圈7×24h优化版)...")
    print(f"参数: delay_window={delay_window}, returns_sum_window={returns_sum_window}, rebalance_period={rebalance_period}, normalization_method={normalization_method}")
    
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
    
    # 创建因子计算函数
    def calculate_alpha19_factor(group):
        if len(group) < max(delay_window, returns_sum_window) + rebalance_period + 20:
            return pd.DataFrame()
        
        # 根据归一化方法处理价格
        if normalization_method == 'relative_price':
            # 方法1: 使用相对价格 (当前价格 / 过去20天平均价格)
            group['normalized_close'] = group['close'] / group['close'].rolling(window=20, min_periods=20).mean()
            
        elif normalization_method == 'market_cap':
            # 方法2: 使用市值加权 (如果有市值数据)
            # 这里使用交易量作为市值的近似
            group['normalized_close'] = group['close'] * group['volume'] / group['volume'].rolling(window=20, min_periods=20).mean()
            
        elif normalization_method == 'price_change':
            # 方法3: 使用价格变化率
            group['normalized_close'] = group['close'].pct_change(periods=5)  # 5日价格变化率
            
        else:
            # 默认使用相对价格方法
            group['normalized_close'] = group['close'] / group['close'].rolling(window=20, min_periods=20).mean()
        
        # 计算未来对数收益率 (为因子分析提供)，使用调仓周期
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'normalized_close', 'close', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 19 因子基础数据...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha19_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    
    # 合并结果
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
        
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    print("计算时序指标...")
    
    # 步骤1: 先按交易对分组计算时序指标
    def calculate_time_series_metrics(group):
        """计算每个交易对的时序指标"""
        if len(group) < max(delay_window, returns_sum_window) + 1:
            return pd.DataFrame()
        
        # 1. 计算收益率
        group['returns'] = group['close'].pct_change()
        
        # 2. 计算延迟价格: delay(close, 7)
        group['delay_close'] = group['close'].shift(delay_window)
        
        # 3. 计算价格变化: delta(close, 7)
        group['delta_close'] = group['close'].diff(delay_window)
        
        # 4. 计算价格差异: close - delay(close, 7)
        group['close_delay_diff'] = group['close'] - group['delay_close']
        
        # 5. 计算复合价格变化: (close - delay(close, 7)) + delta(close, 7)
        group['composite_price_change'] = group['close_delay_diff'] + group['delta_close']
        
        # 6. 计算价格方向信号: sign(((close - delay(close, 7)) + delta(close, 7)))
        group['price_direction_signal'] = np.sign(group['composite_price_change'])
        
        # 7. 计算反转信号: -1 * sign(...)
        group['reversal_signal'] = -1 * group['price_direction_signal']
        
        # 8. 计算长期累积收益: sum(returns, 250)
        group['cumulative_returns'] = group['returns'].rolling(window=returns_sum_window, min_periods=returns_sum_window).sum()
        
        # 9. 计算累积收益调整: 1 + sum(returns, 250)
        group['cumulative_returns_adj'] = 1 + group['cumulative_returns']
        
        return group
    
    # 按交易对分组计算时序指标
    result_dfs = []
    for symbol, group in tqdm(factor_df.groupby('symbol')):
        ts_result = calculate_time_series_metrics(group)
        if not ts_result.empty:
            result_dfs.append(ts_result)
    
    if not result_dfs:
        print("警告: 没有足够的数据计算时序指标")
        return pd.DataFrame()
    
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['reversal_signal', 'cumulative_returns_adj'])
    
    print("计算横截面排名和复合因子...")
    
    # 步骤2: 再按日期分组进行横截面排名
    def calculate_final_factor(df):
        """计算最终因子值"""
        df = df.copy()
        
        # 1. 对累积收益调整进行横截面排名: rank((1 + sum(returns, 250)))
        df['cumulative_returns_rank'] = df['cumulative_returns_adj'].rank(pct=True)
        
        # 2. 计算长期收益强度调整: 1 + rank((1 + sum(returns, 250)))
        df['long_term_strength'] = 1 + df['cumulative_returns_rank']
        
        # 3. 复合因子: 反转信号 * 长期收益强度调整
        df['alpha19_raw'] = df['reversal_signal'] * df['long_term_strength']
        
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
        mean = df['alpha19_raw'].mean()
        std = df['alpha19_raw'].std()
        df['alpha19_raw'] = df['alpha19_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha19_raw'].rank(pct=True)
        df['alpha19_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha19_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha19_reversal_momentum_delay{delay_window}d_returns{returns_sum_window}d_{normalization_method}_rebalance{rebalance_period}d_{today}.csv")
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
    # === Alpha 19 因子参数优化测试 ===
    
    # 策略1: 标准参数 + 相对价格归一化 (推荐)
    # create_alpha19_factor(delay_window=5, returns_sum_window=20, rebalance_period=10, normalization_method='relative_price')
    
    # 策略2: 短期策略 + 价格变化率归一化
    # create_alpha19_factor(delay_window=3, returns_sum_window=100, rebalance_period=3, normalization_method='price_change')
    
    # 策略3: 中期策略 + 市值加权归一化
    # create_alpha19_factor(delay_window=10, returns_sum_window=180, rebalance_period=10, normalization_method='market_cap')
    
    # 策略4: 超短期策略 + 相对价格归一化
    create_alpha19_factor(delay_window=5, returns_sum_window=60, rebalance_period=10, normalization_method='relative_price')
    
    # 策略5: 长期策略 + 相对价格归一化
    # create_alpha19_factor(delay_window=14, returns_sum_window=365, rebalance_period=14, normalization_method='relative_price')
