import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha17_factor(ts_rank_window=10, delta_window=1, volume_rank_window=5, adv_window=20, rebalance_period=5, normalization_method='relative_price'):
    """
    创建 Alpha 17 因子 (币圈7×24h优化版)
    
    参数:
    ts_rank_window: 时序排名窗口，默认为10天
    delta_window: 差分窗口，默认为1天
    volume_rank_window: 成交量排名窗口，默认为5天
    adv_window: 平均成交量窗口，默认为20天
    rebalance_period: 调仓周期，默认为5天
    normalization_method: 归一化方法，可选: 'relative_price', 'market_cap', 'price_change'
    
    原理:
    Alpha 17 = (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
    
    核心逻辑:
    1. -1 * rank(ts_rank(close, 10)): 价格强度反转信号
    2. rank(delta(delta(close, 1), 1)): 价格加速度排名
    3. rank(ts_rank((volume / adv20), 5)): 成交量相对强度排名
    4. 三个部分相乘，形成复合信号
    
    该因子结合了价格强度、价格加速度和交易量强度三个方面的信息，适用于捕捉可能的趋势转变或加速
    """
    print(f"开始构建 Alpha 17 因子 (币圈7×24h优化版)...")
    print(f"参数: ts_rank_window={ts_rank_window}, delta_window={delta_window}, volume_rank_window={volume_rank_window}, adv_window={adv_window}, rebalance_period={rebalance_period}, normalization_method={normalization_method}")
    
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
    def calculate_alpha17_factor(group):
        if len(group) < max(ts_rank_window, adv_window, volume_rank_window) + rebalance_period + 20:  # 增加窗口用于归一化
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
        result = group[['date', 'symbol', 'normalized_close', 'normalized_volume', 'close', 'volume', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 17 因子基础数据...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha17_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    
    # 合并结果
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
        
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    print("计算时序排名和差分...")
    
    # 步骤1: 先按交易对分组计算时序指标
    def calculate_time_series_metrics(group):
        """计算每个交易对的时序指标"""
        if len(group) < max(ts_rank_window, adv_window, volume_rank_window) + 2:  # 需要至少2个点做差分
            return pd.DataFrame()
        
        # 1. 计算时序排名: ts_rank(close, 10)
        group['ts_rank_close'] = group['close'].rolling(window=ts_rank_window, min_periods=ts_rank_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # 2. 计算价格加速度: delta(delta(close, 1), 1) - 二阶差分
        group['delta_close'] = group['close'].diff(delta_window)  # 一阶差分
        group['delta_delta_close'] = group['delta_close'].diff(delta_window)  # 二阶差分（价格加速度）
        
        # 3. 计算平均成交量: adv20
        group['adv'] = group['volume'].rolling(window=adv_window, min_periods=adv_window).mean()
        
        # 4. 计算成交量相对强度: volume / adv20
        group['volume_adv_ratio'] = group['volume'] / group['adv']
        
        # 5. 计算成交量相对强度的时序排名: ts_rank((volume / adv20), 5)
        group['ts_rank_volume_adv'] = group['volume_adv_ratio'].rolling(window=volume_rank_window, min_periods=volume_rank_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
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
    factor_df = factor_df.dropna(subset=['ts_rank_close', 'delta_delta_close', 'ts_rank_volume_adv'])
    
    print("计算横截面排名和复合因子...")
    
    # 步骤2: 再按日期分组进行横截面排名
    def calculate_final_factor(df):
        """计算最终因子值"""
        df = df.copy()
        
        # 1. 价格强度反转排名: rank(ts_rank(close, 10))
        df['rank_ts_rank_close'] = df['ts_rank_close'].rank(pct=True)
        
        # 2. 价格强度反转信号: -1 * rank(ts_rank(close, 10))
        df['price_strength_signal'] = -1 * df['rank_ts_rank_close']
        
        # 3. 价格加速度排名: rank(delta(delta(close, 1), 1))
        df['rank_delta_delta_close'] = df['delta_delta_close'].rank(pct=True)
        
        # 4. 成交量相对强度排名: rank(ts_rank((volume / adv20), 5))
        df['rank_ts_rank_volume_adv'] = df['ts_rank_volume_adv'].rank(pct=True)
        
        # 5. 复合因子: 三个部分相乘
        df['alpha17_raw'] = df['price_strength_signal'] * df['rank_delta_delta_close'] * df['rank_ts_rank_volume_adv']
        
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
        mean = df['alpha17_raw'].mean()
        std = df['alpha17_raw'].std()
        df['alpha17_raw'] = df['alpha17_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha17_raw'].rank(pct=True)
        df['alpha17_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha17_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha17_composite_ts{ts_rank_window}d_delta{delta_window}d_vol{volume_rank_window}d_adv{adv_window}d_{normalization_method}_rebalance{rebalance_period}d_{today}.csv")
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
    # === Alpha 17 因子参数优化测试 ===
    
    # 策略1: 标准参数 + 相对价格归一化 (推荐)
    # create_alpha17_factor(ts_rank_window=20, delta_window=20, volume_rank_window=10, adv_window=20, rebalance_period=5, normalization_method='relative_price')
    
    # 策略2: 短期策略 + 价格变化率归一化
    # create_alpha17_factor(ts_rank_window=5, delta_window=1, volume_rank_window=3, adv_window=10, rebalance_period=3, normalization_method='price_change')
    
    # 策略3: 中期策略 + 市值加权归一化
    create_alpha17_factor(ts_rank_window=15, delta_window=2, volume_rank_window=7, adv_window=30, rebalance_period=7, normalization_method='market_cap')
    
    # 策略4: 超短期策略 + 相对价格归一化
    # create_alpha17_factor(ts_rank_window=3, delta_window=1, volume_rank_window=2, adv_window=5, rebalance_period=2, normalization_method='relative_price')
    
    # 策略5: 长期策略 + 相对价格归一化
    # create_alpha17_factor(ts_rank_window=20, delta_window=3, volume_rank_window=10, adv_window=50, rebalance_period=10, normalization_method='relative_price')
