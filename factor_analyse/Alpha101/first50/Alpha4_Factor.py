import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha4_factor(ts_rank_window=9, rebalance_period=3, price_normalization='relative_price'):
    """
    创建 Alpha 4 因子 (币圈优化版)
    
    参数:
    ts_rank_window: 时序排名窗口，默认为9天
    rebalance_period: 调仓周期，默认为3天
    price_normalization: 价格归一化方法，默认为'relative_price'
    
    原理:
    Alpha 4 = (-1 * Ts_Rank(rank(low), 9))
    
    核心步骤:
    1. 价格归一化：根据币圈特性对最低价进行归一化处理
    2. 横截面排名：对归一化最低价进行横截面排名
    3. 时序趋势分析：计算过去9天最低价排名的时间序列排名
    4. 信号方向反转：对时序排名结果取反，捕捉价格支撑机会
    
    该因子是一个基于价格支撑水平的短期动量型因子，特别适用于风险管理和择时交易
    """
    print(f"开始构建 Alpha 4 因子 (币圈优化版)...")
    print(f"参数: ts_rank_window={ts_rank_window}, rebalance_period={rebalance_period}")
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
    def calculate_alpha4_factor(group):
        if len(group) < ts_rank_window + rebalance_period + 20:  # +20是为了计算相对价格的历史均值
            # 数据不足，返回空DataFrame
            return pd.DataFrame()
        
        # 计算市值 (价格 * 交易量，作为市值的近似)
        group['market_value'] = group['low'] * group['volume']
        
        # 计算价格变化率 (相对于前一天)
        group['price_change'] = group['low'].pct_change()
        
        # 计算相对价格 (相对于历史均值) - 使用20天均值
        group['price_mean'] = group['low'].rolling(window=20, min_periods=1).mean()
        group['relative_price'] = group['low'] / group['price_mean']
        
        # 计算未来对数收益率 (为因子分析提供)，使用调仓周期
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'low', 'volume', 'market_value', 
                      'price_change', 'relative_price', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 4 因子基础数据...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha4_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    
    # 合并结果
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
        
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    print("计算横截面排名...")
    
    # 步骤1: 横截面排名标准化（使用归一化价格）
    def calculate_cross_sectional_rank(df):
        """对每个日期的数据进行横截面排名"""
        df = df.copy()
        
        # 根据选择的归一化方法处理最低价
        if price_normalization == 'market_cap':
            # 使用市值代替绝对价格
            df['low_rank'] = df['market_value'].rank(pct=True)
        elif price_normalization == 'price_change':
            # 使用价格变化率
            df['low_rank'] = df['price_change'].rank(pct=True)
        elif price_normalization == 'relative_price':
            # 使用相对价格（相对于自身历史均值）
            df['low_rank'] = df['relative_price'].rank(pct=True)
        else:
            # 默认使用对数变换的价格
            df['log_low'] = np.log(df['low'])
            df['low_rank'] = df['log_low'].rank(pct=True)
        
        return df
    
    # 按日期分组进行横截面排名
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_cross_sectional_rank)
    
    print("计算时序排名...")
    
    # 步骤2: 计算每个交易对的时序排名 (Ts_Rank)
    def calculate_ts_rank(group):
        """计算每个交易对的时序排名"""
        group = group.copy()
        if len(group) < ts_rank_window:
            group['ts_rank'] = np.nan
            return group
        
        # 计算时序排名的函数
        def rolling_ts_rank(series, window):
            """计算滚动时序排名"""
            ts_rank_values = []
            for i in range(len(series)):
                if i < window - 1:
                    ts_rank_values.append(np.nan)
                else:
                    # 取过去window天的数据
                    window_data = series.iloc[i-window+1:i+1]
                    
                    # 计算当前值在过去window天中的排名百分位
                    if len(window_data) == window:
                        current_value = window_data.iloc[-1]
                        rank_pct = (window_data < current_value).sum() / (window - 1)
                        ts_rank_values.append(rank_pct)
                    else:
                        ts_rank_values.append(np.nan)
            
            return pd.Series(ts_rank_values, index=series.index)
        
        # 计算过去ts_rank_window天的时序排名
        ts_rank = rolling_ts_rank(group['low_rank'], ts_rank_window)
        
        # 步骤3: 信号方向反转 (-1 * Ts_Rank)
        group['alpha4_factor'] = -1 * ts_rank
        
        return group
    
    # 按交易对分组计算时序排名
    factor_df = factor_df.groupby('symbol', group_keys=False).apply(calculate_ts_rank)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['alpha4_factor'])
    
    print("处理极端值和归一化...")
    
    # 检查数据结构
    print(f"数据列: {factor_df.columns.tolist()}")
    print(f"数据形状: {factor_df.shape}")
    
    # 处理极端值
    # 计算每日因子值的上下限(3个标准差)
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha4_factor'].mean()
        std = df['alpha4_factor'].std()
        df['alpha4_factor'] = df['alpha4_factor'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha4_factor'].rank(pct=True)
        df['alpha4_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha4_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha4_{price_normalization}_tsrank{ts_rank_window}d_{today}.csv")
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
    # === Alpha 4 因子测试 ===
    
    # 策略1: 使用相对价格排名 (推荐)
    create_alpha4_factor(ts_rank_window=9, rebalance_period=3, price_normalization='relative_price')
 
