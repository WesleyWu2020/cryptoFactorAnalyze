import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

def create_alpha15_factor(correlation_window=3, sum_window=3, rebalance_period=3, normalization_method='relative_price'):
    """
    创建 Alpha 15 因子 (币圈优化版)
    严格避免未来函数，只使用上一期已出现的token
    
    参数:
    correlation_window: 相关性计算窗口，默认为3天
    sum_window: 求和窗口，默认为3天
    rebalance_period: 调仓周期，默认为3天
    normalization_method: 价格归一化方法，默认为'relative_price'
    
    原理:
    Alpha 15 = (-1 * rank(sum(delta(close, 1), 3)) * rank(correlation(volume, close, 3)))
    
    核心步骤:
    1. 价格归一化：适应币圈巨大的价格差异
    2. 价格变化计算：计算收盘价的日变化
    3. 相关性分析：计算交易量与收盘价的相关性
    4. 复合信号：将价格变化求和排名与相关性排名相乘，然后取反
    5. 🔥 关键：结合历史市值排名，确保只使用上一期已出现的token
    
    该因子是一个基于价格动量和量价相关性的复合因子，特别针对币圈特性进行了优化
    """
    print(f"开始构建 Alpha 15 因子 (币圈优化版)...")
    print(f"参数: correlation_window={correlation_window}, sum_window={sum_window}, rebalance_period={rebalance_period}")
    print(f"价格归一化方法: {normalization_method}")
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))  # 回到Crypto目录
    
    # �� 关键：加载历史市值排名数据，避免未来函数
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
    def calculate_alpha15_factor_with_availability(group, symbol, available_tokens_by_date):
        if len(group) < max(correlation_window, sum_window) + rebalance_period + 20:  # +20是为了计算相对价格的历史均值
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
        
        # 价格归一化处理（适应币圈价格差异）
        if normalization_method == 'relative_price':
            # 计算相对价格基准（相对于自身历史均值）
            group['price_base'] = group['close'].rolling(window=20, min_periods=1).mean()
            group['normalized_close'] = group['close'] / group['price_base']
            group['normalized_volume'] = group['volume'] / group['volume'].rolling(window=20, min_periods=1).mean()
        elif normalization_method == 'log_price':
            # 使用对数价格
            group['normalized_close'] = np.log(group['close'])
            group['normalized_volume'] = np.log(group['volume'])
        else:  # 默认使用原始价格
            group['normalized_close'] = group['close']
            group['normalized_volume'] = group['volume']
        
        # 计算价格变化 (delta(close, 1))
        group['price_delta'] = group['normalized_close'].diff()
        
        # 计算未来对数收益率 (为因子分析提供)
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'normalized_close', 'normalized_volume', 'price_delta', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 15 因子基础数据（避免未来函数）...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha15_factor_with_availability(group, symbol, available_tokens_by_date)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    
    # 合并结果
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
        
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    print("计算时序相关性...")
    
    # 步骤1: 计算每个交易对的时序相关性
    def calculate_correlation(group):
        """计算每个交易对的时序相关性"""
        group = group.copy()
        if len(group) < correlation_window:
            group['correlation'] = np.nan
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
            group['normalized_volume'], 
            group['normalized_close'], 
            correlation_window
        )
        
        group['correlation'] = correlation
        
        return group
    
    # 按交易对分组计算时序相关性
    factor_df = factor_df.groupby('symbol', group_keys=False).apply(calculate_correlation)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['correlation'])
    
    print("计算价格变化求和...")
    
    # 步骤2: 计算价格变化的求和排名
    def calculate_correlation_rank(df):
        """按日期计算相关性排名"""
        df = df.copy()
        
        # 对相关性进行排名
        df['correlation_rank'] = df['correlation'].rank(pct=True)
        
        return df
    
    # 按日期分组计算相关性排名
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_correlation_rank)
    
    # 步骤3: 计算价格变化求和
    def calculate_sum(group):
        """计算每个交易对的价格变化求和"""
        group = group.copy()
        if len(group) < sum_window:
            group['price_delta_sum'] = np.nan
            return group
        
        # 计算滚动求和
        def rolling_sum(x, window):
            """计算滚动求和"""
            sum_values = []
            for i in range(len(x)):
                if i < window - 1:
                    sum_values.append(np.nan)
                else:
                    # 取过去window天的数据
                    x_window = x.iloc[i-window+1:i+1]
                    
                    # 计算求和
                    if len(x_window) == window:
                        sum_val = x_window.sum()
                        sum_values.append(sum_val)
                    else:
                        sum_values.append(np.nan)
            
            return pd.Series(sum_values, index=x.index)
        
        # 计算过去sum_window天的价格变化求和
        price_delta_sum = rolling_sum(group['price_delta'], sum_window)
        
        group['price_delta_sum'] = price_delta_sum
        
        return group
    
    # 按交易对分组计算价格变化求和
    factor_df = factor_df.groupby('symbol', group_keys=False).apply(calculate_sum)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['price_delta_sum'])
    
    print("计算最终因子值...")
    
    # 步骤4: 计算最终因子值
    def calculate_final_factor(df):
        """按日期计算最终因子值"""
        df = df.copy()
        
        # 对价格变化求和进行排名
        df['price_delta_sum_rank'] = df['price_delta_sum'].rank(pct=True)
        
        # 计算复合因子：-1 * (price_delta_sum_rank * correlation_rank)
        df['alpha15_factor'] = -1 * (df['price_delta_sum_rank'] * df['correlation_rank'])
        
        return df
    
    # 按日期分组计算最终因子值
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_final_factor)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['alpha15_factor'])
    
    print("处理极端值和归一化...")
    
    # 检查数据结构
    print(f"数据列: {factor_df.columns.tolist()}")
    print(f"数据形状: {factor_df.shape}")
    
    # 处理极端值
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha15_factor'].mean()
        std = df['alpha15_factor'].std()
        df['alpha15_factor'] = df['alpha15_factor'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha15_factor'].rank(pct=True)
        df['alpha15_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha15_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha15_{normalization_method}_corr{correlation_window}d_sum{sum_window}d_rebalance{rebalance_period}d_{today}.csv")
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
    # === Alpha 15 因子测试 (币圈优化版) ===
    
    # 策略1: 使用相对价格归一化 (推荐)
    create_alpha15_factor(correlation_window=10, sum_window=10, rebalance_period=5, normalization_method='relative_price')
    
    # 策略2: 缩短窗口 (适应币圈高频特性)
    # create_alpha15_factor(correlation_window=5, sum_window=5, rebalance_period=3, normalization_method='relative_price')
    
    # 策略3: 使用对数价格 (处理极端价格差异)
    # create_alpha15_factor(correlation_window=10, sum_window=10, rebalance_period=5, normalization_method='log_price')