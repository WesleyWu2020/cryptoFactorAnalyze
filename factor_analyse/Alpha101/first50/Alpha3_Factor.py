import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

def create_alpha3_factor(correlation_window=10, rebalance_period=5, price_normalization='market_cap'):
    """
    创建 Alpha 3 因子 (币圈优化版)
    严格避免未来函数，只使用上一期已出现的token
    
    参数:
    correlation_window: 相关性计算窗口，默认为10天
    rebalance_period: 调仓周期，默认为5天
    price_normalization: 价格归一化方法，可选: 'market_cap', 'price_change', 'relative_price'
    
    原理:
    Alpha 3 = (-1 * correlation(rank(normalized_open), rank(volume), 10))
    
    核心步骤:
    1. 价格归一化：根据币圈特性对开盘价进行归一化处理
    2. 横截面排名标准化：对归一化价格和交易量分别进行横截面排名
    3. 相关性分析：计算过去10天两排名序列的相关性
    4. 信号方向反转：对相关系数取反，捕捉量价背离的机会
    5. 🔥 关键：结合历史市值排名，确保只使用上一期已出现的token
    
    该因子是一个基于价量关系的中期反转型因子，特别针对币圈价格差异巨大的特性进行了优化
    """
    print(f"开始构建 Alpha 3 因子 (币圈优化版)...")
    print(f"参数: correlation_window={correlation_window}, rebalance_period={rebalance_period}")
    print(f"价格归一化方法: {price_normalization}")
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # 回到Crypto目录
    
    #  关键：加载历史市值排名数据，避免未来函数
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
    def calculate_alpha3_factor_with_availability(group, symbol, available_tokens_by_date):
        if len(group) < correlation_window + rebalance_period:
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
        
        # 计算市值 (价格 * 交易量，作为市值的近似)
        group['market_value'] = group['open'] * group['volume']
        
        # 计算价格变化率 (相对于前一天)
        group['price_change'] = group['open'].pct_change()
        
        # 计算相对价格 (相对于历史均值)
        group['price_mean'] = group['open'].rolling(window=20, min_periods=1).mean()
        group['relative_price'] = group['open'] / group['price_mean']
        
        # 计算未来对数收益率 (为因子分析提供)，使用调仓周期
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'open', 'volume', 'market_value', 
                      'price_change', 'relative_price', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 3 因子基础数据（避免未来函数）...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha3_factor_with_availability(group, symbol, available_tokens_by_date)
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
        
        # 根据选择的归一化方法处理价格
        if price_normalization == 'market_cap':
            # 使用市值代替绝对价格
            df['price_rank'] = df['market_value'].rank(pct=True)
        elif price_normalization == 'price_change':
            # 使用价格变化率
            df['price_rank'] = df['price_change'].rank(pct=True)
        elif price_normalization == 'relative_price':
            # 使用相对价格（相对于自身历史均值）
            df['price_rank'] = df['relative_price'].rank(pct=True)
        else:
            # 默认使用对数变换的价格
            df['log_open'] = np.log(df['open'])
            df['price_rank'] = df['log_open'].rank(pct=True)
        
        # 交易量排名
        df['volume_rank'] = df['volume'].rank(pct=True)
        
        return df
    
    # 按日期分组进行横截面排名
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_cross_sectional_rank)
    
    print("计算时序相关性...")
    
    # 步骤2: 计算每个交易对的时序相关性
    def calculate_temporal_correlation(group):
        """计算每个交易对的时序相关性"""
        group = group.copy()
        if len(group) < correlation_window:
            group['alpha3_factor'] = np.nan
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
            group['price_rank'], 
            group['volume_rank'], 
            correlation_window
        )
        
        # 步骤3: 信号方向反转 (-1 * correlation)
        group['alpha3_factor'] = -1 * correlation
        
        return group
    
    # 按交易对分组计算时序相关性
    factor_df = factor_df.groupby('symbol', group_keys=False).apply(calculate_temporal_correlation)
    
    # 删除NaN值
    factor_df = factor_df.dropna(subset=['alpha3_factor'])
    
    print("处理极端值和归一化...")
    
    # 检查数据结构
    print(f"数据列: {factor_df.columns.tolist()}")
    print(f"数据形状: {factor_df.shape}")
    
    # 处理极端值
    # 计算每日因子值的上下限(3个标准差)
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha3_factor'].mean()
        std = df['alpha3_factor'].std()
        df['alpha3_factor'] = df['alpha3_factor'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)
    
    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        """使用排序归一化到-1到1之间"""
        df = df.copy()
        rank_pct = df['alpha3_factor'].rank(pct=True)
        df['alpha3_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha3_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha3_{price_normalization}_corr{correlation_window}d_rebalance{rebalance_period}d_{today}.csv")
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
    # 测试不同的价格归一化方法
    
    # 方法1: 使用市值排名 (推荐)
    # create_alpha3_factor(correlation_window=7, rebalance_period=3, price_normalization='market_cap')
    
    # 方法2: 使用价格变化率排名
    # create_alpha3_factor(correlation_window=20, rebalance_period=10, price_normalization='price_change')
    
    # 方法3: 使用相对价格排名
    create_alpha3_factor(correlation_window=20, rebalance_period=10, price_normalization='relative_price')

