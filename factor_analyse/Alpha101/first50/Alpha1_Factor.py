import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha1_factor(stddev_window=20, argmax_window=5, rebalance_period=3):
    """
    创建 Alpha 1 因子
    
    参数:
    stddev_window: 标准差计算窗口，默认为20天
    argmax_window: 极值查找窗口，默认为5天
    rebalance_period: 调仓周期，默认为3天
    
    原理:
    Alpha 1 = (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    
    核心步骤:
    1. 若收益率（returns）为负，取过去20天收益率的标准差；若收益率为正，取收盘价
    2. 对条件结果进行平方运算，保留原始符号
    3. 找出过去5天内上述结果的最大值位置
    4. 对极值位置结果进行横截面排名，并减去0.5
    5. 归一化因子值到-1到1之间
    
    该因子捕捉极端事件时点，区分多空趋势下的风险特征
    """
    print(f"开始构建 Alpha 1 因子...")
    print(f"参数: stddev_window={stddev_window}, argmax_window={argmax_window}, rebalance_period={rebalance_period}")
    
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
    def calculate_alpha1_factor(group):
        if len(group) < stddev_window + argmax_window + rebalance_period:
            # 数据不足，返回空DataFrame
            return pd.DataFrame()
        
        # 计算日收益率
        group['returns'] = group['close'].pct_change()
        
        # 计算过去stddev_window天的收益率标准差
        group['returns_stddev'] = group['returns'].rolling(window=stddev_window).std()
        
        # 步骤1: 条件选择 - 负收益时取标准差，正收益时取收盘价
        group['conditional_value'] = np.where(
            group['returns'] < 0,
            group['returns_stddev'],
            group['close']
        )
        
        # 步骤2: 平方与符号保留 (SignedPower)
        # 保留原始符号，对绝对值进行平方
        group['signed_power'] = np.sign(group['conditional_value']) * np.abs(group['conditional_value']) ** 2
        
        # 步骤3: 时序极值定位 (Ts_ArgMax)
        # 找出过去argmax_window天内signed_power的最大值位置
        def ts_argmax(series, window):
            """计算时序极值位置"""
            if len(series) < window:
                return np.nan
            
            # 获取过去window天的数据
            recent_data = series.iloc[-window:]
            # 找到最大值的位置（从0开始计数）
            max_pos = recent_data.argmax()
            return max_pos
        
        # 使用rolling apply计算Ts_ArgMax
        group['ts_argmax'] = group['signed_power'].rolling(window=argmax_window).apply(
            lambda x: ts_argmax(x, argmax_window), raw=False
        )
        
        # 计算未来对数收益率 (为因子分析提供)，使用调仓周期
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])
        
        # 只保留需要的列
        result = group[['date', 'symbol', 'ts_argmax', 'future_ret']].copy()
        
        # 删除NaN值
        result = result.dropna()
        
        return result
    
    # 按交易对分组计算
    print("计算 Alpha 1 因子...")
    result_dfs = []
    
    # 使用tqdm显示进度
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha1_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)
    
    # 合并结果
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()
        
    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    # 处理极端值
    # 计算每日因子值的上下限(3个标准差)
    def winsorize_by_date(df):
        mean = df['ts_argmax'].mean()
        std = df['ts_argmax'].std()
        df['ts_argmax'] = df['ts_argmax'].clip(lower=mean-3*std, upper=mean+3*std)
        return df
    
    # 按日期分组进行极端值处理
    factor_df = factor_df.groupby('date').apply(winsorize_by_date).reset_index(drop=True)
    
    # 步骤4: 横截面标准化 (rank - 0.5)
    def normalize_by_date(df):
        """使用排序归一化到-0.5到0.5之间，然后扩展到-1到1"""
        rank_pct = df['ts_argmax'].rank(pct=True)
        # 先归一化到[-0.5, 0.5]
        df['alpha1_factor'] = rank_pct - 0.5
        # 再扩展到[-1, 1]
        df['alpha1_factor'] = 2 * df['alpha1_factor']
        return df
    
    print("排序归一化因子值到-1到1之间...")
    # 按日期分组进行排序归一化
    factor_df = factor_df.groupby('date').apply(normalize_by_date).reset_index(drop=True)
    
    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha1_factor': 'factor'
    })
    
    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]
    
    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha1_stddev{stddev_window}d_argmax{argmax_window}d_rebalance{rebalance_period}d_{today}.csv")
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
    # 默认使用20天标准差窗口，5天极值窗口，3天调仓
    create_alpha1_factor(stddev_window=30, argmax_window=10, rebalance_period=10) 