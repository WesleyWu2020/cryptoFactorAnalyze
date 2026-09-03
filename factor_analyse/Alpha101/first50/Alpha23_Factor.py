import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha23_factor(rebalance_period=3):
    """
    创建 Alpha 23 因子 (基于价格突破的短期反转因子)

    参数:
    rebalance_period: 调仓周期，默认为3天

    原理:
    Alpha 23 = 
        if (high > 20日最高价均值): -1 * delta(high, 2)
        else: 0

    1. (sum(high, 20) / 20)：20日最高价均值
    2. delta(high, 2)：2日最高价变化
    3. 只在突破时给出信号，其他时候为0
    """
    print(f"开始构建 Alpha 23 因子 ...")
    print(f"参数: rebalance_period={rebalance_period}")

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
    def calculate_alpha23_factor(group):
        if len(group) < 25:
            return pd.DataFrame()

        # 1. 20日最高价均值
        group['high_ma20'] = group['high'].rolling(window=20, min_periods=20).mean()
        # 2. 2日最高价变化
        group['delta_high_2'] = group['high'].diff(2)
        # 3. 因子逻辑
        def alpha23_row(row):
            if pd.isna(row['high_ma20']) or pd.isna(row['delta_high_2']):
                return np.nan
            if row['high'] > row['high_ma20']:
                return -1 * row['delta_high_2']
            else:
                return 0
        group['alpha23_raw'] = group.apply(alpha23_row, axis=1)
        # 4. future return
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])

        # 只保留需要的列
        result = group[['date', 'symbol', 'alpha23_raw', 'future_ret']].copy()
        result = result.dropna()
        return result

    # 按交易对分组计算
    print("计算 Alpha 23 因子基础数据...")
    result_dfs = []
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha23_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)

    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 处理极端值
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha23_raw'].mean()
        std = df['alpha23_raw'].std()
        df['alpha23_raw'] = df['alpha23_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df

    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)

    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        df = df.copy()
        rank_pct = df['alpha23_raw'].rank(pct=True)
        df['alpha23_factor'] = 2 * (rank_pct - 0.5)
        return df

    print("排序归一化因子值到-1到1之间...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)

    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha23_factor': 'factor'
    })

    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]

    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)

    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha23_breakout_reversal_rebalance{rebalance_period}d_{today}.csv")
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
    # === Alpha 23 因子参数优化测试 ===
    create_alpha23_factor(rebalance_period=3)
