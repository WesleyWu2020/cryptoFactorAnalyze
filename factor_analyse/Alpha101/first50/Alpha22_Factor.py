import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha22_factor(rebalance_period=7):
    """
    创建 Alpha 22 因子 (基于价量相关性变化和波动性的短期反转因子)

    参数:
    rebalance_period: 调仓周期，默认为7天

    原理:
    Alpha 22 = -1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))

    1. delta(correlation(high, volume, 5), 5)：监控最高价与交易量相关性的5日变化
    2. rank(stddev(close, 20))：20日收盘价波动性的横截面排名
    3. 反向操作：价量关系正向变化且波动性高时，给出看跌信号
    """
    print(f"开始构建 Alpha 22 因子 ...")
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
    def calculate_alpha22_factor(group):
        if len(group) < 40:
            return pd.DataFrame()

        # 1. 计算5日相关性
        group['corr_high_vol'] = group['high'].rolling(window=10, min_periods=10).corr(group['volume'])
        # 2. 计算5日相关性的5日变化
        group['delta_corr'] = group['corr_high_vol'].diff(10)
        # 3. 计算20日收盘价波动性
        group['std20'] = group['close'].rolling(window=10, min_periods=10).std()
        # 4. 计算future return
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])

        # 只保留需要的列
        result = group[['date', 'symbol', 'delta_corr', 'std20', 'future_ret']].copy()
        result = result.dropna()
        return result

    # 按交易对分组计算
    print("计算 Alpha 22 因子基础数据...")
    result_dfs = []
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha22_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)

    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 步骤2：横截面排名和复合因子
    def calculate_final_factor(df):
        df = df.copy()
        # 1. 波动性横截面排名
        df['std20_rank'] = df['std20'].rank(pct=True)
        # 2. 复合因子
        df['alpha22_raw'] = -1 * (df['delta_corr'] * df['std20_rank'])
        return df

    # 按日期分组计算最终因子
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_final_factor)

    # 处理极端值
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha22_raw'].mean()
        std = df['alpha22_raw'].std()
        df['alpha22_raw'] = df['alpha22_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df

    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)

    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        df = df.copy()
        rank_pct = df['alpha22_raw'].rank(pct=True)
        df['alpha22_factor'] = 2 * (rank_pct - 0.5)
        return df

    print("排序归一化因子值到-1到1之间...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)

    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha22_factor': 'factor'
    })

    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]

    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)

    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha22_corr_volatility_rebalance{rebalance_period}d_{today}.csv")
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
    # === Alpha 22 因子参数优化测试 ===
    create_alpha22_factor(rebalance_period=7)
