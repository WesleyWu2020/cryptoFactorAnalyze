import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha24_factor(rebalance_period=7):
    """
    创建 Alpha 24 因子 (自适应长短期结合因子)

    参数:
    rebalance_period: 调仓周期，默认为7天

    原理:
    Alpha 24 = 
        if (长期价格变化率 <= 5%):
            -1 * (close - 100日最低价)
        else:
            -1 * delta(close, 3)

    1. delta((sum(close, 100) / 100), 100) / delay(close, 100)：长期价格变化率
    2. ts_min(close, 100)：100日最低价
    3. delta(close, 3)：3日价格变化
    4. 根据长期趋势强度切换策略
    """
    print(f"开始构建 Alpha 24 因子 ...")
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
    def calculate_alpha24_factor(group):
        if len(group) < 200:
            return pd.DataFrame()

        # 1. 100日均价
        group['ma100'] = group['close'].rolling(window=30, min_periods=30).mean()
        # 2. 100日均价的100日变化
        group['ma100_delta100'] = group['ma100'].diff(30)
        # 3. 100日前的close
        group['close_delay100'] = group['close'].shift(30)
        # 4. 长期价格变化率
        group['long_trend_rate'] = group['ma100_delta100'] / group['close_delay100']
        # 5. 100日最低价
        group['min100'] = group['close'].rolling(window=30, min_periods=30).min()
        # 6. 3日价格变化
        group['delta_close_3'] = group['close'].diff(3)
        # 7. 因子逻辑
        def alpha24_row(row):
            if pd.isna(row['long_trend_rate']) or pd.isna(row['min100']) or pd.isna(row['delta_close_3']):
                return np.nan
            if (row['long_trend_rate'] < 0.05) or (row['long_trend_rate'] == 0.05):
                return -1 * (row['close'] - row['min100'])
            else:
                return -1 * row['delta_close_3']
        group['alpha24_raw'] = group.apply(alpha24_row, axis=1)
        # 8. future return
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])

        # 只保留需要的列
        result = group[['date', 'symbol', 'alpha24_raw', 'future_ret']].copy()
        result = result.dropna()
        return result

    # 按交易对分组计算
    print("计算 Alpha 24 因子基础数据...")
    result_dfs = []
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha24_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)

    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 处理极端值
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha24_raw'].mean()
        std = df['alpha24_raw'].std()
        df['alpha24_raw'] = df['alpha24_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df

    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)

    # 排序归一化到-1到1之间
    def normalize_by_date(df):
        df = df.copy()
        rank_pct = df['alpha24_raw'].rank(pct=True)
        df['alpha24_factor'] = 2 * (rank_pct - 0.5)
        return df

    print("排序归一化因子值到-1到1之间...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)

    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha24_factor': 'factor'
    })

    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]

    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)

    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha24_adaptive_trend_rebalance{rebalance_period}d_{today}.csv")
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
    # === Alpha 24 因子参数优化测试 ===
    create_alpha24_factor(rebalance_period=7)
