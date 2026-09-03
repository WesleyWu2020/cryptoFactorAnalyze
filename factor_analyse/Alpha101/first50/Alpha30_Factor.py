import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha30_factor(rebalance_period=5, volume_short_window=5, volume_long_window=20):
    """
    创建 Alpha 30 因子 (基于价格连续性和交易量的反转因子)

    参数:
    rebalance_period: 调仓周期，默认为5天
    volume_short_window: 短期交易量窗口，默认为5
    volume_long_window: 长期交易量窗口，默认为20

    原理:
    Alpha 30 = (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))

    1. sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2))) + sign((delay(close, 2) - delay(close, 3)))：连续3天价格变化方向
    2. 1.0 - rank(...)：反向逻辑，价格连续性强时给出较低权重
    3. sum(volume, 5) / sum(volume, 20)：短期与长期交易量比率
    """
    print(f"开始构建 Alpha 30 因子 ...")
    print(f"参数: rebalance_period={rebalance_period}, volume_short_window={volume_short_window}, volume_long_window={volume_long_window}")

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
    def calculate_alpha30_factor(group):
        if len(group) < max(volume_short_window, volume_long_window) + 5:
            return pd.DataFrame()

        # 1. 连续3天价格变化方向
        group['price_change_1'] = np.sign(group['close'] - group['close'].shift(1))
        group['price_change_2'] = np.sign(group['close'].shift(1) - group['close'].shift(2))
        group['price_change_3'] = np.sign(group['close'].shift(2) - group['close'].shift(3))
        
        # 2. 价格连续性总和
        group['price_continuity'] = group['price_change_1'] + group['price_change_2'] + group['price_change_3']
        
        # 3. 短期交易量
        group['volume_short'] = group['volume'].rolling(window=volume_short_window, min_periods=volume_short_window).sum()
        
        # 4. 长期交易量
        group['volume_long'] = group['volume'].rolling(window=volume_long_window, min_periods=volume_long_window).sum()
        
        # 5. 交易量比率
        group['volume_ratio'] = group['volume_short'] / group['volume_long']
        
        # 6. future return
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])

        # 只保留需要的列
        result = group[['date', 'symbol', 'price_continuity', 'volume_ratio', 'future_ret']].copy()
        result = result.dropna()
        return result

    # 按交易对分组计算
    print("计算 Alpha 30 因子基础数据...")
    result_dfs = []
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha30_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)

    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 横截面排名和复合因子计算
    def calculate_final_factor(df):
        df = df.copy()
        # 1. 价格连续性排名
        df['price_continuity_rank'] = df['price_continuity'].rank(pct=True)
        # 2. 反向逻辑
        df['reversal_weight'] = 1.0 - df['price_continuity_rank']
        # 3. 复合因子
        df['alpha30_raw'] = df['reversal_weight'] * df['volume_ratio']
        return df

    # 按日期分组计算最终因子
    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_final_factor)

    # 处理极端值
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha30_raw'].mean()
        std = df['alpha30_raw'].std()
        df['alpha30_raw'] = df['alpha30_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df

    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)

    # 排序归一化到-1到1之间（横截面排名）
    def normalize_by_date(df):
        df = df.copy()
        rank_pct = df['alpha30_raw'].rank(pct=True)
        df['alpha30_factor'] = 2 * (rank_pct - 0.5)
        return df

    print("排序归一化因子值到-1到1之间...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)

    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha30_factor': 'factor'
    })

    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]

    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)

    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha30_price_continuity_volume_rebalance{rebalance_period}d_{today}.csv")
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
    # === Alpha 30 因子参数优化测试 ===
    create_alpha30_factor(rebalance_period=5, volume_short_window=10, volume_long_window=60)
