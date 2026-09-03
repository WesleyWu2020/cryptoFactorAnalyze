import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha28_factor(rebalance_period=5, corr_window=5):
    """
    创建 Alpha 28 因子 (基于流动性与价格关系的均值回归因子)

    参数:
    rebalance_period: 调仓周期，默认为5天
    corr_window: 相关性窗口，默认为5

    原理:
    Alpha 28 = scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))

    1. correlation(adv20, low, 5)：过去5天平均交易量与最低价的相关性
    2. (high + low) / 2：最高价与最低价的中点，代表日内价格中心
    3. + ... - close：将相关性与价格中心相加，再减去收盘价
    4. scale(...)：对结果进行缩放标准化
    """
    print(f"开始构建 Alpha 28 因子 ...")
    print(f"参数: rebalance_period={rebalance_period}, corr_window={corr_window}")

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
    def calculate_alpha28_factor(group):
        if len(group) < corr_window + 20:
            return pd.DataFrame()

        # 1. 20日平均成交量
        group['adv20'] = group['volume'].rolling(window=20, min_periods=20).mean()
        # 2. correlation(adv20, low, 5)
        group['corr_adv20_low'] = group[['adv20', 'low']].rolling(window=corr_window, min_periods=corr_window).corr().iloc[0::2,-1].reset_index(level=1, drop=True)
        # 3. (high + low) / 2 归一化
        group['price_center'] = (group['high'] + group['low']) / 2
        group['price_center_norm'] = group['price_center'] / group['close']
        # 4. 复合项（归一化后）
        group['alpha28_raw'] = (group['corr_adv20_low'] + group['price_center_norm']) - 1
        # 5. future return
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])

        # 只保留需要的列
        result = group[['date', 'symbol', 'alpha28_raw', 'future_ret']].copy()
        result = result.dropna()
        return result

    # 按交易对分组计算
    print("计算 Alpha 28 因子基础数据...")
    result_dfs = []
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha28_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)

    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 处理极端值
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha28_raw'].mean()
        std = df['alpha28_raw'].std()
        df['alpha28_raw'] = df['alpha28_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df

    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)

    # 排序归一化到-1到1之间（横截面排名）
    def normalize_by_date(df):
        df = df.copy()
        rank_pct = df['alpha28_raw'].rank(pct=True)
        df['alpha28_factor'] = 2 * (rank_pct - 0.5)
        return df

    print("排序归一化因子值到-1到1之间...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)

    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha28_factor': 'factor'
    })

    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]

    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)

    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha28_liquidity_price_meanreversion_rebalance{rebalance_period}d_{today}.csv")
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
    # === Alpha 28 因子参数优化测试 ===
    create_alpha28_factor(rebalance_period=5, corr_window=10)
