import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha26_factor(
    rebalance_period=3,
    ts_rank_window=5,   # ts_rank的窗口
    ts_max_window=3     # ts_max的窗口
):
    """
    创建 Alpha 26 因子 (基于价量关系强度的短期反转因子)

    参数:
    rebalance_period: 调仓周期，默认为3天
    ts_rank_window: ts_rank的窗口，默认为5
    ts_max_window: ts_max的窗口，默认为3

    原理:
    Alpha 26 = -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

    1. ts_rank(volume, 5)：5日内volume的时间序列排名
    2. ts_rank(high, 5)：5日内high的时间序列排名
    3. correlation(ts_rank(volume, 5), ts_rank(high, 5), 5)：5日窗口内两者相关性
    4. ts_max(..., 3)：取3日内相关性的最大值
    5. 取负号，相关性越高，因子越低（看跌）
    """
    print(f"开始构建 Alpha 26 因子 ...")
    print(f"参数: rebalance_period={rebalance_period}, ts_rank_window={ts_rank_window}, ts_max_window={ts_max_window}")

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
    def calculate_alpha26_factor(group):
        min_len = max(ts_rank_window, ts_max_window) * 3
        if len(group) < min_len:
            return pd.DataFrame()

        # 1. ts_rank(volume, ts_rank_window)
        group['vol_rank'] = group['volume'].rolling(window=ts_rank_window, min_periods=ts_rank_window).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / ts_rank_window, raw=False
        )
        # 2. ts_rank(high, ts_rank_window)
        group['high_rank'] = group['high'].rolling(window=ts_rank_window, min_periods=ts_rank_window).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / ts_rank_window, raw=False
        )
        # 3. correlation(ts_rank(volume), ts_rank(high), ts_rank_window)
        group['corr_vh'] = group[['vol_rank', 'high_rank']].rolling(window=ts_rank_window, min_periods=ts_rank_window).corr().iloc[0::2,-1].reset_index(level=1, drop=True)
        # 4. ts_max(..., ts_max_window)
        group['corr_vh_max'] = group['corr_vh'].rolling(window=ts_max_window, min_periods=ts_max_window).max()
        # 5. 取负号
        group['alpha26_raw'] = -1 * group['corr_vh_max']
        # 6. future return
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])

        # 只保留需要的列
        result = group[['date', 'symbol', 'alpha26_raw', 'future_ret']].copy()
        result = result.dropna()
        return result

    # 按交易对分组计算
    print("计算 Alpha 26 因子基础数据...")
    result_dfs = []
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha26_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)

    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 排序归一化到-1到1之间（横截面排名）
    def normalize_by_date(df):
        df = df.copy()
        rank_pct = df['alpha26_raw'].rank(pct=True)
        df['alpha26_factor'] = 2 * (rank_pct - 0.5)
        return df

    print("排序归一化因子值到-1到1之间...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)

    # 处理极端值
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha26_factor'].mean()
        std = df['alpha26_factor'].std()
        df['factor'] = df['alpha26_factor'].clip(lower=mean-3*std, upper=mean+3*std)
        return df

    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)

    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument'
    })

    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]

    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)

    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha26_volprice_corr_reversal_rebalance{rebalance_period}d_{today}.csv")
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
    # === Alpha 26 因子参数优化测试 ===
    create_alpha26_factor(rebalance_period=10, ts_rank_window=10, ts_max_window=5)
