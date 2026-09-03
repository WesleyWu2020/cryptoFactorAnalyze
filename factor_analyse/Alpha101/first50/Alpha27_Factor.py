import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha27_factor(rebalance_period=5, corr_window=6, mean_window=2):
    """
    创建 Alpha 27 因子 (基于市场微观结构的二元因子)

    参数:
    rebalance_period: 调仓周期，默认为5天
    corr_window: 相关性窗口，默认为6
    mean_window: 相关性均值窗口，默认为2

    原理:
    Alpha 27 = 
        if (0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))):
            -1
        else:
            1

    1. correlation(rank(volume), rank(vwap), 6)：分析交易量排名与VWAP排名的关系
    2. sum(..., 2) / 2：2天平均相关性，衡量关系稳定性
    3. rank(...)：横截面排名
    4. 二元信号：买入(1)或卖出(-1)
    """
    print(f"开始构建 Alpha 27 因子 ...")
    print(f"参数: rebalance_period={rebalance_period}, corr_window={corr_window}, mean_window={mean_window}")

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
    def calculate_alpha27_factor(group):
        if len(group) < corr_window + mean_window + 5:
            return pd.DataFrame()

        # 1. VWAP
        group['vwap'] = (group['quote_volume'] / group['volume']).replace([np.inf, -np.inf], np.nan)
        # 2. rank(volume), rank(vwap)
        group['vol_rank'] = group['volume'].rank(method='average')
        group['vwap_rank'] = group['vwap'].rank(method='average')
        # 3. correlation(rank(volume), rank(vwap), corr_window)
        group['corr_vol_vwap'] = group[['vol_rank', 'vwap_rank']].rolling(window=corr_window, min_periods=corr_window).corr().iloc[0::2,-1].reset_index(level=1, drop=True)
        # 4. 2天均值
        group['corr_mean'] = group['corr_vol_vwap'].rolling(window=mean_window, min_periods=mean_window).mean()
        # 5. future return
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])

        # 只保留需要的列
        result = group[['date', 'symbol', 'corr_mean', 'future_ret']].copy()
        result = result.dropna()
        return result

    # 按交易对分组计算
    print("计算 Alpha 27 因子基础数据...")
    result_dfs = []
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha27_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)

    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 横截面排名并生成二元信号
    def cross_section_rank_signal(df):
        df = df.copy()
        # 横截面排名
        rank_pct = df['corr_mean'].rank(pct=True)
        # 0.5为阈值，>0.5为-1，<=0.5为1
        df['factor'] = np.where(rank_pct > 0.5, -1, 1)
        return df

    print("横截面排名并生成二元信号...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(cross_section_rank_signal)

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
    output_path = os.path.join(output_dir, f"alpha27_microstructure_binary_rebalance{rebalance_period}d_{today}.csv")
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
    # === Alpha 27 因子参数优化测试 ===
    create_alpha27_factor(rebalance_period=10, corr_window=20, mean_window=10)
