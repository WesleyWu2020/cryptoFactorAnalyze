import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha29_factor(rebalance_period=7, ts_rank_window=5, delay_window=6):
    """
    创建 Alpha 29 因子 (高度复杂的非线性因子)

    参数:
    rebalance_period: 调仓周期，默认为7天
    ts_rank_window: 时间序列排名窗口，默认为5
    delay_window: 延迟窗口，默认为6

    原理:
    Alpha 29 = min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5)

    1. delta((close - 1), 5)：价格绝对变动趋势
    2. 多层排名和标准化处理
    3. 极值与非线性变换
    4. 历史收益关联
    """
    print(f"开始构建 Alpha 29 因子 ...")
    print(f"参数: rebalance_period={rebalance_period}, ts_rank_window={ts_rank_window}, delay_window={delay_window}")

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
    def calculate_alpha29_factor(group):
        if len(group) < max(ts_rank_window, delay_window) + 10:
            return pd.DataFrame()

        # 1. delta((close - 1), 5)
        group['close_minus_1'] = group['close'] - 1
        group['delta_close_5'] = group['close_minus_1'].diff(5)
        
        # 2. -1 * rank(delta((close - 1), 5))
        group['neg_rank_delta'] = -1 * group['delta_close_5'].rank(method='average')
        
        # 3. rank(rank(...))
        group['double_rank'] = group['neg_rank_delta'].rank(method='average').rank(method='average')
        
        # 4. ts_min(rank(rank(...)), 2)
        group['ts_min_double_rank'] = group['double_rank'].rolling(window=2, min_periods=2).min()
        
        # 5. log(sum(ts_min(...), 1))
        group['log_ts_min'] = np.log(group['ts_min_double_rank'].abs() + 1)  # 加1避免log(0)
        
        # 6. scale(log(...))
        group['scaled_log'] = (group['log_ts_min'] - group['log_ts_min'].rolling(window=20, min_periods=20).mean()) / group['log_ts_min'].rolling(window=20, min_periods=20).std()
        
        # 7. rank(rank(scale(...)))
        group['rank_rank_scaled'] = group['scaled_log'].rank(method='average').rank(method='average')
        
        # 8. product(rank(rank(scale(...))), 1)
        group['product_rank'] = group['rank_rank_scaled']
        
        # 9. min(product(...), 1), 5)
        group['min_product'] = group['product_rank'].rolling(window=5, min_periods=5).min()
        
        # 10. delay((-1 * returns), 6)
        group['returns'] = group['close'].pct_change()
        group['neg_returns_delay'] = -1 * group['returns'].shift(delay_window)
        
        # 11. ts_rank(delay(...), 5)
        group['ts_rank_delay'] = group['neg_returns_delay'].rolling(window=ts_rank_window, min_periods=ts_rank_window).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
        )
        
        # 12. 最终因子
        group['alpha29_raw'] = group['min_product'] + group['ts_rank_delay']
        
        # 13. future return
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])

        # 只保留需要的列
        result = group[['date', 'symbol', 'alpha29_raw', 'future_ret']].copy()
        result = result.dropna()
        return result

    # 按交易对分组计算
    print("计算 Alpha 29 因子基础数据...")
    result_dfs = []
    for symbol, group in tqdm(df.groupby('symbol')):
        factor_result = calculate_alpha29_factor(group)
        if not factor_result.empty:
            result_dfs.append(factor_result)

    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 处理极端值
    def winsorize_by_date(df):
        df = df.copy()
        mean = df['alpha29_raw'].mean()
        std = df['alpha29_raw'].std()
        df['alpha29_raw'] = df['alpha29_raw'].clip(lower=mean-3*std, upper=mean+3*std)
        return df

    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)

    # 排序归一化到-1到1之间（横截面排名）
    def normalize_by_date(df):
        df = df.copy()
        rank_pct = df['alpha29_raw'].rank(pct=True)
        df['alpha29_factor'] = 2 * (rank_pct - 0.5)
        return df

    print("排序归一化因子值到-1到1之间...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)

    # 重命名列以符合因子分析框架要求
    factor_df = factor_df.rename(columns={
        'symbol': 'instrument',
        'alpha29_factor': 'factor'
    })

    # 确保列的顺序为 date, instrument, factor, future_ret
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]

    # 创建输出目录
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)

    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"alpha29_complex_nonlinear_rebalance{rebalance_period}d_{today}.csv")
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
    # === Alpha 29 因子参数优化测试 ===
    create_alpha29_factor(rebalance_period=10, ts_rank_window=5, delay_window=6)
