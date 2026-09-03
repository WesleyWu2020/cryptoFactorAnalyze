# factor_analyse/Alpha101/Alpha34_Factor.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

def create_alpha34_factor(std_short=2, std_long=5, delta_lag=1, rebalance_period=3, availability_lookback_days=90):
    """
    Alpha 34
    rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))

    核心思路:
    - 波动率比较: 短期(2) / 中期(5)波动率比值越小越好 → 反向排名(1 - rank)
    - 价格变化: 近1期价格变动(delta)越小越好 → 反向排名(1 - rank)
    - 综合: 两项相加后再做一次横截面rank
    - 平均持有期 2-5 天
    """
    print("开始构建 Alpha 34 因子 (7x24 加密优化)...")
    print(f"参数: std_short={std_short}, std_long={std_long}, delta_lag={delta_lag}, rebalance_period={rebalance_period}")

    # 路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))  # 回到Crypto目录

    # 历史市值排名，避免未来函数
    historical_csv_path = os.path.join(base_dir, "data", "cryptocompare_top50_marketcap_historical.csv")
    if not os.path.exists(historical_csv_path):
        raise FileNotFoundError(f"未找到历史市值排名文件: {historical_csv_path}")

    print("加载历史市值排名数据...")
    historical_df = pd.read_csv(historical_csv_path)
    historical_df['Date'] = pd.to_datetime(historical_df['Date'], format='mixed', errors='coerce')
    historical_df = historical_df.sort_values('Date').reset_index(drop=True)

    def get_available_tokens_by_date(hdf, lookback_period=availability_lookback_days):
        available_tokens_by_date = {}
        for date in hdf['Date'].unique():
            lookback_date = date - timedelta(days=lookback_period)
            available_tokens = hdf[hdf['Date'] <= lookback_date]['Trading_Pair'].unique().tolist()
            if available_tokens:
                available_tokens_by_date[date] = available_tokens
        return available_tokens_by_date

    available_tokens_by_date = get_available_tokens_by_date(historical_df, lookback_period=availability_lookback_days)
    print(f"已创建 {len(available_tokens_by_date)} 个日期的token可用性映射")

    # 数据文件
    data_dir = os.path.join(base_dir, "data", "kline_data")
    kline_files = [f for f in os.listdir(data_dir) if f.startswith("binance_daily_klines_")]
    if not kline_files:
        raise FileNotFoundError("未找到K线数据文件")
    latest_file = sorted(kline_files)[-1]
    data_path = os.path.join(data_dir, latest_file)
    print(f"读取数据文件: {data_path}")

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    # 组件计算（逐symbol，避免未来函数并做可用性过滤）
    def compute_components(group, symbol):
        min_len = max(std_long + 2, delta_lag + 2) + rebalance_period + 3
        if len(group) < min_len:
            return pd.DataFrame()

        # 可用性过滤
        valid_rows = []
        for idx, row in group.iterrows():
            current_date = row['date'].date()
            available_date = None
            for date_key in available_tokens_by_date.keys():
                if date_key.date() <= current_date:
                    available_date = date_key
                else:
                    break
            if available_date and symbol in available_tokens_by_date[available_date]:
                valid_rows.append(idx)
        if not valid_rows:
            return pd.DataFrame()

        group = group.loc[valid_rows].copy()

        # 日收益率
        group['returns'] = group['close'].pct_change()

        # 波动率: stddev(returns, N)
        group['std_s'] = group['returns'].rolling(window=std_short, min_periods=std_short).std()
        group['std_l'] = group['returns'].rolling(window=std_long, min_periods=std_long).std()

        # 比值: std_s / std_l
        group['vol_ratio'] = group['std_s'] / group['std_l']
        group['vol_ratio'] = group['vol_ratio'].replace([np.inf, -np.inf], np.nan)

        # 价格变化: delta(close, lag)
        group['delta_close'] = group['close'] - group['close'].shift(delta_lag)

        # 未来收益（评估用）
        group['future_ret'] = np.log(group['close'].shift(-rebalance_period) / group['close'])

        return group[['date', 'symbol', 'vol_ratio', 'delta_close', 'future_ret']].dropna()

    print("计算 Alpha 34 因子基础数据（避免未来函数）...")
    result_dfs = []
    for symbol, g in tqdm(df.groupby('symbol')):
        r = compute_components(g, symbol)
        if not r.empty:
            result_dfs.append(r)

    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 横截面排名与组合
    def combine_by_date(df_date):
        df_date = df_date.copy()
        # rank(vol_ratio) → 倾向小 → 1 - rank
        r_vol = df_date['vol_ratio'].rank(pct=True)
        inv_vol = 1.0 - r_vol

        # rank(delta_close) → 倾向小 → 1 - rank
        r_delta = df_date['delta_close'].rank(pct=True)
        inv_delta = 1.0 - r_delta

        # 综合后再次rank
        sum_inv = inv_vol + inv_delta
        df_date['alpha34_raw'] = sum_inv.rank(pct=True)
        return df_date

    factor_df = factor_df.groupby('date', group_keys=False).apply(combine_by_date)
    factor_df = factor_df.dropna(subset=['alpha34_raw'])

    # 截尾与归一化
    def winsorize_by_date(df_date):
        df_date = df_date.copy()
        m = df_date['alpha34_raw'].mean()
        sd = df_date['alpha34_raw'].std()
        df_date['alpha34_raw'] = df_date['alpha34_raw'].clip(lower=m - 3 * sd, upper=m + 3 * sd)
        return df_date

    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)

    def normalize_by_date(df_date):
        df_date = df_date.copy()
        rank_pct = df_date['alpha34_raw'].rank(pct=True)
        df_date['alpha34_factor'] = 2 * (rank_pct - 0.5)  # [-1, 1]
        return df_date

    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument', 'alpha34_factor': 'factor'})
    factor_df = factor_df[['date', 'instrument', 'factor', 'future_ret']]

    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(
        output_dir,
        f"alpha34_volratio{std_short}_{std_long}_delta{delta_lag}_rebalance{rebalance_period}d_{today}.csv"
    )
    factor_df.to_csv(output_path, index=False)

    print(f"✅ 因子数据已保存至: {output_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n因子统计信息:")
    print(factor_df['factor'].describe())
    print("\n数据预览:")
    print(factor_df.head())
    return factor_df

if __name__ == "__main__":
    # 默认偏短持有期设定
    create_alpha34_factor(std_short=5, std_long=10, delta_lag=5, rebalance_period=5)
