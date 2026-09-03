import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha31_factor(
    rebalance_period=7,
    delta_long_window=10,     # delta(close, 10)
    decay_window=10,          # decay_linear(..., 10)
    delta_short_window=3,     # delta(close, 3)
    adv_window=20,            # adv20
    corr_window=12            # correlation(adv20, low, 12)
):
    """
    创建 Alpha 31 因子（长期趋势 + 短期动量 + 流动性/支撑 的多维复合因子）

    Alpha 31 =
      rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10))))
      + rank((-1 * delta(close, 3)))
      + sign(scale(correlation(adv20, low, 12)))

    说明：
    - 先在时间序列上构造信号，再在横截面上做 rank/scale，最后合成并标准化输出。
    """
    print("开始构建 Alpha 31 因子 ...")
    print(f"参数: rebalance_period={rebalance_period}, "
          f"delta_long_window={delta_long_window}, decay_window={decay_window}, "
          f"delta_short_window={delta_short_window}, adv_window={adv_window}, corr_window={corr_window}")

    # 目录与数据
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))  # 回到Crypto目录
    data_dir = os.path.join(base_dir, "data", "kline_data")

    kline_files = [f for f in os.listdir(data_dir) if f.startswith("binance_daily_klines_")]
    if not kline_files:
        raise FileNotFoundError("未找到K线数据文件")
    latest_file = sorted(kline_files)[-1]
    data_path = os.path.join(data_dir, latest_file)
    print(f"读取数据文件: {data_path}")

    # 读取
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # ============ 时间序列基础量 ============
    def ts_features(g):
        g = g.copy()
        # delta(close, 10) 与 delta(close, 3)
        g['delta_long'] = g['close'].diff(delta_long_window)
        g['delta_short'] = g['close'].diff(delta_short_window)

        # adv20
        g['adv20'] = g['volume'].rolling(window=adv_window, min_periods=adv_window).mean()

        # correlation(adv20, low, corr_window)
        g['corr_adv_low'] = g['adv20'].rolling(window=corr_window, min_periods=corr_window).corr(g['low'])

        # future return
        g['future_ret'] = np.log(g['close'].shift(-rebalance_period) / g['close'])
        return g

    df = df.groupby('symbol', group_keys=False).apply(ts_features)

    # ============ 横截面 rank（用于 decay_linear 的输入） ============
    # -1 * rank(rank(delta(close, 10)))
    df['cs_rank1_delta_long'] = df.groupby('date')['delta_long'].rank(pct=True)
    df['cs_rank2_delta_long'] = df.groupby('date')['cs_rank1_delta_long'].rank(pct=True)
    df['neg_rank2_delta_long'] = -1.0 * df['cs_rank2_delta_long']

    # ============ 时间序列 decay_linear（对上面的横截面序列做 TS 衰减） ============
    weights = np.arange(1, decay_window + 1, dtype=float)
    w_sum = weights.sum()

    def decay_linear_series(s):
        return s.rolling(window=decay_window, min_periods=decay_window).apply(
            lambda x: np.dot(x, weights) / w_sum, raw=True
        )

    df['decay_linear_val'] = df.groupby('symbol', group_keys=False)['neg_rank2_delta_long'].apply(decay_linear_series)

    # ============ 横截面多层 rank ============
    # rank(rank(rank(decay_linear(...))))
    df['cs_rank_decay_1'] = df.groupby('date')['decay_linear_val'].rank(pct=True)
    df['cs_rank_decay_2'] = df.groupby('date')['cs_rank_decay_1'].rank(pct=True)
    df['cs_rank_decay_3'] = df.groupby('date')['cs_rank_decay_2'].rank(pct=True)

    # rank((-1 * delta(close, 3)))
    df['neg_delta_short'] = -1.0 * df['delta_short']
    df['cs_rank_neg_delta_short'] = df.groupby('date')['neg_delta_short'].rank(pct=True)

    # sign(scale(correlation(adv20, low, 12))) -> 先对 corr 做横截面 z-score
    def cs_zscore(x):
        m = x.mean()
        s = x.std()
        if s == 0 or np.isnan(s):
            return (x - m)  # 避免除0，退化为中心化
        return (x - m) / s

    df['corr_scaled'] = df.groupby('date')['corr_adv_low'].transform(cs_zscore)
    df['corr_sign'] = np.sign(df['corr_scaled'].fillna(0.0))

    # 合成原始因子
    df['alpha31_raw'] = df['cs_rank_decay_3'] + df['cs_rank_neg_delta_short'] + df['corr_sign']

    # 仅保留所需列并去除NaN
    df_keep = df[['date', 'symbol', 'alpha31_raw', 'future_ret']].dropna()

    if df_keep.empty:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    # ============ 极值处理（按日） ============
    def winsorize_by_date(g):
        g = g.copy()
        mu = g['alpha31_raw'].mean()
        sd = g['alpha31_raw'].std()
        g['alpha31_raw'] = g['alpha31_raw'].clip(lower=mu - 3*sd, upper=mu + 3*sd)
        return g

    df_keep = df_keep.groupby('date', group_keys=False).apply(winsorize_by_date)

    # ============ 横截面排序归一化到 [-1, 1] ============
    def normalize_by_date(g):
        g = g.copy()
        r = g['alpha31_raw'].rank(pct=True)
        g['alpha31_factor'] = 2.0 * (r - 0.5)
        return g

    df_keep = df_keep.groupby('date', group_keys=False).apply(normalize_by_date)

    # 输出格式
    factor_df = df_keep.rename(columns={'symbol': 'instrument', 'alpha31_factor': 'factor'})[
        ['date', 'instrument', 'factor', 'future_ret']
    ]

    # 保存
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    out_path = os.path.join(output_dir, f"alpha31_trend_momentum_liquidity_rebalance{rebalance_period}d_{today}.csv")
    factor_df.to_csv(out_path, index=False)

    print(f"✅ 因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("示例预览：")
    print(factor_df.head())

    return factor_df


if __name__ == "__main__":
    # 可按需要调整窗口
    create_alpha31_factor(rebalance_period=10, delta_long_window=20, decay_window=20, delta_short_window=5, adv_window=20, corr_window=18)
