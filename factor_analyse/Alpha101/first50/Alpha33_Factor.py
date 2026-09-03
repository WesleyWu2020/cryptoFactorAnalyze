import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def create_alpha33_factor(rebalance_period=3):
    """
    Alpha 33: rank((-1 * ((1 - (open / close))^1))) == rank((open - close) / close)
    - 日内反转：日内上涨越强，排名越低；预期短期回调
    - rebalance_period: 调仓周期（默认3天）
    """
    print("开始构建 Alpha 33 因子 ...")
    print(f"参数: rebalance_period={rebalance_period}")

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

    # 分组计算
    def calc_group(g):
        if len(g) < rebalance_period + 5:
            return pd.DataFrame()
        # (open - close) / close 等价于 - (1 - open/close)
        g['alpha33_raw'] = (g['open'] - g['close']) / g['close']
        g['alpha33_raw'].replace([np.inf, -np.inf], np.nan, inplace=True)
        # future return
        g['future_ret'] = np.log(g['close'].shift(-rebalance_period) / g['close'])
        return g[['date', 'symbol', 'alpha33_raw', 'future_ret']].dropna()

    print("计算 Alpha 33 因子基础数据...")
    parts = []
    for sym, grp in tqdm(df.groupby('symbol')):
        out = calc_group(grp.copy())
        if not out.empty:
            parts.append(out)
    if not parts:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(parts, ignore_index=True)

    # 按日winsorize极值
    def winsorize_by_date(x):
        m, s = x['alpha33_raw'].mean(), x['alpha33_raw'].std()
        x['alpha33_raw'] = x['alpha33_raw'].clip(lower=m-3*s, upper=m+3*s)
        return x
    factor_df = factor_df.groupby('date', group_keys=False).apply(winsorize_by_date)

    # 横截面排名归一化到[-1,1]
    def normalize_by_date(x):
        r = x['alpha33_raw'].rank(pct=True)
        x['alpha33_factor'] = 2*(r - 0.5)
        return x
    print("排序归一化因子值到-1到1之间...")
    factor_df = factor_df.groupby('date', group_keys=False).apply(normalize_by_date)

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument', 'alpha33_factor': 'factor'})[
        ['date', 'instrument', 'factor', 'future_ret']
    ]
    output_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')
    out_path = os.path.join(output_dir, f"alpha33_intraday_reversion_rebalance{rebalance_period}d_{today}.csv")
    factor_df.to_csv(out_path, index=False)

    print(f"✅ 因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("数据预览：")
    print(factor_df.head())
    return factor_df

if __name__ == "__main__":
    create_alpha33_factor(rebalance_period=1)
