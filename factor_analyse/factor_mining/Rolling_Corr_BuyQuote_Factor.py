import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

def create_rolling_corr_buyquote_factor(window=20, rebalance_period=3):
    """
    因子名称: rolling_corr_buyquote（无未来函数版本）

    思路 —— 过去 window 日内:
      ① 计算收益率 ret = log(close_t / close_{t-1})
      ② 计算 ret 与 taker_buy_quote 的滚动相关系数 corr_t
      ③ 计算同窗口内成交量的平均值 vol̄_t
      ④ 因子值 = corr_t × log(1 + vol̄_t)

    经济含义:
      若主动买单成交额与价格同步上升且在高流动性阶段出现，则因子为正，代表买方力量强。
    """
    print(f"开始构建滚动相关系数因子 (Rolling_Corr_BuyQuote) —— 无未来函数版本")
    print(f"参数: window={window}, rebalance_period={rebalance_period}")

    # ---------- 路径 ----------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(base_dir, "data", "kline_data")
    hist_path = os.path.join(base_dir, "data", "cryptocompare_top50_marketcap_historical.csv")

    # 历史市值Top出现池（避免未来函数）
    print(f"读取历史市值排名数据: {hist_path}")
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"未找到历史市值排名文件: {hist_path}")
    historical_df = pd.read_csv(hist_path)
    historical_df["Date"] = pd.to_datetime(historical_df["Date"], format='mixed', errors='coerce')
    historical_df = historical_df.sort_values("Date").reset_index(drop=True)
    print(f"✅ 历史市值排名数据: {len(historical_df)} 条记录")
    print(f"📅 时间范围: {historical_df['Date'].min():%Y-%m-%d} 到 {historical_df['Date'].max():%Y-%m-%d}")

    def get_available_tokens_by_date(hdf: pd.DataFrame, lookback_period=90):
        """
        对每个日期，给出此前 lookback_period 天内出现过的 token 列表（含当日）。
        """
        available_tokens = {}
        unique_dates = sorted(hdf["Date"].unique())
        for current_date in unique_dates:
            start_date = current_date - timedelta(days=lookback_period)
            tokens = (
                hdf[(hdf["Date"] >= start_date) & (hdf["Date"] <= current_date)]["Trading_Pair"]
                .unique()
                .tolist()
            )
            available_tokens[current_date] = tokens
        return available_tokens

    print("🔍 生成各日期可用token列表（避免未来函数）...")
    available_tokens_by_date = get_available_tokens_by_date(historical_df, lookback_period=90)
    sample_dates = list(available_tokens_by_date.keys())[:3]
    for d in sample_dates:
        print(f"   {pd.to_datetime(d):%Y-%m-%d}: {len(available_tokens_by_date[d])} 个可用token")

    # ---------- K线数据 ----------
    files = sorted([f for f in os.listdir(data_dir) if f.startswith("binance_daily_klines_")])
    if not files:
        raise FileNotFoundError("未找到 K 线数据文件")
    kline_path = os.path.join(data_dir, files[-1])
    print(f"读取数据文件: {kline_path}")

    df = pd.read_csv(kline_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])

    # ---------- 单品种计算 ----------
    def calc_factor(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) < window + rebalance_period:
            return pd.DataFrame()

        gp = group.copy()
        gp["ret"] = np.log(gp["close"]).diff()
        gp["vol_mean"] = gp["volume"].rolling(window).mean()

        # taker_buy_quote 与 ret 的滚动相关
        gp["tbq_mean"] = gp["taker_buy_quote"].rolling(window).mean()
        gp["corr_rbq"] = gp["ret"].rolling(window).corr(gp["tbq_mean"])

        gp["rolling_corr_buyquote"] = gp["corr_rbq"] * np.log1p(gp["vol_mean"])
        # 未来收益（与模板保持一致，可换为 pct_change 版本）
        gp["future_ret"] = gp["close"].pct_change(rebalance_period).shift(-rebalance_period)

        return gp[["date", "symbol", "rolling_corr_buyquote", "future_ret"]]

    # ---------- 计算并按可用token过滤 ----------
    print("计算因子并进行可用性过滤...")
    result_dfs = []
    date_keys_sorted = sorted(available_tokens_by_date.keys())

    for symbol, group in tqdm(df.groupby("symbol")):
        tmp = calc_factor(group)
        if tmp.empty:
            continue

        valid_idx = []
        for idx, row in tmp.iterrows():
            current_date = row["date"].date()
            # 找到最近的可用日期键
            available_date = None
            for dk in date_keys_sorted:
                dk_date = pd.to_datetime(dk).date()
                if dk_date <= current_date:
                    available_date = dk
                else:
                    break
            if available_date is not None and symbol in available_tokens_by_date[available_date]:
                valid_idx.append(idx)

        if not valid_idx:
            continue
        result_dfs.append(tmp.loc[valid_idx])

    if not result_dfs:
        print("⚠️ 没有足够的数据计算因子")
        return pd.DataFrame()

    result = pd.concat(result_dfs, ignore_index=True).dropna(subset=["rolling_corr_buyquote", "future_ret"])

    # ---------- 归一化（按日排位到 [-1,1]） ----------
    result = result.rename(columns={"symbol": "instrument"})
    def normalize_by_date(df_d):
        if len(df_d) == 0 or df_d["rolling_corr_buyquote"].isna().all():
            return df_d
        rank_pct = df_d["rolling_corr_buyquote"].rank(pct=True)
        df_d["factor"] = 2 * (rank_pct - 0.5)
        return df_d
    factor_df = (
        result.groupby("date")
        .apply(normalize_by_date)
        .reset_index(drop=True)[["date", "instrument", "factor", "future_ret"]]
        .dropna()
    )

    # ---------- 诊断 ----------
    print(f"\n🔍 未来函数验证（前5天样本）:")
    for d in sorted(factor_df["date"].unique())[:5]:
        cnt = factor_df.loc[factor_df["date"] == d, "instrument"].nunique()
        print(f"   {pd.to_datetime(d):%Y-%m-%d}: {cnt} 个token")

    # ---------- 保存 ----------
    out_dir = os.path.join(base_dir, "data", "factor_data")
    os.makedirs(out_dir, exist_ok=True)
    fn = f"rolling_corr_buyquote_{window}d_rebalance{rebalance_period}d_{datetime.now():%Y%m%d}.csv"
    out_path = os.path.join(out_dir, fn)
    factor_df.to_csv(out_path, index=False)
    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")

    # 简要统计
    print("\n无未来函数因子统计信息:")
    print(factor_df["factor"].describe())
    print(f"\n归一化验证: 范围[{factor_df['factor'].min():.6f}, {factor_df['factor'].max():.6f}], 均值={factor_df['factor'].mean():.6f}")

    return factor_df

if __name__ == "__main__":
    create_rolling_corr_buyquote_factor(window=20, rebalance_period=10) 