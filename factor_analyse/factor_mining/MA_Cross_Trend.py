# factor_analyse/factor_mining/MA_Cross_Trend.py
"""
均线多头排列强度因子 (MA Cross Trend)
factor = MA_fast / MA_slow - 1

无未来函数检查：
- rolling(window).mean() 均为向历史方向的滑动窗口，无 center=True
- shift 均为正数（历史数据），无负位移
"""
import pandas as pd
import numpy as np

from util_factor import (
    load_kline_df,
    build_available_tokens_by_date_from_kline,
    filter_group_by_availability,
    group_apply_with_progress,
    winsorize_by_date,
    rank_to_unit_by_date,
    save_factor_df,
    future_return,
    print_availability_sample,
    print_factor_summary,
    print_latest_date_inference,
)


def create_ma_cross_trend(fast=20, slow=60, rebalance_period=5, top_n=50):
    """
    均线多头排列强度因子 —— 无未来函数版本
    factor_raw = MA_fast / MA_slow - 1
      > 0: 快线在慢线上方（多头排列），值越大趋势越强
      < 0: 空头排列，值越小下跌趋势越强

    经典趋势跟踪信号：20MA/60MA 是加密量化中最常用的趋势判断工具
    """
    print(f"开始构建均线多头排列因子 (MA Cross Trend) —— 无未来函数版本")
    print(f"参数: fast={fast}, slow={slow}, rebalance_period={rebalance_period}, top_n={top_n}")

    df = load_kline_df()

    print("🔍 基于 K 线数据构建每日前50排名...")
    available_tokens_by_date = build_available_tokens_by_date_from_kline(
        kline_df=df, top_n=top_n, ranking_method="quote_volume",
    )
    print_availability_sample(available_tokens_by_date, n=3)

    ever_top50 = set()
    for symbols in available_tokens_by_date.values():
        ever_top50.update(symbols)
    df = df[df["symbol"].isin(ever_top50)].copy()

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < slow + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # 纯历史滑动窗口，无未来函数
        ma_fast = gp["close"].rolling(fast).mean()
        ma_slow = gp["close"].rolling(slow).mean()

        gp["mac_raw"] = ma_fast / ma_slow - 1

        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)
        return gp[["date", "symbol", "mac_raw", "future_ret"]].dropna(subset=["mac_raw"])

    print("计算因子...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    factor_df = winsorize_by_date(factor_df, col="mac_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="mac_raw", out_col="factor")
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[
        ["date", "instrument", "factor", "future_ret"]
    ]

    out_path = save_factor_df(
        factor_df,
        file_prefix=f"ma_cross_trend_{fast}_{slow}d_rebalance{rebalance_period}d_",
    )
    print_factor_summary(factor_df, out_path)
    print_latest_date_inference(factor_df, factor_df["date"].max(), groups=5)
    return factor_df


if __name__ == "__main__":
    create_ma_cross_trend(fast=20, slow=60, rebalance_period=5, top_n=50)
