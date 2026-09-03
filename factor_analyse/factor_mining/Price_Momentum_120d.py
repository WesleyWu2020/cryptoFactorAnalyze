# factor_analyse/factor_mining/Price_Momentum_120d.py
"""
价格动量因子 120日 (Price Momentum 120D)
factor = log(close_t / close_{t-120})

无未来函数检查：
- shift(120) 为向后平移120天（历史价格），无前视偏差
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


def create_price_momentum_120d(window=120, rebalance_period=20, top_n=50):
    """
    120日价格动量因子 —— 无未来函数版本
    factor_raw = log(close_t / close_{t-window})

    捕捉长期趋势（加密市场 4个月动量：牛熊周期中最有效的择时信号之一）
    """
    print(f"开始构建价格动量因子 (Price Momentum {window}D) —— 无未来函数版本")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, top_n={top_n}")

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
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        gp["mom_raw"] = np.log(gp["close"] / gp["close"].shift(window))
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)
        return gp[["date", "symbol", "mom_raw", "future_ret"]].dropna(subset=["mom_raw"])

    print("计算因子...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    factor_df = winsorize_by_date(factor_df, col="mom_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="mom_raw", out_col="factor")
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[
        ["date", "instrument", "factor", "future_ret"]
    ]

    out_path = save_factor_df(
        factor_df,
        file_prefix=f"price_momentum_{window}d_rebalance{rebalance_period}d_",
    )
    print_factor_summary(factor_df, out_path)
    print_latest_date_inference(factor_df, factor_df["date"].max(), groups=5)
    return factor_df


if __name__ == "__main__":
    create_price_momentum_120d(window=120, rebalance_period=20, top_n=50)
