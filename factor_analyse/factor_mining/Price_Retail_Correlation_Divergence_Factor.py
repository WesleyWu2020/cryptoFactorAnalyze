import pandas as pd

from util_factor import (
    load_kline_df,
    build_available_tokens_by_date_from_kline,
    filter_group_by_availability,
    group_apply_with_progress,
    winsorize_by_date,
    rank_to_unit_by_date,
    save_factor_df,
    print_availability_sample,
    print_factor_summary,
    print_latest_date_inference,
    future_return,
)
from operator_utils import PCT_CHANGE, CORRELATION


def create_price_retail_correlation_divergence_factor(
    window=20, rebalance_period=10, top_n=50
):
    """
    Price-Retail Correlation Divergence factor:
    CORRELATION(PCT_CHANGE(CLOSE,1), PCT_CHANGE(TRADES_COUNT,1), window)
    - CORRELATION(PCT_CHANGE(CLOSE,1), PCT_CHANGE(QUOTE_VOLUME,1), window)
    """
    print("开始构建价量与散户相关性背离因子 (Price-Retail Correlation Divergence)")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, top_n={top_n}")

    df = load_kline_df()

    print("🔍 基于Binance K线数据构建每日前50排名...")
    available_tokens_by_date = build_available_tokens_by_date_from_kline(
        kline_df=df,
        top_n=top_n,
        ranking_method="quote_volume",
        rebalance_period=rebalance_period,
        lookback_buffer=20,
        strict_top_n=True,
    )
    print("🔍 生成各日期可用token列表（每天前50，考虑调仓周期）...")
    print_availability_sample(available_tokens_by_date, n=3)

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()

        gp = group.copy()
        gp["close_ret_1d"] = PCT_CHANGE(gp["close"], n=1)
        gp["trades_ret_1d"] = PCT_CHANGE(gp["trades_count"], n=1)
        gp["qv_ret_1d"] = PCT_CHANGE(gp["quote_volume"], n=1)

        gp["corr_close_trades"] = CORRELATION(gp["close_ret_1d"], gp["trades_ret_1d"], n=window)
        gp["corr_close_qv"] = CORRELATION(gp["close_ret_1d"], gp["qv_ret_1d"], n=window)
        gp["prcd_raw"] = gp["corr_close_trades"] - gp["corr_close_qv"]
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        # Live 推理需要保留 future_ret=NaN 的末端日期
        return gp[["date", "symbol", "prcd_raw", "future_ret"]].dropna(subset=["prcd_raw"])

    print("计算 PRCD 因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df_raw = pd.concat(result_dfs, ignore_index=True)
    factor_df_live = winsorize_by_date(factor_df_raw, col="prcd_raw", n_std=3.0)
    factor_df_live = rank_to_unit_by_date(factor_df_live, col="prcd_raw", out_col="factor")

    # 回测/报告使用：仅保留 future_ret 有效样本
    factor_df_backtest = factor_df_live.dropna(subset=["future_ret"]).copy()
    factor_df_backtest = factor_df_backtest.rename(columns={"symbol": "instrument"})[
        ["date", "instrument", "factor", "future_ret"]
    ]

    out_path = save_factor_df(
        factor_df_backtest,
        file_prefix=f"price_retail_corr_divergence_{window}d_rebalance{rebalance_period}d_",
    )
    print_factor_summary(factor_df_backtest, out_path)

    factor_df_live = factor_df_live.rename(columns={"symbol": "instrument"})[
        ["date", "instrument", "factor", "future_ret"]
    ]
    latest_date = factor_df_live["date"].max()
    print_latest_date_inference(factor_df_live, latest_date, groups=5)

    print(
        f"ℹ️ Live推理日期范围: {factor_df_live['date'].min().date()} ~ {factor_df_live['date'].max().date()} | "
        f"回测有效日期范围: {factor_df_backtest['date'].min().date()} ~ {factor_df_backtest['date'].max().date()}"
    )
    return factor_df_backtest


if __name__ == "__main__":
    create_price_retail_correlation_divergence_factor(
        window=20, rebalance_period=10, top_n=50
    )
