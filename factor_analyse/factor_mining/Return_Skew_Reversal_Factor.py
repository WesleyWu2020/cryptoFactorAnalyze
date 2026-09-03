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
from operator_utils import PCT_CHANGE, TS_SKEW


def create_return_skew_reversal_factor(window=20, rebalance_period=10, top_n=50):
    """
    Return Skew Reversal factor (no look-ahead in factor value):
    1) RET = PCT_CHANGE(CLOSE, 1)
    2) raw = -1 * TS_SKEW(RET, window)
    """
    print("Start building Return Skew Reversal factor (no future leak in factor value)")
    print(f"Params: window={window}, rebalance_period={rebalance_period}, top_n={top_n}")

    df = load_kline_df()

    print("Build daily top-N availability from Binance kline...")
    available_tokens_by_date = build_available_tokens_by_date_from_kline(
        kline_df=df,
        top_n=top_n,
        ranking_method="quote_volume",
        rebalance_period=rebalance_period,
        lookback_buffer=20,
        strict_top_n=True,
    )
    print_availability_sample(available_tokens_by_date, n=3)

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()

        gp = group.copy()
        gp["ret"] = PCT_CHANGE(gp["close"], n=1)
        gp["ret_skew"] = TS_SKEW(gp["ret"], n=window)
        gp["ret_skew_reversal_raw"] = -1.0 * gp["ret_skew"]
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        # Keep tail rows with future_ret NaN for live inference.
        return gp[["date", "symbol", "ret_skew_reversal_raw", "future_ret"]].dropna(
            subset=["ret_skew_reversal_raw"]
        )

    print("Compute factor and apply availability filtering...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("Warning: insufficient data for factor computation.")
        return pd.DataFrame()

    factor_df_raw = pd.concat(result_dfs, ignore_index=True)

    factor_df_live = winsorize_by_date(factor_df_raw, col="ret_skew_reversal_raw", n_std=3.0)
    factor_df_live = rank_to_unit_by_date(
        factor_df_live, col="ret_skew_reversal_raw", out_col="factor"
    )

    # Backtest/report uses rows with valid future_ret.
    factor_df_backtest = factor_df_live.dropna(subset=["future_ret"]).copy()
    factor_df_backtest = factor_df_backtest.rename(columns={"symbol": "instrument"})[
        ["date", "instrument", "factor", "future_ret"]
    ]

    out_path = save_factor_df(
        factor_df_backtest,
        file_prefix=f"return_skew_reversal_{window}d_rebalance{rebalance_period}d_",
    )
    print_factor_summary(factor_df_backtest, out_path)

    factor_df_live = factor_df_live.rename(columns={"symbol": "instrument"})[
        ["date", "instrument", "factor", "future_ret"]
    ]
    latest_date = factor_df_live["date"].max()
    print_latest_date_inference(factor_df_live, latest_date, groups=5)

    print(
        f"Live date range: {factor_df_live['date'].min().date()} ~ {factor_df_live['date'].max().date()} | "
        f"Backtest valid range: {factor_df_backtest['date'].min().date()} ~ {factor_df_backtest['date'].max().date()}"
    )
    return factor_df_backtest


if __name__ == "__main__":
    create_return_skew_reversal_factor(window=20, rebalance_period=10, top_n=50)
