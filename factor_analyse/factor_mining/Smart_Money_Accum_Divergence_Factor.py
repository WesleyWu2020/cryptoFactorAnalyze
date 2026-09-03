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
from operator_utils import PCT_CHANGE


def create_smart_money_accum_divergence_factor(
    window=20, rebalance_period=10, top_n=50, eps=1e-5
):
    """
    Smart Money Accumulation Divergence:
    RANK(PCT_CHANGE(QUOTE_VOLUME / (TRADES_COUNT + eps), window))
    - RANK(PCT_CHANGE(CLOSE, window))
    """
    print("开始构建聪明钱潜伏背离因子 (Smart Money Accumulation Divergence)")
    print(
        f"参数: window={window}, rebalance_period={rebalance_period}, top_n={top_n}, eps={eps}"
    )

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
        gp["ats"] = gp["quote_volume"] / (gp["trades_count"] + eps)
        gp["ats_roc"] = PCT_CHANGE(gp["ats"], n=window)
        gp["price_roc"] = PCT_CHANGE(gp["close"], n=window)
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        # Live 推理需要保留 future_ret=NaN 的末端日期
        return gp[["date", "symbol", "ats_roc", "price_roc", "future_ret"]].dropna(
            subset=["ats_roc", "price_roc"]
        )

    print("计算截面rank背离并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df_raw = pd.concat(result_dfs, ignore_index=True)
    factor_df_raw["ats_roc_rank"] = factor_df_raw.groupby("date")["ats_roc"].rank(
        method="average", pct=True
    )
    factor_df_raw["price_roc_rank"] = factor_df_raw.groupby("date")["price_roc"].rank(
        method="average", pct=True
    )
    factor_df_raw["smart_money_accum_div_raw"] = (
        factor_df_raw["ats_roc_rank"] - factor_df_raw["price_roc_rank"]
    )

    factor_df_live = winsorize_by_date(
        factor_df_raw, col="smart_money_accum_div_raw", n_std=3.0
    )
    factor_df_live = rank_to_unit_by_date(
        factor_df_live, col="smart_money_accum_div_raw", out_col="factor"
    )

    # 回测/报告使用：仅保留 future_ret 有效样本
    factor_df_backtest = factor_df_live.dropna(subset=["future_ret"]).copy()
    factor_df_backtest = factor_df_backtest.rename(columns={"symbol": "instrument"})[
        ["date", "instrument", "factor", "future_ret"]
    ]

    out_path = save_factor_df(
        factor_df_backtest,
        file_prefix=f"smart_money_accum_divergence_{window}d_rebalance{rebalance_period}d_",
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
    create_smart_money_accum_divergence_factor(
        window=20, rebalance_period=10, top_n=50, eps=1e-5
    )
