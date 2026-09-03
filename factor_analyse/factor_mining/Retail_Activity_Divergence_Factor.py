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


def create_retail_activity_divergence_factor(lookback_days=20, rebalance_period=10, top_n=50):
    """
    散户活跃度背离因子 (Retail Activity Divergence) —— 无未来函数版本

    逻辑:
    1. 计算 trades_count 与 quote_volume 的 N 日移动均值
    2. 在每日截面上分别计算两者的相对排名（百分位）
    3. Divergence = rank(trades_ma) - rank(quote_volume_ma)

    含义:
    - 因子越高，表示“交易笔数排名显著高于成交额排名”，通常反映散户噪音交易更活跃。
    """
    print("开始构建散户活跃度背离因子 (Retail Activity Divergence) —— 无未来函数版本")
    print(f"参数: lookback_days={lookback_days}, rebalance_period={rebalance_period}, top_n={top_n}")

    # K线数据（包含完整历史）
    df = load_kline_df()

    # 基于K线构建每日TopN可用token（考虑调仓周期）
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
        if len(group) < lookback_days + rebalance_period + 2:
            return pd.DataFrame()

        gp = group.copy()
        gp["trades_ma"] = gp["trades_count"].rolling(window=lookback_days).mean()
        gp["quote_volume_ma"] = gp["quote_volume"].rolling(window=lookback_days).mean()
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        # Live 推理需要保留 future_ret=NaN 的末端日期
        return gp[["date", "symbol", "trades_ma", "quote_volume_ma", "future_ret"]].dropna(
            subset=["trades_ma", "quote_volume_ma"]
        )

    print("计算移动均值并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df_raw = pd.concat(result_dfs, ignore_index=True)

    # 日截面排名差：交易笔数排名 - 成交额排名
    factor_df_raw["trades_rank"] = factor_df_raw.groupby("date")["trades_ma"].rank(method="average", pct=True)
    factor_df_raw["quote_volume_rank"] = factor_df_raw.groupby("date")["quote_volume_ma"].rank(
        method="average", pct=True
    )
    factor_df_raw["retail_activity_divergence"] = (
        factor_df_raw["trades_rank"] - factor_df_raw["quote_volume_rank"]
    )

    # 去极值 + 按日秩归一化到 [-1, 1]
    factor_df_live = winsorize_by_date(factor_df_raw, col="retail_activity_divergence", n_std=3.0)
    factor_df_live = rank_to_unit_by_date(
        factor_df_live, col="retail_activity_divergence", out_col="factor"
    )

    # 回测/报告使用：仅保留 future_ret 有效样本
    factor_df_backtest = factor_df_live.dropna(subset=["future_ret"]).copy()

    factor_df_backtest = factor_df_backtest.rename(columns={"symbol": "instrument"})[
        ["date", "instrument", "factor", "future_ret"]
    ]
    out_path = save_factor_df(
        factor_df_backtest,
        file_prefix=f"retail_activity_divergence_{lookback_days}d_rebalance{rebalance_period}d_",
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
    create_retail_activity_divergence_factor(lookback_days=20, rebalance_period=10, top_n=50)
