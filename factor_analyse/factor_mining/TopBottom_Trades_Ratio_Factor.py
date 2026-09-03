# factor_analyse/factor_mining/TopBottom_Trades_Ratio_Factor.py
import pandas as pd
import numpy as np

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

def create_topbottom_trades_ratio_factor(window=20, k_top=10, k_bottom=10, rebalance_period=3, availability_lookback_days=90):
    """
    Top-Bottom Trades Ratio 因子 —— 无未来函数版本

    定义:
      - 主动买盘 buy = taker_buy_quote
      - 主动卖盘 sell = quote_volume - taker_buy_quote
      - ratio = buy / sell
      - 对于每个日期，取过去 window 天：
          * 按 trades_count 排序
          * 前 k_top 天的 ratio 之和 / 后 k_bottom 天的 ratio 之和

    参数:
      - window: 滚动窗口长度 (默认 20)
      - k_top: 取 trades_count 排名前 k_top 的天数累积 (默认 10)
      - k_bottom: 取 trades_count 排名后 k_bottom 的天数累积 (默认 10)
      - rebalance_period: 调仓周期 (默认 3)
      - availability_lookback_days: 可用性窗口 (默认 90)
    """
    print(f"开始构建 Top-Bottom Trades Ratio 因子 —— 无未来函数版本")
    print(f"参数: window={window}, k_top={k_top}, k_bottom={k_bottom}, rebalance_period={rebalance_period}")

    # 可用性池
    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(
        historical_df, lookback_days=availability_lookback_days, mode="window"
    )
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # K线数据
    df = load_kline_df()

    # 缺失时构造 taker_sell_quote
    if "taker_sell_quote" not in df.columns:
        if "quote_volume" in df.columns:
            df["taker_sell_quote"] = df["quote_volume"] - df["taker_buy_quote"]
            print("⚠️ 'taker_sell_quote' 缺失，已用 quote_volume - taker_buy_quote 近似")
        else:
            raise KeyError("数据缺少 'taker_sell_quote' 和 'quote_volume'，无法计算 sell 侧金额")

    # 单symbol计算
    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy().reset_index(drop=True)

        # 基础变量
        gp["buy"] = pd.to_numeric(gp["taker_buy_quote"], errors="coerce")
        gp["sell"] = pd.to_numeric(gp["taker_sell_quote"], errors="coerce")
        gp["trades_count"] = pd.to_numeric(gp["trades_count"], errors="coerce")
        gp["ratio"] = gp["buy"] / (gp["sell"] + 1e-8)

        # 预分配
        gp["top_sum_ratio"] = np.nan
        gp["bottom_sum_ratio"] = np.nan

        # 手动滚动窗口：按 trades_count 排序，取 top/bottom 累积 ratio
        for i in range(window - 1, len(gp)):
            w = gp.iloc[i - window + 1:i + 1]
            w_sorted = w.sort_values("trades_count", ascending=False)

            top_sum = w_sorted["ratio"].head(k_top).sum()
            bottom_sum = w_sorted["ratio"].tail(k_bottom).sum()

            gp.loc[i, "top_sum_ratio"] = top_sum
            gp.loc[i, "bottom_sum_ratio"] = bottom_sum

        # 因子：Top / Bottom
        gp["tb_trades_ratio_raw"] = gp["top_sum_ratio"] / (gp["bottom_sum_ratio"] + 1e-8)

        # 未来收益（百分比）
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "tb_trades_ratio_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1, 1]
    factor_df = winsorize_by_date(factor_df, col="tb_trades_ratio_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="tb_trades_ratio_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(
        factor_df,
        file_prefix=f"topbottom_trades_ratio_{window}d_top{k_top}_bottom{k_bottom}_rebalance{rebalance_period}d_"
    )

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 示例：20日窗口，Top10/Bottom10，调仓10日
    create_topbottom_trades_ratio_factor(window=20, k_top=10, k_bottom=10, rebalance_period=10)
