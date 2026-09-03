import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date,
    load_kline_df, filter_group_by_availability, group_apply_with_progress,
    rank_to_unit_by_date, save_factor_df, print_availability_sample
)

def create_net_order_flow_factor(window=20, rebalance_period=3, availability_lookback_days=90):
    """
    因子名称: net_order_flow（无未来函数版本）

    思路 —— 订单不平衡/净主动买卖流量
      ① 计算 daily_flow = taker_buy_quote - taker_sell_quote
         • 若没有 taker_sell_quote，则用 quote_volume - taker_buy_quote 近似
      ② 在过去 window 天内累计: net_flow = Σ(daily_flow) / Σ(quote_volume 或 taker_buy_quote)
      ③ 因子值 = net_flow
    """
    print(f"开始构建净订单流因子 (Net_Order_Flow) —— 无未来函数版本")
    print(f"参数: window={window}, rebalance_period={rebalance_period}")

    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(historical_df, lookback_days=availability_lookback_days, mode="window")
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    df = load_kline_df()

    if "taker_sell_quote" not in df.columns:
        if "quote_volume" in df.columns:
            df["taker_sell_quote"] = df["quote_volume"] - df["taker_buy_quote"]
            print("⚠️ 'taker_sell_quote' 缺失，已用 quote_volume - taker_buy_quote 近似")
        else:
            raise KeyError("数据缺少 'taker_sell_quote' 和 'quote_volume'，无法计算净流量")

    def calc_factor(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period:
            return pd.DataFrame()
        gp = group.copy()
        gp["daily_flow"] = gp["taker_buy_quote"] - gp["taker_sell_quote"]
        denom = gp["quote_volume"] if "quote_volume" in gp.columns else gp["taker_buy_quote"]
        gp["net_flow"] = gp["daily_flow"].rolling(window).sum() / denom.rolling(window).sum().replace(0, np.nan)
        gp["future_ret"] = gp["close"].pct_change(rebalance_period).shift(-rebalance_period)
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)
        return gp[["date", "symbol", "net_flow", "future_ret"]]

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", calc_factor)
    if not result_dfs:
        print("⚠️ 没有足够的数据生成因子")
        return pd.DataFrame()

    result = pd.concat(result_dfs, ignore_index=True).dropna(subset=["net_flow", "future_ret"])
    result = result.rename(columns={"symbol": "instrument"})
    factor_df = rank_to_unit_by_date(result, col="net_flow", out_col="factor")[["date", "instrument", "factor", "future_ret"]]

    out_path = save_factor_df(factor_df, file_prefix=f"net_order_flow_{window}d_rebalance{rebalance_period}d_")
    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n无未来函数因子统计信息:")
    print(factor_df["factor"].describe())
    return factor_df

if __name__ == "__main__":
    create_net_order_flow_factor(window=20, rebalance_period=10) 