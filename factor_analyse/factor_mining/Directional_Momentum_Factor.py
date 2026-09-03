import pandas as pd
import numpy as np

from util_factor import (
    load_kline_df,
    build_available_tokens_by_date_from_kline,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary, print_latest_date_inference
)

def create_directional_momentum_factor(window=20, rebalance_period=10, top_n=50):
    """
    方向动量因子 (Directional Momentum) —— 无未来函数版本
      up_momentum = sum(max(0, daily_return)) for t in [1, window]
      down_momentum = sum(max(0, -daily_return)) for t in [1, window]
      factor_raw = up_momentum / (down_momentum + eps)

    策略：
    - 每天只使用前50个token（基于Binance实际成交额排名）
    - 只对曾进入Top50的token计算因子（跳过从未进入Top50的token）
    - 因子计算使用token的完整历史数据（确保rolling窗口正确），输出时按日期过滤到Top50
    """
    print(f"开始构建方向动量因子 (Directional Momentum) —— 无未来函数版本")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, top_n={top_n}")

    df = load_kline_df()

    # 基于K线数据构建每日前50排名
    print("🔍 基于Binance K线数据构建每日前50排名...")
    available_tokens_by_date = build_available_tokens_by_date_from_kline(
        kline_df=df,
        top_n=top_n,
        ranking_method='quote_volume',
    )
    print("🔍 生成各日期可用token列表（每天前50）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # 预筛选：只保留曾经出现在Top50中的symbol，跳过从未进入Top50的token
    ever_top50 = set()
    for symbols in available_tokens_by_date.values():
        ever_top50.update(symbols)
    before_count = df['symbol'].nunique()
    df = df[df['symbol'].isin(ever_top50)].copy()
    print(f"🔍 预筛选: {before_count} → {df['symbol'].nunique()} 个symbol（仅保留曾进入Top50的token）")

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        gp['daily_return'] = gp['close'].pct_change()

        # 向量化rolling计算，比 rolling().apply(lambda) 快几十倍
        gp['up_momentum'] = gp['daily_return'].clip(lower=0).rolling(window).sum()
        gp['down_momentum'] = gp['daily_return'].clip(upper=0).abs().rolling(window).sum()

        eps = 1e-10
        gp['dm_raw'] = gp['up_momentum'] / (gp['down_momentum'] + eps)

        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        # 可用性过滤：只保留该token在Top50中的日期
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        # 只丢弃因子值为NaN的行，保留future_ret=NaN的行（用于Live推理）
        return gp[["date", "symbol", "dm_raw", "future_ret"]].dropna(subset=["dm_raw"])

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="dm_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="dm_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"directional_momentum_{window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)

    # 最新日期推理（现在能取到真正的最新日期，而非rebalance_period天前）
    latest_date = factor_df['date'].max()
    print_latest_date_inference(factor_df, latest_date, groups=5)

    return factor_df

if __name__ == "__main__":
    create_directional_momentum_factor(window=20, rebalance_period=10, top_n=50)
