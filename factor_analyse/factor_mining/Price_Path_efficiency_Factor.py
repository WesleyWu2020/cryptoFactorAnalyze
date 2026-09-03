# factor_analyse/factor_mining/Price.py
import pandas as pd
import numpy as np

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample
)

def create_price_path_efficiency_factor(window=20, rebalance_period=1, availability_lookback_days=90):
    """
    涨跌距离/路径（价格路径效率，方向性）
    定义:
      net_ret  = close/close.shift(window) - 1
      path_len = sum_{t-window+1..t} |pct_change(close)|
      ppe_raw  = net_ret / path_len   # 路径效率，方向性，约[-1, 1]
    含义:
      同期净涨跌相对“走过的路径”越高，路径效率越高；噪声少、趋势更干净。
    """
    print(f"开始构建 涨跌距离/路径 因子 (Price Path Efficiency)")
    print(f"参数: window={window}, rebalance_period={rebalance_period}")

    # 历史市值可用性池（避免未来函数）
    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(
        historical_df, lookback_days=availability_lookback_days, mode="window"
    )
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # K线数据
    df = load_kline_df()

    # 单symbol计算
    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # 组件
        gp["net_ret"] = gp["close"] / gp["close"].shift(window) - 1
        gp["path_len"] = gp["close"].pct_change().abs().rolling(window).sum()
        gp["ppe_raw"] = gp["net_ret"] / (gp["path_len"] + 1e-8)

        # 未来收益（与短持有期评估对齐：百分比）
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="pct")

        # 历史可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "ppe_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="ppe_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="ppe_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"price_path_efficiency_{window}d_rebalance{rebalance_period}d_")

    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print(f"\n归一化验证: 范围[{factor_df['factor'].min():.6f}, {factor_df['factor'].max():.6f}], 均值={factor_df['factor'].mean():.6f}")
    return factor_df

if __name__ == "__main__":
    # 默认短持有期
    create_price_path_efficiency_factor(window=10, rebalance_period=10)