# factor_analyse/factor_mining/Volatility_Efficiency.py
import pandas as pd
import numpy as np

from util_factor import (
    load_kline_df,
    load_available_tokens_from_index_cache,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary, print_latest_date_inference
)

def create_volatility_efficiency_factor(window=20, rebalance_period=3, top_n=50):
    """
    波动效率 (Volatility Efficiency) —— 无未来函数版本
      roc = close/close.shift(window) - 1
      range_pct = mean( (high - low) / close.shift(1), window )
      vol_eff = roc / range_pct
    
    策略：
    - 使用市值50等权指数的成分股列表（从index_cache_equal_weight_30_50.csv加载）
    - 对于每个日期，使用最近一次调仓日的成分股列表
    - 例如：计算12月17日的因子时，使用12月1日（最近一次调仓日）的成分股
    - 保留这些token的完整历史数据（包括不在成分股时的数据）
    """
    print(f"开始构建波动效率因子 (Volatility Efficiency) —— 无未来函数版本")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, top_n={top_n}")

    # K线数据（包含所有可能进入前50的token的完整数据）
    df = load_kline_df()
    
    # 从指数缓存文件加载市值50成分股列表
    print("🔍 从指数缓存文件加载市值50成分股列表...")
    all_dates = sorted(df['date'].unique())
    available_tokens_by_date = load_available_tokens_from_index_cache(
        cache_file='index_cache_equal_weight_30_50.csv',  # 指定使用市值50等权指数文件
        top_n=top_n,
        all_dates=all_dates  # 传入所有日期，将调仓日的成分股扩展到所有交易日
    )
    
    # 如果缓存文件不存在或为空，报错
    if not available_tokens_by_date:
        raise FileNotFoundError("无法从指数缓存文件加载成分股列表，请确保 index_cache_equal_weight_30_50.csv 文件存在")
    
    print("🔍 生成各日期可用token列表（基于市值50等权指数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # 单symbol计算
    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # 组件
        gp["roc"] = gp["close"] / gp["close"].shift(window) - 1
        gp["range_pct"] = ((gp["high"] - gp["low"]) / (gp["close"].shift(1) + 1e-8)).rolling(window=window).mean()
        gp["ve_raw"] = gp["roc"] / (gp["range_pct"] + 1e-8)

        # 未来收益（保持与原文件一致：百分比）
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "ve_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="ve_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="ve_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"volatility_efficiency_{window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)
    
    # 添加最新日期推理打印
    latest_date = factor_df['date'].max()
    print_latest_date_inference(factor_df, latest_date, groups=5)
    
    return factor_df

if __name__ == "__main__":
    create_volatility_efficiency_factor(window=20, rebalance_period=10) 