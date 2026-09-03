# factor_analyse/factor_mining/Buy_Stability_Factor.py
import pandas as pd
import numpy as np

from util_factor import (
    load_kline_df,
    load_available_tokens_from_index_cache,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary, print_latest_date_inference
)

def create_buy_stability_factor(lookback_days=20, rebalance_period=10, top_n=50):
    """
    买入稳定因子（主动买入/成交额 的稳定性）—— 无未来函数版本
    - 定义: buy_ratio = taker_buy_quote / quote_volume
    - 计算: N日均值、N日标准差，稳定性 = 均值 / 标准差
    - 输出: 截面去极值 + 排名到[-1,1] 的 factor，与 future_ret
    
    策略：
    - 使用市值50等权指数的成分股列表（从index_cache_equal_weight_30_50.csv加载）
    - 对于每个日期，使用最近一次调仓日的成分股列表
    - 例如：计算12月17日的因子时，使用12月1日（最近一次调仓日）的成分股
    - 保留这些token的完整历史数据（包括不在成分股时的数据）
    """
    print(f"开始构建买入稳定因子 (Buy Stability) —— 无未来函数版本")
    print(f"参数: lookback_days={lookback_days}, rebalance_period={rebalance_period}, top_n={top_n}")

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
        if len(group) < lookback_days + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # 数值化
        gp["trades_count"] = pd.to_numeric(gp["trades_count"], errors="coerce").fillna(0)

        # 交易密度（每单位成交额的笔数）→ 比例化并做时间序列归一
        gp["trades_density"] = gp["trades_count"] / (gp["quote_volume"] + 1e-8)
        gp["trades_density_mean"] = gp["trades_density"].rolling(window=lookback_days, min_periods=lookback_days).mean()
        gp["trades_density_ratio"] = gp["trades_density"] / (gp["trades_density_mean"] + 1e-8)

        # 将原 buy/sell 的比值再除以交易密度的相对强度
        gp["buy_ratio"] = (
            gp["taker_buy_quote"] / (gp["quote_volume"] - gp["taker_buy_quote"] + 1e-8)
        ) * (gp["trades_density_ratio"] ** 2 + 1e-8)

        # 稳定性：均值/标准差
        gp["buy_ratio_mean"] = gp["buy_ratio"].rolling(window=lookback_days, min_periods=lookback_days).mean()
        gp["buy_ratio_std"] = gp["buy_ratio"].rolling(window=lookback_days, min_periods=lookback_days).std()
        gp["buy_stability_raw"] = gp["buy_ratio_mean"] / (gp["buy_ratio_std"] + 1e-8)

        # 未来收益（保持与原文件一致：对数收益率）
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "buy_stability_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值 + 截面秩归一
    factor_df = winsorize_by_date(factor_df, col="buy_stability_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="buy_stability_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"buy_stability_{lookback_days}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)
    
    # 添加最新日期推理打印
    latest_date = factor_df['date'].max()
    print_latest_date_inference(factor_df, latest_date, groups=5)
    
    return factor_df

if __name__ == "__main__":
    # 示例：20日窗口、调仓10日
    create_buy_stability_factor(lookback_days=20, rebalance_period=10)
