# factor_analyse/factor_mining/N_Day_Momentum.py
import pandas as pd
import numpy as np

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary, print_latest_date_inference
)

def create_n_day_momentum_factor(window=1, rebalance_period=3, availability_lookback_days=30, 
                                 use_smoothing=False, smoothing_window=3, use_volatility_adjust=False):
    """
    N天动量因子 (N-Day Momentum) —— 无未来函数版本
    基于研究发现：1天滞后收益率的预测能力比所有其他特征组合起来还要强
    
    核心因子: momentum = log(close / close.shift(window))
    当window=1时，直接使用1天滞后收益率作为预测因子
    
    可选增强:
    - use_smoothing: 是否使用平滑处理（移动平均）
    - smoothing_window: 平滑窗口大小
    - use_volatility_adjust: 是否使用波动率调整
    """
    print(f"开始构建N天动量因子 (N-Day Momentum) —— 无未来函数版本")
    print(f"参数: window={window}, rebalance_period={rebalance_period}")
    print(f"增强选项: smoothing={use_smoothing}(window={smoothing_window}), volatility_adjust={use_volatility_adjust}")
    if window == 1:
        print("⚠️  使用1天滞后收益率（最强预测因子）")

    # 可用性池
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
        min_len = window + rebalance_period + 2
        if use_smoothing:
            min_len = max(min_len, smoothing_window + rebalance_period + 2)
        if use_volatility_adjust:
            min_len = max(min_len, 20 + rebalance_period + 2)  # 需要足够数据计算波动率
            
        if len(group) < min_len:
            return pd.DataFrame()
        gp = group.copy()

        # 核心：1天滞后收益率（最强预测因子）
        # 使用对数收益率形式
        gp["momentum_raw"] = np.log(gp["close"] / gp["close"].shift(window))
        
        # 可选增强1：平滑处理（移动平均）
        if use_smoothing and smoothing_window > 1:
            gp["momentum_raw"] = gp["momentum_raw"].rolling(window=smoothing_window).mean()
        
        # 可选增强2：波动率调整（Sharpe-like调整）
        if use_volatility_adjust:
            # 计算滚动波动率
            vol_window = 20
            gp["volatility"] = gp["momentum_raw"].rolling(window=vol_window).std()
            # 波动率调整：momentum / volatility (类似Sharpe ratio)
            gp["momentum_raw"] = gp["momentum_raw"] / (gp["volatility"] + 1e-8)

        # 未来收益（保持与原文件一致：对数收益率）
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "momentum_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="momentum_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="momentum_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"n_day_momentum_{window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)
    
    # 添加最新日期推理打印
    latest_date = factor_df['date'].max()
    print_latest_date_inference(factor_df, latest_date, groups=5)
    
    return factor_df

if __name__ == "__main__":
    # 使用1天滞后收益率（最强预测因子）
    # 默认不使用平滑和波动率调整，直接使用原始1天滞后收益率
    create_n_day_momentum_factor(
        window=1, 
        rebalance_period=1,
        use_smoothing=False,      # 设置为True可启用平滑处理
        smoothing_window=3,       # 平滑窗口大小
        use_volatility_adjust=False  # 设置为True可启用波动率调整
    )

