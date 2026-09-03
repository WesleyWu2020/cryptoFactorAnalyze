# factor_analyse/factor_mining/UBL_Factor.py
import pandas as pd
import numpy as np

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary, print_latest_date_inference
)

def create_ubl_factor(std_window=20, rebalance_period=10, availability_lookback_days=30):
    """
    UBL 因子 —— 无未来函数版本
    UBL = zscore(蜡烛图上影线的标准差) + zscore(威廉下影线的均值)
    
    计算逻辑:
    1. 蜡烛图上影线 = high - max(open, close)
    2. 上影线标准差 = std(上影线, std_window)
    3. 威廉下影线 = close - low
    4. 威廉下影线均值 = 当日下影线 / 7日平均下影线 = (close - low) / (close - low).rolling(7).mean()
    5. zscore(上影线标准差) = 按日期横截面标准化
    6. zscore(威廉下影线均值) = 按日期横截面标准化
    7. UBL = zscore(上影线标准差) + zscore(威廉下影线均值)
    
    参数:
    - std_window: 上影线标准差计算窗口，默认20天
    - rebalance_period: 调仓周期，默认10天
    - availability_lookback_days: 可用性回看天数，默认30天
    """
    print(f"开始构建 UBL 因子 (Upper Shadow Std + Lower Shadow Mean)")
    print(f"参数: std_window={std_window}, rebalance_period={rebalance_period}")

    # 可用性池
    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(
        historical_df, lookback_days=availability_lookback_days, mode="window"
    )
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # K线数据
    df = load_kline_df()

    # 横截面zscore函数
    def cs_zscore(x):
        """按日期横截面标准化"""
        m = x.mean()
        s = x.std()
        if s == 0 or np.isnan(s):
            return (x - m)  # 避免除0，退化为中心化
        return (x - m) / s

    # 单symbol计算
    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < std_window + rebalance_period + 7 + 2:
            return pd.DataFrame()
        gp = group.copy()

        # 1. 计算蜡烛图上影线 = high - max(open, close)
        gp["upper_shadow"] = gp["high"] - gp[["open", "close"]].max(axis=1)
        
        # 2. 计算上影线的标准差（滚动窗口）
        gp["upper_shadow_std"] = gp["upper_shadow"].rolling(window=std_window).std()
        
        # 3. 计算威廉下影线 = close - low
        gp["lower_shadow"] = gp["close"] - gp["low"]
        
        # 4. 计算威廉下影线的均值 = 当日下影线 / n日平均下影线
        gp["lower_shadow_ma"] = gp["lower_shadow"].rolling(window=std_window).mean()
        gp["lower_shadow_mean"] = gp["lower_shadow"] / (gp["lower_shadow_ma"] + 1e-8)  # 避免除零

        # 未来收益
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "upper_shadow_std", "lower_shadow_mean", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 按日期进行横截面zscore标准化
    print("按日期进行横截面zscore标准化...")
    factor_df["zscore_upper_shadow_std"] = factor_df.groupby("date")["upper_shadow_std"].transform(cs_zscore)
    factor_df["zscore_lower_shadow_mean"] = factor_df.groupby("date")["lower_shadow_mean"].transform(cs_zscore)
    
    # 计算UBL因子 = zscore(上影线标准差) + zscore(威廉下影线均值)
    factor_df["ubl_raw"] = factor_df["zscore_lower_shadow_mean"] # factor_df["zscore_upper_shadow_std"] +

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="ubl_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="ubl_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"ubl_{std_window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)
    
    # 添加最新日期推理打印
    latest_date = factor_df['date'].max()
    print_latest_date_inference(factor_df, latest_date, groups=5)
    
    return factor_df

if __name__ == "__main__":
    create_ubl_factor(std_window=30, rebalance_period=10)