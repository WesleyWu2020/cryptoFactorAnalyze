# factor_analyse/factor_mining/APB_Factor.py
import pandas as pd
import numpy as np

from util_factor import (
    load_kline_df,
    build_available_tokens_by_date_from_kline,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary, print_latest_date_inference
)

def create_apb_factor(window=20, rebalance_period=3, top_n=50):
    """
    平均价格偏差 (Average Price Bias, APB)
    
    逻辑来源：
    东方证券 - 因子选股系列研究六十：基于量价关系度量股票的买卖压力
    
    公式：
      Daily_VWAP = quote_volume / volume
      
      分子（Window_TWAP）= (1/T) * Σ(vwap_t) 
        即：过去T天日均价的算术平均值
      
      分母（Window_VWAP）= Σ(volume_t * vwap_t) / Σ(volume_t)
        即：过去T天成交量加权的vwap均值（也就是这个月度区间内的vwap）
      
      APB = ln(Window_TWAP / Window_VWAP)
      
    含义：
      APB > 0: 买压（低位放量）； APB < 0: 卖压（高位放量）。
    
    参数建议：
      window=20 (对应研报 APB_1m)
      window=5  (对应研报 APB_5d)
    """
    print(f"开始构建平均价格偏差因子 (APB) —— 无未来函数版本")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, top_n={top_n}")

    # K线数据
    df = load_kline_df()
    
    # 基于K线数据构建每日前50排名
    print("🔍 基于Binance K线数据构建每日前50排名...")
    available_tokens_by_date = build_available_tokens_by_date_from_kline(
        kline_df=df,
        top_n=top_n,
        ranking_method='quote_volume', 
        rebalance_period=rebalance_period,
        lookback_buffer=20,
        strict_top_n=True
    )
    print("🔍 生成各日期可用token列表（每天前50，考虑调仓周期）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # 单symbol计算
    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # 1. 计算单日均价 (Daily VWAP)
        # 使用 quote_volume / volume 比 (open+close)/2 更能代表当天的真实均价
        # 添加 1e-8 防止除以0
        gp["daily_vwap"] = gp["quote_volume"] / (gp["volume"] + 1e-8)

        # 2. 计算窗口期 TWAP (Time-Weighted Average Price)
        # 即过去 window 天的日均价的算术平均值（分子）
        gp["window_twap"] = gp["daily_vwap"].rolling(window=window).mean()

        # 3. 计算窗口期成交量加权的 VWAP 均值（分母）
        # 分母 = sum(volume_t * vwap_t) / sum(volume_t)
        # 这是各个交易日成交量加权的vwap均值
        gp["vwap_weighted_sum"] = (gp["volume"] * gp["daily_vwap"]).rolling(window=window).sum()
        window_vol = gp["volume"].rolling(window=window).sum()
        gp["window_vwap"] = gp["vwap_weighted_sum"] / (window_vol + 1e-8)

        # 4. 计算 APB
        # ln(TWAP / VWAP)
        gp["apb_raw"] = np.log(gp["window_twap"] / (gp["window_vwap"] + 1e-8))

        # 未来收益
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "apb_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="apb_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="apb_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"apb_factor_{window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)
    
    # 添加最新日期推理打印
    latest_date = factor_df['date'].max()
    print_latest_date_inference(factor_df, latest_date, groups=5)
    
    return factor_df

if __name__ == "__main__":
    # 对应研报中的 APB_1m (20个交易日)
    create_apb_factor(window=3, rebalance_period=3)
    
    # 若要复现 APB_5d，可取消下面注释
    # create_apb_factor(window=5, rebalance_period=1)