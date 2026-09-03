import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

from util_factor import (
    load_kline_df,
    build_available_tokens_by_date_from_kline,
    filter_group_by_availability, group_apply_with_progress,
    winsorize_by_date, rank_to_unit_by_date, save_factor_df, print_availability_sample,
    print_factor_summary, print_latest_date_inference, future_return
)

def calculate_volume_stability(data: pd.DataFrame, 
                              lookback_days: int = 20,
                              volume_col: str = 'quote_volume',
                              trades_col: str = 'trades_count') -> pd.Series:
    """
    计算量稳因子（成交量稳定性）
    
    参数:
        data : 包含加密货币数据的DataFrame（需按时间排序）
        lookback_days : 回溯期（默认20天）
        volume_col : 成交量列名（默认'quote_volume'）
        trades_col : 交易笔数列名（默认'trades_count'）
    
    返回:
        volume_stability : 量稳因子值（pd.Series）
    """
    data = data.copy()
    # 计算每笔交易的平均成交量，作为换手率的代理指标
    data['turnover'] = data[volume_col] / (data[trades_col] + 1e-8)  # 防止除零
    
    # 计算N日均值
    data['turnover_mean'] = data['turnover'].rolling(window=lookback_days).mean()
    
    # 计算N日标准差
    data['turnover_std'] = data['turnover'].rolling(window=lookback_days).std()
    
    # 量稳因子：均值/标准差
    data['volume_stability'] = data['turnover_mean'] / (data['turnover_std'] + 1e-8)  # 防止除零
    
    return data['volume_stability']

def create_volume_stability_factor(lookback_days=20, rebalance_period=3, top_n=50):
    """
    量稳因子 (Volume Stability) —— 无未来函数版本

    原理:
    1. 计算每笔交易的平均成交量作为换手率的代理指标
    2. 计算N日均值和标准差
    3. 量稳因子 = 均值/标准差

    策略：
    - 每天只使用前50个token（基于Binance实际成交额排名）
    - 保留这些token的完整历史数据（包括不在前50时的数据）
    - 考虑调仓周期：确保调仓时即使某个token某天不在前50，但调仓日它进入前50时，我们仍有它的数据
    """
    print(f"开始构建量稳因子 (Volume Stability) —— 无未来函数版本")
    print(f"参数: lookback_days={lookback_days}, rebalance_period={rebalance_period}, top_n={top_n}")

    # K线数据（包含所有可能进入前50的token的完整数据）
    df = load_kline_df()

    # 基于K线数据构建每日前50排名（考虑调仓周期）
    print("🔍 基于Binance K线数据构建每日前50排名...")
    available_tokens_by_date = build_available_tokens_by_date_from_kline(
        kline_df=df,
        top_n=top_n,
        ranking_method='quote_volume',  # 使用成交额排名
        rebalance_period=rebalance_period,  # 考虑调仓周期
        lookback_buffer=20,  # 回看缓冲区
        strict_top_n=True  # 严格限制每天只使用前50个token
    )
    print("🔍 生成各日期可用token列表（每天前50，考虑调仓周期）...")
    print_availability_sample(available_tokens_by_date, n=3)

    df = load_kline_df()

    # 单symbol计算
    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < lookback_days + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # 计算量稳因子
        gp['volume_stability'] = calculate_volume_stability(gp, lookback_days)

        # 未来收益（保持与原文件一致：对数收益率）
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        # Live 推理需要保留 future_ret=NaN 的末端日期，只移除因子本身无效的行
        return gp[["date", "symbol", "volume_stability", "future_ret"]].dropna(subset=["volume_stability"])

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df_raw = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df_live = winsorize_by_date(factor_df_raw, col="volume_stability", n_std=3.0)
    factor_df_live = rank_to_unit_by_date(factor_df_live, col="volume_stability", out_col="factor")

    # 回测/报告使用：只保留 future_ret 有效样本（保持历史统计口径）
    factor_df_backtest = factor_df_live.dropna(subset=["future_ret"]).copy()

    # 输出
    factor_df_backtest = factor_df_backtest.rename(columns={"symbol": "instrument"})[
        ["date", "instrument", "factor", "future_ret"]
    ]
    out_path = save_factor_df(
        factor_df_backtest,
        file_prefix=f"volume_stability_{lookback_days}d_rebalance{rebalance_period}d_"
    )

    print_factor_summary(factor_df_backtest, out_path)

    # 添加最新日期推理打印
    factor_df_live = factor_df_live.rename(columns={"symbol": "instrument"})[
        ["date", "instrument", "factor", "future_ret"]
    ]
    latest_date = factor_df_live['date'].max()
    print_latest_date_inference(factor_df_live, latest_date, groups=5)

    print(
        f"ℹ️ Live推理日期范围: {factor_df_live['date'].min().date()} ~ {factor_df_live['date'].max().date()} | "
        f"回测有效日期范围: {factor_df_backtest['date'].min().date()} ~ {factor_df_backtest['date'].max().date()}"
    )

    return factor_df_backtest

if __name__ == "__main__":
    create_volume_stability_factor(lookback_days=20, rebalance_period=10, top_n=50)
