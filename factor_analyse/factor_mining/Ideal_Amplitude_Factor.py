# factor_analyse/factor_mining/Ideal_Amplitude_Factor.py
import pandas as pd
import numpy as np

from util_factor import (
    load_kline_df,
    load_available_tokens_from_index_cache,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary, print_latest_date_inference
)

def create_ideal_amplitude_factor(window=20, rebalance_period=3, top_n=50, high_pct=0.4, low_pct=0.4):
    """
    理想振幅因子 (Ideal Amplitude Factor) —— 无未来函数版本
      amplitude = high/low - 1
      high_price_days = 收盘价最高的40%交易日
      low_price_days = 收盘价最低的40%交易日
      vhigh = high_price_days的振幅均值
      vlow = low_price_days的振幅均值
      ideal_amplitude = vhigh - vlow

    策略：
    - 使用市值50等权指数的成分股列表（从index_cache_equal_weight_30_50.csv加载）
    - 对于每个日期，使用最近一次调仓日的成分股列表
    - 例如：计算12月17日的因子时，使用12月1日（最近一次调仓日）的成分股
    - 保留这些token的完整历史数据（包括不在成分股时的数据）
    """
    print(f"开始构建理想振幅因子 (Ideal Amplitude Factor) —— 无未来函数版本")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, top_n={top_n}, high_pct={high_pct}, low_pct={low_pct}")

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

        # 计算每日振幅
        gp["amplitude"] = gp["high"] / (gp["low"] + 1e-8) - 1

        # 对每个日期计算理想振幅因子
        results = []

        for i in range(window-1, len(gp)):
            # 获取当前窗口的数据
            window_data = gp.iloc[i-window+1:i+1].copy()

            if len(window_data) < window:
                continue

            # 按收盘价排序
            sorted_by_close = window_data.sort_values('close', ascending=False)

            # 选择收盘价最高的high_pct比例的交易日
            n_high = max(1, int(len(sorted_by_close) * high_pct))
            high_price_days = sorted_by_close.head(n_high)

            # 选择收盘价最低的low_pct比例的交易日
            n_low = max(1, int(len(sorted_by_close) * low_pct))
            low_price_days = sorted_by_close.tail(n_low)

            # 计算高价振幅因子和低价振幅因子
            vhigh = high_price_days['amplitude'].mean()
            vlow = low_price_days['amplitude'].mean()

            # 理想振幅因子
            ideal_amplitude = vhigh - vlow

            # 记录结果
            current_date = window_data['date'].iloc[-1]
            results.append({
                'date': current_date,
                'symbol': symbol,
                'ideal_amplitude_raw': ideal_amplitude,
                'vhigh': vhigh,
                'vlow': vlow
            })

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)

        # 未来收益（保持与原文件一致：百分比）
        # 需要将result_df与原始group合并来计算future_return
        result_df = result_df.merge(gp[['date', 'close']], on='date', how='left')
        result_df["future_ret"] = future_return(result_df["close"], rebalance_period, method="log")

        # 可用性过滤
        result_df = filter_group_by_availability(result_df, symbol, available_tokens_by_date)

        return result_df[["date", "symbol", "ideal_amplitude_raw", "future_ret", "vhigh", "vlow"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="ideal_amplitude_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="ideal_amplitude_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret", "vhigh", "vlow"]]
    out_path = save_factor_df(factor_df, file_prefix=f"ideal_amplitude_{window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)

    # 添加最新日期推理打印
    latest_date = factor_df['date'].max()
    print_latest_date_inference(factor_df, latest_date, groups=5)

    return factor_df

if __name__ == "__main__":
    create_ideal_amplitude_factor(window=20, rebalance_period=10, high_pct=0.2, low_pct=0.2)