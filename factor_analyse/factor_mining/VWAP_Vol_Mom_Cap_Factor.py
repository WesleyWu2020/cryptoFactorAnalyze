import pandas as pd
import numpy as np

from util_factor import (
    load_kline_df,
    build_available_tokens_by_date_from_kline,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary, print_latest_date_inference
)


def create_vwap_vol_mom_cap_factor(window=20, rebalance_period=10, top_n=50):
    """
    VWAP-波动率-动量-市值 融合因子
    
    四个子因子：
    1. VWAP偏离度: (close - vwap) / vwap，其中 vwap = quote_volume / volume
       含义：收盘价相对于当日成交均价的偏离，正值表示尾盘强势/主力吸筹
       
    2. 波动率效率: |ROC| / mean((high-low)/close, window)
       含义：净价格变化相对于振幅的效率，高值表示趋势性强、噪声少
       
    3. 动量: close / close[t-window] - 1
       含义：经典N日价格动量
       
    4. 市值代理: log(quote_volume.rolling(window).mean())
       含义：用成交额均值作为市值/流动性代理，大市值通常更稳定
    
    合成方法：
    - 对每个子因子做横截面排名归一化到 [-1, 1]
    - 等权求和后再做横截面排名归一化
    """
    print(f"开始构建 VWAP-波动率-动量-市值 融合因子")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, top_n={top_n}")

    df = load_kline_df()

    print("🔍 基于Binance K线数据构建每日前50排名...")
    available_tokens_by_date = build_available_tokens_by_date_from_kline(
        kline_df=df,
        top_n=top_n,
        ranking_method='quote_volume',
    )
    print("🔍 生成各日期可用token列表（每天前50）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # 预筛选：只保留曾经出现在Top50中的symbol
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

        eps = 1e-10

        # ========== 子因子1: VWAP偏离度 ==========
        # vwap = quote_volume / volume (当日成交均价)
        gp['vwap'] = gp['quote_volume'] / (gp['volume'] + eps)
        # 偏离度 = (close - vwap) / vwap
        gp['vwap_deviation'] = (gp['close'] - gp['vwap']) / (gp['vwap'] + eps)

        # ========== 子因子2: 波动率效率 ==========
        # ROC = close / close.shift(window) - 1
        gp['roc'] = gp['close'] / gp['close'].shift(window) - 1
        # 日内振幅 = (high - low) / close
        gp['intraday_range'] = (gp['high'] - gp['low']) / (gp['close'] + eps)
        # 平均振幅
        gp['avg_range'] = gp['intraday_range'].rolling(window).mean()
        # 波动率效率 = |ROC| / 平均振幅
        gp['vol_efficiency'] = gp['roc'].abs() / (gp['avg_range'] + eps)

        # ========== 子因子3: 动量 ==========
        gp['momentum'] = gp['roc']  # 直接使用ROC作为动量

        # ========== 子因子4: 市值代理 ==========
        # 使用log(成交额均值)作为市值代理
        gp['avg_quote_volume'] = gp['quote_volume'].rolling(window).mean()
        gp['cap_proxy'] = np.log(gp['avg_quote_volume'] + 1)

        # ========== 未来收益 ==========
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        # 只丢弃子因子为NaN的行，保留future_ret=NaN的行（用于Live推理）
        cols_out = ["date", "symbol", "vwap_deviation", "vol_efficiency", "momentum", "cap_proxy", "future_ret"]
        return gp[cols_out].dropna(subset=["vwap_deviation", "vol_efficiency", "momentum", "cap_proxy"])

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # ========== 对每个子因子去极值 ==========
    for col in ["vwap_deviation", "vol_efficiency", "momentum", "cap_proxy"]:
        factor_df = winsorize_by_date(factor_df, col=col, n_std=3.0)

    # ========== 横截面排名归一化到 [-1, 1] ==========
    factor_df = rank_to_unit_by_date(factor_df, col="vwap_deviation", out_col="vwap_rank")
    factor_df = rank_to_unit_by_date(factor_df, col="vol_efficiency", out_col="voleff_rank")
    factor_df = rank_to_unit_by_date(factor_df, col="momentum", out_col="mom_rank")
    factor_df = rank_to_unit_by_date(factor_df, col="cap_proxy", out_col="cap_rank")

    # ========== 等权合成 ==========
    factor_df["composite_raw"] = (
        factor_df["vwap_rank"] +
        factor_df["voleff_rank"] +
        factor_df["mom_rank"] +
        factor_df["cap_rank"]
    ) / 4.0

    # 对合成因子再做横截面排名归一化
    factor_df = rank_to_unit_by_date(factor_df, col="composite_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"vwap_vol_mom_cap_{window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)

    latest_date = factor_df['date'].max()
    print_latest_date_inference(factor_df, latest_date, groups=5)

    return factor_df


if __name__ == "__main__":
    create_vwap_vol_mom_cap_factor(window=20, rebalance_period=10, top_n=50)
