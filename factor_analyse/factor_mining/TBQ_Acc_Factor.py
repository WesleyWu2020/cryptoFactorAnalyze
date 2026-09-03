# factor_analyse/factor_mining/TBQ_Acc_Factor.py
import pandas as pd
import numpy as np

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

def create_taker_buy_ratio_accumulation_factor(accumulation_window=20, vol_eff_window=20, rebalance_period=5, availability_lookback_days=90):
    """
    主动买入比率累积量 × 波动效率因子 (Taker Buy Ratio Accumulation × Volatility Efficiency Factor)
    
    思路:
    1. 计算主动买入比率: taker_buy_quote / quote_volume
    2. 计算20日累积量: 对主动买入比率进行20日累积
    3. 计算波动效率: roc / range_pct
       - roc = close/close.shift(window) - 1
       - range_pct = mean((high - low) / close.shift(1), window)
    4. 组合: 累积量排名 × 波动效率
    
    参数:
    - accumulation_window: 累积计算窗口，默认20天
    - vol_eff_window: 波动效率计算窗口，默认20天
    - rebalance_period: 调仓周期，默认5天
    
    - 平均持有期: 5-10天
    """
    print(f"开始构建主动买入比率累积量 × 波动效率因子...")
    print(f"参数: accumulation_window={accumulation_window}, vol_eff_window={vol_eff_window}, rebalance_period={rebalance_period}")

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
        if len(group) < max(accumulation_window, vol_eff_window) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 主动买入比率累积量计算 ===
        
        # 1. 计算主动买入比率
        gp['taker_buy_ratio'] = gp['taker_buy_quote'] / (gp['quote_volume'] + 1e-8)
        
        # 2. 处理极端值
        gp['taker_buy_ratio'] = gp['taker_buy_ratio'].replace([np.inf, -np.inf], np.nan)
        gp['taker_buy_ratio'] = gp['taker_buy_ratio'].clip(0, 1)  # 比率应该在[0, 1]范围内
        
        # 3. 计算20日累积量
        gp['accumulation'] = gp['taker_buy_ratio'].rolling(
            window=accumulation_window, 
            min_periods=accumulation_window
        ).sum()
        
        # === 波动效率计算 ===
        
        # 4. 计算收益率 (roc)
        gp['roc'] = gp['close'] / gp['close'].shift(vol_eff_window) - 1
        
        # 5. 计算价格区间百分比 (range_pct)
        gp['range_pct'] = ((gp['high'] - gp['low']) / (gp['close'].shift(1) + 1e-8)).rolling(
            window=vol_eff_window, 
            min_periods=vol_eff_window
        ).mean()
        
        # 6. 计算波动效率
        gp['volatility_efficiency'] = gp['roc'] / (gp['range_pct'] + 1e-8)
        
        # 7. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "accumulation", "volatility_efficiency", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    # 合并所有token的数据
    all_data = pd.concat(result_dfs, ignore_index=True)
    
    # 按日期分组计算横截面排名
    print("计算横截面排名和组合因子...")
    def combine_per_date(df_date):
        df_date = df_date.copy()
        
        # 计算累积量的横截面排名
        df_date['accumulation_rank'] = df_date['accumulation'].rank(pct=True)
        
        # 组合因子: 累积量排名 × 波动效率
        df_date['alpha_raw'] = df_date['accumulation_rank'] * df_date['volatility_efficiency']
        
        return df_date
    
    factor_df = all_data.groupby('date', group_keys=False).apply(combine_per_date)
    factor_df = factor_df.dropna(subset=['alpha_raw'])
    
    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"taker_buy_ratio_accumulation_vol_eff_{accumulation_window}d_{vol_eff_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_taker_buy_ratio_accumulation_factor(accumulation_window=10, vol_eff_window=10, rebalance_period=10)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_taker_buy_ratio_accumulation_factor(accumulation_window=10, vol_eff_window=10, rebalance_period=3)
    
    # 可选：更长的窗口，适应长期趋势
    # create_taker_buy_ratio_accumulation_factor(accumulation_window=30, vol_eff_window=30, rebalance_period=10)
