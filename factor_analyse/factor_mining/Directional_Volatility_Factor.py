import pandas as pd
import numpy as np

from util_factor import (
    load_available_tokens_from_index_cache, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample
)

def create_directional_volatility_factor(window=20, rebalance_period=3, availability_lookback_days=90, top_n=50):
    """
    定向波动率因子:
      up_volatility = (high - open) / open
      down_volatility = (open - low) / open
      up_roc = (high_t - open_{t-window}) / open_{t-window}
      down_roc = (open_{t-window} - low_t) / open_{t-window}
      factor_raw = (up_roc / MA(up_volatility, window)) - (down_roc / MA(down_volatility, window))
    """
    print(f"开始构建定向波动率因子 (Directional_Volatility)... 参数: window={window}, rebalance_period={rebalance_period}")

    # 数据
    df = load_kline_df()

    # 可用性池：与主流程因子统一，使用指数缓存成分股
    print("🔍 从指数缓存文件加载市值50成分股列表...")
    all_dates = sorted(df['date'].unique())
    available_tokens_by_date = load_available_tokens_from_index_cache(
        cache_file='index_cache_equal_weight_30_50.csv',
        top_n=top_n,
        all_dates=all_dates
    )
    if not available_tokens_by_date:
        raise FileNotFoundError("无法从指数缓存文件加载成分股列表，请确保 index_cache_equal_weight_30_50.csv 文件存在")

    print("🔍 生成各日期可用token列表（基于市值50等权指数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        gp['up_volatility'] = (gp['high'] - gp['open']) / gp['open']
        gp['down_volatility'] = (gp['open'] - gp['low']) / gp['open']
        eps = 1e-10
        gp['up_volatility'] = gp['up_volatility'].replace(0, eps)
        gp['down_volatility'] = gp['down_volatility'].replace(0, eps)

        gp['up_roc'] = (gp['high'] - gp['open'].shift(window)) / gp['open'].shift(window)
        gp['down_roc'] = (gp['open'].shift(window) - gp['low']) / gp['open'].shift(window)

        gp['up_eff'] = gp['up_roc'] / gp['up_volatility'].rolling(window).mean()
        gp['down_eff'] = gp['down_roc'] / gp['down_volatility'].rolling(window).mean()

        gp['dv_raw'] = gp['up_eff'] - gp['down_eff']
        gp = gp.replace([np.inf, -np.inf], np.nan)
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)
        return gp[['date', 'symbol', 'dv_raw', 'future_ret']].dropna()

    print("计算定向波动率因子（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='dv_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='dv_raw', out_col='factor')

    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]

    out_path = save_factor_df(factor_df, file_prefix=f"directional_volatility_{window}d_rebalance{rebalance_period}d_")
    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n无未来函数因子统计信息:")
    print(factor_df['factor'].describe())
    return factor_df

if __name__ == "__main__":
    create_directional_volatility_factor(window=20, rebalance_period=10)
