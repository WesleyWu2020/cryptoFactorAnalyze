import pandas as pd
import numpy as np

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample
)

def create_high_volatility_momentum_factor(window=20, atr_window=14, momentum_type="product", rebalance_period=3, availability_lookback_days=90):
    """
    高波动性动量:
      roc = pct_change(window)
      TR = max(high-low, |high-close_prev|, |low-close_prev|)
      ATR = MA(TR, atr_window)
      range_pct = (high-low)/close
      hv_momentum = roc*ATR (product) 或 roc/range_pct (ratio)
    """
    print(f"开始构建高波动性动量因子... 参数: window={window}, atr_window={atr_window}, type={momentum_type}, rebalance={rebalance_period}")

    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(historical_df, lookback_days=availability_lookback_days, mode="window")
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    df = load_kline_df()

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < max(window, atr_window) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        gp['roc'] = gp['close'].pct_change(periods=window)

        gp['high_low'] = gp['high'] - gp['low']
        gp['high_close_prev'] = (gp['high'] - gp['close'].shift(1)).abs()
        gp['low_close_prev'] = (gp['low'] - gp['close'].shift(1)).abs()
        gp['tr'] = gp[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)

        gp['atr'] = gp['tr'].rolling(window=atr_window).mean()
        gp['range_pct'] = gp['high_low'] / gp['close']

        if momentum_type == "product":
            gp['hvm_raw'] = gp['roc'] * gp['atr']
        else:
            gp['hvm_raw'] = gp['roc'] / gp['range_pct'].replace(0, np.nan)

        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)
        return gp[['date', 'symbol', 'hvm_raw', 'future_ret']].dropna()

    print("计算高波动性动量因子（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    factor_df = winsorize_by_date(factor_df, col='hvm_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='hvm_raw', out_col='factor')

    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]

    out_path = save_factor_df(factor_df, file_prefix=f"high_volatility_momentum_{window}d_{momentum_type}_rebalance{rebalance_period}d_")
    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n无未来函数因子统计信息:")
    print(factor_df['factor'].describe())
    return factor_df

if __name__ == "__main__":
    create_high_volatility_momentum_factor(window=20, atr_window=14, momentum_type="ratio", rebalance_period=10)