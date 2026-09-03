# factor_analyse/factor_mining/Volatility_Order_Flow_Factor.py
import pandas as pd
import numpy as np

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample
)

def create_volatility_order_flow_factor(window=20, rebalance_period=3, availability_lookback_days=90):
    """
    波动率-订单流:
      roc = (close/close.shift(window) - 1)
      range_pct = ((high-low)/close.shift(1)).rolling(window).mean()
      vol_eff = roc / range_pct
      avg_trade_size = quote_volume / trades_count
      trades_ratio = trades_count / MA(trades_count, window)
      avg_trade_size_ratio = avg_trade_size / MA(avg_trade_size, window)
      order_flow = trades_ratio / avg_trade_size_ratio
      vof_raw = vol_eff * order_flow
    """
    print(f"开始构建波动率-订单流因子... 参数: window={window}, rebalance={rebalance_period}")

    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(historical_df, lookback_days=availability_lookback_days, mode="window")
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    df = load_kline_df()

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        gp['roc'] = gp['close'] / gp['close'].shift(window) - 1
        gp['range_pct'] = ((gp['high'] - gp['low']) / gp['close'].shift(1)).rolling(window=window).mean()
        gp['vol_eff'] = gp['roc'] / (gp['range_pct'] + 1e-8)

        gp['avg_trade_size'] = gp['quote_volume'] / (gp['trades_count'] + 1e-8)
        gp['trades_ratio'] = gp['trades_count'] / (gp['trades_count'].rolling(window=window).mean() + 1e-8)
        gp['avg_trade_size_ratio'] = gp['avg_trade_size'] / (gp['avg_trade_size'].rolling(window=window).mean() + 1e-8)
        gp['order_flow'] = gp['trades_ratio'] / (gp['avg_trade_size_ratio'] + 1e-8)

        gp['vof_raw'] = gp['vol_eff'] * gp['order_flow']
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)
        return gp[['date', 'symbol', 'vof_raw', 'future_ret']].dropna()

    print("计算波动率-订单流因子（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    factor_df = winsorize_by_date(factor_df, col='vof_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='vof_raw', out_col='factor')
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]

    out_path = save_factor_df(factor_df, file_prefix=f"volatility_order_flow_{window}d_rebalance{rebalance_period}d_")
    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n无未来函数因子统计信息:")
    print(factor_df['factor'].describe())
    return factor_df

if __name__ == "__main__":
    create_volatility_order_flow_factor(window=20, rebalance_period=8)
