# factor_analyse/factor_mining/Taker_Buy_Quote_Extreme_Efficiency.py
import pandas as pd
import numpy as np

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample
)

def create_taker_buy_quote_extreme_efficiency(window=20, rebalance_period=3, availability_lookback_days=90):
    """
    TBQ极值效率:
      roc = (close/close.shift(window) - 1)
      range_pct = ((high-low)/close.shift(1)).rolling(window).mean()
      vol_eff = roc / range_pct
      在每个rolling窗口内，按 taker_buy_quote 排序，Top10 vol_eff 和 / Bottom10 vol_eff 和
    """
    print(f"开始构建 TBQ 极值效率因子... 参数: window={window}, rebalance={rebalance_period}")

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

        vals, dates, syms, fwd = [], [], [], []
        fr = future_return(gp['close'], rebalance_period, method="log")
        for i in range(window - 1, len(gp) - rebalance_period):
            sub = gp.iloc[i - window + 1:i + 1]
            if len(sub) < window or sub['vol_eff'].isna().all():
                continue

            sub_sorted = sub.sort_values('taker_buy_quote', ascending=False)
            if len(sub_sorted) < 20:
                continue
            top10 = sub_sorted.head(10)['vol_eff'].sum()
            bot10 = sub_sorted.tail(10)['vol_eff'].sum()
            if bot10 == 0 or np.isnan(top10) or np.isnan(bot10):
                continue

            factor = top10 / bot10
            if not np.isfinite(factor):
                continue

            dates.append(gp.iloc[i]['date']); syms.append(symbol); vals.append(factor); fwd.append(fr.iloc[i])

        if not vals:
            return pd.DataFrame()

        res = pd.DataFrame({'date': dates, 'symbol': syms, 'tbqe_raw': vals, 'future_ret': fwd}).dropna()
        res = filter_group_by_availability(res, symbol, available_tokens_by_date)
        return res

    print("计算 TBQ 极值效率（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    factor_df = winsorize_by_date(factor_df, col='tbqe_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='tbqe_raw', out_col='factor')
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]

    out_path = save_factor_df(factor_df, file_prefix=f"taker_buy_quote_extreme_efficiency_{window}d_rebalance{rebalance_period}d_")
    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n无未来函数因子统计信息:")
    print(factor_df['factor'].describe())
    return factor_df

if __name__ == "__main__":
    create_taker_buy_quote_extreme_efficiency(window=20, rebalance_period=10) 