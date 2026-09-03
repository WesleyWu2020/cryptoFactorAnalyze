# factor_analyse/factor_mining/Volume_TakerBuyBase_Extreme.py
import pandas as pd
import numpy as np

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample
)

def create_volume_takerbuybas_extreme(window=20, rebalance_period=3, availability_lookback_days=90):
    """
    成交量切分-主动买入Base 极值:
      taker_buy_base_momentum = rolling_sum(taker_buy_base, 10)
      在每个rolling窗口内，按 volume 排序，Top10 TBBase_Mom 和 / Bottom10 TBBase_Mom 和
    """
    print(f"开始构建成交量切分主动买入基础资产极值因子... 参数: window={window}, rebalance={rebalance_period}")

    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(historical_df, lookback_days=availability_lookback_days, mode="window")
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    df = load_kline_df()

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        tb_mom_win = 10
        gp['tb_base_mom'] = gp['taker_buy_base'].rolling(window=tb_mom_win, min_periods=tb_mom_win).sum()

        vals, dates, syms, fwd = [], [], [], []
        fr = future_return(gp['close'], rebalance_period, method="log")
        for i in range(window - 1, len(gp) - rebalance_period):
            sub = gp.iloc[i - window + 1:i + 1]
            if len(sub) < window or sub['tb_base_mom'].isna().all():
                continue

            sub_sorted = sub.sort_values('volume', ascending=False)
            if len(sub_sorted) < 20:
                continue
            top10 = sub_sorted.head(10)['tb_base_mom'].sum()
            bot10 = sub_sorted.tail(10)['tb_base_mom'].sum()
            if bot10 == 0 or np.isnan(top10) or np.isnan(bot10):
                continue

            factor = top10 / bot10
            if not np.isfinite(factor):
                continue

            dates.append(gp.iloc[i]['date']); syms.append(symbol); vals.append(factor); fwd.append(fr.iloc[i])

        if not vals:
            return pd.DataFrame()

        res = pd.DataFrame({'date': dates, 'symbol': syms, 'vtbe_raw': vals, 'future_ret': fwd}).dropna()
        res = filter_group_by_availability(res, symbol, available_tokens_by_date)
        return res

    print("计算成交量切分主动买入基础资产极值（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    factor_df = winsorize_by_date(factor_df, col='vtbe_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='vtbe_raw', out_col='factor')
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]

    out_path = save_factor_df(factor_df, file_prefix=f"volume_takerbuybas_extreme_{window}d_rebalance{rebalance_period}d_")
    print(f"✅ 因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n因子统计信息:")
    print(factor_df['factor'].describe())
    return factor_df

if __name__ == "__main__":
    create_volume_takerbuybas_extreme(window=20, rebalance_period=10)
