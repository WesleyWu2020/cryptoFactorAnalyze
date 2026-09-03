import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date,
    load_kline_df, filter_group_by_availability, group_apply_with_progress,
    winsorize_by_date, rank_to_unit_by_date, save_factor_df, print_availability_sample
)

def create_momentum_volume_extreme(window=20, rebalance_period=3, availability_lookback_days=90):
    """
    创建动量-成交量极值分组因子 (Momentum Volume Extreme)
    严格避免未来函数，只使用上一期已出现的token
    
    参数:
    window: 计算窗口，默认为20天
    rebalance_period: 调仓周期，默认为3天

    逻辑:
    - 计算20日动量
    - 在每个rolling window内，按成交量排序
    - 取最大10天的动量加总，除以最小10天的动量加总，作为该日因子值
    - 🔥 关键：结合历史市值排名，确保只使用上一期已出现的token
    """
    print(f"开始构建动量-成交量极值分组因子 (Momentum Volume Extreme)...")
    print(f"参数: window={window}, rebalance_period={rebalance_period}")
    print("🔥 使用无未来函数版本：只使用上一期已出现的token")
    
    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(historical_df, lookback_days=availability_lookback_days, mode="window")
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    df = load_kline_df()

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period:
            return pd.DataFrame()
        gp = group.copy()
        gp['momentum'] = gp['close'] / gp['close'].shift(window) - 1

        vals, dates, syms, fwd = [], [], [], []
        for i in range(window - 1, len(gp) - rebalance_period):
            current_date = gp.iloc[i]['date']
            sub = gp.iloc[i - window + 1:i + 1]
            if len(sub) < window:
                continue
            sub_sorted = sub.sort_values('volume', ascending=False)
            top10 = sub_sorted.head(10)['momentum'].sum()
            bot10 = sub_sorted.tail(10)['momentum'].sum()
            if bot10 == 0 or np.isnan(top10) or np.isnan(bot10):
                continue
            factor = top10 / bot10
            if np.isnan(factor) or np.isinf(factor):
                continue
            dates.append(current_date); syms.append(symbol); vals.append(factor)
            fwd.append(np.log(gp.iloc[i + rebalance_period]['close'] / gp.iloc[i]['close']) if i + rebalance_period < len(gp) else np.nan)

        if not vals:
            return pd.DataFrame()
        res = pd.DataFrame({'date': dates, 'symbol': syms, 'momentum_volume_extreme': vals, 'future_ret': fwd}).dropna()
        res = filter_group_by_availability(res, symbol, available_tokens_by_date)
        return res

    print("计算动量-成交量极值分组因子（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    factor_df = winsorize_by_date(factor_df, col='momentum_volume_extreme', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='momentum_volume_extreme', out_col='factor')
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]

    out_path = save_factor_df(factor_df, file_prefix=f"momentum_volume_extreme_{window}d_rebalance{rebalance_period}d_")
    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n无未来函数因子统计信息:")
    print(factor_df['factor'].describe())
    return factor_df

if __name__ == "__main__":
    create_momentum_volume_extreme(window=20, rebalance_period=10)
