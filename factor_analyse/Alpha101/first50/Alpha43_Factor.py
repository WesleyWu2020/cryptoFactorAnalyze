# factor_analyse/Alpha101/Alpha43_Factor.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

# 路径以便导入 util_factor
current_dir = os.path.dirname(os.path.abspath(__file__))
factor_mining_dir = os.path.join(current_dir, '..', 'factor_mining')
import sys
sys.path.insert(0, factor_mining_dir)

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

def create_alpha43_factor(adv_window=20, vol_ts_rank_window=20, price_ts_rank_window=8, 
                         delta_window=7, rebalance_period=5, availability_lookback_days=90):
    """
    Alpha 43 因子 (币圈7×24h优化版)
    
    原始定义:
    Alpha#43 = (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    
    思路:
    - 交易量活跃度: ts_rank((volume / adv20), 20)，过去20天该交易量比率的时间序列排名
    - 价格下跌程度: ts_rank((-1 * delta(close, 7)), 8)，过去8天价格变化负值的时间序列排名
    - 交互效应: 两个时间序列排名相乘
    - 信号解读: 当股票交易量相对活跃且价格出现下跌时，可能存在反转机会
    - 平均持有期: 3-7天
    """
    print(f"开始构建 Alpha 43 因子...")
    print(f"参数: adv_window={adv_window}, vol_ts_rank_window={vol_ts_rank_window}, "
          f"price_ts_rank_window={price_ts_rank_window}, delta_window={delta_window}, "
          f"rebalance_period={rebalance_period}")

    # 可用性池
    historical_df = load_historical_marketcap()
    available_tokens_by_date = build_available_tokens_by_date(
        historical_df, lookback_days=availability_lookback_days, mode="window"
    )
    print("🔍 生成各日期可用token列表（避免未来函数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # K线数据
    df = load_kline_df()

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < max(adv_window, vol_ts_rank_window, price_ts_rank_window, delta_window) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 7×24h币圈特殊处理 ===
        
        # 1. 价格归一化处理（适应币圈价格差异）
        gp['price_base'] = gp['close'].rolling(window=20, min_periods=1).mean()
        gp['norm_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        
        # 2. 计算交易量比率 volume / adv20
        gp['adv20'] = gp['volume'].rolling(window=adv_window, min_periods=adv_window).mean()
        gp['vol_ratio'] = gp['volume'] / (gp['adv20'] + 1e-8)
        
        # 3. 计算价格变化负值 -1 * delta(close, 7)
        gp['neg_delta_close'] = -1 * (gp['norm_close'] - gp['norm_close'].shift(delta_window))
        
        # 4. 计算时间序列排名
        # 注意：这里使用rolling.apply而不是rolling.rank，因为我们需要对窗口内的值进行排名
        def ts_rank(series):
            if series.isna().any():
                return np.nan
            # 对序列进行排名，返回最后一个元素的排名（百分比）
            return series.rank(pct=True).iloc[-1]
        
        # 交易量时间序列排名
        gp['vol_ts_rank'] = gp['vol_ratio'].rolling(window=vol_ts_rank_window, min_periods=vol_ts_rank_window).apply(
            ts_rank, raw=False
        )
        
        # 价格变化时间序列排名
        gp['price_ts_rank'] = gp['neg_delta_close'].rolling(window=price_ts_rank_window, min_periods=price_ts_rank_window).apply(
            ts_rank, raw=False
        )
        
        # 5. 计算因子值：两个时间序列排名相乘
        gp['alpha43_raw'] = gp['vol_ts_rank'] * gp['price_ts_rank']
        
        # 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[['date', 'symbol', 'alpha43_raw', 'future_ret']].dropna()

    print("计算 Alpha 43 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)
    
    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha43_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha43_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha43_volume_price_reversal_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数 (量价反转因子)
    create_alpha43_factor(adv_window=20, vol_ts_rank_window=20, price_ts_rank_window=8, 
                         delta_window=7, rebalance_period=5)
    
    # 可选：更短的窗口，更敏感地捕捉短期反转
    # create_alpha43_factor(adv_window=10, vol_ts_rank_window=10, price_ts_rank_window=5, 
    #                      delta_window=3, rebalance_period=3)
