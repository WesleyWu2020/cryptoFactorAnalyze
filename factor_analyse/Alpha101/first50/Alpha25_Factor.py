# factor_analyse/Alpha101/first50/Alpha25_Factor.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

# 路径以便导入 util_factor
current_dir = os.path.dirname(os.path.abspath(__file__))
factor_mining_dir = os.path.join(current_dir, '..', '..', 'factor_mining')
import sys
sys.path.insert(0, factor_mining_dir)

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

def create_alpha25_factor(adv_window=20, rebalance_period=5, availability_lookback_days=90):
    """
    Alpha 25 因子 (多维度复合反弹因子)
    
    原始定义:
    Alpha 25 = rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    
    思路:
    1. -1 * returns：反向收益，体现反转逻辑
    2. adv20：20日平均主动买入金额，流动性权重
    3. vwap：成交量加权平均价，市场认可价格水平
    4. high - close：日内抛压
    5. 四项相乘后横截面排名
    
    - 平均持有期: 5-10天
    """
    print(f"开始构建 Alpha 25 因子...")
    print(f"参数: adv_window={adv_window}, rebalance_period={rebalance_period}")

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
        if len(group) < adv_window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === Alpha 25 因子计算 ===
        
        # 1. 收益率
        gp['returns'] = gp['close'].pct_change()
        
        # 2. 20日平均主动买入金额
        gp['adv20'] = gp['taker_buy_quote'].rolling(window=adv_window, min_periods=adv_window).mean()
        
        # 3. VWAP（成交量加权平均价）归一化
        gp['vwap'] = (gp['quote_volume'] / gp['volume']).replace([np.inf, -np.inf], np.nan)
        gp['vwap_norm'] = gp['vwap'] / gp['close']
        
        # 4. high - close 归一化
        gp['high_close_diff'] = (gp['high'] - gp['close']) / gp['close']
        
        # 5. 复合项
        gp['alpha25_raw'] = (
            (-1 * gp['returns']) *
            gp['adv20'] *
            gp['vwap_norm'] *
            gp['high_close_diff']
        )
        
        # 6. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "alpha25_raw", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha25_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha25_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha25_composite_reversal_adv{adv_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_alpha25_factor(adv_window=7, rebalance_period=10)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_alpha25_factor(adv_window=10, rebalance_period=3)
