# factor_analyse/Alpha101/last51/Alpha57_Factor.py
import pandas as pd
import numpy as np
import sys
import os

# 添加路径以导入util_factor
current_dir = os.path.dirname(os.path.abspath(__file__))
factor_mining_dir = os.path.join(current_dir, '..', '..', 'factor_mining')
sys.path.insert(0, factor_mining_dir)

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

def create_alpha57_factor(
    ts_argmax_window=30,
    decay_window=2,
    rebalance_period=3, 
    availability_lookback_days=90
):
    """
    Alpha #57 因子 —— 无未来函数版本
    
    公式: (0 - (1 * ((close - (taker_buy_quote/taker_buy_base)) / decay_linear(rank(ts_argmax(close, ts_argmax_window)), decay_window))))
    
    核心步骤:
    1. ts_argmax(close, ts_argmax_window)：过去ts_argmax_window天收盘价最大值的位置（时间索引）
    2. rank(...)：对位置进行横截面排名
    3. decay_linear(..., decay_window)：decay_window天线性衰减加权
    4. (close - (taker_buy_quote/taker_buy_base))：收盘价与主动买入价格的差异
    5. ... / ...：差异除以衰减加权排名
    6. 结果取负值
    
    投资逻辑:
    - 当收盘价高于主动买入价格且最近出现历史高点时，可能存在过度乐观
    - 预期回调，体现反转投资逻辑
    
    参数:
    - ts_argmax_window: ts_argmax计算窗口，默认30天
    - decay_window: 线性衰减窗口，默认2天
    - rebalance_period: 调仓周期，默认3天
    
    平均持有期: 2-5天
    """
    print("开始构建 Alpha #57 因子...")
    print(f"参数: ts_argmax_window={ts_argmax_window}, decay_window={decay_window}, rebalance_period={rebalance_period}")

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
        if len(group) < ts_argmax_window + decay_window + 2:  # 需要足够的数据
            return pd.DataFrame()
        gp = group.copy()

        # === Alpha #57 因子计算 ===
        
        # 1. 计算主动买入价格 (taker_buy_quote / taker_buy_base)
        gp['taker_buy_price'] = gp['taker_buy_quote'] / (gp['taker_buy_base'] + 1e-8)
        
        # 2. 计算收盘价与主动买入价格的差异
        gp['price_diff'] = gp['close'] - gp['taker_buy_price']
        
        # 3. 计算过去ts_argmax_window天收盘价最大值的位置（时间索引）
        # 使用rolling apply来获取argmax
        def get_argmax(x):
            if len(x) == 0:
                return np.nan
            return np.argmax(x)
        
        gp['ts_argmax_close'] = gp['close'].rolling(
            window=ts_argmax_window, min_periods=ts_argmax_window
        ).apply(get_argmax)
        
        # 4. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "price_diff", "ts_argmax_close", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    # 合并所有token的数据
    all_data = pd.concat(result_dfs, ignore_index=True)
    
    # 按日期分组计算横截面排名和组合因子
    print("计算横截面排名和组合因子...")
    def combine_per_date(df_date):
        df_date = df_date.copy()
        
        # 对ts_argmax位置进行横截面排名
        df_date['rank_ts_argmax'] = df_date['ts_argmax_close'].rank(pct=True)
        
        # 计算decay_window天线性衰减加权
        df_date['decay_rank'] = df_date['rank_ts_argmax'].rolling(
            window=decay_window, min_periods=decay_window
        ).apply(lambda x: (x * np.arange(1, decay_window + 1)).sum() / sum(range(1, decay_window + 1)))
        
        # 组合因子: (close - taker_buy_price) / decay_linear(rank(ts_argmax), decay_window) 然后取负值
        df_date['alpha57_raw'] = -(df_date['price_diff'] / (df_date['decay_rank'] + 1e-8))
        
        return df_date
    
    factor_df = all_data.groupby('date', group_keys=False).apply(combine_per_date)
    factor_df = factor_df.dropna(subset=['alpha57_raw'])
    
    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha57_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha57_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha57_price_diff_ts_argmax_decay_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # === 参数配置区域 ===
    
    # 标准参数配置
    TS_ARGMAX_WINDOW = 14  # ts_argmax计算窗口
    DECAY_WINDOW = 2      # 线性衰减窗口
    REBALANCE_PERIOD = 7   # 调仓周期
    
    # 执行标准配置
    print("=== 执行标准配置 ===")
    create_alpha57_factor(
        ts_argmax_window=TS_ARGMAX_WINDOW,
        decay_window=DECAY_WINDOW,
        rebalance_period=REBALANCE_PERIOD
    )
    