# factor_analyse/Alpha101/last51/Alpha56_Factor.py
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

def create_alpha56_factor(
    long_window=21, 
    short_window=3, 
    sum_window=7, 
    rebalance_period=5, 
    availability_lookback_days=90
):
    """
    Alpha #56 因子 —— 无未来函数版本
    
    公式: (0 - (1 * (rank((sum(returns, long_window) / sum(sum(returns, short_window), sum_window))) * rank((returns * taker_buy_quote)))))
    
    核心步骤:
    1. sum(returns, long_window)：过去long_window天收益率总和
    2. sum(returns, short_window)：过去short_window天收益率总和  
    3. sum(..., sum_window)：过去sum_window天的short_window收益率总和
    4. ... / ...：长期与短期收益率的比率
    5. rank(...)：对比率进行横截面排名
    6. (returns * taker_buy_quote)：收益率乘以主动买入金额
    7. rank(...)：对主动买入金额加权收益进行横截面排名
    8. 两个排名相乘后取负值
    
    投资逻辑:
    - 当币种长期收益率相对较好且主动买入金额加权收益较高时，可能存在过度反应
    - 预期回调，体现均值回归的投资逻辑
    
    参数:
    - long_window: 长期收益率计算窗口，默认21天
    - short_window: 短期收益率计算窗口，默认3天
    - sum_window: 短期收益率累积窗口，默认7天
    - rebalance_period: 调仓周期，默认5天
    
    平均持有期: 5-10天
    """
    print("开始构建 Alpha #56 因子...")
    print(f"参数: long_window={long_window}, short_window={short_window}, sum_window={sum_window}, rebalance_period={rebalance_period}")

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
        if len(group) < long_window + sum_window + 2:  # 需要足够的数据
            return pd.DataFrame()
        gp = group.copy()

        # === Alpha #56 因子计算 ===
        
        # 1. 计算收益率
        gp['returns'] = gp['close'].pct_change()
        
        # 2. 计算过去long_window天收益率总和
        gp['sum_returns_long'] = gp['returns'].rolling(window=long_window, min_periods=long_window).sum()
        
        # 3. 计算过去short_window天收益率总和
        gp['sum_returns_short'] = gp['returns'].rolling(window=short_window, min_periods=short_window).sum()
        
        # 4. 计算过去sum_window天的short_window收益率总和
        gp['sum_sum_returns_short'] = gp['sum_returns_short'].rolling(window=sum_window, min_periods=sum_window).sum()
        
        # 5. 计算长期与短期收益率的比率
        gp['returns_ratio'] = gp['sum_returns_long'] / (gp['sum_sum_returns_short'] + 1e-8)
        
        # 6. 计算收益率乘以主动买入金额
        gp['returns_taker_buy'] = gp['returns'] * gp['taker_buy_quote']
        
        # 7. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "returns_ratio", "returns_taker_buy", "future_ret"]].dropna()

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
        
        # 计算收益率比率的横截面排名
        df_date['rank_returns_ratio'] = df_date['returns_ratio'].rank(pct=True)
        
        # 计算主动买入金额加权收益的横截面排名
        df_date['rank_returns_taker_buy'] = df_date['returns_taker_buy'].rank(pct=True)
        
        # 组合因子: 两个排名相乘后取负值
        df_date['alpha56_raw'] = -(df_date['rank_returns_ratio'] * df_date['rank_returns_taker_buy'])
        
        return df_date
    
    factor_df = all_data.groupby('date', group_keys=False).apply(combine_per_date)
    factor_df = factor_df.dropna(subset=['alpha56_raw'])
    
    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha56_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha56_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha56_returns_taker_buy_{long_window}d_{short_window}d_{sum_window}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # === 参数配置区域 ===
    
    # 标准参数配置
    LONG_WINDOW = 21      # 长期收益率计算窗口
    SHORT_WINDOW = 3      # 短期收益率计算窗口  
    SUM_WINDOW = 7        # 短期收益率累积窗口
    REBALANCE_PERIOD = 2  # 调仓周期
    
    # 执行标准配置
    print("=== 执行标准配置 ===")
    create_alpha56_factor(
        long_window=LONG_WINDOW,
        short_window=SHORT_WINDOW, 
        sum_window=SUM_WINDOW,
        rebalance_period=REBALANCE_PERIOD
    )
