# factor_analyse/Alpha101/last51/Alpha60_Factor.py
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

def create_alpha60_factor(
    ts_argmax_window=10,
    rebalance_period=5, 
    availability_lookback_days=90
):
    """
    Alpha #60 因子 —— 无未来函数版本
    
    公式: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * taker_buy_quote)))) - scale(rank(ts_argmax(close, ts_argmax_window))))))
    
    核心步骤:
    1. (close - low) - (high - close)：收盘价相对于日内中点的偏离
    2. ... / (high - low)：标准化处理
    3. ... * taker_buy_quote：乘以主动买入金额
    4. rank(...)：横截面排名
    5. scale(...)：缩放标准化
    6. 2 * ...：乘以2
    7. ts_argmax(close, ts_argmax_window)：过去ts_argmax_window天收盘价最大值的位置
    8. rank(...)：对位置进行横截面排名
    9. scale(...)：缩放标准化
    10. 第一部分减去第二部分后取负值
    
    投资逻辑:
    - 结合日内位置分析、主动买入金额分析和历史高点分析
    - 当币种日内位置较好且有主动买入支持，但接近历史高点时，因子给出复合信号
    
    参数:
    - ts_argmax_window: ts_argmax计算窗口，默认10天
    - rebalance_period: 调仓周期，默认5天
    
    平均持有期: 3-7天
    """
    print("开始构建 Alpha #60 因子...")
    print(f"参数: ts_argmax_window={ts_argmax_window}, rebalance_period={rebalance_period}")

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
        if len(group) < ts_argmax_window + 2:  # 需要足够的数据
            return pd.DataFrame()
        gp = group.copy()

        # === Alpha #60 因子计算 ===
        
        # 1. 计算日内位置指标
        gp['intraday_position'] = ((gp['close'] - gp['low']) - (gp['high'] - gp['close'])) / (gp['high'] - gp['low'] + 1e-8)
        
        # 2. 乘以主动买入金额
        gp['position_taker_buy'] = gp['intraday_position'] * gp['taker_buy_quote']
        
        # 3. 计算过去ts_argmax_window天收盘价最大值的位置
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

        return gp[["date", "symbol", "position_taker_buy", "ts_argmax_close", "future_ret"]].dropna()

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
        
        # 第一部分：日内位置 × 主动买入金额的排名和缩放
        df_date['rank_position_taker_buy'] = df_date['position_taker_buy'].rank(pct=True)
        df_date['scale_position_taker_buy'] = (df_date['rank_position_taker_buy'] - 0.5) * 2  # 缩放到[-1, 1]
        
        # 第二部分：ts_argmax位置的排名和缩放
        df_date['rank_ts_argmax'] = df_date['ts_argmax_close'].rank(pct=True)
        df_date['scale_ts_argmax'] = (df_date['rank_ts_argmax'] - 0.5) * 2  # 缩放到[-1, 1]
        
        # 组合因子: (2 * scale(rank(position_taker_buy)) - scale(rank(ts_argmax))) 然后取负值
        df_date['alpha60_raw'] = -(2 * df_date['scale_position_taker_buy'] - df_date['scale_ts_argmax'])
        
        return df_date
    
    factor_df = all_data.groupby('date', group_keys=False).apply(combine_per_date)
    factor_df = factor_df.dropna(subset=['alpha60_raw'])
    
    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="alpha60_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="alpha60_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha60_intraday_position_ts_argmax_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # === 参数配置区域 ===
    
    # 标准参数配置
    TS_ARGMAX_WINDOW = 49  # ts_argmax计算窗口
    REBALANCE_PERIOD = 3   # 调仓周期
    
    # 执行标准配置
    print("=== 执行标准配置 ===")
    create_alpha60_factor(
        ts_argmax_window=TS_ARGMAX_WINDOW,
        rebalance_period=REBALANCE_PERIOD
    )
    
    # # === 可选参数配置 ===
    
    # # 快速交易配置 - 适应加密货币市场的快速节奏
    # print("\n=== 执行快速交易配置 ===")
    # create_alpha60_factor(
    #     ts_argmax_window=7,       # 更短的ts_argmax窗口
    #     rebalance_period=3        # 更短的调仓周期
    # )
    
    # # 中期配置 - 平衡敏感度和稳定性
    # print("\n=== 执行中期配置 ===")
    # create_alpha60_factor(
    #     ts_argmax_window=15,      # 中等ts_argmax窗口
    #     rebalance_period=5        # 中等调仓周期
    # )
    
    # # 长期趋势配置 - 适应长期趋势分析
    # print("\n=== 执行长期趋势配置 ===")
    # create_alpha60_factor(
    #     ts_argmax_window=20,      # 更长的ts_argmax窗口
    #     rebalance_period=7        # 更长的调仓周期
    # )
