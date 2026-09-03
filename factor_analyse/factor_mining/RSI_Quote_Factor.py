# factor_analyse/factor_mining/RSI_Quote_Factor.py
import pandas as pd
import numpy as np

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

def create_rsi_quote_factor(window=14, rebalance_period=5, availability_lookback_days=90):
    """
    RSI 截面排序因子 (RSI Cross-Sectional Rank Factor) —— 无未来函数版本
    
    思路:
    1. 计算传统RSI: RSI = 100 - (100 / (1 + RS))
       - RS = 平均上涨幅度 / 平均下跌幅度
       - 使用指数移动平均计算，避免未来函数
    2. 横截面排名: 对每个日期的RSI值进行排名
    3. 标准化: 将排名转换为[-1, 1]区间
    
    参数:
    - window: RSI计算窗口，默认14天
    - rebalance_period: 调仓周期，默认5天
    
    - 平均持有期: 5-10天
    """
    print(f"开始构建 RSI 截面排序因子...")
    print(f"参数: window={window}, rebalance_period={rebalance_period}")

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
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === RSI 计算 ===
        
        # 1. 计算价格变化
        gp['price_change'] = gp['close'].diff()
        
        # 2. 分离上涨和下跌
        gp['gain'] = np.where(gp['price_change'] > 0, gp['price_change'], 0)
        gp['loss'] = np.where(gp['price_change'] < 0, -gp['price_change'], 0)
        
        # 3. 计算指数移动平均（避免未来函数）
        # 使用简单的滚动平均作为近似
        gp['avg_gain'] = gp['gain'].rolling(window=window, min_periods=window).mean()
        gp['avg_loss'] = gp['loss'].rolling(window=window, min_periods=window).mean()
        
        # 4. 计算RS和RSI
        gp['rs'] = gp['avg_gain'] / (gp['avg_loss'] + 1e-8)  # 避免除零
        gp['rsi'] = 100 - (100 / (1 + gp['rs']))
        
        # 5. 处理极端值
        gp['rsi'] = gp['rsi'].clip(0, 100)  # RSI应该在[0, 100]范围内
        
        # 6. 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "rsi", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="rsi", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="rsi", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"rsi_cross_sectional_{window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 标准参数
    create_rsi_quote_factor(window=14, rebalance_period=5)
    
    # 可选：更短的窗口，适应加密货币市场的更快节奏
    # create_rsi_quote_factor(window=7, rebalance_period=3)
    
    # 可选：更长的窗口，适应长期趋势
    # create_rsi_quote_factor(window=21, rebalance_period=10)
