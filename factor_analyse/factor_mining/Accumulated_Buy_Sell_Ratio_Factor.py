# factor_analyse/factor_mining/Accumulated_Buy_Sell_Ratio_Factor.py
import pandas as pd
import numpy as np

from util_factor import (
    load_kline_df,
    load_available_tokens_from_index_cache,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary, print_latest_date_inference
)

def create_accumulated_buy_sell_ratio_factor(window=20, rebalance_period=3, top_n=50):
    """
    累积主买/累积主卖比率因子 —— 无未来函数版本
    
    定义：
    - 主动买盘 = taker_buy_quote
    - 主动卖盘 = quote_volume - taker_buy_quote
    
    逻辑：
    1. 统计过去window个交易日
    2. 收盘为涨的日期：累积主动买盘
    3. 收盘为负的日期：累积主动卖盘
    4. 因子 = 累积主买 / 累积主卖
    
    投资逻辑：
    - 当累积主动买盘远大于累积主动卖盘时，表明上涨趋势得到资金支持
    - 当累积主动卖盘远大于累积主动买盘时，表明下跌趋势得到资金确认
    - 比值越高，上涨趋势越强；比值越低，下跌趋势越强
    
    策略：
    - 使用市值50等权指数的成分股列表（从index_cache_equal_weight_30_50.csv加载）
    - 对于每个日期，使用最近一次调仓日的成分股列表
    - 例如：计算12月17日的因子时，使用12月1日（最近一次调仓日）的成分股
    - 保留这些token的完整历史数据（包括不在成分股时的数据）
    """
    print(f"开始构建累积主买/累积主卖比率因子 —— 无未来函数版本")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, top_n={top_n}")

    # K线数据（包含所有可能进入前50的token的完整数据）
    df = load_kline_df()
    
    # 从指数缓存文件加载市值50成分股列表
    print("🔍 从指数缓存文件加载市值50成分股列表...")
    all_dates = sorted(df['date'].unique())
    available_tokens_by_date = load_available_tokens_from_index_cache(
        cache_file='index_cache_equal_weight_30_50.csv',  # 指定使用市值50等权指数文件
        top_n=top_n,
        all_dates=all_dates  # 传入所有日期，将调仓日的成分股扩展到所有交易日
    )
    
    # 如果缓存文件不存在或为空，报错
    if not available_tokens_by_date:
        raise FileNotFoundError("无法从指数缓存文件加载成分股列表，请确保 index_cache_equal_weight_30_50.csv 文件存在")
    
    print("🔍 生成各日期可用token列表（基于市值50等权指数）...")
    print_availability_sample(available_tokens_by_date, n=3)

    # 单symbol计算
    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy().reset_index(drop=True)

        # 计算日收益率
        gp["daily_return"] = gp["close"] / gp["close"].shift(1) - 1
        
        # 定义主动买盘和主动卖盘
        gp["taker_sell_quote"] = gp["quote_volume"] - gp["taker_buy_quote"]  # 主动卖盘
        
        # 🔥 向量化优化：使用条件筛选和rolling sum替代手动循环
        # 对于收盘为涨的日期，累积主动买盘；对于收盘为负的日期，累积主动卖盘
        # 使用条件乘法：True时保留原值，False时为0，然后rolling sum
        buy_conditional = (gp["daily_return"] > 0) * gp["taker_buy_quote"]
        sell_conditional = (gp["daily_return"] < 0) * gp["taker_sell_quote"]
        
        # 使用rolling sum计算窗口内的累积值
        gp["accumulated_buy"] = buy_conditional.rolling(window=window, min_periods=window).sum()
        gp["accumulated_sell"] = sell_conditional.rolling(window=window, min_periods=window).sum()
        
        # 计算因子：累积主买 / 累积主卖
        gp["abs_ratio"] = gp["accumulated_buy"] / (gp["accumulated_sell"] + 1e-8)
        
        # 未来收益（保持与其他因子一致：对数收益率）
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "abs_ratio", "future_ret"]].dropna()

    print("计算因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与按日秩归一化到[-1,1]
    factor_df = winsorize_by_date(factor_df, col="abs_ratio", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="abs_ratio", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"accumulated_buy_sell_ratio_{window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)
    
    # 添加最新日期推理打印
    latest_date = factor_df['date'].max()
    print_latest_date_inference(factor_df, latest_date, groups=5)
    
    return factor_df

if __name__ == "__main__":
    create_accumulated_buy_sell_ratio_factor(window=30, rebalance_period=5)
