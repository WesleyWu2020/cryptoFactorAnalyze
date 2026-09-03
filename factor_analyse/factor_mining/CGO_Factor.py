import pandas as pd
import numpy as np

from util_factor import (
    load_kline_df,
    load_available_tokens_from_index_cache,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary, print_latest_date_inference
)

def calculate_rp_recursive(vwap_arr, turnover_arr):
    """
    使用递归方式计算参考价格 RP (Reference Price)
    逻辑: RP_t = RP_{t-1} * (1 - V_{t-1}) + P_{t-1} * V_{t-1}
    
    解释：
    - RP代表"参考持仓成本"
    - 每天一部分仓位(V_{t-1})按昨日价格(P_{t-1})换手
    - 剩余仓位(1-V_{t-1})保持原成本(RP_{t-1})
    """
    T = len(vwap_arr)
    rp_arr = np.zeros(T)
    rp_arr[0] = vwap_arr[0]  # 初始值

    curr_rp = vwap_arr[0]
    for t in range(1, T):
        v_prev = turnover_arr[t-1]
        p_prev = vwap_arr[t-1]
        curr_rp = curr_rp * (1 - v_prev) + p_prev * v_prev
        rp_arr[t] = curr_rp
        
    return rp_arr

def calculate_turnover_volume_ratio(volume_series, window=60):
    """
    基于成交量占比的换手率估计（保守版本）
    
    逻辑：
    1. 计算滚动平均成交量作为基准
    2. 当日成交量相对基准的比率反映市场活跃度
    3. 使用tanh函数将比率映射到合理的换手率区间
    4. EMA平滑避免高频噪音
    
    参数：
    - volume_series: 成交量序列
    - window: 滚动窗口（默认60天）
    
    返回：
    - turnover: 估计的换手率（范围：0.01-0.25）
    
    映射逻辑：
    - volume_ratio = 1.0（成交量等于平均值）-> turnover ≈ 0.03（3%）
    - volume_ratio = 2.0（成交量是平均值2倍）-> turnover ≈ 0.11（11%）
    - volume_ratio = 0.5（成交量是平均值一半）-> turnover ≈ 0.01（1%）
    """
    # 计算滚动平均成交量作为基准
    avg_volume = volume_series.rolling(window=window, min_periods=1).mean()
    
    # 当日成交量 / 平均成交量的比率
    volume_ratio = volume_series / (avg_volume + 1e-8)
    
    # 映射到换手率（使用更保守的映射）
    base_rate = 0.03  # 基础换手率：3%
    sensitivity = 0.08  # 敏感度系数
    
    # 使用tanh函数进行平滑映射
    # tanh将(-∞, +∞)映射到(-1, 1)，更能处理极端值
    turnover = base_rate + sensitivity * np.tanh((volume_ratio - 1) / 2)
    
    # EMA平滑处理，避免高频波动
    # span=5 意味着大约用最近5天的数据进行指数加权平均
    turnover = turnover.ewm(span=5, adjust=False).mean()
    
    # 截断在合理范围：最低1%，最高25%
    turnover = turnover.clip(0.01, 0.25)
    
    return turnover

def create_cgo_factor(window=100, rebalance_period=1, top_n=50):
    """
    资本利得突出量因子 (CGO - Capital Gain Outstanding) —— 无未来函数版本
    - 定义: CGO = (收盘价 - 参考成本) / 参考成本
    - 参考成本通过递归方式计算，考虑换手率
    - 输出: 截面去极值 + 排名到[-1,1] 的 factor，与 future_ret
    
    策略：
    - 使用市值50等权指数的成分股列表（从index_cache_equal_weight_30_50.csv加载）
    - 对于每个日期，使用最近一次调仓日的成分股列表
    - 例如：计算12月17日的因子时，使用12月1日（最近一次调仓日）的成分股
    - 保留这些token的完整历史数据（包括不在成分股时的数据）
    """
    print(f"开始构建资本利得突出量因子 (CGO) —— 无未来函数版本")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, top_n={top_n}")

    # 1. 加载K线数据（包含所有可能进入前50的token的完整数据）
    df = load_kline_df()
    
    # 2. 从指数缓存文件加载市值50成分股列表
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

    # 3. 单symbol计算逻辑
    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # --- A. 计算成交均价 (VWAP) ---
        if 'quote_volume' in gp.columns and 'volume' in gp.columns:
            gp['vwap'] = np.where(gp['volume'] > 0, gp['quote_volume'] / gp['volume'], gp['close'])
        else:
            gp['vwap'] = gp['close']

        # --- B. 计算换手率 (完全严格复现你的 Snippet) ---
        
        # 【细节1】加1平滑 (严格复现)
        log_vol = np.log(gp['volume'] + 1)
        
        # 【细节2】滚动窗口的极值 (严格复现)
        roll_min = log_vol.rolling(window=window, min_periods=1).min()
        roll_max = log_vol.rolling(window=window, min_periods=1).max()
        
        # 【细节3】分母防零处理 (严格复现)
        denominator = roll_max - roll_min
        denominator = denominator.replace(0, 1) # 这里如果用Series操作，replace(0, 1)是安全的
        
        # 计算相对位置 position
        position = (log_vol - roll_min) / denominator
        
        # 【细节4】映射到合理的换手率区间 (严格复现)
        max_turnover_cap = 0.8
        estimated_turnover = position * max_turnover_cap
        
        # 【细节5】给予基础底噪 (严格复现：混合逻辑)
        # 这里的公式：base + estimated * (1 - base) 
        # 含义是：在 base 之上的剩余空间 (1-base) 里进行映射，比简单的加法更严谨
        base_turnover = 0.02
        gp['turnover'] = base_turnover + estimated_turnover * (1 - base_turnover)
        
        # 最终双重保险截断
        gp['turnover'] = gp['turnover'].clip(0, 0.99)

        # --- C. 计算参考价格 (RP) ---
        vwap_values = gp['vwap'].values
        turnover_values = gp['turnover'].values
        gp['rp'] = calculate_rp_recursive(vwap_values, turnover_values)

        # --- D. 计算 CGO 因子 ---
        # CGO = (收盘价 - 参考成本) / 参考成本
        # 注：分母加 1e-8 防止除零
        gp['cgo_raw'] = (gp['close'] - gp['rp']) / (gp['rp'] + 1e-8)

        # --- E. 后续处理 ---
        gp["future_ret"] = future_return(gp["close"], rebalance_period, method="log")
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "cgo_raw", "future_ret"]].dropna()

    print("计算 CGO 因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    
    if not result_dfs:
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 4. 去极值与标准化
    factor_df = winsorize_by_date(factor_df, col="cgo_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="cgo_raw", out_col="factor")

    # 5. 保存与输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"cgo_strict_log_{window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)
    latest_date = factor_df['date'].max()
    print_latest_date_inference(factor_df, latest_date, groups=5)
    
    return factor_df

if __name__ == "__main__":
    create_cgo_factor(window=10, rebalance_period=3)