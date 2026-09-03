import pandas as pd
import numpy as np

from util_factor import (
    load_kline_df,
    build_available_tokens_by_date_from_kline,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary, print_latest_date_inference
)

def create_return_taker_ratio_diff_factor(window=20, rebalance_period=10, top_n=50):
    """
    改进版：主动买入占比-收益差因子 (W-Cutting 逻辑变种)
    
    逻辑改进：
    1. 排序指标：使用 taker_buy_quote / quote_volume (主动买入占比)。
       - 相比绝对金额，占比能剥离行情冷热的影响，纯粹衡量多空力量对比。
    2. 合成方式：使用 Subtraction (做差)。
       - Factor = Sum(Ret_High_Ratio) - Sum(Ret_Low_Ratio)
    
    含义：
    - 因子值越大，说明该币种在“多头主动进攻”的日子里涨得好，且在“空头砸盘”的日子里跌得少（抗跌）。
    - 这是一个衡量“上涨质量”的因子。
    """
    print(f"开始构建主动买入占比-收益差因子")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, top_n={top_n}")

    # 加载数据
    df = load_kline_df()
    
    # 筛选前50
    available_tokens_by_date = build_available_tokens_by_date_from_kline(
        kline_df=df, top_n=top_n, ranking_method='quote_volume',
        rebalance_period=rebalance_period, lookback_buffer=20, strict_top_n=True
    )

    def compute_one(group: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if len(group) < window + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()
        
        # 1. 基础指标计算
        gp['ret'] = np.log(gp['close'] / gp['close'].shift(1))
        
        # 核心改进A: 使用占比而不是绝对值
        # taker_buy_ratio 越高，说明当日主动买入意愿越强
        if 'taker_buy_quote' in gp.columns and 'quote_volume' in gp.columns:
            gp['buy_ratio'] = gp['taker_buy_quote'] / (gp['quote_volume'] + 1e-8)
        else:
            return pd.DataFrame()

        # 预先计算未来收益（提高效率）
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")
        
        vals, dates, fwd = [], [], []
        
        # 提取 numpy array 加速
        ret_arr = gp['ret'].values
        ratio_arr = gp['buy_ratio'].values
        dates_arr = gp['date'].values
        future_ret_arr = gp['future_ret'].values
        
        # 2. 滚动窗口计算 (W-Cutting 风格)
        # 从 window 开始（而不是 window - 1），确保有完整的窗口数据
        for i in range(window, len(gp) - rebalance_period):
            # 获取窗口数据 [i - window : i]（不包含当前日）
            window_ret = ret_arr[i - window : i]
            window_ratio = ratio_arr[i - window : i]
            
            # 检查窗口数据完整性
            if np.isnan(window_ret).any() or np.isnan(window_ratio).any():
                continue
            
            # 按买入占比排序
            # argsort 从小到大排序
            sorted_idx = np.argsort(window_ratio)
            
            # 切割：取两头
            # 20天窗口，取头尾各5天或10天（这里取一半对半切，或者 Top/Bottom 5）
            # 研报中是取两头各10天(总共20天)，也就是全覆盖。我们也取各10天。
            n_split = window // 2
            
            low_ratio_idx = sorted_idx[:n_split]   # 买入占比最低的10天 (空头主导/散户抛售)
            high_ratio_idx = sorted_idx[n_split:]  # 买入占比最高的10天 (多头主导/大户吸筹)
            
            # 核心改进B: 使用差值 (Subtraction)
            # M_high: 多头主导日的累计涨幅
            # M_low:  空头主导日的累计涨幅
            m_high = np.sum(window_ret[high_ratio_idx])
            m_low = np.sum(window_ret[low_ratio_idx])
            
            # 因子 = 多头日涨幅 / 空头日涨幅
            # 避免除零和无效值
            if m_low == 0 or np.isnan(m_high) or np.isnan(m_low):
                continue
            
            factor_val = m_high - m_low
            
            # 检查因子值有效性
            if np.isnan(factor_val) or np.isinf(factor_val):
                continue
            
            vals.append(factor_val)
            dates.append(dates_arr[i])
            fwd.append(future_ret_arr[i])

        if not vals:
            return pd.DataFrame()

        res = pd.DataFrame({
            'date': dates, 
            'symbol': symbol, 
            'factor_raw': vals, 
            'future_ret': fwd
        })
        
        res = filter_group_by_availability(res, symbol, available_tokens_by_date)
        return res

    print("计算因子中...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    
    if not result_dfs:
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    # 去极值与标准化
    factor_df = winsorize_by_date(factor_df, col='factor_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='factor_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"return_taker_ratio_diff_{window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)
    latest_date = factor_df['date'].max()
    print_latest_date_inference(factor_df, latest_date, groups=5)
    
    return factor_df

if __name__ == "__main__":
    create_return_taker_ratio_diff_factor(window=10, rebalance_period=5)