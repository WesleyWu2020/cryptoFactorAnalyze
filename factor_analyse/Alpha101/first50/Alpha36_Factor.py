# factor_analyse/Alpha101/Alpha36_Factor.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加 factor_mining 目录到 Python 路径
factor_mining_dir = os.path.join(current_dir, '..', 'factor_mining')
import sys
sys.path.insert(0, factor_mining_dir)

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample
)

def create_alpha36_factor(corr_window=15, ts_rank_window=5, delay_lag=6, vwap_adv_corr=6, 
                         long_ma_window=200, rebalance_period=10, availability_lookback_days=90):
    """
    Alpha 36 因子 (币圈7×24h优化版)
    严格避免未来函数，只使用上一期已出现的token
    
    原始定义:
    (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + 
        (0.7 * rank((open - close)))) + 
        (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + 
        rank(abs(correlation(vwap, adv20, 6)))) + 
        (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open))))
    
    权重分配: 2.21, 0.7, 0.73, 1.0, 0.6
    
    币圈7×24h特殊设计:
    1. 相对价格归一化：适应币圈巨大的价格差异
    2. 成交量标准化：处理不同币种的成交量量级差异
    3. 多维度信号：结合日内、短期、中期和长期信号
    4. 🔥 关键：结合历史市值排名，确保只使用上一期已出现的token
    
    该因子适用于多因素选股，平均持有期约5-15天
    """
    print(f"开始构建 Alpha 36 因子 (币圈7×24h优化版)...")
    print(f"参数: corr_window={corr_window}, ts_rank_window={ts_rank_window}, delay_lag={delay_lag}")
    print(f"vwap_adv_corr={vwap_adv_corr}, long_ma_window={long_ma_window}, rebalance_period={rebalance_period}")

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
        if len(group) < max(corr_window, ts_rank_window, delay_lag, vwap_adv_corr, long_ma_window) + rebalance_period + 2:
            return pd.DataFrame()
        gp = group.copy()

        # === 增强的7×24h币圈价格归一化处理 ===
        
        # 1. 多层次价格归一化（适应币圈巨大价格差异）
        # 短期基准：20日移动平均
        gp['price_base_short'] = gp['close'].rolling(window=20, min_periods=1).mean()
        # 中期基准：60日移动平均
        gp['price_base_medium'] = gp['close'].rolling(window=60, min_periods=1).mean()
        # 长期基准：120日移动平均
        gp['price_base_long'] = gp['close'].rolling(window=120, min_periods=1).mean()
        
        # 2. 自适应价格基准选择
        # 根据价格波动性选择最合适的基准
        gp['price_volatility'] = gp['close'].rolling(window=20).std() / gp['close'].rolling(window=20).mean()
        
        # 高波动币种使用短期基准，低波动币种使用长期基准
        def select_price_base(row):
            if pd.isna(row['price_volatility']):
                return row['price_base_medium']
            elif row['price_volatility'] > 0.1:  # 高波动
                return row['price_base_short']
            elif row['price_volatility'] < 0.05:  # 低波动
                return row['price_base_long']
            else:  # 中等波动
                return row['price_base_medium']
        
        gp['price_base'] = gp.apply(select_price_base, axis=1)
        
        # 3. 增强的价格归一化
        gp['normalized_open'] = gp['open'] / (gp['price_base'] + 1e-8)
        gp['normalized_close'] = gp['close'] / (gp['price_base'] + 1e-8)
        gp['vwap'] = (gp['high'] + gp['low'] + gp['close']) / 3
        gp['normalized_vwap'] = gp['vwap'] / (gp['price_base'] + 1e-8)
        
        # 4. 成交量标准化增强（处理不同币种成交量量级差异）
        # 使用相对成交量而非绝对成交量
        gp['volume_ma_short'] = gp['volume'].rolling(window=10, min_periods=1).mean()
        gp['volume_ma_medium'] = gp['volume'].rolling(window=30, min_periods=1).mean()
        gp['volume_ma_long'] = gp['volume'].rolling(window=60, min_periods=1).mean()
        
        # 自适应成交量基准
        def select_volume_base(row):
            if pd.isna(row['price_volatility']):
                return row['volume_ma_medium']
            elif row['price_volatility'] > 0.1:  # 高波动用短期
                return row['volume_ma_short']
            elif row['price_volatility'] < 0.05:  # 低波动用长期
                return row['volume_ma_long']
            else:  # 中等波动
                return row['volume_ma_medium']
        
        gp['volume_base'] = gp.apply(select_volume_base, axis=1)
        gp['normalized_volume'] = gp['volume'] / (gp['volume_base'] + 1e-8)
        
        # 5. 价格变化率标准化（避免绝对价格差异影响）
        # 使用对数收益率而非绝对价格差
        gp['log_returns'] = np.log(gp['normalized_close'] / gp['normalized_close'].shift(1))
        gp['log_intraday_ret'] = np.log(gp['normalized_close'] / gp['normalized_open'])
        
        # 6. 计算因子组件（使用标准化后的数据）
        
        # 组件1: 日内模式与交易量关系 (权重: 2.21)
        # 使用对数收益率替代绝对价格差
        gp['intraday_ret'] = gp['log_intraday_ret']
        gp['volume_delay'] = gp['normalized_volume'].shift(1)
        gp['corr_intraday_volume'] = gp['intraday_ret'].rolling(window=corr_window).corr(gp['volume_delay'])
        
        # 组件2: 日内反转 (权重: 0.7)
        # 使用对数收益率
        gp['intraday_reversal'] = -gp['log_intraday_ret']  # 反转信号
        
        # 组件3: 历史收益影响 (权重: 0.73)
        # 使用对数收益率
        gp['returns'] = gp['log_returns']
        gp['neg_returns_delay'] = (-1 * gp['returns']).shift(delay_lag)
        
        # 计算Ts_Rank (时序排名)
        def ts_rank(series, window):
            if len(series) < window:
                return np.nan
            recent_data = series.iloc[-window:]
            rank_pct = recent_data.rank(pct=True).iloc[-1]
            return rank_pct
        
        gp['ts_rank_neg_returns'] = gp['neg_returns_delay'].rolling(window=ts_rank_window).apply(
            lambda x: ts_rank(x, ts_rank_window), raw=False
        )
        
        # 组件4: 流动性分析 (权重: 1.0)
        gp['adv20'] = gp['normalized_volume'].rolling(window=20).mean()
        gp['corr_vwap_adv20'] = gp['normalized_vwap'].rolling(window=vwap_adv_corr).corr(gp['adv20'])
        gp['abs_corr_vwap_adv20'] = np.abs(gp['corr_vwap_adv20'])
        
        # 组件5: 长期趋势与日内结合 (权重: 0.6)
        # 使用相对价格变化
        gp['long_ma'] = gp['normalized_close'].rolling(window=long_ma_window).mean()
        gp['trend_intraday'] = (gp['long_ma'] - gp['normalized_open']) * (gp['normalized_close'] - gp['normalized_open'])
        
        # 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="log")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[['date', 'symbol', 'corr_intraday_volume', 'intraday_reversal', 
                  'ts_rank_neg_returns', 'abs_corr_vwap_adv20', 'trend_intraday', 'future_ret']].dropna()

    print("计算 Alpha 36 因子基础数据（考虑token可用性）...")
    result_dfs = group_apply_with_progress(df, 'symbol', compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    factor_df = pd.concat(result_dfs, ignore_index=True)

    print("计算横截面排名和因子值...")
    
    # 按日期计算Alpha 36因子
    def calculate_alpha36_by_date(df_date):
        df_date = df_date.copy()
        
        # 对各个组件进行排名
        df_date['corr_intraday_volume_rank'] = df_date['corr_intraday_volume'].rank(pct=True)
        df_date['intraday_reversal_rank'] = df_date['intraday_reversal'].rank(pct=True)
        df_date['ts_rank_neg_returns_rank'] = df_date['ts_rank_neg_returns'].rank(pct=True)
        df_date['abs_corr_vwap_adv20_rank'] = df_date['abs_corr_vwap_adv20'].rank(pct=True)
        df_date['trend_intraday_rank'] = df_date['trend_intraday'].rank(pct=True)
        
        # 按权重组合因子
        df_date['alpha36_raw'] = (
            2.21 * df_date['corr_intraday_volume_rank'] +
            0.7 * df_date['intraday_reversal_rank'] +
            0.73 * df_date['ts_rank_neg_returns_rank'] +
            1.0 * df_date['abs_corr_vwap_adv20_rank'] +
            0.6 * df_date['trend_intraday_rank']
        )
        
        return df_date

    factor_df = factor_df.groupby('date', group_keys=False).apply(calculate_alpha36_by_date)
    factor_df = factor_df.dropna(subset=['alpha36_raw'])

    # 去极值与归一化
    factor_df = winsorize_by_date(factor_df, col='alpha36_raw', n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col='alpha36_raw', out_col='factor')

    # 输出
    factor_df = factor_df.rename(columns={'symbol': 'instrument'})[['date', 'instrument', 'factor', 'future_ret']]
    out_path = save_factor_df(factor_df, file_prefix=f"alpha36_multi_factor_")

    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n因子统计信息:")
    print(factor_df['factor'].describe())
    return factor_df

if __name__ == "__main__":
    # === Alpha 36 因子测试 (币圈7×24h优化版) ===
    
    # 策略1: 标准参数 (推荐)
    create_alpha36_factor(corr_window=10, ts_rank_window=5, delay_lag=5, 
                         vwap_adv_corr=5, long_ma_window=50, rebalance_period=3)
    
    # 策略2: 缩短窗口 (适应币圈高频特性)
    # create_alpha36_factor(corr_window=10, ts_rank_window=3, delay_lag=3, 
    #                      vwap_adv_corr=5, long_ma_window=100, rebalance_period=5)
    
    # 策略3: 极短窗口 (适应7×24h快速变化)
    # create_alpha36_factor(corr_window=7, ts_rank_window=3, delay_lag=2, 
    #                      vwap_adv_corr=3, long_ma_window=50, rebalance_period=3)