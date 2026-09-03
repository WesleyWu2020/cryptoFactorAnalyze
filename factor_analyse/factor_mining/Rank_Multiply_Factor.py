# factor_analyse/factor_mining/Rank_Multiply_Factor.py
import pandas as pd
import numpy as np

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

def create_ic_weighted_rank_factor(window=20, rebalance_period=10, availability_lookback_days=90, ic_lookback_days=60):
    """
    IC等权Rank因子 —— 基于历史IC动态权重组合量稳因子（负向）和波动效率因子（正向）
    
    核心思路：
    1. 计算 Volume Stability 因子（负向：值越小越好）
    2. 计算 Volatility Efficiency 因子（正向：值越大越好）
    3. 对量稳因子取负值，使其变为正向
    4. 基于历史IC值动态分配两个因子的权重
    5. 加权组合后按日秩归一化到[-1,1]
    
    投资逻辑：
    - 量稳因子：负向，取负值后变为正向（稳定性越高越好）
    - 波动效率因子：正向（效率越高越好）
    - 使用IC等权动态调整，适应市场变化
    """
    print(f"开始构建 IC等权Rank因子...")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, ic_lookback_days={ic_lookback_days}")

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

        # === Volume Stability 因子计算（负向因子） ===
        
        # 计算每笔交易的平均成交量，作为换手率的代理指标
        gp['turnover'] = gp['quote_volume'] / (gp['trades_count'] + 1e-8)  # 防止除零
        
        # 计算N日均值
        gp['turnover_mean'] = gp['turnover'].rolling(window=window, min_periods=window).mean()
        
        # 计算N日标准差
        gp['turnover_std'] = gp['turnover'].rolling(window=window, min_periods=window).std()
        
        # 量稳因子：均值/标准差（负向：值越小越好）
        gp['volume_stability'] = gp['turnover_mean'] / (gp['turnover_std'] + 1e-8)  # 防止除零
        
        # === Volatility Efficiency 因子计算（正向因子） ===
        
        # 计算收益率
        gp['roc'] = gp['close'] / gp['close'].shift(window) - 1
        
        # 计算价格区间百分比
        gp['range_pct'] = ((gp['high'] - gp['low']) / (gp['close'].shift(1) + 1e-8)).rolling(
            window=window, min_periods=window
        ).mean()
        
        # 波动效率：收益率/价格区间（正向：值越大越好）
        gp['volatility_efficiency'] = gp['roc'] / (gp['range_pct'] + 1e-8)
        
        # 未来收益
        gp['future_ret'] = future_return(gp['close'], rebalance_period, method="pct")

        # 可用性过滤
        gp = filter_group_by_availability(gp, symbol, available_tokens_by_date)

        return gp[["date", "symbol", "volume_stability", "volatility_efficiency", "future_ret"]].dropna()

    print("计算两个基础因子并进行可用性过滤...")
    result_dfs = group_apply_with_progress(df, "symbol", compute_one)
    if not result_dfs:
        print("警告: 没有足够的数据计算因子")
        return pd.DataFrame()

    # 合并所有token的数据
    all_data = pd.concat(result_dfs, ignore_index=True)
    
    # 按日期分组计算IC权重和因子组合
    print("计算IC权重和因子组合...")
    
    def calculate_ic_weights_and_combine(df_date):
        """计算IC权重并组合因子"""
        df_date = df_date.copy()
        
        # 对两个因子分别进行截面rank排名到[0,1]
        df_date['volume_stability_rank'] = df_date['volume_stability'].rank(pct=True)
        df_date['volatility_efficiency_rank'] = df_date['volatility_efficiency'].rank(pct=True)
        
        # 量稳因子取负值（负向因子转为正向）
        df_date['volume_stability_rank_positive'] = 1 - df_date['volume_stability_rank']
        
        return df_date
    
    # 先计算rank排名
    all_data = all_data.groupby('date', group_keys=False).apply(calculate_ic_weights_and_combine)
    
    # 计算历史IC值（用于权重分配）
    def calculate_historical_ic(df_grouped, lookback_days):
        """计算历史IC值"""
        dates = sorted(df_grouped['date'].unique())
        ic_data = []
        
        for i in range(lookback_days, len(dates)):
            # 获取历史数据窗口
            start_idx = i - lookback_days
            end_idx = i
            historical_dates = dates[start_idx:end_idx]
            
            # 计算历史IC
            historical_data = df_grouped[df_grouped['date'].isin(historical_dates)]
            
            if len(historical_data) > 10:  # 确保有足够样本
                # 计算量稳因子的IC（负向因子，取负值后）
                ic_volstab = historical_data['volume_stability_rank_positive'].corr(historical_data['future_ret'])
                
                # 计算波动效率因子的IC（正向因子）
                ic_vol_eff = historical_data['volatility_efficiency_rank'].corr(historical_data['future_ret'])
                
                ic_data.append({
                    'date': dates[i],
                    'ic_volstab': ic_volstab if not np.isnan(ic_volstab) else 0.0,
                    'ic_vol_eff': ic_vol_eff if not np.isnan(ic_vol_eff) else 0.0
                })
        
        return pd.DataFrame(ic_data)
    
    print(f"计算历史IC值（回看{ic_lookback_days}天）...")
    ic_df = calculate_historical_ic(all_data, ic_lookback_days)
    
    if len(ic_df) == 0:
        print("警告: 无法计算IC值，使用等权重")
        volstab_weight, vol_eff_weight = 0.5, 0.5
    else:
        # 计算平均IC值
        avg_ic_volstab = ic_df['ic_volstab'].mean()
        avg_ic_vol_eff = ic_df['ic_vol_eff'].mean()
        
        # 使用IC绝对值作为权重依据
        volstab_weight = abs(avg_ic_volstab) if not np.isnan(avg_ic_volstab) else 0.5
        vol_eff_weight = abs(avg_ic_vol_eff) if not np.isnan(avg_ic_vol_eff) else 0.5
        
        # 归一化权重
        total_weight = volstab_weight + vol_eff_weight
        if total_weight > 0:
            volstab_weight /= total_weight
            vol_eff_weight /= total_weight
        else:
            volstab_weight, vol_eff_weight = 0.5, 0.5
        
        print(f"IC权重分配: 量稳因子={volstab_weight:.3f}, 波动效率因子={vol_eff_weight:.3f}")
        print(f"平均IC值: 量稳因子={avg_ic_volstab:.4f}, 波动效率因子={avg_ic_vol_eff:.4f}")
    
    # 使用IC权重组合因子
    def combine_factors_with_ic_weights(df_date):
        """使用IC权重组合因子"""
        df_date = df_date.copy()
        
        # 加权组合：量稳因子（负向转正向）+ 波动效率因子（正向）
        df_date['ic_weighted_rank'] = (
            volstab_weight * df_date['volume_stability_rank_positive'] + 
            vol_eff_weight * df_date['volatility_efficiency_rank']
        )
        
        return df_date
    
    # 组合因子
    all_data = all_data.groupby('date', group_keys=False).apply(combine_factors_with_ic_weights)
    
    # 去极值与按日秩归一化到[-1,1]
    factor_df = all_data[["date", "symbol", "ic_weighted_rank", "future_ret"]].copy()
    factor_df = factor_df.rename(columns={"ic_weighted_rank": "rank_multiply_raw"})
    factor_df = factor_df.dropna(subset=['rank_multiply_raw'])
    
    factor_df = winsorize_by_date(factor_df, col="rank_multiply_raw", n_std=3.0)
    factor_df = rank_to_unit_by_date(factor_df, col="rank_multiply_raw", out_col="factor")

    # 输出
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret"]]
    out_path = save_factor_df(factor_df, file_prefix=f"ic_weighted_rank_{window}d_rebalance{rebalance_period}d_")

    print_factor_summary(factor_df, out_path)
    
    # 显示IC权重信息
    print(f"\n最终IC权重分配:")
    print(f"  量稳因子（负向转正向）: {volstab_weight:.3f}")
    print(f"  波动效率因子（正向）: {vol_eff_weight:.3f}")
    
    return factor_df

if __name__ == "__main__":
    create_ic_weighted_rank_factor(window=20, rebalance_period=10, ic_lookback_days=30)
