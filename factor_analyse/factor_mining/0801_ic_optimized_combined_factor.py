import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_ic_optimized_combined_factor(window=20, rebalance_period=5, lookback_periods=60):
    """
    创建基于IC（信息系数）优化的组合因子
    
    核心思路：
    1. 计算每个因子与未来收益的IC值
    2. 基于历史IC的均值和稳定性进行权重分配
    3. 使用IC_IR（IC均值/IC标准差）作为权重依据
    4. 动态调整权重，适应市场变化
    """
    print(f"开始构建IC优化组合因子...")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, lookback_periods={lookback_periods}")

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))  # 回到Crypto目录

    # 获取最新的因子数据文件
    factor_data_dir = os.path.join(base_dir, "data", "factor_data")
    
    # 自动获取最新的因子文件
    def get_latest_factor_file(prefix):
        """获取指定前缀的最新因子文件"""
        files = [f for f in os.listdir(factor_data_dir) if f.startswith(prefix)]
        if not files:
            raise FileNotFoundError(f"未找到以 {prefix} 开头的因子文件")
        return sorted(files)[-1]  # 返回最新的文件
    
    # 因子文件名（自动获取最新文件）
    factor_files = {
        "volume_stability_20d": get_latest_factor_file("volume_stability_20d_rebalance10d_"),
        "volatility_efficiency_20d": get_latest_factor_file("volatility_efficiency_20d_rebalance10d_"),
    }
    
    print("使用的因子文件:")
    for factor_name, filename in factor_files.items():
        print(f"  {factor_name}: {filename}")

    print("加载因子数据...")
    dfs = []
    for factor, filename in factor_files.items():
        path = os.path.join(factor_data_dir, filename)
        try:
            df = pd.read_csv(path)
            # 只选择有效的因子值（非空）
            df = df.dropna(subset=['factor'])
            df = df.rename(columns={'factor': factor})
            dfs.append(df[['date', 'instrument', factor]])
            print(f"✅ 成功加载因子: {factor}, 数据量: {len(df)}")
        except Exception as e:
            print(f"❌ 加载因子失败: {factor}, 错误: {e}")
            continue

    if len(dfs) < 2:
        print("❌ 可用因子数量不足，无法进行组合")
        return None

    # 合并因子
    from functools import reduce
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['date', 'instrument'], how='inner'), dfs)
    
    print(f"合并后数据形状: {df_merged.shape}")
    print(f"可用因子: {[f for f in factor_files.keys() if f in df_merged.columns]}")

    # 获取可用因子列表
    available_factors = [f for f in factor_files.keys() if f in df_merged.columns]

    # 对每个因子进行排序归一化到-1到1之间
    def normalize_factor_by_date(df, factor_col):
        """对因子使用排序归一化到-1到1之间"""
        rank_pct = df[factor_col].rank(pct=True)
        df[factor_col] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("对单个因子进行排序归一化...")
    for factor in available_factors:
        df_merged = df_merged.groupby('date').apply(
            lambda x: normalize_factor_by_date(x, factor)
        ).reset_index(drop=True)

    # 修复：加载K线数据获取价格信息
    print("加载K线数据用于计算未来收益...")
    kline_dir = os.path.join(base_dir, "data", "kline_data")
    kline_files = [f for f in os.listdir(kline_dir) if f.startswith("binance_daily_klines_")]
    if not kline_files:
        raise FileNotFoundError("未找到K线数据文件")
    
    kline_path = os.path.join(kline_dir, sorted(kline_files)[-1])
    kline_df = pd.read_csv(kline_path)
    kline_df['date'] = pd.to_datetime(kline_df['date'])
    kline_df = kline_df[['date', 'symbol', 'close']].copy()
    kline_df = kline_df.rename(columns={'symbol': 'instrument'})

    # 计算每个因子的IC值 - 修复未来函数问题
    def calculate_daily_ic_simplified(df, factors, kline_data):
        """
        计算每个因子每日的IC值 - 使用真实价格计算未来收益
        """
        ic_data = []
        dates = sorted(df['date'].unique())
        
        for i in range(len(dates) - rebalance_period):
            current_date = dates[i]
            future_date = dates[i + rebalance_period]
            
            current_data = df[df['date'] == current_date].copy()
            future_data = kline_data[kline_data['date'] == future_date].copy()
            
            # 修复：正确合并当前因子数据和未来价格数据
            # 当前数据包含因子值，未来数据包含价格
            merged_data = pd.merge(
                current_data, 
                future_data[['instrument', 'close']], 
                on='instrument', 
                how='inner'
            )
            
            # 重命名未来价格列，避免与当前价格混淆
            merged_data = merged_data.rename(columns={'close': 'close_future'})
            
            # 现在需要获取当前日期的价格数据
            current_prices = kline_data[kline_data['date'] == current_date][['instrument', 'close']]
            merged_data = pd.merge(merged_data, current_prices, on='instrument', how='inner')
            
            if len(merged_data) > 10:  # 确保有足够样本
                ic_row = {'date': current_date}
                
                for factor in factors:
                    # 现在应该有 close 和 close_future 两列
                    if 'close' in merged_data.columns and 'close_future' in merged_data.columns:
                        # 使用真实的未来价格收益
                        future_return_real = (merged_data['close_future'] - merged_data['close']) / (merged_data['close'] + 1e-8)
                        
                        # 计算因子值与真实未来收益的相关系数
                        ic = merged_data[factor].corr(future_return_real)
                        ic_row[factor] = ic if not np.isnan(ic) else 0
                    else:
                        print(f"警告: 日期 {current_date} 缺少价格列，可用列: {merged_data.columns.tolist()}")
                        ic_row[factor] = 0
                
                ic_data.append(ic_row)
        
        return pd.DataFrame(ic_data)

    print("计算因子IC值...")
    ic_data = calculate_daily_ic_simplified(df_merged, available_factors, kline_df)
    
    if len(ic_data) == 0:
        print("❌ 无法计算IC值，请检查数据")
        return None
    
    print("IC值统计:")
    print(ic_data[available_factors].describe())

    # 基于IC的权重计算
    def calculate_ic_based_weights(ic_data, factors, lookback_periods):
        """
        基于历史IC计算权重 - 使用IC_IR作为权重依据
        """
        if len(ic_data) < lookback_periods:
            # 数据不足时使用等权重
            n_factors = len(factors)
            return np.array([1/n_factors] * n_factors)
        
        # 使用最近的IC数据
        recent_ic = ic_data.tail(lookback_periods)
        
        # 计算每个因子的IC_IR
        factor_scores = {}
        for factor in factors:
            ic_mean = recent_ic[factor].mean()
            ic_std = recent_ic[factor].std()
            
            # IC_IR = IC均值 / IC标准差
            if ic_std > 0:
                ic_ir = ic_mean / ic_std
            else:
                ic_ir = ic_mean
            
            # 使用IC_IR的绝对值作为权重依据，确保正数
            factor_scores[factor] = max(abs(ic_ir), 0.01)
        
        # 将分数转换为权重
        total_score = sum(factor_scores.values())
        if total_score > 0:
            weights = np.array([factor_scores[factor] / total_score for factor in factors])
        else:
            n_factors = len(factors)
            weights = np.array([1/n_factors] * n_factors)
        
        return weights

    # 计算动态权重并应用
    print("开始IC优化权重计算...")
    
    # 为每个日期计算优化权重
    unique_dates = sorted(df_merged['date'].unique())
    optimized_factors = []
    
    # 存储权重历史用于分析
    weight_history = []
    
    # 权重应用时确保无未来函数
    for i, current_date in enumerate(unique_dates):
        # 获取当前日期的数据
        current_data = df_merged[df_merged['date'] == current_date].copy()
        
        if i < lookback_periods:
            # 前期数据不足时使用等权重
            weights = np.array([1/len(available_factors)] * len(available_factors))
        else:
            # 修复：只使用当前日期之前的IC数据计算权重
            historical_ic = ic_data[ic_data['date'] < current_date]
            weights = calculate_ic_based_weights(historical_ic, available_factors, lookback_periods)
        
        # 记录权重
        weight_record = {'date': current_date}
        for j, factor in enumerate(available_factors):
            weight_record[factor] = weights[j]
        weight_history.append(weight_record)
        
        # 应用权重计算组合因子
        weighted_factor = np.zeros(len(current_data))
        for j, factor in enumerate(available_factors):
            weighted_factor += weights[j] * current_data[factor].values
        
        current_data['ic_optimized_factor'] = weighted_factor
        optimized_factors.append(current_data[['date', 'instrument', 'ic_optimized_factor']])
        
        if i % 50 == 0:
            print(f"已处理 {i+1}/{len(unique_dates)} 个日期")
            print(f"当前权重: {dict(zip(available_factors, weights))}")

    # 合并所有结果
    factor_df = pd.concat(optimized_factors, ignore_index=True)
    
    # 对最终组合因子进行排序归一化到-1到1之间
    def normalize_final_factor_by_date(df):
        """对组合因子使用排序归一化到-1到1之间"""
        rank_pct = df['ic_optimized_factor'].rank(pct=True)
        df['ic_optimized_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("对组合因子进行排序归一化...")
    factor_df = factor_df.groupby('date').apply(normalize_final_factor_by_date).reset_index(drop=True)
    
    factor_df = factor_df.rename(columns={'ic_optimized_factor': 'factor'})

    # 输出目录
    output_dir = factor_data_dir
    os.makedirs(output_dir, exist_ok=True)

    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"ic_optimized_factor_{window}d_rebalance{rebalance_period}d_{today}.csv")
    factor_df.to_csv(output_path, index=False)

    print(f"✅ IC优化组合因子数据已保存至: {output_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n因子统计信息:")
    print(factor_df['factor'].describe())

    # 验证最终组合因子的归一化结果
    print(f"\n最终组合因子范围: [{factor_df['factor'].min():.6f}, {factor_df['factor'].max():.6f}]")

    # 分析权重变化
    weight_df = pd.DataFrame(weight_history)
    print(f"\n权重变化统计:")
    for factor in available_factors:
        if factor in weight_df.columns:
            print(f"{factor}: 平均权重={weight_df[factor].mean():.4f}, 标准差={weight_df[factor].std():.4f}")

    # 显示最终权重信息
    if len(ic_data) >= lookback_periods:
        final_ic = ic_data.tail(lookback_periods)
        final_weights = calculate_ic_based_weights(final_ic, available_factors, lookback_periods)
        print(f"\n最终优化权重:")
        for factor, weight in zip(available_factors, final_weights):
            print(f"{factor}: {weight:.4f}")
        
        # 显示最终IC_IR
        print(f"\n最终各因子IC_IR:")
        for factor in available_factors:
            ic_mean = final_ic[factor].mean()
            ic_std = final_ic[factor].std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else ic_mean
            print(f"{factor}: IC={ic_mean:.4f}, IC_IR={ic_ir:.4f}")

    print("\n数据预览:")
    print(factor_df.head())

    return factor_df

if __name__ == "__main__":
    create_ic_optimized_combined_factor(window=20, rebalance_period=10, lookback_periods=60) 