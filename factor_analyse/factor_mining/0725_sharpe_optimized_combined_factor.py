import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def create_sharpe_optimized_factor(window=20, rebalance_period=1, lookback_periods=60):
    """
    创建基于夏普比率优化的组合因子
    """
    print(f"开始构建夏普比率优化组合因子...")
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
        "alpha5_relative_price_vwap20d": get_latest_factor_file("alpha5_relative_price_vwap20d"),
        "volume_stability_20d": get_latest_factor_file("volume_stability_20d"),
        "volatility_efficiency_20d": get_latest_factor_file("volatility_efficiency_20d"),
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

    # 计算因子收益率（作为优化的基础）
    def calculate_factor_returns(df, factors, periods=1):
        """
        计算因子的"收益率"（这里用因子值的变化率作为代理）
        """
        factor_returns = pd.DataFrame()
        factor_returns['date'] = df['date'].unique()[periods:]
        
        for factor in factors:
            returns = []
            dates = df['date'].unique()
            
            for i in range(periods, len(dates)):
                current_date = dates[i]
                prev_date = dates[i-periods]
                
                current_factor = df[df['date'] == current_date][factor].mean()
                prev_factor = df[df['date'] == prev_date][factor].mean()
                
                if prev_factor != 0:
                    factor_return = (current_factor - prev_factor) / prev_factor
                else:
                    factor_return = 0
                
                returns.append(factor_return)
            
            factor_returns[factor] = returns
        
        return factor_returns

    print("计算因子收益率...")
    factor_returns = calculate_factor_returns(df_merged, available_factors, periods=rebalance_period)
    
    # 夏普比率优化函数
    def sharpe_objective(weights, mean_returns, cov_matrix):
        """
        夏普比率优化目标函数（最小化负夏普比率）
        """
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if portfolio_volatility == 0:
            return -np.inf
        
        sharpe_ratio = portfolio_return / portfolio_volatility
        return -sharpe_ratio  # 负值用于最小化

    def optimize_weights(factor_returns_data, lookback_periods):
        """
        使用滚动窗口优化权重
        """
        if len(factor_returns_data) < lookback_periods:
            # 如果数据不足，使用等权重
            n_factors = len(available_factors)
            return np.array([1/n_factors] * n_factors)
        
        # 使用最近lookback_periods期的数据
        recent_data = factor_returns_data.tail(lookback_periods)
        
        # 计算均值和协方差矩阵
        mean_returns = recent_data[available_factors].mean().values
        cov_matrix = recent_data[available_factors].cov().values
        
        # 检查协方差矩阵是否有效
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            n_factors = len(available_factors)
            return np.array([1/n_factors] * n_factors)
        
        # 优化约束条件
        n_factors = len(available_factors)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_factors))
        
        # 初始权重（等权）
        initial_weights = np.array([1/n_factors] * n_factors)
        
        try:
            # 执行优化
            result = minimize(
                sharpe_objective,
                initial_weights,
                args=(mean_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                print(f"优化失败，使用等权重: {result.message}")
                return initial_weights
                
        except Exception as e:
            print(f"优化过程出错，使用等权重: {e}")
            return initial_weights

    # 计算动态权重并应用
    print("开始夏普比率优化...")
    
    # 为每个日期计算优化权重
    unique_dates = sorted(df_merged['date'].unique())
    optimized_factors = []
    
    for i, current_date in enumerate(unique_dates):
        # 获取当前日期的数据
        current_data = df_merged[df_merged['date'] == current_date].copy()
        
        if i < lookback_periods:
            # 前期数据不足时使用等权重
            weights = np.array([1/len(available_factors)] * len(available_factors))
        else:
            # 使用历史数据优化权重
            historical_returns = factor_returns.iloc[:i]
            weights = optimize_weights(historical_returns, lookback_periods)
        
        # 应用权重计算组合因子
        weighted_factor = np.zeros(len(current_data))
        for j, factor in enumerate(available_factors):
            weighted_factor += weights[j] * current_data[factor].values
        
        current_data['sharpe_optimized_factor'] = weighted_factor
        optimized_factors.append(current_data[['date', 'instrument', 'sharpe_optimized_factor']])
        
        if i % 50 == 0:
            print(f"已处理 {i+1}/{len(unique_dates)} 个日期")
            print(f"当前权重: {dict(zip(available_factors, weights))}")

    # 合并所有结果
    factor_df = pd.concat(optimized_factors, ignore_index=True)
    
    # 对最终组合因子进行排序归一化到-1到1之间
    def normalize_final_factor_by_date(df):
        """对组合因子使用排序归一化到-1到1之间"""
        rank_pct = df['sharpe_optimized_factor'].rank(pct=True)
        df['sharpe_optimized_factor'] = 2 * (rank_pct - 0.5)  # 将[0,1]映射到[-1,1]
        return df
    
    print("对组合因子进行排序归一化...")
    factor_df = factor_df.groupby('date').apply(normalize_final_factor_by_date).reset_index(drop=True)
    
    factor_df = factor_df.rename(columns={'sharpe_optimized_factor': 'factor'})

    # 输出目录
    output_dir = factor_data_dir
    os.makedirs(output_dir, exist_ok=True)

    # 保存因子数据
    today = datetime.now().strftime('%Y%m%d')
    output_path = os.path.join(output_dir, f"sharpe_optimized_factor_{window}d_rebalance{rebalance_period}d_{today}.csv")
    factor_df.to_csv(output_path, index=False)

    print(f"✅ 夏普比率优化组合因子数据已保存至: {output_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    print("\n因子统计信息:")
    print(factor_df['factor'].describe())

    # 验证最终组合因子的归一化结果
    print(f"\n最终组合因子范围: [{factor_df['factor'].min():.6f}, {factor_df['factor'].max():.6f}]")

    # 显示最终权重信息
    final_returns = factor_returns.tail(lookback_periods)
    final_weights = optimize_weights(final_returns, lookback_periods)
    print(f"\n最终优化权重:")
    for factor, weight in zip(available_factors, final_weights):
        print(f"{factor}: {weight:.4f}")

    print("\n数据预览:")
    print(factor_df.head())

    return factor_df

if __name__ == "__main__":
    create_sharpe_optimized_factor(window=20, rebalance_period=10, lookback_periods=34) 