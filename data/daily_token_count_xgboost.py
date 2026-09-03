"""
统计xgboost因子文件中每天有多少种token
"""
import pandas as pd
import os

def count_daily_tokens(csv_file_path):
    """
    统计CSV文件中每天有多少种不同的token
    
    Parameters:
    -----------
    csv_file_path : str
        CSV文件路径
    
    Returns:
    --------
    pd.DataFrame
        包含日期和token数量的DataFrame
    """
    # 读取CSV文件
    print(f"正在读取文件: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    # 确保date列是日期类型
    df['date'] = pd.to_datetime(df['date'])
    
    # 按日期分组，统计每天有多少种不同的token（instrument）
    daily_token_count = df.groupby('date')['instrument'].nunique().reset_index()
    daily_token_count.columns = ['date', 'token_count']
    
    # 按日期排序
    daily_token_count = daily_token_count.sort_values('date')
    
    # 显示统计信息
    print(f"\n总共有 {len(daily_token_count)} 个交易日")
    print(f"平均每天有 {daily_token_count['token_count'].mean():.2f} 种token")
    print(f"最少token数量: {daily_token_count['token_count'].min()}")
    print(f"最多token数量: {daily_token_count['token_count'].max()}")
    
    # 显示前10条和后10条数据
    print("\n前10天的token数量:")
    print(daily_token_count.head(10).to_string(index=False))
    
    print("\n后10天的token数量:")
    print(daily_token_count.tail(10).to_string(index=False))
    
    # 保存结果到CSV文件
    output_file = csv_file_path.replace('.csv', '_daily_token_count.csv')
    daily_token_count.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")
    
    return daily_token_count

if __name__ == "__main__":
    # 文件路径
    csv_file = "/Users/dmiwu/work/PythonProject/QuantTest/factor_stock/个人投研之因子分析/Crypto/data/factor_data/volatility_efficiency_20d_rebalance10d_20251228.csv"
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: 文件不存在 {csv_file}")
    else:
        # 执行统计
        result = count_daily_tokens(csv_file)
        
        # 显示完整结果（可选）
        # print("\n完整统计结果:")
        # print(result.to_string(index=False))

