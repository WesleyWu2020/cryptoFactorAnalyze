import pandas as pd
import os
import shutil
from datetime import datetime
from factor_analyse_custom import factor_miner
import numpy as np
import sys

# 导入因子配置
from factor_config import FACTOR_CONFIG, get_factor_config, list_all_factors

def main(factor_type="volume_stability_20", rebalance_period=3):
    """
    主函数，用于分析不同类型的因子
    
    参数:
    factor_type: 因子类型，可选值为FACTOR_CONFIG中定义的因子
    rebalance_period: 调仓周期，默认为3天
    """
    # 检查因子类型是否支持
    if factor_type not in FACTOR_CONFIG:
        print(f"错误: 不支持的因子类型 '{factor_type}'")
        print(f"支持的因子类型: {', '.join(FACTOR_CONFIG.keys())}")
        return
    
    # 获取因子配置
    config = FACTOR_CONFIG[factor_type]
    file_prefix = config["file_prefix"]
    factor_name = config["factor_name"]
    factor_direction = config["factor_direction"]
    factor_desc = config["factor_desc"]
    # 使用配置中的调仓周期（如果存在），否则使用默认值
    factor_rebalance_period = config.get("rebalance_period", rebalance_period)
    
    print(f"开始分析 {factor_desc} ({factor_type}) - {factor_rebalance_period}天调仓")
     
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取最新的因子数据文件
    factor_data_dir = os.path.join(os.path.dirname(current_dir), "data", "factor_data")
    
    # 先尝试查找包含特定调仓周期的文件
    rebalance_pattern = f"rebalance{factor_rebalance_period}d"
    rebalance_files = [f for f in os.listdir(factor_data_dir) 
                      if f.startswith(file_prefix) and rebalance_pattern in f]

    if rebalance_files:
        print(f"找到调仓周期为{factor_rebalance_period}天的因子文件")
        latest_file = sorted(rebalance_files)[-1]
    else:
        # 如果没有找到特定调仓周期的文件，则使用普通文件
        factor_files = [f for f in os.listdir(factor_data_dir) if f.startswith(file_prefix)]
        if not factor_files:
            raise FileNotFoundError(f"未找到{factor_desc}数据文件")
        
        print(f"警告: 未找到调仓周期为{factor_rebalance_period}天的因子文件，使用默认文件")
        latest_file = sorted(factor_files)[-1]

    factor_path = os.path.join(factor_data_dir, latest_file)
    print(f"选择的因子文件: {latest_file}")
    
    # 加载数据
    """
    数据必须要有三列: 
    |date|instrument|factor|
    分别是日期列, 代码列表列, 还有因子值一列
    
    注意事项: 
    1. 其中date必须要是 yyyy-mm-dd 的形式
    2. instrument列是加密货币交易对符号
    """
    df = pd.read_csv(factor_path)
    
    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['date'] = pd.to_datetime(df['date'])
    
    # 重命名因子列，使其与factor_name参数匹配
    df = df.rename(columns={'factor': factor_name})
    
    # 检查因子值的分布情况
    print("\n因子值统计信息:")
    print(df[factor_name].describe())
    
    # 检测并移除因子值完全相同的币种
    print("\n检测因子值完全相同的币种...")
    
    # 按币种分组计算标准差
    std_by_instrument = df.groupby('instrument')[factor_name].std().reset_index()
    std_by_instrument.columns = ['instrument', 'factor_std']
    
    # 找出因子值标准差为0的币种（表示该币种的所有因子值都相同）
    zero_std_instruments = std_by_instrument[std_by_instrument['factor_std'] == 0]['instrument'].tolist()
    
    if zero_std_instruments:
        print(f"发现 {len(zero_std_instruments)} 个币种的因子值完全相同，将被移除:")
        for i, ins in enumerate(zero_std_instruments[:10]):  # 只显示前10个
            print(f"  {i+1}. {ins}")
        if len(zero_std_instruments) > 10:
            print(f"  ...以及其他 {len(zero_std_instruments) - 10} 个币种")
            
        # 移除这些币种
        df = df[~df['instrument'].isin(zero_std_instruments)]
        print(f"移除后剩余 {df['instrument'].nunique()} 个币种")
    else:
        print("未发现因子值完全相同的币种")
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(current_dir), "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出报告文件路径
    today = datetime.now().strftime('%Y%m%d')
    report_path = os.path.join(output_dir, f"{factor_type}_rebalance{factor_rebalance_period}d_report_{today}.html")
    
    # 使用已有的HTML模板
    board_template = os.path.join(current_dir, "board.html")
    if os.path.exists(board_template):
        print(f"使用已有HTML模板: {board_template}")
    
    # 实例化因子分析库
    f = factor_miner(
        factor_data=df, 
        factor_name=factor_name, 
        factor_direction=factor_direction, 
        render_path=report_path,
        n_groups=10,  # 使用10个分组
        rebalance_period=factor_rebalance_period  # 设置调仓周期
    )
    
    print(f"开始生成因子分析报告...")
    f.render()
    print(f"✅ 因子分析报告已生成: {report_path}")
    
    # 可选：将生成的报告复制到web服务目录
    web_dir = os.path.join(os.path.dirname(current_dir), "web")
    if os.path.exists(web_dir):
        web_report_path = os.path.join(web_dir, f"{factor_type}_rebalance{factor_rebalance_period}d_latest.html")
        shutil.copy2(report_path, web_report_path)
        print(f"✅ 报告已复制到web目录: {web_report_path}")

if __name__ == "__main__":
    # 从命令行参数获取因子类型和调仓周期
    factor_type = "Retail_Friction_Illiquidity_Factor"
    rebalance_period = None # 默认为None，会使用因子配置中的值或main函数的默认值
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list" or sys.argv[1] == "-l":
            list_all_factors()
            sys. exit(0)
        factor_type = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            rebalance_period = int(sys.argv[2])
            if rebalance_period <= 0:
                print("警告: 调仓周期必须大于0，使用默认值")
                rebalance_period = None
        except ValueError:
            print("警告: 调仓周期必须是整数，使用默认值")
    
    # 如果提供了调仓周期参数，则使用它；否则使用None（会使用因子配置中的值或默认值）
    main(factor_type, rebalance_period)

