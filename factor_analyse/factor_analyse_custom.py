import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import sys
import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from scipy import stats  # 添加scipy.stats用于计算t值和p值

warnings.filterwarnings('ignore')

"""
加密货币因子分析, 获取组内的收益、换手率、夏普、胜率、盈亏比、ic、波动率、年化波动、回撤
采用pyecharts可视化, 部署前端, 保证可以每日都定时更新
支持样本内外分析，默认最后180天作为样本外
"""
class factor_miner:
    @staticmethod
    def _group_apply_preserve_columns(df: pd.DataFrame, by, func) -> pd.DataFrame:
        """
        在新版 pandas 下，groupby.apply 可能不再自动保留分组列。
        这里改为显式逐组处理并拼接，确保原列稳定保留。
        """
        chunks = []
        for _, group in df.groupby(by, sort=False):
            result = func(group.copy())
            if result is not None and not result.empty:
                chunks.append(result)
        if not chunks:
            return df.iloc[0:0].copy()
        return pd.concat(chunks, ignore_index=True)

    def __init__(self, 
                 factor_data: pd.DataFrame, 
                 factor_name: str, 
                 factor_direction: int, 
                 render_path: str=None,
                 api_key: str=None,
                 api_secret: str=None,
                 n_groups: int=5,
                 rebalance_period: int=1,
                 out_of_sample_days: int=180):  # 添加样本外天数参数
        """
        最终在https://hostlocal:5000/factors/因子名 中访问
        :params factor_data: 因子数据, 必须包含三个字段, 日期、币种代码、因子
        :params factor_name: 因子名称
        :params factor_direction: 因子方向, 主观上正相关则为1, 否则为-1
        :params render_path: 渲染页面保存路径
        :params api_key: 币安API Key
        :params api_secret: 币安API Secret
        :params n_groups: 分组数量，默认为5
        :params rebalance_period: 调仓周期，默认为1天
        :params out_of_sample_days: 样本外天数，默认为180天
        """
        self.factor_data = factor_data
        self.factor_name = factor_name
        self.factor_direction = factor_direction
        self.render_path = render_path
        self.api_key = api_key
        self.api_secret = api_secret
        self.n_groups = n_groups  # 保存分组数
        self.rebalance_period = rebalance_period
        self.out_of_sample_days = out_of_sample_days  # 保存样本外天数
        self.factor_data['date'] = pd.to_datetime(self.factor_data['date'])
        self.factor_data.sort_values('date', inplace=True)
        
        # 初始化币安API客户端
        self.client = None

        print(f'=============================因子分组（{n_groups}组）=================================')
        # 因子分组，允许重复边界值
        def fac_group(df):
            # 先删除因子值为空的行
            df = df.dropna(subset=[factor_name])
            
            # 检查数据量
            if len(df) == 0:
                return df
            
            # 如果数据量少于分组数，降低分组数
            actual_groups = min(n_groups, len(df))
            
            # 删除重复值（基于 instrument 和 factor_name）
            df = df.drop_duplicates(subset=['instrument', factor_name])
            
            # 再次检查数据量
            if len(df) < actual_groups:
                actual_groups = max(1, len(df))
            
            try:
                # 尝试使用 qcut（分位数分组）
                df['group'] = pd.qcut(
                    df[factor_name], 
                    q=actual_groups, 
                    labels=range(actual_groups), 
                    duplicates='drop'
                )
            except Exception as e:
                try:
                    # 如果 qcut 失败，尝试使用 cut（等距分组）
                    df['group'] = pd.cut(
                        df[factor_name], 
                        bins=actual_groups, 
                        labels=range(actual_groups), 
                        include_lowest=True,
                        duplicates='drop'
                    )
                except Exception as e2:
                    # 如果都失败，使用简单的排序分组
                    print(f"    警告: 该日期分组失败，使用排序分组。样本数={len(df)}")
                    df = df.sort_values(factor_name)
                    group_size = max(1, len(df) // actual_groups)
                    groups = []
                    for i in range(len(df)):
                        group_id = min(i // group_size, actual_groups - 1)
                        groups.append(group_id)
                    df['group'] = groups
            
            return df
        
        self.factor_data = self._group_apply_preserve_columns(self.factor_data, 'date', fac_group)

        # 如果数据中没有future_ret列，则从本地数据或币安API获取价格数据计算
        if 'future_ret' not in self.factor_data.columns:
            print("数据中没有future_ret列，尝试从本地K线数据计算...")
            self._calculate_future_returns()
        
        # 合并数据
        self.data = self.factor_data.copy()
        
        # 因子预处理
        self.data = self._processing()
        
        # 分割样本内外数据
        self._split_in_out_sample()

    def _split_in_out_sample(self):
        """
        分割样本内外数据
        """
        # 获取所有日期
        all_dates = sorted(self.data['date'].unique())
        
        if len(all_dates) <= self.out_of_sample_days:
            print(f"警告: 总数据天数({len(all_dates)})小于等于样本外天数({self.out_of_sample_days})，无法进行样本内外分割")
            # 如果数据不足，将所有数据作为样本内
            self.split_date = all_dates[-1]
            self.data_in_sample = self.data.copy()
            self.data_out_sample = pd.DataFrame()
            self.in_sample_dates = len(all_dates)
            self.out_sample_dates = 0
        else:
            # 计算分割点
            split_index = len(all_dates) - self.out_of_sample_days
            self.split_date = all_dates[split_index - 1]  # 样本内的最后一天
            
            # 分割数据
            self.data_in_sample = self.data[self.data['date'] <= self.split_date].copy()
            self.data_out_sample = self.data[self.data['date'] > self.split_date].copy()
            
            self.in_sample_dates = len(self.data_in_sample['date'].unique())
            self.out_sample_dates = len(self.data_out_sample['date'].unique())
            
        print(f"样本内数据: {self.in_sample_dates} 天 (至 {self.split_date.strftime('%Y-%m-%d')})")
        print(f"样本外数据: {self.out_sample_dates} 天")

    def _calculate_future_returns(self):
        """
        计算未来收益率
        仅从本地K线数据计算；不再从币安API获取
        """
        try:
            # 尝试从本地K线数据计算
            self._calculate_future_returns_from_local()
        except Exception as e:
            print(f"从本地数据计算future_ret失败: {e}")
            print("离线模式: 跳过future_ret补全；请在因子生成阶段写入future_ret或确保本地K线可用。")
    
    def _calculate_future_returns_from_local(self):
        """
        从本地K线数据计算未来收益率，考虑调仓周期
        """
        import os
        
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # K线数据文件路径
        kline_dir = os.path.join(os.path.dirname(current_dir), "data", "kline_data")
        
        # 查找最新的K线数据文件
        kline_files = [f for f in os.listdir(kline_dir) if f.startswith("binance_daily_klines_")]
        if not kline_files:
            raise FileNotFoundError("未找到K线数据文件")
        
        latest_file = sorted(kline_files)[-1]
        kline_path = os.path.join(kline_dir, latest_file)
        
        print(f"读取K线数据: {kline_path}")
        
        # 读取K线数据
        klines_df = pd.read_csv(kline_path)
        
        # 确保日期格式正确
        klines_df['date'] = pd.to_datetime(klines_df['date'])
        
        # 计算未来收益率，考虑调仓周期
        def calculate_returns(group):
            group = group.sort_values('date')
            # 使用调仓周期计算未来收益率
            group['future_ret'] = group['close'].shift(-self.rebalance_period) / group['close'] - 1
            return group
        
        klines_df = self._group_apply_preserve_columns(klines_df, 'symbol', calculate_returns)
        
        # 只保留需要的列
        price_data = klines_df[['date', 'symbol', 'future_ret']].dropna()
        
        # 重命名列以匹配因子数据
        price_data = price_data.rename(columns={'symbol': 'instrument'})
        
        # 合并到因子数据
        self.factor_data = pd.merge(
            self.factor_data, 
            price_data[['date', 'instrument', 'future_ret']], 
            on=['date', 'instrument'], 
            how='left'
        )
        
        print(f"成功从本地K线数据计算future_ret (调仓周期: {self.rebalance_period}天)，共 {len(self.factor_data[~self.factor_data['future_ret'].isna()])} 条有效数据")
    
    def _calculate_future_returns_from_api(self):
        """
        已禁用：离线模式不再从币安API获取数据
        """
        raise RuntimeError("已禁用：离线模式不从币安API获取价格数据。请提供本地K线或在因子生成阶段写入future_ret。")
    
    def _processing(self):
        """
        数据预处理, 去极值
        """
        print('============================因子去极值==================================')
        def nor(df):
            up = df[self.factor_name].mean() + 3 * df[self.factor_name].std()
            down = df[self.factor_name].mean() - 3 * df[self.factor_name].std()
            df = df[(df[self.factor_name] <= up) & (df[self.factor_name] >= down)]
            return df

        return self._group_apply_preserve_columns(self.data, 'date', nor)
    
    def _turnover(self, group_num):
        """
        计算换手率
        :params group_num: 组数
        """
        # 按照日期和组类聚合所有币种
        df = self.data[self.data['group']==group_num]
        df = df[['date', 'instrument']]
        
        # 修复这里的代码，避免 'date' 列重复
        grouped = df.groupby(['date']).apply(lambda x: set(x['instrument'].to_list()))
        df_sets = pd.DataFrame({'ins_set': grouped.values}, index=grouped.index)
        df_sets = df_sets.reset_index()  # 现在只有一个 'date' 列
        
        df_sets['yesterday_set'] = df_sets['ins_set'].shift(1)
        df_sets['repeat_set'] = df_sets.apply(lambda x: x['ins_set'].intersection(x['yesterday_set']) if isinstance(x['yesterday_set'], set) else set(), axis=1)
        df_sets = df_sets.dropna()
        df_sets['repeat_num'] = df_sets['repeat_set'].apply(len)
        df_sets['total_num'] = df_sets['ins_set'].apply(len)
        df_sets['turn'] = 1 - df_sets['repeat_num'] / df_sets['total_num']
        return np.nanmean(df_sets['turn'])

    def _turnover_with_data(self, group_num, data):
        """
        使用指定数据计算换手率
        :param group_num: 组数
        :param data: 指定的数据
        """
        # 按照日期和组类聚合所有币种
        df = data[data['group']==group_num]
        df = df[['date', 'instrument']]
        
        # 修复这里的代码，避免 'date' 列重复
        grouped = df.groupby(['date']).apply(lambda x: set(x['instrument'].to_list()))
        df_sets = pd.DataFrame({'ins_set': grouped.values}, index=grouped.index)
        df_sets = df_sets.reset_index()  # 现在只有一个 'date' 列
        
        df_sets['yesterday_set'] = df_sets['ins_set'].shift(1)
        df_sets['repeat_set'] = df_sets.apply(lambda x: x['ins_set'].intersection(x['yesterday_set']) if isinstance(x['yesterday_set'], set) else set(), axis=1)
        df_sets = df_sets.dropna()
        df_sets['repeat_num'] = df_sets['repeat_set'].apply(len)
        df_sets['total_num'] = df_sets['ins_set'].apply(len)
        df_sets['turn'] = 1 - df_sets['repeat_num'] / df_sets['total_num']
        return np.nanmean(df_sets['turn'])

    def IC(self):
        # 假设我们有future_ret列
        if 'future_ret' not in self.data.columns:
            print("警告: 数据中没有future_ret列，无法计算IC")
            return 0, pd.DataFrame({'date': [], 'acc_ic': []}), 0, 0, 0, 0, 0, 0
            
        # 按日期分组计算每日IC值和Rank IC值
        def calculate_daily_ic_and_rank_ic(x):
            # 计算普通IC (Pearson相关系数)
            ic = x[self.factor_name].corr(x['future_ret'])
            
            # 计算Rank IC (Spearman秩相关系数)
            rank_ic = x[self.factor_name].corr(x['future_ret'], method='spearman')
            
            return pd.Series({'ic': ic, 'rank_ic': rank_ic})
        
        daily_ic = self.data.groupby('date').apply(calculate_daily_ic_and_rank_ic).reset_index()
        
        # 计算平均IC值和平均Rank IC值
        ic = daily_ic['ic'].mean()
        rank_ic = daily_ic['rank_ic'].mean()
        
        # 计算IC的t值和p值
        ic_std = daily_ic['ic'].std()
        ic_count = len(daily_ic['ic'].dropna())
        if ic_std > 0 and ic_count > 1:
            t_stat = ic / (ic_std / np.sqrt(ic_count))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=ic_count-1))  # 双尾检验
            # 处理极小的p值，避免显示为0.0000
            if p_value < 0.0001:
                p_value = 0.0001  # 设置为最小显示值
        else:
            t_stat = 0
            p_value = 1
        
        # 计算IC_IR (IC均值/IC标准差)
        ic_ir = daily_ic['ic'].mean() / daily_ic['ic'].std() if daily_ic['ic'].std() != 0 else 0
        
        # 计算IR (年化收益率/年化波动率)
        # 首先计算多空组合收益
        df = self.data.copy()
        if self.factor_direction == 1:
            # 因子值越高，预期收益越高
            high_group = df[df['group'] == self.n_groups - 1]
            low_group = df[df['group'] == 0]
        else:
            # 因子值越低，预期收益越高
            high_group = df[df['group'] == 0]
            low_group = df[df['group'] == self.n_groups - 1]
        
        # 计算每日多空组合收益
        high_returns = high_group.groupby('date')['future_ret'].mean()
        low_returns = low_group.groupby('date')['future_ret'].mean()
        
        # 确保索引对齐
        common_dates = high_returns.index.intersection(low_returns.index)
        long_short_returns = high_returns.loc[common_dates] - low_returns.loc[common_dates]
        
        # 计算IR
        ir = long_short_returns.mean() / long_short_returns.std() * np.sqrt(365) if long_short_returns.std() != 0 else 0

        # 计算秩相关系数（单调性检验）
        # 计算各分组的平均收益率
        group_returns = []
        group_ranks = []
        for i in range(self.n_groups):
            group_data = df[df['group'] == i]
            if len(group_data) > 0:
                avg_return = group_data.groupby('date')['future_ret'].mean().mean()
                group_returns.append(avg_return)
                group_ranks.append(i)
        
        if len(group_returns) > 1:
            # 计算斯皮尔曼秩相关系数
            spearman_corr, spearman_p = stats.spearmanr(group_ranks, group_returns)
            # 确保秩相关系数有足够的小数位数
            if abs(spearman_corr) < 0.0001:
                spearman_corr = 0.0
        else:
            spearman_corr = 0
            spearman_p = 1

        # 累计IC
        daily_ic['acc_ic'] = daily_ic['ic'].cumsum()
        
        return ic, daily_ic[['date', 'acc_ic']], ir, ic_ir, t_stat, p_value, spearman_corr, rank_ic, daily_ic

    def IC_with_sample_split(self, data=None, sample_type="全样本"):
        """
        计算IC指标，支持样本内外分析
        :param data: 数据，如果为None则使用self.data
        :param sample_type: 样本类型标识，用于打印信息
        """
        if data is None:
            data = self.data
            
        # 假设我们有future_ret列
        if 'future_ret' not in data.columns or len(data) == 0:
            print(f"警告: {sample_type}数据中没有future_ret列或数据为空，无法计算IC")
            return 0, pd.DataFrame({'date': [], 'acc_ic': []}), 0, 0, 0, 0, 0, 0
            
        # 按日期分组计算每日IC值和Rank IC值
        def calculate_daily_ic_and_rank_ic(x):
            # 计算普通IC (Pearson相关系数)
            ic = x[self.factor_name].corr(x['future_ret'])
            
            # 计算Rank IC (Spearman秩相关系数)
            rank_ic = x[self.factor_name].corr(x['future_ret'], method='spearman')
            
            return pd.Series({'ic': ic, 'rank_ic': rank_ic})
        
        daily_ic = data.groupby('date').apply(calculate_daily_ic_and_rank_ic).reset_index()
        
        # 计算平均IC值和平均Rank IC值
        ic = daily_ic['ic'].mean()
        rank_ic = daily_ic['rank_ic'].mean()
        
        # 计算IC的t值和p值
        ic_std = daily_ic['ic'].std()
        ic_count = len(daily_ic['ic'].dropna())
        if ic_std > 0 and ic_count > 1:
            t_stat = ic / (ic_std / np.sqrt(ic_count))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=ic_count-1))  # 双尾检验
            # 处理极小的p值，避免显示为0.0000
            if p_value < 0.0001:
                p_value = 0.0001  # 设置为最小显示值
        else:
            t_stat = 0
            p_value = 1
        
        # 计算IC_IR (IC均值/IC标准差)
        ic_ir = daily_ic['ic'].mean() / daily_ic['ic'].std() if daily_ic['ic'].std() != 0 else 0
        
        # 计算IR (年化收益率/年化波动率)
        # 首先计算多空组合收益
        df = data.copy()
        if self.factor_direction == 1:
            # 因子值越高，预期收益越高
            high_group = df[df['group'] == self.n_groups - 1]
            low_group = df[df['group'] == 0]
        else:
            # 因子值越低，预期收益越高
            high_group = df[df['group'] == 0]
            low_group = df[df['group'] == self.n_groups - 1]
        
        # 计算每日多空组合收益
        high_returns = high_group.groupby('date')['future_ret'].mean()
        low_returns = low_group.groupby('date')['future_ret'].mean()
        
        # 确保索引对齐
        common_dates = high_returns.index.intersection(low_returns.index)
        long_short_returns = high_returns.loc[common_dates] - low_returns.loc[common_dates]
        
        # 计算IR
        ir = long_short_returns.mean() / long_short_returns.std() * np.sqrt(365) if long_short_returns.std() != 0 else 0

        # 计算秩相关系数（单调性检验）
        # 计算各分组的平均收益率
        group_returns = []
        group_ranks = []
        for i in range(self.n_groups):
            group_data = df[df['group'] == i]
            if len(group_data) > 0:
                avg_return = group_data.groupby('date')['future_ret'].mean().mean()
                group_returns.append(avg_return)
                group_ranks.append(i)
        
        if len(group_returns) > 1:
            # 计算斯皮尔曼秩相关系数
            spearman_corr, spearman_p = stats.spearmanr(group_ranks, group_returns)
            # 确保秩相关系数有足够的小数位数
            if abs(spearman_corr) < 0.0001:
                spearman_corr = 0.0
        else:
            spearman_corr = 0
            spearman_p = 1

        # 累计IC
        daily_ic['acc_ic'] = daily_ic['ic'].cumsum()
        
        return ic, daily_ic[['date', 'acc_ic']], ir, ic_ir, t_stat, p_value, spearman_corr, rank_ic, daily_ic

    def performance(self, group_num):
        """
        获取投资组合相关指标
        :params group_num: 组数
        """
        df = self.data[self.data['group']==group_num]

        # 换手
        turn = self._turnover(group_num)
        
        # 假设我们有future_ret列
        if 'future_ret' not in df.columns:
            print(f"警告: 数据中没有future_ret列，无法计算组 {group_num} 的性能指标")
            return pd.DataFrame({'group': [str(group_num)], 
                               'turnover': [round(turn, 4)]})
        
        # 计算组内IC
        ic = df['future_ret'].corr(df[self.factor_name])

        # 计算组内平均收益 - 修正：将N天收益率转换为日收益率
        daily_future_ret = df.groupby('date').apply(lambda x: x['future_ret'].mean())
        
        # 将N天收益率转换为等效日收益率
        # 方法1：简单平均（适用于小收益率）
        daily_adjusted_ret = daily_future_ret / self.rebalance_period
        
        # 方法2：复利转换（更精确）
        # daily_adjusted_ret = (1 + daily_future_ret) ** (1/self.rebalance_period) - 1
        
        return_ratio = np.sum(daily_adjusted_ret)
        # 加密货币市场是24/7的，使用365天而不是242天
        annual_return_ratio = return_ratio * 365 / len(daily_adjusted_ret)

        sharp_ratio = daily_adjusted_ret.mean() / daily_adjusted_ret.std(ddof=1) * np.sqrt(len(daily_adjusted_ret))   # 夏普比率
        annual_vo_ratio = daily_adjusted_ret.std() * np.sqrt(365)                 # 年化波动率，使用365天

        # 最大回撤 - 修正计算方法
        def get_max_drawdown_fast(returns):
            """
            计算最大回撤
            
            参数:
                returns: 日收益率序列
                
            返回:
                最大回撤百分比
            """
            # 处理空数组或只有一个元素的情况
            if len(returns) <= 1:
                return 0.0
                
            # 计算累计收益率序列
            cum_returns = (1 + returns).cumprod() - 1
            
            # 转换为净值序列  
            nav = 1 + cum_returns
            
            # 计算累计最大净值
            running_max = np.maximum.accumulate(nav)
            
            # 计算回撤
            drawdowns = (running_max - nav) / running_max
            
            # 获取最大回撤
            max_drawdown = drawdowns.max()
            
            return round(max_drawdown * 100, 2)  # 转换为百分比并四舍五入到两位小数
            
        max_drawdowns = get_max_drawdown_fast(daily_adjusted_ret)

        # 计算最大回撤周期与回撤恢复时长
        def get_max_drawdown_metrics(returns):
            """
            返回: (max_drawdown_pct, drawdown_days, recovery_days)
            - max_drawdown_pct: 最大回撤比例(0-1)
            - drawdown_days: 从峰值到谷底的天数
            - recovery_days: 从谷底到重新突破该峰值的天数；若未恢复返回None
            """
            if len(returns) <= 1:
                return 0.0, 0, None
            nav = (1 + returns).cumprod()
            nav_values = nav.values.astype(float)
            running_max = np.maximum.accumulate(nav_values)
            drawdowns = (running_max - nav_values) / running_max
            trough_idx = int(np.argmax(drawdowns))
            if trough_idx <= 0:
                return float(drawdowns[trough_idx]), 0, 0
            peak_idx = int(np.argmax(nav_values[:trough_idx+1]))
            peak_value = nav_values[peak_idx]
            # 计算天数
            idx_dates = pd.to_datetime(nav.index)
            drawdown_days = (idx_dates[trough_idx] - idx_dates[peak_idx]).days
            # 寻找恢复点
            recovery_idx = None
            for i in range(trough_idx + 1, len(nav_values)):
                if nav_values[i] >= peak_value:
                    recovery_idx = i
                    break
            if recovery_idx is not None:
                recovery_days = (idx_dates[recovery_idx] - idx_dates[trough_idx]).days
            else:
                recovery_days = None
            return float(drawdowns[trough_idx]), int(max(drawdown_days, 0)), (int(recovery_days) if recovery_days is not None else None)

        _, dd_days, rec_days = get_max_drawdown_metrics(daily_adjusted_ret)

        win_percent = len(daily_adjusted_ret[daily_adjusted_ret>0]) / len(daily_adjusted_ret)   # 胜率

        if self.factor_direction == 1:
            if group_num == self.n_groups - 1:  # 最后一组
                group_num = 'long'
            elif group_num == 0:  # 第一组
                group_num = 'short'
        else:
            if group_num == 0:  # 第一组
                group_num = 'long'
            elif group_num == self.n_groups - 1:  # 最后一组
                group_num = 'short'
        
        group_num = str(group_num)
        return pd.DataFrame({'group': [group_num], 
                             'return': [round(return_ratio, 4)], 
                             'turnover': [round(turn, 4)], 
                             'annual_return': [round(annual_return_ratio, 4)], 
                             'sharp': [round(sharp_ratio, 4)], 
                             'IC': [round(ic, 4)], 
                             'volatility': [round(daily_adjusted_ret.std(), 4)], 
                             'annual_volatility': [round(annual_vo_ratio, 4)], 
                             'max_drawback': [round(max_drawdowns, 4)], 
                             'md_period_days': [dd_days if dd_days is not None else '-'], 
                             'recovery_period_days': [rec_days if rec_days is not None else '-'], 
                             'win_percent': [round(win_percent, 4)], 
                             'profit-loss ratio': [round(daily_adjusted_ret[daily_adjusted_ret>0].sum() / np.abs(daily_adjusted_ret[daily_adjusted_ret<0].sum()), 4)]})

    def performance_with_sample_split(self, group_num, data=None, sample_type="全样本"):
        """
        获取投资组合相关指标，支持样本内外分析
        :param group_num: 组数
        :param data: 数据，如果为None则使用self.data
        :param sample_type: 样本类型标识
        """
        if data is None:
            data = self.data
            
        df = data[data['group']==group_num]

        # 换手 - 需要使用特定数据计算
        turn = self._turnover_with_data(group_num, data)
        
        # 假设我们有future_ret列
        if 'future_ret' not in df.columns or len(df) == 0:
            print(f"警告: {sample_type}数据中没有future_ret列或组 {group_num} 数据为空，无法计算性能指标")
            return pd.DataFrame({'group': [str(group_num)], 
                               'turnover': [round(turn, 4)]})
        
        # 计算组内IC
        ic = df['future_ret'].corr(df[self.factor_name])

        # 计算组内平均收益 - 修正：将N天收益率转换为日收益率
        daily_future_ret = df.groupby('date').apply(lambda x: x['future_ret'].mean())
        
        # 将N天收益率转换为等效日收益率
        # 方法1：简单平均（适用于小收益率）
        daily_adjusted_ret = daily_future_ret / self.rebalance_period
        
        # 方法2：复利转换（更精确）
        # daily_adjusted_ret = (1 + daily_future_ret) ** (1/self.rebalance_period) - 1
        
        return_ratio = np.sum(daily_adjusted_ret)
        # 加密货币市场是24/7的，使用365天而不是242天
        annual_return_ratio = return_ratio * 365 / len(daily_adjusted_ret)

        sharp_ratio = daily_adjusted_ret.mean() / daily_adjusted_ret.std(ddof=1) * np.sqrt(len(daily_adjusted_ret))   # 夏普比率
        annual_vo_ratio = daily_adjusted_ret.std() * np.sqrt(365)                 # 年化波动率，使用365天

        # 最大回撤 - 修正计算方法
        def get_max_drawdown_fast(returns):
            """
            计算最大回撤
            
            参数:
                returns: 日收益率序列
                
            返回:
                最大回撤百分比
            """
            # 处理空数组或只有一个元素的情况
            if len(returns) <= 1:
                return 0.0
                
            # 计算累计收益率序列
            cum_returns = (1 + returns).cumprod() - 1
            
            # 转换为净值序列  
            nav = 1 + cum_returns
            
            # 计算累计最大净值
            running_max = np.maximum.accumulate(nav)
            
            # 计算回撤
            drawdowns = (running_max - nav) / running_max
            
            # 获取最大回撤
            max_drawdown = drawdowns.max()
            
            return round(max_drawdown * 100, 2)  # 转换为百分比并四舍五入到两位小数
            
        max_drawdowns = get_max_drawdown_fast(daily_adjusted_ret)

        # 计算最大回撤周期与回撤恢复时长（样本内/外）
        def get_max_drawdown_metrics(returns):
            if len(returns) <= 1:
                return 0.0, 0, None
            nav = (1 + returns).cumprod()
            nav_values = nav.values.astype(float)
            running_max = np.maximum.accumulate(nav_values)
            drawdowns = (running_max - nav_values) / running_max
            trough_idx = int(np.argmax(drawdowns))
            if trough_idx <= 0:
                return float(drawdowns[trough_idx]), 0, 0
            peak_idx = int(np.argmax(nav_values[:trough_idx+1]))
            peak_value = nav_values[peak_idx]
            idx_dates = pd.to_datetime(nav.index)
            drawdown_days = (idx_dates[trough_idx] - idx_dates[peak_idx]).days
            recovery_idx = None
            for i in range(trough_idx + 1, len(nav_values)):
                if nav_values[i] >= peak_value:
                    recovery_idx = i
                    break
            if recovery_idx is not None:
                recovery_days = (idx_dates[recovery_idx] - idx_dates[trough_idx]).days
            else:
                recovery_days = None
            return float(drawdowns[trough_idx]), int(max(drawdown_days, 0)), (int(recovery_days) if recovery_days is not None else None)

        _, dd_days, rec_days = get_max_drawdown_metrics(daily_adjusted_ret)

        win_percent = len(daily_adjusted_ret[daily_adjusted_ret>0]) / len(daily_adjusted_ret)   # 胜率

        if self.factor_direction == 1:
            if group_num == self.n_groups - 1:  # 最后一组
                group_num = 'long'
            elif group_num == 0:  # 第一组
                group_num = 'short'
        else:
            if group_num == 0:  # 第一组
                group_num = 'long'
            elif group_num == self.n_groups - 1:  # 最后一组
                group_num = 'short'
        
        group_num = str(group_num)
        return pd.DataFrame({'group': [group_num], 
                             'return': [round(return_ratio, 4)], 
                             'turnover': [round(turn, 4)], 
                             'annual_return': [round(annual_return_ratio, 4)], 
                             'sharp': [round(sharp_ratio, 4)], 
                             'IC': [round(ic, 4)], 
                             'volatility': [round(daily_adjusted_ret.std(), 4)], 
                             'annual_volatility': [round(annual_vo_ratio, 4)], 
                             'max_drawback': [round(max_drawdowns, 4)], 
                             'md_period_days': [dd_days if dd_days is not None else '-'], 
                             'recovery_period_days': [rec_days if rec_days is not None else '-'], 
                             'win_percent': [round(win_percent, 4)], 
                             'profit-loss ratio': [round(daily_adjusted_ret[daily_adjusted_ret>0].sum() / np.abs(daily_adjusted_ret[daily_adjusted_ret<0].sum()), 4)]})

    def show_plot(self):
        """
        分组收益和累计IC图
        """
        from pyecharts.charts import Line
        from pyecharts import options as opts

        # 假设我们有future_ret列
        if 'future_ret' not in self.data.columns:
            print("警告: 数据中没有future_ret列，无法绘制收益曲线")
            # 创建空的图表
            line = Line()
            line.add_xaxis([])
            line.add_yaxis("无数据", [])
            
            line2 = Line()
            line2.add_xaxis([])
            line2.add_yaxis("无数据", [])
            
            line3 = Line() # 累计Rank IC图
            line3.add_xaxis([])
            line3.add_yaxis("无数据", [])

            line4 = Line() # Rank IC衰减图
            line4.add_xaxis([])
            line4.add_yaxis("无数据", [])

            line5 = Line() # Rank IC自相关图
            line5.add_xaxis([])
            line5.add_yaxis("无数据", [])
            
            return line, line2, line3, line4, line5
            
        # 计算分组累计收益 - 修正：将N天收益率转换为日收益率
        df = self.data[['date', 'group', 'future_ret']].copy()
        df = df.groupby(['date', 'group'], as_index=False)['future_ret'].mean()
        df['ret'] = df['future_ret'] / self.rebalance_period
        df = pd.pivot_table(df, values=['ret'], index='date', columns='group').cumsum()
        
        # 确保列名是正确的格式
        df.columns = [col[1] for col in df.columns]
        
        # 确保所有分组都有数据
        for i in range(self.n_groups):
            if i not in df.columns:
                print(f"警告: 分组 {i} 没有数据，添加零值序列")
                df[i] = 0.0
        
        # 重新排序列，确保按照0,1,2,...,n_groups-1的顺序
        df = df.reindex(sorted(df.columns), axis=1)
        
        columns = list(df.columns)

        # 求多空组合累计收益曲线
        if self.factor_direction == 1:
            df['long_short'] = df[columns[-1]] - df[columns[0]]
        else:
            df['long_short'] = df[columns[0]] - df[columns[-1]]

        # 绘制折线图
        x = list(df.index)
        line = Line()
        line.add_xaxis(x)
        
        # 确保所有分组都被添加到图表中
        for i in range(self.n_groups):
            if i in df.columns:
                # 使用更明确的分组标签格式
                group_label = f"Group {i}"
                line.add_yaxis(
                    group_label, 
                    df[i].tolist(),
                    is_symbol_show=False,
                    linestyle_opts=opts.LineStyleOpts(width=2)
                )
        
        # 添加多空组合曲线
        line.add_yaxis(
            "Long-Short", 
            df['long_short'].tolist(),
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=3, type_="dashed")
        )
        
        # 设置图表选项
        line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 隐藏数字
        line.set_global_opts(
            title_opts=opts.TitleOpts(title="分组收益曲线"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="5%"),
            xaxis_opts=opts.AxisOpts(type_="time"),
            yaxis_opts=opts.AxisOpts(
                name="累计收益",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            ),
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside")
            ]
        )

        # 绘制累计IC图 - IC计算本身应该是正确的，因为它是相关性而非收益率
        self._ic, acc_ic, self._ir, self._ic_ir, self._t_stat, self._p_value, self._spearman_corr, self._rank_ic, _ = self.IC()
        x = list(acc_ic['date'])
        y = list(acc_ic['acc_ic'])
        line2 = Line()
        line2.add_xaxis(x)
        line2.add_yaxis(
            '累计IC', 
            y, 
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2, color="#ffbe0b")
        )
        line2.set_global_opts(
            title_opts=opts.TitleOpts(title="累计IC曲线"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(type_="time"),
            yaxis_opts=opts.AxisOpts(
                name="累计IC值",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            ),
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside")
            ]
        )
        
        # 绘制累计Rank IC图
        # 计算累计Rank IC
        daily_rank_ic = self.data.groupby('date').apply(
            lambda x: x[self.factor_name].corr(x['future_ret'], method='spearman')
        ).reset_index()
        daily_rank_ic.columns = ['date', 'rank_ic']
        daily_rank_ic = daily_rank_ic.dropna()
        daily_rank_ic['cum_rank_ic'] = daily_rank_ic['rank_ic'].cumsum()
        
        x3 = list(daily_rank_ic['date'])
        y3 = list(daily_rank_ic['cum_rank_ic'])
        line3 = Line()
        line3.add_xaxis(x3)
        line3.add_yaxis(
            '累计Rank IC',
            y3,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2, color="#ff7f0e")
        )
        line3.set_global_opts(
            title_opts=opts.TitleOpts(title="累计Rank IC曲线"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(type_="time"),
            yaxis_opts=opts.AxisOpts(
                name="累计Rank IC值",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            ),
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside")
            ]
        )

        # 绘制Rank IC衰减图
        decay_data = self.calculate_rank_ic_decay(max_lag=10)
        x4 = [f"Lag{lag}" for lag in decay_data['lag']]
        y4 = decay_data['rank_ic'].tolist()
        
        from pyecharts.charts import Bar
        line4 = Bar()  # 使用柱状图更直观
        line4.add_xaxis(x4)
        line4.add_yaxis(
            "Rank IC", 
            y4,
            itemstyle_opts=opts.ItemStyleOpts(color="#3a86ff")
        )
        line4.set_global_opts(
            title_opts=opts.TitleOpts(title="Rank IC衰减图"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(name="滞后期"),
            yaxis_opts=opts.AxisOpts(
                name="Rank IC值",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            )
        )

        # 绘制Rank IC自相关图
        autocorr_data = self.calculate_rank_ic_autocorr(max_lag=20)
        x5 = [f"Lag{lag}" for lag in autocorr_data['lag']]
        y5 = autocorr_data['autocorr'].tolist()
        
        line5 = Line()
        line5.add_xaxis(x5)
        line5.add_yaxis(
            "自相关系数", 
            y5,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2, color="#ff6b6b"),
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(y=0)]  # 添加0线参考
            )
        )
        line5.set_global_opts(
            title_opts=opts.TitleOpts(title="Rank IC自相关图"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(name="滞后期"),
            yaxis_opts=opts.AxisOpts(
                name="自相关系数",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            )
        )
        
        return line, line2, line3, line4, line5
    
    def get_latest_group_symbols(self, include_incomplete_data=True):
        """
        获取最新一期各个分组的币种和因子值
        
        参数:
        include_incomplete_data: 是否包含未来收益率为NaN的最新数据
        """
        if include_incomplete_data:
            # 获取原始因子数据的最新日期
            latest_date = self.factor_data['date'].max()
        
        # 筛选最新日期的数据
            latest_data = self.factor_data[self.factor_data['date'] == latest_date].copy()
            
            # 如果没有分组信息，需要手动计算
            if 'group' not in latest_data.columns:
                # 复制因子分组逻辑
                def fac_group(df):
                    df = df.drop_duplicates(self.factor_name)
                    try:
                        df['group'] = pd.qcut(df[self.factor_name], q=self.n_groups, labels=range(self.n_groups), duplicates='drop')
                        if df['group'].nunique() < self.n_groups:
                            raise ValueError("qcut分组数不足")
                    except Exception as e:
                        min_val = df[self.factor_name].min()
                        max_val = df[self.factor_name].max()
                        if min_val == max_val:
                            df['group'] = np.random.randint(0, self.n_groups, size=len(df))
                        else:
                            df['group'] = pd.cut(df[self.factor_name], bins=self.n_groups, labels=range(self.n_groups), duplicates='drop')
                            if df['group'].nunique() < self.n_groups:
                                df['group'] = pd.qcut(np.arange(len(df)), q=self.n_groups, labels=range(self.n_groups))
                    return df
                
                latest_data = fac_group(latest_data)
        else:
            # 使用已处理数据中的最新日期
            latest_date = self.data['date'].max()
        latest_data = self.data[self.data['date'] == latest_date].copy()
        
        # 按组别排序
        latest_data = latest_data.sort_values(['group', self.factor_name], ascending=[True, False])
        
        # 选择需要的列
        latest_data = latest_data[['group', 'instrument', self.factor_name]]
        
        # 按组分组
        group_data = {}
        for i in range(self.n_groups):
            group_df = latest_data[latest_data['group'] == i].copy()
            # 将因子值格式化为四位小数
            group_df[self.factor_name] = group_df[self.factor_name].apply(lambda x: round(x, 4))
            group_data[i] = group_df[['instrument', self.factor_name]].values.tolist()
        
        return latest_date, group_data
    
    def calculate_hedged_returns_with_fees(self, fee_rate=0.0003):
        return self.calculate_hedged_returns_with_fees_sample_split(self.data, fee_rate, "全样本")

    def calculate_hedged_returns_with_fees_sample_split(self, data=None, fee_rate=0.0003, sample_type="全样本"):
        """
        计算考虑交易手续费的对冲收益曲线，支持样本内外分析
        
        参数:
        data: 指定数据，如果为None则使用self.data
        fee_rate: 单边交易手续费率，默认为万分之三(0.0003)
        sample_type: 样本类型标识
        
        返回:
        包含日期和累计收益的DataFrame
        """
        if data is None:
            data = self.data
            
        if 'future_ret' not in data.columns or len(data) == 0:
            print(f"警告: {sample_type}数据中没有future_ret列或数据为空，无法计算对冲收益曲线")
            return pd.DataFrame({'date': [], 'hedged_return': []}), {}, {}
        
        # 复制数据以避免修改原始数据
        df = data.copy()
        
        # 确定多空组合的构成
        if self.factor_direction == 1:
            # 因子值越高，预期收益越高
            long_group = df[df['group'] == self.n_groups - 1]
            short_group = df[df['group'] == 0]
        else:
            # 因子值越低，预期收益越高
            long_group = df[df['group'] == 0]
            short_group = df[df['group'] == self.n_groups - 1]
        
        # 计算每日多空组合收益 - 修正：转换为日收益率，并处理NaN值
        long_returns = long_group.groupby('date')['future_ret'].apply(lambda x: x.mean() if not x.isna().all() else np.nan)
        short_returns = short_group.groupby('date')['future_ret'].apply(lambda x: x.mean() if not x.isna().all() else np.nan)
        
        # 将N天收益率转换为等效日收益率
        long_returns_daily = long_returns / self.rebalance_period
        short_returns_daily = short_returns / self.rebalance_period
        
        # 确保索引对齐
        common_dates = sorted(list(set(long_returns_daily.index).intersection(set(short_returns_daily.index))))
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=common_dates)
        result.index.name = 'date'
        result['long_return'] = long_returns_daily.loc[common_dates]
        result['short_return'] = short_returns_daily.loc[common_dates]
        
        # 计算不含手续费的对冲收益（已经是日收益率）
        result['adjusted_hedged_return_no_fee'] = result['long_return'] - result['short_return']
        
        # 处理NaN值：将NaN替换为0，或者删除包含NaN的行
        # 方法1：删除包含NaN的行（推荐）
        initial_length = len(result)
        result = result.dropna()
        final_length = len(result)
        
        if initial_length != final_length:
            print(f"警告: {sample_type}删除了 {initial_length - final_length} 个包含NaN的日期数据点")
        
        if len(result) == 0:
            print(f"警告: {sample_type}所有数据都包含NaN，无法计算收益曲线")
            return pd.DataFrame({'date': [], 'hedged_return': []}), {}, {}
        
        # 计算换手率（滚动调仓策略下，每天的换手率是总仓位的1/调仓周期）
        result['adjusted_turnover'] = 1.0 / self.rebalance_period
        
        # 计算手续费
        result['adjusted_fee'] = result['adjusted_turnover'] * fee_rate * 2  # 买入和卖出
        
        # 计算含手续费的收益
        result['adjusted_hedged_return_with_fee'] = result['adjusted_hedged_return_no_fee'] - result['adjusted_fee']
        
        # 计算累计收益 - 使用单利而非复利
        result['cum_return_no_fee'] = result['adjusted_hedged_return_no_fee'].cumsum()
        result['cum_return_with_fee'] = result['adjusted_hedged_return_with_fee'].cumsum()
        
        # 计算年化收益率、夏普比率等指标
        days = len(result)
        if days == 0:
            print(f"警告: {sample_type}没有有效数据计算指标")
            return result, {}, {}
            
        total_days = (result.index[-1] - result.index[0]).days + 1
        
        # 修正年化收益率计算方式
        total_return_no_fee = result['cum_return_no_fee'].iloc[-1]
        total_return_with_fee = result['cum_return_with_fee'].iloc[-1]
        
        # 使用单利计算年化收益率
        if total_days > 0:
            annualized_return_no_fee = total_return_no_fee * (365 / total_days)
            annualized_return_with_fee = total_return_with_fee * (365 / total_days)
        else:
            annualized_return_no_fee = 0
            annualized_return_with_fee = 0
        
        result_no_fee = {
            'annualized_return': annualized_return_no_fee,
            'sharpe_ratio': result['adjusted_hedged_return_no_fee'].mean() / result['adjusted_hedged_return_no_fee'].std() * np.sqrt(365) if result['adjusted_hedged_return_no_fee'].std() != 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(result['cum_return_no_fee']),
            'win_rate': (result['adjusted_hedged_return_no_fee'] > 0).mean()
        }
        
        result_with_fee = {
            'annualized_return': annualized_return_with_fee,
            'sharpe_ratio': result['adjusted_hedged_return_with_fee'].mean() / result['adjusted_hedged_return_with_fee'].std() * np.sqrt(365) if result['adjusted_hedged_return_with_fee'].std() != 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(result['cum_return_with_fee']),
            'win_rate': (result['adjusted_hedged_return_with_fee'] > 0).mean()
        }
        
        return result, result_no_fee, result_with_fee

    def _calculate_max_drawdown(self, cum_returns):
        """
        计算最大回撤
        
        参数:
            cum_returns: 累计收益率序列（如0.5表示50%收益）
            
        返回:
            最大回撤百分比（0到1之间）
        """
        # 转换为净值序列 - 适用于单利计算
        nav = 1 + cum_returns
        
        # 计算累计最大净值
        running_max = np.maximum.accumulate(nav)
        
        # 计算回撤
        drawdowns = (running_max - nav) / running_max
        
        # 获取最大回撤
        max_drawdown = drawdowns.max()
        
        return max_drawdown
    
    def calculate_rank_ic_decay(self, max_lag=10):
        """
        计算Rank IC衰减图数据
        
        参数:
        max_lag: 最大滞后期，默认为10天
        
        返回:
        DataFrame包含lag和对应的rank_ic值
        """
        if 'future_ret' not in self.data.columns:
            print("警告: 数据中没有future_ret列，无法计算Rank IC衰减")
        
        decay_results = []
        
        for lag in range(1, max_lag + 1):
            # 计算每个滞后期的未来收益率
            lag_returns = []
            
            for symbol in self.data['instrument'].unique():
                symbol_data = self.data[self.data['instrument'] == symbol].copy()
                symbol_data = symbol_data.sort_values('date')
                
                # 计算lag天后的收益率
                symbol_data[f'future_ret_lag{lag}'] = symbol_data['future_ret'].shift(-lag)
                lag_returns.append(symbol_data[['date', 'instrument', self.factor_name, f'future_ret_lag{lag}']])
            
            # 合并所有symbol的数据
            lag_data = pd.concat(lag_returns, ignore_index=True)
            lag_data = lag_data.dropna()
            
            if len(lag_data) == 0:
                rank_ic = 0
            else:
                # 计算该滞后期的Rank IC
                rank_ic = lag_data[self.factor_name].corr(lag_data[f'future_ret_lag{lag}'], method='spearman')
                if pd.isna(rank_ic):
                    rank_ic = 0
            
            decay_results.append({'lag': lag, 'rank_ic': rank_ic})
        
        return pd.DataFrame(decay_results)
    
    def calculate_rank_ic_autocorr(self, max_lag=20):
        """
        计算Rank IC自相关性
        
        参数:
        max_lag: 最大滞后期，默认为20
        
        返回:
        DataFrame包含lag和对应的autocorr值
        """
        if 'future_ret' not in self.data.columns:
            print("警告: 数据中没有future_ret列，无法计算Rank IC自相关性")
            return pd.DataFrame({'lag': range(1, max_lag+1), 'autocorr': [0]*max_lag})
        
        # 按日期分组计算每日Rank IC值
        daily_rank_ic = self.data.groupby('date').apply(
            lambda x: x[self.factor_name].corr(x['future_ret'], method='spearman')
        ).reset_index()
        daily_rank_ic.columns = ['date', 'rank_ic']
        daily_rank_ic = daily_rank_ic.dropna()
        
        if len(daily_rank_ic) < max_lag + 1:
            print(f"警告: 数据长度不足，无法计算{max_lag}期自相关性")
            return pd.DataFrame({'lag': range(1, max_lag+1), 'autocorr': [0]*max_lag})
        
        autocorr_results = []
        rank_ic_series = daily_rank_ic['rank_ic']
        
        for lag in range(1, max_lag + 1):
            try:
                # 计算自相关系数
                autocorr = rank_ic_series.autocorr(lag=lag)
                if pd.isna(autocorr):
                    autocorr = 0
            except:
                autocorr = 0
            
            autocorr_results.append({'lag': lag, 'autocorr': autocorr})
        
        return pd.DataFrame(autocorr_results)
    
    def render(self):
        """
        直接生成静态HTML报告，只包含样本内外分析，不生成全样本图表
        """
        print(f'生成因子分析报告: {self.render_path}')
        
        # 导入必要的库
        from pyecharts.charts import Line
        from pyecharts import options as opts
        
        # =========================== 全样本分析 ===========================
        print("正在计算全样本指标...")
        
        # 加载多头和空头绩效情况
        if self.factor_direction == 1:
            performance_table_full = pd.concat([self.performance(self.n_groups - 1), self.performance(0)], axis=0).to_html(index=False)
        else:
            performance_table_full = pd.concat([self.performance(0), self.performance(self.n_groups - 1)], axis=0).to_html(index=False)
        
        # 计算含手续费的对冲收益曲线
        hedged_returns_full, stats_no_fee_full, stats_with_fee_full = self.calculate_hedged_returns_with_fees(fee_rate=0.0003)
        
        # 获取IC值和其他指标
        ic_value_full, _, ir_value_full, ic_ir_value_full, t_value_full, p_value_full, spearman_corr_full, rank_ic_value_full, daily_ic_full = self.IC()
        
        # =========================== 样本内分析 ===========================
        print("正在计算样本内指标...")
        
        # 样本内绩效
        if len(self.data_in_sample) > 0:
            if self.factor_direction == 1:
                performance_table_in = pd.concat([
                    self.performance_with_sample_split(self.n_groups - 1, self.data_in_sample, "样本内"), 
                    self.performance_with_sample_split(0, self.data_in_sample, "样本内")
                ], axis=0).to_html(index=False)
            else:
                performance_table_in = pd.concat([
                    self.performance_with_sample_split(0, self.data_in_sample, "样本内"), 
                    self.performance_with_sample_split(self.n_groups - 1, self.data_in_sample, "样本内")
                ], axis=0).to_html(index=False)
            
            # 样本内图表
            line_in1, line_in2, line_in3, line_in4, line_in5, line_in6, line_in7, line_in8 = self.show_plot_for_sample(self.data_in_sample, "样本内")
            
            # 样本内对冲收益
            hedged_returns_in, stats_no_fee_in, stats_with_fee_in = self.calculate_hedged_returns_with_fees_sample_split(
                self.data_in_sample, fee_rate=0.0003, sample_type="样本内"
            )
            
            # 样本内IC指标
            ic_value_in, _, ir_value_in, ic_ir_value_in, t_value_in, p_value_in, spearman_corr_in, rank_ic_value_in, daily_ic_in = self.IC_with_sample_split(
                self.data_in_sample, "样本内"
            )
        else:
            performance_table_in = "<p>样本内数据不足</p>"
            # 创建空图表
            line_in1 = line_in2 = line_in3 = line_in4 = line_in5 = line_in6 = line_in7 = line_in8 = Line()
            hedged_returns_in = pd.DataFrame()
            stats_no_fee_in = stats_with_fee_in = {}
            ic_value_in = ir_value_in = ic_ir_value_in = t_value_in = p_value_in = spearman_corr_in = rank_ic_value_in = 0
            daily_ic_in = pd.DataFrame()
        
        # =========================== 样本外分析 ===========================
        print("正在计算样本外指标...")
        
        # 样本外绩效
        if len(self.data_out_sample) > 0:
            if self.factor_direction == 1:
                performance_table_out = pd.concat([
                    self.performance_with_sample_split(self.n_groups - 1, self.data_out_sample, "样本外"), 
                    self.performance_with_sample_split(0, self.data_out_sample, "样本外")
                ], axis=0).to_html(index=False)
            else:
                performance_table_out = pd.concat([
                    self.performance_with_sample_split(0, self.data_out_sample, "样本外"), 
                    self.performance_with_sample_split(self.n_groups - 1, self.data_out_sample, "样本外")
                ], axis=0).to_html(index=False)
            
            # 样本外图表
            line_out1, line_out2, line_out3, line_out4, line_out5, line_out6, line_out7, line_out8 = self.show_plot_for_sample(self.data_out_sample, "样本外")
            
            # 样本外对冲收益
            hedged_returns_out, stats_no_fee_out, stats_with_fee_out = self.calculate_hedged_returns_with_fees_sample_split(
                self.data_out_sample, fee_rate=0.0003, sample_type="样本外"
            )
            
            # 样本外IC指标
            ic_value_out, _, ir_value_out, ic_ir_value_out, t_value_out, p_value_out, spearman_corr_out, rank_ic_value_out, daily_ic_out = self.IC_with_sample_split(
                self.data_out_sample, "样本外"
            )
        else:
            performance_table_out = "<p>样本外数据不足</p>"
            # 创建空图表
            line_out1 = line_out2 = line_out3 = line_out4 = line_out5 = line_out6 = line_out7 = line_out8 = Line()
            hedged_returns_out = pd.DataFrame()
            stats_no_fee_out = stats_with_fee_out = {}
            ic_value_out = ir_value_out = ic_ir_value_out = t_value_out = p_value_out = spearman_corr_out = rank_ic_value_out = 0
            daily_ic_out = pd.DataFrame()
        
        # =========================== 样本内外对比图表 ===========================
        
        # 样本内外对比图表 - 正确的版本
        line7 = Line()
        
        # 准备数据
        in_sample_dates = []
        in_sample_returns = []
        out_sample_dates = []
        out_sample_returns = []
        
        # 处理样本内数据
        if not hedged_returns_in.empty:
            in_sample_dates = hedged_returns_in.index.strftime('%Y-%m-%d').tolist()
            in_sample_returns = hedged_returns_in['cum_return_no_fee'].tolist()
        
        # 处理样本外数据
        if not hedged_returns_out.empty:
            out_sample_dates = hedged_returns_out.index.strftime('%Y-%m-%d').tolist()
            out_sample_returns = hedged_returns_out['cum_return_no_fee'].tolist()
            
            # 如果有样本内数据，样本外收益需要从样本内最后一点开始（单利累加）
            if in_sample_returns:
                last_in_sample_return = in_sample_returns[-1]
                # 对于单利，直接加上样本内最后一个收益点的值
                out_sample_returns = [last_in_sample_return + ret for ret in out_sample_returns]
        
        # 创建完整的时间轴（所有日期）
        all_dates = []
        if in_sample_dates:
            all_dates.extend(in_sample_dates)
        if out_sample_dates:
            all_dates.extend(out_sample_dates)
        
        if all_dates:
            line7.add_xaxis(all_dates)
            
            # 添加样本内数据
            if in_sample_dates and in_sample_returns:
                # 样本内数据只在样本内期间有值
                full_in_sample_data = in_sample_returns + [None] * len(out_sample_dates)
                line7.add_yaxis(
                    '样本内表现', 
                    full_in_sample_data,
                    is_symbol_show=False,
                    linestyle_opts=opts.LineStyleOpts(width=2, color="#1f77b4"),
                    label_opts=opts.LabelOpts(is_show=False)
                )
            
            # 添加样本外数据  
            if out_sample_dates and out_sample_returns:
                # 样本外数据只在样本外期间有值
                full_out_sample_data = [None] * len(in_sample_dates) + out_sample_returns
                line7.add_yaxis(
                    '样本外表现', 
                    full_out_sample_data,
                    is_symbol_show=False,
                    linestyle_opts=opts.LineStyleOpts(width=2, color="#166534", type_="dashed"),
                    label_opts=opts.LabelOpts(is_show=False)
                )
        
        line7.set_global_opts(
            title_opts=opts.TitleOpts(title="样本内外对冲收益对比"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="5%"),
            xaxis_opts=opts.AxisOpts(type_="time"),
            yaxis_opts=opts.AxisOpts(
                name="累计收益",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            ),
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside")
            ]
        )
        
        # 叠加市值50等权指数（因子池版）
        try:
            import os
            mc50_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'index_cache_equal_weight_30_50_daily.csv')
            if os.path.exists(mc50_path):
                mc50 = pd.read_csv(mc50_path)
                mc50['date'] = pd.to_datetime(mc50['date'])
                mc50['cum_return'] = mc50['index_value'] - 1.0

                # 限定到all_dates范围
                if all_dates:
                    all_dates_dt = pd.to_datetime(all_dates)
                    start_dt, end_dt = all_dates_dt.min(), all_dates_dt.max()
                    mc50_filtered = mc50[(mc50['date'] >= start_dt) & (mc50['date'] <= end_dt)].copy()
                    
                    if len(mc50_filtered) > 0:
                        # 构建调仓日到累计收益的映射
                        rebalance_map = dict(zip(mc50_filtered['date'], mc50_filtered['cum_return']))
                        rebalance_dates = sorted(rebalance_map.keys())
                        
                        # 对于每个交易日，使用最近一次调仓日的指数值（向前填充）
                        mc50_series = []
                        for current_date_str in all_dates:
                            current_date = pd.to_datetime(current_date_str)
                            # 找到小于等于当前日期的最近一次调仓日
                            closest_rebalance_date = None
                            for rb_date in reversed(rebalance_dates):
                                if rb_date <= current_date:
                                    closest_rebalance_date = rb_date
                                    break
                            
                            # 如果找到调仓日，使用其指数值；否则为None
                            if closest_rebalance_date is not None:
                                mc50_series.append(rebalance_map[closest_rebalance_date])
                            else:
                                mc50_series.append(None)

                        line7.add_yaxis(
                            '市值50等权指数(因子池)',
                            mc50_series,
                            is_symbol_show=False,
                            linestyle_opts=opts.LineStyleOpts(width=2, color="#F97316")
                        )
                    else:
                        print(f"警告: 市值50指数数据在日期范围内为空")
            else:
                print(f"警告: 未找到市值50指数文件: {mc50_path}")
        except Exception as e:
            print(f"叠加市值50指数失败: {e}")
        
        # =========================== 创建对比表格 ===========================
        
        # 综合指标对比表格 - 包含IC指标和对冲收益指标
        def _calc_dd_metrics_from_returns(df_returns):
            """
            从对冲日收益序列计算最大回撤期与修复期（单位: 天）。
            返回 (drawdown_days, recovery_days)；若不可计算则返回(None, None)。
            """
            try:
                if df_returns is None or len(df_returns) == 0:
                    return None, None
                returns = df_returns['adjusted_hedged_return_no_fee']
                if returns.isna().all() or len(returns) <= 1:
                    return None, None
                nav = (1 + returns).cumprod()
                nav_values = nav.values.astype(float)
                running_max = np.maximum.accumulate(nav_values)
                drawdowns = (running_max - nav_values) / running_max
                trough_idx = int(np.argmax(drawdowns))
                if trough_idx <= 0:
                    return 0, 0
                peak_idx = int(np.argmax(nav_values[:trough_idx+1]))
                peak_value = nav_values[peak_idx]
                idx_dates = pd.to_datetime(nav.index)
                drawdown_days = (idx_dates[trough_idx] - idx_dates[peak_idx]).days
                recovery_idx = None
                for i in range(trough_idx + 1, len(nav_values)):
                    if nav_values[i] >= peak_value:
                        recovery_idx = i
                        break
                if recovery_idx is not None:
                    recovery_days = (idx_dates[recovery_idx] - idx_dates[trough_idx]).days
                else:
                    recovery_days = None
                return int(max(drawdown_days, 0)), (int(recovery_days) if recovery_days is not None else None)
            except Exception:
                return None, None

        dd_days_full, rec_days_full = _calc_dd_metrics_from_returns(hedged_returns_full if isinstance(hedged_returns_full, pd.DataFrame) and not hedged_returns_full.empty else None)
        dd_days_in, rec_days_in = _calc_dd_metrics_from_returns(hedged_returns_in if 'hedged_returns_in' in locals() and isinstance(hedged_returns_in, pd.DataFrame) and not hedged_returns_in.empty else None)
        dd_days_out, rec_days_out = _calc_dd_metrics_from_returns(hedged_returns_out if 'hedged_returns_out' in locals() and isinstance(hedged_returns_out, pd.DataFrame) and not hedged_returns_out.empty else None)

        # 全样本
        decay_full = self.calculate_rank_ic_decay(max_lag=20)
        halflife_full = self.calc_rankic_halflife(decay_full, threshold=0.5)

        # 样本内
        decay_in = self.calculate_rank_ic_decay_for_sample(self.data_in_sample, max_lag=20) if len(self.data_in_sample)>0 else None
        halflife_in = self.calc_rankic_halflife(decay_in, threshold=0.5) if decay_in is not None else None

        # 样本外
        decay_out = self.calculate_rank_ic_decay_for_sample(self.data_out_sample, max_lag=20) if len(self.data_out_sample)>0 else None
        halflife_out = self.calc_rankic_halflife(decay_out, threshold=0.5) if decay_out is not None else None

        # 计算IC概率
        def calculate_ic_probabilities(daily_ic_df):
            """计算IC值小于-0.02和大于0.02的概率"""
            if daily_ic_df.empty or 'ic' not in daily_ic_df.columns:
                return "N/A", "N/A"
            ic_values = daily_ic_df['ic'].dropna()
            if len(ic_values) == 0:
                return "N/A", "N/A"
            p_ic_neg = (ic_values < -0.02).mean()
            p_ic_pos = (ic_values > 0.02).mean()
            return f"{p_ic_neg:.1%}", f"{p_ic_pos:.1%}"

        p_ic_neg_full, p_ic_pos_full = calculate_ic_probabilities(daily_ic_full)
        p_ic_neg_in, p_ic_pos_in = calculate_ic_probabilities(daily_ic_in)
        p_ic_neg_out, p_ic_pos_out = calculate_ic_probabilities(daily_ic_out)

        ic_comparison_table = pd.DataFrame({
            'IC': [
                f"{ic_value_full:.4f}",
                f"{ic_value_in:.4f}",
                f"{ic_value_out:.4f}"
            ],
            'Rank IC': [
                f"{rank_ic_value_full:.4f}",
                f"{rank_ic_value_in:.4f}",
                f"{rank_ic_value_out:.4f}"
            ],
            'IR': [
                f"{ir_value_full:.4f}",
                f"{ir_value_in:.4f}",
                f"{ir_value_out:.4f}"
            ],
            'IC_IR': [
                f"{ic_ir_value_full:.4f}",
                f"{ic_ir_value_in:.4f}",
                f"{ic_ir_value_out:.4f}"
            ],
            'RankIC Half-Life': [
                halflife_full if halflife_full is not None else 'N/A',
                halflife_in if halflife_in is not None else 'N/A',
                halflife_out if halflife_out is not None else 'N/A',
            ],
            'P(IC<-0.02)': [
                p_ic_neg_full,
                p_ic_neg_in,
                p_ic_neg_out
            ],
            'P(IC>0.02)': [
                p_ic_pos_full,
                p_ic_pos_in,
                p_ic_pos_out
            ],
            't值': [
                f"{t_value_full:.4f}",
                f"{t_value_in:.4f}",
                f"{t_value_out:.4f}"
            ],
            'p值': [
                f"{p_value_full:.4f}" if p_value_full >= 0.0001 else "<0.0001",
                f"{p_value_in:.4f}" if p_value_in >= 0.0001 else "<0.0001",
                f"{p_value_out:.4f}" if p_value_out >= 0.0001 else "<0.0001"
            ],
            '单调性': [
                f"{spearman_corr_full:.4f}",
                f"{spearman_corr_in:.4f}",
                f"{spearman_corr_out:.4f}"
            ],
            '年化收益率': [
                f"{stats_no_fee_full['annualized_return']:.2%}" if stats_no_fee_full else "N/A",
                f"{stats_no_fee_in['annualized_return']:.2%}" if stats_no_fee_in else "N/A",
                f"{stats_no_fee_out['annualized_return']:.2%}" if stats_no_fee_out else "N/A"
            ],
            '夏普比率': [
                f"{stats_no_fee_full['sharpe_ratio']:.2f}" if stats_no_fee_full else "N/A",
                f"{stats_no_fee_in['sharpe_ratio']:.2f}" if stats_no_fee_in else "N/A",
                f"{stats_no_fee_out['sharpe_ratio']:.2f}" if stats_no_fee_out else "N/A"
            ],
            '最大回撤': [
                f"{stats_no_fee_full['max_drawdown']:.2%}" if stats_no_fee_full else "N/A",
                f"{stats_no_fee_in['max_drawdown']:.2%}" if stats_no_fee_in else "N/A",
                f"{stats_no_fee_out['max_drawdown']:.2%}" if stats_no_fee_out else "N/A"
            ],
            '最大回撤期(天)': [
                f"{dd_days_full}" if dd_days_full is not None else "N/A",
                f"{dd_days_in}" if dd_days_in is not None else "N/A",
                f"{dd_days_out}" if dd_days_out is not None else "N/A"
            ],
            '修复期(天)': [
                f"{rec_days_full}" if rec_days_full is not None else "N/A",
                f"{rec_days_in}" if rec_days_in is not None else "N/A",
                f"{rec_days_out}" if rec_days_out is not None else "N/A"
            ],
            '胜率': [
                f"{stats_no_fee_full['win_rate']:.2%}" if stats_no_fee_full else "N/A",
                f"{stats_no_fee_in['win_rate']:.2%}" if stats_no_fee_in else "N/A",
                f"{stats_no_fee_out['win_rate']:.2%}" if stats_no_fee_out else "N/A"
            ]
        }, index=['全样本', '样本内', '样本外']).to_html()
        
        # 对冲收益指标对比表格 - 同样行列对换
        hedge_comparison_table = pd.DataFrame({
            '年化收益率': [
                f"{stats_no_fee_full['annualized_return']:.2%}" if stats_no_fee_full else "N/A",
                f"{stats_no_fee_in['annualized_return']:.2%}" if stats_no_fee_in else "N/A",
                f"{stats_no_fee_out['annualized_return']:.2%}" if stats_no_fee_out else "N/A"
            ],
            '夏普比率': [
                f"{stats_no_fee_full['sharpe_ratio']:.2f}" if stats_no_fee_full else "N/A",
                f"{stats_no_fee_in['sharpe_ratio']:.2f}" if stats_no_fee_in else "N/A",
                f"{stats_no_fee_out['sharpe_ratio']:.2f}" if stats_no_fee_out else "N/A"
            ],
            '最大回撤': [
                f"{stats_no_fee_full['max_drawdown']:.2%}" if stats_no_fee_full else "N/A",
                f"{stats_no_fee_in['max_drawdown']:.2%}" if stats_no_fee_in else "N/A",
                f"{stats_no_fee_out['max_drawdown']:.2%}" if stats_no_fee_out else "N/A"
            ],
            '胜率': [
                f"{stats_no_fee_full['win_rate']:.2%}" if stats_no_fee_full else "N/A",
                f"{stats_no_fee_in['win_rate']:.2%}" if stats_no_fee_in else "N/A",
                f"{stats_no_fee_out['win_rate']:.2%}" if stats_no_fee_out else "N/A"
            ]
        }, index=['全样本(不含费)', '样本内(不含费)', '样本外(不含费)']).to_html()
        
        # 获取最新一期的分组数据
        latest_date, group_symbols = self.get_latest_group_symbols(include_incomplete_data=True)
        
        # =========================== 生成HTML ===========================
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.factor_name} 因子分析报告 ({self.n_groups}组, {self.rebalance_period}天调仓) - 样本内外分析</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; margin-bottom: 10px; }}
        .section {{ margin-bottom: 20px; }}
        .highlight {{ background-color: #f0f8ff; padding: 10px; border-left: 4px solid #1f77b4; margin-bottom: 20px; }}
        .warning {{ background-color: #fff8dc; padding: 10px; border-left: 4px solid #ffa500; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        table, th, td {{ border: 1px solid #ddd; }}
        th, td {{ padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart {{ width: 100%; height: 400px; margin-top: 0; }}
        .chart-container {{ margin-top: 5px; }}
        .comparison-section {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .groups-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            width: 100%;
            justify-content: flex-start;
            align-items: flex-start;
        }}
        .group-box {{
            flex: 1 1 300px;
            min-width: 250px;
            max-width: 350px;
            margin: 0;
            box-sizing: border-box;
            border: 1px solid #eee;
            padding: 10px;
            background-color: #fafafa;
        }}
    </style>
</head>
<body>
    <h1>{self.factor_name} 因子分析报告 ({self.n_groups}组, {self.rebalance_period}天调仓)</h1>
    
    <div class="highlight">
        <h3>📊 样本分割信息</h3>
        <strong>样本内:</strong> {self.in_sample_dates} 天 (至 {self.split_date.strftime('%Y-%m-%d')})
        <strong>样本外:</strong> {self.out_sample_dates} 天 (从 {(self.split_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')})
        <strong>样本外天数设定:</strong> {self.out_of_sample_days} 天
    </div>
    
    <div class="comparison-section">
        <h2>📈 样本内外核心指标对比</h2>
        
        <h3>综合指标对比</h3>
        {ic_comparison_table}
        
        <div class="section">
            <h3>样本内外收益对比图</h3>
            <div class="chart-container">
                <div id="chart7" class="chart"></div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📋 全样本分析结果</h2>
        
        <h3>多空组合绩效 (全样本)</h3>
        {performance_table_full}
    </div>
    
    <div class="section">
        <h2>📋 样本内分析结果</h2>
        <h3>多空组合绩效 (样本内)</h3>
        {performance_table_in}
        
        <div class="section">
            <h3>分组收益曲线 (样本内)</h3>
            <div class="chart-container">
                <div id="chart_in1" class="chart"></div>
            </div>
        </div>
        
        <div class="section">
            <h3>累计IC曲线 (样本内)</h3>
            <div class="chart-container">
                <div id="chart_in2" class="chart"></div>
            </div>
        </div>
        
        <div class="section">
            <h3>累计Rank IC曲线 (样本内)</h3>
            <div class="chart-container">
                <div id="chart_in3" class="chart"></div>
            </div>
        </div>
        
        <div class="section">
            <h3>Rank IC衰减图 (样本内)</h3>
            <div class="chart-container">
                <div id="chart_in4" class="chart"></div>
            </div>
        </div>
        
        <div class="section">
            <h3>Rank IC自相关图 (样本内)</h3>
            <div class="chart-container">
                <div id="chart_in5" class="chart"></div>
            </div>
        </div>
        
        <div class="section">
            <h3>对冲收益曲线 (样本内)</h3>
            <div class="chart-container">
                <div id="chart_in6" class="chart"></div>
            </div>
        </div>

        <div class="section">
            <h3>IC分布图 (样本内)</h3>
            <div class="chart-container">
                <div id="chart_in7" class="chart"></div>
            </div>
        </div>

        <div class="section">
            <h3>Rank IC分布图 (样本内)</h3>
            <div class="chart-container">
                <div id="chart_in8" class="chart"></div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📋 样本外分析结果</h2>
        <h3>多空组合绩效 (样本外)</h3>
        {performance_table_out}
        
        <div class="section">
            <h3>分组收益曲线 (样本外)</h3>
            <div class="chart-container">
                <div id="chart_out1" class="chart"></div>
            </div>
        </div>
        
        <div class="section">
            <h3>累计IC曲线 (样本外)</h3>
            <div class="chart-container">
                <div id="chart_out2" class="chart"></div>
            </div>
        </div>
        
        <div class="section">
            <h3>累计Rank IC曲线 (样本外)</h3>
            <div class="chart-container">
                <div id="chart_out3" class="chart"></div>
            </div>
        </div>
        
        <div class="section">
            <h3>Rank IC衰减图 (样本外)</h3>
            <div class="chart-container">
                <div id="chart_out4" class="chart"></div>
            </div>
        </div>
        
        <div class="section">
            <h3>Rank IC自相关图 (样本外)</h3>
            <div class="chart-container">
                <div id="chart_out5" class="chart"></div>
            </div>
        </div>
        
        <div class="section">
            <h3>对冲收益曲线 (样本外)</h3>
            <div class="chart-container">
                <div id="chart_out6" class="chart"></div>
            </div>
        </div>

        <div class="section">
            <h3>IC分布图 (样本外)</h3>
            <div class="chart-container">
                <div id="chart_out7" class="chart"></div>
            </div>
        </div>

        <div class="section">
            <h3>Rank IC分布图 (样本外)</h3>
            <div class="chart-container">
                <div id="chart_out8" class="chart"></div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 最新一期分组币种 ({latest_date.strftime('%Y-%m-%d')})</h2>
        <div class="groups-row">
"""
        
        # 为每个分组生成HTML
        for i in range(self.n_groups):
            # 设置分组名称
            if i == 0 and self.factor_direction == -1:
                group_name = f"Group {i} (Long)"
            elif i == 0 and self.factor_direction == 1:
                group_name = f"Group {i} (Short)"
            elif i == self.n_groups - 1 and self.factor_direction == -1:
                group_name = f"Group {i} (Short)"
            elif i == self.n_groups - 1 and self.factor_direction == 1:
                group_name = f"Group {i} (Long)"
            else:
                group_name = f"Group {i}"
            
            # 添加分组框
            html_content += f'<div class="group-box"><h3>{group_name}</h3>'
            
            # 添加表格头部
            html_content += '<table><tr><th>Symbol</th><th>Factor Value</th></tr>'
            
            # 检查当前组是否有数据
            if i in group_symbols and group_symbols[i]:
                # 添加每个币种的行
                for symbol, factor_value in group_symbols[i]:
                    html_content += f'<tr><td>{symbol}</td><td>{factor_value}</td></tr>'
            else:
                # 如果没有数据，显示提示
                html_content += '<tr><td colspan="2">No data available</td></tr>'
            
            # 关闭表格和分组框
            html_content += '</table></div>'
        
        # 完成HTML内容，添加图表初始化脚本
        html_content += """
        </div>
    </div>
    <script type="text/javascript">
"""
        
        # 添加图表初始化脚本
        chart7_options = line7.dump_options_with_quotes()
        
        # 样本内图表
        chart_in1_options = line_in1.dump_options_with_quotes()
        chart_in2_options = line_in2.dump_options_with_quotes()
        chart_in3_options = line_in3.dump_options_with_quotes()
        chart_in4_options = line_in4.dump_options_with_quotes()
        chart_in5_options = line_in5.dump_options_with_quotes()
        chart_in6_options = line_in6.dump_options_with_quotes()
        chart_in7_options = line_in7.dump_options_with_quotes()
        chart_in8_options = line_in8.dump_options_with_quotes()
        
        # 样本外图表
        chart_out1_options = line_out1.dump_options_with_quotes()
        chart_out2_options = line_out2.dump_options_with_quotes()
        chart_out3_options = line_out3.dump_options_with_quotes()
        chart_out4_options = line_out4.dump_options_with_quotes()
        chart_out5_options = line_out5.dump_options_with_quotes()
        chart_out6_options = line_out6.dump_options_with_quotes()
        chart_out7_options = line_out7.dump_options_with_quotes()
        chart_out8_options = line_out8.dump_options_with_quotes()
        
        html_content += f"""
        var chart7 = echarts.init(document.getElementById('chart7'));
        chart7.setOption({chart7_options});
        
        // 样本内图表
        var chart_in1 = echarts.init(document.getElementById('chart_in1'));
        chart_in1.setOption({chart_in1_options});
        
        var chart_in2 = echarts.init(document.getElementById('chart_in2'));
        chart_in2.setOption({chart_in2_options});

        var chart_in3 = echarts.init(document.getElementById('chart_in3'));
        chart_in3.setOption({chart_in3_options});
        
        var chart_in4 = echarts.init(document.getElementById('chart_in4'));
        chart_in4.setOption({chart_in4_options});
        
        var chart_in5 = echarts.init(document.getElementById('chart_in5'));
        chart_in5.setOption({chart_in5_options});

        var chart_in6 = echarts.init(document.getElementById('chart_in6'));
        chart_in6.setOption({chart_in6_options});

        var chart_in7 = echarts.init(document.getElementById('chart_in7'));
        chart_in7.setOption({chart_in7_options});

        var chart_in8 = echarts.init(document.getElementById('chart_in8'));
        chart_in8.setOption({chart_in8_options});
        
        // 样本外图表
        var chart_out1 = echarts.init(document.getElementById('chart_out1'));
        chart_out1.setOption({chart_out1_options});
        
        var chart_out2 = echarts.init(document.getElementById('chart_out2'));
        chart_out2.setOption({chart_out2_options});

        var chart_out3 = echarts.init(document.getElementById('chart_out3'));
        chart_out3.setOption({chart_out3_options});
        
        var chart_out4 = echarts.init(document.getElementById('chart_out4'));
        chart_out4.setOption({chart_out4_options});
        
        var chart_out5 = echarts.init(document.getElementById('chart_out5'));
        chart_out5.setOption({chart_out5_options});

        var chart_out6 = echarts.init(document.getElementById('chart_out6'));
        chart_out6.setOption({chart_out6_options});

        var chart_out7 = echarts.init(document.getElementById('chart_out7'));
        chart_out7.setOption({chart_out7_options});

        var chart_out8 = echarts.init(document.getElementById('chart_out8'));
        chart_out8.setOption({chart_out8_options});
    </script>
</body>
</html>
"""
        
        # 保存HTML文件
        if self.render_path:
            with open(self.render_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ 因子分析报告(含样本内外对比)已保存至: {self.render_path}")

    def show_plot_for_sample(self, data, sample_type="全样本"):
        """
        为指定样本数据生成图表
        :param data: 样本数据
        :param sample_type: 样本类型标识，用于图表标题
        """
        from pyecharts.charts import Line, Bar
        from pyecharts import options as opts

        # 假设我们有future_ret列
        if 'future_ret' not in data.columns or len(data) == 0:
            print(f"警告: {sample_type}数据中没有future_ret列或数据为空，无法绘制收益曲线")
            # 创建空的图表
            line = Line()
            line.add_xaxis([])
            line.add_yaxis("无数据", [])
            
            line2 = Line()
            line2.add_xaxis([])
            line2.add_yaxis("无数据", [])
            
            line3 = Line() # 累计Rank IC图
            line3.add_xaxis([])
            line3.add_yaxis("无数据", [])

            line4 = Line() # Rank IC衰减图
            line4.add_xaxis([])
            line4.add_yaxis("无数据", [])

            line5 = Line() # Rank IC自相关图
            line5.add_xaxis([])
            line5.add_yaxis("无数据", [])
            
            line6 = Line() # 对冲收益图
            line6.add_xaxis([])
            line6.add_yaxis("无数据", [])
            
            line7 = Line() # IC分布图
            line7.add_xaxis([])
            line7.add_yaxis("无数据", [])

            line8 = Line() # Rank IC分布图
            line8.add_xaxis([])
            line8.add_yaxis("无数据", [])

            return line, line2, line3, line4, line5, line6, line7, line8
            
        # 计算分组累计收益 - 修正：将N天收益率转换为日收益率
        df = data[['date', 'group', 'future_ret']].copy()
        df = df.groupby(['date', 'group'], as_index=False)['future_ret'].mean()
        df['ret'] = df['future_ret'] / self.rebalance_period
        df = pd.pivot_table(df, values=['ret'], index='date', columns='group').cumsum()
        
        # 确保列名是正确的格式
        df.columns = [col[1] for col in df.columns]
        
        # 确保所有分组都有数据
        for i in range(self.n_groups):
            if i not in df.columns:
                print(f"警告: {sample_type}分组 {i} 没有数据，添加零值序列")
                df[i] = 0.0
        
        # 重新排序列，确保按照0,1,2,...,n_groups-1的顺序
        df = df.reindex(sorted(df.columns), axis=1)
        
        columns = list(df.columns)

        # 求多空组合累计收益曲线
        if self.factor_direction == 1:
            df['long_short'] = df[columns[-1]] - df[columns[0]]
        else:
            df['long_short'] = df[columns[0]] - df[columns[-1]]

        # 绘制折线图
        x = list(df.index)
        line = Line()
        line.add_xaxis(x)
        
        # 确保所有分组都被添加到图表中
        for i in range(self.n_groups):
            if i in df.columns:
                # 使用更明确的分组标签格式
                group_label = f"Group {i}"
                line.add_yaxis(
                    group_label, 
                    df[i].tolist(),
                    is_symbol_show=False,
                    linestyle_opts=opts.LineStyleOpts(width=2)
                )
        
        # 添加多空组合曲线
        line.add_yaxis(
            "Long-Short", 
            df['long_short'].tolist(),
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=3, type_="dashed")
        )
        
        # 设置图表选项
        line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 隐藏数字
        line.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{sample_type}分组收益曲线"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="5%"),
            xaxis_opts=opts.AxisOpts(type_="time"),
            yaxis_opts=opts.AxisOpts(
                name="累计收益",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            ),
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside")
            ]
        )

        # 绘制累计IC图 - 修正：使用IC方法返回的累计IC数据
        _, acc_ic, _, _, _, _, _, _, _ = self.IC_with_sample_split(data, sample_type)
        x = list(acc_ic['date'])
        y = list(acc_ic['acc_ic'])
        line2 = Line()
        line2.add_xaxis(x)
        line2.add_yaxis(
            '累计IC', 
            y, 
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2, color="#ffbe0b")
        )
        line2.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{sample_type}累计IC曲线"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(type_="time"),
            yaxis_opts=opts.AxisOpts(
                name="累计IC值",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            ),
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside")
            ]
        )
        
        # 绘制累计Rank IC图
        # 计算累计Rank IC
        daily_rank_ic = data.groupby('date').apply(
            lambda x: x[self.factor_name].corr(x['future_ret'], method='spearman')
        ).reset_index()
        daily_rank_ic.columns = ['date', 'rank_ic']
        daily_rank_ic = daily_rank_ic.dropna()
        daily_rank_ic['cum_rank_ic'] = daily_rank_ic['rank_ic'].cumsum()
        
        x3 = list(daily_rank_ic['date'])
        y3 = list(daily_rank_ic['cum_rank_ic'])
        line3 = Line()
        line3.add_xaxis(x3)
        line3.add_yaxis(
            '累计Rank IC',
            y3,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2, color="#ff7f0e")
        )
        line3.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{sample_type}累计Rank IC曲线"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(type_="time"),
            yaxis_opts=opts.AxisOpts(
                name="累计Rank IC值",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            ),
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside")
            ]
        )

        # 绘制Rank IC衰减图
        decay_data = self.calculate_rank_ic_decay_for_sample(data, max_lag=10)
        x4 = [f"Lag{lag}" for lag in decay_data['lag']]
        y4 = decay_data['rank_ic'].tolist()
        
        line4 = Bar()  # 使用柱状图更直观
        line4.add_xaxis(x4)
        line4.add_yaxis(
            "Rank IC", 
            y4,
            itemstyle_opts=opts.ItemStyleOpts(color="#3a86ff")
        )
        line4.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{sample_type}Rank IC衰减图"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(name="滞后期"),
            yaxis_opts=opts.AxisOpts(
                name="Rank IC值",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            )
        )

        # 绘制Rank IC自相关图
        autocorr_data = self.calculate_rank_ic_autocorr_for_sample(data, max_lag=20)
        x5 = [f"Lag{lag}" for lag in autocorr_data['lag']]
        y5 = autocorr_data['autocorr'].tolist()
        
        line5 = Line()
        line5.add_xaxis(x5)
        line5.add_yaxis(
            "自相关系数", 
            y5,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2, color="#ff6b6b"),
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(y=0)]  # 添加0线参考
            )
        )
        line5.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{sample_type}Rank IC自相关图"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(name="滞后期"),
            yaxis_opts=opts.AxisOpts(
                name="自相关系数",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            )
        )
        
        # 绘制对冲收益图
        hedged_returns, _, _ = self.calculate_hedged_returns_with_fees_sample_split(data, fee_rate=0.0003, sample_type=sample_type)
        line6 = Line()
        if not hedged_returns.empty:
            line6.add_xaxis(hedged_returns.index.strftime('%Y-%m-%d').tolist())
            line6.add_yaxis(
                '不含手续费', 
                hedged_returns['cum_return_no_fee'].tolist(),
                is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(width=2)
            )
            line6.add_yaxis(
                '含手续费(万分之三)', 
                hedged_returns['cum_return_with_fee'].tolist(),
                is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(width=2, type_="dashed")
            )
        line6.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{sample_type}对冲收益曲线(含/不含手续费)"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="5%"),
            xaxis_opts=opts.AxisOpts(type_="time"),
            yaxis_opts=opts.AxisOpts(
                name="累计收益",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            ),
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside")
            ]
        )
        
        # 计算IC和Rank IC的分布
        _, _, _, _, _, _, _, _, daily_ic_data = self.IC_with_sample_split(data, sample_type)

        # 绘制IC分布图
        from pyecharts.charts import Bar, Line
        ic_chart = Bar()
        if not daily_ic_data.empty and 'ic' in daily_ic_data.columns:
            ic_values = daily_ic_data['ic'].dropna()
            if len(ic_values) > 0:
                # 计算统计量
                ic_skew = ic_values.skew()
                ic_kurt = ic_values.kurtosis()

                # 直方图数据
                hist, bin_edges = np.histogram(ic_values, bins=20, density=False)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 0.05
                bin_labels = [f"{bin_edges[i]:.4f} ~ {bin_edges[i+1]:.4f}" for i in range(len(bin_edges) - 1)]

                ic_chart.add_xaxis(bin_labels)
                ic_chart.add_yaxis(
                    "Histogram",
                    hist.tolist(),
                    category_gap="0%",
                    bar_width="95%",
                    itemstyle_opts=opts.ItemStyleOpts(color="#6baed6", opacity=0.7),
                    label_opts=opts.LabelOpts(is_show=False)
                )

                # 密度曲线（缩放到频数尺度）
                from scipy import stats
                kde = stats.gaussian_kde(ic_values)
                density_counts = (kde(bin_centers) * len(ic_values) * bin_width).tolist()
                line_density = Line()
                line_density.add_xaxis(bin_labels)
                line_density.add_yaxis(
                    "Density Curve",
                    density_counts,
                    is_symbol_show=False,
                    is_smooth=True,
                    linestyle_opts=opts.LineStyleOpts(width=2.2, color="#1f77ff"),
                    areastyle_opts=opts.AreaStyleOpts(opacity=0.08, color="#1f77ff"),
                    label_opts=opts.LabelOpts(is_show=False)
                )

                ic_chart.set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"{sample_type} IC分布图 (skew={ic_skew:.3f}, kurt={ic_kurt:.3f})",
                        subtitle=f"均值: {ic_values.mean():.4f}, 标准差: {ic_values.std():.4f}"
                    ),
                    xaxis_opts=opts.AxisOpts(name="IC区间", type_="category"),
                    yaxis_opts=opts.AxisOpts(name="频数", type_="value"),
                    legend_opts=opts.LegendOpts(pos_top="10%"),
                    datazoom_opts=[opts.DataZoomOpts(type_="inside"), opts.DataZoomOpts()],
                    tooltip_opts=opts.TooltipOpts(trigger="axis")
                )

                ic_chart = ic_chart.overlap(line_density)
            else:
                ic_chart.add_xaxis([])
                ic_chart.add_yaxis("无数据", [])
                ic_chart.set_global_opts(title_opts=opts.TitleOpts(title=f"{sample_type} IC分布图 (无数据)"))
        else:
            ic_chart.add_xaxis([])
            ic_chart.add_yaxis("无数据", [])
            ic_chart.set_global_opts(title_opts=opts.TitleOpts(title=f"{sample_type} IC分布图 (无数据)"))

        # 绘制Rank IC分布图
        rank_ic_chart = Bar()
        if not daily_ic_data.empty and 'rank_ic' in daily_ic_data.columns:
            rank_ic_values = daily_ic_data['rank_ic'].dropna()
            if len(rank_ic_values) > 0:
                # 计算统计量
                rank_ic_skew = rank_ic_values.skew()
                rank_ic_kurt = rank_ic_values.kurtosis()

                # 直方图数据
                hist_rank, bin_edges_rank = np.histogram(rank_ic_values, bins=20, density=False)
                bin_centers_rank = (bin_edges_rank[:-1] + bin_edges_rank[1:]) / 2
                bin_width_rank = float(bin_edges_rank[1] - bin_edges_rank[0]) if len(bin_edges_rank) > 1 else 0.05
                bin_labels_rank = [f"{bin_edges_rank[i]:.4f} ~ {bin_edges_rank[i+1]:.4f}" for i in range(len(bin_edges_rank) - 1)]

                rank_ic_chart.add_xaxis(bin_labels_rank)
                rank_ic_chart.add_yaxis(
                    "Histogram",
                    hist_rank.tolist(),
                    category_gap="0%",
                    bar_width="95%",
                    itemstyle_opts=opts.ItemStyleOpts(color="#74c476", opacity=0.7),
                    label_opts=opts.LabelOpts(is_show=False)
                )

                # 密度曲线（缩放到频数尺度）
                from scipy import stats
                kde_rank = stats.gaussian_kde(rank_ic_values)
                density_rank_counts = (kde_rank(bin_centers_rank) * len(rank_ic_values) * bin_width_rank).tolist()
                line_rank_density = Line()
                line_rank_density.add_xaxis(bin_labels_rank)
                line_rank_density.add_yaxis(
                    "Density Curve",
                    density_rank_counts,
                    is_symbol_show=False,
                    is_smooth=True,
                    linestyle_opts=opts.LineStyleOpts(width=2.2, color="#ff7f0e"),
                    areastyle_opts=opts.AreaStyleOpts(opacity=0.08, color="#ff7f0e"),
                    label_opts=opts.LabelOpts(is_show=False)
                )

                rank_ic_chart.set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"{sample_type} Rank IC分布图 (skew={rank_ic_skew:.3f}, kurt={rank_ic_kurt:.3f})",
                        subtitle=f"均值: {rank_ic_values.mean():.4f}, 标准差: {rank_ic_values.std():.4f}"
                    ),
                    xaxis_opts=opts.AxisOpts(name="Rank IC区间", type_="category"),
                    yaxis_opts=opts.AxisOpts(name="频数", type_="value"),
                    legend_opts=opts.LegendOpts(pos_top="10%"),
                    datazoom_opts=[opts.DataZoomOpts(type_="inside"), opts.DataZoomOpts()],
                    tooltip_opts=opts.TooltipOpts(trigger="axis")
                )

                rank_ic_chart = rank_ic_chart.overlap(line_rank_density)
            else:
                rank_ic_chart.add_xaxis([])
                rank_ic_chart.add_yaxis("无数据", [])
                rank_ic_chart.set_global_opts(title_opts=opts.TitleOpts(title=f"{sample_type} Rank IC分布图 (无数据)"))
        else:
            rank_ic_chart.add_xaxis([])
            rank_ic_chart.add_yaxis("无数据", [])
            rank_ic_chart.set_global_opts(title_opts=opts.TitleOpts(title=f"{sample_type} Rank IC分布图 (无数据)"))

        return line, line2, line3, line4, line5, line6, ic_chart, rank_ic_chart
    
    def calculate_rank_ic_decay_for_sample(self, data, max_lag=10):
        """
        为指定样本计算Rank IC衰减图数据
        """
        if 'future_ret' not in data.columns:
            print("警告: 数据中没有future_ret列，无法计算Rank IC衰减")
            return pd.DataFrame({'lag': range(1, max_lag+1), 'rank_ic': [0]*max_lag})
        
        decay_results = []
        
        for lag in range(1, max_lag + 1):
            # 计算每个滞后期的未来收益率
            lag_returns = []
            
            for symbol in data['instrument'].unique():
                symbol_data = data[data['instrument'] == symbol].copy()
                symbol_data = symbol_data.sort_values('date')
                
                # 计算lag天后的收益率
                symbol_data[f'future_ret_lag{lag}'] = symbol_data['future_ret'].shift(-lag)
                lag_returns.append(symbol_data[['date', 'instrument', self.factor_name, f'future_ret_lag{lag}']])
            
            # 合并所有symbol的数据
            lag_data = pd.concat(lag_returns, ignore_index=True)
            lag_data = lag_data.dropna()
            
            if len(lag_data) == 0:
                rank_ic = 0
            else:
                # 计算该滞后期的Rank IC
                rank_ic = lag_data[self.factor_name].corr(lag_data[f'future_ret_lag{lag}'], method='spearman')
                if pd.isna(rank_ic):
                    rank_ic = 0
            
            decay_results.append({'lag': lag, 'rank_ic': rank_ic})
        
        return pd.DataFrame(decay_results)

    def calc_rankic_halflife(self, df_or_decay_curve, threshold=0.5):
        """
        使用指数拟合法自动估计Rank IC半衰期（t_half）。
        ln(|RankIC|) ≈ ln(a) - lambda * lag
        半衰期 t_half = ln(2)/lambda
        threshold 参数仅作外部兼容，不影响指数法
        """
        # 取绝对值
        if isinstance(df_or_decay_curve, pd.DataFrame):
            vals = df_or_decay_curve['rank_ic'].abs().values
            lags = df_or_decay_curve['lag'].values
        else:
            vals = np.abs(np.asarray(df_or_decay_curve))
            lags = np.arange(1, len(vals) + 1)
        # 屏蔽无效点
        mask = (vals > 1e-6) & np.isfinite(vals)
        vals, lags = vals[mask], lags[mask]
        if len(vals) < 3:
            return None
        logy = np.log(vals)
        # 最小二乘拟合
        A = np.vstack([lags, np.ones_like(lags)]).T
        slope, intercept = np.linalg.lstsq(A, logy, rcond=None)[0]
        if slope >= 0:  # 异常，不衰减，返回无穷大或最大lag
            return float('inf')
        t_half = np.log(2) / -slope
        return t_half

    def calculate_rank_ic_autocorr_for_sample(self, data, max_lag=20):
        """
        为指定样本计算Rank IC自相关性
        """
        if 'future_ret' not in data.columns:
            print("警告: 数据中没有future_ret列，无法计算Rank IC自相关性")
            return pd.DataFrame({'lag': range(1, max_lag+1), 'autocorr': [0]*max_lag})
        
        # 按日期分组计算每日Rank IC值
        daily_rank_ic = data.groupby('date').apply(
            lambda x: x[self.factor_name].corr(x['future_ret'], method='spearman')
        ).reset_index()
        daily_rank_ic.columns = ['date', 'rank_ic']
        daily_rank_ic = daily_rank_ic.dropna()
        
        if len(daily_rank_ic) < max_lag + 1:
            print(f"警告: 数据长度不足，无法计算{max_lag}期自相关性")
            return pd.DataFrame({'lag': range(1, max_lag+1), 'autocorr': [0]*max_lag})
        
        autocorr_results = []
        rank_ic_series = daily_rank_ic['rank_ic']
        
        for lag in range(1, max_lag + 1):
            try:
                # 计算自相关系数
                autocorr = rank_ic_series.autocorr(lag=lag)
                if pd.isna(autocorr):
                    autocorr = 0
            except:
                autocorr = 0
            
            autocorr_results.append({'lag': lag, 'autocorr': autocorr})
        
        return pd.DataFrame(autocorr_results)
