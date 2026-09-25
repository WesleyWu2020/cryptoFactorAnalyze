# factor_analyse/factor_mining/util_factor.py
import os
from datetime import timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 可视化相关导入
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# 机器学习相关导入
try:
    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    from sklearn.decomposition import PCA
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# 贝叶斯优化相关导入
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from scipy import stats
import joblib
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> Crypto
DATA_DIR = os.path.join(BASE_DIR, "data")
KLINE_DIR = os.path.join(DATA_DIR, "kline_data")
FACTOR_DIR = os.path.join(DATA_DIR, "factor_data")
HIST_MCAP_PATH = os.path.join(DATA_DIR, "cryptocompare_top50_marketcap_historical.csv")
# 遗传算法因子路径
GA_FEATURE_TXT = os.path.join(BASE_DIR, "factor_analyse", "Genetic_Algorithm", "feature.txt")
GA_EXPR_LOG = os.path.join(FACTOR_DIR, "ga_top_expr_log.csv")

# 注意：稳定币在K线数据获取时已经排除，这里不需要再次排除

def load_historical_marketcap(path: str = HIST_MCAP_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到历史市值排名文件: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format='mixed', errors='coerce')
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def build_available_tokens_by_date(hdf: pd.DataFrame, lookback_days: int = 90, mode: str = "window") -> dict:
    """
    mode:
      - 'window': tokens in [current - lookback, current] (推荐)
      - 'cutoff': tokens where Date <= (current - lookback)
    """
    available_tokens = {}
    unique_dates = sorted(hdf["Date"].unique())
    for current_date in unique_dates:
        if mode == "window":
            start_date = current_date - timedelta(days=lookback_days)
            tokens = hdf[(hdf["Date"] >= start_date) & (hdf["Date"] <= current_date)]["Trading_Pair"].unique().tolist()
        else:  # cutoff
            cutoff_date = current_date - timedelta(days=lookback_days)
            tokens = hdf[hdf["Date"] <= cutoff_date]["Trading_Pair"].unique().tolist()
        available_tokens[current_date] = tokens
    return available_tokens

def build_daily_top50_ranking_from_kline(
    kline_df: pd.DataFrame,
    ranking_method: str = 'quote_volume',
    top_n: int = 50
) -> Dict[pd.Timestamp, List[str]]:
    """
    基于K线数据构建每日前N的token排名
    
    策略：每天基于Binance实际成交额排名，选择前N个token
    这样可以确保每天只使用前50个token，但保留所有token的完整K线数据
    
    注意：K线数据在获取时已经排除了稳定币，这里不需要再次排除
    
    Args:
        kline_df: K线数据DataFrame（包含所有可能进入前50的token的完整数据，已排除稳定币）
        ranking_method: 排名方法 ('quote_volume', 'volume')
        top_n: 选择前N个token，默认50
        
    Returns:
        Dict[pd.Timestamp, List[str]]: 每日前N的token符号列表
    """
    daily_top50 = {}
    
    for date, group in kline_df.groupby('date'):
        # 计算排名指标
        if ranking_method == 'quote_volume':
            # 使用成交额（USDT）排名
            ranking_df = group.groupby('symbol').agg({
                'quote_volume': 'sum',
                'base_asset': 'first'
            }).reset_index()
            ranking_df['rank_score'] = ranking_df['quote_volume']
        elif ranking_method == 'volume':
            # 使用成交量排名
            ranking_df = group.groupby('symbol').agg({
                'volume': 'sum',
                'base_asset': 'first'
            }).reset_index()
            ranking_df['rank_score'] = ranking_df['volume']
        else:
            raise ValueError(f"未知的排名方法: {ranking_method}")
        
        # 🔥 关键修复：过滤掉无效值（NaN、0、负数）
        ranking_df = ranking_df[
            (ranking_df['rank_score'].notna()) & 
            (ranking_df['rank_score'] > 0)
        ].copy()
        
        # 排序并选择前N个
        ranking_df = ranking_df.sort_values('rank_score', ascending=False)
        top_n_symbols = ranking_df.head(top_n)['symbol'].tolist()
        
        daily_top50[date] = top_n_symbols
    
    return daily_top50
    
def load_available_tokens_from_index_cache(
    cache_file: str = None,
    top_n: int = 50,
    all_dates: List[pd.Timestamp] = None
) -> Dict[pd.Timestamp, List[str]]:
    """
    从指数缓存CSV文件加载每日可用token列表
    
    Args:
        cache_file: 缓存文件路径，None表示自动查找最新的index_cache文件
        top_n: 选择前N个token，默认50（用于限制数量，如果CSV中超过top_n）
        all_dates: 所有需要填充的日期列表，如果提供则会将调仓日的成分股扩展到所有日期
        
    Returns:
        Dict[pd.Timestamp, List[str]]: 每日可用token列表
    """
    if cache_file is None:
        # 自动查找最新的index_cache文件
        cache_files = [f for f in os.listdir(DATA_DIR) 
                       if f.startswith("index_cache_") and f.endswith(".csv")]
        if not cache_files:
            return {}
        # 选择最新的文件（按文件名排序）
        cache_file = sorted(cache_files)[-1]
    
    cache_path = os.path.join(DATA_DIR, cache_file)
    
    if not os.path.exists(cache_path):
        print(f"⚠️  指数缓存文件不存在: {cache_path}")
        return {}
    
    print(f"📊 从指数缓存文件加载成分股列表: {cache_file}")
    
    try:
        df = pd.read_csv(cache_path)
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        
        # 检查必要的列
        if 'components' not in df.columns:
            print(f"⚠️  缓存文件缺少components列: {cache_path}")
            return {}
        
        # 🔥 修改：先构建调仓日的成分股字典，只保存非空的成分股
        rebalance_components = {}
        has_rebalance_day_col = 'is_rebalance_day' in df.columns
        
        for _, row in df.iterrows():
            date = row['date']
            if pd.isna(date):
                continue
            
            # 解析components列（逗号分隔的字符串）
            components_str = str(row['components']) if pd.notna(row['components']) else ''
            components = [c.strip() for c in components_str.split(',') if c.strip()]
            
            # 🔥 关键修改：只保存非空的成分股列表
            if len(components) == 0:
                continue
            
            # 限制为top_n个
            if len(components) > top_n:
                components = components[:top_n]
            
            rebalance_components[date] = components
        
        if not rebalance_components:
            print(f"⚠️  缓存文件中没有有效的成分股数据")
            return {}
        
        # 如果提供了all_dates，将调仓日的成分股扩展到所有日期
        if all_dates is not None:
            available_tokens = {}
            rebalance_dates = sorted(rebalance_components.keys())
            
            for current_date in all_dates:
                # 🔥 修改：优先查找同月的调仓日，如果没有则找最近的调仓日
                closest_rebalance_date = None
                
                # 1. 首先查找同月内最近的调仓日（小于等于当前日期）
                current_year_month = (current_date.year, current_date.month)
                for rb_date in reversed(rebalance_dates):
                    if rb_date <= current_date:
                        rb_year_month = (rb_date.year, rb_date.month)
                        if rb_year_month == current_year_month:
                            closest_rebalance_date = rb_date
                            break
                
                # 2. 如果同月内没有找到，使用最近的调仓日
                if closest_rebalance_date is None:
                    for rb_date in reversed(rebalance_dates):
                        if rb_date <= current_date:
                            closest_rebalance_date = rb_date
                            break
                
                # 如果找到调仓日，使用其成分股；否则跳过
                if closest_rebalance_date is not None:
                    components = rebalance_components[closest_rebalance_date]
                    available_tokens[current_date] = components
            
            print(f"✅ 成功加载 {len(rebalance_components)} 个调仓日的成分股列表")
            print(f"   扩展到 {len(available_tokens)} 个交易日的成分股列表")
            
            # 🔥 新增：显示每月使用的调仓日信息
            monthly_rebalance_usage = {}
            for date in sorted(available_tokens.keys()):
                year_month = (date.year, date.month)
                if year_month not in monthly_rebalance_usage:
                    # 找出这个月使用的是哪个调仓日
                    for rb_date in reversed(rebalance_dates):
                        if rb_date <= date:
                            rb_year_month = (rb_date.year, rb_date.month)
                            if rb_year_month == year_month:
                                monthly_rebalance_usage[year_month] = rb_date
                                break
                    # 如果同月没有，使用最近的
                    if year_month not in monthly_rebalance_usage:
                        for rb_date in reversed(rebalance_dates):
                            if rb_date <= date:
                                monthly_rebalance_usage[year_month] = rb_date
                                break
            
            # 显示最近几个月的调仓日使用情况
            recent_months = sorted(monthly_rebalance_usage.keys())[-3:]
            for year, month in recent_months:
                rb_date = monthly_rebalance_usage[(year, month)]
                print(f"   {year}-{month:02d}: 使用 {rb_date.strftime('%Y-%m-%d')} 的成分股 ({len(rebalance_components[rb_date])} 个币种)")
        else:
            # 如果没有提供all_dates，只返回调仓日的成分股
            available_tokens = rebalance_components
            print(f"✅ 成功加载 {len(rebalance_components)} 个调仓日的成分股列表")
        
        if available_tokens:
            dates = sorted(available_tokens.keys())
            print(f"   时间范围: {dates[0].strftime('%Y-%m-%d')} 至 {dates[-1].strftime('%Y-%m-%d')}")
            # 显示几个日期的币种数量
            sample_dates = dates[:3] + dates[-3:] if len(dates) > 6 else dates
            for d in sample_dates:
                print(f"   {d.strftime('%Y-%m-%d')}: {len(available_tokens[d])} 个币种")
        
        return available_tokens
        
    except Exception as e:
        print(f"❌ 读取指数缓存文件失败: {e}")
        return {}

def build_available_tokens_by_date_from_kline(
    kline_df: pd.DataFrame,
    top_n: int = 50,
    ranking_method: str = 'quote_volume',
    rebalance_period: int = 10,
    lookback_buffer: int = 20,
    strict_top_n: bool = True
) -> Dict[pd.Timestamp, List[str]]:
    """
    基于K线数据构建每日可用token列表（前N名）
    
    关键特性：
    1. 每天只选择前N个token（基于成交额排名）
    2. 但保留这些token的完整历史数据（包括它们不在前N时的数据）
    3. 考虑调仓周期：如果10天调仓，需要确保10天后token仍有数据
    
    Args:
        kline_df: K线数据DataFrame（包含所有可能进入前50的token的完整数据）
        top_n: 每天选择前N个token，默认50
        ranking_method: 排名方法 ('quote_volume', 'volume')
        rebalance_period: 调仓周期（天），默认10天
        lookback_buffer: 回看缓冲区（天），确保有足够的历史数据，默认20天
        strict_top_n: 是否严格限制为top_n个token（True：每天只使用top_n个；False：包含未来可能进入前N的token），默认True
        
    Returns:
        Dict[pd.Timestamp, List[str]]: 每日可用token列表
    """
    # 构建每日前N排名
    daily_top50 = build_daily_top50_ranking_from_kline(
        kline_df,
        ranking_method=ranking_method,
        top_n=top_n
    )
    
    available_tokens = {}
    
    # 获取所有日期
    all_dates = sorted(kline_df['date'].unique())
    
    if strict_top_n:
        # 严格模式：每天只使用前N个token
        # 但我们需要收集所有可能需要的token（当前前N + 未来可能进入前N的），以便保留它们的完整历史数据
        all_needed_symbols = set()
        for date in all_dates:
            all_needed_symbols.update(daily_top50.get(date, []))
            # 未来rebalance_period天内可能进入前N的token
            future_dates = [d for d in all_dates if date < d <= date + timedelta(days=rebalance_period)]
            for future_date in future_dates:
                all_needed_symbols.update(daily_top50.get(future_date, []))
        
        # 对于每个日期，只返回当日前N的token（严格限制）
        # 但all_needed_symbols包含了所有需要保留数据的token
        for current_date in all_dates:
            current_top50 = daily_top50.get(current_date, [])
            # 严格限制为top_n个
            available_tokens[current_date] = current_top50[:top_n] if len(current_top50) > top_n else current_top50
    else:
        # 非严格模式：包含当前前N + 未来可能进入前N的token
        for current_date in all_dates:
            # 当前日期前N的token
            current_top50 = set(daily_top50.get(current_date, []))
            
            # 未来rebalance_period天内可能进入前N的token
            future_dates = [d for d in all_dates if current_date < d <= current_date + timedelta(days=rebalance_period)]
            future_top50 = set()
            for future_date in future_dates:
                future_top50.update(daily_top50.get(future_date, []))
            
            # 合并：当前前N + 未来可能进入前N的token
            available_tokens[current_date] = list(current_top50 | future_top50)
    
    return available_tokens

def latest_kline_path(prefix: str = "binance_daily_klines_") -> str:
    # 排除统计文件（包含_daily_token_count的文件）
    files = sorted([f for f in os.listdir(KLINE_DIR)
                   if f.startswith(prefix) and '_daily_token_count' not in f])
    if not files:
        raise FileNotFoundError("未找到K线数据文件")
    return os.path.join(KLINE_DIR, files[-1])

def load_kline_df(path: str | None = None) -> pd.DataFrame:
    path = path or latest_kline_path()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])
    return df

def filter_group_by_availability(group: pd.DataFrame, symbol: str, available_tokens_by_date: dict) -> pd.DataFrame:
    """
    根据可用token列表筛选数据
    
    注意：此函数会保留symbol的完整历史数据（包括不在可用列表时的数据）
    这样可以确保即使某个token某天不在前50，但调仓日它进入前50时，我们仍有它的历史数据
    """
    if group.empty:
        return group
    
    # 如果available_tokens_by_date的key是pd.Timestamp，需要转换
    date_keys_sorted = sorted(available_tokens_by_date.keys())
    
    valid_idx = []
    for idx, row in group.iterrows():
        current = row["date"]
        
        # 找到最接近的可用日期
        available_date = None
        for dk in date_keys_sorted:
            if isinstance(dk, pd.Timestamp):
                dk_date = dk
            else:
                dk_date = pd.to_datetime(dk)
            
            if dk_date <= current:
                available_date = dk
            else:
                break
        
        # 检查token在该日期是否可用
        if available_date is not None:
            if symbol in available_tokens_by_date[available_date]:
                valid_idx.append(idx)
    
    # 如果valid_idx为空，返回空DataFrame
    if not valid_idx:
        return group.iloc[0:0].copy()
    
    return group.loc[valid_idx].copy()

def future_return(close: pd.Series, period: int, method: str = "log") -> pd.Series:
    if method == "pct":
        return close.pct_change(period).shift(-period)
    return np.log(close.shift(-period) / close)

def group_apply_with_progress(df: pd.DataFrame, by: str, func) -> list[pd.DataFrame]:
    out = []
    for key, gp in tqdm(df.groupby(by)):
        r = func(gp, key) if func.__code__.co_argcount >= 2 else func(gp)
        if r is not None and not r.empty:
            out.append(r)
    return out

def winsorize_by_date(df: pd.DataFrame, col: str, n_std: float = 3.0) -> pd.DataFrame:
    def _w(d):
        m = d[col].mean()
        sd = d[col].std()
        d[col] = d[col].clip(lower=m - n_std * sd, upper=m + n_std * sd)
        return d
    return df.groupby("date", group_keys=False).apply(_w)

def rank_to_unit_by_date(df: pd.DataFrame, col: str, out_col: str = "factor") -> pd.DataFrame:
    def _n(d):
        rank_pct = d[col].rank(pct=True)
        d[out_col] = 2 * (rank_pct - 0.5)
        return d
    return df.groupby("date", group_keys=False).apply(_n)

def dedupe_predictions_keep_latest_window(
    df: pd.DataFrame,
    date_col: str = "date",
    symbol_col: str = "symbol",
    window_col: str = "window_idx",
) -> tuple[pd.DataFrame, int]:
    """
    对重叠滚动窗口输出按 (date, symbol) 去重，保留最新窗口记录。
    返回 (去重后的DataFrame, 删除行数)。
    """
    if df.empty:
        return df.copy(), 0

    out = df.copy()
    required_cols = {date_col, symbol_col}
    if not required_cols.issubset(out.columns):
        return out, 0

    out[date_col] = pd.to_datetime(out[date_col])
    if window_col in out.columns:
        out = out.sort_values([date_col, symbol_col, window_col], kind="mergesort")
    else:
        out = out.sort_values([date_col, symbol_col], kind="mergesort")

    before = len(out)
    out = out.drop_duplicates(subset=[date_col, symbol_col], keep="last").reset_index(drop=True)
    return out, before - len(out)

def apply_label_purge_to_split(
    train_data: list,
    val_data: list,
    test_data: list,
    purge_days: int = 0,
    date_index: int = 4,
) -> tuple[list, list, list]:
    """
    对(train, val, test)应用时间purge，避免标签窗口跨边界。
    - train 保留 date < min(val_date) - purge_days
    - val   保留 date < min(test_date) - purge_days
    """
    if purge_days is None or int(purge_days) <= 0:
        return train_data, val_data, test_data

    delta = pd.Timedelta(days=int(purge_days))

    if val_data:
        val_start = min(pd.to_datetime(item[date_index]) for item in val_data)
        train_cutoff = val_start - delta
        train_data = [item for item in train_data if pd.to_datetime(item[date_index]) < train_cutoff]

    if test_data:
        test_start = min(pd.to_datetime(item[date_index]) for item in test_data)
        val_cutoff = test_start - delta
        val_data = [item for item in val_data if pd.to_datetime(item[date_index]) < val_cutoff]

    return train_data, val_data, test_data

def save_factor_df(factor_df: pd.DataFrame, file_prefix: str, suffix: str = "") -> str:
    os.makedirs(FACTOR_DIR, exist_ok=True)
    today = pd.Timestamp.today().strftime("%Y%m%d")
    fname = f"{file_prefix}{suffix}{today}.csv"
    out_path = os.path.join(FACTOR_DIR, fname)
    factor_df.to_csv(out_path, index=False)
    return out_path

def print_availability_sample(available_tokens_by_date: dict, n: int = 3):
    keys = list(available_tokens_by_date.keys())[:n]
    for d in keys:
        print(f"   {pd.to_datetime(d):%Y-%m-%d}: {len(available_tokens_by_date[d])} 个可用token")

def analyze_token_performance(factor_df: pd.DataFrame, n_days: int = 5) -> None:
    """
    分析某个token的前N日因子值和收益率表现
    
    参数:
    factor_df: 包含 date, instrument, factor, future_ret 的因子数据
    n_days: 分析的天数，默认5天
    """
    print(f"\n=== 某个Token的前{n_days}日因子值分析 ===")
    
    # 选择因子值变化较大的token进行分析
    token_factor_std = factor_df.groupby('instrument')['factor'].std()
    if not token_factor_std.empty:
        # 选择因子值标准差最大的token
        selected_token = token_factor_std.idxmax()
        print(f"选择分析Token: {selected_token} (因子值标准差最大)")
        
        # 获取该token的数据
        token_data = factor_df[factor_df['instrument'] == selected_token].copy()
        token_data = token_data.sort_values('date')
        
        if len(token_data) >= n_days:
            # 显示前N日数据
            recent_data = token_data.tail(n_days)
            print(f"\n{selected_token} 前{n_days}日数据:")
            print("=" * 80)
            print(f"{'日期':<12} {'因子值':<12} {'未来收益率':<12} {'因子排名':<12}")
            print("-" * 80)
            
            for _, row in recent_data.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                factor_val = f"{row['factor']:.6f}"
                future_ret = f"{row['future_ret']:.6f}"
                
                # 计算当日因子排名
                daily_data = factor_df[factor_df['date'] == row['date']]
                if len(daily_data) > 0:
                    rank_pct = daily_data['factor'].rank(pct=True)
                    token_rank = rank_pct[daily_data['instrument'] == selected_token].iloc[0]
                    rank_str = f"{token_rank:.3f}"
                else:
                    rank_str = "N/A"
                
                print(f"{date_str:<12} {factor_val:<12} {future_ret:<12} {rank_str:<12}")
            
            print("=" * 80)
            
            # 显示该token的统计信息
            print(f"\n{selected_token} 统计信息:")
            print(f"因子值范围: [{token_data['factor'].min():.6f}, {token_data['factor'].max():.6f}]")
            print(f"因子均值: {token_data['factor'].mean():.6f}")
            print(f"因子标准差: {token_data['factor'].std():.6f}")
            print(f"未来收益率均值: {token_data['future_ret'].mean():.6f}")
            print(f"未来收益率标准差: {token_data['future_ret'].std():.6f}")
            
            # 计算因子值与未来收益率的相关性
            correlation = token_data['factor'].corr(token_data['future_ret'])
            print(f"因子值与未来收益率相关性: {correlation:.6f}")
            
        else:
            print(f"Token {selected_token} 数据不足{n_days}天，无法显示前{n_days}日分析")
    else:
        print("没有可用的token数据进行前N日分析")

def print_factor_summary(factor_df: pd.DataFrame, out_path: str) -> None:
    """
    打印因子数据总结信息
    
    参数:
    factor_df: 因子数据DataFrame
    out_path: 输出文件路径
    """
    print(f"✅ 无未来函数的因子数据已保存至: {out_path}")
    print(f"总计生成 {len(factor_df)} 条因子记录")
    
    # 显示因子数据统计信息
    print("\n因子统计信息:")
    print(factor_df['factor'].describe())
    
    # 分析某个token的前5日表现
    analyze_token_performance(factor_df, n_days=5)
    
    # 显示数据预览
    print("\n=== 整体数据预览 ===")
    print(factor_df.head(10))
    
def enhance_factor_dispersion(df, col='factor', power=1.5):
    """增强因子的分散度，保持排名不变但拉大极值差异"""
    df = df.copy()
    # 按日期分组处理
    def process_group(group):
        group = group.copy()
        # 将因子值映射到[-1, 1]
        factor_values = group[col].values
        # 应用幂次变换（保持符号）
        enhanced = np.sign(factor_values) * (np.abs(factor_values) ** power)
        group[col] = enhanced
        return group
    
    return df.groupby('date', group_keys=False).apply(process_group)


# ===== XGBoost 相关类和函数 =====

class AdaptiveSelector:
    """自适应特征选择器，支持pickle序列化"""
    def __init__(self, pre_selector, selected_pre_indices):
        self.pre_selector = pre_selector
        self.selected_pre_indices = selected_pre_indices

    def fit(self, X, y):
        return self

    def transform(self, X):
        X_pre = self.pre_selector.transform(X)
        return X_pre[:, self.selected_pre_indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

class CombinedSelector:
    """组合选择器：先应用正交化，再应用互信息选择，支持pickle序列化"""
    def __init__(self, orthogonalizer, pre_selector, selected_pre_indices):
        self.orthogonalizer = orthogonalizer
        self.pre_selector = pre_selector
        self.selected_pre_indices = selected_pre_indices

    def fit(self, X, y):
        return self

    def transform(self, X):
        if self.orthogonalizer is not None:
            X_ortho = self.orthogonalizer.transform(X)
        else:
            X_ortho = X

        if self.pre_selector is not None:
            X_pre = self.pre_selector.transform(X_ortho)
            return X_pre[:, self.selected_pre_indices]
        else:
            return X_ortho[:, self.selected_pre_indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

class ConservativeSelector:
    """保守特征选择器：先正交化，再基于重要性选择，支持pickle序列化"""
    def __init__(self, orthogonalizer, selected_indices):
        self.orthogonalizer = orthogonalizer
        self.selected_indices = selected_indices

    def fit(self, X, y):
        return self

    def transform(self, X):
        if self.orthogonalizer is not None:
            X_ortho = self.orthogonalizer.transform(X)
            return X_ortho[:, self.selected_indices]
        else:
            # 如果没有正交化器，直接从原始特征中选择
            return X[:, self.selected_indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

class FeatureOrthogonalizer:
    """
    特征正交化处理器：去除高度相关的特征并进行正交化
    支持两种模式：
    1. 相关性过滤：去除高度相关的冗余特征
    2. PCA正交化：将特征转换为正交主成分（可选）
    """
    def __init__(self, correlation_threshold=0.95, use_pca=False, pca_components=None, pca_variance_ratio=0.95):
        """
        Parameters:
        - correlation_threshold: 相关性阈值，超过此值的特征对将被去除其中一个
        - use_pca: 是否使用PCA进行正交化
        - pca_components: PCA主成分数量（None表示自动选择）
        - pca_variance_ratio: PCA保留的方差比例（当pca_components为None时使用）
        """
        self.correlation_threshold = correlation_threshold
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca_variance_ratio = pca_variance_ratio
        self.selected_features_ = None
        self.pca_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        """
        拟合正交化器

        Parameters:
        - X: 特征矩阵 (n_samples, n_features)
        - y: 目标变量（可选，用于基于重要性的特征选择）
        """
        X = np.array(X)
        n_features = X.shape[1]

        # 1. 相关性过滤：去除高度相关的特征
        print(f"🔄 特征相关性分析（阈值={self.correlation_threshold}）...")

        # 计算相关性矩阵（使用pandas DataFrame更高效）
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().values
        np.fill_diagonal(corr_matrix, 0)  # 对角线设为0

        # 如果提供了y，计算每个特征与y的相关性
        if y is not None:
            y_corrs = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(n_features)])
            y_corrs = np.nan_to_num(y_corrs, nan=0)
        else:
            # 如果没有y，使用特征的方差作为重要性指标
            y_corrs = np.var(X, axis=0)
            y_corrs = np.nan_to_num(y_corrs, nan=0)

        # 找出高度相关的特征对，并选择要移除的特征
        to_remove = set()
        for i in range(n_features):
            if i in to_remove:
                continue
            for j in range(i+1, n_features):
                if j in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    # 优先保留与y相关性更高（或方差更大）的特征
                    if y_corrs[i] >= y_corrs[j]:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)

        # 保留的特征索引
        self.selected_features_ = np.array([i for i in range(n_features) if i not in to_remove])

        removed_count = len(to_remove)
        print(f"   去除 {removed_count} 个高度相关特征，保留 {len(self.selected_features_)} 个特征")

        # 应用相关性过滤
        X_filtered = X[:, self.selected_features_]

        # 2. PCA正交化（可选）
        if self.use_pca:
            print(f"🔄 PCA正交化处理...")
            self.pca_ = PCA(n_components=self.pca_components,
                           svd_solver='full' if self.pca_components is None else 'auto')
            self.pca_.fit(X_filtered)

            # 如果未指定主成分数量，根据方差比例自动选择
            if self.pca_components is None:
                cumsum_variance = np.cumsum(self.pca_.explained_variance_ratio_)
                n_components = np.argmax(cumsum_variance >= self.pca_variance_ratio) + 1
                n_components = min(n_components, X_filtered.shape[1])
                self.pca_ = PCA(n_components=n_components, svd_solver='full')
                self.pca_.fit(X_filtered)

            explained_variance = np.sum(self.pca_.explained_variance_ratio_)
            print(f"   PCA主成分数: {self.pca_.n_components_}, 解释方差: {explained_variance:.4f}")
        else:
            self.pca_ = None

        return self

    def transform(self, X):
        """转换特征矩阵"""
        X = np.array(X)

        # 1. 应用相关性过滤
        X_filtered = X[:, self.selected_features_]

        # 2. 应用PCA正交化（如果启用）
        if self.pca_ is not None:
            X_transformed = self.pca_.transform(X_filtered)
        else:
            X_transformed = X_filtered

        return X_transformed

    def fit_transform(self, X, y=None):
        """拟合并转换"""
        self.fit(X, y)
        return self.transform(X)

    def get_feature_importance_mapping(self):
        """
        获取PCA主成分到原始特征的映射（如果使用PCA）
        返回主成分的载荷矩阵
        """
        if self.pca_ is not None:
            # 载荷矩阵：每个主成分对原始特征的贡献
            return self.pca_.components_.T  # (n_features, n_components)
        else:
            return np.eye(len(self.selected_features_))  # 单位矩阵（无变换）

def huber_loss_objective(y_pred, y_true, delta=1.0):
    """
    Huber Loss 自定义目标函数（用于XGBoost）

    Huber Loss 结合了 MAE 和 MSE 的优点：
    - 当误差小于delta时，使用MSE（二次损失，平滑）
    - 当误差大于delta时，使用MAE（线性损失，对异常值鲁棒）

    Parameters:
    - y_pred: 预测值
    - y_true: 真实值（DMatrix中的label）
    - delta: 阈值参数（默认1.0），控制从二次损失切换到线性损失的临界点

    Returns:
    - grad: 一阶梯度
    - hess: 二阶梯度（Hessian）
    """
    residual = y_pred - y_true
    abs_residual = np.abs(residual)

    # 计算梯度（一阶导数）
    # 当 |residual| <= delta: grad = residual
    # 当 |residual| > delta: grad = delta * sign(residual)
    grad = np.where(
        abs_residual <= delta,
        residual,  # MSE部分的梯度
        delta * np.sign(residual)  # MAE部分的梯度
    )

    # 计算Hessian（二阶导数）
    # 当 |residual| <= delta: hess = 1
    # 当 |residual| > delta: hess = 0
    hess = np.where(
        abs_residual <= delta,
        1.0,  # MSE部分的Hessian
        0.0   # MAE部分的Hessian（线性部分）
    )

    return grad, hess

def huber_loss_eval_metric(y_pred, y_true, delta=1.0):
    """
    Huber Loss 评估指标（用于XGBoost的eval_metric）

    Returns:
    - 损失值（标量）
    """
    residual = y_pred - y_true
    abs_residual = np.abs(residual)

    # Huber Loss 值
    loss = np.where(
        abs_residual <= delta,
        0.5 * residual ** 2,  # MSE部分
        delta * abs_residual - 0.5 * delta ** 2  # MAE部分
    )

    return 'huber_loss', np.mean(loss)

# 全局变量用于存储Huber Loss的delta值（用于自定义目标函数）
_huber_delta_global = 1.0

def _huber_obj_global(y_pred, dtrain):
    """
    全局Huber Loss目标函数（使用全局变量_huber_delta_global）

    XGBoost自定义目标函数签名: (y_pred, dtrain) -> (grad, hess)
    - y_pred: 预测值数组（numpy array）
    - dtrain: DMatrix对象，需要用get_label()获取真实标签

    Returns:
    - grad: 一阶梯度
    - hess: 二阶梯度（Hessian）
    """
    y_true = dtrain.get_label()
    return huber_loss_objective(y_pred, y_true, delta=_huber_delta_global)

def _huber_metric_global(y_pred, dtrain):
    """
    全局Huber Loss评估指标（使用全局变量_huber_delta_global）

    XGBoost自定义评估指标签名: (y_pred, dtrain) -> (metric_name, metric_value)
    - y_pred: 预测值数组（numpy array）
    - dtrain: DMatrix对象，需要用get_label()获取真实标签

    Returns:
    - (metric_name, metric_value) 元组
    """
    y_true = dtrain.get_label()
    return huber_loss_eval_metric(y_pred, y_true, delta=_huber_delta_global)

def get_default_xgb_params(huber_delta=1.0):
    """获取默认的XGBoost参数（强正则化版本，使用Huber Loss）"""
    # 设置全局delta值
    global _huber_delta_global
    _huber_delta_global = huber_delta

    return {
        # 注意：使用obj参数传递自定义目标函数，custom_metric使用Huber Loss用于early stopping
        'learning_rate': 0.01,      # 从0.03进一步降低到0.01，增强正则化
        'max_depth': 3,             # 从4降低到3，减少过拟合
        'min_child_weight': 25,     # 从12增加到25，增强正则化
        'gamma': 2.0,              # 从0.8增加到2.0，增强后剪枝
        'subsample': 0.6,          # 从0.7降低到0.6，增强正则化
        'colsample_bytree': 0.4,   # 从0.5降低到0.4，增强正则化
        'colsample_bynode': 0.6,   # 从0.8降低到0.6，增强正则化
        'reg_alpha': 8.0,          # 从3.0增加到8.0，增强L1正则化
        'reg_lambda': 20.0,        # 从10.0增加到20.0，增强L2正则化
        'n_estimators': 2500,      # 从3000降低到2500，减少过拟合
        'early_stopping_rounds': 200,  # 从300降低到200，更严格的早停
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
    }

def get_enhanced_xgb_params(huber_delta=1.0):
    """增强的正则化参数（超强正则化版本，使用Huber Loss）"""
    # 设置全局delta值
    global _huber_delta_global
    _huber_delta_global = huber_delta

    return {
        # 注意：使用obj参数传递自定义目标函数，custom_metric使用Huber Loss用于early stopping
        'learning_rate': 0.008,     # 从0.015进一步降低到0.008，极强正则化
        'max_depth': 2,             # 从3降低到2，极度限制树深度
        'min_child_weight': 35,     # 从20增加到35，极强最小叶子权重
        'gamma': 4.0,              # 从2.0增加到4.0，极强后剪枝
        'subsample': 0.4,          # 从0.5降低到0.4，极低子采样比例
        'colsample_bytree': 0.25,  # 从0.3降低到0.25，极低特征采样比例
        'colsample_bynode': 0.35,  # 从0.5降低到0.35，极低节点特征采样
        'reg_alpha': 12.0,         # 从5.0增加到12.0，极强L1正则化
        'reg_lambda': 30.0,        # 从15.0增加到30.0，极强L2正则化
        'n_estimators': 2000,      # 从3000降低到2000，减少迭代次数
        'early_stopping_rounds': 100,  # 从150降低到100，极严格早停
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
    }

def get_ultra_regularized_xgb_params(huber_delta=1.0):
    """超强正则化参数（最大限度防止过拟合，使用Huber Loss）"""
    # 设置全局delta值
    global _huber_delta_global
    _huber_delta_global = huber_delta

    return {
        # 注意：使用obj参数传递自定义目标函数，custom_metric使用Huber Loss用于early stopping
        'learning_rate': 0.005,     # 极低学习率，最大限度正则化
        'max_depth': 2,             # 极浅的树
        'min_child_weight': 50,     # 极高的最小叶子权重
        'gamma': 8.0,              # 极强的后剪枝
        'subsample': 0.3,          # 极低的子采样比例
        'colsample_bytree': 0.15,  # 极低的特征采样比例
        'colsample_bynode': 0.2,   # 极低的节点特征采样
        'reg_alpha': 20.0,         # 极强的L1正则化
        'reg_lambda': 50.0,        # 极强的L2正则化
        'n_estimators': 1500,      # 极少的迭代次数
        'early_stopping_rounds': 50,   # 极严格的早停
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
    }

def get_feature_names(use_improved_features: bool = False, use_ga_features: bool = False):
    """
    获取特征名称列表（与特征生成函数保持一致）

    Parameters:
    - use_improved_features: 是否使用改进的特征工程
    - use_ga_features: 是否包含遗传算法因子（来自 feature.txt）

    Returns:
    - feature_names: 特征名称列表
    """
    if use_improved_features:
        # 精简版特征名称
        feature_names = [
            # 价格动量特征
            'momentum_5', 'momentum_vol_adj_5', 'range_efficiency_5',
            'momentum_10', 'momentum_vol_adj_10', 'range_efficiency_10',
            'momentum_20', 'momentum_vol_adj_20', 'range_efficiency_20',
            # 量价关系特征
            'price_volume_divergence_5', 'vwap_premium_5',
            'price_volume_divergence_10', 'vwap_premium_10',
            # 均值回复特征
            'ma_deviation_10', 'rsi_10',
            'ma_deviation_20', 'rsi_20',
            # 市场微观结构特征（如果可用）
            'money_flow_5', 'large_order_ratio_5',
            'money_flow_10', 'large_order_ratio_10',
            # 技术指标特征
            'macd_signal', 'bollinger_position_20',
            # Multi-Transformer量价时序特征（13个）
            'pvo', 'sse', 'liq', 'skew', 'gap', 'vts', 'vcll', 'rvr', 'csl', 'rev', 'vclb', 'kurtosis', 'lip'
        ]
    else:
        # 完整版特征名称
        feature_names = [
            # 基础价格特征
            'open', 'high', 'low', 'close', 'volume', 'vwap',
            # 累积买卖比率因子
            'cum_buy_sell_ratio',
            # 买入稳定因子
            'buy_stability_5', 'buy_stability_10', 'buy_stability_20', 'buy_stability_30',
            # 方向动量因子
            'directional_momentum_5', 'directional_momentum_10', 'directional_momentum_20', 'directional_momentum_30',
            # 定向波动率因子
            'directional_volatility_5', 'directional_volatility_10', 'directional_volatility_20', 'directional_volatility_30',
            # 高波动性动量
            'hv_momentum_product_5', 'hv_momentum_ratio_5',
            'hv_momentum_product_10', 'hv_momentum_ratio_10',
            'hv_momentum_product_20', 'hv_momentum_ratio_20',
            'hv_momentum_product_30', 'hv_momentum_ratio_30',
            # 动量-主动买入极值
            'momentum_tbq_extreme_10', 'momentum_tbq_extreme_20', 'momentum_tbq_extreme_30',
            # 动量-成交量极值分组因子
            'momentum_volume_extreme_10', 'momentum_volume_extreme_20', 'momentum_volume_extreme_30',
            # 净订单流
            'net_order_flow_5', 'net_order_flow_10', 'net_order_flow_20', 'net_order_flow_30',
            # 涨跌距离/路径效率
            'price_path_efficiency_5', 'price_path_efficiency_10', 'price_path_efficiency_20', 'price_path_efficiency_30',
            # 滚动相关性
            'rolling_corr_buyquote_5', 'rolling_corr_buyquote_10', 'rolling_corr_buyquote_20', 'rolling_corr_buyquote_30',
            # RSI截面排序因子
            'rsi_14', 'rsi_21', 'rsi_30',
            # TBQ极值效率
            'tbq_extreme_efficiency_10', 'tbq_extreme_efficiency_20', 'tbq_extreme_efficiency_30',
            # 交易比
            'top_bottom_trades_ratio_10', 'top_bottom_trades_ratio_20', 'top_bottom_trades_ratio_30',
            # 波动效率
            'volatility_efficiency_5', 'volatility_efficiency_10', 'volatility_efficiency_20', 'volatility_efficiency_30',
            # 波动率-订单流
            'volatility_order_flow_5', 'volatility_order_flow_10', 'volatility_order_flow_20', 'volatility_order_flow_30',
            # 量稳因子
            'volume_stability_10', 'volume_stability_20', 'volume_stability_30',
            # 成交量切分
            'volume_tbb_extreme_10', 'volume_tbb_extreme_20', 'volume_tbb_extreme_30',
            # 买卖强度稳定性
            'buy_intensity_stability_5', 'buy_intensity_stability_10', 'buy_intensity_stability_20', 'buy_intensity_stability_30',
            # 价格路径平滑度
            'price_path_stability_5', 'price_path_stability_10', 'price_path_stability_20', 'price_path_stability_30',
            'price_efficiency_stability_5', 'price_efficiency_stability_10', 'price_efficiency_stability_20', 'price_efficiency_stability_30',
            # 成交笔数稳定性
            'trades_frequency_stability_5', 'trades_frequency_stability_10', 'trades_frequency_stability_20', 'trades_frequency_stability_30',
            'trade_size_stability_5', 'trade_size_stability_10', 'trade_size_stability_20', 'trade_size_stability_30',
            # 买卖压力稳定性
            'pressure_balance_stability_5', 'pressure_balance_stability_10', 'pressure_balance_stability_20', 'pressure_balance_stability_30',
            'pressure_ratio_stability_5', 'pressure_ratio_stability_10', 'pressure_ratio_stability_20', 'pressure_ratio_stability_30',
            # Multi-Transformer量价时序特征（13个）
            'pvo', 'sse', 'liq', 'skew', 'gap', 'vts', 'vcll', 'rvr', 'csl', 'rev', 'vclb', 'kurtosis', 'lip'
        ]

        # 条件特征（如果有相关数据才添加）
        conditional_features = ['quote_volume']  # 基础价格特征已包含
        # 如果有taker_buy_quote数据，会添加相关的特征（已在上面包含）

    # 遗传算法因子放在最前面（与数据列顺序一致，便于特征选择时优先筛选）
    if use_ga_features:
        ga_names = _load_ga_feature_names()
        if ga_names:
            feature_names = ga_names + feature_names

    return feature_names

def add_multi_transformer_features(feat: pd.DataFrame, close: pd.Series, open_price: pd.Series, vol: pd.Series,
                                   open_raw: pd.Series = None, eps: float = 1e-8):
    """
    添加Multi-Transformer模型的13个量价时序特征

    Parameters:
    - feat: 特征DataFrame
    - close: 收盘价序列（已shift(1)处理）
    - open_price: 开盘价序列（已shift(1)处理）
    - vol: 成交量序列（已shift(1)处理）
    - open_raw: 原始开盘价序列（未shift，用于计算gap特征，避免未来函数）
    - eps: 防止除零的小值
    """
    # 计算日收益率
    returns = close.pct_change()

    # 1. 量价背离度 (pvo): T日收盘价减去其5日指数移动平均，再除以20日成交量移动平均
    ema_5 = close.ewm(span=5, adjust=False).mean()
    vol_ma_20 = vol.rolling(20).mean()
    feat['pvo'] = (close - ema_5) / (vol_ma_20 + eps)

    # 2. 压力支撑效率 (sse): 过去20日最高收盘价减去最低收盘价，再除以20日成交量移动平均
    close_max_20 = close.rolling(20).max()
    close_min_20 = close.rolling(20).min()
    feat['sse'] = (close_max_20 - close_min_20) / (vol_ma_20 + eps)

    # 3. 流动性冲击系数 (liq): T日成交量与昨日成交量的绝对差值，除以过去20日成交量标准差
    vol_diff = (vol - vol.shift(1)).abs()
    vol_std_20 = vol.rolling(20).std()
    feat['liq'] = vol_diff / (vol_std_20 + eps)

    # 4. 波动率偏度 (skew): 过去20日T日收益率的三阶标准化矩
    returns_mean_20 = returns.rolling(20).mean()
    returns_std_20 = returns.rolling(20).std()
    returns_normalized = (returns - returns_mean_20) / (returns_std_20 + eps)
    feat['skew'] = returns_normalized.rolling(20).apply(lambda x: x.pow(3).mean(), raw=False)

    # 5. 隔夜跳空强度 (gap): T日开盘价减去昨日收盘价，再除以T-1日收盘价
    # 注意：close已经shift(1)，所以close[i]是T-1日的收盘价
    # 如果提供了open_raw（未shift），使用open_raw[i]作为T日开盘价
    # 否则使用open_price[i]（T-1日开盘价）作为近似，但这不是完全准确的
    if open_raw is not None:
        # open_raw[i]是T日的开盘价，close[i]是T-1日的收盘价
        feat['gap'] = (open_raw - close) / (close + eps)
    else:
        # 回退方案：使用open_price（T-1日开盘价）和close.shift(1)（T-2日收盘价）
        # 这计算的是T-1日的gap，不是T日的gap，但避免了未来函数
        feat['gap'] = (open_price - close.shift(1)) / (close.shift(1) + eps)

    # 6. 波动率期限结构 (vts): 过去10日收益率标准差除以过去30日收益率标准差
    returns_std_10 = returns.rolling(10).std()
    returns_std_30 = returns.rolling(30).std()
    feat['vts'] = returns_std_10 / (returns_std_30 + eps)

    # 7. 量能聚集度 (vcll): T日成交量除以过去20日成交量移动平均
    feat['vcll'] = vol / (vol_ma_20 + eps)

    # 8. 收益率波动比 (rvr): T日收益率除以过去20日收益率标准差
    feat['rvr'] = returns / (returns_std_20 + eps)

    # 9. 筹码松动度 (csl): T日收盘价减去其60日移动平均，再除以60日收盘价移动平均
    close_ma_60 = close.rolling(60).mean()
    feat['csl'] = (close - close_ma_60) / (close_ma_60 + eps)

    # 10. 反转效应强度 (rev): 过去20日今日收益率与昨日收益率的相关系数
    returns_yesterday = returns.shift(1)
    feat['rev'] = returns.rolling(20).corr(returns_yesterday)

    # 11. 波动率聚集度 (vclb): 过去5日收益率标准差除以过去20日收益率标准差
    returns_std_5 = returns.rolling(5).std()
    feat['vclb'] = returns_std_5 / (returns_std_20 + eps)

    # 12. 收益分布峰度 (kurtosis): 过去20日收益率的四阶标准化矩
    feat['kurtosis'] = returns_normalized.rolling(20).apply(lambda x: x.pow(4).mean(), raw=False)

    # 13. 流动性冲击持续性 (lip): 过去三天的流动性冲击系数的加权和
    # 使用指数衰减权重：[0.5, 0.3, 0.2] 对应最近3天
    # 注意：需要先计算liq，然后再计算lip
    liq_shift1 = feat['liq'].shift(1)
    liq_shift2 = feat['liq'].shift(2)
    feat['lip'] = 0.5 * feat['liq'] + 0.3 * liq_shift1.fillna(0) + 0.2 * liq_shift2.fillna(0)

    return feat

def create_features_symbol_fast_based_on_feature_txt(gp: pd.DataFrame, rebalance_period: int, min_history_days: int = 60, use_improved_features: bool = False):
    """
    统一的特征生成函数，基于feature.txt的特征定义创建特征
    仅使用历史滚动值（rolling/ewm/shift），无未来信息泄露

    **重要**: 为避免未来函数，所有基础价格特征都使用T-1日的数据
    即：在T日做决策时，只能使用截至T-1日收盘的所有数据

    Parameters:
    - gp: 单个symbol的K线数据
    - rebalance_period: 重平衡周期
    - min_history_days: 最小历史天数
    - use_improved_features: 如果True，使用精简版特征（23个特征，减少过拟合风险）
                           如果False，使用完整版特征（100+个特征，基于feature.txt）
    """
    eps = 1e-8
    gp = gp.sort_values('date').reset_index(drop=True).copy()

    if len(gp) < min_history_days + rebalance_period + 20:
        return pd.DataFrame()

    # 🔥 关键修改：使用T-1日的数据作为特征，避免未来函数
    # 在T日做决策时，只能使用截至T-1日的所有数据
    close = gp['close'].shift(1)
    high  = gp['high'].shift(1)
    low   = gp['low'].shift(1)
    open = gp['open'].shift(1)
    open_raw = gp['open']  # 保留未shift的原始open，用于计算gap特征
    vol   = gp['volume'].shift(1)
    qv    = gp['quote_volume'].shift(1) if 'quote_volume' in gp.columns else None
    tbq   = gp['taker_buy_quote'].shift(1) if 'taker_buy_quote' in gp.columns else None
    tbb   = gp['taker_buy_base'].shift(1) if 'taker_buy_base' in gp.columns else None
    trades_count = gp['trades_count'].shift(1) if 'trades_count' in gp.columns else None

    feat = pd.DataFrame({
        'date': gp['date'],
        'symbol': gp['symbol']
    })

    # ========== 精简版特征（use_improved_features=True）==========
    if use_improved_features:
        # 1. 基础价格动量特征（减少窗口数量）
        for window in [5, 10, 20]:
            # 价格动量
            feat[f'momentum_{window}'] = close.pct_change(window)
            # 波动率调整动量
            volatility = close.pct_change().rolling(window).std()
            feat[f'momentum_vol_adj_{window}'] = feat[f'momentum_{window}'] / (volatility + eps)

            # 高低波动特征
            high_low_ratio = (high / low - 1).rolling(window).mean()
            feat[f'range_efficiency_{window}'] = feat[f'momentum_{window}'] / (high_low_ratio + eps)

        # 2. 量价关系特征
        for window in [5, 10]:
            # 量价背离
            price_change = close.pct_change(window)
            volume_change = vol.pct_change(window)
            feat[f'price_volume_divergence_{window}'] = price_change - volume_change

            # 成交量加权价格
            vwap = (vol * (high + low + close) / 3).rolling(window).sum() / (vol.rolling(window).sum() + eps)
            feat[f'vwap_premium_{window}'] = (close - vwap) / close

        # 3. 均值回复特征
        for window in [10, 20]:
            # 价格与均线的偏离
            ma = close.rolling(window).mean()
            feat[f'ma_deviation_{window}'] = (close - ma) / ma

            # RSI改进版本
            returns = close.pct_change()
            gain = returns.clip(lower=0).rolling(window).mean()
            loss = (-returns.clip(upper=0)).rolling(window).mean()
            rs = gain / (loss + eps)
            feat[f'rsi_{window}'] = 100 - 100 / (1 + rs)

        # 4. 市场微观结构特征
        if tbq is not None and qv is not None:
            for window in [5, 10]:
                # 资金流强度
                money_flow = (tbq - (qv - tbq)) / (qv + eps)
                feat[f'money_flow_{window}'] = money_flow.rolling(window).mean()

                # 大单净流入
                large_order_ratio = (tbq / (qv + eps)).rolling(window).mean()
                feat[f'large_order_ratio_{window}'] = large_order_ratio

        # 5. 技术指标特征
        # MACD信号
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        feat['macd_signal'] = macd.ewm(span=9).mean()

        # 布林带位置
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        feat['bollinger_position_20'] = (close - bb_middle) / (2 * bb_std + eps)

        # 6. Multi-Transformer量价时序特征（13个特征）
        feat = add_multi_transformer_features(feat, close, open, vol, open_raw=open_raw, eps=eps)

        # 目标变量: 注意close已经是shift(1)，所以这里相当于预测从T日到T+rebalance_period日的收益
        # 即：用T-1日及之前的特征，预测T日到T+rebalance_period日的收益
        feat["future_ret"] = np.log(gp['close'].shift(-rebalance_period) / gp['close'])

        # 清理数据
        feat = feat.iloc[min_history_days:-rebalance_period].dropna().reset_index(drop=True)
        return feat

    # ========== 完整版特征（use_improved_features=False，基于feature.txt）==========

    # 基础价格特征（已经通过shift(1)处理，使用T-1日数据）
    feat['open']   = open
    feat['high']   = high
    feat['low']    = low
    feat['close']  = close
    feat['volume'] = vol

    if qv is not None:
        feat['quote_volume'] = qv
        feat['vwap'] = qv / (vol + eps)
    else:
        feat['vwap'] = close

    # ===== 特征1: 累积主买/累积主卖比率因子 =====
    if tbq is not None and qv is not None:
        tbs = qv - tbq  # 主动卖盘
        # 收盘为涨的日期：累积主动买盘
        up_days = close > close.shift(1)
        cum_buy_up = (tbq * up_days).rolling(window=20, min_periods=1).sum()
        cum_sell_down = (tbs * (~up_days)).rolling(window=20, min_periods=1).sum()
        feat['cum_buy_sell_ratio'] = cum_buy_up / (cum_sell_down + eps)

    # ===== 特征2: 买入稳定因子 =====
    if tbq is not None and qv is not None:
        buy_ratio = tbq / (qv + eps)
        for n in (5, 10, 20, 30):
            buy_ratio_mean = buy_ratio.rolling(n).mean()
            buy_ratio_std = buy_ratio.rolling(n).std()
            feat[f'buy_stability_{n}'] = buy_ratio_mean / (buy_ratio_std + eps)

    # ===== 特征3: 方向动量因子 =====
    for window in (5, 10, 20, 30):
        up_momentum = (close.diff().clip(lower=0)).rolling(window).sum()
        down_momentum = ((-close.diff()).clip(lower=0)).rolling(window).sum()
        feat[f'directional_momentum_{window}'] = up_momentum / (down_momentum + eps)

    # ===== 特征4: 定向波动率因子 =====
    for window in (5, 10, 20, 30):
        up_volatility = ((high - open) / (open + eps)).rolling(window).mean()
        down_volatility = ((open - low) / (open + eps)).rolling(window).mean()
        up_roc = (high - open.shift(window)) / (open.shift(window) + eps)
        down_roc = (open.shift(window) - low) / (open.shift(window) + eps)
        feat[f'directional_volatility_{window}'] = (up_roc / (up_volatility + eps)) - (down_roc / (down_volatility + eps))

    # ===== 特征5: 高波动性动量 =====
    for window in (5, 10, 20, 30):
        roc = close.pct_change(window)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        range_pct = (high - low) / (close + eps)
        feat[f'hv_momentum_product_{window}'] = roc * atr
        feat[f'hv_momentum_ratio_{window}'] = roc / (range_pct + eps)

    # ===== 特征6: 动量-主动买入极值 =====
    if tbq is not None:
        for window in (10, 20, 30):
            momentum = close.pct_change(window)
            # 简化实现：使用滚动窗口内的相关性或分位数
            tbq_rank = tbq.rolling(window).rank(pct=True)
            feat[f'momentum_tbq_extreme_{window}'] = momentum * tbq_rank

    # ===== 特征7: 动量-成交量极值分组因子 =====
    for window in (10, 20, 30):
        momentum = close.pct_change(window)
        vol_rank = vol.rolling(window).rank(pct=True)
        feat[f'momentum_volume_extreme_{window}'] = momentum * vol_rank

    # ===== 特征8: net_order_flow =====
    if tbq is not None and qv is not None:
        for window in (5, 10, 20, 30):
            daily_flow = tbq - (qv - tbq)  # taker_buy_quote - taker_sell_quote
            net_flow = daily_flow.rolling(window).sum() / (qv.rolling(window).sum() + eps)
            feat[f'net_order_flow_{window}'] = net_flow

    # ===== 特征9: 涨跌距离/路径（价格路径效率） =====
    for window in (5, 10, 20, 30):
        net_ret = close.pct_change(window)
        path_len = close.pct_change().abs().rolling(window).sum()
        feat[f'price_path_efficiency_{window}'] = net_ret / (path_len + eps)

    # ===== 特征10: rolling_corr_buyquote =====
    if tbq is not None:
        for window in (5, 10, 20, 30):
            ret = np.log(close / close.shift(1))
            corr = ret.rolling(window).corr(tbq)
            vol_mean = vol.rolling(window).mean()
            feat[f'rolling_corr_buyquote_{window}'] = corr * np.log(1 + vol_mean)

    # ===== 特征11: RSI 截面排序因子 =====
    for period in (14, 21, 30):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + eps)
        rsi = 100 - 100 / (1 + rs)
        feat[f'rsi_{period}'] = rsi

    # ===== 特征12: TBQ极值效率 =====
    if tbq is not None:
        for window in (10, 20, 30):
            roc = close.pct_change(window)
            range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
            vol_eff = roc / (range_pct + eps)
            tbq_rank = tbq.rolling(window).rank(pct=True)
            feat[f'tbq_extreme_efficiency_{window}'] = vol_eff * tbq_rank

    # ===== 特征13: Top-Bottom Trades Ratio =====
    if trades_count is not None and tbq is not None and qv is not None:
        for window in (10, 20, 30):
            buy_ratio = tbq / (qv + eps)
            trades_rank = trades_count.rolling(window).rank(pct=True)
            feat[f'top_bottom_trades_ratio_{window}'] = buy_ratio * trades_rank

    # ===== 特征14: 波动效率 =====
    for window in (5, 10, 20, 30):
        roc = close.pct_change(window)
        range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
        feat[f'volatility_efficiency_{window}'] = roc / (range_pct + eps)

    # ===== 特征15: 波动率-订单流 =====
    if trades_count is not None and qv is not None:
        for window in (5, 10, 20, 30):
            roc = close.pct_change(window)
            range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
            vol_eff = roc / (range_pct + eps)
            avg_trade_size = qv / (trades_count + eps)
            trades_ratio = trades_count / (trades_count.rolling(window).mean() + eps)
            avg_trade_size_ratio = avg_trade_size / (avg_trade_size.rolling(window).mean() + eps)
            order_flow = trades_ratio / (avg_trade_size_ratio + eps)
            feat[f'volatility_order_flow_{window}'] = vol_eff * order_flow

    # ===== 特征16: 量稳因子（成交量稳定性） =====
    if trades_count is not None:
        for lookback_days in (10, 20, 30):
            volume_mean = vol.rolling(lookback_days).mean()
            volume_std = vol.rolling(lookback_days).std()
            feat[f'volume_stability_{lookback_days}'] = volume_mean / (volume_std + eps)

    # ===== 特征17: 成交量切分-主动买入Base 极值 =====
    if tbb is not None:
        for window in (10, 20, 30):
            tb_base_mom = tbb.rolling(10).sum()
            vol_rank = vol.rolling(window).rank(pct=True)
            feat[f'volume_tbb_extreme_{window}'] = tb_base_mom * vol_rank
    # =====================================================================

    # ===== 特征18: 买卖强度稳定性 =====
    if tbq is not None and tbb is not None and qv is not None:
        for window in (5, 10, 20, 30):
            buy_intensity = (tbq + tbb) / (qv + eps)  # 总主动买入强度
            buy_intensity_std = buy_intensity.rolling(window).std()
            buy_intensity_mean = buy_intensity.rolling(window).mean()
            feat[f'buy_intensity_stability_{window}'] = buy_intensity_mean / (buy_intensity_std + eps)
    # ===== 特征19: 价格路径的平滑度 =====
    for window in (5, 10, 20, 30):
        ma = close.rolling(window).mean()
        deviation = (close - ma) / ma
        deviation_std = deviation.rolling(window).std()
        feat[f'price_path_stability_{window}'] = 1 / (deviation_std + eps)

        # 或者用路径效率：实际收益 / 路径长度
        net_return = close.pct_change(window)
        path_length = close.pct_change().abs().rolling(window).sum()
        feat[f'price_efficiency_stability_{window}'] = net_return / (path_length + eps)

    # ===== 特征20: 成交笔数波动的稳定性 =====
    if trades_count is not None:
        for window in (5, 10, 20, 30):
            trades_std = trades_count.rolling(window).std()
            trades_mean = trades_count.rolling(window).mean()
            feat[f'trades_frequency_stability_{window}'] = trades_mean / (trades_std + eps)

            if vol is not None:
                avg_trade_size = vol / (trades_count + eps)
                trade_size_std = avg_trade_size.rolling(window).std()
                trade_size_mean = avg_trade_size.rolling(window).mean()
                feat[f'trade_size_stability_{window}'] = trade_size_mean / (trade_size_std + eps)

    # ===== 特征21: 买卖压力的稳定性（主动买入 vs 主动卖出） =====
    if tbq is not None and qv is not None:
        for window in (5, 10, 20, 30):
            buy_pressure = tbq / (qv + eps)  # 主动买入比例
            sell_pressure = (qv - tbq) / (qv + eps)  # 主动卖出比例

            pressure_diff = buy_pressure - sell_pressure
            pressure_diff_std = pressure_diff.rolling(window).std()
            pressure_diff_mean = pressure_diff.rolling(window).mean()
            feat[f'pressure_balance_stability_{window}'] = pressure_diff_mean / (pressure_diff_std + eps)

            pressure_ratio = buy_pressure / (sell_pressure + eps)
            pressure_ratio_std = pressure_ratio.rolling(window).std()
            pressure_ratio_mean = pressure_ratio.rolling(window).mean()
            feat[f'pressure_ratio_stability_{window}'] = pressure_ratio_mean / (pressure_ratio_std + eps)

    # ===== Multi-Transformer量价时序特征（13个特征） =====
    feat = add_multi_transformer_features(feat, close, open, vol, eps)

    # 然后是目标变量
    feat["future_ret"] = np.log(gp['close'].shift(-rebalance_period) / gp['close'])

    # 改为：
    feat = feat.iloc[min_history_days : -rebalance_period].copy()
    # 只删除future_ret是NaN的行（因为这是目标变量，必须有效）
    feat = feat[~feat['future_ret'].isna()].copy()
    # 对于特征列的NaN，使用前值填充（更保守的处理）
    feature_cols = [col for col in feat.columns if col not in ['date', 'symbol', 'future_ret']]
    feat[feature_cols] = feat[feature_cols].fillna(method='ffill').fillna(0)
    feat = feat.reset_index(drop=True)
    return feat

def create_features_symbol_inference_latest(gp: pd.DataFrame, min_history_days: int = 60,
                                           use_improved_features: bool = False) -> pd.DataFrame:
    """
    仅用于推理：构造与训练一致的特征，但不生成future_ret，也不丢弃末尾。
    返回该symbol在其各自最新日期的一行特征。
    使用基于feature.txt的特征定义。

    **重要**: 为避免未来函数，所有基础价格特征都使用T-1日的数据
    推理时：在最新日期T做预测时，使用截至T-1日收盘的所有数据

    Parameters:
    - gp: 单个symbol的K线数据
    - min_history_days: 最小历史天数
    - use_improved_features: 是否使用改进的特征工程（需与训练时一致）
    """
    eps = 1e-8
    gp = gp.sort_values('date').reset_index(drop=True).copy()

    # 推理时只需要min_history_days的历史数据，不需要rebalance_period的未来数据
    if len(gp) < min_history_days + 20:
        return pd.DataFrame()

    # 🔥 关键修改：使用T-1日的数据作为特征，避免未来函数（与训练时保持一致）
    close = gp['close'].shift(1)
    high  = gp['high'].shift(1)
    low   = gp['low'].shift(1)
    open = gp['open'].shift(1)
    open_raw = gp['open']  # 保留未shift的原始open，用于计算gap特征
    vol   = gp['volume'].shift(1)
    qv    = gp['quote_volume'].shift(1) if 'quote_volume' in gp.columns else None
    tbq   = gp['taker_buy_quote'].shift(1) if 'taker_buy_quote' in gp.columns else None
    tbb   = gp['taker_buy_base'].shift(1) if 'taker_buy_base' in gp.columns else None
    trades_count = gp['trades_count'].shift(1) if 'trades_count' in gp.columns else None

    feat = pd.DataFrame({
        'date': gp['date'],
        'symbol': gp['symbol']
    })

    # ========== 精简版特征（use_improved_features=True）==========
    if use_improved_features:
        # 1. 基础价格动量特征（减少窗口数量）
        for window in [5, 10, 20]:
            # 价格动量
            feat[f'momentum_{window}'] = close.pct_change(window)
            # 波动率调整动量
            volatility = close.pct_change().rolling(window).std()
            feat[f'momentum_vol_adj_{window}'] = feat[f'momentum_{window}'] / (volatility + eps)

            # 高低波动特征
            high_low_ratio = (high / low - 1).rolling(window).mean()
            feat[f'range_efficiency_{window}'] = feat[f'momentum_{window}'] / (high_low_ratio + eps)

        # 2. 量价关系特征
        for window in [5, 10]:
            # 量价背离
            price_change = close.pct_change(window)
            volume_change = vol.pct_change(window)
            feat[f'price_volume_divergence_{window}'] = price_change - volume_change

            # 成交量加权价格
            vwap = (vol * (high + low + close) / 3).rolling(window).sum() / (vol.rolling(window).sum() + eps)
            feat[f'vwap_premium_{window}'] = (close - vwap) / close

        # 3. 均值回复特征
        for window in [10, 20]:
            # 价格与均线的偏离
            ma = close.rolling(window).mean()
            feat[f'ma_deviation_{window}'] = (close - ma) / ma

            # RSI改进版本
            returns = close.pct_change()
            gain = returns.clip(lower=0).rolling(window).mean()
            loss = (-returns.clip(upper=0)).rolling(window).mean()
            rs = gain / (loss + eps)
            feat[f'rsi_{window}'] = 100 - 100 / (1 + rs)

        # 4. 市场微观结构特征
        if tbq is not None and qv is not None:
            for window in [5, 10]:
                # 资金流强度
                money_flow = (tbq - (qv - tbq)) / (qv + eps)
                feat[f'money_flow_{window}'] = money_flow.rolling(window).mean()

                # 大单净流入
                large_order_ratio = (tbq / (qv + eps)).rolling(window).mean()
                feat[f'large_order_ratio_{window}'] = large_order_ratio

        # 5. 技术指标特征
        # MACD信号
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        feat['macd_signal'] = macd.ewm(span=9).mean()

        # 布林带位置
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        feat['bollinger_position_20'] = (close - bb_middle) / (2 * bb_std + eps)

        # 6. Multi-Transformer量价时序特征（13个特征）
        feat = add_multi_transformer_features(feat, close, open, vol, open_raw=open_raw, eps=eps)

        # 推理时不需要future_ret，也不丢弃最后几行
        # 只丢弃前面不完整的行
        feat = feat.iloc[min_history_days:].dropna().reset_index(drop=True)
        if feat.empty:
            return pd.DataFrame()
        return feat.iloc[[-1]]  # 返回最新一行

    # ========== 完整版特征（use_improved_features=False，基于feature.txt）==========

    # 基础价格特征（已经通过shift(1)处理，使用T-1日数据）
    feat['open']   = open
    feat['high']   = high
    feat['low']    = low
    feat['close']  = close
    feat['volume'] = vol

    if qv is not None:
        feat['quote_volume'] = qv
        feat['vwap'] = qv / (vol + eps)
    else:
        feat['vwap'] = close

    # ===== 特征1: 累积主买/累积主卖比率因子 =====
    if tbq is not None and qv is not None:
        tbs = qv - tbq  # 主动卖盘
        # 收盘为涨的日期：累积主动买盘
        up_days = close > close.shift(1)
        cum_buy_up = (tbq * up_days).rolling(window=20, min_periods=1).sum()
        cum_sell_down = (tbs * (~up_days)).rolling(window=20, min_periods=1).sum()
        feat['cum_buy_sell_ratio'] = cum_buy_up / (cum_sell_down + eps)

    # ===== 特征2: 买入稳定因子 =====
    if tbq is not None and qv is not None:
        buy_ratio = tbq / (qv + eps)
        for n in (5, 10, 20, 30):
            buy_ratio_mean = buy_ratio.rolling(n).mean()
            buy_ratio_std = buy_ratio.rolling(n).std()
            feat[f'buy_stability_{n}'] = buy_ratio_mean / (buy_ratio_std + eps)

    # ===== 特征3: 方向动量因子 =====
    for window in (5, 10, 20, 30):
        up_momentum = (close.diff().clip(lower=0)).rolling(window).sum()
        down_momentum = ((-close.diff()).clip(lower=0)).rolling(window).sum()
        feat[f'directional_momentum_{window}'] = up_momentum / (down_momentum + eps)

    # ===== 特征4: 定向波动率因子 =====
    for window in (5, 10, 20, 30):
        up_volatility = ((high - open) / (open + eps)).rolling(window).mean()
        down_volatility = ((open - low) / (open + eps)).rolling(window).mean()
        up_roc = (high - open.shift(window)) / (open.shift(window) + eps)
        down_roc = (open.shift(window) - low) / (open.shift(window) + eps)
        feat[f'directional_volatility_{window}'] = (up_roc / (up_volatility + eps)) - (down_roc / (down_volatility + eps))

    # ===== 特征5: 高波动性动量 =====
    for window in (5, 10, 20, 30):
        roc = close.pct_change(window)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        range_pct = (high - low) / (close + eps)
        feat[f'hv_momentum_product_{window}'] = roc * atr
        feat[f'hv_momentum_ratio_{window}'] = roc / (range_pct + eps)

    # ===== 特征6: 动量-主动买入极值 =====
    if tbq is not None:
        for window in (10, 20, 30):
            momentum = close.pct_change(window)
            # 简化实现：使用滚动窗口内的相关性或分位数
            tbq_rank = tbq.rolling(window).rank(pct=True)
            feat[f'momentum_tbq_extreme_{window}'] = momentum * tbq_rank

    # ===== 特征7: 动量-成交量极值分组因子 =====
    for window in (10, 20, 30):
        momentum = close.pct_change(window)
        vol_rank = vol.rolling(window).rank(pct=True)
        feat[f'momentum_volume_extreme_{window}'] = momentum * vol_rank

    # ===== 特征8: net_order_flow =====
    if tbq is not None and qv is not None:
        for window in (5, 10, 20, 30):
            daily_flow = tbq - (qv - tbq)  # taker_buy_quote - taker_sell_quote
            net_flow = daily_flow.rolling(window).sum() / (qv.rolling(window).sum() + eps)
            feat[f'net_order_flow_{window}'] = net_flow

    # ===== 特征9: 涨跌距离/路径（价格路径效率） =====
    for window in (5, 10, 20, 30):
        net_ret = close.pct_change(window)
        path_len = close.pct_change().abs().rolling(window).sum()
        feat[f'price_path_efficiency_{window}'] = net_ret / (path_len + eps)

    # ===== 特征10: rolling_corr_buyquote =====
    if tbq is not None:
        for window in (5, 10, 20, 30):
            ret = np.log(close / close.shift(1))
            corr = ret.rolling(window).corr(tbq)
            vol_mean = vol.rolling(window).mean()
            feat[f'rolling_corr_buyquote_{window}'] = corr * np.log(1 + vol_mean)

    # ===== 特征11: RSI 截面排序因子 =====
    for period in (14, 21, 30):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + eps)
        rsi = 100 - 100 / (1 + rs)
        feat[f'rsi_{period}'] = rsi

    # ===== 特征12: TBQ极值效率 =====
    if tbq is not None:
        for window in (10, 20, 30):
            roc = close.pct_change(window)
            range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
            vol_eff = roc / (range_pct + eps)
            tbq_rank = tbq.rolling(window).rank(pct=True)
            feat[f'tbq_extreme_efficiency_{window}'] = vol_eff * tbq_rank

    # ===== 特征13: Top-Bottom Trades Ratio =====
    if trades_count is not None and tbq is not None and qv is not None:
        for window in (10, 20, 30):
            buy_ratio = tbq / (qv + eps)
            trades_rank = trades_count.rolling(window).rank(pct=True)
            feat[f'top_bottom_trades_ratio_{window}'] = buy_ratio * trades_rank

    # ===== 特征14: 波动效率 =====
    for window in (5, 10, 20, 30):
        roc = close.pct_change(window)
        range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
        feat[f'volatility_efficiency_{window}'] = roc / (range_pct + eps)

    # ===== 特征15: 波动率-订单流 =====
    if trades_count is not None and qv is not None:
        for window in (5, 10, 20, 30):
            roc = close.pct_change(window)
            range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
            vol_eff = roc / (range_pct + eps)
            avg_trade_size = qv / (trades_count + eps)
            trades_ratio = trades_count / (trades_count.rolling(window).mean() + eps)
            avg_trade_size_ratio = avg_trade_size / (avg_trade_size.rolling(window).mean() + eps)
            order_flow = trades_ratio / (avg_trade_size_ratio + eps)
            feat[f'volatility_order_flow_{window}'] = vol_eff * order_flow

    # ===== 特征16: 量稳因子（成交量稳定性） =====
    if trades_count is not None:
        for lookback_days in (10, 20, 30):
            volume_mean = vol.rolling(lookback_days).mean()
            volume_std = vol.rolling(lookback_days).std()
            feat[f'volume_stability_{lookback_days}'] = volume_mean / (volume_std + eps)

    # ===== 特征17: 成交量切分-主动买入Base 极值 =====
    if tbb is not None:
        for window in (10, 20, 30):
            tb_base_mom = tbb.rolling(10).sum()
            vol_rank = vol.rolling(window).rank(pct=True)
            feat[f'volume_tbb_extreme_{window}'] = tb_base_mom * vol_rank
    # =====================================================================

    # ===== 特征18: 买卖强度稳定性 =====
    if tbq is not None and tbb is not None and qv is not None:
        for window in (5, 10, 20, 30):
            buy_intensity = (tbq + tbb) / (qv + eps)  # 总主动买入强度
            buy_intensity_std = buy_intensity.rolling(window).std()
            buy_intensity_mean = buy_intensity.rolling(window).mean()
            feat[f'buy_intensity_stability_{window}'] = buy_intensity_mean / (buy_intensity_std + eps)
    # ===== 特征19: 价格路径的平滑度 =====
    for window in (5, 10, 20, 30):
        ma = close.rolling(window).mean()
        deviation = (close - ma) / ma
        deviation_std = deviation.rolling(window).std()
        feat[f'price_path_stability_{window}'] = 1 / (deviation_std + eps)

        # 或者用路径效率：实际收益 / 路径长度
        net_return = close.pct_change(window)
        path_length = close.pct_change().abs().rolling(window).sum()
        feat[f'price_efficiency_stability_{window}'] = net_return / (path_length + eps)

    # ===== 特征20: 成交笔数波动的稳定性 =====
    if trades_count is not None:
        for window in (5, 10, 20, 30):
            trades_std = trades_count.rolling(window).std()
            trades_mean = trades_count.rolling(window).mean()
            feat[f'trades_frequency_stability_{window}'] = trades_mean / (trades_std + eps)

            if vol is not None:
                avg_trade_size = vol / (trades_count + eps)
                trade_size_std = avg_trade_size.rolling(window).std()
                trade_size_mean = avg_trade_size.rolling(window).mean()
                feat[f'trade_size_stability_{window}'] = trade_size_mean / (trade_size_std + eps)

    # ===== 特征21: 买卖压力的稳定性（主动买入 vs 主动卖出） =====
    if tbq is not None and qv is not None:
        for window in (5, 10, 20, 30):
            buy_pressure = tbq / (qv + eps)  # 主动买入比例
            sell_pressure = (qv - tbq) / (qv + eps)  # 主动卖出比例

            pressure_diff = buy_pressure - sell_pressure
            pressure_diff_std = pressure_diff.rolling(window).std()
            pressure_diff_mean = pressure_diff.rolling(window).mean()
            feat[f'pressure_balance_stability_{window}'] = pressure_diff_mean / (pressure_diff_std + eps)

            pressure_ratio = buy_pressure / (sell_pressure + eps)
            pressure_ratio_std = pressure_ratio.rolling(window).std()
            pressure_ratio_mean = pressure_ratio.rolling(window).mean()
            feat[f'pressure_ratio_stability_{window}'] = pressure_ratio_mean / (pressure_ratio_std + eps)

    # ===== Multi-Transformer量价时序特征（13个特征） =====
    feat = add_multi_transformer_features(feat, close, open, vol, eps)

    # 推理时不需要future_ret，也不丢弃最后几行
    # 只丢弃前面不完整的行
    feat = feat.iloc[min_history_days:].dropna().reset_index(drop=True)
    if feat.empty:
        return pd.DataFrame()
    return feat.iloc[[-1]]  # 返回最新一行

def _get_selected_features_from_selector(selector):
    """从选择器中提取被选择的特征索引"""
    if hasattr(selector, 'selected_indices'):
        return selector.selected_indices
    elif hasattr(selector, 'pre_selector') and hasattr(selector.pre_selector, 'get_support'):
        selected_pre = np.where(selector.pre_selector.get_support())[0]
        if hasattr(selector, 'selected_pre_indices'):
            return selected_pre[selector.selected_pre_indices]
        else:
            return selected_pre
    else:
        return np.array([])

def bayesian_optimize_xgboost(X_train, y_train, X_val, y_val, dates_val=None, symbols_val=None,
                              n_calls=50, n_random_starts=10,
                              ic_ir_weight=1.0, rankIC=1.0, ir_weight=1.0, monotonicity_weight=1.0,
                              horizon_days=None,
                              bayes_objective='ic_ir'):
    """
    使用贝叶斯优化寻找最优的XGBoost参数

    Parameters:
    - bayes_objective: 目标函数模式
        'ic_ir'    (推荐) 单指标：仅最大化非重叠IC_IR，无需手工调权重，从根本上
                   避免「权重本身通过验证集调出」导致的验证泄露。
        'weighted' (兼容) 多指标加权和：IC_IR * ic_ir_weight + |IC| * rankIC +
                   IR * ir_weight + 单调性 * monotonicity_weight。
                   注意：若这四个权重通过实验调出，验证集IC不再是真正的样本外指标。
    - ic_ir_weight, rankIC, ir_weight, monotonicity_weight:
        仅在 bayes_objective='weighted' 时生效；默认均为 1.0（等权），
        避免手工调出的不等权把数据挖掘偏差引入目标函数。
    - horizon_days: 前瞻收益的天数（=rebalance_period），用于对日IC序列做非重叠
        子采样，校正重叠标签导致的IC_IR虚高。

    Returns:
    - best_params: 最优参数字典
    - best_score: 最优验证得分
    """
    if not SKOPT_AVAILABLE:
        print("⚠️ Scikit-optimize不可用，使用默认参数")
        return get_default_xgb_params(), None

    if bayes_objective == 'ic_ir':
        print("🚀 开始贝叶斯优化XGBoost参数（目标：单指标 非重叠IC_IR，无手工权重）...")
    else:
        print("🚀 开始贝叶斯优化XGBoost参数（目标：加权和 IC_IR+IC+IR+单调性）...")
    print(f"   优化迭代次数: {n_calls}, 随机启动: {n_random_starts}, horizon_days={horizon_days}")
    if bayes_objective == 'weighted':
        print(f"   权重设置: IC_IR={ic_ir_weight}, RankIC={rankIC}, IR={ir_weight}, 单调性={monotonicity_weight}")

    # 检查是否有日期和符号信息用于计算IC
    use_ic_objective = dates_val is not None and symbols_val is not None and len(dates_val) == len(y_val)
    if not use_ic_objective:
        print("⚠️ 警告: 缺少dates_val或symbols_val，将使用MAE作为目标函数")
        use_ic_objective = False

    # 定义参数搜索空间
    param_space = [
        # # 学习率
        # Real(0.05, 0.3, name='learning_rate'),

        # # 树的最大深度
        # Integer(4, 8, name='max_depth'),

        # # 最小叶子节点样本数
        # Integer(5, 50, name='min_child_weight'),

        # # gamma参数，用于控制是否后剪枝
        # Real(0.0, 2.0, name='gamma'),

        # # 训练样本的子采样比例
        # Real(0.5, 1.0, name='subsample'),

        # # 每棵树列的子采样比例
        # Real(0.5, 1.0, name='colsample_bytree'),

        # # 每级树列的子采样比例
        # Real(0.5, 1.0, name='colsample_bynode'),

        # # L1正则化参数（扩大搜索范围）
        # Real(5.0, 30.0, name='reg_alpha'),   # 从(2.0, 10.0)扩大到(5.0, 30.0)

        # # L2正则化参数（扩大搜索范围）
        # Real(10.0, 50.0, name='reg_lambda'), # 从(5.0, 20.0)扩大到(10.0, 50.0)
        # 学习率
        Real(0.008, 0.15, name='learning_rate'),

        # 树的最大深度
        Integer(4, 6, name='max_depth'),

        # 最小叶子节点样本数
        Integer(20, 50, name='min_child_weight'),

        # gamma参数，用于控制是否后剪枝
        Real(3.0, 8.0, name='gamma'),

        # 训练样本的子采样比例
        Real(0.1, 0.3, name='subsample'),

        # 每棵树列的子采样比例
        Real(0.1, 0.3, name='colsample_bytree'),

        # 每级树列的子采样比例
        Real(0.1, 0.3, name='colsample_bynode'),

        # L1正则化参数（扩大搜索范围）
        Real(15.0, 25.0, name='reg_alpha'),   # 从(2.0, 10.0)扩大到(5.0, 30.0)

        # L2正则化参数（扩大搜索范围）
        Real(25.0, 50.0, name='reg_lambda'), # 从(5.0, 20.0)扩大到(10.0, 50.0)

        # 树的数量 (固定为较大的值，由early stopping控制)
        Integer(2000, 4000, name='n_estimators'),
    ]

    # 全局变量用于存储最佳分数
    global_best_score = [-float('inf')]  # 改为负无穷，因为我们要最大化

    @use_named_args(param_space)
    def objective(**params):
        """目标函数：最大化IC_IR、IC、IR的加权和（返回负值以便gp_minimize最小化）"""
        try:
            # 提取参数
            learning_rate = params['learning_rate']
            max_depth = params['max_depth']
            min_child_weight = params['min_child_weight']
            gamma = params['gamma']
            subsample = params['subsample']
            colsample_bytree = params['colsample_bytree']
            colsample_bynode = params['colsample_bynode']
            reg_alpha = params['reg_alpha']
            reg_lambda = params['reg_lambda']
            n_estimators = params['n_estimators']

            # 设置XGBoost参数（使用Huber Loss增强鲁棒性，delta=1.0）
            # 使用全局函数，通过全局变量传递delta值
            global _huber_delta_global
            _huber_delta_global = 1.0  # 贝叶斯优化中使用固定delta=1.0

            xgb_params = {
                # 注意：使用obj参数传递自定义目标函数
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_child_weight': min_child_weight,
                'gamma': gamma,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'colsample_bynode': colsample_bynode,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'tree_method': 'hist',
                'random_state': 42,
                'n_jobs': -1,
            }

            # 创建DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            # 训练模型（不使用early stopping，因为自定义目标函数）
            # 使用 obj 参数传递自定义目标函数
            evals_result = {}
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=min(n_estimators, 500),  # 限制迭代次数避免过拟合
                verbose_eval=False,
                obj=_huber_obj_global  # 使用全局Huber Loss目标函数
            )

            # 获取验证集预测
            val_pred = model.predict(dval)

            if use_ic_objective:
                # 计算IC相关指标
                val_df = pd.DataFrame({
                    'date': dates_val,
                    'symbol': symbols_val,
                    'pred': val_pred,
                    'actual': y_val
                })

                # 按日期分组计算每日IC
                daily_ic_by_date = {}
                for date, group in val_df.groupby('date'):
                    if len(group) >= 5:  # 至少5个样本才计算IC
                        ic = group['pred'].corr(group['actual'])
                        if not np.isnan(ic):
                            daily_ic_by_date[pd.to_datetime(date)] = float(ic)

                ic_series = pd.Series(daily_ic_by_date).sort_index() if daily_ic_by_date else pd.Series(dtype=float)
                if horizon_days is not None and int(horizon_days) > 1 and not ic_series.empty:
                    ic_series = ic_series.iloc[::int(horizon_days)]

                if ic_series.shape[0] > 0:
                    # 计算IC统计量
                    ic_mean = float(ic_series.mean())
                    ic_std = float(ic_series.std(ddof=0))

                    # IC_IR = IC均值 / IC标准差
                    ic_ir = ic_mean / ic_std if ic_std > 1e-8 else ic_mean

                    # 计算分组单调性：将预测值分为5组，计算每组平均收益，然后计算斯皮尔曼相关系数
                    val_df_sorted = val_df.sort_values('pred', ascending=False)
                    n_groups = 5
                    group_size = len(val_df_sorted) // n_groups

                    group_returns = []
                    for i in range(n_groups):
                        start_idx = i * group_size
                        end_idx = (i + 1) * group_size if i < n_groups - 1 else len(val_df_sorted)
                        group_data = val_df_sorted.iloc[start_idx:end_idx]
                        group_return = group_data['actual'].mean()
                        group_returns.append(group_return)

                    # 计算分组单调性（斯皮尔曼相关系数：组序号 vs 组收益）
                    from scipy import stats as scipy_stats
                    group_ranks = list(range(1, n_groups + 1))  # [1, 2, 3, 4, 5]
                    if len(group_returns) == n_groups:
                        monotonicity_corr, _ = scipy_stats.spearmanr(group_ranks, group_returns)
                        monotonicity = abs(monotonicity_corr)  # 取绝对值，确保为正
                    else:
                        monotonicity = 0

                    # IR = 年化收益率 / 年化波动率
                    # 简化计算：使用多空组合收益（前20% - 后20%）
                    n_top = max(1, int(len(val_df_sorted) * 0.2))
                    n_bottom = max(1, int(len(val_df_sorted) * 0.2))

                    top_returns = val_df_sorted.head(n_top)['actual'].mean()
                    bottom_returns = val_df_sorted.tail(n_bottom)['actual'].mean()
                    long_short_return = top_returns - bottom_returns

                    # 计算多空组合的波动率（按日期分组）
                    daily_ls_by_date = {}
                    for date, group in val_df.groupby('date'):
                        if len(group) >= 10:  # 至少10个样本才计算
                            group_sorted = group.sort_values('pred', ascending=False)
                            n_top_daily = max(1, int(len(group_sorted) * 0.2))
                            n_bottom_daily = max(1, int(len(group_sorted) * 0.2))
                            top_ret = group_sorted.head(n_top_daily)['actual'].mean()
                            bottom_ret = group_sorted.tail(n_bottom_daily)['actual'].mean()
                            daily_ls_by_date[pd.to_datetime(date)] = float(top_ret - bottom_ret)

                    ls_series = pd.Series(daily_ls_by_date).sort_index() if daily_ls_by_date else pd.Series(dtype=float)
                    if horizon_days is not None and int(horizon_days) > 1 and not ls_series.empty:
                        ls_series = ls_series.iloc[::int(horizon_days)]

                    if ls_series.shape[0] > 1:
                        ls_return_mean = float(ls_series.mean())
                        ls_return_std = float(ls_series.std(ddof=0))
                        # 年化IR（假设252个交易日）
                        ir = (ls_return_mean / ls_return_std) * np.sqrt(252) if ls_return_std > 1e-8 else 0
                    else:
                        ir = 0

                    # 加权和（使用绝对值确保为正）
                    weighted_score = (ic_ir_weight * abs(ic_ir) +
                                     rankIC * abs(ic_mean) +
                                     ir_weight * abs(ir) +
                                     monotonicity_weight * monotonicity)

                    # 更新全局最佳分数（取负值，因为gp_minimize是最小化）
                    if weighted_score > global_best_score[0]:
                        global_best_score[0] = weighted_score

                    # 返回负值，因为gp_minimize是最小化，我们要最大化weighted_score
                    return -weighted_score
                else:
                    # 没有有效的IC数据，返回一个很大的正值（表示很差）
                    return 1e6
            else:
                # 回退到MAE
                val_mae = evals_result['val']['mae'][-1]
                if val_mae < abs(global_best_score[0]):
                    global_best_score[0] = -val_mae  # 存储负值以保持一致性
            return val_mae

        except Exception as e:
            print(f"参数组合训练失败: {e}")
            return float('inf')  # 返回无穷大表示失败

    # 执行贝叶斯优化
    try:
        result = gp_minimize(
            objective,
            param_space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            random_state=42,
            verbose=True
        )

        # 提取最优参数
        best_params = {
            'learning_rate': result.x[0],
            'max_depth': result.x[1],
            'min_child_weight': result.x[2],
            'gamma': result.x[3],
            'subsample': result.x[4],
            'colsample_bytree': result.x[5],
            'colsample_bynode': result.x[6],
            'reg_alpha': result.x[7],
            'reg_lambda': result.x[8],
            'n_estimators': result.x[9],
        }

        # 添加固定参数（使用Huber Loss，delta=1.0）
        # 设置全局delta值
        global _huber_delta_global
        _huber_delta_global = 1.0

        best_params.update({
            # 注意：不在params中设置objective和eval_metric，而是在xgb.train()中使用obj和custom_metric参数
            'early_stopping_rounds': 300,
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
        })

        print("✅ 贝叶斯优化完成!")
        if use_ic_objective:
            # result.fun是负值（因为我们返回的是-weighted_score），需要取绝对值
            best_score = abs(result.fun)
            print(f"   最优加权IC指标: {best_score:.6f}")
            print(f"   (IC_IR权重={ic_ir_weight}, RankIC权重={rankIC}, IR权重={ir_weight}, 单调性权重={monotonicity_weight})")
        else:
            best_score = result.fun
            print(f"   最优验证MAE: {best_score:.6f}")
        print(f"   最优参数: learning_rate={best_params['learning_rate']:.4f}, "
              f"max_depth={best_params['max_depth']}, "
              f"min_child_weight={best_params['min_child_weight']}")

        return best_params, best_score

    except Exception as e:
        print(f"❌ 贝叶斯优化失败: {e}")
        print("   使用默认参数")
        return get_default_xgb_params(), None

def prepare_robust_data(df, rebalance_period=5, min_history_days=60, n_jobs=-1,
                        cross_sectional_standardize=True, use_rank_target=True, use_improved_features=False,
                        available_tokens_by_date=None, use_ga_features=False):
    """
    更稳健的数据预处理
    关键改进点：
    1. 对特征进行更严格的异常值处理
    2. 使用中位数和MAD进行稳健标准化
    3. 对目标变量使用更保守的处理方法

    🔥 关键修复：先计算特征（在全量连续数据上），再过滤（只保留Top50期间的样本）
    这样可以避免K线数据不连续导致的特征计算错误
    """
    from joblib import Parallel, delayed
    from scipy import stats

    symbols = df['symbol'].unique().tolist()

    # 预先构建快速查询集 (date -> set of symbols) 用于高效过滤
    if available_tokens_by_date is not None:
        available_tokens_sets = {}
        for d, syms in available_tokens_by_date.items():
            if isinstance(d, pd.Timestamp):
                available_tokens_sets[d] = set(syms)
            else:
                available_tokens_sets[pd.to_datetime(d)] = set(syms)
    else:
        available_tokens_sets = None

    def run_one(sym):
        gp = df[df['symbol'] == sym].copy()
        if gp.empty: return pd.DataFrame()

        # 🔥 关键修复：1. 先计算特征（此时gp是完整的连续时间序列）
        feat_df = create_features_symbol_fast_based_on_feature_txt(gp, rebalance_period, min_history_days, use_improved_features)

        if feat_df.empty: return pd.DataFrame()

        # 🔥 关键修复：2. 计算完后再过滤（只保留Top50期间的样本）
        if available_tokens_sets is not None:
            # 向量化过滤：只保留该symbol在Top50名单里的日期的行
            # 构建日期到可用symbol集合的映射（已排序）
            sorted_dates = sorted(available_tokens_sets.keys())

            # 将feat_df的日期转换为Timestamp
            feat_df['_date_ts'] = pd.to_datetime(feat_df['date'])

            # 使用merge_asof或直接查找最接近的日期
            valid_mask = []
            for dt in feat_df['_date_ts']:
                # 找到最接近的可用日期（小于等于当前日期）
                available_date = None
                for d in sorted_dates:
                    if d <= dt:
                        available_date = d
                    else:
                        break
                if available_date is not None:
                    valid_mask.append(sym in available_tokens_sets[available_date])
                else:
                    valid_mask.append(False)

            feat_df = feat_df[valid_mask].drop(columns=['_date_ts'])

        return feat_df

    dfs = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(run_one)(s) for s in symbols)
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return np.array([]), np.array([]), np.array([]), [], []

    out = pd.concat(dfs, ignore_index=True)

    # 合并遗传算法因子（来自 feature.txt）
    if use_ga_features:
        ga_df = _compute_ga_factors_df(df)
        if ga_df.empty:
            print("⚠️ use_ga_features=True 但未加载到遗传算法因子，请检查 feature.txt 和 ga_top_expr_log.csv")
        else:
            ga_cols = [c for c in ga_df.columns if c not in ["date", "symbol"]]
            if ga_cols:
                out["date_norm"] = pd.to_datetime(out["date"]).dt.normalize()
                ga_df["date_norm"] = pd.to_datetime(ga_df["date"]).dt.normalize()
                out = out.merge(
                    ga_df[["date_norm", "symbol"] + ga_cols],
                    on=["date_norm", "symbol"],
                    how="left",
                    suffixes=("", "_ga"),
                )
                out = out.drop(columns=["date_norm"])
                out[ga_cols] = out.groupby("symbol", sort=False)[ga_cols].ffill().fillna(0)
                # 将 GA 因子列放在最前面，便于特征选择时优先筛选
                base_cols = [c for c in out.columns if c not in ["date", "symbol", "future_ret"] and c not in ga_cols]
                out = out[["date", "symbol"] + ga_cols + base_cols + ["future_ret"]]

    # 保存原始目标变量（用于IC计算）
    y_original = out['future_ret'].copy().to_numpy(dtype=np.float32)

    # 横截面标准化：按日期分组，使用中位数和MAD进行稳健标准化
    if cross_sectional_standardize:
        print("🔄 进行稳健横截面标准化（按日期分组，使用中位数和MAD）...")
        feature_cols = [col for col in out.columns if col not in ['date', 'symbol', 'future_ret']]

        def robust_standardize_by_date(group):
            """对每个日期的特征进行稳健标准化（使用中位数和MAD）"""
            group = group.copy()
            for col in feature_cols:
                # 使用中位数和MAD进行稳健标准化
                median_val = group[col].median()
                mad_val = (group[col] - median_val).abs().median()
                if mad_val > 1e-8:
                    # MAD标准化：MAD = median(|x - median(x)|)
                    group[col] = (group[col] - median_val) / mad_val
                else:
                    group[col] = 0.0  # 如果MAD为0，设为0
            return group

        out = out.groupby('date', group_keys=False).apply(robust_standardize_by_date)
        print(f"   已完成稳健横截面标准化，处理了 {out['date'].nunique()} 个交易日")

    # 目标变量排名转换：按日期分组，使用更保守的处理方法
    if use_rank_target:
        print("🔄 将目标变量转换为横截面排名（按日期分组，保守处理）...")
        def conservative_target_processing(group):
            """对每个日期的目标变量进行保守的排名转换"""
            group = group.copy()
            # 使用更稳健的排名方法
            rank_pct = group['future_ret'].rank(pct=True, method='average')
            # 避免极端值，使用更窄的clip范围（0.01-0.99）
            rank_pct_clipped = rank_pct.clip(0.01, 0.99)
            group['future_ret'] = stats.norm.ppf(rank_pct_clipped)
            return group

        out = out.groupby('date', group_keys=False).apply(conservative_target_processing)
        print(f"   已完成保守目标变量排名转换，处理了 {out['date'].nunique()} 个交易日")

    X = out.drop(columns=['date','symbol','future_ret']).to_numpy(dtype=np.float32)
    y = out['future_ret'].to_numpy(dtype=np.float32)
    syms = out['symbol'].tolist()
    dates = pd.to_datetime(out['date']).tolist()

    target_type = "排名" if use_rank_target else "绝对值"
    print(f"✅ 稳健特征 向量化+并行 数据样本: {len(out)}，特征维度: {X.shape[1]}，目标变量类型: {target_type}")
    print(f"数据列: {df.columns.tolist()}")
    return X, y, y_original, syms, dates


def _load_ga_feature_names(feature_txt_path: str = None) -> List[str]:
    """从 feature.txt 加载遗传算法因子名称列表"""
    path = feature_txt_path or GA_FEATURE_TXT
    if not os.path.exists(path):
        return []
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                names.append(name)
    return names


def _compute_ga_factors_df(df: pd.DataFrame, feature_txt_path: str = None,
                          ga_log_path: str = None) -> pd.DataFrame:
    """
    计算遗传算法因子（原始值，用于合并到XGBoost特征）
    返回 DataFrame 列: date, symbol, ga_feat1, ga_feat2, ...
    """
    ft_path = feature_txt_path or GA_FEATURE_TXT
    gl_path = ga_log_path or GA_EXPR_LOG
    if not os.path.exists(ft_path) or not os.path.exists(gl_path):
        return pd.DataFrame()

    # 动态导入 Genetic_Algorithm 模块（避免循环依赖）
    import sys
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    FACTOR_ANALYSE_DIR = os.path.dirname(THIS_DIR)
    GA_DIR = os.path.join(FACTOR_ANALYSE_DIR, "Genetic_Algorithm")
    if GA_DIR not in sys.path:
        sys.path.insert(0, GA_DIR)

    from fitness import _prepare_derived_features  # type: ignore
    from expr_parser import parse_expr, load_slug_to_expr_map  # type: ignore

    feature_names = _load_ga_feature_names(ft_path)
    if not feature_names:
        return pd.DataFrame()

    slug_to_expr = load_slug_to_expr_map(gl_path)
    if not slug_to_expr:
        return pd.DataFrame()

    g = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = _prepare_derived_features(g)

    result = g[["date", "symbol"]].copy()
    n_ga = 0
    for name in feature_names:
        expr_str = slug_to_expr.get(name)
        if not expr_str:
            continue
        try:
            node = parse_expr(expr_str)
            raw = node.eval(g)
            if isinstance(raw, pd.Series):
                raw_values = raw.to_numpy()
            else:
                raw_values = np.asarray(raw)
            raw_values = np.where(np.isfinite(raw_values), raw_values, np.nan)
            result[name] = raw_values
            n_ga += 1
        except Exception:
            continue

    if n_ga == 0:
        return pd.DataFrame()
    print(f"✅ 已加载 {n_ga} 个遗传算法因子（来自 feature.txt），已置于特征列最前")
    return result


def prepare_xgboost_data_fast(df, rebalance_period=5, min_history_days=60, n_jobs=-1,
                              cross_sectional_standardize=True, use_rank_target=True, use_improved_features=False,
                              available_tokens_by_date=None, use_ga_features=False):
    """
    向量化 + 并行 的数据准备。返回与原接口兼容的五元组（增加原始目标变量）。
    使用基于feature.txt的特征定义或改进的特征。

    Parameters:
    - cross_sectional_standardize: 是否对特征进行横截面标准化（按日期分组）
    - use_rank_target: 是否将目标变量转换为横截面排名（预测排名而非绝对值）
    - use_improved_features: 是否使用改进的特征工程（减少过拟合风险）
    - available_tokens_by_date: Top50币种列表（dict，date -> list of symbols），用于过滤样本
                               如果为None，则不进行过滤
    - use_ga_features: 是否加入遗传算法挖出的因子（来自 Genetic_Algorithm/feature.txt）

    🔥 关键修复：先计算特征（在全量连续数据上），再过滤（只保留Top50期间的样本）
    这样可以避免K线数据不连续导致的特征计算错误

    Returns:
    - X: 特征矩阵
    - y: 目标变量（如果use_rank_target=True，则为排名；否则为原始值）
    - y_original: 原始目标变量（用于IC计算）
    - syms: 符号列表
    - dates: 日期列表
    """
    from joblib import Parallel, delayed

    symbols = df['symbol'].unique().tolist()

    # 预先构建快速查询集 (date -> set of symbols) 用于高效过滤
    if available_tokens_by_date is not None:
        available_tokens_sets = {}
        for d, syms in available_tokens_by_date.items():
            if isinstance(d, pd.Timestamp):
                available_tokens_sets[d] = set(syms)
            else:
                available_tokens_sets[pd.to_datetime(d)] = set(syms)
    else:
        available_tokens_sets = None

    def run_one(sym):
        gp = df[df['symbol'] == sym].copy()
        if gp.empty: return pd.DataFrame()

        # 🔥 关键修复：1. 先计算特征（此时gp是完整的连续时间序列）
        feat_df = create_features_symbol_fast_based_on_feature_txt(gp, rebalance_period, min_history_days, use_improved_features)

        if feat_df.empty: return pd.DataFrame()

        # 🔥 关键修复：2. 计算完后再过滤（只保留Top50期间的样本）
        if available_tokens_sets is not None:
            # 向量化过滤：只保留该symbol在Top50名单里的日期的行
            # 构建日期到可用symbol集合的映射（已排序）
            sorted_dates = sorted(available_tokens_sets.keys())

            # 将feat_df的日期转换为Timestamp
            feat_df['_date_ts'] = pd.to_datetime(feat_df['date'])

            # 使用merge_asof或直接查找最接近的日期
            valid_mask = []
            for dt in feat_df['_date_ts']:
                # 找到最接近的可用日期（小于等于当前日期）
                available_date = None
                for d in sorted_dates:
                    if d <= dt:
                        available_date = d
                    else:
                        break
                if available_date is not None:
                    valid_mask.append(sym in available_tokens_sets[available_date])
                else:
                    valid_mask.append(False)

            feat_df = feat_df[valid_mask].drop(columns=['_date_ts'])

        return feat_df

    dfs = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(run_one)(s) for s in symbols)
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return np.array([]), np.array([]), np.array([]), [], []

    out = pd.concat(dfs, ignore_index=True)

    # 合并遗传算法因子（来自 feature.txt）
    if use_ga_features:
        ga_df = _compute_ga_factors_df(df)
        if ga_df.empty:
            print("⚠️ use_ga_features=True 但未加载到遗传算法因子，请检查 feature.txt 和 ga_top_expr_log.csv")
        else:
            ga_cols = [c for c in ga_df.columns if c not in ["date", "symbol"]]
            if ga_cols:
                out["date_norm"] = pd.to_datetime(out["date"]).dt.normalize()
                ga_df["date_norm"] = pd.to_datetime(ga_df["date"]).dt.normalize()
                out = out.merge(
                    ga_df[["date_norm", "symbol"] + ga_cols],
                    on=["date_norm", "symbol"],
                    how="left",
                    suffixes=("", "_ga"),
                )
                out = out.drop(columns=["date_norm"])
                # 填充 GA 因子的 NaN（按 symbol 前向填充，其余填 0）
                out[ga_cols] = out.groupby("symbol", sort=False)[ga_cols].ffill().fillna(0)
                # 将 GA 因子列放在最前面，便于特征选择时优先筛选
                base_cols = [c for c in out.columns if c not in ["date", "symbol", "future_ret"] and c not in ga_cols]
                out = out[["date", "symbol"] + ga_cols + base_cols + ["future_ret"]]

    # 保存原始目标变量（用于IC计算）
    y_original = out['future_ret'].copy().to_numpy(dtype=np.float32)

    # 横截面标准化：按日期分组，对每个日期的特征进行标准化
    if cross_sectional_standardize:
        print("🔄 进行横截面标准化（按日期分组）...")
        feature_cols = [col for col in out.columns if col not in ['date', 'symbol', 'future_ret']]

        def standardize_by_date(group):
            """对每个日期的特征进行横截面标准化"""
            group = group.copy()
            for col in feature_cols:
                mean_val = group[col].mean()
                std_val = group[col].std()
                if std_val > 1e-8:  # 避免除零
                    group[col] = (group[col] - mean_val) / std_val
                else:
                    group[col] = 0.0  # 如果标准差为0，设为0
            return group

        out = out.groupby('date', group_keys=False).apply(standardize_by_date)
        print(f"   已完成横截面标准化，处理了 {out['date'].nunique()} 个交易日")

    # 目标变量排名转换：按日期分组，将future_ret转换为横截面排名
    if use_rank_target:
        print("🔄 将目标变量转换为横截面排名（按日期分组）...")
        def rank_target_by_date(group):
            """对每个日期的目标变量进行排名转换"""
            group = group.copy()
            # 使用百分比排名（0-1之间），然后转换为标准正态分布的分位数
            # 这样可以保持相对顺序，同时使分布更接近正态分布
            rank_pct = group['future_ret'].rank(pct=True, method='average')
            # 将百分比排名转换为标准正态分布的分位数（使用scipy.stats.norm.ppf）
            from scipy import stats
            # 避免边界值（0和1）导致无穷大，使用clip
            rank_pct_clipped = rank_pct.clip(0.001, 0.999)
            group['future_ret'] = stats.norm.ppf(rank_pct_clipped)
            return group

        out = out.groupby('date', group_keys=False).apply(rank_target_by_date)
        print(f"   已完成目标变量排名转换，处理了 {out['date'].nunique()} 个交易日")

    X = out.drop(columns=['date','symbol','future_ret']).to_numpy(dtype=np.float32)
    y = out['future_ret'].to_numpy(dtype=np.float32)
    syms = out['symbol'].tolist()
    dates = pd.to_datetime(out['date']).tolist()

    target_type = "排名" if use_rank_target else "绝对值"
    print(f"✅ 特征 向量化+并行 数据样本: {len(out)}，特征维度: {X.shape[1]}，目标变量类型: {target_type}")
    return X, y, y_original, syms, dates

def adaptive_feature_selection(X_train, y_train, X_val, y_val, X_test,
                             target_features=50, importance_threshold=0.001,
                             correlation_threshold=0.95, use_orthogonalization=True, use_pca=False):
    """
    自适应特征选择：结合多种方法，动态调整选择数量
    新增：特征正交化处理，去除高度相关的特征

    Parameters:
    - correlation_threshold: 相关性阈值，超过此值的特征对将被去除其中一个
    - use_orthogonalization: 是否使用正交化处理（相关性过滤）
    - use_pca: 是否使用PCA进行正交化（会进一步降低特征维度）
    """
    print("🔄 自适应特征选择...")

    # 0. 特征正交化处理（去除高度相关的特征）
    orthogonalizer = None
    if use_orthogonalization:
        orthogonalizer = FeatureOrthogonalizer(
            correlation_threshold=correlation_threshold,
            use_pca=use_pca,
            pca_components=None,  # 自动选择主成分数量
            pca_variance_ratio=0.95
        )
        X_train_ortho = orthogonalizer.fit_transform(X_train, y_train)
        X_val_ortho = orthogonalizer.transform(X_val)
        X_test_ortho = orthogonalizer.transform(X_test)

        print(f"正交化后特征维度: {X_train.shape[1]} -> {X_train_ortho.shape[1]}")

        # 使用正交化后的特征继续后续处理
        X_train_for_selection = X_train_ortho
        X_val_for_selection = X_val_ortho
        X_test_for_selection = X_test_ortho
    else:
        X_train_for_selection = X_train
        X_val_for_selection = X_val
        X_test_for_selection = X_test

    # 1. 初步筛选（互信息）
    n_preselect = min(120, X_train_for_selection.shape[1])
    pre_selector = SelectKBest(score_func=mutual_info_regression, k=n_preselect)
    X_train_pre = pre_selector.fit_transform(X_train_for_selection, y_train)

    # 2. 训练临时模型评估特征重要性
    temp_model = xgb.XGBRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        random_state=42, n_jobs=-1
    )
    temp_model.fit(X_train_pre, y_train)

    # 3. 获取训练集特征重要性
    train_importance = temp_model.feature_importances_

    # 4. 计算验证集特征置换重要性（针对预选特征）
    # 注意：需要转换验证集到与训练集相同的预选特征空间
    X_val_pre = pre_selector.transform(X_val_for_selection)
    val_importance = permutation_feature_importance(
        temp_model, X_val_pre, y_val, n_repeats=3, random_state=42
    )

    # 5. 结合训练集和验证集的重要性（加权平均）
    train_weight = 0.7
    val_weight = 0.3

    # 归一化训练集重要性
    if np.max(train_importance) > 0:
        train_importance_norm = train_importance / np.max(train_importance)
    else:
        train_importance_norm = train_importance

    # 组合重要性评分
    combined_importance = (train_weight * train_importance_norm +
                          val_weight * val_importance)

    # 6. 根据组合重要性动态选择特征
    sorted_indices = np.argsort(combined_importance)[::-1]
    cumulative_importance = np.cumsum(combined_importance[sorted_indices])

    # 确保不超过实际可用特征数
    max_available = len(sorted_indices)

    # 选择重要性累积到80%的特征，或者至少target_features个特征
    target_cumulative = 0.8
    n_selected = np.where(cumulative_importance >= target_cumulative)[0]
    n_selected = n_selected[0] + 1 if len(n_selected) > 0 else max_available
    # 确保不超过可用特征数，且至少选择target_features个（如果可用的话）
    n_selected = min(max(target_features, n_selected), max_available)

    # 映射回原始特征索引
    selected_pre_indices = sorted_indices[:n_selected]

    print(f"📊 自适应特征选择重要性统计:")
    print(f"   训练集重要性均值: {train_importance_norm.mean():.4f}")
    print(f"   验证集重要性均值: {val_importance.mean():.4f}")
    print(f"   组合重要性均值: {combined_importance.mean():.4f}")
    print(f"   选择特征数: {n_selected}")

    # 创建组合选择器
    if orthogonalizer is not None:
        # 使用模块级别的CombinedSelector类（支持pickle序列化）
        selector = CombinedSelector(orthogonalizer, pre_selector, selected_pre_indices)
    else:
        selector = AdaptiveSelector(pre_selector, selected_pre_indices)

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    print(f"自适应特征选择: {X_train.shape[1]} -> {X_train_selected.shape[1]}")
    if n_selected > 0:
        print(f"累积组合重要性覆盖: {cumulative_importance[n_selected-1]:.3f}")
        print(f"训练集R²: {temp_model.score(X_train_pre, y_train):.4f}")
        print(f"验证集R²: {temp_model.score(X_val_pre, y_val):.4f}")
    else:
        print(f"累积组合重要性覆盖: 0.000")

    return selector, X_train_selected, X_val_selected, X_test_selected

def permutation_feature_importance(model, X_val, y_val, n_repeats=3, random_state=42):
    """
    计算特征置换重要性（Permutation Feature Importance）

    原理：随机打乱特征值，观察模型性能变化
    如果某个特征重要，打乱后性能下降明显

    Parameters:
    - model: 已训练的模型
    - X_val: 验证集特征
    - y_val: 验证集目标
    - n_repeats: 每个特征重复打乱次数
    - random_state: 随机种子

    Returns:
    - importance_scores: 特征重要性评分数组
    """
    np.random.seed(random_state)
    baseline_score = model.score(X_val, y_val)  # 使用R²作为基准
    n_features = X_val.shape[1]
    importance_scores = np.zeros(n_features)

    print("🔄 计算特征置换重要性...")
    for feature_idx in range(n_features):
        feature_scores = []

        for _ in range(n_repeats):
            # 复制验证集并打乱当前特征
            X_permuted = X_val.copy()
            np.random.shuffle(X_permuted[:, feature_idx])

            # 计算打乱后的性能
            permuted_score = model.score(X_permuted, y_val)
            # 重要性 = 基准性能 - 打乱后性能（下降越多说明越重要）
            importance_drop = baseline_score - permuted_score
            feature_scores.append(importance_drop)

        # 取平均重要性
        importance_scores[feature_idx] = np.mean(feature_scores)

    # 归一化到0-1范围
    if np.max(importance_scores) > 0:
        importance_scores = importance_scores / np.max(importance_scores)

    return importance_scores

def conservative_feature_selection(X_train, y_train, X_val, y_val, X_test,
                                 max_features=40, correlation_threshold=0.9):
    """
    保守的特征选择策略（结合训练集和验证集的重要性）
    更严格的特征筛选，减少过拟合风险

    Parameters:
    - max_features: 最大特征数量（默认40）
    - correlation_threshold: 相关性阈值（默认0.9，更严格）
    """
    print("🔄 保守特征选择（训练集+验证集联合评估）...")

    # 1. 先去除高相关性特征
    orthogonalizer = FeatureOrthogonalizer(
        correlation_threshold=correlation_threshold,
        use_pca=False  # 不使用PCA，保持可解释性
    )
    X_train_ortho = orthogonalizer.fit_transform(X_train, y_train)
    X_val_ortho = orthogonalizer.transform(X_val)
    X_test_ortho = orthogonalizer.transform(X_test)

    print(f"正交化后特征维度: {X_train.shape[1]} -> {X_train_ortho.shape[1]}")

    # 2. 基于XGBoost重要性进行选择
    temp_model = xgb.XGBRegressor(
        n_estimators=20,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_alpha=8.0,
        reg_lambda=20.0,
        n_jobs=-1
    )
    temp_model.fit(X_train_ortho, y_train)

    # 3. 获取训练集特征重要性（Gini重要性）
    train_importance = temp_model.feature_importances_

    # 4. 计算验证集特征置换重要性
    val_importance = permutation_feature_importance(
        temp_model, X_val_ortho, y_val, n_repeats=3, random_state=42
    )

    # 5. 结合训练集和验证集的重要性（加权平均）
    # 训练集权重70%，验证集权重30%（更重视验证集的泛化能力）
    train_weight = 0.7
    val_weight = 0.3

    # 归一化训练集重要性
    if np.max(train_importance) > 0:
        train_importance_norm = train_importance / np.max(train_importance)
    else:
        train_importance_norm = train_importance

    # 组合重要性评分
    combined_importance = (train_weight * train_importance_norm +
                          val_weight * val_importance)

    print(f"📊 重要性统计:")
    print(f"   训练集重要性均值: {train_importance_norm.mean():.4f}")
    print(f"   验证集重要性均值: {val_importance.mean():.4f}")
    print(f"   组合重要性均值: {combined_importance.mean():.4f}")

    # 6. 更严格的重要性筛选（基于组合重要性）
    positive_importance = combined_importance[combined_importance > 0]
    if len(positive_importance) > 0:
        importance_threshold = np.percentile(positive_importance, 30)  # 30分位数
    else:
        importance_threshold = 0

    selected_indices = np.where(combined_importance >= importance_threshold)[0]

    # 7. 限制最大特征数量
    if len(selected_indices) > max_features:
        # 按组合重要性排序选择前max_features个
        top_indices = np.argsort(combined_importance[selected_indices])[-max_features:]
        selected_indices = selected_indices[top_indices]

    # 创建选择器
    selector = ConservativeSelector(orthogonalizer, selected_indices)

    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    print(f"保守特征选择: {X_train.shape[1]} -> {X_train_selected.shape[1]}")
    if len(selected_indices) > 0:
        print(f"   重要性阈值: {importance_threshold:.6f}, 选择特征数: {len(selected_indices)}")
        print(f"   训练集R²: {temp_model.score(X_train_ortho, y_train):.4f}")
        print(f"   验证集R²: {temp_model.score(X_val_ortho, y_val):.4f}")

    return selector, X_train_selected, X_val_selected, X_test_selected

def robust_feature_selection(X_train, y_train, X_val, y_val, X_test,
                           use_rolling=True, use_cv=True,
                           rolling_params=None, cv_params=None,
                           **base_selection_params):
    """
    集成多种特征选择方法，提供更稳健的特征选择

    Parameters:
    - use_rolling: 是否使用滚动窗口
    - use_cv: 是否使用交叉验证
    - rolling_params: 滚动窗口参数
    - cv_params: 交叉验证参数
    - **base_selection_params: 传递给基础特征选择函数的参数
    """
    print("🛡️ 集成稳健特征选择...")

    if rolling_params is None:
        rolling_params = {'window_size': 1000, 'step_size': 500, 'min_stability': 0.7}
    if cv_params is None:
        cv_params = {'n_splits': 5, 'min_cv_stability': 0.8}

    selected_sets = []
    method_names = []

    # 滚动窗口特征选择
    if use_rolling:
        print("📊 执行滚动窗口特征选择...")
        rolling_features, rolling_stability = rolling_window_feature_selection(
            X_train, y_train, X_val, y_val, X_test, **rolling_params, **base_selection_params
        )
        if len(rolling_features) > 0:
            selected_sets.append(set(rolling_features))
            method_names.append("滚动窗口")
            print(f"   滚动窗口选择: {len(rolling_features)} 个特征")

    # 交叉验证特征选择
    if use_cv:
        print("📊 执行交叉验证特征选择...")
        cv_features, cv_stability, cv_importances = cv_based_feature_selection(
            X_train, y_train, X_val, y_val, X_test, **cv_params, **base_selection_params
        )
        if len(cv_features) > 0:
            selected_sets.append(set(cv_features))
            method_names.append("交叉验证")
            print(f"   CV选择: {len(cv_features)} 个特征")

    # 确定最终特征
    if selected_sets:
        # 取交集：只保留在所有方法中都被选择的特征
        final_features = list(set.intersection(*selected_sets))
        selection_method = "交集"
        print(f"✅ {selection_method}策略 - 最终稳健特征: {len(final_features)} 个特征")

        if len(final_features) == 0:
            print("⚠️  交集为空，使用并集策略...")
            final_features = list(set.union(*selected_sets))
            selection_method = "并集"
            print(f"✅ {selection_method}策略 - 最终稳健特征: {len(final_features)} 个特征")
    else:
        # 如果没有启用任何稳健方法，使用基础方法
        print("📊 使用基础保守特征选择...")
        selector, _, _, _ = conservative_feature_selection(
            X_train, y_train, X_val, y_val, X_test, **base_selection_params
        )
        final_features = _get_selected_features_from_selector(selector)
        selection_method = "基础方法"
        print(f"✅ {selection_method} - 选择特征: {len(final_features)} 个特征")

    # 🔥 对最终合并的特征再次应用相关性筛选
    if len(final_features) > 0 and 'correlation_threshold' in base_selection_params:
        correlation_threshold = base_selection_params['correlation_threshold']
        print(f"🔄 对最终特征应用相关性筛选（阈值={correlation_threshold}）...")

        # 提取最终选择的特征
        final_features_array = np.array(final_features)
        if final_features_array.max() < X_train.shape[1]:
            X_train_final = X_train[:, final_features_array]

            # 计算相关性矩阵
            X_train_final_df = pd.DataFrame(X_train_final)
            corr_matrix = X_train_final_df.corr().values
            np.fill_diagonal(corr_matrix, 0)  # 对角线设为0

            # 如果提供了y，计算每个特征与y的相关性（用于决定保留哪个特征）
            if y_train is not None:
                y_corrs = np.abs([np.corrcoef(X_train_final[:, i], y_train)[0, 1]
                                 for i in range(X_train_final.shape[1])])
                y_corrs = np.nan_to_num(y_corrs, nan=0)
            else:
                # 如果没有y，使用特征的方差作为重要性指标
                y_corrs = np.var(X_train_final, axis=0)
                y_corrs = np.nan_to_num(y_corrs, nan=0)

            # 找出高度相关的特征对，并选择要移除的特征
            to_remove = set()
            for i in range(len(final_features)):
                if i in to_remove:
                    continue
                for j in range(i+1, len(final_features)):
                    if j in to_remove:
                        continue
                    if abs(corr_matrix[i, j]) > correlation_threshold:
                        # 优先保留与y相关性更高（或方差更大）的特征
                        if y_corrs[i] >= y_corrs[j]:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)

            # 保留的特征索引（映射回原始特征索引）
            kept_indices = [i for i in range(len(final_features)) if i not in to_remove]
            final_features = [final_features[i] for i in kept_indices]

            removed_count = len(to_remove)
            print(f"   去除 {removed_count} 个高度相关特征，保留 {len(final_features)} 个特征")
        else:
            print(f"   ⚠️ 特征索引超出范围，跳过相关性筛选")

    # 创建最终的选择器
    final_selector = ConservativeSelector(None, np.array(final_features))

    # 应用到所有数据集
    X_train_selected = final_selector.transform(X_train)
    X_val_selected = final_selector.transform(X_val)
    X_test_selected = final_selector.transform(X_test)

    print(f"🛡️ 集成特征选择完成: {selection_method}")
    print(f"   方法组合: {', '.join(method_names) if method_names else '基础方法'}")
    print(f"   最终特征数: {len(final_features)}")

    return final_selector, X_train_selected, X_val_selected, X_test_selected

def rolling_window_feature_selection(X_train, y_train, X_val, y_val, X_test,
                                   window_size=1000, step_size=500,
                                   min_stability=0.7, **feature_selection_params):
    """
    使用滚动窗口进行特征选择

    Parameters:
    - window_size: 每个窗口的大小
    - step_size: 窗口移动步长
    - min_stability: 特征在不同窗口中出现的比例阈值
    - **feature_selection_params: 传递给基础特征选择函数的参数
    """
    print("🔄 滚动窗口特征选择...")
    n_samples = len(X_train)
    n_features = X_train.shape[1]
    feature_counts = np.zeros(n_features)  # 记录每个特征被选择的次数

    # 滑动窗口
    window_starts = list(range(0, n_samples - window_size + 1, step_size))
    if not window_starts:
        # 如果训练数据不够一个窗口，使用整个训练集
        window_starts = [0]
        window_size = n_samples

    print(f"   窗口大小: {window_size}, 步长: {step_size}, 窗口数量: {len(window_starts)}")

    for start_idx in window_starts:
        end_idx = min(start_idx + window_size, n_samples)

        # 当前窗口的数据
        X_window = X_train[start_idx:end_idx]
        y_window = y_train[start_idx:end_idx]

        # 在当前窗口进行特征选择
        try:
            selector, _, _, _ = conservative_feature_selection(
                X_window, y_window, X_val, y_val, X_test, **feature_selection_params
            )

            # 记录被选择的特征
            selected_features = _get_selected_features_from_selector(selector)
            feature_counts[selected_features] += 1

        except Exception as e:
            print(f"   ⚠️ 窗口 {start_idx}-{end_idx} 特征选择失败: {e}")
            continue

    # 计算稳定性（特征被选择的频率）
    total_windows = len(window_starts)
    stability_scores = feature_counts / total_windows

    # 选择稳定特征
    stable_features = np.where(stability_scores >= min_stability)[0]

    print(f"✅ 滚动窗口特征选择完成:")
    print(f"   总窗口数: {total_windows}")
    print(f"   稳定性阈值: {min_stability}")
    print(f"   选择稳定特征: {len(stable_features)} 个")

    if len(stable_features) > 0:
        avg_stability = np.mean(stability_scores[stable_features])
        print(f"   平均稳定性: {avg_stability:.3f}")

    return stable_features, stability_scores

def cv_based_feature_selection(X_train, y_train, X_val, y_val, X_test,
                              n_splits=5, min_cv_stability=0.8, **feature_selection_params):
    """
    使用交叉验证进行特征选择

    Parameters:
    - n_splits: 交叉验证折数
    - min_cv_stability: 特征在CV中出现的比例阈值
    - **feature_selection_params: 传递给基础特征选择函数的参数
    """
    print("🔄 交叉验证特征选择...")
    from sklearn.model_selection import TimeSeriesSplit

    # 使用时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=n_splits)
    n_features = X_train.shape[1]
    feature_selection_counts = np.zeros(n_features)
    feature_importance_sums = np.zeros(n_features)

    print(f"   CV折数: {n_splits}, 时间序列分割")

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        try:
            # 当前折的训练数据
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]

            # 在当前折进行特征选择
            selector, X_fold_selected, _, _ = conservative_feature_selection(
                X_fold_train, y_fold_train, X_val, y_val, X_test, **feature_selection_params
            )

            # 收集特征重要性信息（如果可用）
            if hasattr(selector, 'temp_model') and hasattr(selector.temp_model, 'feature_importances_'):
                fold_importances = selector.temp_model.feature_importances_
                # 确保维度匹配
                if len(fold_importances) <= len(feature_importance_sums):
                    feature_importance_sums[:len(fold_importances)] += fold_importances

            # 记录被选择的特征
            selected_features = _get_selected_features_from_selector(selector)
            feature_selection_counts[selected_features] += 1

            fold_results.append({
                'fold': fold_idx,
                'n_selected': len(selected_features),
                'selected_features': selected_features.copy()
            })

        except Exception as e:
            print(f"   ⚠️ 折 {fold_idx} 特征选择失败: {e}")
            continue

    # 计算CV稳定性
    cv_stability = feature_selection_counts / n_splits

    # 选择稳定特征
    stable_features = np.where(cv_stability >= min_cv_stability)[0]

    # 计算平均重要性（对于被选择的特征）
    avg_importances = feature_importance_sums / np.maximum(feature_selection_counts, 1)

    print(f"✅ CV特征选择完成:")
    print(f"   CV折数: {n_splits}")
    print(f"   稳定性阈值: {min_cv_stability}")
    print(f"   选择稳定特征: {len(stable_features)} 个")

    if len(stable_features) > 0:
        avg_stability = np.mean(cv_stability[stable_features])
        print(f"   平均CV稳定性: {avg_stability:.3f}")

        # 显示各折的结果统计
        fold_sizes = [result['n_selected'] for result in fold_results if 'n_selected' in result]
        if fold_sizes:
            print(f"   各折选择特征数: {fold_sizes}")
            print(f"   平均每折特征数: {np.mean(fold_sizes):.1f}")

    return stable_features, cv_stability, avg_importances

def analyze_selected_features_correlation(X_selected, n_features, title="特征相关性分析",
                                         save_path=None, show_plot=True, feature_names=None):
    """
    分析并可视化选定特征的相关性

    Parameters:
    - X_selected: 选定的特征矩阵
    - n_features: 特征数量
    - title: 图表标题
    - save_path: 保存路径（可选）
    - show_plot: 是否显示图形
    - feature_names: 特征名称列表（可选）
    """
    print(f"\n🔍 {title}")
    print("=" * 60)

    # 创建特征名称
    if feature_names is None or len(feature_names) != n_features:
        # 如果没有提供特征名称或数量不匹配，使用默认编号
        if n_features <= 20:
            feature_names = [f'feature_{i+1}' for i in range(n_features)]
        else:
            # 对于大量特征，使用编号
            feature_names = [f'f{i+1}' for i in range(n_features)]

    # 计算相关性矩阵
    corr_matrix = np.corrcoef(X_selected.T)

    # 相关性统计信息
    if n_features > 1:
        corr_flat = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        corr_abs = np.abs(corr_flat)

        print("📊 相关性统计:")
        print(f"   特征数量: {n_features}")
        print(f"   平均相关性: {corr_abs.mean():.6f}")
        print(f"   中位数相关性: {np.median(corr_abs):.6f}")
        print(f"   最大相关性: {corr_abs.max():.6f}")
        print(f"   最小相关性: {corr_abs.min():.6f}")
        print(f"   相关性标准差: {corr_abs.std():.6f}")

        # 找出高度相关的特征对
        high_corr_pairs = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                if abs(corr_matrix[i, j]) >= 0.5:  # 相关性阈值
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
    else:
        print("📊 相关性统计:")
        print(f"   特征数量: {n_features}")
        print("   单个特征，无相关性统计")
        high_corr_pairs = []

    if high_corr_pairs:
        print(f"\n⚠️  高度相关特征对 (|corr| >= 0.5): {len(high_corr_pairs)} 对")
        for i, j, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:  # 只显示前10个
            print(f"      f{i+1} ↔ f{j+1}: {corr:.4f}")
    else:
        print("\n✅ 无高度相关特征对 (|corr| >= 0.5)")

    # 可视化相关性矩阵
    if VISUALIZATION_AVAILABLE and show_plot:
        try:
            if n_features == 1:
                # 单个特征的情况
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f"单个特征: {feature_names[0]}\n相关性矩阵为1×1单位矩阵",
                        ha='center', va='center', fontsize=14,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                plt.title(title, fontsize=14, pad=20)
                plt.axis('off')
            elif n_features <= 30:  # 小型特征集：完整热力图
                plt.figure(figsize=(max(10, min(20, n_features * 0.5)), max(8, min(18, n_features * 0.45))))

                # 创建遮罩，只显示上三角
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

                # 绘制热力图
                sns.heatmap(corr_matrix,
                           mask=mask,
                           annot=n_features <= 12,  # 只有很少特征时显示数值
                           cmap='coolwarm',
                           vmin=-1, vmax=1,
                           center=0,
                           square=True,
                           xticklabels=feature_names,
                           yticklabels=feature_names,
                           cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                           linewidths=0.5 if n_features <= 15 else 0)

                plt.title(title, fontsize=14, pad=20)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()

            elif n_features <= 100:  # 中型特征集：简化热力图
                plt.figure(figsize=(12, 10))

                # 只显示相关性绝对值大于阈值的部分
                high_corr_mask = np.abs(corr_matrix) >= 0.5
                corr_display = corr_matrix * high_corr_mask

                sns.heatmap(corr_display,
                           cmap='coolwarm',
                           vmin=-1, vmax=1,
                           center=0,
                           square=False,
                           xticklabels=False,
                           yticklabels=False,
                           cbar_kws={'shrink': 0.8, 'label': 'Correlation (|corr| >= 0.5)'})

                plt.title(f"{title} (|corr| >= 0.5)", fontsize=14, pad=20)
                plt.tight_layout()

                print("ℹ️  特征数量较多，只显示高相关性部分 (|corr| >= 0.5)")

            else:  # 大型特征集：相关性分布图
                plt.figure(figsize=(12, 8))

                # 相关性绝对值的分布
                plt.hist(corr_abs, bins=50, alpha=0.7, edgecolor='black', density=True)
                plt.xlabel('Absolute Correlation')
                plt.ylabel('Density')
                plt.title(f"{title} - Correlation Distribution", fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.axvline(0.7, color='red', linestyle='--', label='High correlation threshold (0.7)')
                plt.axvline(corr_abs.mean(), color='blue', linestyle='--',
                           label=f'Mean: {corr_abs.mean():.3f}')
                plt.legend()

                print("ℹ️  特征数量过多，显示相关性分布而非热力图")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 相关性分析图表已保存: {save_path}")

            # plt.show()

        except Exception as e:
            print(f"⚠️  绘图失败: {e}")
            print("   将显示文本形式的相关性摘要")

    # 对于大型特征集，提供文本摘要
    if n_features > 30 and not (VISUALIZATION_AVAILABLE and show_plot):
        print("ℹ️  特征数量较多，提供相关性分布摘要:")

        # 相关性分布统计
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        hist, _ = np.histogram(corr_abs, bins=bins)
        print("   相关性分布:")
        for i in range(len(bins)-1):
            pct = hist[i] / len(corr_abs) * 100
            print(".1f")
        # 显示最相关的特征对
        if len(high_corr_pairs) > 0:
            print(f"\n🔗 最相关的10个特征对:")
            for i, j, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
                print(".4f")

    print("=" * 60)

def analyze_feature_importance(X_train, y_train, n_features, title="特征重要性分析",
                              save_path=None, show_plot=True, top_n=20, feature_names=None):
    """
    分析并可视化特征重要性

    Parameters:
    - X_train: 训练特征
    - y_train: 训练目标
    - n_features: 特征数量
    - title: 图表标题
    - save_path: 保存路径（可选）
    - show_plot: 是否显示图形
    - top_n: 显示前N个最重要的特征
    - feature_names: 特征名称列表（可选）
    """
    print(f"\n📈 {title}")
    print("=" * 60)

    # 创建特征名称
    if feature_names is None or len(feature_names) != n_features:
        # 如果没有提供特征名称或数量不匹配，使用默认编号
        if n_features <= top_n:
            feature_names = [f'feature_{i+1}' for i in range(n_features)]
        else:
            feature_names = [f'f{i+1}' for i in range(n_features)]

    try:
        # 训练临时模型获取特征重要性
        temp_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )

        temp_model.fit(X_train, y_train)
        importances = temp_model.feature_importances_

        # 创建重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # 重要性统计
        print("📊 特征重要性统计:")
        print(f"   总特征数: {n_features}")
        print(f"   平均重要性: {importances.mean():.6f}")
        print(f"   最大重要性: {importances.max():.6f}")
        print(f"   最小重要性: {importances.min():.6f}")
        print(f"   重要性标准差: {importances.std():.6f}")
        # 显示前N个重要特征
        display_n = min(top_n, n_features)
        print(f"\n🏆 前{display_n}个最重要的特征:")
        for i, row in importance_df.head(display_n).iterrows():
            rank = i + 1
            print(f"  #{rank:2d} {row['feature']}: {row['importance']:.6f}")
        # 重要性分布分析
        importance_df['importance_pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100
        cumulative_pct = importance_df['importance_pct'].cumsum()

        # 找到解释80%重要性的特征数量
        n_80pct = (cumulative_pct >= 80).idxmax() + 1
        print(f"   80%重要性所需特征数: {n_80pct} (占总数的{n_80pct/n_features*100:.1f}%)")
        # 可视化
        if VISUALIZATION_AVAILABLE and show_plot:
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # 特征重要性条形图
                top_features = importance_df.head(display_n)
                ax1.barh(range(len(top_features)), top_features['importance'][::-1])
                ax1.set_yticks(range(len(top_features)))
                ax1.set_yticklabels(top_features['feature'][::-1])
                ax1.set_xlabel('Importance')
                ax1.set_title(f'Top {display_n} Feature Importances')
                ax1.grid(True, alpha=0.3)

                # 重要性分布直方图
                ax2.hist(importances, bins=20, alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Importance')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Feature Importance Distribution')
                ax2.grid(True, alpha=0.3)
                ax2.axvline(importances.mean(), color='red', linestyle='--',
                           label=f'Mean: {importances.mean():.4f}')
                ax2.legend()

                plt.suptitle(title, fontsize=14)
                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"💾 特征重要性图表已保存: {save_path}")

                # plt.show()

            except Exception as e:
                print(f"⚠️  绘图失败: {e}")

        return importance_df

    except Exception as e:
        print(f"❌ 特征重要性分析失败: {e}")
        return None

def _artifact_paths(window: int, rebalance_period: int, model_type: str = 'xgb'):
    base_dir = Path(os.path.dirname(__file__)) / "models"
    base_dir.mkdir(parents=True, exist_ok=True)
    model_type = (model_type or 'xgb').lower()
    if model_type == 'lgbm':
        tag = f"lgbm_enhanced_w{window}_r{rebalance_period}"
        model_path = base_dir / f"{tag}.txt"
    else:
        # tag = f"xgb_enhanced_w{window}_r5"
        tag = f"xgb_enhanced_w{window}_r{rebalance_period}"
        # tag = f"xgb_w{window}_r{rebalance_period}"
        model_path = base_dir / f"{tag}.json"
    selector_path = base_dir / f"{tag}_selector.pkl"
    scaler_path = base_dir / f"{tag}_scaler.pkl"
    return model_path, selector_path, scaler_path

def create_strict_time_split(data_with_dates, test_days=180, validation_days=180):
    """
    严格的时间序列分割
    支持4元组 (X, y, symbol, date) 或 5元组 (X, y, y_original, symbol, date)
    """
    print("🔄 严格时间序列分割...")

    # 检测数据格式：4元组还是5元组
    is_5tuple = len(data_with_dates[0]) == 5

    if is_5tuple:
        # 5元组: (X, y, y_original, symbol, date)
        data_with_dates.sort(key=lambda x: x[4])
        dates = [item[4] for item in data_with_dates]
    else:
        # 4元组: (X, y, symbol, date)
        data_with_dates.sort(key=lambda x: x[3])
        dates = [item[3] for item in data_with_dates]

    min_date, max_date = min(dates), max(dates)

    # 测试集：最后test_days天
    test_cutoff = max_date - pd.Timedelta(days=test_days - 1)
    if is_5tuple:
        test_data = [it for it in data_with_dates if it[4] >= test_cutoff]
    else:
        test_data = [it for it in data_with_dates if it[3] >= test_cutoff]

    # 验证集：测试集之前的validation_days天
    val_cutoff = test_cutoff - pd.Timedelta(days=validation_days)
    if is_5tuple:
        val_data = [it for it in data_with_dates if val_cutoff <= it[4] < test_cutoff]
        train_data = [it for it in data_with_dates if it[4] < val_cutoff]
    else:
        val_data = [it for it in data_with_dates if val_cutoff <= it[3] < test_cutoff]
        train_data = [it for it in data_with_dates if it[3] < val_cutoff]

    print(f"数据时间范围: {min_date} ~ {max_date}")
    print(f"训练集: {min_date} ~ {val_cutoff.date()} ({len(train_data)} 样本)")
    print(f"验证集: {val_cutoff.date()} ~ {test_cutoff.date()} ({len(val_data)} 样本)")
    print(f"测试集: {test_cutoff.date()} ~ {max_date} ({len(test_data)} 样本)")

    return train_data, val_data, test_data

def create_rolling_time_split(data_with_dates, train_window_days=365, val_window_days=90,
                             test_window_days=90, roll_step_days=180, min_train_samples=1000):
    """
    滚动时间序列分割 - 支持滚动窗口训练

    Parameters:
    - train_window_days: 训练窗口长度（天数）
    - val_window_days: 验证窗口长度（天数）
    - test_window_days: 测试窗口长度（天数）
    - roll_step_days: 滚动步长（天数）
    - min_train_samples: 最少训练样本数

    Returns:
    - List of tuples: [(train_data, val_data, test_data, window_start_date, window_end_date), ...]
    """
    print("🔄 滚动时间序列分割...")

    # 检测数据格式：4元组还是5元组
    is_5tuple = len(data_with_dates[0]) == 5

    if is_5tuple:
        # 5元组: (X, y, y_original, symbol, date)
        data_with_dates.sort(key=lambda x: x[4])
        dates = [item[4] for item in data_with_dates]
    else:
        # 4元组: (X, y, symbol, date)
        data_with_dates.sort(key=lambda x: x[3])
        dates = [item[3] for item in data_with_dates]

    min_date, max_date = min(dates), max(dates)
    print(f"数据时间范围: {min_date} ~ {max_date}")

    # 计算滚动窗口
    rolling_windows = []

    # 第一个窗口的结束日期（测试集结束）
    current_end_date = max_date

    while True:
        # 计算当前窗口的各个部分
        test_start = current_end_date - pd.Timedelta(days=test_window_days - 1)
        val_start = test_start - pd.Timedelta(days=val_window_days)
        train_start = val_start - pd.Timedelta(days=train_window_days)

        # 检查是否有足够的训练数据
        if is_5tuple:
            train_data = [it for it in data_with_dates if train_start <= it[4] < val_start]
        else:
            train_data = [it for it in data_with_dates if train_start <= it[3] < val_start]

        if len(train_data) < min_train_samples:
            print(f"训练样本不足 ({len(train_data)} < {min_train_samples})，停止滚动")
            break

        # 提取验证集和测试集
        if is_5tuple:
            val_data = [it for it in data_with_dates if val_start <= it[4] < test_start]
            test_data = [it for it in data_with_dates if test_start <= it[4] <= current_end_date]
        else:
            val_data = [it for it in data_with_dates if val_start <= it[3] < test_start]
            test_data = [it for it in data_with_dates if test_start <= it[3] <= current_end_date]

        window_info = {
            'window_start': train_start,
            'train_end': val_start - pd.Timedelta(days=1),
            'val_end': test_start - pd.Timedelta(days=1),
            'test_end': current_end_date,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data)
        }

        rolling_windows.append((train_data, val_data, test_data, window_info))

        # 移动到下一个滚动窗口
        current_end_date = current_end_date - pd.Timedelta(days=roll_step_days)

        # 如果训练开始日期已经超出数据范围，停止
        if train_start < min_date:
            break

    print(f"生成 {len(rolling_windows)} 个滚动窗口")
    for i, (_, _, _, window_info) in enumerate(rolling_windows):
        print(f"窗口 {i+1}: 训练 {window_info['window_start'].date()} ~ {window_info['train_end'].date()} "
              f"({window_info['train_samples']} 样本) -> "
              f"验证 {window_info['train_end'].date()} ~ {window_info['val_end'].date()} "
              f"({window_info['val_samples']} 样本) -> "
              f"测试 {window_info['val_end'].date()} ~ {window_info['test_end'].date()} "
              f"({window_info['test_samples']} 样本)")

    return rolling_windows


def create_expanding_rolling_time_split(data_with_dates, initial_train_days=365, val_window_days=90,
                                       test_window_days=90, roll_step_days=90, min_train_samples=1000):
    """
    扩展训练集的滚动时间序列分割

    特点：
    - 训练集不断扩展（包含越来越多的历史数据）
    - 验证集和测试集是未来固定的时间段
    - 每次滚动时，训练集会包含更多的历史数据

    Parameters:
    - initial_train_days: 初始训练窗口长度（天数）
    - val_window_days: 验证窗口长度（天数）
    - test_window_days: 测试窗口长度（天数）
    - roll_step_days: 滚动步长（天数）
    - min_train_samples: 最少训练样本数

    Returns:
    - List of tuples: [(train_data, val_data, test_data, window_info), ...]
    """
    print("🔄 扩展训练集滚动时间序列分割...")

    # 检测数据格式：4元组还是5元组
    is_5tuple = len(data_with_dates[0]) == 5

    if is_5tuple:
        # 5元组: (X, y, y_original, symbol, date)
        data_with_dates.sort(key=lambda x: x[4])
        dates = [item[4] for item in data_with_dates]
    else:
        # 4元组: (X, y, symbol, date)
        data_with_dates.sort(key=lambda x: x[3])
        dates = [item[3] for item in data_with_dates]

    min_date, max_date = min(dates), max(dates)
    print(f"数据时间范围: {min_date} ~ {max_date}")

    # 计算滚动窗口
    rolling_windows = []

    # 从最早的可能开始日期开始
    current_train_end = min_date + pd.Timedelta(days=initial_train_days)

    window_idx = 0
    while True:
        # 计算当前窗口的各个部分
        val_start = current_train_end
        val_end = val_start + pd.Timedelta(days=val_window_days)
        test_start = val_end
        test_end = test_start + pd.Timedelta(days=test_window_days)

        # 检查是否还有足够的数据创建窗口
        # 如果验证集开始日期已经超出数据范围，停止
        if val_start >= max_date:
            break
        
        # 如果测试集结束日期超出数据范围，但测试集开始日期还在范围内，创建最后一个窗口（允许测试集不足90天）
        is_last_window = False
        if test_end > max_date:
            if test_start < max_date:
                # 这是最后一个窗口，测试集不足test_window_days天，但仍然创建
                is_last_window = True
                test_end = max_date  # 将测试集结束日期调整为数据最大日期
            else:
                # 测试集开始日期也超出范围，无法创建窗口
                break

        # 训练集：从数据开始到当前训练结束日期
        if is_5tuple:
            train_data = [it for it in data_with_dates if it[4] < val_start]
            val_data = [it for it in data_with_dates if val_start <= it[4] < val_end]
            test_data = [it for it in data_with_dates if test_start <= it[4] < test_end]
        else:
            train_data = [it for it in data_with_dates if it[3] < val_start]
            val_data = [it for it in data_with_dates if val_start <= it[3] < val_end]
            test_data = [it for it in data_with_dates if test_start <= it[3] < test_end]

        if len(train_data) < min_train_samples:
            print(f"训练样本不足 ({len(train_data)} < {min_train_samples})，停止滚动")
            break

        window_info = {
            'window_idx': window_idx,
            'train_start': min_date,
            'train_end': val_start - pd.Timedelta(days=1),
            'val_start': val_start,
            'val_end': val_end - pd.Timedelta(days=1),
            'test_start': test_start,
            'test_end': test_end - pd.Timedelta(days=1),
            'window_start': min_date,
            'window_end': test_end - pd.Timedelta(days=1),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'is_last_window': is_last_window  # 标记是否为最后一个窗口
        }

        rolling_windows.append((train_data, val_data, test_data, window_info))
        
        if is_last_window:
            actual_test_days = len(test_data)  # 使用实际测试集样本数
            expected_test_days = test_window_days
            print(f"⚠️ 最后一个窗口：测试集不足 {expected_test_days} 天，实际测试集样本数: {actual_test_days} 个")
            break

        # 移动到下一个滚动窗口（训练集结束日期前进）
        current_train_end = current_train_end + pd.Timedelta(days=roll_step_days)
        window_idx += 1

    print(f"生成 {len(rolling_windows)} 个扩展滚动窗口")
    for i, (_, _, _, window_info) in enumerate(rolling_windows):
        last_window_marker = " [最后一个窗口]" if window_info.get('is_last_window', False) else ""
        print(f"窗口 {i+1}{last_window_marker}: 训练 {window_info['train_start'].date()} ~ {window_info['train_end'].date()} "
              f"({window_info['train_samples']} 样本) -> "
              f"验证 {window_info['val_start'].date()} ~ {window_info['val_end'].date()} "
              f"({window_info['val_samples']} 样本) -> "
              f"测试 {window_info['test_start'].date()} ~ {window_info['test_end'].date()} "
              f"({window_info['test_samples']} 样本)")

    return rolling_windows

def create_quarterly_expanding_rolling_time_split(data_with_dates, initial_train_days=365, val_window_days=90,
                                                  test_window_days=90, min_train_samples=1000):
    """
    按季度第一天（1月1日、4月1日、7月1日、10月1日）扩展训练集的滚动时间序列分割
    
    特点：
    - 训练集不断扩展（包含越来越多的历史数据）
    - 验证集和测试集是未来固定的时间段
    - 每次滚动时，训练集会包含更多的历史数据
    - 只在每个季度的第一天（1/1, 4/1, 7/1, 10/1）重新训练
    
    Parameters:
    - initial_train_days: 初始训练窗口长度（天数）
    - val_window_days: 验证窗口长度（天数）
    - test_window_days: 测试窗口长度（天数）
    - min_train_samples: 最少训练样本数
    
    Returns:
    - List of tuples: [(train_data, val_data, test_data, window_info), ...]
    """
    print("🔄 按季度第一天扩展训练集滚动时间序列分割...")
    
    # 检测数据格式：4元组还是5元组
    is_5tuple = len(data_with_dates[0]) == 5
    
    if is_5tuple:
        # 5元组: (X, y, y_original, symbol, date)
        data_with_dates.sort(key=lambda x: x[4])
        dates = [item[4] for item in data_with_dates]
    else:
        # 4元组: (X, y, symbol, date)
        data_with_dates.sort(key=lambda x: x[3])
        dates = [item[3] for item in data_with_dates]
    
    min_date, max_date = min(dates), max(dates)
    print(f"数据时间范围: {min_date} ~ {max_date}")
    
    # 辅助函数：获取下一个季度第一天
    def get_next_quarter_start(date):
        """获取下一个季度第一天（1/1, 4/1, 7/1, 10/1）"""
        year = date.year
        month = date.month
        
        if month <= 3:
            # 当前在Q1，下一个季度是4月1日
            return pd.Timestamp(year=year, month=4, day=1)
        elif month <= 6:
            # 当前在Q2，下一个季度是7月1日
            return pd.Timestamp(year=year, month=7, day=1)
        elif month <= 9:
            # 当前在Q3，下一个季度是10月1日
            return pd.Timestamp(year=year, month=10, day=1)
        else:
            # 当前在Q4，下一个季度是下一年的1月1日
            return pd.Timestamp(year=year + 1, month=1, day=1)
    
    def get_current_quarter_start(date):
        """获取当前日期所在季度的第一天"""
        year = date.year
        month = date.month
        
        if month <= 3:
            return pd.Timestamp(year=year, month=1, day=1)
        elif month <= 6:
            return pd.Timestamp(year=year, month=4, day=1)
        elif month <= 9:
            return pd.Timestamp(year=year, month=7, day=1)
        else:
            return pd.Timestamp(year=year, month=10, day=1)
    
    # 计算滚动窗口
    rolling_windows = []
    
    # 从最早的可能开始日期开始
    # 找到第一个季度第一天作为初始训练结束日期
    first_possible_train_end = min_date + pd.Timedelta(days=initial_train_days)
    
    # 找到第一个大于等于 first_possible_train_end 的季度第一天
    # 先尝试当前季度第一天
    current_quarter_start = get_current_quarter_start(first_possible_train_end)
    
    if current_quarter_start >= first_possible_train_end:
        # 当前季度第一天满足要求，使用它
        current_train_end = current_quarter_start
    else:
        # 当前季度第一天太早，使用下一个季度第一天
        current_train_end = get_next_quarter_start(first_possible_train_end)
    
    window_idx = 0
    while True:
        # 计算当前窗口的各个部分
        val_start = current_train_end
        val_end = val_start + pd.Timedelta(days=val_window_days)
        test_start = val_end
        test_end = test_start + pd.Timedelta(days=test_window_days)
        
        # 检查是否还有足够的数据创建窗口
        # 如果验证集开始日期已经超出数据范围，停止
        if val_start >= max_date:
            break
        
        # 如果测试集结束日期超出数据范围，但测试集开始日期还在范围内，创建最后一个窗口（允许测试集不足90天）
        is_last_window = False
        if test_end > max_date:
            if test_start < max_date:
                # 这是最后一个窗口，测试集不足test_window_days天，但仍然创建
                is_last_window = True
                test_end = max_date  # 将测试集结束日期调整为数据最大日期
            else:
                # 测试集开始日期也超出范围，无法创建窗口
                break
        
        # 训练集：从数据开始到当前训练结束日期
        if is_5tuple:
            train_data = [it for it in data_with_dates if it[4] < val_start]
            val_data = [it for it in data_with_dates if val_start <= it[4] < val_end]
            test_data = [it for it in data_with_dates if test_start <= it[4] < test_end]
        else:
            train_data = [it for it in data_with_dates if it[3] < val_start]
            val_data = [it for it in data_with_dates if val_start <= it[3] < val_end]
            test_data = [it for it in data_with_dates if test_start <= it[3] < test_end]
        
        if len(train_data) < min_train_samples:
            print(f"训练样本不足 ({len(train_data)} < {min_train_samples})，停止滚动")
            break
        
        window_info = {
            'window_idx': window_idx,
            'train_start': min_date,
            'train_end': val_start - pd.Timedelta(days=1),
            'val_start': val_start,
            'val_end': val_end - pd.Timedelta(days=1),
            'test_start': test_start,
            'test_end': test_end - pd.Timedelta(days=1),
            'window_start': min_date,
            'window_end': test_end - pd.Timedelta(days=1),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'is_last_window': is_last_window,  # 标记是否为最后一个窗口
            'quarter_start': current_train_end  # 记录季度第一天
        }
        
        rolling_windows.append((train_data, val_data, test_data, window_info))
        
        if is_last_window:
            actual_test_days = len(test_data)  # 使用实际测试集样本数
            expected_test_days = test_window_days
            print(f"⚠️ 最后一个窗口：测试集不足 {expected_test_days} 天，实际测试集样本数: {actual_test_days} 个")
            break
        
        # 移动到下一个季度第一天
        current_train_end = get_next_quarter_start(current_train_end)
        window_idx += 1
    
    print(f"生成 {len(rolling_windows)} 个季度滚动窗口")
    for i, (_, _, _, window_info) in enumerate(rolling_windows):
        last_window_marker = " [最后一个窗口]" if window_info.get('is_last_window', False) else ""
        quarter_date = window_info.get('quarter_start', window_info['val_start'])
        print(f"窗口 {i+1}{last_window_marker} (季度开始: {quarter_date.date()}): "
              f"训练 {window_info['train_start'].date()} ~ {window_info['train_end'].date()} "
              f"({window_info['train_samples']} 样本) -> "
              f"验证 {window_info['val_start'].date()} ~ {window_info['val_end'].date()} "
              f"({window_info['val_samples']} 样本) -> "
              f"测试 {window_info['test_start'].date()} ~ {window_info['test_end'].date()} "
              f"({window_info['test_samples']} 样本)")
    
    return rolling_windows


def create_monthly_sliding_window_time_split(data_with_dates, train_window_days=360,
                                              val_window_days=60, test_window_days=30,
                                              roll_step_days=30, min_train_samples=1000):
    """
    固定长度滑动窗口时间序列分割（非扩展式）

    特点：
    - 训练集长度固定（不会越来越大），只用最近 N 天数据
    - 每隔 roll_step_days 滑动一次，重新训练模型
    - 适应 regime change 更快
    """
    print(f"🔄 固定窗口滑动时间序列分割（训练{train_window_days}天, 验证{val_window_days}天, "
          f"测试{test_window_days}天, 步长{roll_step_days}天）...")

    is_5tuple = len(data_with_dates[0]) == 5
    date_idx = 4 if is_5tuple else 3
    data_with_dates.sort(key=lambda x: x[date_idx])
    dates = [item[date_idx] for item in data_with_dates]

    min_date, max_date = min(dates), max(dates)
    print(f"数据时间范围: {min_date.date()} ~ {max_date.date()}")

    rolling_windows = []
    train_start = min_date
    window_idx = 0

    while True:
        train_end = train_start + pd.Timedelta(days=train_window_days)
        val_start = train_end
        val_end = val_start + pd.Timedelta(days=val_window_days)
        test_start = val_end
        test_end = test_start + pd.Timedelta(days=test_window_days)

        if val_start >= max_date:
            break

        is_last_window = False
        if test_end > max_date:
            if test_start < max_date:
                is_last_window = True
                test_end = max_date
            else:
                break

        train_data = [it for it in data_with_dates if train_start <= it[date_idx] < val_start]
        val_data = [it for it in data_with_dates if val_start <= it[date_idx] < val_end]
        test_data = [it for it in data_with_dates if test_start <= it[date_idx] < test_end]

        if len(train_data) < min_train_samples:
            train_start += pd.Timedelta(days=roll_step_days)
            continue

        if len(val_data) == 0 or len(test_data) == 0:
            if is_last_window:
                break
            train_start += pd.Timedelta(days=roll_step_days)
            continue

        window_info = {
            'window_idx': window_idx,
            'window_start': train_start,
            'train_start': train_start,
            'train_end': val_start - pd.Timedelta(days=1),
            'val_start': val_start,
            'val_end': val_end - pd.Timedelta(days=1),
            'test_start': test_start,
            'test_end': test_end - pd.Timedelta(days=1),
            'window_end': test_end - pd.Timedelta(days=1),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'is_last_window': is_last_window,
        }

        rolling_windows.append((train_data, val_data, test_data, window_info))

        if is_last_window:
            break

        train_start += pd.Timedelta(days=roll_step_days)
        window_idx += 1

    print(f"生成 {len(rolling_windows)} 个滑动窗口")
    for i, (_, _, _, wi) in enumerate(rolling_windows):
        last_tag = " [LAST]" if wi.get('is_last_window') else ""
        print(f"  窗口 {i+1}{last_tag}: "
              f"训练 {wi['train_start'].date()}~{wi['train_end'].date()} ({wi['train_samples']}样本) -> "
              f"验证 {wi['val_start'].date()}~{wi['val_end'].date()} ({wi['val_samples']}样本) -> "
              f"测试 {wi['test_start'].date()}~{wi['test_end'].date()} ({wi['test_samples']}样本)")

    return rolling_windows


def ic_stability_feature_selection(X_train, y_train, dates_train, symbols_train,
                                   X_val, y_val, dates_val, symbols_val,
                                   X_test,
                                   max_features=30, correlation_threshold=0.5,
                                   n_sub_periods=4, min_positive_ratio=0.6,
                                   recent_ic_days=60, min_recent_ic=0.0):
    """
    基于 IC 稳定性的特征选择：选择在多个子时段上 IC 都稳定为正的特征，
    并过滤最近 N 天 IC 衰减为 0 或变负的特征。

    Parameters:
    - n_sub_periods: 将训练集切成多少个子时段检查 IC 稳定性
    - min_positive_ratio: 特征在至少该比例的子时段中 IC > 0 才保留
    - recent_ic_days: 检查最近 N 天的 IC（使用验证集末尾或训练集末尾）
    - min_recent_ic: 最近 IC 必须 >= 该值才保留（默认 0）
    """
    print(f"🧬 IC 稳定性特征选择（{n_sub_periods} 子时段, 最近{recent_ic_days}天衰减检查）...")

    n_features = X_train.shape[1]
    dates_train_arr = np.array(dates_train)

    # 1. 去除高相关性特征
    orthogonalizer = FeatureOrthogonalizer(
        correlation_threshold=correlation_threshold,
        use_pca=False
    )
    X_train_ortho = orthogonalizer.fit_transform(X_train, y_train)
    X_val_ortho = orthogonalizer.transform(X_val)
    X_test_ortho = orthogonalizer.transform(X_test)
    n_ortho = X_train_ortho.shape[1]
    print(f"  正交化: {n_features} -> {n_ortho}")

    # 2. 子时段 IC 稳定性检查
    unique_dates = np.sort(np.unique(dates_train_arr))
    period_size = len(unique_dates) // n_sub_periods
    if period_size < 5:
        n_sub_periods = max(1, len(unique_dates) // 5)
        period_size = len(unique_dates) // n_sub_periods

    feature_positive_count = np.zeros(n_ortho)
    feature_ic_sums = np.zeros(n_ortho)

    for p in range(n_sub_periods):
        start_idx = p * period_size
        end_idx = (p + 1) * period_size if p < n_sub_periods - 1 else len(unique_dates)
        period_dates = set(unique_dates[start_idx:end_idx])
        mask = np.array([d in period_dates for d in dates_train_arr])
        if mask.sum() < 20:
            continue

        X_sub = X_train_ortho[mask]
        y_sub = y_train[mask]
        dates_sub = dates_train_arr[mask]

        for fi in range(n_ortho):
            df_tmp = pd.DataFrame({'date': dates_sub, 'feat': X_sub[:, fi], 'y': y_sub})
            daily_ics = []
            for _, grp in df_tmp.groupby('date'):
                if len(grp) >= 5:
                    ic = grp['feat'].corr(grp['y'])
                    if np.isfinite(ic):
                        daily_ics.append(ic)
            if daily_ics:
                mean_ic = np.mean(daily_ics)
                feature_ic_sums[fi] += mean_ic
                if mean_ic > 0:
                    feature_positive_count[fi] += 1

    positive_ratio = feature_positive_count / max(n_sub_periods, 1)
    print(f"  子时段 IC 稳定性: 通过率 {(positive_ratio >= min_positive_ratio).sum()}/{n_ortho}")

    # 3. 最近 N 天 IC 衰减检查（使用验证集数据）
    dates_val_arr = np.array(dates_val)
    unique_val_dates = np.sort(np.unique(dates_val_arr))
    recent_cutoff = unique_val_dates[-1] - pd.Timedelta(days=recent_ic_days) if len(unique_val_dates) > 0 else None

    recent_ic = np.full(n_ortho, np.nan)
    if recent_cutoff is not None:
        recent_mask = dates_val_arr >= recent_cutoff
        if recent_mask.sum() >= 20:
            X_recent = X_val_ortho[recent_mask]
            y_recent = y_val[recent_mask]
            dates_recent = dates_val_arr[recent_mask]
            for fi in range(n_ortho):
                df_tmp = pd.DataFrame({'date': dates_recent, 'feat': X_recent[:, fi], 'y': y_recent})
                daily_ics = []
                for _, grp in df_tmp.groupby('date'):
                    if len(grp) >= 5:
                        ic = grp['feat'].corr(grp['y'])
                        if np.isfinite(ic):
                            daily_ics.append(ic)
                if daily_ics:
                    recent_ic[fi] = np.mean(daily_ics)

    recent_pass = np.where(np.isfinite(recent_ic), recent_ic >= min_recent_ic, True)
    print(f"  近期 IC 衰减检查: 通过 {recent_pass.sum()}/{n_ortho}")

    # 4. 综合筛选
    stability_pass = positive_ratio >= min_positive_ratio
    combined_pass = stability_pass & recent_pass

    # 用 IC 总和排序
    candidate_indices = np.where(combined_pass)[0]
    if len(candidate_indices) == 0:
        print("  ⚠️ 没有特征通过 IC 稳定性筛选，回退到 top IC 特征")
        candidate_indices = np.argsort(-feature_ic_sums)[:max_features]

    if len(candidate_indices) > max_features:
        sorted_by_ic = np.argsort(-feature_ic_sums[candidate_indices])
        candidate_indices = candidate_indices[sorted_by_ic[:max_features]]

    selector = ConservativeSelector(orthogonalizer, candidate_indices)
    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)

    print(f"  IC 稳定性特征选择: {X_train.shape[1]} -> {X_train_sel.shape[1]}")
    if len(candidate_indices) > 0:
        avg_ratio = positive_ratio[candidate_indices].mean()
        avg_ic = feature_ic_sums[candidate_indices].mean() / max(n_sub_periods, 1)
        print(f"  入选特征: 平均稳定率={avg_ratio:.2f}, 平均子时段IC={avg_ic:.4f}")
        finite_recent = recent_ic[candidate_indices]
        finite_recent = finite_recent[np.isfinite(finite_recent)]
        if len(finite_recent) > 0:
            print(f"  入选特征: 近期平均IC={finite_recent.mean():.4f}")

    return selector, X_train_sel, X_val_sel, X_test_sel


def train_robust_xgboost(X_train, y_train, X_val, y_val, dates_val=None, symbols_val=None,
                        use_bayesian_opt=True, ic_ir_weight=3.35, rankIC=2.0, ir_weight=1.3, monotonicity_weight=0.7,
                        huber_delta=1.0):
    """更稳健的训练过程"""
    print("🛡️ 使用稳健训练策略...")

    # 1. 更严格的异常值处理和数据增强
    def robust_winsorize(y, lower_percentile=2, upper_percentile=98):
        lower_bound = np.percentile(y, lower_percentile)
        upper_bound = np.percentile(y, upper_percentile)
        return np.clip(y, lower_bound, upper_bound)

    # 添加数据增强正则化（轻微噪声注入）
    def add_noise_regularization(X, noise_level=0.01):
        """添加轻微噪声作为正则化"""
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise

    y_train_processed = robust_winsorize(y_train)
    y_val_processed = robust_winsorize(y_val)

    # 应用数据增强正则化
    X_train_augmented = add_noise_regularization(X_train, noise_level=0.005)

    print(f"目标收益率统计（稳健处理）:")
    print(f"  训练集: 均值={y_train_processed.mean():.6f}, 标准差={y_train_processed.std():.6f}")
    print(f"  验证集: 均值={y_val_processed.mean():.6f}, 标准差={y_val_processed.std():.6f}")

    # 2. 参数优化（如果启用）
    if use_bayesian_opt and SKOPT_AVAILABLE:
        print("🎯 使用贝叶斯优化寻找最优参数...")
        params, best_score = bayesian_optimize_xgboost(
            X_train, y_train_processed, X_val, y_val_processed,
            dates_val=dates_val, symbols_val=symbols_val,
            n_calls=25, n_random_starts=5,
            ic_ir_weight=ic_ir_weight, rankIC=rankIC, ir_weight=ir_weight, monotonicity_weight=monotonicity_weight
        )
        if best_score is not None:
            if dates_val is not None and symbols_val is not None:
                print(f"✅ 贝叶斯优化完成，最优加权IC指标: {best_score:.6f}")
            else:
                print(f"✅ 贝叶斯优化完成，最优验证MAE: {best_score:.6f}")
    else:
        print("📋 使用增强的正则化参数")
        params = get_enhanced_xgb_params(huber_delta=huber_delta)

    # 3. 增强正则化参数（如果未指定则使用更强的默认值）
    if 'reg_alpha' not in params:
        params['reg_alpha'] = 15.0  # 增强L1正则化
    if 'reg_lambda' not in params:
        params['reg_lambda'] = 25.0  # 增强L2正则化
    if 'gamma' not in params:
        params['gamma'] = 5.0  # 增强后剪枝
    if 'min_child_weight' not in params:
        params['min_child_weight'] = 30  # 增强最小叶子权重
    if 'subsample' not in params:
        params['subsample'] = 0.35  # 降低子采样
    if 'colsample_bytree' not in params:
        params['colsample_bytree'] = 0.2  # 降低特征采样

    dtrain = xgb.DMatrix(X_train_augmented, label=y_train_processed)
    dval = xgb.DMatrix(X_val, label=y_val_processed)

    early_stopping_rounds = int(params.pop('early_stopping_rounds', 150))
    n_estimators = int(params.pop('n_estimators', 1000))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=min(n_estimators, 1500),
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100,
        obj=_huber_obj_global,
        custom_metric=_huber_metric_global,
    )

    best_iteration = getattr(model, 'best_iteration', None)
    if best_iteration is not None:
        print(f"  Early stopping: 最优轮数 {best_iteration}")

    val_pred = model.predict(dval)
    val_rmse = np.sqrt(mean_squared_error(y_val_processed, val_pred))
    val_mae = mean_absolute_error(y_val_processed, val_pred)
    val_r2 = r2_score(y_val_processed, val_pred)

    print(f"[验证集-稳健训练] RMSE={val_rmse:.6f}, MAE={val_mae:.6f}, R2={val_r2:.4f}")

    return model, val_pred

def train_enhanced_xgboost_model(X_train, y_train, X_val, y_val, symbols_train, symbols_val, dates_train, dates_val,
                                 use_bayesian_opt=True, use_robust_training=False,
                                 ic_ir_weight=1.0, rankIC=1.0, ir_weight=1.0, monotonicity_weight=1.0,
                                 regularization_strength='default', huber_delta=1.0,
                                 horizon_days=None,
                                 bayes_objective='ic_ir'):
    """训练增强版XGBoost模型，支持贝叶斯优化和稳健训练"""
    if use_robust_training:
        return train_robust_xgboost(X_train, y_train, X_val, y_val,
                                   dates_val=dates_val, symbols_val=symbols_val,
                                   use_bayesian_opt=use_bayesian_opt,
                                   ic_ir_weight=ic_ir_weight, rankIC=rankIC, ir_weight=ir_weight, monotonicity_weight=monotonicity_weight,
                                   huber_delta=huber_delta)

    print("🚀 开始增强版XGBoost训练...")

    # 更智能的异常值处理
    def smart_clip_values(y, clip_percentile=98):
        """使用更智能的异常值处理"""
        # 使用IQR方法检测异常值
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 结合分位数方法
        lower_bound = max(lower_bound, np.percentile(y, 100 - clip_percentile))
        upper_bound = min(upper_bound, np.percentile(y, clip_percentile))

        return np.clip(y, lower_bound, upper_bound)

    y_train_clipped = smart_clip_values(y_train, clip_percentile=98)
    y_val_clipped = smart_clip_values(y_val, clip_percentile=98)

    print(f"目标收益率统计:")
    print(f"  训练集: 均值={y_train_clipped.mean():.6f}, 标准差={y_train_clipped.std():.6f}")
    print(f"  验证集: 均值={y_val_clipped.mean():.6f}, 标准差={y_val_clipped.std():.6f}")

    # 贝叶斯优化参数（可选）
    if use_bayesian_opt and SKOPT_AVAILABLE:
        print("🎯 使用贝叶斯优化寻找最优参数...")
        params, best_score = bayesian_optimize_xgboost(
            X_train, y_train_clipped, X_val, y_val_clipped,
            dates_val=dates_val, symbols_val=symbols_val,
            n_calls=30, n_random_starts=5,  # 减少迭代次数以加快速度
            ic_ir_weight=ic_ir_weight, rankIC=rankIC, ir_weight=ir_weight, monotonicity_weight=monotonicity_weight,
            horizon_days=horizon_days,
            bayes_objective=bayes_objective,
        )
        if best_score is not None:
            if dates_val is not None and symbols_val is not None:
                print(f"✅ 贝叶斯优化完成，最优加权IC指标: {best_score:.6f}")
            else:
                print(f"✅ 贝叶斯优化完成，最优验证MAE: {best_score:.6f}")
        else:
            print(f"📋 使用{regularization_strength}XGBoost参数")
            # 注意：train_enhanced_xgboost_model 函数内部无法访问 huber_delta，使用默认值1.0
            if regularization_strength == 'ultra':
                params = get_ultra_regularized_xgb_params(huber_delta=huber_delta)
            elif regularization_strength == 'enhanced':
                params = get_enhanced_xgb_params(huber_delta=huber_delta)
            else:  # 'default'
                params = get_default_xgb_params(huber_delta=huber_delta)
    else:
        print(f"📋 使用{regularization_strength}XGBoost参数")
        if regularization_strength == 'ultra':
            params = get_ultra_regularized_xgb_params(huber_delta=huber_delta)
        elif regularization_strength == 'enhanced':
            params = get_enhanced_xgb_params(huber_delta=huber_delta)
        else:  # 'default'
            params = get_default_xgb_params(huber_delta=huber_delta)

    dtrain = xgb.DMatrix(X_train, label=y_train_clipped)
    dval = xgb.DMatrix(X_val, label=y_val_clipped)

    early_stopping_rounds = int(params.pop('early_stopping_rounds', 150))
    n_estimators = int(params.pop('n_estimators', 1500))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=min(n_estimators, 2000),
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100,
        obj=_huber_obj_global,
        custom_metric=_huber_metric_global,
    )

    best_iteration = getattr(model, 'best_iteration', None)
    if best_iteration is not None:
        print(f"  Early stopping: 最优轮数 {best_iteration}")

    val_pred = model.predict(dval)
    val_rmse = np.sqrt(mean_squared_error(y_val_clipped, val_pred))
    val_mae = mean_absolute_error(y_val_clipped, val_pred)
    val_r2 = r2_score(y_val_clipped, val_pred)

    print(f"[验证集] RMSE={val_rmse:.6f}, MAE={val_mae:.6f}, R2={val_r2:.4f}")

    return model, val_pred

def print_latest_daily_groups_live(window: int, rebalance_period: int, min_history_days: int = 60,
                                   groups: int = 5, use_improved_features: bool = False,
                                   use_robust_preprocessing: bool = False, use_ga_features: bool = False,
                                   model_type: str = 'xgb'):
    """
    使用已训练的模型，对原始K线的"全局最新日期"做一次前向预测并分组打印。
    类似 Xgboost_Prediction_original.py 中的实现，但适配新的代码结构。
    只使用每日前50个币种。
    """
    # 读取数据与工件
    df = load_kline_df()

    # 🔥 关键：优先从指数缓存文件加载成分股列表，如果没有则从K线数据计算
    print(f"📊 Live推理：加载每日前50排名，只使用前50个币种...")

    # 尝试从指数缓存文件加载
    all_dates = sorted(df['date'].unique())
    available_tokens_by_date = load_available_tokens_from_index_cache(
        cache_file=None,  # 自动查找最新的index_cache文件
        top_n=50,
        all_dates=all_dates  # 传入所有日期，将调仓日的成分股扩展到所有交易日
    )

    # 如果缓存文件不存在或为空，回退到从K线数据计算
    if not available_tokens_by_date:
        print(f"📊 指数缓存文件不存在或为空，从K线数据构建每日前50排名...")
        available_tokens_by_date = build_available_tokens_by_date_from_kline(
            kline_df=df,
            top_n=50,
            ranking_method='quote_volume',
            rebalance_period=rebalance_period,
            strict_top_n=True  # 严格限制为50个币种
        )

    # 对于滚动训练，查找最新的窗口模型文件
    base_dir = Path(os.path.dirname(__file__)) / "models"
    base_dir.mkdir(parents=True, exist_ok=True)
    model_type = (model_type or 'xgb').lower()
    if model_type == 'lgbm':
        tag = f"lgbm_enhanced_w{window}_r{rebalance_period}"
        model_ext = ".txt"
    else:
        tag = f"xgb_enhanced_w{window}_r{rebalance_period}"
        model_ext = ".json"

    # 查找所有带窗口索引的模型文件
    model_files = list(base_dir.glob(f"{tag}_window*{model_ext}"))
    selector_files = list(base_dir.glob(f"{tag}_selector_window*.pkl"))

    if not model_files or not selector_files:
        # 如果没有找到窗口模型文件，尝试查找基础模型文件
        model_path, selector_path, scaler_path = _artifact_paths(window, rebalance_period, model_type=model_type)
        if not (model_path.exists() and selector_path.exists()):
            print("未找到已训练模型/预处理器，跳过live日度分组打印。")
            return
        # 使用基础模型文件
        model_path_to_use = model_path
        selector_path_to_use = selector_path
    else:
        # 找到最新的窗口模型文件（最大索引）
        model_files.sort(key=lambda x: int(x.stem.split('_window')[-1]))
        selector_files.sort(key=lambda x: int(x.stem.split('_selector_window')[-1]))

        model_path_to_use = model_files[-1]  # 最新的模型文件
        selector_path_to_use = selector_files[-1]  # 对应的选择器文件

        print(f"📦 使用最新的窗口模型: {model_path_to_use.name}")

    if model_type == 'lgbm':
        if not LGBM_AVAILABLE:
            print("LightGBM不可用，跳过live日度分组打印。")
            return
        booster = lgb.Booster(model_file=str(model_path_to_use))
    else:
        booster = xgb.Booster()
        booster.load_model(str(model_path_to_use))
    selector = joblib.load(selector_path_to_use)
    # 不再需要scaler，因为特征已经按日期进行了横截面标准化

    # 获取最新日期（确保类型一致）
    latest_date = df['date'].max()

    # 获取最新日期的前50个币种（尝试精确匹配，如果失败则找最接近的日期）
    latest_top50 = available_tokens_by_date.get(latest_date, [])
    if not latest_top50:
        # 如果精确匹配失败，找最接近的日期
        available_dates = sorted(available_tokens_by_date.keys())
        if available_dates:
            # 找到小于等于latest_date的最大日期
            closest_date = None
            for d in reversed(available_dates):
                if d <= latest_date:
                    closest_date = d
                    break
            if closest_date:
                latest_top50 = available_tokens_by_date[closest_date]
                latest_date = closest_date  # 使用找到的日期
                print(f"注意: 使用最接近的日期 {latest_date.date()} 的前50排名")

    if not latest_top50:
        print(f"最新日期 {latest_date.date()} 没有前50排名数据。")
        return

    print(f"最新日期 {latest_date.date()} 的前50个币种: {len(latest_top50)} 个")

    # 只为前50个币种构造"最新一行"特征
    feats = []
    symbols_with_features = []
    for sym in latest_top50:
        gp = df[df['symbol'] == sym]
        if gp.empty:
            continue
        row = create_features_symbol_inference_latest(
            gp.copy(),
            min_history_days=min_history_days,
            use_improved_features=use_improved_features
        )
        if row is not None and not row.empty:
            feats.append(row)
            symbols_with_features.append(sym)

    if not feats:
        print("无法构造最新日度特征，跳过live分组打印。")
        return

    feat_df = pd.concat(feats, ignore_index=True)

    # 合并遗传算法因子（与训练时保持一致）
    if use_ga_features:
        ga_df = _compute_ga_factors_df(df)
        if not ga_df.empty:
            ga_cols = [c for c in ga_df.columns if c not in ["date", "symbol"]]
            if ga_cols:
                feat_df["date_norm"] = pd.to_datetime(feat_df["date"]).dt.normalize()
                ga_df["date_norm"] = pd.to_datetime(ga_df["date"]).dt.normalize()
                feat_df = feat_df.merge(
                    ga_df[["date_norm", "symbol"] + ga_cols],
                    on=["date_norm", "symbol"],
                    how="left",
                )
                feat_df = feat_df.drop(columns=["date_norm"])
                feat_df[ga_cols] = feat_df.groupby("symbol", sort=False)[ga_cols].ffill().fillna(0)
                # 将 GA 因子列放在最前面（与训练时一致）
                base_cols = [c for c in feat_df.columns if c not in ["date", "symbol"] and c not in ga_cols]
                feat_df = feat_df[["date", "symbol"] + ga_cols + base_cols]

    # 🔥 关键修复：不应该按latest_date过滤，因为每个币种的特征已经是对应其各自最新日期的
    feat_df['date'] = pd.to_datetime(feat_df['date'])

    # 检查日期分布
    date_counts = feat_df['date'].value_counts().sort_index(ascending=False)
    most_common_date = date_counts.index[0]
    print(f"特征日期分布: {dict(date_counts.head(5))}")

    # 使用最常见的日期（通常是latest_date或接近latest_date）
    date_diff = abs((most_common_date - latest_date).days)
    if date_diff <= 2:
        target_date = most_common_date
        print(f"使用最常见的日期 {target_date.date()} 进行推理（与latest_date {latest_date.date()} 差异 {date_diff} 天）")
    else:
        target_date = latest_date
        print(f"使用latest_date {target_date.date()} 进行推理")

    # 选择目标日期或接近目标日期的特征（允许1天差异）
    day = feat_df[
        (feat_df['date'] == target_date) |
        (abs((feat_df['date'] - target_date).dt.days) <= 1)
    ].copy()

    if day.empty:
        print(f"目标日期 {target_date.date()} 无可推理样本。")
        return

    # 如果有多个日期，优先选择target_date的，然后选择最接近的
    if day['date'].nunique() > 1:
        day = day.sort_values('date', key=lambda x: abs((x - target_date).dt.days))
        # 对于每个symbol，只保留最接近target_date的那一行
        day = day.drop_duplicates(subset=['symbol'], keep='first')

    print(f"✅ 成功为 {len(day)} 个币种生成特征（目标日期: {target_date.date()}）")

    # 横截面标准化（与训练时保持一致）
    feature_cols = [col for col in day.columns if col not in ['date', 'symbol']]

    if use_robust_preprocessing:
        # 使用稳健标准化（中位数和MAD）
        def robust_standardize_by_date(group):
            """对每个日期的特征进行稳健标准化（使用中位数和MAD）"""
            group = group.copy()
            for col in feature_cols:
                median_val = group[col].median()
                mad_val = (group[col] - median_val).abs().median()
                if mad_val > 1e-8:
                    group[col] = (group[col] - median_val) / mad_val
                else:
                    group[col] = 0.0
            return group
        day = day.groupby('date', group_keys=False).apply(robust_standardize_by_date)
    else:
        # 使用标准标准化（均值和标准差）
        def standardize_by_date(group):
            """对每个日期的特征进行横截面标准化"""
            group = group.copy()
            for col in feature_cols:
                mean_val = group[col].mean()
                std_val = group[col].std()
                if std_val > 1e-8:
                    group[col] = (group[col] - mean_val) / std_val
                else:
                    group[col] = 0.0
            return group
        day = day.groupby('date', group_keys=False).apply(standardize_by_date)

    X = day.drop(columns=['date','symbol']).to_numpy(dtype=np.float32)
    X_sel = selector.transform(X)
    # 移除scaler.transform，因为已经在上面按日期进行了横截面标准化
    X_scl = X_sel.astype(np.float32, copy=False)
    if model_type == 'lgbm':
        preds = booster.predict(X_scl)
    else:
        preds = booster.predict(xgb.DMatrix(X_scl))

    out = pd.DataFrame({'instrument': day['symbol'].values, 'pred': preds})
    # 横截面秩值归一化到[-1,1]，并按百分位分组，展示更直观
    out['pct'] = out['pred'].rank(pct=True, method='average')
    out['score'] = 2 * out['pct'] - 1  # [-1, 1]

    try:
        out['group'] = pd.qcut(out['pct'], q=groups, labels=list(range(groups)), duplicates='drop')
    except Exception:
        bins = np.linspace(out['pct'].min() - 1e-9, out['pct'].max() + 1e-9, groups + 1)
        out['group'] = pd.cut(out['pct'], bins=bins, labels=list(range(groups)), include_lowest=True)
    out['group'] = out['group'].astype(int)

    print(f"\n==== {rebalance_period}日调仓视角 | 最新日期: {latest_date.date()} (Live 推理) ====")

    # CSV表头
    print("date,group,instrument,score,pred")

    for g in range(groups):
        sub = out[out['group'] == g].sort_values('score')
        for _, r in sub.iterrows():
            print(f"{latest_date.date()},{g},{r['instrument']},{r['score']:.4f},{r['pred']:.6f}")
