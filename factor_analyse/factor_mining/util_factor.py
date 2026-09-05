# factor_analyse/factor_mining/util_factor.py
import os
from datetime import timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> Crypto
DATA_DIR = os.path.join(BASE_DIR, "data")
KLINE_DIR = os.path.join(DATA_DIR, "kline_data")
FACTOR_DIR = os.path.join(DATA_DIR, "factor_data")
HIST_MCAP_PATH = os.path.join(DATA_DIR, "coingecko_top100_marketcap_historical.csv")
HIST_MCAP_ALT_PATH = os.path.join(DATA_DIR, "binance_coingecko_top100_marketcap_historical.csv")

# 注意：稳定币在K线数据获取时已经排除，这里不需要再次排除
EXCLUDED_FIAT_BASE_ASSETS = {
    "U", "EUR", "TRY", "RUB", "BRL", "UAH", "BIDR", "IDRT", "NGN", "PLN", "RON",
    "JPY", "GBP", "AUD", "CAD", "CHF", "NZD", "HKD", "SGD", "ZAR", "MXN",
    "ARS", "CLP", "COP", "PEN", "KES", "GHS", "MAD", "EGP", "SAR", "AED",
    "QAR", "KWD", "BHD", "OMR", "JOD", "THB", "VND", "MYR", "IDR", "PHP", "TWD"
}


def _is_excluded_fiat_symbol(symbol: str) -> bool:
    sym = str(symbol).upper()
    if not sym.endswith("USDT"):
        return False
    return sym[:-4] in EXCLUDED_FIAT_BASE_ASSETS

def _resolve_marketcap_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if path == HIST_MCAP_PATH and os.path.exists(HIST_MCAP_ALT_PATH):
        return HIST_MCAP_ALT_PATH
    raise FileNotFoundError(f"未找到历史市值排名文件: {path}")


def _normalize_trading_pair_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容历史文件列名差异：
    - 新文件常见字段: Trading_Pairs（逗号分隔）
    - 旧逻辑字段: Trading_Pair（单值）
    """
    if "Trading_Pair" in df.columns:
        df["Trading_Pair"] = df["Trading_Pair"].astype(str).str.strip()
        return df

    if "Trading_Pairs" in df.columns:
        work = df.copy()
        work["Trading_Pairs"] = work["Trading_Pairs"].fillna("").astype(str)
        work["Trading_Pair"] = work["Trading_Pairs"].str.split(",")
        work = work.explode("Trading_Pair")
        work["Trading_Pair"] = work["Trading_Pair"].fillna("").astype(str).str.strip()
        work = work[work["Trading_Pair"] != ""].copy()
        return work

    raise KeyError("历史市值文件缺少 Trading_Pair/Trading_Pairs 列")


def load_historical_marketcap(path: str = HIST_MCAP_PATH) -> pd.DataFrame:
    resolved_path = _resolve_marketcap_path(path)
    df = pd.read_csv(resolved_path)
    df["Date"] = pd.to_datetime(df["Date"], format='mixed', errors='coerce')
    df = df.dropna(subset=["Date"])
    df = _normalize_trading_pair_column(df)
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

        # 排除法币类交易对（如 EURUSDT、UUSDT），避免进入前N
        ranking_df = ranking_df[
            ~ranking_df["symbol"].astype(str).map(_is_excluded_fiat_symbol)
        ].copy()
        
        # 过滤掉无效值（NaN、0、负数）
        ranking_df = ranking_df[
            (ranking_df['rank_score'].notna()) &
            (ranking_df['rank_score'] > 0)
        ].copy()

        # 排序并选择前N个
        ranking_df = ranking_df.sort_values('rank_score', ascending=False)
        top_n_symbols = ranking_df.head(top_n)['symbol'].tolist()
        
        daily_top50[date] = top_n_symbols
    
    return daily_top50

def build_available_tokens_by_date_from_kline(
    kline_df: pd.DataFrame,
    top_n: int = 50,
    ranking_method: str = 'quote_volume',
    rebalance_period: int = 10,
    lookback_buffer: int = 20,
    strict_top_n: bool = True  # 保留参数兼容性，但已无实际作用
) -> Dict[pd.Timestamp, List[str]]:
    """
    基于K线数据构建每日可用token列表（前N名）
    
    每天基于当日成交额排名，严格选择前N个token，不使用未来信息。
    
    Args:
        kline_df: K线数据DataFrame（包含所有可能进入前50的token的完整数据）
        top_n: 每天选择前N个token，默认50
        ranking_method: 排名方法 ('quote_volume', 'volume')
        rebalance_period: 调仓周期（天），默认10天（保留参数兼容性）
        lookback_buffer: 回看缓冲区（天），默认20天（保留参数兼容性）
        strict_top_n: 已废弃，保留兼容性
        
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
    
    for current_date in all_dates:
        current_top50 = daily_top50.get(current_date, [])
        available_tokens[current_date] = current_top50[:top_n] if len(current_top50) > top_n else current_top50
    
    return available_tokens

def load_available_tokens_from_index_cache(
    cache_file: str = None,
    top_n: int = 50,
    all_dates: List[pd.Timestamp] = None
) -> Dict[pd.Timestamp, List[str]]:
    """
    从指数缓存CSV文件加载每日可用token列表
    
    关键特性：
    - 对于每个日期，使用该月最近一次调仓日的成分股列表
    - 例如：计算12月17日的因子时，使用12月1日（该月最近一次调仓日）的成分股
    - 如果components为空，会查找同月内最近的一次有效调仓日
    
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

        def _as_bool(v) -> bool:
            if isinstance(v, bool):
                return v
            if pd.isna(v):
                return False
            if isinstance(v, (int, float)):
                return bool(v)
            return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

        use_rebalance_flag = 'is_rebalance_day' in df.columns
        for _, row in df.iterrows():
            date = row['date']
            if pd.isna(date):
                continue

            # 如果文件显式提供调仓日标记，则只把调仓日视为“成分锚点”
            if use_rebalance_flag and not _as_bool(row.get('is_rebalance_day')):
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

def latest_kline_path(prefix: str = "binance_daily_klines_") -> str:
    # 排除统计文件（包含_daily_token_count的文件）
    files = sorted([f for f in os.listdir(KLINE_DIR)
                   if f.startswith(prefix) and f.endswith(".csv") and '_daily_token_count' not in f])
    if not files:
        raise FileNotFoundError("未找到K线数据文件")
    return os.path.join(KLINE_DIR, files[-1])

def load_kline_df(path: str | None = None) -> pd.DataFrame:
    path = path or latest_kline_path()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "symbol" in df.columns:
        df = df[~df["symbol"].astype(str).map(_is_excluded_fiat_symbol)].copy()
    df = df.sort_values(["symbol", "date"])
    return df

def filter_group_by_availability(group: pd.DataFrame, symbol: str, available_tokens_by_date: dict) -> pd.DataFrame:
    """
    根据可用token列表筛选数据：只保留symbol在当日Top50中的行，移除不在列表中的行。
    
    注意：因子计算应在调用此函数之前完成（使用token的完整历史），
    此函数仅过滤输出行，确保最终结果只包含token在投资宇宙中的日期。
    """
    if group.empty or not available_tokens_by_date:
        return group

    # 统一日期key并转为set，后续 membership O(1)
    normalized_map: Dict[pd.Timestamp, set] = {}
    for k, tokens in available_tokens_by_date.items():
        dk = pd.to_datetime(k)
        normalized_map[dk] = set(tokens)

    available_dates = np.array(sorted(normalized_map.keys()), dtype="datetime64[ns]")
    if len(available_dates) == 0:
        return group.iloc[0:0].copy()

    group_dates = pd.to_datetime(group["date"])
    unique_group_dates = np.sort(group_dates.unique().astype("datetime64[ns]"))
    nearest_idx = np.searchsorted(available_dates, unique_group_dates, side="right") - 1

    # 为每个交易日预计算是否可用，避免逐行逐日期双循环
    day_allowed: Dict[pd.Timestamp, bool] = {}
    for gd, idx in zip(unique_group_dates, nearest_idx):
        gdt = pd.Timestamp(gd)
        if idx < 0:
            day_allowed[gdt] = False
            continue
        nearest_date = pd.Timestamp(available_dates[idx])
        day_allowed[gdt] = symbol in normalized_map.get(nearest_date, set())

    mask = group_dates.map(day_allowed).fillna(False).to_numpy(dtype=bool)
    if not mask.any():
        return group.iloc[0:0].copy()
    return group.loc[mask].copy()

def filter_by_daily_top50(
    kline_df: pd.DataFrame,
    daily_top50: Dict[pd.Timestamp, List[str]],
    rebalance_period: int = 10
) -> pd.DataFrame:
    """
    根据每日前50排名筛选数据，只保留曾经出现在前50中的token的完整K线历史。
    
    不使用未来信息：仅收集截至各日期已知的前50 token。
    
    Args:
        kline_df: 完整的K线数据
        daily_top50: 每日前50的token列表
        rebalance_period: 调仓周期（天），保留参数兼容性
        
    Returns:
        筛选后的DataFrame
    """
    all_needed_symbols = set()
    for symbols in daily_top50.values():
        all_needed_symbols.update(symbols)

    filtered_df = kline_df[kline_df['symbol'].isin(all_needed_symbols)].copy()
    return filtered_df

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
    """Delegate to the shared daily MAD preprocessing implementation."""
    from factor_common.preprocessing import winsorize_by_date as winsorize
    return winsorize(df, col, n_std)

def rank_to_unit_by_date(df: pd.DataFrame, col: str, out_col: str = "factor") -> pd.DataFrame:
    """Delegate to the shared daily rank preprocessing implementation."""
    from factor_common.preprocessing import rank_to_unit_by_date as rank
    return rank(df, col, out_col)

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

def print_latest_date_inference(factor_df: pd.DataFrame, latest_date: pd.Timestamp, groups: int = 5) -> None:
    """
    打印最新日期的因子推理结果
    
    参数:
    factor_df: 因子数据DataFrame，包含 date, instrument, factor, future_ret
    latest_date: 最新日期
    groups: 分组数量，默认为5
    """
    # 获取最新日期的数据
    latest_data = factor_df[factor_df['date'] == latest_date].copy()
    
    if latest_data.empty:
        print(f"警告: 最新日期 {latest_date.date()} 没有因子数据")
        return
    
    # 计算百分位排名和分数
    latest_data['pct'] = latest_data['factor'].rank(pct=True, method='average')
    latest_data['score'] = 2 * latest_data['pct'] - 1  # [-1, 1]
    
    # 分组
    try:
        latest_data['group'] = pd.qcut(latest_data['pct'], q=groups, labels=list(range(groups)))
    except Exception:
        bins = np.linspace(latest_data['pct'].min() - 1e-9, latest_data['pct'].max() + 1e-9, groups + 1)
        latest_data['group'] = pd.cut(latest_data['pct'], bins=bins, labels=list(range(groups)), include_lowest=True)
    latest_data['group'] = latest_data['group'].astype(int)
    
    print(f"\n==== 最新日期推理 | 日期: {latest_date.date()} (Live 推理) ====")
    print("分组说明: G0=第一组(低分组) | G4=最后一组(高分组)")

    # 重点展示首尾分组，避免输出过长
    key_groups = [0, groups - 1] if groups > 1 else [0]
    max_rows_per_group = 12

    for g in key_groups:
        sub = latest_data[latest_data['group'] == g].copy()
        if sub.empty:
            print(f"\n--- G{g} ---")
            print("无样本")
            continue

        # G0 看最低分，G4 看最高分，更直观地区分首尾组
        if g == 0:
            sub = sub.sort_values(['score', 'factor'], ascending=[True, True])
            title = f"G{g} 第一组(低分组)"
        else:
            sub = sub.sort_values(['score', 'factor'], ascending=[False, False])
            title = f"G{g} 最后一组(高分组)"

        total_n = len(sub)
        show = sub.head(max_rows_per_group)
        print(f"\n--- {title} | 样本数={total_n} | 展示前{len(show)}条 ---")
        print("instrument,score,factor")
        for _, r in show.iterrows():
            print(f"{r['instrument']},{r['score']:.4f},{r['factor']:.6f}")
        if total_n > len(show):
            print(f"... 省略 {total_n - len(show)} 条")

    # 显示全分组统计（紧凑格式）
    print("\n分组统计:")
    group_stats = latest_data.groupby('group').agg(
        count=('instrument', 'count'),
        score_min=('score', 'min'),
        score_max=('score', 'max'),
        score_mean=('score', 'mean'),
        factor_min=('factor', 'min'),
        factor_max=('factor', 'max'),
        factor_mean=('factor', 'mean'),
    ).round(4)
    for g, row in group_stats.iterrows():
        if g == 0:
            label = "G0(第一组/低分)"
        elif g == groups - 1:
            label = f"G{g}(最后一组/高分)"
        else:
            label = f"G{g}"
        print(
            f"{label}: n={int(row['count'])}, "
            f"score[{row['score_min']:.4f},{row['score_max']:.4f}], "
            f"score_mean={row['score_mean']:.4f}, "
            f"factor[{row['factor_min']:.6f},{row['factor_max']:.6f}], "
            f"factor_mean={row['factor_mean']:.6f}"
        )
