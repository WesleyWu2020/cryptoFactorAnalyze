import pandas as pd
import os
import time
import requests
import json
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta
import csv
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Binance API credentials - replace with your own if you have rate limit concerns
API_KEY = ""  # 可以为空
API_SECRET = ""  # 可以为空

# 法币类 base asset（需要排除）
# 重点排除：U/EUR（用户指定）
EXCLUDED_FIAT_BASE_ASSETS = {
    "U", "EUR", "TRY", "RUB", "BRL", "UAH", "BIDR", "IDRT", "NGN", "PLN", "RON",
    "JPY", "GBP", "AUD", "CAD", "CHF", "NZD", "HKD", "SGD", "ZAR", "MXN",
    "ARS", "CLP", "COP", "PEN", "KES", "GHS", "MAD", "EGP", "SAR", "AED",
    "QAR", "KWD", "BHD", "OMR", "JOD", "THB", "VND", "MYR", "IDR", "PHP", "TWD"
}


def write_kline_metadata(csv_path: str, schema_version: str = "v1") -> None:
    """写入最小元信息，便于回放和排障。"""
    if not os.path.exists(csv_path):
        return
    try:
        df = pd.read_csv(csv_path, usecols=['date', 'symbol'])
        date_series = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        meta = {
            "dataset": os.path.basename(csv_path),
            "schema_version": schema_version,
            "source": "binance_daily_klines",
            "row_count": int(len(df)),
            "symbol_count": int(df['symbol'].nunique()),
            "date_min": str(date_series.min().date()) if date_series.notna().any() else None,
            "date_max": str(date_series.max().date()) if date_series.notna().any() else None,
            "updated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(f"{csv_path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️  写入K线元信息失败: {e}")


def resolve_incremental_output_path(
    output_dir: str,
    prefix: str = "binance_daily_klines_",
    now: datetime | None = None,
) -> str:
    """
    解析增量更新目标文件：
    - 若目录下已有K线主文件，始终复用最新一个进行追加更新（避免每天新建CSV）
    - 若不存在，则按当天日期创建首个文件名
    """
    now = now or datetime.now()
    os.makedirs(output_dir, exist_ok=True)
    existing_files = sorted(
        [
            f for f in os.listdir(output_dir)
            if f.startswith(prefix) and f.endswith(".csv") and "_daily_token_count" not in f
        ]
    )
    if existing_files:
        return os.path.join(output_dir, existing_files[-1])
    return os.path.join(output_dir, f"{prefix}{now.strftime('%Y%m%d')}.csv")


def load_existing_kline_data(path: str) -> pd.DataFrame:
    """加载已有K线文件；不存在或为空时返回空DataFrame。"""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return df
    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"], format="mixed", errors="coerce")
        df = df[date_series.notna()].copy()
        df["date"] = date_series[date_series.notna()].dt.strftime("%Y-%m-%d")
    return df


def build_incremental_fetch_plan(
    valid_trading_pairs: dict,
    existing_kline_data: pd.DataFrame,
    now: datetime | None = None,
) -> dict:
    """
    基于历史候选池与已有K线数据生成增量抓取计划。
    对已有symbol，仅抓其最后一个交易日之后的数据；新symbol从首次出现前90天开始抓。
    """
    now = now or datetime.now()
    today = pd.Timestamp(now.date())
    last_date_by_symbol = {}

    if not existing_kline_data.empty and {"symbol", "date"}.issubset(existing_kline_data.columns):
        existing = existing_kline_data[["symbol", "date"]].copy()
        existing["date"] = pd.to_datetime(existing["date"], format="mixed", errors="coerce")
        existing = existing[existing["date"].notna()]
        if not existing.empty:
            last_date_by_symbol = existing.groupby("symbol")["date"].max().to_dict()

    plan = {}
    for pair, date_info in valid_trading_pairs.items():
        default_start = pd.Timestamp(date_info["first_date"]) - pd.Timedelta(days=90)
        last_date = last_date_by_symbol.get(pair)
        if pd.notna(last_date):
            fetch_start = pd.Timestamp(last_date) + pd.Timedelta(days=1)
        else:
            fetch_start = default_start

        if fetch_start > today:
            continue

        row = date_info.copy()
        row["fetch_start_date"] = pd.Timestamp(fetch_start).normalize()
        plan[pair] = row

    return plan


def merge_kline_data(existing_kline_data: pd.DataFrame, new_kline_data: pd.DataFrame) -> pd.DataFrame:
    """合并已有数据与新抓取数据，按(symbol,date)去重并按(date,symbol)排序。"""
    if existing_kline_data is None or existing_kline_data.empty:
        merged = new_kline_data.copy()
    elif new_kline_data is None or new_kline_data.empty:
        merged = existing_kline_data.copy()
    else:
        merged = pd.concat([existing_kline_data, new_kline_data], ignore_index=True)

    if merged.empty:
        return merged

    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    if "datetime" in merged.columns:
        merged["datetime"] = pd.to_datetime(merged["datetime"], format="mixed", errors="coerce")

    merged = merged.drop_duplicates(subset=["symbol", "date"], keep="last")
    merged = merged.sort_values(["date", "symbol"]).reset_index(drop=True)
    return merged


def filter_valid_pairs_by_latest_guide_date(valid_trading_pairs: dict, expanded_df: pd.DataFrame) -> dict:
    """
    仅保留指引CSV最新日期出现的交易对，避免每天对历史全部token做增量更新。
    """
    if not valid_trading_pairs:
        return {}
    if expanded_df is None or expanded_df.empty or "Date" not in expanded_df.columns or "Trading_Pair" not in expanded_df.columns:
        return valid_trading_pairs

    dates = pd.to_datetime(expanded_df["Date"], format="mixed", errors="coerce")
    if dates.notna().sum() == 0:
        return valid_trading_pairs

    latest_date = dates.max().date()
    latest_pairs = set(
        expanded_df.loc[dates.dt.date == latest_date, "Trading_Pair"].astype(str).str.strip()
    )
    return {pair: info for pair, info in valid_trading_pairs.items() if pair in latest_pairs}

def fetch_historical_klines_from_coingecko_csv(max_workers: int = 10):
    """
    基于历史流动性/市值候选CSV获取相关 token 的 Binance USDT K 线数据
    严格避免未来函数，按时间顺序处理token
    
    Args:
        max_workers: 并发线程数，默认10（可根据网络情况调整，建议5-20）
    """
    # 初始化Binance客户端（增加超时时间）
    client = Client(
        API_KEY, 
        API_SECRET,
        requests_params={'timeout': 30}  # 增加超时时间到30秒
    )
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 读取历史候选币种文件
    csv_path = os.path.join(current_dir, "binance_coingecko_top100_marketcap_historical.csv")
    
    print(f"📊 正在读取历史候选币种数据: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return
    
    historical_df = pd.read_csv(csv_path)
    
    print(f"✅ 成功读取 {len(historical_df)} 条历史记录")
    
    # 检查必要的列
    required_columns = ['Date', 'Symbol', 'Trading_Pairs']
    missing_columns = [col for col in required_columns if col not in historical_df.columns]
    if missing_columns:
        print(f"❌ CSV文件中缺少必要的列: {missing_columns}")
        print(f"   现有列: {list(historical_df.columns)}")
        return
    
    # 将Date列转换为datetime（支持混合格式）
    historical_df['Date'] = pd.to_datetime(historical_df['Date'], format='mixed', errors='coerce')
    
    print(f"📅 数据时间范围: {historical_df['Date'].min()} 到 {historical_df['Date'].max()}")
    
    # 按日期排序，确保时间顺序处理
    historical_df = historical_df.sort_values('Date').reset_index(drop=True)
    
    # 🔥 处理Trading_Pairs列（可能包含多个交易对，用逗号分隔）
    # 展开Trading_Pairs，将每个交易对单独处理
    expanded_rows = []
    for idx, row in historical_df.iterrows():
        trading_pairs_str = str(row['Trading_Pairs']) if pd.notna(row['Trading_Pairs']) else ''
        # 分割交易对（可能用逗号分隔）
        trading_pairs = [tp.strip() for tp in trading_pairs_str.split(',') if tp.strip()]
        
        if not trading_pairs:
            continue
        
        # 为每个交易对创建一行
        for tp in trading_pairs:
            expanded_row = row.copy()
            expanded_row['Trading_Pair'] = tp  # 添加单数形式的列
            expanded_rows.append(expanded_row)
    
    if not expanded_rows:
        print("❌ 没有找到有效的交易对数据")
        return
    
    expanded_df = pd.DataFrame(expanded_rows)
    
    # 🔥 优化：只获取CSV文件中每个token实际出现过的日期范围内的数据
    # 统计每个token在CSV中出现的日期范围（首次和最后出现时间）
    token_date_ranges = {}
    for pair, group in expanded_df.groupby('Trading_Pair'):
        min_date = group['Date'].min()
        max_date = group['Date'].max()
        token_date_ranges[pair] = {
            'first_date': min_date,
            'last_date': max_date,
            'appearance_dates': set(group['Date'].dt.date)  # 记录所有出现过的日期
        }
    
    print(f"🪙 发现 {len(token_date_ranges)} 个独特的交易对")
    print(f"📋 前10个交易对及出现时间范围:")
    for i, (pair, date_info) in enumerate(list(token_date_ranges.items())[:10]):
        print(f"   {pair}: {date_info['first_date'].strftime('%Y-%m-%d')} 至 {date_info['last_date'].strftime('%Y-%m-%d')}")
    
    # 动态拉取币安USDT交易对清单，优先按交易所返回做校验（降低硬编码名单维护成本）
    binance_usdt_symbols = set()
    try:
        exchange_info = client.get_exchange_info()
        symbols = exchange_info.get('symbols', [])
        binance_usdt_symbols = {
            s.get('symbol', '')
            for s in symbols
            if s.get('status') == 'TRADING' and s.get('quoteAsset') == 'USDT'
        }
        print(f"✅ 动态获取币安可交易USDT交易对: {len(binance_usdt_symbols)} 个")
    except Exception as e:
        print(f"⚠️  动态获取exchangeInfo失败，将回退到静态规则: {e}")

    # 过滤出币安支持的交易对
    valid_trading_pairs = {}
    excluded_pairs = []
    
    # 一些可能不在币安的交易对预过滤
    # CoinGecko数据中可能包含一些币安不支持的交易对
    known_excluded = {
        'STETHUSDT',      # Lido Staked Ether - 可能不在币安
        'FIGR_HELOCUSDT', # Figure Heloc - 可能不在币安
        'BSC-USDUSDT',    # Binance Bridged USDT - 格式问题（包含连字符）
        'CBBTCUSDT',      # Coinbase Wrapped BTC - 可能不在币安
        'HYPEUSDT',       # Hyperliquid - 可能不在币安或名称不同
        'SUSDEUSDT',      # Ethena Staked USDe - 可能不在币安
        'USDT0USDT',      # USDT0 - 可能是稳定币变种
        'XUSDUSDT',       # XUSD - 稳定币
        'SUSDSUSDT',      # sUSDS - 可能是稳定币变种
        'BUIDLUSDT',      # BUIDL - 不在币安
        'C1USDUSDT',      # C1USD - 不在币安
        'CCUSDT',         # CC - 不在币安
        'EZETHUSDT',      # EZETH - 不在币安
        'FBTCUSDT',       # FBTC - 不在币安
        'FLRUSDT',        # FLR - 不在币安
        'FTNUSDT',        # FTN - 不在币安
        'HASHUSDT',       # HASH - 不在币安
        'HTXUSDT',        # HTX - 不在币安
        'IPUSDT',         # IP - 不在币安
        'JAAAUSDT',       # JAAA - 不在币安
        'JITOSOLUSDT',    # JITOSOL - 不在币安
        'JLPUSDT',        # JLP - 不在币安
        'KASUSDT',        # KAS - 不在币安
        'KHYPEUSDT',      # KHYPE - 不在币安
        'LBTCUSDT',       # LBTC - 不在币安
        'LSETHUSDT',      # LSETH - 不在币安
    }
    
    for pair, date_info in token_date_ranges.items():
        # 排除法币类交易对（如 EURUSDT、UUSDT）
        if pair.endswith('USDT'):
            base = pair[:-4].upper()
            if base in EXCLUDED_FIAT_BASE_ASSETS:
                excluded_pairs.append(pair)
                continue

        # ⚠️ 幸存者偏差修复：不再按 exchange_info 的 status=TRADING 硬排除。
        # Binance 对已下架币种的历史 K 线通常仍保留，应尝试抓取，由 fetch_single_pair
        # 的 "Invalid symbol" 分支兜底处理真正不可获取的 symbol。
        # if binance_usdt_symbols and pair not in binance_usdt_symbols:
        #     excluded_pairs.append(pair)
        #     continue

        # 排除已知不支持的交易对
        if pair in known_excluded:
            excluded_pairs.append(pair)
            continue
        
        # 排除包含连字符或下划线的交易对（币安通常不支持）
        if '-' in pair or '_' in pair:
            excluded_pairs.append(pair)
            continue
        
        # 排除明显是稳定币的交易对
        if pair.startswith('USDT') or pair.startswith('USDC') or pair.startswith('BUSD'):
            excluded_pairs.append(pair)
            continue
        
        valid_trading_pairs[pair] = date_info
    
    # ⚠️ 幸存者偏差修复：不再按"指引 CSV 最新日期"收窄候选池。
    # 原逻辑只抓取当前 Top100，导致历史上曾入榜但现已掉出的币种（2021-2025 年共 279 个）
    # 从未进入抓取清单，K 线数据完全缺失，回测 panel 存在严重 survivorship bias。
    # 保留该函数定义以便未来可选择性启用；此处默认放行全部历史候选。
    # valid_trading_pairs = filter_valid_pairs_by_latest_guide_date(valid_trading_pairs, expanded_df)

    print(f"🎯 将尝试获取 {len(valid_trading_pairs)} 个币安交易对（包含历史曾入榜币种，修复幸存者偏差）")
    if excluded_pairs:
        print(f"❌ 预先排除 {len(excluded_pairs)} 个可能不支持的交易对: {excluded_pairs}")
    
    print(f"\n💡 策略说明:")
    print(f"   1. 只获取指引CSV最新日期出现的交易对（排除稳定币后约{len(valid_trading_pairs)}个）")
    print(f"   2. 增量更新：优先复用现有K线主文件，只抓每个token缺失日期数据")
    print(f"   3. 这样避免获取token已不在前100之后的数据，减少数据量和API调用")
    print(f"   4. 每天基于Binance实际数据选择前50个（使用build_market_cap_ranking_from_binance.py）")
    
    # 创建输出目录
    output_dir = os.path.join(current_dir, "kline_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析最终CSV文件路径（增量复用最新主文件，避免每天新建）
    final_path = resolve_incremental_output_path(output_dir)
    existing_kline_data = load_existing_kline_data(final_path)
    existing_rows = len(existing_kline_data)
    if os.path.exists(final_path):
        print(f"📁 增量目标文件: {final_path}（已存在 {existing_rows} 行）")
    else:
        print(f"📁 增量目标文件: {final_path}（首次创建）")

    incremental_trading_pairs = build_incremental_fetch_plan(valid_trading_pairs, existing_kline_data)
    if not incremental_trading_pairs:
        print("✅ 现有数据已是最新，无需追加。")
        if os.path.exists(final_path):
            write_kline_metadata(final_path)
        return
    
    # 创建一个空的DataFrame来存储本次新增数据
    new_kline_data = pd.DataFrame()
    
    # 统计信息（使用线程安全的锁）
    stats_lock = Lock()
    success_count = 0
    error_count = 0
    invalid_symbol_count = 0
    
    print(f"\n🚀 开始并发获取 {len(incremental_trading_pairs)} 个交易对的增量K线数据（并发数: {max_workers}）...")
    print("⚠️ 重要：新token首次抓取含90天历史缓冲，老token仅抓最后日期之后的数据")
    
    def fetch_single_pair(trading_pair: str, date_info: dict) -> tuple:
        """
        获取单个交易对的K线数据（工作函数）
        
        Args:
            trading_pair: 交易对符号
            date_info: 包含first_date, last_date, appearance_dates的字典
        
        Returns:
            tuple: (success: bool, trading_pair: str, df: pd.DataFrame or None, error_type: str)
        """
        nonlocal success_count, error_count, invalid_symbol_count
        
        # 为每个线程创建独立的客户端（避免线程安全问题）
        thread_client = Client(
            API_KEY, 
            API_SECRET,
            requests_params={'timeout': 30}
        )
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 增量抓取：优先从symbol已有最后日期+1开始；新symbol从首次出现前90天开始
                start_date = pd.Timestamp(date_info.get('fetch_start_date', date_info['first_date'] - timedelta(days=90)))
                # 🔥 修改：采集到今天，而不是CSV中的最后日期（确保获取最新数据）
                end_date = datetime.now() + timedelta(days=1)
                
                # 获取日K线数据
                klines = thread_client.get_historical_klines(
                    symbol=trading_pair,
                    interval=Client.KLINE_INTERVAL_1DAY,
                    start_str=start_date.strftime('%Y-%m-%d'),
                    end_str=end_date.strftime('%Y-%m-%d')
                )
                
                # 如果没有数据，返回失败
                if not klines:
                    with stats_lock:
                        error_count += 1
                    return (False, trading_pair, None, "no_data")
                
                # 将K线数据转换为DataFrame
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 
                    'volume', 'close_time', 'quote_volume', 'trades_count', 
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                # 添加交易对信息
                df['symbol'] = trading_pair
                df['base_asset'] = trading_pair.replace('USDT', '')
                
                # 处理日期
                df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.strftime('%Y-%m-%d')
                df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
                df['date_only'] = df['datetime'].dt.date
                
                # 只保留本次增量区间的数据
                df = df[
                    df['datetime'] >= start_date
                ].copy()
                
                if len(df) == 0:
                    with stats_lock:
                        error_count += 1
                    return (False, trading_pair, None, "no_data_in_range")
                
                # 标记哪些日期是CSV中实际出现的（可选，用于后续分析）
                df['in_csv'] = df['date_only'].isin(date_info['appearance_dates'])
                
                # 转换数据类型
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                                 'taker_buy_base', 'taker_buy_quote']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['trades_count'] = pd.to_numeric(df['trades_count'], errors='coerce').astype('Int64')
                
                # 只保留基础K线数据列
                kline_columns = [
                    'symbol', 'base_asset', 'date', 'datetime',
                    'open', 'high', 'low', 'close', 
                    'volume', 'quote_volume', 'trades_count',
                    'taker_buy_base', 'taker_buy_quote',
                    'open_time', 'close_time', 'in_csv'  # 可选：保留标记
                ]
                
                existing_columns = [col for col in kline_columns if col in df.columns]
                df = df[existing_columns]
                
                with stats_lock:
                    success_count += 1
                
                return (True, trading_pair, df, None)
                
            except BinanceAPIException as e:
                if "Invalid symbol" in str(e):
                    with stats_lock:
                        invalid_symbol_count += 1
                    return (False, trading_pair, None, "invalid_symbol")
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2 ** retry_count)  # 指数退避
                    else:
                        with stats_lock:
                            error_count += 1
                        return (False, trading_pair, None, "api_error")
                        
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, 
                    requests.exceptions.SSLError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(5 * retry_count)
                else:
                    with stats_lock:
                        error_count += 1
                    return (False, trading_pair, None, "network_error")
                    
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(3 * retry_count)
                else:
                    with stats_lock:
                        error_count += 1
                    return (False, trading_pair, None, "unknown_error")
        
        return (False, trading_pair, None, "max_retries")
    
    # 使用线程池并发处理
    results = []
    failed_pairs = {}  # 记录失败的交易对及其date_info，用于重试（排除invalid_symbol）
    
    def run_fetch_batch(pairs_to_fetch: dict, desc: str) -> list:
        """执行一批获取任务，返回 (results, failed_pairs)"""
        nonlocal success_count, error_count, invalid_symbol_count
        batch_results = []
        batch_failed = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(fetch_single_pair, pair, date_info): (pair, date_info)
                for pair, date_info in pairs_to_fetch.items()
            }
            
            with tqdm(total=len(pairs_to_fetch), desc=desc) as pbar:
                for future in as_completed(future_to_pair):
                    pair, date_info = future_to_pair[future]
                    try:
                        success, trading_pair, df, error_type = future.result()
                        
                        if success and df is not None:
                            batch_results.append((trading_pair, df))
                            if len(batch_results) % 10 == 0:
                                pbar.set_postfix({
                                    '成功': success_count,
                                    '失败': error_count,
                                    '无效': invalid_symbol_count
                                })
                        else:
                            # invalid_symbol 不纳入重试
                            if error_type != "invalid_symbol":
                                batch_failed[trading_pair] = date_info
                                print(f"⚠️  {trading_pair} 获取失败: {error_type}")
                            else:
                                pass  # 静默跳过无效交易对
                        
                    except Exception as e:
                        with stats_lock:
                            error_count += 1
                        batch_failed[pair] = date_info
                        print(f"⚠️  处理 {pair} 时出错: {e}")
                    
                    pbar.update(1)
        
        return batch_results, batch_failed
    
    # 第一轮获取
    results, failed_pairs = run_fetch_batch(incremental_trading_pairs, "获取增量K线数据")
    
    # 对失败的交易对进行重试（最多重试2轮）
    max_retry_rounds = 2
    for retry_round in range(max_retry_rounds):
        if not failed_pairs:
            break
        # 重试前等待，避免 API 限流
        time.sleep(5)
        print(f"\n🔄 重试第 {retry_round + 1} 轮: {len(failed_pairs)} 个失败交易对...")
        retry_results, failed_pairs = run_fetch_batch(failed_pairs, f"重试第{retry_round + 1}轮")
        results.extend(retry_results)
        if failed_pairs:
            print(f"⚠️  仍有 {len(failed_pairs)} 个交易对失败: {list(failed_pairs.keys())}")
    
    if failed_pairs:
        print(f"\n❌ 最终仍有 {len(failed_pairs)} 个交易对无法获取: {list(failed_pairs.keys())}")
    
    # 合并所有结果
    print(f"\n📊 合并 {len(results)} 个成功获取的数据...")
    for trading_pair, df in results:
        new_kline_data = pd.concat([new_kline_data, df], ignore_index=True)
    
    # 打印成功获取的交易对（每10个一行）
    # 使用实际结果计数（重试后成功的不应重复计入 error_count）
    actual_success_count = len(results)
    actual_fail_count = len(failed_pairs)
    if actual_success_count > 0:
        successful_pairs = sorted([pair for pair, _ in results])
        print(f"\n✅ 成功获取 {actual_success_count} 个交易对:")
        for i, pair in enumerate(successful_pairs):
            if i % 10 == 0:
                print()
            print(f"{pair:<12}", end="")
        print()
    
    # 保存所有数据
    try:
        print(f"\n💾 正在保存本次新增 {len(new_kline_data)} 条K线数据...")
        all_kline_data = merge_kline_data(existing_kline_data, new_kline_data)
        appended_rows = max(len(all_kline_data) - existing_rows, 0)
        
        # 保存到CSV
        all_kline_data.to_csv(final_path, index=False)
        print(f"✅ 所有K线数据已保存至: {final_path}")
        write_kline_metadata(final_path)
        
        # 打印数据预览
        print("\n📊 数据预览:")
        print(all_kline_data[['symbol', 'base_asset', 'date', 'open', 'high', 'low', 'close', 'volume']].head(10))
        
        print(f"\n📈 数据统计:")
        print(f"   成功获取: {actual_success_count} 个交易对")
        print(f"   失败: {actual_fail_count} 个交易对")
        print(f"   无效交易对: {invalid_symbol_count} 个")
        print(f"   本次净新增记录: {appended_rows}")
        print(f"   总记录数: {len(all_kline_data)}")
        print(f"   独特交易对: {all_kline_data['symbol'].nunique()}")
        if len(all_kline_data) > 0:
            print(f"   时间跨度: {all_kline_data['date'].min()} 到 {all_kline_data['date'].max()}")
        
        # 验证未来函数修复
        print(f"\n🔍 未来函数检查:")
        for pair in all_kline_data['symbol'].unique()[:5]:
            pair_data = all_kline_data[all_kline_data['symbol'] == pair]
            first_data_date = pair_data['date'].min()
            last_data_date = pair_data['date'].max()
            if pair in valid_trading_pairs:
                date_info = valid_trading_pairs[pair]
                first_appearance = date_info['first_date'].strftime('%Y-%m-%d')
                last_appearance = date_info['last_date'].strftime('%Y-%m-%d')
                print(f"   {pair}: CSV范围 {first_appearance} 至 {last_appearance}, "
                      f"K线数据 {first_data_date} 至 {last_data_date}")
        
    except Exception as e:
        print(f"❌ 保存数据时出错: {e}")


def analyze_trading_pairs():
    """
    分析CSV中的交易对与时间的关系
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "binance_coingecko_top100_marketcap_historical.csv")
    
    if not os.path.exists(csv_path):
        print("❌ CSV文件不存在")
        return
    
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    
    if 'Trading_Pairs' not in df.columns:
        print("❌ CSV文件中缺少Trading_Pairs列")
        return
    
    # 展开Trading_Pairs列
    expanded_rows = []
    for idx, row in df.iterrows():
        trading_pairs_str = str(row['Trading_Pairs']) if pd.notna(row['Trading_Pairs']) else ''
        trading_pairs = [tp.strip() for tp in trading_pairs_str.split(',') if tp.strip()]
        
        for tp in trading_pairs:
            expanded_row = row.copy()
            expanded_row['Trading_Pair'] = tp
            expanded_rows.append(expanded_row)
    
    if not expanded_rows:
        print("❌ 没有找到有效的交易对数据")
        return
    
    expanded_df = pd.DataFrame(expanded_rows)
    
    # 分析token首次出现时间
    token_first_appearance = expanded_df.groupby('Trading_Pair')['Date'].min()
    token_last_appearance = expanded_df.groupby('Trading_Pair')['Date'].max()
    
    print(f"\n📊 交易对时间分析:")
    print(f"   CSV中独特交易对数量: {len(token_first_appearance)}")
    print(f"   数据时间跨度: {df['Date'].min().strftime('%Y-%m-%d')} 到 {df['Date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\n📈 token首次出现时间分布:")
    appearance_counts = token_first_appearance.dt.strftime('%Y-%m').value_counts().sort_index()
    for date_month, count in appearance_counts.head(10).items():
        print(f"   {date_month}: {count} 个新token")


if __name__ == "__main__":
    print("🚀 基于Binance历史数据获取币安USDT合约K线数据（无未来函数版本）")
    print("=" * 80)
    
    # 先分析交易对时间分布
    analyze_trading_pairs()
    
    print("\n" + "=" * 80)
    # 获取历史K线数据（使用并发加速）
    fetch_historical_klines_from_coingecko_csv(max_workers=5)  # 可根据网络情况调整（建议5-20）
    
    print("\n🎉 数据获取完成！")
