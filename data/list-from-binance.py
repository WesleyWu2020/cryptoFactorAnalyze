#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过Binance API获取历史流动性排名

特性：
- 重试机制：自动重试失败的请求
- 冷却机制：避免API速率限制
- 每月获取一次：固定月频决策，便于回放
- 增量更新：自动检测最新日期，只获取新数据
- 每若干个月保存一次：避免数据丢失
- 基于流动性排名：使用Binance的历史K线数据计算成交额

⚠️ 未来函数修复：
- 在每月第1天（如5月1日）做决策时，聚合的是上一个自然月（如4月1日到4月30日）总交易量
- 这样在5月1日做决策时，使用的是上月已收盘数据，避免未来函数
- CSV中的Date字段标记为决策日期（如5月1日），而聚合窗口在字段中显式记录
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, date
import time
import random
import os
import json
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from pipeline_time_utils import add_months, previous_month_window

# API配置
BINANCE_API = "https://api.binance.com/api/v3"

# 稳定币报价货币
STABLE_QUOTES = ("USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI", "USDP")

# 稳定币列表（需要排除的）
STABLECOINS = {
    'usdt', 'usdc', 'busd', 'dai', 'ust', 'tusd', 'usdp', 'usdn', 'fei', 'frax',
    'lusd', 'alusd', 'gusd', 'ousd', 'susd', 'musd', 'rsv', 'husd', 'cusdc', 'cdai',
    'usde', 'usds', 'usdd', 'mim', 'ustc', 'pax', 'paxg', 'usdr', 'usdb', 'usdx',
    'vai', 'dola', 'gho', 'pyusd','bfusd','usd1','BFUSD','USDT0','C1USD','BUIDL','buidl','USDF','usdf','FBTC','fbtc',
    'USTB','ustb','FDUSD','fdusd','USDY','usdy','USDAI','usdai','FDUSDUSDT','fdusdusdt',
    'bsc-usd', 'usd0', 'usdtb', 'usdg', 'rlusd', 'usdc.e', 'solvbtc', 'lbtc',
    'susds', 'susde', 'usyc', 'syrupusdc', 'syrupusdt', 'steakusdc', 'eurc', 'cusd', 'crvusd', 'usr',
    'xusd',
    'cbbtc', 'clbtc', 'tbtc', 'sbtc', 'btc.b', 'jitosol', 'bnsol', 'msol', 'jupsol'
}

# Wrapped tokens（需要排除的）
WRAPPED_TOKENS = {
    'wbtc', 'weth', 'wusdt', 'wusdc', 'wbusd', 'wdai', 'wtusd',
    'wbnb', 'wmatic', 'wavax', 'wftm', 'wone', 'wsol'
}

# 平台币（需要排除的）
EXCHANGE_TOKENS = {
    'okb', 'gt', 'bgb', 'xt', 'cro', 'ftt', 'ht', 'kcs', 'leo'
}

# 法币类 base asset（需要排除）
# 重点排除：U/EUR（用户指定）+ 常见法币代码，防止进入Binance Top池
FIAT_BASE_ASSETS = {
    "U", "EUR", "TRY", "RUB", "BRL", "UAH", "BIDR", "IDRT", "NGN", "PLN", "RON",
    "JPY", "GBP", "AUD", "CAD", "CHF", "NZD", "HKD", "SGD", "ZAR", "MXN",
    "ARS", "CLP", "COP", "PEN", "KES", "GHS", "MAD", "EGP", "SAR", "AED",
    "QAR", "KWD", "BHD", "OMR", "JOD", "THB", "VND", "MYR", "IDR", "PHP", "TWD"
}


def is_stablecoin(symbol: str) -> bool:
    """
    判断是否为稳定币
    
    Args:
        symbol: 币种符号
        
    Returns:
        bool: 是否为稳定币
    """
    symbol_lower = symbol.lower()
    
    # 检查是否在稳定币列表中
    if symbol_lower in STABLECOINS:
        return True
    
    # 检查是否为wrapped token
    if symbol_lower in WRAPPED_TOKENS:
        return True
    
    # 检查是否为平台币
    if symbol_lower in EXCHANGE_TOKENS:
        return True
    
    # 检查W开头的短符号（可能是wrapped token）
    if symbol_lower.startswith('w') and len(symbol_lower) <= 6:
        # 排除一些正常的币种
        if symbol_lower not in ['wld', 'wif', 'woo', 'win', 'waves', 'wax']:
            return True
    
    return False


def is_excluded_base_asset(base_asset: str) -> bool:
    """
    统一判断是否应排除该 base 资产（稳定币/包装币/平台币/法币）
    """
    if is_stablecoin(base_asset):
        return True
    return str(base_asset).upper() in FIAT_BASE_ASSETS


def get_latest_date_from_csv(csv_file: str) -> Optional[date]:
    """从现有CSV文件中获取最新日期"""
    if not os.path.exists(csv_file):
        return None
    
    try:
        df = pd.read_csv(csv_file)
        if 'Date' in df.columns and len(df) > 0:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            df = df.dropna(subset=['Date'])
            if len(df) > 0:
                latest_date = df['Date'].max()
                return latest_date.date()
    except Exception as e:
        print(f"❌ 读取CSV文件时出错: {e}")
    
    return None


def write_dataset_metadata(csv_file: str, source: str, schema_version: str = "v1") -> None:
    """写入最小元信息，便于数据回放与问题定位。"""
    if not os.path.exists(csv_file):
        return
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            return
        date_series = pd.to_datetime(df.get("Date"), format="mixed", errors="coerce")
        meta = {
            "dataset": os.path.basename(csv_file),
            "schema_version": schema_version,
            "source": source,
            "row_count": int(len(df)),
            "date_min": str(date_series.min().date()) if date_series.notna().any() else None,
            "date_max": str(date_series.max().date()) if date_series.notna().any() else None,
            "updated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "columns": list(df.columns),
        }
        with open(f"{csv_file}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️  写入元信息失败: {e}")


class BinanceScraper:
    """Binance 数据爬虫（基于流动性排名）"""

    def __init__(self, request_delay: float = 0.1, max_retries: int = 3, max_workers: int = 20, only_usdt: bool = True):
        """
        初始化爬虫

        Args:
            request_delay: 请求间隔（秒），默认0.1秒以避免速率限制（并发模式下不使用）
            max_retries: 最大重试次数
            max_workers: 并发线程数，默认20
            only_usdt: 是否只查询USDT交易对，默认True（可大幅减少请求数量）
        """
        self.binance_api = BINANCE_API
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.only_usdt = only_usdt
        self.session = self._create_session()
        self.lock = Lock()  # 用于线程安全的计数器

    def _create_session(self):
        """创建带重试机制的requests session"""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        session.headers.update(headers)
        return session

    def safe_request(self, url: str, params: Dict = None, max_retries: int = 3) -> Optional[requests.Response]:
        """安全的请求函数，带重试和错误处理"""
        for attempt in range(max_retries):
            try:
                r = self.session.get(url, params=params, timeout=30)
                if r.status_code == 200:
                    return r
                elif r.status_code == 429:  # 请求过多
                    wait_time = 2 ** attempt
                    print(f"   ⚠️  速率限制，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"   ⚠️  HTTP {r.status_code}，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        print(f"   ❌ HTTP {r.status_code}，重试失败")
                        return None
            except (requests.exceptions.SSLError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"   ⚠️  网络错误 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__}，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ 网络错误，重试失败: {type(e).__name__}")
                    return None
        return None

    def get_binance_symbols(self) -> List[Dict]:
        """从Binance获取所有交易对列表"""
        print(f"   📋 从Binance获取交易对列表...")
        url = f"{self.binance_api}/exchangeInfo"
        r = self.safe_request(url)
        
        if r is None:
            print(f"   ❌ 无法获取Binance交易对列表")
            return []
        
        try:
            data = r.json()
            symbols = data.get("symbols", [])
            
            # 提取以稳定币为报价货币的交易对
            valid_symbols = []
            quotes_to_use = ["USDT"] if self.only_usdt else STABLE_QUOTES
            
            for s in symbols:
                symbol = s.get("symbol", "")
                quote = s.get("quoteAsset", "")
                
                if quote in quotes_to_use and s.get("status") == "TRADING":
                    base = s.get("baseAsset", "")
                    if not is_excluded_base_asset(base):
                        valid_symbols.append({
                            "symbol": symbol,
                            "base": base,
                            "quote": quote
                        })
            
            quote_type = "USDT" if self.only_usdt else "稳定币"
            print(f"   ✅ 获取到 {len(valid_symbols)} 个{quote_type}交易对")
            return valid_symbols
        except Exception as e:
            print(f"   ❌ 解析Binance数据失败: {e}")
            return []

    def check_symbol_exists_in_range(self, symbol: str, start_date: date, end_date: date) -> bool:
        """
        快速检查交易对在日期范围内是否存在数据
        直接查询日期范围内的数据，如果存在则返回True
        
        Args:
            symbol: 交易对符号
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            bool: 如果存在数据返回True
        """
        # 查询日期范围内的K线数据
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        start_time = int(start_datetime.timestamp() * 1000)
        end_time = int(end_datetime.timestamp() * 1000)
        
        url = f"{self.binance_api}/klines"
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1
        }
        
        r = self.safe_request(url, params=params)
        if r is None:
            return False
        
        try:
            klines = r.json()
            # 如果有数据，说明该交易对在日期范围内存在
            return klines is not None and len(klines) > 0
        except (ValueError, IndexError, KeyError):
            return False

    def get_historical_klines(self, symbol: str, start_date: date, end_date: date) -> Optional[Dict]:
        """
        获取指定日期范围内的K线数据并聚合
        
        Args:
            symbol: 交易对符号（如 BTCUSDT）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict: 包含价格和交易量的字典，如果失败返回None
        """
        # 计算时间戳（毫秒）
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        start_time = int(start_datetime.timestamp() * 1000)
        end_time = int(end_datetime.timestamp() * 1000)
        
        # Binance API限制每次最多返回1000条，需要分页获取
        all_klines = []
        current_start = start_time
        
        while current_start < end_time:
            url = f"{self.binance_api}/klines"
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000  # 最大限制
            }
            
            r = self.safe_request(url, params=params)
            if r is None:
                break
            
            try:
                klines = r.json()
                if not klines or len(klines) == 0:
                    break
                
                all_klines.extend(klines)
                
                # 如果返回的数据少于1000条，说明已经获取完所有数据
                if len(klines) < 1000:
                    break
                
                # 更新起始时间为最后一条K线的收盘时间+1
                last_kline_time = klines[-1][6]  # 收盘时间（毫秒）
                current_start = last_kline_time + 1
                
            except (ValueError, IndexError, KeyError):
                break
        
        if not all_klines:
            return None
        
        try:
            # 聚合所有K线的交易量
            total_base_volume = 0.0
            total_quote_volume = 0.0
            last_close_price = 0.0
            
            # K线数据格式：[开盘时间, 开盘价, 最高价, 最低价, 收盘价, 成交量, 收盘时间, 成交额, 成交笔数, ...]
            for k in all_klines:
                base_volume = float(k[5])   # base volume (基础资产成交量)
                close_price = float(k[4])   # close price (收盘价)
                quote_volume = float(k[7]) if len(k) > 7 else base_volume * close_price  # 成交额（如果可用）
                
                total_base_volume += base_volume
                total_quote_volume += quote_volume
                last_close_price = close_price  # 使用最后一天的收盘价作为价格
            
            return {
                "price": last_close_price,
                "base_volume": total_base_volume,
                "quote_volume": total_quote_volume
            }
        except (ValueError, IndexError, KeyError) as e:
            return None

    def _check_exists_worker(self, symbol_info: Dict, start_date: date, end_date: date) -> Optional[Dict]:
        """工作线程函数：快速检查交易对在日期范围内是否存在"""
        symbol = symbol_info["symbol"]
        if self.check_symbol_exists_in_range(symbol, start_date, end_date):
            return symbol_info
        return None

    def _fetch_kline_worker(self, symbol_info: Dict, start_date: date, end_date: date) -> Optional[Dict]:
        """工作线程函数：获取单个交易对的K线数据（聚合日期范围内的数据）"""
        symbol = symbol_info["symbol"]
        base_symbol = symbol_info["base"]
        
        kline_data = self.get_historical_klines(symbol, start_date, end_date)
        
        if kline_data is None:
            return None
        
        return {
            "symbol": symbol,
            "base": base_symbol,
            "quote_volume": kline_data["quote_volume"],
            "price": kline_data["price"]
        }

    def get_historical_market_data(self, start_date: date, end_date: date, limit: int = 100) -> Optional[List[Dict]]:
        """
        获取指定日期范围内的市场数据（基于流动性排名，聚合日期范围内的总成交额，使用并发加速）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            limit: 获取的排名数量
            
        Returns:
            List[Dict]: 币种数据列表，如果失败返回None
        """
        # 第一步：从Binance获取交易对列表
        binance_symbols = self.get_binance_symbols()
        if not binance_symbols:
            return None
        
        total_symbols = len(binance_symbols)
        print(f"   📋 第一步：快速检查哪些交易对在 {start_date} 到 {end_date} 范围内存在数据...")
        
        # 第二步：快速检查哪些交易对在日期范围内存在（两阶段查询）
        valid_symbols = []
        checked_count = 0
        start_time = time.time()
        
        # 第一阶段：快速检查存在性
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._check_exists_worker, symbol_info, start_date, end_date): symbol_info
                for symbol_info in binance_symbols
            }
            
            for future in as_completed(future_to_symbol):
                checked_count += 1
                if checked_count % 100 == 0 or checked_count == total_symbols:
                    elapsed = time.time() - start_time
                    speed = checked_count / elapsed if elapsed > 0 else 0
                    print(f"   ⏳ 检查进度: {checked_count}/{total_symbols} ({checked_count*100//total_symbols}%), "
                          f"找到 {len(valid_symbols)} 个有效交易对, 速度: {speed:.1f} 请求/秒")
                
                try:
                    result = future.result()
                    if result is not None:
                        valid_symbols.append(result)
                except Exception:
                    pass
        
        if not valid_symbols:
            print(f"   ❌ 在 {start_date} 到 {end_date} 范围内没有找到任何有效的交易对数据")
            return None
        
        print(f"   ✅ 找到 {len(valid_symbols)} 个在日期范围内存在的交易对（从 {total_symbols} 个中筛选）")
        print(f"   📊 第二步：开始获取这 {len(valid_symbols)} 个交易对的详细数据（聚合 {start_date} 到 {end_date} 的交易量，并发数: {self.max_workers}）...")
        
        # 第三步：使用线程池并发获取K线数据（只查询有效的交易对）
        base_symbol_data = {}  # {base: {quote_volume_sum, price, symbol, name}}
        processed_count = 0
        failed_count = 0
        success_count = 0
        
        start_time = time.time()
        
        # 使用线程池并发请求（只查询有效的交易对）
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务（只查询有效的交易对）
            future_to_symbol = {
                executor.submit(self._fetch_kline_worker, symbol_info, start_date, end_date): symbol_info
                for symbol_info in valid_symbols
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_symbol):
                processed_count += 1
                symbol_info = future_to_symbol[future]
                
                # 显示进度（每50个显示一次）
                if processed_count % 50 == 0 or processed_count == len(valid_symbols):
                    elapsed = time.time() - start_time
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    print(f"   ⏳ 进度: {processed_count}/{total_symbols} ({processed_count*100//total_symbols}%), "
                          f"成功: {success_count}, 失败: {failed_count}, 速度: {speed:.1f} 请求/秒")
                
                try:
                    result = future.result()
                    if result is None:
                        failed_count += 1
                    else:
                        base_symbol = result["base"]
                        quote_volume = result["quote_volume"]
                        price = result["price"]
                        symbol = result["symbol"]
                        
                        # 聚合同一币种不同交易对的交易量
                        if base_symbol not in base_symbol_data:
                            base_symbol_data[base_symbol] = {
                                "symbol": base_symbol.upper(),
                                "name": base_symbol.upper(),
                                "quote_volume": 0,
                                "price": price,
                                "trading_pairs": []
                            }
                        
                        base_symbol_data[base_symbol]["quote_volume"] += quote_volume
                        base_symbol_data[base_symbol]["trading_pairs"].append(symbol)
                        if price > 0:
                            base_symbol_data[base_symbol]["price"] = price
                        
                        success_count += 1
                except Exception as e:
                    failed_count += 1
        
        # 转换为列表并按交易量排序
        coins_data = []
        for base, data in base_symbol_data.items():
            if data["quote_volume"] > 0:
                coins_data.append({
                    "symbol": data["symbol"],
                    "name": data["name"],
                    "quote_volume": data["quote_volume"],
                    "price": data["price"],
                    "trading_pairs": ",".join(data["trading_pairs"])
                })
        
        if not coins_data:
            print(f"   ❌ 未获取到有效数据（成功: {processed_count - failed_count}, 失败: {failed_count}）")
            return None
        
        # 按交易量排序并取前limit个
        coins_data_sorted = sorted(coins_data, key=lambda x: x["quote_volume"], reverse=True)
        top_coins = coins_data_sorted[:limit]
        
        # 附加排名
        for idx, item in enumerate(top_coins, 1):
            item["rank"] = idx
            # 明确标记为流动性分数，避免误认为真实市值
            item["liquidity_score"] = item["quote_volume"]
        
        print(f"   📊 处理结果: 处理{processed_count}个交易对, 成功{processed_count - failed_count}个, 有效币种{len(coins_data)}个, 前{limit}名")
        return top_coins

    def scrape_historical_data(
        self,
        start_date: date,
        end_date: date,
        output_file: str,
        month_interval: int = 1,
        limit: int = 100
    ):
        """
        获取指定时间范围内的历史数据（月频）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            output_file: 输出文件路径
            month_interval: 月份间隔（默认1个月）
            limit: 排名数量
        """
        print("=" * 80)
        print("📊 Binance 历史交易量排名数据获取")
        print("=" * 80)
        print(f"📅 时间范围: {start_date} 到 {end_date}")
        print(f"📈 排名数量: 前{limit}名（排除稳定币）")
        print(f"⏱️  月份间隔: {month_interval}个月（月频）")
        print(f"⏳ 请求延迟: {self.request_delay}秒")
        print("=" * 80)
        
        # 检查现有数据
        latest_existing_date = get_latest_date_from_csv(output_file)
        if latest_existing_date:
            print(f"📖 发现现有数据，最新日期: {latest_existing_date}")
            # 从最新决策日的下一期开始继续
            next_decision = add_months(latest_existing_date.replace(day=1), month_interval)
            if next_decision >= start_date:
                start_date = next_decision
                print(f"🔄 从 {start_date} 开始增量更新（月频）")
        
        if start_date > end_date:
            print("✅ 数据已是最新，无需更新")
            return
        
        all_data = []
        current_date = start_date.replace(day=1)
        success_count = 0
        fail_count = 0
        month_count = 0
        save_interval = 6  # 每6个月保存一次
        
        # 计算总月份数
        total_months = 0
        temp_date = current_date
        while temp_date <= end_date:
            total_months += 1
            temp_date = add_months(temp_date, month_interval)
        
        month_num = 0
        
        while current_date <= end_date:
            month_num += 1
            month_count += 1
            
            # 修复未来函数：月初决策使用上月完整窗口
            window_start_date, window_end_date = previous_month_window(current_date)

            date_str = current_date.strftime('%Y-%m-%d')
            window_start_str = window_start_date.strftime('%Y-%m-%d')
            window_end_str = window_end_date.strftime('%Y-%m-%d')

            print(f"\n[{month_num}/{total_months}] 📅 {date_str} (决策日期) -> 聚合 {window_start_str} 到 {window_end_str} 的数据（上月总交易量）")
            
            # 获取上一个自然月的总交易量数据
            coins = self.get_historical_market_data(window_start_date, window_end_date, limit=limit)
            
            if coins:
                for coin in coins:
                    all_data.append({
                        'Date': date_str,
                        'Rank': coin['rank'],
                        'Symbol': coin['symbol'],
                        'Name': coin['name'],
                        'Price_USD': coin['price'],
                        # 兼容旧字段名，保持下游不崩
                        'Market_Cap_USD': coin['liquidity_score'],
                        'Liquidity_Score_USD': coin['liquidity_score'],
                        'Quote_Volume_USD': coin['quote_volume'],
                        'Trading_Pairs': coin.get('trading_pairs', ''),
                        'Decision_Window_Start': window_start_str,
                        'Decision_Window_End': window_end_str,
                    })
                
                success_count += 1
                print(f"   ✅ 成功获取 {len(coins)} 个币种")
                
                # 每N个月保存一次
                if month_count >= save_interval:
                    print(f"   💾 保存数据（已处理 {month_count} 个月，总计 {len(all_data)} 条记录）...")
                    save_data_to_csv(all_data, output_file)
                    all_data = []
                    month_count = 0
            else:
                fail_count += 1
                print(f"   ❌ 获取失败")
            
            # 移动到下N个月
            current_date = add_months(current_date, month_interval)
            
            # 冷却延迟
            if current_date <= end_date:
                delay = max(0.0, self.request_delay + random.uniform(-0.2, 0.2))  # 确保延迟不为负数
                if delay > 0:
                    print(f"   ⏳ 冷却 {delay:.1f} 秒...")
                    time.sleep(delay)
        
        # 保存剩余数据
        if all_data:
            print(f"   💾 保存剩余数据（{month_count} 个月，总计 {len(all_data)} 条记录）...")
            save_data_to_csv(all_data, output_file)
        
        # 统计信息
        print("\n" + "=" * 80)
        print("📊 获取完成统计")
        print("=" * 80)
        print(f"✅ 成功: {success_count}/{total_months} 个月")
        print(f"❌ 失败: {fail_count}/{total_months} 个月")
        
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            print(f"📈 总记录数: {len(df)} 条")
            print(f"🪙 涉及币种数: {df['Symbol'].nunique()} 个")
            print(f"📅 日期范围: {df['Date'].min()} 到 {df['Date'].max()}")
            write_dataset_metadata(output_file, source="binance_api_monthly_liquidity")


def save_data_to_csv(data: List[Dict], output_file: str):
    """保存数据到CSV文件（增量更新）"""
    file_exists = os.path.exists(output_file)
    
    existing_data = []
    if file_exists:
        try:
            existing_df = pd.read_csv(output_file)
            existing_data = existing_df.to_dict('records')
        except Exception as e:
            print(f"⚠️  读取现有数据时出错: {e}")
            existing_data = []
    
    all_data = existing_data + data
    
    if all_data:
        df = pd.DataFrame(all_data)
        # 兼容历史字段：统一补齐 liquidity 字段
        if "Liquidity_Score_USD" not in df.columns and "Market_Cap_USD" in df.columns:
            df["Liquidity_Score_USD"] = df["Market_Cap_USD"]
        if "Market_Cap_USD" not in df.columns and "Liquidity_Score_USD" in df.columns:
            df["Market_Cap_USD"] = df["Liquidity_Score_USD"]
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values(['Date', 'Rank']).reset_index(drop=True)
        df = df.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
        df = df.sort_values(['Date', 'Rank']).reset_index(drop=True)
        df.to_csv(output_file, index=False)
        print(f"✅ 数据已保存: {len(df)} 条记录 (新增 {len(data)} 条)")
        write_dataset_metadata(output_file, source="binance_api_monthly_liquidity")


def main():
    """主函数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "binance_coingecko_top100_marketcap_historical.csv")
    
    target_start_date = date(2021, 1, 1)
    end_date = date.today() - timedelta(days=1)
    
    scraper = BinanceScraper(
        request_delay=0.1,  # 并发模式下不使用
        max_retries=3,
        max_workers=20,     # 并发线程数，可根据网络情况调整（建议10-30）
        only_usdt=True      # 只查询USDT交易对，可大幅减少请求数量
    )
    
    scraper.scrape_historical_data(
        start_date=target_start_date,
        end_date=end_date,
        output_file=output_file,
        month_interval=1,
        limit=100
    )
    
    print("\n🎉 程序执行完成！")


if __name__ == "__main__":
    main()
