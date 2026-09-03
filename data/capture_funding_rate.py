import pandas as pd
import os
import time
import requests
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv


# Default symbols list (Binance USDT-M perpetual top tokens)
DEFAULT_SYMBOLS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT",
    "MATIC", "TRX", "LTC", "BCH", "UNI", "ATOM", "XLM", "NEAR", "FIL", "APT",
    "ARB", "OP", "INJ", "SUI", "SEI", "TIA", "PYTH", "JTO", "WLD", "ORDI",
    "FTM", "ICP", "RUNE", "AAVE", "MKR", "SAND", "MANA", "AXS", "GALA", "FLOW",
    "IMX", "LDO", "GRT", "ALGO", "THETA", "EGLD", "FTT", "HBAR", "VET", "ETC"
]

BINANCE_FUNDING_RATE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def load_symbols(symbols_file: str) -> list[str]:
    """
    Load symbols from CSV file or fallback to default list.
    
    If symbols_file is provided and exists, read the first column.
    Otherwise, return the default hardcoded list.
    """
    if symbols_file and os.path.exists(symbols_file):
        try:
            df = pd.read_csv(symbols_file)
            # Assume first column contains symbol names
            symbols = df.iloc[:, 0].str.replace("USDT", "", regex=False).unique().tolist()
            return symbols
        except Exception as e:
            print(f"⚠️  Failed to load symbols from {symbols_file}: {e}. Using default list.")
            return DEFAULT_SYMBOLS
    return DEFAULT_SYMBOLS


def fetch_symbol(
    symbol: str,
    start_ms: int,
    end_ms: int | None = None,
    pause: float = 0.25,
    out_dir: str = "data/funding_rate_data"
) -> tuple[int, str, str]:
    """
    Fetch funding rates for a symbol from Binance USDT-M perpetuals API.
    
    Args:
        symbol: Coin symbol without USDT suffix (e.g., "BTC")
        start_ms: Start time in milliseconds (Unix epoch)
        end_ms: End time in milliseconds; if None, use current time
        pause: Sleep duration (seconds) between API requests
        out_dir: Output directory for CSV files
    
    Returns:
        Tuple of (rows_fetched, min_date, max_date)
    """
    symbol_full = f"{symbol}USDT"
    csv_path = os.path.join(out_dir, f"{symbol_full}_funding.csv")
    
    os.makedirs(out_dir, exist_ok=True)
    
    if end_ms is None:
        end_ms = int(datetime.utcnow().timestamp() * 1000)
    
    all_rows = []
    current_start = start_ms
    total_rows = 0
    
    while current_start < end_ms:
        # Request up to 1000 records
        params = {
            "symbol": symbol_full,
            "limit": 1000,
            "startTime": current_start,
            "endTime": end_ms
        }
        
        try:
            time.sleep(pause)
            response = requests.get(BINANCE_FUNDING_RATE_URL, params=params, timeout=10)
            
            if response.status_code != 200:
                # Retry once on error
                print(f"⚠️  HTTP {response.status_code} for {symbol_full}. Retrying...")
                time.sleep(5)
                response = requests.get(BINANCE_FUNDING_RATE_URL, params=params, timeout=10)
                
                if response.status_code != 200:
                    print(f"❌ Failed to fetch {symbol_full}: HTTP {response.status_code}")
                    return (total_rows, "", "")
            
            data = response.json()
            
            if not data or len(data) == 0:
                break
            
            # Parse and convert funding rates
            for record in data:
                funding_time_ms = int(record["fundingTime"])
                funding_rate = float(record["fundingRate"])
                
                # Convert to ISO UTC datetime string
                dt_utc = datetime.fromtimestamp(funding_time_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)
                funding_time_iso = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
                
                all_rows.append({
                    "symbol": symbol_full,
                    "fundingTime": funding_time_iso,
                    "fundingRate": funding_rate,
                    "fundingTimeMs": funding_time_ms
                })
            
            total_rows += len(data)
            
            # If we got fewer than 1000 records, we've reached the end
            if len(data) < 1000:
                break
            
            # Advance to next batch: last fundingTime + 1ms
            last_funding_time = int(data[-1]["fundingTime"]) + 1
            current_start = last_funding_time
            
        except requests.RequestException as e:
            print(f"❌ Network error fetching {symbol_full}: {e}")
            return (total_rows, "", "")
        except (ValueError, KeyError) as e:
            print(f"❌ Parse error for {symbol_full}: {e}")
            return (total_rows, "", "")
    
    if not all_rows:
        return (0, "", "")
    
    # Write to CSV
    df = pd.DataFrame(all_rows)
    
    # Check if file exists and has data
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, df], ignore_index=True)
        # Remove duplicates (if any) based on fundingTimeMs
        df = df.drop_duplicates(subset=["fundingTimeMs"], keep="last")
        df = df.sort_values("fundingTimeMs").reset_index(drop=True)
    
    df.to_csv(csv_path, index=False)
    
    # Get date range
    df["fundingTime"] = pd.to_datetime(df["fundingTime"])
    min_date = df["fundingTime"].min().strftime("%Y-%m-%d %H:%M:%S")
    max_date = df["fundingTime"].max().strftime("%Y-%m-%d %H:%M:%S")
    
    return (len(all_rows), min_date, max_date)


def get_last_funding_time_ms(csv_path: str) -> int | None:
    """
    Read the last fundingTimeMs from an existing CSV file.
    
    Returns:
        Last fundingTimeMs or None if file doesn't exist/is empty
    """
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path, usecols=["fundingTimeMs"])
        if len(df) > 0:
            return int(df["fundingTimeMs"].iloc[-1])
    except Exception as e:
        print(f"⚠️  Failed to read last fundingTimeMs from {csv_path}: {e}")
    
    return None


def date_to_ms(date_str: str) -> int:
    """Convert YYYY-MM-DD format to milliseconds since epoch (UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Binance USDT-M perpetual funding rates"
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=None,
        help="Path to symbols CSV (default: data/binance_top50_symbols.csv if exists)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD, default: 2024-01-01)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/funding_rate_data",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.25,
        help="Sleep duration between requests (seconds)"
    )
    
    args = parser.parse_args()
    
    # Determine symbols file
    symbols_file = args.symbols_file
    if not symbols_file:
        if os.path.exists("data/binance_top50_symbols.csv"):
            symbols_file = "data/binance_top50_symbols.csv"
    
    symbols = load_symbols(symbols_file)
    
    # Parse dates
    start_ms = date_to_ms(args.start)
    
    if args.end:
        end_ms = date_to_ms(args.end) + 86400 * 1000  # End of day
    else:
        end_ms = int(datetime.utcnow().timestamp() * 1000)
    
    print(f"📥 Fetching funding rates for {len(symbols)} symbols")
    print(f"   Period: {args.start} to {args.end or 'today'}")
    print(f"   Output: {args.out_dir}")
    print()
    
    summary = []
    
    for symbol in symbols:
        csv_path = os.path.join(args.out_dir, f"{symbol}USDT_funding.csv")
        
        # Check if incremental update is possible
        effective_start_ms = start_ms
        last_funding_ms = get_last_funding_time_ms(csv_path)
        
        if last_funding_ms is not None:
            # Start from last record + 1ms
            effective_start_ms = last_funding_ms + 1
        
        rows_fetched, min_date, max_date = fetch_symbol(
            symbol,
            effective_start_ms,
            end_ms,
            pause=args.pause,
            out_dir=args.out_dir
        )
        
        if rows_fetched > 0:
            summary.append(f"✅ {symbol:8s}: {rows_fetched:4d} rows  ({min_date} ~ {max_date})")
        else:
            summary.append(f"⏭️  {symbol:8s}: no new data")
    
    print("\n📊 Summary:")
    for line in summary:
        print(f"   {line}")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
