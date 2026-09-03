import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from data import capture_data_binance as cdb


class CaptureDataBinanceIncrementalTest(unittest.TestCase):
    def test_resolve_incremental_output_path_prefers_latest_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            old_file = td_path / "binance_daily_klines_20260326.csv"
            latest_file = td_path / "binance_daily_klines_20260327.csv"
            old_file.write_text("date,symbol\n", encoding="utf-8")
            latest_file.write_text("date,symbol\n", encoding="utf-8")
            (td_path / "binance_daily_klines_20260328.csv.meta.json").write_text("{}", encoding="utf-8")

            resolved = cdb.resolve_incremental_output_path(str(td_path))
            self.assertEqual(resolved, str(latest_file))

    def test_resolve_incremental_output_path_creates_today_name_when_empty(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            resolved = cdb.resolve_incremental_output_path(str(td_path), now=datetime(2026, 3, 30))
            self.assertEqual(resolved, str(td_path / "binance_daily_klines_20260330.csv"))

    def test_build_fetch_plan_uses_existing_max_date_per_symbol(self):
        valid_pairs = {
            "BTCUSDT": {
                "first_date": pd.Timestamp("2021-01-01"),
                "last_date": pd.Timestamp("2026-03-29"),
                "appearance_dates": set(),
            },
            "NEWUSDT": {
                "first_date": pd.Timestamp("2024-01-01"),
                "last_date": pd.Timestamp("2026-03-29"),
                "appearance_dates": set(),
            },
        }
        existing_df = pd.DataFrame(
            {
                "symbol": ["BTCUSDT", "BTCUSDT"],
                "date": ["2026-03-28", "2026-03-29"],
            }
        )

        plan = cdb.build_incremental_fetch_plan(valid_pairs, existing_df, now=datetime(2026, 3, 30))
        self.assertEqual(plan["BTCUSDT"]["fetch_start_date"], pd.Timestamp("2026-03-30"))
        self.assertEqual(plan["NEWUSDT"]["fetch_start_date"], pd.Timestamp("2023-10-03"))

    def test_merge_existing_with_new_rows_deduplicates_symbol_date(self):
        existing_df = pd.DataFrame(
            {
                "symbol": ["BTCUSDT", "ETHUSDT"],
                "date": ["2026-03-29", "2026-03-29"],
                "close": [100.0, 200.0],
            }
        )
        new_df = pd.DataFrame(
            {
                "symbol": ["BTCUSDT", "BTCUSDT"],
                "date": ["2026-03-29", "2026-03-30"],
                "close": [101.0, 102.0],
            }
        )

        merged = cdb.merge_kline_data(existing_df, new_df)
        self.assertEqual(len(merged), 3)
        btc_0329 = merged[(merged["symbol"] == "BTCUSDT") & (merged["date"] == "2026-03-29")]
        self.assertEqual(float(btc_0329["close"].iloc[0]), 101.0)

    def test_filter_valid_pairs_to_latest_guide_date_only(self):
        valid_pairs = {
            "BTCUSDT": {"first_date": pd.Timestamp("2021-01-01")},
            "ETHUSDT": {"first_date": pd.Timestamp("2021-01-01")},
            "DOGEUSDT": {"first_date": pd.Timestamp("2021-01-01")},
        }
        expanded_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-02-01", "2026-03-01", "2026-03-01"]),
                "Trading_Pair": ["DOGEUSDT", "BTCUSDT", "ETHUSDT"],
            }
        )

        filtered = cdb.filter_valid_pairs_by_latest_guide_date(valid_pairs, expanded_df)
        self.assertEqual(sorted(filtered.keys()), ["BTCUSDT", "ETHUSDT"])


if __name__ == "__main__":
    unittest.main()
