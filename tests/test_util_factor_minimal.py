import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FACTOR_MINING_DIR = REPO_ROOT / "factor_analyse" / "factor_mining"
if str(FACTOR_MINING_DIR) not in sys.path:
    sys.path.insert(0, str(FACTOR_MINING_DIR))

import util_factor  # noqa: E402


class UtilFactorMinimalTest(unittest.TestCase):
    def test_load_historical_marketcap_supports_trading_pairs_column(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "sample.csv"
            df = pd.DataFrame(
                {
                    "Date": ["2026-01-01", "2026-01-01"],
                    "Trading_Pairs": ["BTCUSDT,ETHUSDT", "SOLUSDT"],
                }
            )
            df.to_csv(csv_path, index=False)

            loaded = util_factor.load_historical_marketcap(str(csv_path))
            self.assertIn("Trading_Pair", loaded.columns)
            self.assertEqual(sorted(loaded["Trading_Pair"].unique().tolist()), ["BTCUSDT", "ETHUSDT", "SOLUSDT"])

    def test_latest_kline_path_ignores_non_csv_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "binance_daily_klines_20250101.csv").write_text("date,symbol\n", encoding="utf-8")
            (td_path / "binance_daily_klines_20250102.csv.meta.json").write_text("{}", encoding="utf-8")
            (td_path / "binance_daily_klines_20250103.tmp").write_text("", encoding="utf-8")
            (td_path / "binance_daily_klines_20250104.csv").write_text("date,symbol\n", encoding="utf-8")

            old_dir = util_factor.KLINE_DIR
            try:
                util_factor.KLINE_DIR = str(td_path)
                latest = util_factor.latest_kline_path()
            finally:
                util_factor.KLINE_DIR = old_dir

            self.assertTrue(latest.endswith("binance_daily_klines_20250104.csv"))

    def test_filter_group_by_availability_matches_latest_snapshot_logic(self):
        group = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-06", "2024-01-10"]),
                "value": [1, 2, 3],
            }
        )
        available = {
            "2024-01-01": ["ETHUSDT"],
            pd.Timestamp("2024-01-05"): ["BTCUSDT"],
            pd.Timestamp("2024-01-09"): ["BTCUSDT"],
        }

        out = util_factor.filter_group_by_availability(group, "BTCUSDT", available)
        out_dates = out["date"].dt.strftime("%Y-%m-%d").tolist()
        self.assertEqual(out_dates, ["2024-01-06", "2024-01-10"])

    def test_winsorize_and_rank_keep_date_column(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "dv_raw": [1.0, 1000.0, 2.0],
            }
        )
        w = util_factor.winsorize_by_date(df, col="dv_raw", n_std=3.0)
        self.assertIn("date", w.columns)
        r = util_factor.rank_to_unit_by_date(w, col="dv_raw", out_col="factor")
        self.assertIn("date", r.columns)
        self.assertIn("factor", r.columns)


if __name__ == "__main__":
    unittest.main()
