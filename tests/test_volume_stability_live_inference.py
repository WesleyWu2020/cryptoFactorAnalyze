import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FACTOR_MINING_DIR = REPO_ROOT / "factor_analyse" / "factor_mining"
if str(FACTOR_MINING_DIR) not in sys.path:
    sys.path.insert(0, str(FACTOR_MINING_DIR))

import Volume_Stability_Factor as volume_factor  # noqa: E402


class VolumeStabilityLiveInferenceTest(unittest.TestCase):
    def test_live_inference_keeps_latest_date_without_changing_backtest_cutoff(self):
        dates = pd.date_range("2026-01-01", periods=40, freq="D")
        symbols = ["BTCUSDT", "ETHUSDT"]

        rows = []
        for d in dates:
            for i, s in enumerate(symbols):
                close = 100 + i * 5 + (d - dates[0]).days * 0.2
                quote_volume = 1000 + i * 100 + (d - dates[0]).days * 10
                trades_count = 100 + i * 5 + ((d - dates[0]).days % 7)
                rows.append(
                    {
                        "date": d,
                        "symbol": s,
                        "close": close,
                        "quote_volume": quote_volume,
                        "trades_count": trades_count,
                    }
                )
        kline_df = pd.DataFrame(rows)

        available_tokens = {d: symbols for d in dates}
        expected_live_max = dates.max()
        rebalance_period = 3
        expected_backtest_max = expected_live_max - pd.Timedelta(days=rebalance_period)

        with mock.patch.object(
            volume_factor,
            "load_kline_df",
            side_effect=[kline_df.copy(), kline_df.copy()],
        ), mock.patch.object(
            volume_factor,
            "build_available_tokens_by_date_from_kline",
            return_value=available_tokens,
        ), mock.patch.object(
            volume_factor,
            "save_factor_df",
            return_value="/tmp/volume_stability_test.csv",
        ), mock.patch.object(
            volume_factor,
            "print_factor_summary",
        ) as summary_mock, mock.patch.object(
            volume_factor,
            "print_latest_date_inference",
        ) as live_print_mock:
            returned_df = volume_factor.create_volume_stability_factor(
                lookback_days=5,
                rebalance_period=rebalance_period,
                top_n=2,
            )

        backtest_df = summary_mock.call_args[0][0]
        live_df = live_print_mock.call_args[0][0]
        latest_date_arg = live_print_mock.call_args[0][1]

        self.assertEqual(pd.Timestamp(latest_date_arg), expected_live_max)
        self.assertEqual(pd.Timestamp(live_df["date"].max()), expected_live_max)
        self.assertEqual(pd.Timestamp(backtest_df["date"].max()), expected_backtest_max)
        self.assertEqual(pd.Timestamp(returned_df["date"].max()), expected_backtest_max)

        self.assertEqual(int(backtest_df["future_ret"].isna().sum()), 0)
        latest_live_rows = live_df[live_df["date"] == expected_live_max]
        self.assertGreater(len(latest_live_rows), 0)
        self.assertTrue(latest_live_rows["future_ret"].isna().all())


if __name__ == "__main__":
    unittest.main()
