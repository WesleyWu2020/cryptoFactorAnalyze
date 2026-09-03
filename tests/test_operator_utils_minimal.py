import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FACTOR_MINING_DIR = REPO_ROOT / "factor_analyse" / "factor_mining"
if str(FACTOR_MINING_DIR) not in sys.path:
    sys.path.insert(0, str(FACTOR_MINING_DIR))

import operator_utils as op  # noqa: E402


class OperatorUtilsMinimalTest(unittest.TestCase):
    def test_category1_direct_ops(self):
        x = pd.Series([-1.0, 0.0, 1.0])

        self.assertEqual(op.ABS(x).tolist(), [1.0, 0.0, 1.0])
        self.assertTrue(np.allclose(op.LOGABS(x).to_numpy(), np.log(np.array([1.0, 0.0, 1.0]) + 1e-12)))
        self.assertEqual(op.AS_FLOAT(x > 0).tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(op.RD(pd.Series([1.2345]), 3).iloc[0], 1.234)
        self.assertEqual(op.SIGN(x).tolist(), [-1.0, 0.0, 1.0])
        self.assertAlmostEqual(float(op.SIN(pd.Series([0.0])).iloc[0]), 0.0, places=10)
        self.assertAlmostEqual(float(op.ARCTAN(pd.Series([1.0])).iloc[0]), np.pi / 4, places=10)

    def test_ts_ops_grouped_by_symbol(self):
        df = pd.DataFrame(
            {
                "symbol": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "x": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            }
        )

        d1 = op.delay(df["x"], 1, by=df["symbol"])
        self.assertTrue(np.isnan(d1.iloc[0]))
        self.assertEqual(d1.iloc[1], 1.0)
        self.assertTrue(np.isnan(d1.iloc[4]))
        self.assertEqual(d1.iloc[5], 10.0)

        m2 = op.ts_mean(df["x"], 2, by=df["symbol"])
        self.assertTrue(np.isnan(m2.iloc[0]))
        self.assertAlmostEqual(m2.iloc[1], 1.5, places=10)
        self.assertAlmostEqual(m2.iloc[2], 2.5, places=10)
        self.assertAlmostEqual(m2.iloc[5], 15.0, places=10)

        s2 = op.ts_sum(df["x"], 2, by=df["symbol"])
        self.assertTrue(np.isnan(s2.iloc[0]))
        self.assertEqual(s2.iloc[1], 3.0)
        self.assertEqual(s2.iloc[6], 50.0)

    def test_cs_ops_grouped_by_date(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-02"]
                ),
                "x": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
            }
        )

        rk = op.cs_rank(df["x"], by_date=df["date"])
        self.assertAlmostEqual(rk.iloc[0], 1 / 3, places=10)
        self.assertAlmostEqual(rk.iloc[1], 2 / 3, places=10)
        self.assertAlmostEqual(rk.iloc[2], 1.0, places=10)

        sc = op.cs_scale(df["x"], by_date=df["date"])
        self.assertAlmostEqual(sc.iloc[0], -1.0, places=10)
        self.assertAlmostEqual(sc.iloc[1], 0.0, places=10)
        self.assertAlmostEqual(sc.iloc[2], 1.0, places=10)

        zs = op.cs_zscore(df["x"], by_date=df["date"])
        self.assertAlmostEqual(float(zs.iloc[:3].mean()), 0.0, places=10)
        self.assertAlmostEqual(float(zs.iloc[3:].mean()), 0.0, places=10)

    def test_rolling_corr_and_aliases(self):
        df = pd.DataFrame(
            {
                "symbol": ["A"] * 5 + ["B"] * 5,
                "x": [1, 2, 3, 4, 5, 2, 4, 6, 8, 10],
                "y": [2, 4, 6, 8, 10, 3, 6, 9, 12, 15],
            }
        )
        c = op.corr(df["x"], df["y"], 3, by=df["symbol"])
        valid = c.dropna()
        self.assertTrue((valid > 0.999999).all())

        # alias smoke test
        c2 = op.CORRELATION(df["x"], df["y"], 3, by=df["symbol"])
        self.assertTrue(np.allclose(c.fillna(0).to_numpy(), c2.fillna(0).to_numpy()))

    def test_category3_time_condition_ops(self):
        cond = pd.Series([False, True, False, False, True, True, False])
        bl = op.BARSLAST(cond)
        # before first True -> NaN
        self.assertTrue(np.isnan(bl.iloc[0]))
        self.assertEqual(bl.iloc[1], 0.0)
        self.assertEqual(bl.iloc[2], 1.0)
        self.assertEqual(bl.iloc[3], 2.0)

        blc = op.BARSLASTCOUNT(cond)
        self.assertEqual(blc.tolist(), [0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0])

        consted = op.CONST(pd.Series([1, 2, 3]))
        self.assertEqual(consted.tolist(), [3, 3, 3])

    def test_category4_power_ops(self):
        x = pd.Series([-2.0, -1.0, 0.0, 2.0])
        self.assertEqual(op.POWER(op.ABS(x), 2).tolist(), [4.0, 1.0, 0.0, 4.0])
        self.assertEqual(op.SIGNEDPOWER(x, 2).tolist(), [-4.0, -1.0, 0.0, 4.0])

    def test_category5_extended_ops(self):
        df = pd.DataFrame(
            {
                "symbol": ["A"] * 6,
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        by = df["symbol"]
        x = df["x"]

        self.assertEqual(op.ROC(x, 1, by=by).iloc[1], 1.0)
        self.assertEqual(op.PCT_CHANGE(x, 1, by=by).iloc[2], 0.5)
        self.assertEqual(op.TS_ARGMAX(x, 3, by=by).iloc[4], 3.0)
        self.assertEqual(op.TS_ARGMIN(x, 3, by=by).iloc[4], 1.0)
        self.assertEqual(op.HHVBARS(x, 3, by=by).iloc[4], 0.0)
        self.assertEqual(op.LLVBARS(x, 3, by=by).iloc[4], 2.0)

        cond = pd.Series([False, True, False, False, True, False])
        self.assertEqual(op.COUNT(cond, 3, by=by).iloc[5], 1.0)
        self.assertEqual(bool(op.EVERY(pd.Series([True] * 6), 3, by=by).iloc[5]), True)
        self.assertEqual(bool(op.EXIST(cond, 3, by=by).iloc[5]), True)

        # 线性序列: 斜率接近1，截距随窗口变化
        self.assertAlmostEqual(op.SLOPE(x, 4, by=by).iloc[5], 1.0, places=10)
        self.assertAlmostEqual(op.ANGLE(x, 4, by=by).iloc[5], 45.0, places=10)
        self.assertAlmostEqual(op.INTERCEPT(x, 4, by=by).iloc[5], 3.0, places=10)
        self.assertAlmostEqual(op.FORCAST(x, 4, by=by).iloc[5], 7.0, places=10)

        # 价格变化类
        self.assertAlmostEqual(op.SUM_ABS_PRICE_CHANGE(x, 3, by=by).iloc[5], 3.0, places=10)
        self.assertAlmostEqual(op.MEAN_ABS_PRICE_CHANGE(x, 3, by=by).iloc[5], 1.0, places=10)

    def test_category5_ema_dma_wma_and_future_returns(self):
        df = pd.DataFrame({"symbol": ["A"] * 6, "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        by = df["symbol"]
        x = df["x"]

        ema = op.EMA(x, 3, by=by)
        dma = op.DMA(x, 0.5, by=by)
        wma = op.WMA(x, 3, by=by)
        fr = op.FUTURE_RETURNS(x, 1, by=by)

        self.assertTrue(np.isnan(ema.iloc[0]))
        self.assertAlmostEqual(dma.iloc[0], 1.0, places=10)
        self.assertAlmostEqual(wma.iloc[2], (1 * 1 + 2 * 2 + 3 * 3) / 6, places=10)
        self.assertAlmostEqual(fr.iloc[0], 1.0, places=10)
        self.assertTrue(np.isnan(fr.iloc[-1]))

    def test_if_else_with_series_and_scalar(self):
        x = pd.Series([1.0, -1.0, 0.0])
        out = op.if_else(x > 0, 1.0, -1.0)
        self.assertEqual(out.tolist(), [1.0, -1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
