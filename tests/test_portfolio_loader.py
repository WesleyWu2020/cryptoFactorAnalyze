import unittest
import tempfile
import pathlib
import sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.factor_pool.loader import load_factor_csv, load_many  # noqa: E402


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "instrument": ["BTCUSDT", "ETHUSDT"],
            "factor": [0.1, 0.2],
            "future_ret": [0.01, 0.02],
        }).to_csv(self.d / "f1.csv", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_factor_csv_drops_future_ret(self):
        df = load_factor_csv(self.d / "f1.csv", name="f1")
        self.assertEqual(list(df.columns), ["date", "instrument", "f1"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df["f1"].iloc[0], 0.1)

    def test_load_factor_csv_parses_date(self):
        df = load_factor_csv(self.d / "f1.csv", name="f1")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))

    def test_load_many(self):
        pd.DataFrame({
            "date": ["2024-01-01"],
            "instrument": ["BTCUSDT"],
            "factor": [1.5],
        }).to_csv(self.d / "f2.csv", index=False)
        out = load_many({"f1": self.d / "f1.csv", "f2": self.d / "f2.csv"})
        self.assertEqual(set(out.keys()), {"f1", "f2"})
        self.assertIn("f1", out["f1"].columns)
        self.assertIn("f2", out["f2"].columns)


if __name__ == "__main__":
    unittest.main()
