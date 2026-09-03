import unittest
import pathlib
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.pipeline import run_pipeline  # noqa: E402
from portfolio.config import PortfolioConfig  # noqa: E402


class TestPipeline(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.tmpdir = pathlib.Path(tempfile.mkdtemp())

        # Kline: 60 days, 20 instruments
        n_days, n_ins = 60, 20
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        instruments = [f"C{i:02d}USDT" for i in range(n_ins)]
        kline_rows = []
        prices = {ins: 100.0 for ins in instruments}
        for d in dates:
            for ins in instruments:
                kline_rows.append({"symbol": ins, "date": d, "close": prices[ins]})
                prices[ins] *= (1 + rng.normal(0.001, 0.02))
        self.kline_csv = self.tmpdir / "kline.csv"
        pd.DataFrame(kline_rows).to_csv(self.kline_csv, index=False)

        # Factor: 60 days, 20 instruments, 2 factors
        factor_rows1, factor_rows2 = [], []
        for d in dates:
            for ins in instruments:
                factor_rows1.append({"date": d.date(), "instrument": ins, "factor": rng.standard_normal()})
                factor_rows2.append({"date": d.date(), "instrument": ins, "factor": rng.standard_normal()})
        f1_csv = self.tmpdir / "f1.csv"
        f2_csv = self.tmpdir / "f2.csv"
        pd.DataFrame(factor_rows1).to_csv(f1_csv, index=False)
        pd.DataFrame(factor_rows2).to_csv(f2_csv, index=False)
        self.factor_paths = {"f1": str(f1_csv), "f2": str(f2_csv)}

        # Universe CSV: one window covering all dates
        univ_csv = self.tmpdir / "universe.csv"
        pairs = ",".join(instruments)
        univ_csv.write_text(
            "Decision_Window_Start,Decision_Window_End,Trading_Pairs,Market_Cap_USD,Quote_Volume_USD\n"
            f"2024-01-01,2024-03-31,\"{pairs}\",100,100\n"
        )
        self.universe_csv = univ_csv

        self.cfg = PortfolioConfig(
            top_n=5, ic_window=10, label_period=1,
            fee_rate=0.001, orthogonalize=False,
        )

    def test_returns_series(self):
        ret = run_pipeline(self.factor_paths, self.kline_csv, self.universe_csv, self.cfg)
        self.assertIsInstance(ret, pd.Series)
        self.assertGreater(len(ret), 0)

    def test_html_output(self):
        out = self.tmpdir / "report.html"
        run_pipeline(self.factor_paths, self.kline_csv, self.universe_csv, self.cfg, output_path=out)
        self.assertTrue(out.exists())

    def test_returns_finite(self):
        ret = run_pipeline(self.factor_paths, self.kline_csv, self.universe_csv, self.cfg)
        import numpy as np
        self.assertTrue(np.isfinite(ret.values).all())


if __name__ == "__main__":
    unittest.main()
