"""End-to-end smoke test: runs the full CLI pipeline with synthetic data."""
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _write_synthetic_data(tmpdir: pathlib.Path) -> tuple[dict, pathlib.Path, pathlib.Path]:
    rng = np.random.default_rng(99)
    n_days, n_ins = 50, 15
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    instruments = [f"C{i:02d}USDT" for i in range(n_ins)]

    # Kline
    kline_rows = []
    prices = {ins: 100.0 for ins in instruments}
    for d in dates:
        for ins in instruments:
            kline_rows.append({"symbol": ins, "date": d.date(), "close": prices[ins]})
            prices[ins] *= (1 + rng.normal(0.001, 0.02))
    kline_csv = tmpdir / "kline.csv"
    pd.DataFrame(kline_rows).to_csv(kline_csv, index=False)

    # Factors
    factor_paths = {}
    for fname in ["mom", "rev"]:
        rows = []
        for d in dates:
            for ins in instruments:
                rows.append({"date": d.date(), "instrument": ins, "factor": rng.standard_normal()})
        fcsv = tmpdir / f"{fname}.csv"
        pd.DataFrame(rows).to_csv(fcsv, index=False)
        factor_paths[fname] = str(fcsv)

    # Universe
    pairs = ",".join(instruments)
    univ_csv = tmpdir / "universe.csv"
    univ_csv.write_text(
        "Decision_Window_Start,Decision_Window_End,Trading_Pairs,Market_Cap_USD,Quote_Volume_USD\n"
        f"2024-01-01,2024-12-31,\"{pairs}\",100,100\n"
    )

    return factor_paths, kline_csv, univ_csv


class TestE2ESmoke(unittest.TestCase):
    def setUp(self):
        self.tmpdir = pathlib.Path(tempfile.mkdtemp())
        self.factor_paths, self.kline_csv, self.univ_csv = _write_synthetic_data(self.tmpdir)
        self.report_path = self.tmpdir / "smoke_report.html"

    def test_cli_runs_and_produces_report(self):
        factor_args = [f"{k}={v}" for k, v in self.factor_paths.items()]
        cmd = [
            sys.executable, str(ROOT / "portfolio" / "main.py"),
            "--factors", *factor_args,
            "--kline", str(self.kline_csv),
            "--universe", str(self.univ_csv),
            "--top-n", "5",
            "--ic-window", "10",
            "--no-orthogonalize",
            "--output", str(self.report_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(result.returncode, 0, f"CLI failed:\n{result.stderr}")
        self.assertTrue(self.report_path.exists(), "HTML report not created")
        content = self.report_path.read_text()
        self.assertIn("Sharpe", content)
        self.assertIn("base64", content)

    def test_cli_prints_summary(self):
        factor_args = [f"{k}={v}" for k, v in self.factor_paths.items()]
        cmd = [
            sys.executable, str(ROOT / "portfolio" / "main.py"),
            "--factors", *factor_args,
            "--kline", str(self.kline_csv),
            "--universe", str(self.univ_csv),
            "--top-n", "5",
            "--ic-window", "10",
            "--no-orthogonalize",
            "--output", str(self.report_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        self.assertIn("Annual return", result.stdout)
        self.assertIn("Sharpe ratio", result.stdout)


if __name__ == "__main__":
    unittest.main()
