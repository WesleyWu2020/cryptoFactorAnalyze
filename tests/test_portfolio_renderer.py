import unittest
import pathlib
import sys
import tempfile
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.reporter.html_renderer import render_html  # noqa: E402


class TestHtmlRenderer(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        self.returns = pd.Series(rng.normal(0.001, 0.02, 100), index=dates)
        self.tmpdir = tempfile.mkdtemp()

    def test_creates_html_file(self):
        out = render_html(self.returns, pathlib.Path(self.tmpdir) / "report.html")
        self.assertTrue(out.exists())
        self.assertTrue(out.suffix == ".html")

    def test_html_contains_metrics(self):
        out = render_html(self.returns, pathlib.Path(self.tmpdir) / "report2.html")
        content = out.read_text()
        self.assertIn("Sharpe", content)
        self.assertIn("Max Drawdown", content)
        self.assertIn("Annual Return", content)

    def test_html_contains_chart(self):
        out = render_html(self.returns, pathlib.Path(self.tmpdir) / "report3.html")
        content = out.read_text()
        self.assertIn("base64", content)  # embedded image


if __name__ == "__main__":
    unittest.main()
