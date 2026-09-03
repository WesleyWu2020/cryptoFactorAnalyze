import unittest
import pathlib
import sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.portfolio_builder.universe import build_universe  # noqa: E402
from portfolio.portfolio_builder.constraints import DEFAULT_BLACKLIST  # noqa: E402


class TestUniverseBuilder(unittest.TestCase):
    def setUp(self):
        # Build a minimal in-memory CSV
        import io, tempfile, os
        data = """Decision_Window_Start,Decision_Window_End,Trading_Pairs,Market_Cap_USD,Quote_Volume_USD
2024-01-01,2024-01-15,"BTCUSDT,ETHUSDT,USDTUSDT,WBTCUSDT,SOLUSDT",100,100
2024-01-16,2024-01-31,"BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT",100,100
"""
        self.tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        self.tmp.write(data)
        self.tmp.close()
        self.path = self.tmp.name

    def tearDown(self):
        import os
        os.unlink(self.path)

    def test_blacklist_filtered(self):
        univ = build_universe(self.path)
        # USDTUSDT and WBTCUSDT should be filtered out
        for instruments in univ.values():
            self.assertNotIn("USDTUSDT", instruments)
            self.assertNotIn("WBTCUSDT", instruments)

    def test_date_mapping(self):
        univ = build_universe(self.path)
        d = pd.Timestamp("2024-01-05")
        self.assertIn(d, univ)
        self.assertIn("BTCUSDT", univ[d])

    def test_window_boundary(self):
        univ = build_universe(self.path)
        # Jan 16 is the start of window 2
        d16 = pd.Timestamp("2024-01-16")
        self.assertIn("BNBUSDT", univ[d16])
        self.assertNotIn("BNBUSDT", univ[pd.Timestamp("2024-01-15")])


if __name__ == "__main__":
    unittest.main()
