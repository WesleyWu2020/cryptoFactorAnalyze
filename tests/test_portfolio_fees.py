import unittest
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.backtester.fees import compute_fee  # noqa: E402


class TestFees(unittest.TestCase):
    def test_no_turnover_no_fee(self):
        w = {"A": 0.5, "B": 0.5}
        self.assertAlmostEqual(compute_fee(w, w, fee_rate=0.001), 0.0, places=8)

    def test_full_turnover(self):
        old = {"A": 1.0}
        new = {"B": 1.0}
        # |1-0| + |0-1| = 2.0 total, * 0.001 / 2 = 0.001
        self.assertAlmostEqual(compute_fee(old, new, fee_rate=0.001), 0.001, places=8)

    def test_partial_rebalance(self):
        old = {"A": 0.6, "B": 0.4}
        new = {"A": 0.4, "B": 0.6}
        # |0.4-0.6| + |0.6-0.4| = 0.4, * 0.001 / 2 = 0.0002
        self.assertAlmostEqual(compute_fee(old, new, fee_rate=0.001), 0.0002, places=8)


if __name__ == "__main__":
    unittest.main()


def test_perp_fee_basic():
    from portfolio.backtester.fees import compute_perp_fee
    # 从 0 变为 0.6 空头 → turnover 0.6，单边费率 5bps → 3bps
    fee = compute_perp_fee(old_hedge=0.0, new_hedge=0.6, fee_rate=0.0005)
    assert abs(fee - 0.6 * 0.0005) < 1e-12


def test_perp_fee_hold_zero():
    from portfolio.backtester.fees import compute_perp_fee
    fee = compute_perp_fee(old_hedge=0.3, new_hedge=0.3, fee_rate=0.0005)
    assert fee == 0.0


def test_perp_fee_decrease():
    from portfolio.backtester.fees import compute_perp_fee
    fee = compute_perp_fee(old_hedge=0.8, new_hedge=0.3, fee_rate=0.0005)
    assert abs(fee - 0.5 * 0.0005) < 1e-12
