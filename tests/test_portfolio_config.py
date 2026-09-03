import unittest
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.config import PortfolioConfig  # noqa: E402


class TestPortfolioConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PortfolioConfig()
        self.assertEqual(cfg.top_n, 10)
        self.assertEqual(cfg.ic_window, 20)
        self.assertEqual(cfg.label_period, 1)
        self.assertAlmostEqual(cfg.fee_rate, 0.001)
        self.assertEqual(cfg.winsorize_pct, (0.01, 0.99))
        self.assertTrue(cfg.orthogonalize)
        self.assertEqual(cfg.output_dir, "portfolio/output")

    def test_from_dict(self):
        cfg = PortfolioConfig.from_dict({"top_n": 20, "fee_rate": 0.002})
        self.assertEqual(cfg.top_n, 20)
        self.assertAlmostEqual(cfg.fee_rate, 0.002)
        self.assertEqual(cfg.ic_window, 20)  # default unchanged

    def test_custom_instantiation(self):
        cfg = PortfolioConfig(top_n=5, ic_window=10)
        self.assertEqual(cfg.top_n, 5)
        self.assertEqual(cfg.ic_window, 10)
        self.assertEqual(cfg.label_period, 1)  # default unchanged


def test_hedged_config_defaults():
    cfg = PortfolioConfig()
    # IS/OOS
    assert cfg.is_end_date == "2023-12-31"
    assert cfg.min_ic_ir == 0.05
    # Trend
    assert cfg.ema_short == 50
    assert cfg.ema_long == 200
    assert cfg.zscore_window == 120
    assert cfg.ema_ratio_saturation == 0.15
    assert cfg.zscore_saturation == 2.0
    assert cfg.t_smoothing_span == 5
    # Exposure
    assert cfg.alt_max_exposure == 1.0
    assert cfg.alt_min_exposure == 0.5
    assert cfg.hedge_cap_multiplier == 1.2
    assert cfg.beta_window == 60
    assert cfg.beta_prior == 1.3
    assert cfg.alt_exposure_ema_alpha == 0.05
    # Costs
    assert cfg.alt_fee_rate == 0.001
    assert cfg.perp_fee_rate == 0.0005
    assert cfg.funding_rate_annual == 0.1095


def test_hedged_config_backward_compat():
    """Old fields must be retained with unchanged defaults."""
    cfg = PortfolioConfig()
    assert cfg.top_n == 10  # default not changed; pipeline calls with 5 explicitly
    assert cfg.rebalance_period == 1
    assert cfg.fee_rate == 0.001


def test_from_dict_with_hedged_fields():
    """New hedged fields are correctly loaded via from_dict."""
    cfg = PortfolioConfig.from_dict({
        "ema_short": 30,
        "ema_long": 150,
        "beta_window": 90,
        "alt_max_exposure": 1.5,
    })
    assert cfg.ema_short == 30
    assert cfg.ema_long == 150
    assert cfg.beta_window == 90
    assert cfg.alt_max_exposure == 1.5
    assert cfg.top_n == 10  # unset fields keep defaults


def test_hedged_config_validation():
    """Invalid configs raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="ema_short"):
        PortfolioConfig(ema_short=200, ema_long=50)
    with pytest.raises(ValueError, match="alt_min_exposure"):
        PortfolioConfig(alt_min_exposure=1.5, alt_max_exposure=1.0)
    with pytest.raises(ValueError, match="ema_ratio_saturation"):
        PortfolioConfig(ema_ratio_saturation=0.0)
    with pytest.raises(ValueError, match="alt_exposure_ema_alpha"):
        PortfolioConfig(alt_exposure_ema_alpha=0.0)


if __name__ == "__main__":
    unittest.main()
