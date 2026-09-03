"""Tests for Path B portfolio configuration (BTC core + Alt short)."""
import pathlib
import sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.config import PortfolioConfig  # noqa: E402


def test_pathB_defaults():
    """Path B fields should have sensible defaults."""
    cfg = PortfolioConfig()
    assert 0.0 <= cfg.btc_base_min <= cfg.btc_base_max <= 1.0
    assert cfg.alt_short_exposure_max < 0
    assert 0 < cfg.universe_rank_min < cfg.universe_rank_max
    assert 0 < cfg.funding_filter_percentile < 1
    assert cfg.squeeze_stop_return > 0 and cfg.squeeze_stop_window >= 1


def test_pathB_invalid_range_raises():
    """Invalid Path B parameter ranges should raise ValueError."""
    with pytest.raises(ValueError):
        PortfolioConfig(btc_base_min=0.8, btc_base_max=0.5)
    with pytest.raises(ValueError):
        PortfolioConfig(universe_rank_min=100, universe_rank_max=30)
    with pytest.raises(ValueError):
        PortfolioConfig(alt_short_exposure_max=0.1)


def test_pathB_valid_instantiation():
    """Valid Path B configurations should instantiate without error."""
    cfg = PortfolioConfig(
        btc_base_min=0.3,
        btc_base_max=0.8,
        alt_short_exposure_max=-0.5,
        universe_rank_min=20,
        universe_rank_max=150,
        funding_filter_percentile=0.5,
        squeeze_stop_return=0.1,
        squeeze_stop_window=3,
        short_top_n=3,
    )
    assert cfg.btc_base_min == 0.3
    assert cfg.btc_base_max == 0.8
    assert cfg.alt_short_exposure_max == -0.5
    assert cfg.universe_rank_min == 20
    assert cfg.universe_rank_max == 150
    assert cfg.funding_filter_percentile == 0.5
    assert cfg.squeeze_stop_return == 0.1
    assert cfg.squeeze_stop_window == 3
    assert cfg.short_top_n == 3


def test_pathB_from_dict():
    """Path B fields should be loadable via from_dict()."""
    cfg = PortfolioConfig.from_dict({
        "btc_base_min": 0.5,
        "btc_base_max": 0.75,
        "alt_short_exposure_max": -0.25,
    })
    assert cfg.btc_base_min == 0.5
    assert cfg.btc_base_max == 0.75
    assert cfg.alt_short_exposure_max == -0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
