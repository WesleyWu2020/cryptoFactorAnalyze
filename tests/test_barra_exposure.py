import numpy as np
import pandas as pd
import pytest

from barra.crypto_barra_exposure import (
    BarraConfig, _rolling_beta, analyze_exposures, resample_alpha_to_profile,
)


def test_beta_uses_matching_missing_samples():
    dates = pd.date_range("2024-01-01", periods=40)
    market = pd.Series(np.random.default_rng(5).normal(size=40), index=dates)
    returns = pd.DataFrame({"a": 2 * market, "b": -3 * market})
    returns.loc[dates[::3], "a"] = np.nan
    beta = _rolling_beta(returns, market, 20)
    np.testing.assert_allclose(beta["a"].dropna(), 2, atol=1e-12)
    np.testing.assert_allclose(beta["b"].dropna(), -3, atol=1e-12)


def test_long_wide_and_invalid_inputs():
    cfg = BarraConfig()
    wide = pd.DataFrame([[1., 2.], [3., 4.]], index=pd.date_range("2024-01-01", periods=2), columns=["A", "B"])
    long = wide.rename_axis(index="date", columns="instrument").stack().rename("factor").reset_index()
    pd.testing.assert_frame_equal(resample_alpha_to_profile(wide, cfg), resample_alpha_to_profile(long, cfg))
    with pytest.raises(ValueError, match="duplicate"):
        resample_alpha_to_profile(pd.concat([long, long]), cfg)
    wide.index += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="midnight"):
        resample_alpha_to_profile(wide, cfg)


@pytest.mark.parametrize("kwargs", [{"min_count": 0}, {"vol_window": 0}, {"winsor_q": .5}, {"signal_freq": "4h"}])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        BarraConfig(**kwargs)


def test_regression_residual_and_rank_diagnostics():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=3)
    x = pd.DataFrame(rng.normal(size=(3, 40)), index=dates)
    noise = pd.DataFrame(rng.normal(scale=.01, size=(3, 40)), index=dates)
    x.columns = x.columns.map(str)
    noise.columns = x.columns
    cfg = BarraConfig(winsor_q=0)
    result = analyze_exposures(3 * x + noise, {"x": x}, cfg=cfg)
    assert result["daily_barra_regression"]["r2"].min() > .999
    assert result["alpha_barra_residual"].mean(axis=1).abs().max() < 1e-12
    rank = analyze_exposures(x, {"x": x, "duplicate": 2 * x}, cfg=cfg)
    assert rank["daily_barra_regression"]["status"].eq("rank_deficient").all()
    assert rank["alpha_barra_residual"].isna().all().all()
    sparse = analyze_exposures(x.iloc[:, :5], {"x": x}, cfg=cfg)
    assert sparse["daily_barra_regression"]["status"].eq("insufficient_data").all()
    constant = analyze_exposures(x, {"constant": x * 0}, cfg=cfg)
    assert constant["alpha_barra_residual"].isna().all().all()
