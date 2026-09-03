import pandas as pd
import numpy as np


def test_exposure_strong_bull():
    from portfolio.sizing.exposure_controller import compute_exposure_and_hedge
    alt_exp, hedge = compute_exposure_and_hedge(
        T=1.0, beta_port=1.3,
        alt_min=0.5, alt_max=1.0, hedge_cap_mult=1.2,
    )
    assert abs(alt_exp - 1.0) < 1e-9  # Alt满仓
    assert abs(hedge - 0.0) < 1e-9    # 不对冲


def test_exposure_strong_bear():
    from portfolio.sizing.exposure_controller import compute_exposure_and_hedge
    alt_exp, hedge = compute_exposure_and_hedge(
        T=-1.0, beta_port=1.3,
        alt_min=0.5, alt_max=1.0, hedge_cap_mult=1.2,
    )
    assert abs(alt_exp - 0.5) < 1e-9  # Alt半仓
    # hedge_raw = 1.3*0.5*(1-(-1)) = 1.3; cap = 0.5*1.2 = 0.6 → 0.6
    assert abs(hedge - 0.6) < 1e-9


def test_exposure_neutral():
    from portfolio.sizing.exposure_controller import compute_exposure_and_hedge
    alt_exp, hedge = compute_exposure_and_hedge(
        T=0.0, beta_port=1.0,
        alt_min=0.5, alt_max=1.0, hedge_cap_mult=1.2,
    )
    assert abs(alt_exp - 0.75) < 1e-9
    # hedge_raw = 1.0*0.75*1 = 0.75; cap = 0.75*1.2=0.9 → 0.75
    assert abs(hedge - 0.75) < 1e-9


def test_exposure_clips_T_out_of_range():
    from portfolio.sizing.exposure_controller import compute_exposure_and_hedge
    alt_exp, hedge = compute_exposure_and_hedge(T=2.0, beta_port=1.0)
    assert alt_exp <= 1.0
    assert hedge >= 0.0


def test_ema_smoothing_reduces_jitter():
    from portfolio.sizing.exposure_controller import smooth_exposure_series
    # 突变信号
    raw = pd.Series([0.5]*10 + [1.0]*10, index=pd.date_range("2024-01-01", periods=20))
    smoothed = smooth_exposure_series(raw, alpha=0.05)
    assert len(smoothed) == len(raw)
    # 平滑后第11天应远低于1.0（只吸收5%）
    assert smoothed.iloc[10] < 0.6
    # 最终收敛接近1.0
    assert smoothed.iloc[-1] < 1.0
