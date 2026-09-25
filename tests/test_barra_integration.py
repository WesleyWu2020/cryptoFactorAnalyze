import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from barra.crypto_barra_exposure import (
    BarraConfig, analyze_and_write, analyze_evaluate_result, build_barra_exposures,
)


@pytest.fixture
def daily_store(tmp_path):
    from data.crypto_quant.store import CryptoQuantStore
    from data.crypto_quant.schemas import TABLE_SPECS
    rng = np.random.default_rng(24)
    dates = pd.date_range("2023-10-01", periods=150)
    symbols = [f"C{i}USDT" for i in range(35)]
    close = 100 * np.exp(np.cumsum(rng.normal(0, .03, (150, 35)), axis=0))
    rows = []
    panel = []
    for i, day in enumerate(dates):
        for j, symbol in enumerate(symbols):
            row = {key: 1. for key in TABLE_SPECS["klines_daily"].columns}
            row.update(date=day, symbol=symbol, close_time=day + pd.Timedelta(hours=23),
                       close=close[i, j], open=close[i, j], high=close[i, j]*1.01,
                       low=close[i, j]*.99, quote_volume=float(rng.lognormal(12, 2)))
            rows.append(row)
            panel.append({"date": day, "binance_symbol": symbol, "funding_rate_mean": rng.normal(0, .001)})
    universe = pd.DataFrame([
        {"decision_date": dates[0] if j < 30 else dates[125],
         "effective_date": dates[0] if j < 30 else dates[126],
         "effective_end_date": pd.NaT, "binance_symbol": symbol}
        for j, symbol in enumerate(symbols)
    ])
    path = tmp_path / "market.h5"
    store = CryptoQuantStore(path)
    store.replace("klines_daily", pd.DataFrame(rows))
    # Only columns consumed by provider, stored as queryable tables.
    with pd.HDFStore(path, mode="a") as hdf:
        hdf.put("universe_monthly", universe, format="table", data_columns=["decision_date"])
        hdf.put("research_panel_daily", pd.DataFrame(panel), format="table", data_columns=["date"])
    alpha = pd.DataFrame(rng.normal(size=(50, 35)), index=dates[100:], columns=symbols)
    return path, alpha


def test_missing_close_does_not_create_zero_returns(daily_store):
    path, alpha = daily_store
    day = alpha.index[5]
    with pd.HDFStore(path, mode="a") as hdf:
        klines = hdf["klines_daily"]
        klines.loc[(klines["date"] == day) & (klines["symbol"] == "C0USDT"), "close"] = np.nan
        hdf.put("klines_daily", klines, format="table", data_columns=["date"])
    styles = build_barra_exposures(alpha, h5_path=path, cfg=BarraConfig())
    assert pd.isna(styles["reversal_1d"].loc[day, "C0USDT"])
    assert pd.isna(styles["reversal_1d"].loc[day + pd.Timedelta(days=1), "C0USDT"])


def test_h5_cutoff_future_members_and_outputs(daily_store, tmp_path):
    path, alpha = daily_store
    cfg = BarraConfig()
    before = path.stat()
    full = analyze_and_write({"factor_value": alpha}, h5_path=path, out_dir=tmp_path / "out", cfg=cfg)
    assert full["daily_barra_regression"]["status"].eq("ok").all()
    exposures = build_barra_exposures(alpha, h5_path=path, cfg=cfg)
    for cutoff in (alpha.index[15], alpha.index[35]):
        partial = analyze_evaluate_result({"factor_value": alpha}, h5_path=path, cfg=cfg, as_of=cutoff)
        for key in ("alpha_daily", "alpha_barra_residual"):
            expected = full[key].loc[:cutoff].reindex(columns=partial[key].columns)
            pd.testing.assert_frame_equal(expected, partial[key], atol=1e-12, rtol=0)
        truncated = build_barra_exposures(alpha.loc[:cutoff], h5_path=path, cfg=cfg, as_of=cutoff)
        for name, frame in truncated.items():
            pd.testing.assert_frame_equal(exposures[name].loc[:cutoff].reindex(columns=frame.columns), frame, atol=1e-12, rtol=0)
    assert before.st_mtime_ns == path.stat().st_mtime_ns
    meta = json.loads((tmp_path / "out/metadata.json").read_text())
    assert meta["valid_regression_days"] == 50
    long = pd.read_parquet(tmp_path / "out/alpha_barra_residual_long.parquet")
    assert list(long.columns) == ["date", "instrument", "factor"]
    assert np.isfinite(long["factor"]).all()
    with pytest.warns(UserWarning, match="open_interest"):
        build_barra_exposures(alpha, h5_path=path, cfg=BarraConfig(include_crypto_optional=True))


def test_cli_parquet(daily_store, tmp_path):
    path, alpha = daily_store
    source = tmp_path / "alpha.parquet"
    alpha.to_parquet(source)
    proc = subprocess.run([sys.executable, "-m", "barra.crypto_barra_exposure", "--factor-parquet", str(source),
                           "--h5", str(path), "--out-dir", str(tmp_path / "cli")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "valid_regression_days=50/50" in proc.stdout
