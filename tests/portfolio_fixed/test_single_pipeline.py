import numpy as np
import pandas as pd

from factor_common.data_provider import DataProvider
from factor_common.grouping import target_weights
from factor_common.loader import load_factor
from factor_common.profiles import resolve_profile
from factor_common.backtest import run_backtest
from portfolio.fixed_config import FixedConfig
from portfolio.fixed_pipeline import run_fixed
from portfolio.portfolio_builder.fixed_weights import member_targets


def test_invalid_factor_day_is_zero_not_full_position(factor_paths):
    spec = load_factor(factor_paths[0])
    dates = pd.date_range("2024-01-01", periods=2, name="date")
    values = pd.DataFrame([[np.nan, np.nan], [3.0, 3.0]], index=dates, columns=["A", "B"])
    targets, diag = member_targets(
        values, spec, resolve_profile("perp_1d", {"n_groups": 2}), 2
    )
    assert (targets == 0).all().all()
    assert not diag["valid_signal"].any()


def test_single_pipeline_matches_shared_factor_accounting(config_dict):
    config = FixedConfig.from_dict(config_dict)
    result = run_fixed(config)
    assert result["status"] == "complete"
    spec = load_factor(config.factors[0].path)
    values = result["values"][spec.factor_id]
    provider = DataProvider(config.h5_path, as_of=config.as_of)
    index = result["targets"].index
    opens = provider.get_single_data("open", start=index[0], end=index[-1])
    quality = provider.get_quality(start=index[0], end=index[-1], symbols=values.columns)
    events = provider.get_funding(start=index[0], end=index[-1], symbols=values.columns)
    profile = resolve_profile(
        "perp_1d",
        {
            "n_groups": config.n_groups,
            "factor_direction": spec.setting["factor_direction"],
            "rebalance_days": config.rebalance_days,
            "anchor_date": config.anchor_date,
            "fee_rate": config.fee_rate,
            "slippage": config.slippage,
        },
    )
    baseline = run_backtest(
        values.reindex(index), opens, events, quality, profile,
        signal_start=config.signal_start, signal_end=config.signal_end,
    )
    for name in ("gross", "trading_net", "all_costs"):
        for key in ("ledger", "orders", "positions", "funding"):
            pd.testing.assert_frame_equal(
                result["accounting"]["scenarios"][name][key],
                baseline["scenarios"][name][key],
            )


def test_member_baselines_are_fully_liquidated(config_dict):
    result = run_fixed(FixedConfig.from_dict(config_dict))

    for baseline in result["baselines"].values():
        assert baseline["status"] == "complete"
        for scenario in baseline["scenarios"].values():
            assert scenario["status"] == "complete"
            assert scenario["diagnostics"]["liquidation_reached"] is True
            assert scenario["diagnostics"]["final_quantities"] == {}
