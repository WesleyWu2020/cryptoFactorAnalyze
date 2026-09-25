import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "engines" / "unified_v2" / "composition_v3_strict_config.json"
SCRIPT = Path("/root/crypto-research/users/wesleywu/tmp/run_gp_v3.sh")


EXPECTED_TS = {
    "none",
    "ewm_residual",
    "fast_slow_diff",
    "rolling_tstat",
    "robust_zscore",
    "persistence_ratio",
    "event_count",
    "drawup",
    "drawdown",
    "half_life_decay",
}
EXPECTED_CS = {
    "none",
    "cs_rank",
    "cs_zscore",
    "cs_robust_zscore",
    "cs_winsor_zscore",
    "cs_tail_rank",
    "cs_neutralize_mcap",
    "cs_neutralize_liquidity",
    "cs_group_rank_mcap",
    "cs_group_rank_liquidity",
}


def test_v3_strict_config_contains_only_requested_composition_operators():
    config = json.loads(CONFIG.read_text())

    assert set(config["ts_comp_ops"]) == EXPECTED_TS
    assert set(config["cs_comp_ops"]) == EXPECTED_CS
    assert config["mode4_ops_primary"] == []
    assert config["mode4_ops_explore"] == []
    assert config["intraday_share"] == 0.0


def test_v3_script_defaults_to_2022_2023_train_and_2024_2025_oos():
    text = SCRIPT.read_text()
    assert "END_DATE=${GP_RUN_END_DATE:-20260101}" in text
    assert "TRAIN_END_DATE=${GP_RUN_TRAIN_END_DATE:-20231230}" in text
    assert "END_DATE=" in text


def test_strict_config_never_initializes_disabled_mode4():
    import numpy as np
    from . import config as C
    from . import evolution as E

    config = json.loads(CONFIG.read_text())
    C.set_search_space_overrides({
        key: config[key]
        for key in ("intraday_share", "path_shape_share", "composition_cs_share",
                    "ts_comp_ops", "cs_comp_ops", "mode4_ops_primary",
                    "mode4_ops_explore")
    })
    state = C.build_search_space_state()
    row = E._generate_policy_individual(
        np.random.default_rng(7), state, explore=True, mode_override=3
    )

    assert row[6] != 3
