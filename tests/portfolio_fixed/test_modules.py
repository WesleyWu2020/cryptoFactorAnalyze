from portfolio.fixed_config import FixedConfig
from portfolio.fixed_provenance import code_snapshot
from portfolio.factor_pool.module_loader import load_members


def test_direction_is_read_from_explicit_module(config_dict, factor_paths):
    config_dict["factors"] = [{"path": str(factor_paths[1]), "allocation": 1.0}]
    config_dict["code_hashes"] = code_snapshot(factor_paths[1:])
    specs = load_members(FixedConfig.from_dict(config_dict))
    assert specs[0].factor_id == "ReverseClose"
    assert specs[0].setting["factor_direction"] == -1


def test_modified_source_invalidates_freeze(config_dict, factor_paths):
    factor_paths[0].write_text(factor_paths[0].read_text() + "\n# changed\n")
    try:
        load_members(FixedConfig.from_dict(config_dict))
    except ValueError as exc:
        assert "code snapshot" in str(exc)
    else:
        raise AssertionError("modified source must invalidate frozen snapshot")


def test_repository_gp_negative_direction_is_loaded():
    from factor_common.loader import load_factor

    spec = load_factor("factor_analyse/factor_mining/GP_Factor/GP_064185107a8f267e.py")
    assert spec.setting["factor_direction"] == -1
