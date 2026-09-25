"""Regression checks for information-preserving GP semantics and tie handling."""
import numpy as np
import pandas as pd
import pytest

from Genetic_Algorithm import export as export_module
from Genetic_Algorithm.evaluator import evaluate_tree
from Genetic_Algorithm.export import export_factor
from Genetic_Algorithm.expression import Node, validate_tree
from Genetic_Algorithm.operators import rolling_hit_rate
from factor_common.grouping import target_weights
from factor_common.loader import load_factor
from factor_common.profiles import resolve_profile


def profile(**kwargs):
    return resolve_profile("perp_1d", {"n_groups": 5, "group_tie_policy": "symmetric_fractional", **kwargs})


def test_hit_rate_preserves_missing_nonfinite_and_real_zero():
    a = pd.DataFrame({"x": [1., np.nan, 1., 0., -1., np.inf, -np.inf, 1., 1.]})
    result = rolling_hit_rate(a, 2).x
    assert result.iloc[[0, 1, 2, 5, 6, 7]].isna().all()
    assert result.iloc[[3, 4, 8]].tolist() == [.5, 0., 1.]


@pytest.mark.parametrize("child", [
    Node("rank", (Node("taker_base_ratio"),)),
    Node("rolling_mean", (Node("rank", (Node("close"),)),), window=3),
    Node("negate", (Node("abs", (Node("return_1d"),)),)),
])
def test_provably_degenerate_hit_rate_is_rejected(child):
    with pytest.raises(ValueError, match="degenerate rolling_hit_rate"):
        validate_tree(Node("rolling_hit_rate", (child,), window=3), {})


def test_nonnegative_is_not_strictly_positive():
    validate_tree(Node("rolling_hit_rate", (Node("abs", (Node("return_1d"),)),), window=3), {})


def test_symmetric_ties_are_name_and_column_invariant():
    values = pd.DataFrame([[0, 0, 0, 1, 1, 1, 1], [1] * 7],
                          index=pd.date_range("2024-01-01", periods=2), columns=list("ABCDEFG"))
    expected = target_weights(values, profile())
    rename = dict(zip(values.columns, reversed(values.columns)))
    actual = target_weights(values.rename(columns=rename).iloc[:, ::-1], profile())
    for name in expected:
        restored = actual[name].rename(columns=rename).reindex(columns=values.columns)
        pd.testing.assert_frame_equal(expected[name], restored)
        if name != "long_short":
            np.testing.assert_allclose(expected[name].sum(axis=1), 1.)
    assert expected["long_short"].iloc[1].eq(0).all()
    assert expected["long_short"].iloc[0].abs().sum() == pytest.approx(1.)
    assert expected["long_short"].sum(axis=1).abs().max() < 1e-12
    inverse = target_weights(values, profile(factor_direction=-1))
    pd.testing.assert_frame_equal(inverse["long_short"], -expected["long_short"])


def test_fractional_weights_preserve_untied_legacy_and_insufficient_days():
    values = pd.DataFrame([[5, 2, 7, 1, 4, 3, 6], [np.nan] * 6 + [1]],
                          index=pd.date_range("2024-01-01", periods=2))
    new = target_weights(values, profile())
    old = target_weights(values, profile(group_tie_policy="legacy_instrument"))
    for key in old:
        pd.testing.assert_frame_equal(new[key], old[key])


def test_missing_cells_do_not_receive_fractional_weight():
    values = pd.DataFrame([[1, 1, 2, 2, 2, np.nan, np.inf]],
                          index=pd.date_range("2024-01-01", periods=1))
    for weight in target_weights(values, profile()).values():
        assert weight.iloc[0, -2:].eq(0).all()


def test_partial_tie_cancellation_does_not_restore_gross_exposure():
    values = pd.DataFrame([[0] * 9 + [1]], index=pd.date_range("2024-01-01", periods=1))
    weights = target_weights(values, profile())["long_short"].iloc[0]
    np.testing.assert_allclose(weights.iloc[:9], -1 / 36)
    assert weights.iloc[9] == pytest.approx(.25)
    assert weights.abs().sum() == pytest.approx(.5)


def test_changed_shared_runtime_cannot_reuse_training_or_frozen_evidence(tmp_path, monkeypatch):
    from argparse import Namespace
    from Genetic_Algorithm import cli
    from Genetic_Algorithm.artifacts import write_artifact

    hashes = cli._runtime_hashes()
    provenance = {"selected_code_content_hashes": hashes,
                  "backtest_profile": {"group_tie_policy": "symmetric_fractional"}}
    cli._verify_training_runtime(provenance)
    for key in ("factor_common/grouping.py", "factor_common/profiles.py"):
        assert key in hashes
    frozen = tmp_path / "frozen.json"
    write_artifact(frozen, {"runtime_source_hashes": hashes, "candidates": []}, immutable=True)
    monkeypatch.setattr(cli, "_runtime_hashes", lambda: {**hashes, "factor_common/grouping.py": "changed"})
    with pytest.raises(ValueError, match="training runtime/semantics mismatch"):
        cli._verify_training_runtime(provenance)
    with pytest.raises(ValueError, match="frozen runtime/semantics mismatch"):
        cli._export(Namespace(manifest=frozen, output_dir=tmp_path / "exports"))
    assert not (tmp_path / "exports").exists()


def test_hit_rate_prefix_consistency_and_export_runtime_guard(tmp_path, monkeypatch):
    index = pd.date_range("2024-01-01", periods=30)
    close = pd.DataFrame(np.exp(np.random.default_rng(21).normal(size=(30, 8))), index=index)
    close.iloc[7, 0] = np.nan
    eligible = pd.DataFrame(True, index=index, columns=close.columns)
    eligible.iloc[15:19, 1] = False
    tree = Node("rolling_hit_rate", (Node("return_1d"),), window=5)
    full = evaluate_tree(tree, {"close": close}, eligible)
    truncated = evaluate_tree(tree, {"close": close.iloc[:20]}, eligible.iloc[:20])
    pd.testing.assert_frame_equal(full.iloc[:20], truncated, check_exact=True)
    exported = export_factor({"tree": tree}, tmp_path, direction=1)
    spec = load_factor(exported.path)
    assert spec.setting["group_tie_policy"] == "symmetric_fractional"
    pd.testing.assert_frame_equal(spec.calc_factor({"close": close, "__eligible__": eligible}), full)
    actual_hashes = export_module._runtime_module_hashes()
    monkeypatch.setattr(export_module, "_runtime_module_hashes", lambda: {**actual_hashes, "changed": "hash"})
    with pytest.raises(ValueError, match="runtime/semantics mismatch"):
        spec.calc_factor({"close": close, "__eligible__": eligible})
    with pytest.raises(ValueError, match="runtime/semantics mismatch"):
        export_factor({"tree": tree}, tmp_path, direction=1)


def test_unversioned_old_export_is_rejected_by_loader(tmp_path):
    exported = export_factor({"tree": Node("close")}, tmp_path, direction=1)
    lines = exported.path.read_text().splitlines()
    source = "\n".join(line for line in lines
                       if not line.startswith("_GP_SEMANTICS_VERSION")
                       and not line.lstrip().startswith("verify_export_runtime("))
    exported.path.write_text(source)
    with pytest.raises(ValueError, match="GP semantics version mismatch"):
        load_factor(exported.path)


def test_buy_ratio_is_invariant_to_token_denomination():
    index = pd.date_range("2024-01-01", periods=10)
    rng = np.random.default_rng(91)
    volume = pd.DataFrame(rng.uniform(1, 100, (10, 7)), index=index)
    taker = volume * rng.uniform(.1, .9, volume.shape)
    eligible = volume.notna()
    tree = Node("taker_base_ratio")
    original = evaluate_tree(tree, {"volume": volume, "taker_buy_base_volume": taker}, eligible)
    unit_change = pd.Series([1, 10, 100, 2, 3, 7, 1000], index=volume.columns)
    changed = evaluate_tree(tree, {"volume": volume * unit_change, "taker_buy_base_volume": taker * unit_change}, eligible)
    np.testing.assert_allclose(original, changed, rtol=1e-12)
    pd.testing.assert_frame_equal(target_weights(original, profile())["long_short"],
                                  target_weights(changed, profile())["long_short"])


def test_training_and_replay_share_symmetric_ties(gp_h5, tmp_path):
    from Genetic_Algorithm.config import SearchConfig, Stage
    from Genetic_Algorithm.cost_fitness import evaluate_cost_window
    from Genetic_Algorithm.data import load_stage
    from Genetic_Algorithm.replay import replay
    from factor_common.reporting import _latest_groups_section

    stage = Stage("parity", "2024-04-01", "2024-04-12")
    tree = Node("subtract", (Node("close"), Node("close")))
    data = load_stage(gp_h5, stage, 0, ["close"], include_accounting=True)
    values = evaluate_tree(tree, data.features, data.eligible).loc[data.opens.index]
    cfg = SearchConfig(fitness_mode="all_costs_sharpe", n_groups=5, render_reports=False)
    metrics, returns = evaluate_cost_window(values, data, cfg, 1, stage.start, stage.end)
    result = replay({"tree": tree}, stage, gp_h5, tmp_path, direction=1, n_groups=5, render_reports=False)
    assert metrics["total_return"] == result["metrics"]["total_return"] == 0.
    assert returns.eq(0).all()
    assert result["profile"].group_tie_policy == "symmetric_fractional"
    html = _latest_groups_section({"factor_value": values,
                                  "metadata": {"n_groups": 5, "group_tie_policy": "symmetric_fractional"}})
    assert "weight " in html
