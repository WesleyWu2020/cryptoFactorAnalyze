"""Tests for exporting GP candidates as standalone regular factor modules."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import pytest

from factor_common.loader import load_factor
from Genetic_Algorithm import export as export_module
from Genetic_Algorithm.artifacts import read_verified_manifest
from Genetic_Algorithm.evaluator import evaluate_tree
from Genetic_Algorithm.evolution import Candidate
from Genetic_Algorithm.export import export_factor
from Genetic_Algorithm.expression import (
    Node,
    canonical_tree,
    expression_hash,
    history_days,
    node_count,
    required_fields,
)

NESTED_RANK_TREE = Node(
    "rank",
    (
        Node(
            "add",
            (
                Node("rank", (Node("rolling_mean", (Node("close"),), window=2),)),
                Node("open"),
            ),
        ),
    ),
)


def _ctx():
    dates = pd.date_range("2025-01-01", periods=6, freq="D", name="date")
    symbols = ["A", "B", "C", "D"]
    close = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 4.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.5, 2.5, 3.5, 4.5],
            [2.5, 1.5, 4.5, 3.5],
        ],
        index=dates,
        columns=symbols,
        dtype="float64",
    )
    open_ = pd.DataFrame(
        [
            [4.0, 3.0, 2.0, 1.0],
            [3.0, 4.0, 1.0, 2.0],
            [2.0, 1.0, 4.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.5, 3.5, 2.5, 1.5],
            [3.5, 4.5, 1.5, 2.5],
        ],
        index=dates,
        columns=symbols,
        dtype="float64",
    )
    return {"close": close, "open": open_}, dates, symbols


def _eligible(dates, symbols, blocked=()):
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    for day, instrument in blocked:
        eligible.loc[day, instrument] = False
    return eligible


def _candidate(tree=NESTED_RANK_TREE):
    canonical = canonical_tree(tree)
    return Candidate(
        canonical, expression_hash(canonical), (0.1, 0.05, -node_count(canonical))
    )


def test_export_round_trip_matches_evaluate_tree_under_changing_membership(tmp_path):
    candidate = _candidate()
    exported = export_factor(candidate, tmp_path, direction=1)

    assert exported.identifier == f"GP_{expression_hash(candidate.tree)[:16]}"
    assert exported.path == tmp_path / f"{exported.identifier}.py"
    assert exported.path.is_file()

    spec = load_factor(exported.path)
    assert spec.factor_id == exported.identifier
    assert spec.meta["factor_name"] == exported.identifier
    assert spec.meta["level"] == "daily"
    assert spec.setting["preprocessing"] == "none"
    assert spec.setting["universe"] == "historical_top50"
    assert spec.setting["context_eligible"] is True
    assert spec.setting["factor_direction"] == 1
    assert spec.setting["warmup_bars"] == history_days(candidate.tree)
    assert list(spec.setting["data_needed"]) == sorted(required_fields(candidate.tree))

    ctx, dates, symbols = _ctx()
    blocked_first = [(dates[1], "C"), (dates[3], "A")]
    eligible = _eligible(dates, symbols, blocked_first)
    loaded = spec.calc_factor({**ctx, "__eligible__": eligible})
    expected = evaluate_tree(candidate.tree, ctx, eligible)
    pd.testing.assert_frame_equal(loaded, expected)

    # Changing membership changes the cross-sectional ranks; the exported
    # formula must track evaluate_tree under the new mask exactly.
    changed = _eligible(dates, symbols, [(dates[1], "B"), (dates[4], "D")])
    loaded_changed = spec.calc_factor({**ctx, "__eligible__": changed})
    expected_changed = evaluate_tree(candidate.tree, ctx, changed)
    pd.testing.assert_frame_equal(loaded_changed, expected_changed)
    assert not loaded_changed.equals(loaded)


def test_export_is_idempotent_for_the_same_expression(tmp_path):
    candidate = _candidate()
    first = export_factor(candidate, tmp_path, direction=-1)
    content = first.path.read_bytes()
    second = export_factor(candidate, tmp_path, direction=-1)

    assert second.identifier == first.identifier
    assert second.path == first.path
    assert second.expression_hash == first.expression_hash
    assert first.path.read_bytes() == content


def test_export_reuse_rejects_direction_mismatch(tmp_path):
    """The expression hash covers only the tree: reuse must also match the
    baked-in direction, or a -1 caller would silently reuse a +1 module."""
    candidate = _candidate()
    exported = export_factor(candidate, tmp_path, direction=1)
    with pytest.raises(ValueError, match="direction"):
        export_factor(candidate, tmp_path, direction=-1)

    # The baked-in direction is unchanged and the manifest is intact.
    spec = load_factor(exported.path)
    assert spec.setting["factor_direction"] == 1
    manifest = read_verified_manifest(exported.manifest_path)
    assert manifest["factor_direction"] == 1


def test_export_refuses_overwrite_on_prefix_collision_with_different_hash(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        export_module, "_identifier", lambda full_hash: "GP_forcedcollision"
    )
    tree_a = Node("rank", (Node("close"),))
    tree_b = Node("rank", (Node("open"),))
    assert expression_hash(tree_a) != expression_hash(tree_b)

    export_factor(_candidate(tree_a), tmp_path, direction=1)
    with pytest.raises(FileExistsError, match="collision"):
        export_factor(_candidate(tree_b), tmp_path, direction=1)


def test_export_refuses_to_overwrite_a_foreign_file(tmp_path):
    candidate = _candidate()
    exported = export_factor(candidate, tmp_path, direction=1)
    exported.path.write_text("# foreign content without a hash marker\n")

    with pytest.raises(FileExistsError, match="overwrite"):
        export_factor(candidate, tmp_path, direction=1)


def test_export_rejects_expression_id_mismatch(tmp_path):
    candidate = Candidate(NESTED_RANK_TREE, "bogus-id", (0.1, 0.05, -5))
    with pytest.raises(ValueError, match="expression_id"):
        export_factor(candidate, tmp_path, direction=1)


@pytest.mark.parametrize("direction", [0, 2, -2, True, 1.0])
def test_export_rejects_invalid_direction(tmp_path, direction):
    with pytest.raises((TypeError, ValueError), match="direction"):
        export_factor(_candidate(), tmp_path, direction=direction)


def test_export_manifest_hashes_transitive_runtime_modules(tmp_path):
    candidate = _candidate()
    exported = export_factor(candidate, tmp_path, direction=1)

    manifest_path = tmp_path / f"{exported.identifier}.export_manifest.json"
    assert exported.manifest_path == manifest_path
    manifest = read_verified_manifest(manifest_path)
    assert manifest["identifier"] == exported.identifier
    assert manifest["expression_hash"] == expression_hash(candidate.tree)
    assert manifest["factor_direction"] == 1
    assert manifest["data_needed"] == sorted(required_fields(candidate.tree))
    assert manifest["warmup_bars"] == history_days(candidate.tree)
    assert manifest["wrapper_sha256"] == hashlib.sha256(
        exported.path.read_bytes()
    ).hexdigest()

    runtime = manifest["runtime_module_hashes"]
    package_dir = Path(export_module.__file__).resolve().parent
    expected_modules = {
        f"Genetic_Algorithm/{name}.py"
        for name in ("expression", "evaluator", "operators", "features")
    }
    assert expected_modules <= set(runtime)
    for relative in expected_modules:
        digest = runtime[relative]
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        actual = hashlib.sha256(
            (package_dir / relative.split("/", 1)[1]).read_bytes()
        ).hexdigest()
        assert digest == actual


def test_exported_module_rejects_tampered_ast_or_hash(tmp_path):
    candidate = _candidate()
    exported = export_factor(candidate, tmp_path, direction=1)
    source = exported.path.read_text(encoding="utf-8")
    marker = f'_EXPRESSION_HASH = "{expression_hash(candidate.tree)}"'
    assert marker in source
    tampered = source.replace(marker, '_EXPRESSION_HASH = "' + "0" * 64 + '"')
    exported.path.write_text(tampered, encoding="utf-8")

    spec = load_factor(exported.path)
    ctx, dates, symbols = _ctx()
    eligible = _eligible(dates, symbols)
    with pytest.raises(ValueError, match="hash"):
        spec.calc_factor({**ctx, "__eligible__": eligible})


def test_exported_module_passes_static_future_leak_scan(tmp_path):
    from factor_common.validation import scan_future_leaks

    exported = export_factor(_candidate(), tmp_path, direction=1)
    assert scan_future_leaks([exported.path]) == []
