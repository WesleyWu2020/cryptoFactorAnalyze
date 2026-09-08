import json

import pytest

from Genetic_Algorithm.artifacts import freeze_candidates, read_verified_manifest, write_artifact
from Genetic_Algorithm.config import SearchConfig, Stage
from Genetic_Algorithm.expression import Node
from Genetic_Algorithm.replay import require_test_access, test_manifest
from Genetic_Algorithm import search as search_module
from Genetic_Algorithm.selection import select_validation


def _candidate(identifier="candidate", direction=1):
    return {
        "expression_id": identifier,
        "tree": Node("field", field="close"),
        "direction": direction,
        "complexity": 2,
    }


def _validation(**overrides):
    result = {
        "direction": 1,
        "day_coverage": 0.9,
        "cell_coverage": 0.9,
        "quarter_valid_days": {"Q1": 45, "Q2": 45, "Q3": 45, "Q4": 45},
        "mean_ic": 0.02,
        "quarter_means": {"Q1": 0.01, "Q2": 0.02, "Q3": 0.01, "Q4": -0.01},
        "all_costs_cumulative_return": 0.1,
        "net_sharpe": 1.2,
        "turnover": 0.3,
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"all_costs_cumulative_return": -0.01}, "cumulative return"),
        ({"net_sharpe": None}, "Sharpe"),
        ({"net_sharpe": 1.0}, "Sharpe"),
        ({"direction": -1}, "direction"),
    ],
)
def test_validation_selection_rejects_invalid_net_metrics_and_direction(update, reason):
    result = select_validation([_candidate()], {"candidate": _validation(**update)}, SearchConfig())
    assert result.accepted == ()
    assert reason in result.rejection_reasons["candidate"][0]


def test_validation_selection_allows_zero_candidates():
    result = select_validation([], {}, SearchConfig())
    assert result.accepted == ()
    assert result.rejected == ()


def test_validation_selection_rejects_missing_fourth_calendar_quarter():
    result = select_validation(
        [_candidate()], {"candidate": _validation(quarter_valid_days={"Q1": 45, "Q2": 45, "Q3": 45})}, SearchConfig()
    )
    assert result.accepted == ()
    assert "Q4" in " ".join(result.rejection_reasons["candidate"])


def test_validation_selection_rejects_candidate_without_frozen_training_direction():
    candidate = _candidate()
    candidate.pop("direction")
    result = select_validation([candidate], {"candidate": _validation()}, SearchConfig())
    assert result.accepted == ()
    assert "training direction" in " ".join(result.rejection_reasons["candidate"])


def test_nonempty_validation_input_can_validly_select_zero_with_duplicate_rejection_evidence():
    candidates = [_candidate("duplicate"), _candidate("duplicate")]
    result = select_validation(candidates, {"duplicate": _validation(net_sharpe=None)}, SearchConfig())
    assert result.accepted == ()
    assert len(result.rejected) == 2
    assert len(result.rejection_reasons["duplicate"]) == 2


def test_validation_selection_honors_configured_coverage_and_quarter_thresholds():
    result = select_validation([_candidate()], {"candidate": _validation(day_coverage=0.81, cell_coverage=0.81,
        quarter_valid_days={"Q1": 50, "Q2": 50, "Q3": 50, "Q4": 50})},
        {"min_day_coverage": 0.9, "min_cell_coverage": 0.9, "min_quarter_days": 60, "frozen_limit": 5})
    assert result.accepted == ()
    assert len(result.rejection_reasons["candidate"]) == 6


def test_malformed_validation_nested_values_are_rejected_not_raised():
    result = select_validation([_candidate()], {"candidate": _validation(quarter_valid_days=None, quarter_means=None)}, SearchConfig())
    assert result.accepted == ()
    assert "quarter" in " ".join(result.rejection_reasons["candidate"])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("candidates", 0, "ast", "field"), "open"),
        (("config", "seed"), 7),
        (("candidates", 0, "direction"), -1),
        (("runtime_source_hashes", "selection.py"), "changed-source"),
    ],
)
def test_frozen_manifest_mutation_prevents_test_evaluation(tmp_path, path, value):
    frozen = freeze_candidates(
        [_candidate()],
        training_fingerprint="train",
        validation_fingerprint="validation",
        config={"seed": 42},
        profile={"factor_direction": 1},
        runtime_source_hashes={"selection.py": "source"},
        selection_results={"accepted": ["candidate"]},
        path=tmp_path / "frozen_manifest.json",
    )
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    frozen.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash verification"):
        test_manifest(
            frozen,
            test_stage=Stage("test", "2026-01-01", "2026-01-10"),
            run_dir=tmp_path,
            training_fingerprint="train",
            validation_fingerprint="validation",
            load_test_data=lambda stage: {"fingerprint": "test", "history_start": stage.start, "end": stage.end},
            evaluate=lambda candidate, stage, data, direction: {"ok": True},
            runtime_source_hashes=lambda: {"selection.py": "source"},
            permitted_history_start="2026-01-01",
        )


def _frozen(tmp_path):
    return freeze_candidates(
        [_candidate("good"), _candidate("bad")], training_fingerprint="train", validation_fingerprint="validation",
        config={"seed": 42}, profile={"factor_direction": 1}, runtime_source_hashes={"runtime": "old"},
        selection_results={"accepted": ["good", "bad"]}, path=tmp_path / "frozen_manifest.json",
    )


def test_freeze_rejects_candidate_set_that_does_not_match_selected_ids(tmp_path):
    with pytest.raises(ValueError, match="selection"):
        freeze_candidates([_candidate("actual")], training_fingerprint="train", validation_fingerprint="validation",
                          config={}, profile={}, runtime_source_hashes={}, selection_results={"accepted": ["other"]},
                          path=tmp_path / "frozen_manifest.json")


def test_freeze_uses_training_direction_and_rejects_conflicting_direction_fields(tmp_path):
    candidate = _candidate()
    candidate["training_direction"] = -1
    candidate["direction"] = 1
    with pytest.raises(ValueError, match="conflicting"):
        freeze_candidates([candidate], training_fingerprint="train", validation_fingerprint="validation",
                          config={}, profile={}, runtime_source_hashes={}, selection_results={"accepted": ["candidate"]},
                          path=tmp_path / "frozen_manifest.json")


def _data(stage):
    return {"fingerprint": "test", "history_start": "2025-12-27", "end": stage.end.isoformat()}


def test_runtime_source_change_blocks_test_evaluation(tmp_path):
    with pytest.raises(ValueError, match="runtime source"):
        test_manifest(_frozen(tmp_path), test_stage=Stage("test", "2026-01-01", "2026-01-10"), run_dir=tmp_path,
                      training_fingerprint="train", validation_fingerprint="validation", load_test_data=_data,
                      evaluate=lambda *args: {}, runtime_source_hashes=lambda: {"runtime": "new"},
                      permitted_history_start="2025-12-27")


def test_test_manifest_enforces_loader_bounds(tmp_path):
    with pytest.raises(ValueError, match="bounded"):
        test_manifest(_frozen(tmp_path), test_stage=Stage("test", "2026-01-01", "2026-01-10"), run_dir=tmp_path,
                      training_fingerprint="train", validation_fingerprint="validation",
                      load_test_data=lambda stage: {"fingerprint": "test", "history_start": "2025-12-26", "end": stage.end},
                      evaluate=lambda *args: {}, runtime_source_hashes=lambda: {"runtime": "old"},
                      permitted_history_start="2025-12-27")


def test_test_manifest_retains_failures_marks_repeat_and_records_access(tmp_path):
    manifest = _frozen(tmp_path)
    kwargs = dict(test_stage=Stage("test", "2026-01-01", "2026-01-10"), run_dir=tmp_path,
                  training_fingerprint="train", validation_fingerprint="validation", load_test_data=_data,
                  evaluate=lambda candidate, *_: (_ for _ in ()).throw(RuntimeError("bad")) if candidate["expression_id"] == "bad" else {"ok": True},
                  runtime_source_hashes=lambda: {"runtime": "old"}, permitted_history_start="2025-12-27")
    first = test_manifest(manifest, **kwargs)
    second = test_manifest(manifest, **kwargs)
    assert first != second
    assert len(json.loads(first.read_text())['outcomes']) == 2
    assert json.loads(first.read_text())['outcomes'][0]['status'] == 'complete'
    assert json.loads(first.read_text())['outcomes'][1]['status'] == 'failed'
    assert json.loads(second.read_text())['repeat'] is True
    assert require_test_access(tmp_path, manifest).parent == tmp_path / "test_receipts"


def test_test_access_commitment_survives_loader_failure_and_ignores_alternate_ledger(tmp_path):
    manifest = _frozen(tmp_path)
    with pytest.raises(RuntimeError, match="loader failed"):
        test_manifest(manifest, test_stage=Stage("test", "2026-01-01", "2026-01-10"), run_dir=tmp_path / "untrusted",
                      training_fingerprint="train", validation_fingerprint="validation",
                      load_test_data=lambda _: (_ for _ in ()).throw(RuntimeError("loader failed")), evaluate=lambda *_: {},
                      runtime_source_hashes=lambda: {"runtime": "old"}, permitted_history_start="2025-12-27")
    assert require_test_access(tmp_path / "untrusted", manifest).parent == tmp_path


def test_repeat_is_derived_from_immutable_commitment_after_loader_failure_and_log_deletion(tmp_path):
    manifest = _frozen(tmp_path)
    kwargs = dict(test_stage=Stage("test", "2026-01-01", "2026-01-10"), run_dir=tmp_path,
                  training_fingerprint="train", validation_fingerprint="validation", runtime_source_hashes=lambda: {"runtime": "old"},
                  permitted_history_start="2025-12-27")
    with pytest.raises(RuntimeError):
        test_manifest(manifest, load_test_data=lambda _: (_ for _ in ()).throw(RuntimeError()), evaluate=lambda *_: {}, **kwargs)
    (tmp_path / "test_access_log.json").unlink(missing_ok=True)
    receipt = test_manifest(manifest, load_test_data=_data, evaluate=lambda *_: {}, **kwargs)
    assert json.loads(receipt.read_text())["repeat"] is True


def test_new_search_cannot_claim_unseen_holdout_after_a_test_receipt(tmp_path):
    manifest = _frozen(tmp_path)
    test_manifest(manifest, test_stage=Stage("test", "2026-01-01", "2026-01-10"), run_dir=tmp_path,
                  training_fingerprint="train", validation_fingerprint="validation", load_test_data=_data,
                  evaluate=lambda *_: {}, runtime_source_hashes=lambda: {"runtime": "old"},
                  permitted_history_start="2025-12-27")
    with pytest.raises(ValueError, match="unseen holdout"):
        search_module.run_search(
            "not-read.h5", tmp_path / "audit.json", stage=Stage("train", "2024-01-01", "2024-01-02"),
            warmup_days=0, fields=(), search_stage=lambda _: pytest.fail("search must not run"),
            artifact_dir=tmp_path / "next", holdout_manifest=manifest, claim_unseen_holdout=True,
        )


def test_nested_test_receipt_is_recognized_as_holdout_access(tmp_path):
    manifest = _frozen(tmp_path)
    frozen = read_verified_manifest(manifest)
    commitment = write_artifact(tmp_path / "test_access_commitment_nested.json", {
        "commitment_version": 1,
        "manifest_sha256": frozen["sha256"],
    }, immutable=True)
    commitment_document = read_verified_manifest(commitment)
    receipt = tmp_path / "test_receipts" / "receipt.json"
    write_artifact(receipt, {
        "receipt_version": 1,
        "manifest_sha256": frozen["sha256"],
        "access_commitment_sha256": commitment_document["sha256"],
    }, immutable=True)

    assert require_test_access(tmp_path, manifest) == receipt


def test_nested_receipt_with_unlinked_commitment_is_not_holdout_access(tmp_path):
    manifest = _frozen(tmp_path)
    frozen = read_verified_manifest(manifest)
    write_artifact(tmp_path / "test_receipts" / "receipt.json", {
        "receipt_version": 1,
        "manifest_sha256": frozen["sha256"],
        "access_commitment_sha256": "not-a-real-commitment",
    }, immutable=True)

    with pytest.raises(ValueError, match="no recorded holdout access"):
        require_test_access(tmp_path, manifest)
