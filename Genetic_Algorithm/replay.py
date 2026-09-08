"""Replay exported GP candidates over bounded stages via FactorManager.

A replay exports the candidate's wrapper under the run directory, then runs
the standard ``FactorManager`` evaluation with the manager knowledge cutoff
pinned at ``stage.end`` and the signal window ``[stage.start,
stage.signal_end]`` — the two-day execution tail lands exactly on the stage
end, so no read can cross the stage boundary. The replay profile is the
resolved ``perp_1d`` profile with ``rebalance_days=1``, ``n_groups=10``, the
fixed training direction, fees 0.0005, slippage 0.001, strict funding, and
unit gross exposure; the framework's 50/50 long-short weights and accounting
are used unchanged.

Stage-wide ``all_costs`` metrics are extracted from the full certified ledger
via the framework's existing summary helpers. The framework's automatic
in-sample/out-of-sample split is disclosed in the replay artifact but never
used as the GP stage selector. A non-complete ``all_costs`` accounting is a
failed evaluation (``ReplayEvaluationError``), never a zero return.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from factor_common.manager import FactorManager
from factor_common.profiles import resolve_profile

from .artifacts import write_artifact
from .config import Stage
from .export import export_factor


class ReplayEvaluationError(RuntimeError):
    """A stage replay failed to produce a certified complete evaluation."""


_REPLAY_OVERRIDES = {
    "rebalance_days": 1,
    "n_groups": 10,
    "fee_rate": 0.0005,
    "slippage": 0.001,
    "include_funding": True,
}

_SPLIT_DISCLOSURE = (
    "the framework report's in_sample/out_of_sample blocks come from its "
    "automatic internal split (split_date / out_of_sample_days); they are "
    "disclosed for completeness and are not the train/validation/test "
    "research split — the GP stage metrics use the full stage-wide "
    "all_costs ledger only"
)


def _iso(day: Any) -> str:
    return pd.Timestamp(day).date().isoformat()


def replay(
    candidate: Any,
    stage: Stage,
    h5_path: str | Path,
    run_dir: str | Path,
    *,
    direction: int,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Replay one candidate on one stage and return the replay outcome.

    ``direction`` is the fixed training direction (1 or -1), applied to both
    the exported module and the resolved profile without re-derivation.
    """
    if not isinstance(stage, Stage):
        raise TypeError("stage must be a Genetic_Algorithm.config.Stage")
    run_dir = Path(run_dir)
    exported = export_factor(
        candidate, run_dir / "exported_factors", direction=direction
    )
    overrides = {**_REPLAY_OVERRIDES, "factor_direction": direction}
    profile = resolve_profile("perp_1d", overrides)
    if profile.gross_exposure != 1.0 or profile.funding_price_mode != "strict":
        raise ValueError(
            "replay requires strict funding and unit gross exposure from the "
            "resolved profile"
        )

    manager = FactorManager(
        h5_path=h5_path,
        base_dir=run_dir / "factor_results",
        reports_dir=Path(reports_dir) if reports_dir is not None else run_dir / "reports",
        persist_evaluations=True,
        as_of=stage.end,
    )
    result = manager.evaluate(
        str(exported.path),
        params={
            "start": stage.start,
            "end": stage.signal_end,
            **overrides,
        },
        plot=False,
    )

    all_costs = result["factor_result"]["scenarios"]["all_costs"]
    diagnostics = all_costs["diagnostics"]
    if all_costs["status"] != "complete":
        raise ReplayEvaluationError(
            f"stage {stage.name!r} replay of {exported.identifier} is not "
            f"certified: all_costs status={all_costs['status']!r} "
            f"halt_reason={diagnostics.get('halt_reason')!r}"
        )
    ledger = all_costs["ledger"]
    if ledger.empty:
        raise ReplayEvaluationError(
            f"stage {stage.name!r} replay of {exported.identifier} produced "
            "no usable signals; an empty ledger is a failed evaluation, not "
            "a zero return"
        )
    liquidation = pd.Timestamp(diagnostics["liquidation_date"])
    if not diagnostics["liquidation_reached"] or liquidation > stage.end:
        raise ReplayEvaluationError(
            f"stage {stage.name!r} replay liquidation "
            f"{diagnostics['liquidation_date']} exceeds the stage end "
            f"{_iso(stage.end)}"
        )
    if diagnostics["final_quantities"]:
        raise ReplayEvaluationError(
            f"stage {stage.name!r} replay retains unliquidated positions: "
            f"{sorted(diagnostics['final_quantities'])}"
        )
    positions = all_costs["positions"]
    if not positions.empty and bool((positions.iloc[-1] != 0.0).any()):
        raise ReplayEvaluationError(
            f"stage {stage.name!r} replay final positions are not zero"
        )

    # Stage-wide metrics: the framework's "full" sample summarizes the whole
    # certified ledger; its internal IS/OOS split is never consulted here.
    metrics = result["factor_performance"]["scenarios"]["all_costs"]["full"]
    if metrics is None:
        raise ReplayEvaluationError(
            f"stage {stage.name!r} replay of {exported.identifier} has no "
            "stage-wide all_costs metrics"
        )

    report_name = f"{exported.identifier}_{stage.name}_{result['run_id']}.html"
    report = manager.plot_result(
        result, output_path=manager.reports_dir / report_name
    )

    artifact_path = write_artifact(
        run_dir / f"replay_{stage.name}.json",
        {
            "replay_version": 1,
            "stage": {
                "name": stage.name,
                "start": _iso(stage.start),
                "end": _iso(stage.end),
                "signal_end": _iso(stage.signal_end),
            },
            "identifier": exported.identifier,
            "expression_hash": exported.expression_hash,
            "factor_direction": direction,
            "profile": asdict(profile),
            "run_id": result["run_id"],
            "evaluation_id": result["evaluation_id"],
            "h5_path": str(h5_path),
            "exported_factor_path": str(exported.path),
            "export_manifest_path": str(exported.manifest_path),
            "report_path": report["output_path"],
            "factor_path": result["paths"]["factor_path"],
            "metadata_path": result["paths"]["metadata_path"],
            "evaluation_dir": result["paths"]["evaluation_dir"],
            "all_costs_metrics": metrics,
            "framework_split": result["factor_performance"]["split"],
            "framework_split_disclosure": _SPLIT_DISCLOSURE,
            "liquidation_date": diagnostics["liquidation_date"],
            "manager_metadata": result["metadata"],
        },
    )
    return {
        "stage": stage,
        "identifier": exported.identifier,
        "expression_id": exported.expression_hash,
        "direction": direction,
        "profile": profile,
        "metrics": metrics,
        "result": result,
        "export": exported,
        "run_id": result["run_id"],
        "evaluation_id": result["evaluation_id"],
        "report_path": report["output_path"],
        "artifact_path": str(artifact_path),
    }


__all__ = ["ReplayEvaluationError", "replay"]
