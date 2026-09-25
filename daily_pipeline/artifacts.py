"""Atomic daily-run artifacts and idempotent signal publication."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _comparable_signal(value: dict) -> dict:
    return {key: item for key, item in value.items()
            if key not in {"run_id", "generated_at"}}


def write_run(result: dict, output_root: str | Path) -> Path:
    root = Path(output_root)
    signal_date = result["signal"]["signal_date"]
    day_root = root / "runs" / signal_date
    day_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    staged = day_root / f".partial_{run_id}"
    final = day_root / run_id
    staged.mkdir()
    manifest = {**result["manifest"], "run_id": run_id, "artifact_complete": True}
    _json(staged / "manifest.json", manifest)
    _json(staged / "readiness.json", result["readiness"])
    _json(staged / "member_diagnostics.json", result["member_diagnostics"])
    _json(staged / "signal.json", {**result["signal"], "run_id": run_id})
    result["factor_values"].to_parquet(staged / "factor_values.parquet", index=False)
    result["factor_groups"].to_parquet(staged / "factor_groups.parquet", index=False)
    result["member_targets"].to_parquet(staged / "member_targets.parquet", index=False)
    result["combined_targets"].to_parquet(staged / "combined_targets.parquet", index=False)
    result["risk"].to_parquet(staged / "risk.parquet")
    _json(staged / "artifact_complete.json", {"run_id": run_id, "written": True})
    os.rename(staged, final)
    return final


def write_failure(output_root: str | Path, signal_date: str, exc: Exception) -> Path:
    """Persist a diagnostic failure without publishing any trading instruction."""
    root = Path(output_root) / "runs" / signal_date
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    staged = root / f".partial_{run_id}"
    final = root / run_id
    staged.mkdir()
    _json(staged / "manifest.json", {
        "run_id": run_id,
        "status": "failed",
        "signal_date": signal_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "artifact_complete": True,
    })
    _json(staged / "artifact_complete.json", {"run_id": run_id, "written": True})
    os.rename(staged, final)
    return final


def publish_signal(run_dir: str | Path, output_root: str | Path) -> Path:
    run_dir = Path(run_dir)
    signal = json.loads((run_dir / "signal.json").read_text(encoding="utf-8"))
    published = Path(output_root) / "published"
    published.mkdir(parents=True, exist_ok=True)
    immutable = published / f"{signal['signal_id']}.json"
    payload = json.dumps(signal, ensure_ascii=False, indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
    if immutable.exists():
        existing = json.loads(immutable.read_text(encoding="utf-8"))
        if _comparable_signal(existing) != _comparable_signal(signal):
            raise RuntimeError(f"signal id collision: {signal['signal_id']}")
    else:
        immutable.write_text(payload, encoding="utf-8")
    temporary = published / f".latest.{uuid.uuid4().hex}.tmp"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, published / "latest.json")
    return immutable
