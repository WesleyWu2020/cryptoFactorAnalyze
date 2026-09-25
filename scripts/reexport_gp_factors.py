"""Re-export the 9 GP factor artifacts under the current runtime semantics.

Context: the uncommitted runtime upgrade added GP_SEMANTICS_VERSION=2 and a
loader gate (factor_common/loader.py) that rejects legacy export_version=1
wrappers, which lack _GP_SEMANTICS_VERSION. The error message's sanctioned
fix is "re-export under current semantics". Exports are immutable, so the
legacy artifacts are first moved to a backup directory, then re-exported
from the AST recorded in each legacy manifest (the AST is the complete
formula; canonical hashing yields the same identifier).

Usage: ./.venv/bin/python scripts/reexport_gp_factors.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Genetic_Algorithm.export import export_factor  # noqa: E402

GP_DIR = PROJECT_ROOT / "factor_analyse" / "factor_mining" / "GP_Factor"
BACKUP_DIR = PROJECT_ROOT / "tmp" / "gp_export_v1_backup_20260924"


def main() -> int:
    manifests = sorted(GP_DIR.glob("GP_*.export_manifest.json"))
    if not manifests:
        raise SystemExit("no GP export manifests found")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identifier = manifest["identifier"]
        wrapper_path = GP_DIR / f"{identifier}.py"
        if manifest.get("export_version") != 1:
            print(f"SKIP {identifier}: export_version={manifest.get('export_version')}")
            continue

        # Move the legacy pair aside; exports are immutable, never overwritten.
        shutil.move(str(wrapper_path), str(BACKUP_DIR / wrapper_path.name))
        shutil.move(str(manifest_path), str(BACKUP_DIR / manifest_path.name))

        result = export_factor(
            {"ast": manifest["ast"]},
            GP_DIR,
            direction=int(manifest["factor_direction"]),
        )
        assert result.identifier == identifier, (
            f"identifier drift: {result.identifier} != {identifier}"
        )
        assert result.expression_hash == manifest["expression_hash"]
        results.append((identifier, result.path))
        print(f"RE-EXPORTED {identifier} -> {result.path.name}")

    print(f"\nbackup: {BACKUP_DIR}")
    print(f"re-exported: {len(results)} factor(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
