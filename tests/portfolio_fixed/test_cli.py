from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from portfolio.main_fixed import main


def _write_config(config_dict: dict, tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict), encoding="utf-8")
    return path


def test_run_cli_writes_report_and_returns_zero(config_dict, tmp_path):
    path = _write_config(config_dict, tmp_path)
    output = tmp_path / "reports"
    assert main(["run", "--config", str(path), "--output-root", str(output)]) == 0
    assert len(list(output.glob("*/report.html"))) == 1


def test_run_cli_incomplete_returns_two(config_dict, tmp_path):
    config_dict["as_of"] = "2024-01-24"
    path = _write_config(config_dict, tmp_path)
    output = tmp_path / "reports"
    assert main(["run", "--config", str(path), "--output-root", str(output)]) == 2
    assert len(list(output.glob("*/report.html"))) == 1


def test_freeze_does_not_overwrite_existing_config(fixed_h5, factor_paths, tmp_path):
    path = tmp_path / "config.json"
    args = [
        "freeze", "--h5", str(fixed_h5), "--factor", str(factor_paths[0]),
        "--allocation", "1", "--start", "2024-01-22", "--end", "2024-01-24",
        "--as-of", "2024-01-26", "--output", str(path),
    ]
    assert main(args) == 0
    original = path.read_bytes()
    assert main(args) == 1
    assert path.read_bytes() == original


def test_audit_cli_writes_verified_receipt(config_dict, tmp_path):
    path = _write_config(config_dict, tmp_path)
    receipt = tmp_path / "audit.json"
    assert main([
        "audit", "--config", str(path), "--cutoff", "2024-01-23",
        "--output", str(receipt),
    ]) == 0
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "verified"


def test_run_cli_works_in_fresh_interpreter(config_dict, tmp_path):
    path = _write_config(config_dict, tmp_path)
    output = tmp_path / "subprocess-reports"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", "-m", "portfolio.main_fixed", "run",
         "--config", str(path), "--output-root", str(output)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert len(list(output.glob("*/report.html"))) == 1
