from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "portfolio" / "run_fixed_portfolio.sh"


def _fake_python(tmp_path: Path) -> Path:
    executable = tmp_path / "python"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$PORTFOLIO_TEST_LOG\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_runner_builds_equal_weight_freeze_run_and_audit_commands(tmp_path):
    factors = tmp_path / "factors.txt"
    factors.write_text(
        "# selected factors\n"
        "example_momentum.py\n"
        "Price_Momentum_60d.py\n",
        encoding="utf-8",
    )
    log = tmp_path / "commands.log"
    env = os.environ | {
        "PORTFOLIO_PYTHON": str(_fake_python(tmp_path)),
        "PORTFOLIO_TEST_LOG": str(log),
    }

    result = subprocess.run(
        [
            "bash", str(SCRIPT),
            "--factors", str(factors),
            "--start", "2025-01-01",
            "--end", "2025-03-31",
            "--as-of", "2025-04-05",
            "--rebalance-days", "7",
            "--cutoff", "2025-02-15",
            "--output-root", str(tmp_path / "reports"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = log.read_text(encoding="utf-8").splitlines()
    assert len(commands) == 3
    assert commands[0].startswith("-m portfolio.main_fixed freeze ")
    assert commands[0].count("--factor ") == 2
    assert commands[0].count("--allocation 0.5") == 2
    assert commands[1].startswith("-m portfolio.main_fixed run ")
    assert commands[2].startswith("-m portfolio.main_fixed audit ")
    assert "--cutoff 2025-02-15" in commands[2]


def test_runner_rejects_unsafe_factor_path(tmp_path):
    factors = tmp_path / "factors.txt"
    factors.write_text("../outside.py\n", encoding="utf-8")
    log = tmp_path / "commands.log"
    env = os.environ | {
        "PORTFOLIO_PYTHON": str(_fake_python(tmp_path)),
        "PORTFOLIO_TEST_LOG": str(log),
    }

    result = subprocess.run(
        [
            "bash", str(SCRIPT),
            "--factors", str(factors),
            "--start", "2025-01-01",
            "--end", "2025-03-31",
            "--as-of", "2025-04-05",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsafe factor path" in result.stderr
    assert not log.exists()


def test_runner_reports_missing_option_value_cleanly():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--start"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "missing value for --start" in result.stderr
    assert "unbound variable" not in result.stderr
