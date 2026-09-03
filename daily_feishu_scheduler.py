#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests


TZ = ZoneInfo("Asia/Shanghai")
ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "logs"
ENV_FILE = ROOT_DIR / ".env"
DEFAULT_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
CAPTURE_SCRIPT = ROOT_DIR / "data" / "capture_data_binance.py"
FACTOR_SCRIPT = (
    ROOT_DIR
    / "factor_analyse"
    / "factor_mining"
    / "Retail_Friction_Illiquidity_Factor.py"
)


@dataclass
class StepResult:
    script: Path
    ok: bool
    returncode: int
    duration_sec: float
    stdout: str
    stderr: str


def load_env_file(path: Path, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if (not override) and (key in os.environ):
            continue
        os.environ[key] = value


def now_bjt() -> datetime:
    return datetime.now(TZ)


def format_bjt(dt: Optional[datetime] = None) -> str:
    current = dt or now_bjt()
    return current.strftime("%Y-%m-%d %H:%M:%S %Z")


def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    ensure_log_dir()
    line = f"[{format_bjt()}] {msg}"
    print(line, flush=True)
    log_path = LOG_DIR / "daily_feishu_scheduler.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_python_script(script: Path, python_bin: Path) -> StepResult:
    started = time.monotonic()
    child_env = os.environ.copy()
    # 默认保留代理变量；仅当 CLEAR_PROXY=1 时才移除。
    if child_env.get("CLEAR_PROXY", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        for k in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            child_env.pop(k, None)
    proc = subprocess.run(
        [str(python_bin), str(script)],
        cwd=str(script.parent),
        env=child_env,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started
    return StepResult(
        script=script,
        ok=(proc.returncode == 0),
        returncode=proc.returncode,
        duration_sec=elapsed,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def tail_lines(text: str, n: int = 80) -> str:
    lines = text.strip().splitlines()
    if len(lines) <= n:
        return "\n".join(lines)
    return "\n".join(lines[-n:])


def extract_latest_group_block(stdout_text: str) -> Optional[str]:
    marker = "==== 最新日期推理 | 日期:"
    idx = stdout_text.rfind(marker)
    if idx < 0:
        return None

    section = stdout_text[idx:]
    if not section.strip():
        return None

    end_candidates = [
        "\nℹ️ Live推理日期范围:",
        "\nLive推理日期范围:",
    ]
    end_idx = len(section)
    for token in end_candidates:
        cur = section.find(token)
        if cur >= 0:
            end_idx = min(end_idx, cur)
    block = section[:end_idx].strip()
    if not block:
        return None
    return block


def build_feishu_sign(bot_secret: str, timestamp: str) -> str:
    string_to_sign = f"{timestamp}\n{bot_secret}"
    hmac_code = hmac.new(
        bot_secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    return base64.b64encode(hmac_code).decode("utf-8")


def post_to_feishu(webhook_url: str, text: str, bot_secret: Optional[str] = None) -> None:
    payload = {"msg_type": "text", "content": {"text": text}}
    if bot_secret:
        ts = str(int(time.time()))
        payload["timestamp"] = ts
        payload["sign"] = build_feishu_sign(bot_secret, ts)
    resp = requests.post(webhook_url, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    code = data.get("code", 0)
    if code != 0:
        raise RuntimeError(f"Feishu API error: {data}")


def build_failure_message(
    trigger_time: datetime, step_name: str, result: StepResult
) -> str:
    stderr_tail = tail_lines(result.stderr, n=60) or "(stderr empty)"
    stdout_tail = tail_lines(result.stdout, n=40) or "(stdout empty)"
    return (
        f"[Crypto Daily 08:00 Failed]\n"
        f"Trigger(BJT): {format_bjt(trigger_time)}\n"
        f"Step: {step_name}\n"
        f"Script: {result.script}\n"
        f"Return code: {result.returncode}\n"
        f"Duration: {result.duration_sec:.1f}s\n\n"
        f"stderr tail:\n{stderr_tail}\n\n"
        f"stdout tail:\n{stdout_tail}"
    )


def build_success_message(
    trigger_time: datetime,
    capture_result: StepResult,
    factor_result: StepResult,
    group_block: Optional[str],
) -> str:
    group_part = group_block or "Latest grouping block not found in factor script output."
    return (
        f"[Crypto Daily 08:00 OK]\n"
        f"Trigger(BJT): {format_bjt(trigger_time)}\n"
        f"capture_data_binance.py: {capture_result.duration_sec:.1f}s\n"
        f"Retail_Friction_Illiquidity_Factor.py: {factor_result.duration_sec:.1f}s\n\n"
        f"{group_part}"
    )


def run_once(
    python_bin: Path,
    webhook_url: Optional[str],
    bot_secret: Optional[str],
    dry_run: bool,
) -> bool:
    trigger_time = now_bjt()
    log("Run started")

    capture_result = run_python_script(CAPTURE_SCRIPT, python_bin)
    log(
        f"capture_data_binance.py finished: ok={capture_result.ok}, "
        f"code={capture_result.returncode}, elapsed={capture_result.duration_sec:.1f}s"
    )
    if not capture_result.ok:
        message = build_failure_message(trigger_time, "capture_data_binance.py", capture_result)
        if dry_run:
            log("Dry run enabled; Feishu message not sent.")
            log(message)
        else:
            post_to_feishu(webhook_url, message, bot_secret=bot_secret)
        return False

    factor_result = run_python_script(FACTOR_SCRIPT, python_bin)
    log(
        f"Retail_Friction_Illiquidity_Factor.py finished: ok={factor_result.ok}, "
        f"code={factor_result.returncode}, elapsed={factor_result.duration_sec:.1f}s"
    )
    if not factor_result.ok:
        message = build_failure_message(
            trigger_time, "Retail_Friction_Illiquidity_Factor.py", factor_result
        )
        if dry_run:
            log("Dry run enabled; Feishu message not sent.")
            log(message)
        else:
            post_to_feishu(webhook_url, message, bot_secret=bot_secret)
        return False

    group_block = extract_latest_group_block(factor_result.stdout)
    message = build_success_message(trigger_time, capture_result, factor_result, group_block)
    if dry_run:
        log("Dry run enabled; Feishu message not sent.")
        log(message)
    else:
        post_to_feishu(webhook_url, message, bot_secret=bot_secret)
        log("Feishu success message sent.")
    return True


def seconds_to_next_run(hour: int, minute: int) -> tuple[int, datetime]:
    now = now_bjt()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    wait_sec = int((target - now).total_seconds())
    return wait_sec, target


def sleep_interruptible(total_sec: int, step_sec: int = 30) -> None:
    remaining = total_sec
    while remaining > 0:
        chunk = min(step_sec, remaining)
        time.sleep(chunk)
        remaining -= chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run daily crypto pipeline at fixed BJT time and push latest grouping to Feishu."
    )
    parser.add_argument("--hour", type=int, default=8, help="BJT hour for daily run")
    parser.add_argument("--minute", type=int, default=0, help="BJT minute for daily run")
    parser.add_argument(
        "--python-bin",
        default=str(DEFAULT_PYTHON),
        help="Python interpreter path for subprocess scripts",
    )
    parser.add_argument(
        "--webhook-url",
        default=os.getenv("FEISHU_WEBHOOK_URL", ""),
        help="Feishu robot webhook URL (or set FEISHU_WEBHOOK_URL env var)",
    )
    parser.add_argument(
        "--bot-secret",
        default=os.getenv("FEISHU_BOT_SECRET", ""),
        help="Optional Feishu bot secret for signature (or set FEISHU_BOT_SECRET)",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run immediately once before entering scheduler loop",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (no scheduler loop)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send Feishu message; print message to log only",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Optional[str], Optional[str]]:
    # 不使用 resolve()，避免把 venv 解释器符号链接解析成系统 Python
    python_bin = Path(args.python_bin).expanduser()
    if not python_bin.is_absolute():
        python_bin = (ROOT_DIR / python_bin).absolute()
    if not python_bin.exists():
        raise FileNotFoundError(f"Python bin not found: {python_bin}")
    if not CAPTURE_SCRIPT.exists():
        raise FileNotFoundError(f"Script not found: {CAPTURE_SCRIPT}")
    if not FACTOR_SCRIPT.exists():
        raise FileNotFoundError(f"Script not found: {FACTOR_SCRIPT}")
    webhook_url = args.webhook_url.strip()
    bot_secret = args.bot_secret.strip()
    if (not args.dry_run) and (not webhook_url):
        raise ValueError(
            "Missing Feishu webhook URL. Use --webhook-url or FEISHU_WEBHOOK_URL env var."
        )
    return python_bin, webhook_url if webhook_url else None, bot_secret if bot_secret else None


def main() -> int:
    load_env_file(ENV_FILE, override=False)
    args = parse_args()
    try:
        python_bin, webhook_url, bot_secret = validate_inputs(args)
    except Exception as e:
        print(f"Input validation failed: {e}", file=sys.stderr)
        return 1

    if args.once:
        ok = run_once(python_bin, webhook_url, bot_secret, args.dry_run)
        return 0 if ok else 2

    if args.run_now:
        run_once(python_bin, webhook_url, bot_secret, args.dry_run)

    log(
        f"Scheduler started, daily at {args.hour:02d}:{args.minute:02d} BJT, "
        f"python={python_bin}"
    )
    while True:
        wait_sec, target = seconds_to_next_run(args.hour, args.minute)
        log(f"Next run at {format_bjt(target)}; sleeping {wait_sec}s")
        sleep_interruptible(wait_sec, step_sec=30)
        try:
            run_once(python_bin, webhook_url, bot_secret, args.dry_run)
        except Exception as e:
            err_msg = f"Unexpected scheduler error: {e}"
            log(err_msg)
            if not args.dry_run and webhook_url:
                try:
                    post_to_feishu(
                        webhook_url,
                        f"[Crypto Daily Scheduler Error]\nTime(BJT): {format_bjt()}\n{err_msg}",
                        bot_secret=bot_secret,
                    )
                except Exception as send_e:
                    log(f"Failed to send scheduler error to Feishu: {send_e}")


if __name__ == "__main__":
    raise SystemExit(main())
