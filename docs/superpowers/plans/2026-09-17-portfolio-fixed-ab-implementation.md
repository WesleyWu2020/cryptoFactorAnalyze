# Portfolio A/B 固定组合与共享账本 Implementation Plan

> **For agentic workers:** 按本文件任务顺序实施，使用 `- [ ]` 记录进度。用户本轮明确指定 `writing-plans` 生成执行文档；本文件不自动授权启用其他 Superpowers 技能、派生代理、真实下单或发布操作。代码实施在后续实施任务中执行。

**Goal:** 完成 A「单因子等价迁移」和 B「固定多因子组合闭环」，产出可复核的目标持仓、成交、资金费、净值、风险诊断与 HTML 报告。

**Architecture:** 新增 `portfolio.main_fixed` 作为日频永续多空研究入口，复用 `factor_common` 的模块加载、H5 数据访问、因子预处理、分组、资金费与指标。将共享回测器扩展为可接收目标权重，同时保持原 `run_backtest(values, ...)` 的调用契约。组合采用固定资本占比合并各因子的多空目标，再统一缩小敞口满足约束，最后只对净额订单记账。

**Tech Stack:** Python、pandas、NumPy、pytest、现有 PyTables/H5、Parquet 引擎、标准库 argparse/json/html/hashlib；不引入优化器、数据库或交易框架。

---

## 0. 范围、事实与实施约束

### 0.1 用户已确认的交付范围

- A：模块加载、冻结元信息、时间规则、共享目标权重记账入口、单因子迁移等价测试。
- B：固定多因子目标合并、净额订单、风险约束、统一指标与可追溯报告。
- 不包含 C：交易所 API、实盘账户、影子订单服务、定时调度、自动熔断服务。
- 不包含因子挖掘、挑选历史收益最优组合、重新修改因子参数、动态 IC 权重、自动聚类、正交化、BTC 择时/对冲、Path B 改造。
- 研究模式固定为 `perp_long_short`。用户尚未确定实盘现货/永续；本计划不替用户作实盘选择。现货模式必须单独定义并验收，不能只关闭资金费冒充现货。

### 0.2 已核查的本地事实

1. `factor_common/grouping.py::target_weights` 的 `long_short` 为方向调整后的 50/50 多空组合；旧 `portfolio/main.py` 为 Top-N 多头，两者不是同一策略。
2. 旧 `portfolio/backtester/fees.py` 把声明为单边的费率除以 2；旧测试也固化了这个结果。
3. 旧 `portfolio/backtester/metrics.py` 使用 252 天、平均收益复利，且最大回撤未纳入初始净值。
4. 旧 Top-N 构造在分数全为 NaN 时仍可给出满仓；旧 aligner 无期限前向填充。
5. 旧 `pipeline.py::_load_factor_directions` 不扫描 GP 子目录。
6. 旧多日标签调权未按收益实现时间过滤；当日对冲敞口与当日收益存在时间错配。
7. 当前 `factor_common/backtest.py` 在无法估值时按上次价格记强制成交，与其模块注释不一致。新路径必须采用严格中止规则。
8. 上一轮只读检查运行 `tests/test_portfolio_*.py`，结果为 **132 passed in 35.26s**。这是旧测试基线，不代表本计划已经实现或新路径已经通过。

### 0.3 不破坏用户工作区

当前工作区存在用户未提交的因子、GP、数据研究和测试改动。实施前重新运行 `git status --short`、`git diff -- factor_common portfolio tests`。不重置、不清理、不全量暂存，不重写已有报告。

本计划采用新入口与小范围共享接口扩展，旧 `portfolio/main.py`、`main_hedged.py`、`main_pathB.py` 保留原调用方式，但在 README 中明确标记为旧研究路径。旧路径已知问题不在 A/B 中逐一修复；**新路径不得调用它们的 fees、metrics、aligner、IC weighter 或 backtest engine**。

每个任务完成后检查本任务 diff。只有隔离了属于本任务的变更才提交；下文提交信息是检查点建议，不授权把用户原有改动一起提交。

### 0.4 冻结的业务语义

| 项目 | A/B 规则 |
|---|---|
| 时间 | 日频 UTC 日期，完整自然日日历；因子信号日期 t 在 t+1 开盘生效 |
| 交易价格 | 现有共享引擎的 open 基准；滑点是名义金额上的现金成本，不伪称真实成交价 |
| 数据可用性 | `as_of` 表示已完整观察的 UTC 日；不是精确的历史数据到达时间，也不能恢复供应商历史修订 |
| 调仓 | 所有成员共用 `rebalance_days`、`anchor_date`；按自然日模运算，不按剩余数据行数 |
| 方向 | 仅模块 `SETTING.factor_direction` 决定；不按回测期 IC 翻转 |
| 因子资本占比 | 所有 allocation > 0，和为 1；成员缺失时不重新分配其预算 |
| 单因子目标 | 复用模块预处理及共享分组，原始总敞口为 1，多/空各 0.5 |
| 无效成员日 | 有限值数量不足或截面无区分度，成员目标为全零；记录原因 |
| 信号缺失 | 不使用昨天的因子值补当天信号；下一计划调仓日按该成员全零目标退出 |
| 非调仓日 | 维持实际成交数量，允许权重漂移；不隐含每日再平衡 |
| 净额合并 | 同币不同因子的正负权重直接抵消；不把抵消后的剩余仓位放大 |
| 风险约束 | 对整个合并向量施加一个不大于 1 的比例；不做优化求解或补仓 |
| 无效价格 | 持仓无法估值则 `incomplete`，保留数量；不假设按旧价格成交 |
| 资金费 | 复用事件级现金流与 coverage；缺失不能填零冒充完整 |
| 尾部 | 保留末次计划仓位的完整持有与退出区间；尾部不足为 `incomplete` |
| 指标 | 日频 365 天、真实复利、初始净值计入回撤；不完整全期指标为 null |
| 风险上限含义 | 限制调仓前费前权益口径的目标权重，不保证持仓漂移后或扣费后始终满足同一比例 |
| 空间 | 不使用实时交易精度、最小订单额、保证金/强平模型；报告必须明确这些限制 |

“信号有效性”“分组资格”使用当时可用的因子值和币池；执行日整天的质量统计只能用来事后诊断，不能反过来决定当天开盘买哪个币。历史缺失的原始数据修订时间不是本次 cutoff 测试可证明的事项。

## 1. 文件与接口地图

| 文件 | 动作 | 责任 |
|---|---|---|
| `factor_common/backtest.py` | 小范围修改 | 保留单因子 API，抽出目标权重入口、严格估值和开仓资格规则 |
| `portfolio/fixed_config.py` | 新增 | 严格配置、固定资本占比、参数边界 |
| `portfolio/factor_pool/module_loader.py` | 新增 | 显式模块加载、共享 compute_factor、冻结代码核对 |
| `portfolio/fixed_provenance.py` | 新增 | 代码快照、输入数据读取摘要、源文件变化检查 |
| `portfolio/portfolio_builder/fixed_weights.py` | 新增 | 成员目标与固定权重合并、比例风控 |
| `portfolio/fixed_pipeline.py` | 新增 | A/B 编排、三情景账本、结构化返回 |
| `portfolio/fixed_artifacts.py` | 新增 | 每次运行的 Parquet/JSON/HTML 与成功标志 |
| `portfolio/main_fixed.py` | 新增 | 冻结配置、运行、cutoff 审计三个 CLI 子命令 |
| `portfolio/fixed_audit.py` | 新增 | 从截断 provider 重算并比较值、目标、账本、订单 |
| `portfolio/README.md` | 增补 | 新旧入口边界、运行命令、结果解释 |
| `tests/portfolio_fixed/conftest.py` | 新增 | 小型因子模块和复用 H5 fixture |
| `tests/portfolio_fixed/test_*.py` | 新增 | 配置、成员、组合、风险、CLI、cutoff、失败状态 |
| `tests/factor_common/test_target_backtest.py` | 新增 | 共享 API、费用、资金费、估值、终止语义 |

不新增因子注册表。不改 `factor_miner` 契约。不移动旧文件。新输出目录为 `reports/portfolio/<run_id>/`；旧 `portfolio/output/` 保留。

依赖顺序：A0 → A1 → A2 → A3 → A4 → B1 → B2 → B3 → B4。A4 是第一验收点；B4 是最终交付点。

## 2. Task A0：建立独立测试夹具与基线

**Files:** Create `tests/portfolio_fixed/conftest.py`。

- [ ] 在仓库根目录运行基线，保留输出，不将既有失败误归到本任务。

```bash
./.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/factor_common tests/test_portfolio_*.py
git diff -- factor_common portfolio tests
```

预期：取得当前实际结果；132 是上轮旧 portfolio 子集数量，不写死整体测试数量。

- [ ] 新增以下完整测试夹具。fixture 会写临时 H5 与临时因子，不修改真实因子。

```python
# tests/portfolio_fixed/conftest.py
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.factor_common.conftest import (
    EXAMPLE_CALENDAR,
    _complete_funding_schedule,
    _example_funding_events,
    write_h5_fixture,
)


@pytest.fixture
def fixed_h5(tmp_path):
    return write_h5_fixture(
        tmp_path / "fixed.h5",
        calendar=EXAMPLE_CALENDAR,
        funding_schedule=_complete_funding_schedule(),
        funding_events=_example_funding_events(),
    )


@pytest.fixture
def factor_paths(tmp_path):
    paths = []
    for name, direction in (("FixedClose", 1), ("ReverseClose", -1)):
        path = tmp_path / f"{name}.py"
        source = (
            "TYPE = 'regular'\n"
            f"META = {{'factor_name': {name!r}, 'author': 'test', "
            "'level': 'daily', 'category': 'test', 'description': 'fixture'}\n"
            "SETTING = {'data_needed': ['close'], 'universe': 'historical_top50', "
            "'warmup_bars': 0, 'preprocessing': 'none', 'params': {}, "
            f"'factor_direction': {direction}}}\n"
            "def calc_factor(data_ctx):\n"
            "    return data_ctx['close'].copy()\n"
        )
        path.write_text(source, encoding="utf-8")
        paths.append(path)
    return paths


@pytest.fixture
def config_dict(fixed_h5, factor_paths):
    from portfolio.fixed_config import FixedConfig, FactorAllocation
    from portfolio.fixed_provenance import code_snapshot

    cfg = FixedConfig(
        h5_path=str(fixed_h5),
        signal_start="2024-01-22",
        signal_end="2024-01-24",
        as_of="2024-01-26",
        factors=(FactorAllocation(str(factor_paths[0]), 1.0),),
        code_hashes=code_snapshot(factor_paths[:1]),
        n_groups=2,
        min_valid_instruments=2,
        gross_limit=1.0,
        long_limit=1.0,
        short_limit=1.0,
        single_limit=1.0,
        net_limit=1.0,
        fee_rate=0.001,
        slippage=0.0,
    )
    return asdict(cfg)
```

该文件只有在配置与 provenance 模块实现后调用 `config_dict` 时才需要它们；fixture 本身先被收集不会执行这些导入。

- [ ] 执行 `./.venv/bin/python -B -m pytest --collect-only -q tests/portfolio_fixed`，确认无语法或 conftest 导入错误。当前没有测试项允许退出码 5；后续任务出现测试后应为 0。
- [ ] 检查点提交信息：`test: add fixed portfolio research fixtures`。

## 3. Task A1：严格配置

**Files:** Create `portfolio/fixed_config.py`, `tests/portfolio_fixed/test_config.py`。

- [ ] 先新增以下测试，再运行，预期首次为模块不存在。

```python
# tests/portfolio_fixed/test_config.py
from dataclasses import asdict

import pytest

from portfolio.fixed_config import FactorAllocation, FixedConfig


def sample():
    return FixedConfig(
        h5_path="data/crypto_quant.h5",
        signal_start="2024-01-01",
        signal_end="2024-02-01",
        as_of="2024-02-03",
        factors=(FactorAllocation("factor.py", 1.0),),
        code_hashes={"factor.py": "a" * 64},
    )


def test_roundtrip_and_unknown_key():
    raw = asdict(sample())
    assert FixedConfig.from_dict(raw) == sample()
    with pytest.raises(ValueError, match="unknown"):
        FixedConfig.from_dict({**raw, "gross_limt": 0.2})


@pytest.mark.parametrize("change", [
    {"rebalance_days": True}, {"fee_rate": -1}, {"slippage": float("nan")},
    {"gross_limit": 1.1}, {"net_limit": -0.1}, {"min_valid_instruments": 1},
    {"market_mode": "spot"}, {"as_of": "2023-12-31"},
    {"signal_start": "2024-01-01T12:00:00"},
    {"factors": [{"path": "factor.py", "allocation": 0.8}]},
    {"factors": [{"path": "factor.py", "allocation": -1.0}]},
])
def test_reject_invalid_config(change):
    with pytest.raises((ValueError, TypeError)):
        FixedConfig.from_dict({**asdict(sample()), **change})
```

Run: `./.venv/bin/python -B -m pytest -q tests/portfolio_fixed/test_config.py`。

- [ ] 新增完整实现。结构冻结不代表嵌套 dict 绝对不可变；pipeline 在进入时会 JSON 化复制并核对快照，不在运行中修改配置。

```python
# portfolio/fixed_config.py
from dataclasses import dataclass, fields
import math
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FactorAllocation:
    path: str
    allocation: float


def daily_date(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError("date must be UTC-naive daily midnight")
    return day


@dataclass(frozen=True)
class FixedConfig:
    h5_path: str
    signal_start: str
    signal_end: str
    as_of: str
    factors: tuple[FactorAllocation, ...]
    code_hashes: dict[str, str]
    schema_version: int = 1
    market_mode: str = "perp_long_short"
    rebalance_days: int = 1
    anchor_date: str = "2024-01-01"
    n_groups: int = 5
    min_valid_instruments: int = 10
    fee_rate: float = 0.0005
    slippage: float = 0.001
    gross_limit: float = 0.5
    long_limit: float = 0.25
    short_limit: float = 0.25
    single_limit: float = 0.05
    net_limit: float = 0.05
    initial_equity: float = 1.0
    frozen_at: str = ""

    def __post_init__(self):
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("unsupported schema_version")
        if self.market_mode != "perp_long_short":
            raise ValueError("only perp_long_short research is supported")
        start, end, cutoff = map(daily_date, (
            self.signal_start, self.signal_end, self.as_of,
        ))
        daily_date(self.anchor_date)
        if not start <= end or cutoff < start:
            raise ValueError("invalid date interval")
        for name, minimum in (("rebalance_days", 1), ("n_groups", 2),
                              ("min_valid_instruments", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"invalid {name}")
        if self.min_valid_instruments < self.n_groups:
            raise ValueError("min_valid_instruments must cover n_groups")
        for name in ("fee_rate", "slippage", "gross_limit", "long_limit",
                     "short_limit", "single_limit", "net_limit", "initial_equity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"invalid {name}")
            if not math.isfinite(value):
                raise ValueError(f"invalid {name}")
        if not 0 <= self.fee_rate < 1 or not 0 <= self.slippage < 1:
            raise ValueError("invalid transaction cost")
        for name in ("gross_limit", "long_limit", "short_limit", "single_limit"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"invalid {name}")
        if not 0 <= self.net_limit <= 1 or self.initial_equity <= 0:
            raise ValueError("invalid net_limit or initial_equity")
        if not self.factors:
            raise ValueError("factors must not be empty")
        resolved = []
        for member in self.factors:
            if not isinstance(member, FactorAllocation):
                raise TypeError("factor must be FactorAllocation")
            if not isinstance(member.path, str) or not member.path.strip():
                raise ValueError("factor path required")
            if (isinstance(member.allocation, bool)
                    or not isinstance(member.allocation, (int, float))
                    or not math.isfinite(member.allocation)
                    or member.allocation <= 0):
                raise ValueError("allocation must be finite and positive")
            resolved.append(str(Path(member.path).resolve()))
        if len(resolved) != len(set(resolved)):
            raise ValueError("duplicate factor path")
        if not math.isclose(sum(x.allocation for x in self.factors), 1.0,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("allocations must sum to one")
        if not isinstance(self.code_hashes, dict) or not self.code_hashes:
            raise ValueError("frozen code_hashes required")
        for key, value in self.code_hashes.items():
            if not isinstance(key, str) or not isinstance(value, str) or len(value) != 64:
                raise ValueError("invalid code hash")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError("invalid code hash") from exc

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict):
            raise TypeError("config must be a JSON object")
        allowed = {item.name for item in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        data = dict(raw)
        members = []
        for member in data.get("factors", []):
            if not isinstance(member, dict) or set(member) != {"path", "allocation"}:
                raise ValueError("factor requires exactly path and allocation")
            members.append(FactorAllocation(**member))
        data["factors"] = tuple(members)
        return cls(**data)
```

- [ ] 重跑配置测试，预期全部 PASS。额外确认默认风险数值只是研究示例：A 等价测试会显式设成不触发约束的值，不能把这些默认值宣传为最优风控参数。
- [ ] 检查点提交信息：`feat: define strict fixed portfolio configuration`。

## 4. Task A2：冻结依赖、显式加载模块、记录输入

**Files:** Create `portfolio/fixed_provenance.py`, `portfolio/factor_pool/module_loader.py`, `tests/portfolio_fixed/test_modules.py`。

- [ ] 新增测试。

```python
# tests/portfolio_fixed/test_modules.py
from dataclasses import asdict

import pytest

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
    with pytest.raises(ValueError, match="code snapshot"):
        load_members(FixedConfig.from_dict(config_dict))


def test_repository_gp_negative_direction_is_loaded():
    from factor_common.loader import load_factor
    spec = load_factor(
        "factor_analyse/factor_mining/GP_Factor/GP_064185107a8f267e.py"
    )
    assert spec.setting["factor_direction"] == -1
```

Run: `./.venv/bin/python -B -m pytest -q tests/portfolio_fixed/test_modules.py`。预期首次缺少新模块；不能通过硬编码 GP 方向让测试通过。

- [ ] 新增 provenance 实现。复用已有 `snapshot_source_stats` 与 `_json_safe`，不复制其逻辑；这里对这些现有内部辅助接口建立显式测试依赖，后续若公开 API 再统一迁移。

```python
# portfolio/fixed_provenance.py
import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from factor_common.storage import _json_safe, snapshot_source_stats

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = (
    "factor_common", "portfolio", "Genetic_Algorithm", "data/crypto_quant",
    "factor_analyse/factor_mining",
)


def code_snapshot(extra_paths):
    paths = {Path(path).resolve(strict=True) for path in extra_paths}
    for directory in CODE_DIRS:
        paths.update(path.resolve() for path in (ROOT / directory).rglob("*.py"))
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def frame_hash(frame):
    # 包含轴、列顺序和值；to_csv 只生成字符串，不写文件。
    return hashlib.sha256(frame.to_csv().encode("utf-8")).hexdigest()


class RecordingProvider:
    def __init__(self, provider):
        self.provider = provider
        self.reads = []

    @property
    def symbols(self):
        return self.provider.symbols

    def _record(self, method, *args, **kwargs):
        frame = getattr(self.provider, method)(*args, **kwargs)
        self.reads.append({
            "method": method, "args": _json_safe(args),
            "kwargs": _json_safe(kwargs), "sha256": frame_hash(frame),
            "shape": list(frame.shape),
        })
        return frame

    def get_single_data(self, field, *, start, end):
        return self._record("get_single_data", field, start=start, end=end)

    def get_universe(self, *, start, end):
        return self._record("get_universe", start=start, end=end)

    def get_quality(self, *, start, end, symbols):
        return self._record("get_quality", start=start, end=end, symbols=list(symbols))

    def get_funding(self, *, start, end, symbols):
        return self._record("get_funding", start=start, end=end, symbols=list(symbols))


def environment_metadata():
    versions = {"python": platform.python_version()}
    for package in ("pandas", "numpy", "scipy", "tables", "pyarrow", "fastparquet"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def assert_source_unchanged(paths, before):
    if snapshot_source_stats(paths) != before:
        raise RuntimeError("input source changed during portfolio evaluation")
```

代码快照保守覆盖研究运行涉及的本地目录，改动未使用因子也可能导致需要重新冻结。这是有意选择的简单一致性规则；哈希不是依赖环境锁，也不能证明代码无未来函数。绝对路径限制配置的跨工作区直接复用，移动工作区后须重新冻结并保留旧 manifest。

- [ ] 新增显式加载实现。静态扫描只扫描指定因子源文件；GP evaluator 等依赖通过代码冻结和后续动态 cutoff 检查覆盖，不能宣称扫描了全部依赖即通过。

```python
# portfolio/factor_pool/module_loader.py
from factor_common.loader import load_factor
from factor_common.validation import scan_future_leaks
from portfolio.fixed_provenance import code_snapshot


def load_members(config):
    paths = [member.path for member in config.factors]
    if code_snapshot(paths) != config.code_hashes:
        raise ValueError("code snapshot differs from frozen configuration")
    findings = scan_future_leaks(paths)
    if findings:
        raise ValueError(f"factor source future-leak findings: {findings}")
    specs = [load_factor(path) for path in paths]
    names = [spec.factor_id for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate factor identifier")
    return specs
```

- [ ] 重跑 `test_config.py`、`test_modules.py`，预期 PASS。
- [ ] 检查点提交信息：`feat: freeze factor modules and record portfolio inputs`。

## 5. Task A3：扩展共享账本，保留旧入口

**Files:** Modify `factor_common/backtest.py`; Create `tests/factor_common/test_target_backtest.py`。

这是唯一允许改动共享记账核心的任务。不能复制 `_run_scenario` 到 portfolio；不能改变旧 API 的列名或返回字段。新入口严格模式与旧入口保留模式的结果差异必须来自明确列出的异常数据政策。

### A3.1 先写独立现金流测试

- [ ] 新增测试文件；以下使用现有手算 fixture 辅助函数，不重复搭建市场数据。

```python
# tests/factor_common/test_target_backtest.py
import numpy as np
import pandas as pd
import pytest

from factor_common.backtest import run_backtest, run_target_backtest
from factor_common.grouping import target_weights
from tests.factor_common.test_backtest import _frames, _profile, _quality, EMPTY_EVENTS


def run_targets(targets, opens, profile=None, quality=None):
    return run_target_backtest(
        targets, opens, EMPTY_EVENTS,
        _quality(len(opens), opens.columns) if quality is None else quality,
        _profile() if profile is None else profile,
        signal_start="2024-01-01", signal_end="2024-01-01",
    )


def basic_frames():
    return _frames(3, {0: {"A": 2.0, "B": 1.0}}, {
        0: {"A": 100.0, "B": 100.0},
        1: {"A": 100.0, "B": 100.0},
        2: {"A": 100.0, "B": 100.0},
    })


def test_single_factor_accounting_equivalence():
    values, opens = basic_frames()
    profile = _profile(fee_rate=0.001, slippage=0.002)
    quality = _quality(3, opens.columns)
    old = run_backtest(values, opens, EMPTY_EVENTS, quality, profile,
                       signal_start="2024-01-01", signal_end="2024-01-01")
    new = run_targets(target_weights(values, profile)["long_short"], opens,
                      profile, quality)
    for scenario in ("gross", "trading_net", "all_costs"):
        for key in ("ledger", "orders", "positions", "funding"):
            pd.testing.assert_frame_equal(old["scenarios"][scenario][key],
                                          new["scenarios"][scenario][key])


def test_entry_and_exit_charge_actual_one_way_notional():
    values, opens = basic_frames()
    profile = _profile(fee_rate=0.001)
    result = run_targets(target_weights(values, profile)["long_short"], opens, profile)
    account = result["scenarios"]["trading_net"]
    orders = account["orders"]
    assert orders["fee"].sum() == pytest.approx(0.002)
    assert account["ledger"].iloc[-1]["equity"] == pytest.approx(0.998)
    assert (orders["fee"] == orders["notional"] * 0.001).all()


def test_missing_held_price_halts_without_invented_fill():
    values, opens = basic_frames()
    opens.loc["2024-01-03", "A"] = np.nan
    targets = target_weights(values, _profile())["long_short"]
    account = run_targets(targets, opens)["scenarios"]["gross"]
    assert account["status"] == "incomplete"
    assert account["diagnostics"]["halt_reason"] == "unpriceable_holding"
    assert account["diagnostics"]["final_quantities"]["A"] == pytest.approx(0.005)
    assert not (account["orders"]["reason"] == "unpriceable_forced_exit").any()
    assert account["ledger"].index.max() == pd.Timestamp("2024-01-02")


def test_partial_nan_target_is_rejected():
    values, opens = basic_frames()
    targets = target_weights(values, _profile())["long_short"]
    targets.loc["2024-01-01", "A"] = np.nan
    with pytest.raises(ValueError, match="partial"):
        run_targets(targets, opens)


def test_all_zero_target_is_explicit_cash():
    values, opens = basic_frames()
    targets = target_weights(values, _profile())["long_short"]
    targets.loc["2024-01-01"] = 0.0
    account = run_targets(targets, opens)["scenarios"]["all_costs"]
    assert account["orders"].empty
    assert account["diagnostics"]["final_quantities"] == {}


def test_future_execution_price_cannot_change_entry_quantity():
    values, opens = basic_frames()
    targets = target_weights(values, _profile())["long_short"]
    first = run_targets(targets, opens)["scenarios"]["gross"]["orders"]
    changed = opens.copy()
    changed.loc["2024-01-03", "A"] = 200.0
    second = run_targets(targets, changed)["scenarios"]["gross"]["orders"]
    cutoff = pd.Timestamp("2024-01-02")
    pd.testing.assert_frame_equal(first[first.date <= cutoff].reset_index(drop=True),
                                  second[second.date <= cutoff].reset_index(drop=True))
```

Run: `./.venv/bin/python -B -m pytest -q tests/factor_common/test_target_backtest.py`。首次预期 ImportError。

### A3.2 抽出公共目标入口

- [ ] 在 `factor_common/backtest.py` 中，将原 `run_backtest` 重命名为 `_run_target_backtest`，保留 `values` 参数名作为局部载体，新增两个关键字参数。完整新签名：

```diff
-def run_backtest(
+def _run_target_backtest(
    values: pd.DataFrame,
    opens: pd.DataFrame,
    events: pd.DataFrame,
    quality: pd.DataFrame,
    profile: BacktestProfile,
    *,
    signal_start,
    signal_end,
+    portfolio_name: str,
+    strict_execution: bool,
) -> dict:
     """Simulate fixed-quantity daily accounting; see the module docstring."""
```

- [ ] 仅在该函数内作以下精确替换，保留日历、资金费事件分桶、尾部和返回字段的其他代码：

```diff
-    weights = target_weights(values, profile)[PORTFOLIO]
+    weights = values
```

两个返回字典中的 `"portfolio": PORTFOLIO` 改为 `"portfolio": portfolio_name`。构造 `_Context` 时新增：

```python
        strict_execution=strict_execution,
```

- [ ] 在模块末尾、`__all__` 之前新增两个完整公共函数：

```python
def run_backtest(values, opens, events, quality, profile, *, signal_start, signal_end):
    """Compatibility entry: preserve the existing single-factor contract."""
    if not isinstance(profile, BacktestProfile):
        raise TypeError("profile must be a BacktestProfile")
    _validate_axes(values, name="values")
    return _run_target_backtest(
        target_weights(values, profile)[PORTFOLIO], opens, events, quality, profile,
        signal_start=signal_start, signal_end=signal_end,
        portfolio_name=PORTFOLIO, strict_execution=False,
    )


def run_target_backtest(targets, opens, events, quality, profile, *,
                        signal_start, signal_end):
    """Strict research accounting for signed, signal-dated target weights."""
    if not isinstance(profile, BacktestProfile):
        raise TypeError("profile must be a BacktestProfile")
    _validate_axes(targets, name="targets")
    if np.isinf(targets.to_numpy(dtype=float)).any():
        raise ValueError("infinite target weight")
    partial = targets.isna().any(axis=1) & ~targets.isna().all(axis=1)
    if partial.any():
        raise ValueError("partial NaN target row")
    gross = targets.abs().sum(axis=1)
    if (gross > profile.gross_exposure + 1e-12).any():
        raise ValueError("target gross exceeds profile gross_exposure")
    return _run_target_backtest(
        targets, opens, events, quality, profile,
        signal_start=signal_start, signal_end=signal_end,
        portfolio_name="fixed_targets", strict_execution=True,
    )
```

将 `run_target_backtest` 加入现有 `__all__`。旧 API 原来的轴校验仍在内部，且输出 `portfolio` 仍是 `long_short`。新 API 输出 `fixed_targets`。

### A3.3 严格估值：先检查完整旧仓，再更新现金流

- [ ] `_run_scenario` 每日循环在 `held_pre` 构造之后、任何价格盈亏或费用变更之前，插入：

```python
        if ctx.strict_execution:
            unpriceable = sorted(
                str(inst) for inst in held_pre if not _valid_price(open_row[inst])
            )
            if unpriceable:
                halt = {"reason": "unpriceable_holding", "date": day,
                        "detail": ",".join(unpriceable)}
                break
```

不能在逐币更新到一半时中止，否则可能保留部分当天 PnL，制造不完整的日状态。保留旧分支仅供原 API 历史兼容；新 API 永远不会走到其假设强制退出逻辑。

### A3.4 严格执行资格：阻止新增风险，允许已有风险减少

- [ ] 在 `target_qty = weight * equity_before_trade / float(price)` 之后，将旧的 `if weight != 0.0 and current == 0.0 and not ctx.eligible(day, inst):` 及其整个条件体替换为如下完整块；下一行仍为 `delta = target_qty - current`。

```python
                requested_qty = target_qty
                if ctx.strict_execution and not ctx.eligible(day, inst):
                    if current == 0.0:
                        target_qty = 0.0
                    elif requested_qty * current < 0.0:
                        target_qty = 0.0
                    elif abs(requested_qty) > abs(current):
                        target_qty = current
                    if target_qty != requested_qty:
                        blocked.append({
                            "date": _iso(day), "instrument": str(inst),
                            "requested_quantity": requested_qty,
                            "allowed_quantity": target_qty,
                            "reason": "prior_bar_ineligible_risk_increase",
                        })
                legacy_block = (
                    not ctx.strict_execution and weight != 0.0 and current == 0.0
                    and not ctx.eligible(day, inst)
                )
                if legacy_block:
                    failed_orders += 1
                    blocked.append({"date": _iso(day), "instrument": str(inst)})
                    order_rows.append({
                        "date": day,
                        "instrument": inst,
                        "side": "buy" if weight > 0 else "sell",
                        "target_weight": weight,
                        "target_quantity": target_qty,
                        "order_quantity": np.nan,
                        "price": float(price),
                        "notional": 0.0,
                        "fee": 0.0,
                        "slippage": 0.0,
                        "status": "failed",
                        "reason": "prior_bar_ineligible",
                    })
                    continue
```

原 eligibility 仅检查 placeholder；新严格路径对未知/不完整前一日 K 线也要拒绝新增风险。在 `_run_target_backtest` 中，用以下内容替换现有 `eligible` 局部函数：

```python
    def eligible(day: pd.Timestamp, instrument) -> bool:
        previous = day - _ONE_DAY
        if placeholder.get((previous, str(instrument)), False):
            return False
        if not strict_execution:
            return True
        key = (previous, instrument)
        if key not in quality.index:
            return False
        if "has_complete_kline" not in quality.columns:
            return False
        complete = quality.at[key, "has_complete_kline"]
        return not pd.isna(complete) and bool(complete)
```

**同步测试 fixture：** 本任务新测试的 `run_targets` 和 `test_single_factor_accounting_equivalence` 在生成 quality 后都设置 `quality["has_complete_kline"] = True`。这只能用于合成数据明确完整的场景，不得在生产 provider 上补 True。将 helper 中的 quality 构造写成以下完整形式：

```python
def run_targets(targets, opens, profile=None, quality=None):
    if quality is None:
        quality = _quality(len(opens), opens.columns)
        quality["has_complete_kline"] = True
    return run_target_backtest(
        targets, opens, EMPTY_EVENTS, quality,
        _profile() if profile is None else profile,
        signal_start="2024-01-01", signal_end="2024-01-01",
    )
```

对等价测试的 `quality = _quality(3, opens.columns)` 后新增 `quality["has_complete_kline"] = True`。

- [ ] 增加缺失质量拒绝开仓的测试：

```python
def test_unknown_prior_quality_blocks_new_exposure():
    values, opens = basic_frames()
    targets = target_weights(values, _profile())["long_short"]
    quality = _quality(3, opens.columns)
    account = run_targets(targets, opens, quality=quality)["scenarios"]["gross"]
    assert account["orders"].empty
    assert len(account["diagnostics"]["blocked_orders"]) == 2
```

- [ ] 执行 `./.venv/bin/python -B -m pytest -q tests/factor_common/test_backtest.py tests/factor_common/test_target_backtest.py tests/factor_common/test_funding.py`。预期全部 PASS；旧手算测试的列和金额不应为了通过而改写。
- [ ] 更新共享模块注释：旧入口有显式保留的 legacy 退出假设，新入口严格中止；不得继续用无条件“缺价即中止”描述两者。注释必须记录 `strict_execution` 与 `fixed_targets`。
- [ ] 检查点提交信息：`feat: expose strict target-weight accounting with legacy compatibility`。

## 6. Task A4：单成员目标与结构化研究流水线

**Files:** Create `portfolio/portfolio_builder/fixed_weights.py`, `portfolio/fixed_pipeline.py`, `tests/portfolio_fixed/test_single.py`。

### A4.1 成员目标

- [ ] 新增成员目标实现。方向只通过 `replace(profile, factor_direction=...)` 应用一次；不额外取负、不再 rank/z-score。

```python
# portfolio/portfolio_builder/fixed_weights.py
from dataclasses import replace

import numpy as np
import pandas as pd

from factor_common.grouping import target_weights


def member_targets(values, spec, profile, min_valid):
    clean = values.replace([np.inf, -np.inf], np.nan)
    count = clean.notna().sum(axis=1)
    spread = clean.max(axis=1) - clean.min(axis=1)
    valid = (count >= min_valid) & (spread > 0.0)
    unit_profile = replace(profile, gross_exposure=1.0,
                           factor_direction=spec.setting["factor_direction"])
    weights = target_weights(clean, unit_profile)["long_short"]
    # 无信号的明确研究政策是到计划调仓日退出，不沿用过期值。
    weights.loc[~valid, :] = 0.0
    weights = weights.fillna(0.0)
    diagnostics = pd.DataFrame({
        "valid_count": count,
        "valid_signal": valid,
        "reason": np.where(count < min_valid, "insufficient_values",
                           np.where(spread > 0, "valid", "no_cross_section_spread")),
    })
    return weights, diagnostics


def combine_targets(members, allocations, config):
    # A 阶段只做一个成员的无约束等价迁移，B1 将替换本函数。
    if len(members) != 1 or set(members) != set(allocations):
        raise ValueError("stage A requires exactly one allocated member")
    weights = next(iter(members.values())).copy()
    tests = (
        (weights.abs().sum(axis=1), config.gross_limit),
        (weights.clip(lower=0).sum(axis=1), config.long_limit),
        (-weights.clip(upper=0).sum(axis=1), config.short_limit),
        (weights.abs().max(axis=1), config.single_limit),
        (weights.sum(axis=1).abs(), config.net_limit),
    )
    if any((series > limit + 1e-12).any() for series, limit in tests):
        raise ValueError("stage A equivalence requires nonbinding limits")
    diagnostics = pd.DataFrame({"scale": 1.0}, index=weights.index)
    return weights, diagnostics
```

- [ ] 新增单成员及全 NaN 行测试，第一次运行应因 pipeline 缺失失败。

```python
# tests/portfolio_fixed/test_single.py
import numpy as np
import pandas as pd

from factor_common.backtest import run_backtest
from factor_common.data_provider import DataProvider
from factor_common.loader import load_factor
from factor_common.profiles import resolve_profile
from portfolio.fixed_config import FixedConfig
from portfolio.fixed_pipeline import run_fixed
from portfolio.portfolio_builder.fixed_weights import member_targets


def test_invalid_factor_day_is_zero_not_full_position(factor_paths):
    spec = load_factor(factor_paths[0])
    dates = pd.date_range("2024-01-01", periods=2, name="date")
    values = pd.DataFrame([[np.nan, np.nan], [3.0, 3.0]],
                          index=dates, columns=["A", "B"])
    targets, diag = member_targets(values, spec, resolve_profile("perp_1d", {"n_groups": 2}), 2)
    assert (targets == 0).all().all()
    assert not diag.valid_signal.any()


def test_single_pipeline_matches_shared_factor_accounting(config_dict):
    config = FixedConfig.from_dict(config_dict)
    result = run_fixed(config)
    assert result["status"] == "complete"
    spec = load_factor(config.factors[0].path)
    values = result["values"][spec.factor_id]
    provider = DataProvider(config.h5_path, as_of=config.as_of)
    index = result["targets"].index
    opens = provider.get_single_data("open", start=index[0], end=index[-1])
    quality = provider.get_quality(start=index[0], end=index[-1], symbols=values.columns)
    events = provider.get_funding(start=index[0], end=index[-1], symbols=values.columns)
    profile = resolve_profile("perp_1d", {
        "n_groups": config.n_groups, "factor_direction": spec.setting["factor_direction"],
        "rebalance_days": config.rebalance_days, "anchor_date": config.anchor_date,
        "fee_rate": config.fee_rate, "slippage": config.slippage,
    })
    baseline = run_backtest(values.reindex(index), opens, events, quality, profile,
                            signal_start=config.signal_start, signal_end=config.signal_end)
    for name in ("gross", "trading_net", "all_costs"):
        for key in ("ledger", "orders", "positions", "funding"):
            pd.testing.assert_frame_equal(result["accounting"]["scenarios"][name][key],
                                          baseline["scenarios"][name][key])
```

Run: `./.venv/bin/python -B -m pytest -q tests/portfolio_fixed/test_single.py`。

### A4.2 完整流水线

- [ ] 新增以下完整文件。流水线只依赖共享公共计算函数与本计划已定义函数；不调用旧组合模块。

```python
# portfolio/fixed_pipeline.py
from dataclasses import asdict, replace

import pandas as pd

from factor_common.backtest import run_target_backtest
from factor_common.data_provider import DataProvider
from factor_common.metrics import summarize_returns
from factor_common.profiles import resolve_profile
from factor_common.storage import snapshot_source_stats
from factor_common.value_engine import compute_factor, value_pipeline_fingerprint
from portfolio.factor_pool.module_loader import load_members
from portfolio.fixed_config import FixedConfig, daily_date
from portfolio.fixed_provenance import (
    RecordingProvider, assert_source_unchanged, code_snapshot, environment_metadata,
)
from portfolio.portfolio_builder.fixed_weights import member_targets, combine_targets


def summarize_accounting(accounting):
    summaries = {}
    for name, scenario in accounting["scenarios"].items():
        ledger = scenario["ledger"]
        complete = scenario["status"] == "complete"
        metrics = summarize_returns(ledger["return"], periods_per_year=365)
        summaries[name] = {
            "status": scenario["status"],
            "full": metrics if complete else None,
            "known_segment": None if complete else metrics,
            "fee_total": float(ledger["fee"].sum()),
            "slippage_total": float(ledger["slippage"].sum()),
            "funding_total": float(ledger["funding_cashflow"].sum()),
        }
    return summaries


def run_fixed(config, *, as_of=None):
    # 克隆配置，避免调用者运行时修改嵌套列表或字典。
    cfg = FixedConfig.from_dict(asdict(config))
    cutoff = daily_date(cfg.as_of if as_of is None else as_of)
    if cutoff > daily_date(cfg.as_of) or cutoff < daily_date(cfg.signal_start):
        raise ValueError("audit cutoff outside configured knowledge interval")
    signal_end = min(daily_date(cfg.signal_end), cutoff)
    tail_end = min(daily_date(cfg.signal_end)
                   + pd.Timedelta(days=1 + cfg.rebalance_days), cutoff)
    source_before = snapshot_source_stats([cfg.h5_path])
    specs = load_members(cfg)
    provider = RecordingProvider(DataProvider(cfg.h5_path, as_of=cutoff))
    profile = resolve_profile("perp_1d", {
        "n_groups": cfg.n_groups, "rebalance_days": cfg.rebalance_days,
        "anchor_date": cfg.anchor_date, "fee_rate": cfg.fee_rate,
        "slippage": cfg.slippage, "gross_exposure": cfg.gross_limit,
        "initial_equity": cfg.initial_equity, "include_funding": True,
        "funding_price_mode": "strict",
    })
    values, members, diagnostics, allocations = {}, {}, {}, {}
    for allocation, spec in zip(cfg.factors, specs):
        matrix, compute_diag = compute_factor(
            spec, provider, start=cfg.signal_start, end=signal_end,
        )
        target, member_diag = member_targets(
            matrix, spec, profile, cfg.min_valid_instruments,
        )
        values[spec.factor_id] = matrix
        members[spec.factor_id] = target
        allocations[spec.factor_id] = allocation.allocation
        diagnostics[spec.factor_id] = {
            "compute": compute_diag, "daily": member_diag,
            "direction": spec.setting["factor_direction"],
            "source_sha256": spec.source_sha256,
        }
    combined, risk = combine_targets(members, allocations, cfg)
    calendar = pd.date_range(cfg.signal_start, tail_end, freq="D", name="date")
    targets = combined.reindex(calendar)
    opens = provider.get_single_data("open", start=calendar[0], end=calendar[-1])
    opens = opens.reindex(index=calendar, columns=targets.columns)
    events = provider.get_funding(start=calendar[0], end=calendar[-1], symbols=targets.columns)
    quality = provider.get_quality(start=calendar[0], end=calendar[-1], symbols=targets.columns)
    accounting = run_target_backtest(
        targets, opens, events, quality, profile,
        signal_start=cfg.signal_start, signal_end=cfg.signal_end,
    )
    # 基准使用同一数据、费用与执行语义，仅将每个成员独立以总敞口 1 运行。
    # 基准不是组合的可加性成本归因，不能把各成员净值相加冒充组合。
    baselines = {}
    for name, member in members.items():
        baselines[name] = run_target_backtest(
            member.reindex(calendar), opens, events, quality,
            replace(profile, gross_exposure=1.0),
            signal_start=cfg.signal_start, signal_end=cfg.signal_end,
        )
    assert_source_unchanged([cfg.h5_path], source_before)
    if code_snapshot([item.path for item in cfg.factors]) != cfg.code_hashes:
        raise RuntimeError("code changed during portfolio evaluation")
    any_signal = any(item["daily"]["valid_signal"].any() for item in diagnostics.values())
    status = accounting["status"] if any_signal else "no_valid_signals"
    return {
        "status": status,
        "configuration": asdict(cfg),
        "knowledge_cutoff": cutoff.isoformat(),
        "environment": environment_metadata(),
        "source_stat": source_before,
        "input_reads": provider.reads,
        "value_pipeline_fingerprint": value_pipeline_fingerprint(),
        "values": values, "member_targets": members,
        "member_diagnostics": diagnostics,
        "targets": targets, "risk": risk,
        "accounting": accounting,
        "metrics": summarize_accounting(accounting),
        "baselines": baselines,
    }
```

重要：`as_of` 审计覆盖值而不改变原 `signal_end`；截断执行尾部不能伪装成新的策略终点并提前清仓。三情景继续各自独立记账。`known_segment` 仅来自已认证 ledger；中断当天 orders 里可能有已发生动作，不应拿它们计算完整日收益。

- [ ] 执行 `./.venv/bin/python -B -m pytest -q tests/portfolio_fixed tests/factor_common/test_target_backtest.py`，预期 PASS。
- [ ] A 验收：有效截面且约束不触发、质量完整的数据上，单因子与共享原接口的目标、订单、ledger、positions、funding 一致。无效截面、未知质量、缺价等差异属于本计划明确修正，单独测试，不要求与旧错误行为相等。
- [ ] 检查点提交信息：`feat: complete single-factor fixed portfolio migration`。

## 7. Task B1：固定目标合并与比例风控

**Files:** Modify `portfolio/portfolio_builder/fixed_weights.py`; Create `tests/portfolio_fixed/test_combination.py`。

- [ ] 新增以下全部手算测试，A 版本应因多成员限制或风控不支持而失败。

```python
# tests/portfolio_fixed/test_combination.py
from types import SimpleNamespace

import pandas as pd
import pytest

from portfolio.portfolio_builder.fixed_weights import combine_targets


def limits(**changes):
    values = dict(gross_limit=1.0, long_limit=1.0, short_limit=1.0,
                  single_limit=1.0, net_limit=1.0)
    values.update(changes)
    return SimpleNamespace(**values)


def matrix(row):
    return pd.DataFrame([row], index=pd.date_range("2024-01-01", periods=1, name="date"))


def test_opposite_members_net_to_zero_without_releveraging():
    first = matrix({"A": 0.5, "B": -0.5})
    second = -first
    targets, risk = combine_targets({"f1": first, "f2": second},
                                    {"f1": 0.5, "f2": 0.5}, limits())
    assert (targets == 0).all().all()
    assert risk.iloc[0]["post_gross"] == 0.0


def test_missing_member_budget_is_not_redistributed():
    first = matrix({"A": 0.5, "B": -0.5})
    second = first * 0
    targets, risk = combine_targets({"f1": first, "f2": second},
                                    {"f1": 0.5, "f2": 0.5}, limits())
    assert targets.loc["2024-01-01", "A"] == pytest.approx(0.25)
    assert risk.iloc[0]["post_gross"] == pytest.approx(0.5)


def test_single_cap_scales_whole_vector_and_preserves_neutrality():
    first = matrix({"A": 0.5, "B": -0.5})
    targets, risk = combine_targets({"f1": first}, {"f1": 1.0}, limits(single_limit=0.1))
    assert targets.loc["2024-01-01", "A"] == pytest.approx(0.1)
    assert targets.loc["2024-01-01", "B"] == pytest.approx(-0.1)
    assert risk.iloc[0]["scale"] == pytest.approx(0.2)


def test_zero_net_limit_does_not_destroy_already_neutral_book():
    first = matrix({"A": 0.5, "B": -0.5})
    targets, risk = combine_targets({"f1": first}, {"f1": 1.0}, limits(net_limit=0.0))
    pd.testing.assert_frame_equal(targets, first)


def test_axis_mismatch_is_not_silently_filled():
    first = matrix({"A": 0.5, "B": -0.5})
    second = first.rename(columns={"A": "C"})
    with pytest.raises(ValueError, match="axes"):
        combine_targets({"f1": first, "f2": second}, {"f1": 0.5, "f2": 0.5}, limits())
```

Run: `./.venv/bin/python -B -m pytest -q tests/portfolio_fixed/test_combination.py`。

- [ ] 用以下完整实现替换 A 阶段 `combine_targets`。同文件已有 NumPy、pandas 导入可直接复用。

```python
def combine_targets(members, allocations, config):
    if not members or set(members) != set(allocations):
        raise ValueError("members and allocations must have identical keys")
    if not np.isclose(sum(allocations.values()), 1.0, rtol=0, atol=1e-12):
        raise ValueError("allocations must sum to one")
    if any(not np.isfinite(x) or x <= 0 for x in allocations.values()):
        raise ValueError("allocations must be finite and positive")
    first = next(iter(members.values()))
    total = first * 0.0
    for name, frame in members.items():
        if not frame.index.equals(first.index) or not frame.columns.equals(first.columns):
            raise ValueError("member axes differ")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("member targets must be finite; invalid signals must be explicit zero")
        total = total + frame * allocations[name]
    gross = total.abs().sum(axis=1)
    long = total.clip(lower=0).sum(axis=1)
    short = -total.clip(upper=0).sum(axis=1)
    single = total.abs().max(axis=1)
    net = total.sum(axis=1).abs()
    scales = pd.DataFrame({"identity": 1.0}, index=total.index)
    for label, measure, cap in (
        ("gross", gross, config.gross_limit),
        ("long", long, config.long_limit),
        ("short", short, config.short_limit),
        ("single", single, config.single_limit),
        ("net", net, config.net_limit),
    ):
        # 浮点级的中性残差不触发净敞口为零时的全仓缩放。
        scales[label] = (cap / measure.where(measure > 1e-12)).fillna(1.0).clip(upper=1.0)
    scale = scales.min(axis=1).clip(lower=0.0, upper=1.0)
    output = total.mul(scale, axis=0)
    risk = pd.DataFrame({
        "pre_gross": gross, "pre_long": long, "pre_short": short,
        "pre_single": single, "pre_abs_net": net,
        "scale": scale, "binding_constraint": scales.idxmin(axis=1),
        "post_gross": output.abs().sum(axis=1),
        "post_long": output.clip(lower=0).sum(axis=1),
        "post_short": -output.clip(upper=0).sum(axis=1),
        "post_abs_net": output.sum(axis=1).abs(),
        "post_single": output.abs().max(axis=1),
    })
    return output, risk
```

只有统一缩小，不逐币 clip 后重新归一化，因此能保留目标的净额关系。该实现牺牲部分资本利用率换取可解释性，属于第一版明确取舍。

- [ ] 在 `test_combination.py` 增加端到端抵消账本测试：

```python
def test_equal_opposite_factors_have_no_combined_orders(config_dict, factor_paths):
    from portfolio.fixed_config import FixedConfig
    from portfolio.fixed_pipeline import run_fixed
    from portfolio.fixed_provenance import code_snapshot

    config_dict["factors"] = [{"path": str(path), "allocation": 0.5} for path in factor_paths]
    config_dict["code_hashes"] = code_snapshot(factor_paths)
    result = run_fixed(FixedConfig.from_dict(config_dict))
    account = result["accounting"]["scenarios"]["all_costs"]
    assert result["status"] == "complete"
    assert account["orders"].empty
    assert account["ledger"]["fee"].sum() == 0.0
    assert account["ledger"]["funding_cashflow"].sum() == 0.0
```

- [ ] 执行 `./.venv/bin/python -B -m pytest -q tests/portfolio_fixed tests/factor_common/test_target_backtest.py`，预期 PASS。A 等价测试不能在 B 实现后退化。
- [ ] 检查点提交信息：`feat: combine fixed factor targets with transparent exposure limits`。

## 8. Task B2：统一报告与完整结果目录

**Files:** Create `portfolio/fixed_artifacts.py`, `tests/portfolio_fixed/test_artifacts.py`。

### B2.1 输出约定

每次输出一个新目录，不能覆盖历史版本：

```text
reports/portfolio/<UTC时间>_<随机标识>/
  manifest.json
  metrics.json
  report.html
  targets.parquet
  risk.parquet
  input_reads.json
  members/<factor_id>/values.parquet
  members/<factor_id>/targets.parquet
  members/<factor_id>/diagnostics.parquet
  scenarios/<gross|trading_net|all_costs>/
    ledger.parquet
    orders.parquet
    positions.parquet
    valuation_prices.parquet
    funding.parquet
    funding_coverage.parquet
    observed_exposure.parquet
    diagnostics.json
  baselines/<factor_id>/<scenario>/ledger.parquet
  baselines/<factor_id>/metrics.json
  baseline_return_correlation.parquet
  baseline_return_pair_counts.parquet
  target_overlap.parquet
  artifact_complete.json
```

`artifact_complete.json` 只表示文件集写完，不表示账本完整或策略通过实盘验收。完整性以 manifest.status 及各 scenario.status 为准。个别基准不完整时，不给出其完整收益相关性；不得删掉失败基准使组合看起来更强。

报告必须列出：策略类型/方向、冻结配置、费用、调仓日历、目标风险限额、三情景指标、成员基准、已实现敞口诊断、约束缩放、失败与阻断订单、缺价/资金费缺失、未纳入的执行成本。基准按单位 gross 运行，组合可能 gross<1，必须标明暴露不可直接等量比较；不能将基准绝对收益差解释为 alpha 增量。

### B2.2 测试

- [ ] 新增测试并运行；预期首次缺少新模块。

```python
# tests/portfolio_fixed/test_artifacts.py
import json

import pandas as pd

from portfolio.fixed_artifacts import write_result
from portfolio.fixed_config import FixedConfig
from portfolio.fixed_pipeline import run_fixed


def test_complete_artifacts_are_readable_and_not_overwritten(config_dict, tmp_path):
    result = run_fixed(FixedConfig.from_dict(config_dict))
    first = write_result(result, tmp_path / "out")
    second = write_result(result, tmp_path / "out")
    assert first != second
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert (first / "artifact_complete.json").exists()
    # Parquet 不保留 pandas 的可选 freq 缓存；日期轴和值仍需完全一致。
    pd.testing.assert_frame_equal(pd.read_parquet(first / "targets.parquet"),
                                  result["targets"], check_freq=False)
    report = (first / "report.html").read_text()
    assert "研究回测" in report
    assert "365" in report
    assert "保证金" in report


def test_incomplete_full_metrics_remain_null(config_dict, tmp_path):
    result = run_fixed(FixedConfig.from_dict(config_dict), as_of="2024-01-24")
    assert result["status"] == "incomplete"
    output = write_result(result, tmp_path / "out")
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["all_costs"]["full"] is None
    assert metrics["all_costs"]["known_segment"] is not None
```

Run: `./.venv/bin/python -B -m pytest -q tests/portfolio_fixed/test_artifacts.py`。

### B2.3 完整写入与 HTML 实现

- [ ] 新增以下完整文件。首次写入在隐藏临时目录完成，最后改名发布；失败时保留临时目录用于诊断，不递归删除。Parquet 引擎沿用仓库已有依赖。

```python
# portfolio/fixed_artifacts.py
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from factor_common.storage import _json_safe
from portfolio.fixed_pipeline import summarize_accounting

TABLES = ("ledger", "orders", "positions", "valuation_prices", "funding", "funding_coverage")
LIMITATION = (
    "研究回测：日频 UTC，365 天年化，信号在下一日 open 基准执行。"
    "滑点按成交名义金额扣款；不是实际成交回放。未建模最小订单、数量精度、"
    "盘口冲击、保证金、强平、交易所故障或数据历史修订。"
    "基准各按单位总敞口运行，与受风险约束的组合暴露不同。"
    "历史分组成绩、冻结时间和 cutoff 一致性均不证明未来盈利。"
)


def write_json(path, payload):
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False,
                               indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def observed_exposure(account):
    positions = account["positions"]
    marked = positions.mul(account["valuation_prices"])
    # 未持仓列价格为 NaN 是允许的；持仓列缺价已由严格账本拦截。
    marked = marked.where(positions != 0.0, 0.0)
    weight = marked.div(account["ledger"]["equity"], axis=0)
    return pd.DataFrame({
        "gross": weight.abs().sum(axis=1),
        "long": weight.clip(lower=0).sum(axis=1),
        "short": -weight.clip(upper=0).sum(axis=1),
        "net": weight.sum(axis=1),
        "single": weight.abs().max(axis=1),
    })


def comparison_tables(result):
    returns = {}
    for name, baseline in result["baselines"].items():
        account = baseline["scenarios"]["all_costs"]
        if account["status"] == "complete":
            returns[name] = account["ledger"]["return"]
    names = list(result["baselines"])
    panel = pd.DataFrame(returns).reindex(columns=names)
    correlation = panel.corr(min_periods=20)
    valid = panel.notna().astype("int64")
    counts = valid.T.dot(valid)
    overlap = pd.DataFrame(np.nan, index=names, columns=names)
    for first in names:
        left = result["member_targets"][first]
        for second in names:
            right = result["member_targets"][second]
            active = (left.abs().sum(axis=1) > 0) & (right.abs().sum(axis=1) > 0)
            shared = np.minimum(left.abs(), right.abs()).where(left * right > 0, 0.0).sum(axis=1)
            denominator = 0.5 * (left.abs().sum(axis=1) + right.abs().sum(axis=1))
            series = shared.div(denominator.where(denominator > 0))
            overlap.loc[first, second] = series[active].mean()
    return correlation, counts, overlap


def report_html(result, correlation, counts, overlap):
    rows = []
    for name, item in result["metrics"].items():
        rows.append({"scenario": name, "status": item["status"], **(item["full"] or {})})
    summaries = pd.DataFrame(rows).to_html(index=False, escape=True)
    baselines = []
    for name, baseline in result["baselines"].items():
        item = summarize_accounting(baseline)["all_costs"]
        baselines.append({"factor": name, "status": item["status"], **(item["full"] or {})})
    diagnostics = {
        name: account["diagnostics"]
        for name, account in result["accounting"]["scenarios"].items()
    }
    config = escape(json.dumps(_json_safe(result["configuration"]), ensure_ascii=False, indent=2))
    diag = escape(json.dumps(_json_safe(diagnostics), ensure_ascii=False, indent=2))
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>Fixed Portfolio Research</title><body>'
        '<h1>固定组合研究回测</h1>'
        f'<p>运行状态：{escape(result["status"])}</p><p>{escape(LIMITATION)}</p>'
        '<h2>组合三情景</h2>' + summaries
        + '<p>全期指标为空时，只能在 metrics.json 中查看明确标记的 known_segment。</p>'
        + '<h2>单位总敞口单因子基准</h2>' + pd.DataFrame(baselines).to_html(index=False, escape=True)
        + '<h2>基准净收益相关性（至少 20 个共同日）</h2>' + correlation.to_html(escape=True)
        + '<h3>共同有效日数</h3>' + counts.to_html(escape=True)
        + '<h2>成员目标同向持仓重合度</h2>' + overlap.to_html(escape=True)
        + '<p>相关性和重合度仅用于事后诊断，未用于选择成员或调整权重。</p>'
        + '<h2>目标缩放摘要</h2>' + result["risk"].describe(include="all").to_html(escape=True)
        + '<h2>实际持仓敞口摘要（含漂移与扣费影响）</h2>'
        + observed_exposure(result["accounting"]["scenarios"]["all_costs"]).describe().to_html(escape=True)
        + '<h2>配置与版本</h2><pre>' + config + '</pre>'
        + '<h2>账本诊断</h2><pre>' + diag + '</pre>'
        + '<p><a href="manifest.json">Manifest</a> · <a href="metrics.json">Metrics</a> · '
        '<a href="scenarios/all_costs/ledger.parquet">净账本</a></p></body></html>'
    )


def write_result(result, output_root):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
    staged = root / (".partial_" + run_id)
    final = root / run_id
    staged.mkdir(exist_ok=False)
    scalar_keys = ("status", "configuration", "knowledge_cutoff", "environment",
                   "source_stat", "value_pipeline_fingerprint")
    manifest = {key: result[key] for key in scalar_keys}
    manifest.update({"run_id": run_id, "research_only": True,
                     "cutoff_audit_attached": False, "limitations": LIMITATION})
    write_json(staged / "manifest.json", manifest)
    write_json(staged / "metrics.json", result["metrics"])
    write_json(staged / "input_reads.json", result["input_reads"])
    result["targets"].to_parquet(staged / "targets.parquet")
    result["risk"].to_parquet(staged / "risk.parquet")
    for name, values in result["values"].items():
        path = staged / "members" / name
        path.mkdir(parents=True)
        values.to_parquet(path / "values.parquet")
        result["member_targets"][name].to_parquet(path / "targets.parquet")
        result["member_diagnostics"][name]["daily"].to_parquet(path / "diagnostics.parquet")
        write_json(path / "metadata.json", {
            key: value for key, value in result["member_diagnostics"][name].items() if key != "daily"
        })
    for name, account in result["accounting"]["scenarios"].items():
        path = staged / "scenarios" / name
        path.mkdir(parents=True)
        for table in TABLES:
            account[table].to_parquet(path / f"{table}.parquet")
        observed_exposure(account).to_parquet(path / "observed_exposure.parquet")
        write_json(path / "diagnostics.json", account["diagnostics"])
    for factor, baseline in result["baselines"].items():
        path = staged / "baselines" / factor
        path.mkdir(parents=True)
        write_json(path / "metrics.json", summarize_accounting(baseline))
        for scenario, account in baseline["scenarios"].items():
            child = path / scenario
            child.mkdir()
            account["ledger"].to_parquet(child / "ledger.parquet")
            write_json(child / "diagnostics.json", account["diagnostics"])
    correlation, counts, overlap = comparison_tables(result)
    correlation.to_parquet(staged / "baseline_return_correlation.parquet")
    counts.to_parquet(staged / "baseline_return_pair_counts.parquet")
    overlap.to_parquet(staged / "target_overlap.parquet")
    (staged / "report.html").write_text(report_html(result, correlation, counts, overlap), encoding="utf-8")
    write_json(staged / "artifact_complete.json", {"run_id": run_id, "written": True})
    staged.rename(final)
    return final
```

- [ ] 运行 artifacts 测试，预期 PASS。再检查 `metrics.json` 不包含非标准 JSON 的 NaN/Infinity；HTML 的配置文本经过 escape，不能直接插入未转义名称。
- [ ] 检查点提交信息：`feat: persist auditable fixed portfolio research artifacts`。

## 9. Task B3：从原始输入重算的 cutoff 审计

**Files:** Create `portfolio/fixed_audit.py`, `tests/portfolio_fixed/test_cutoff.py`。

### B3.1 审计必须比较的对象

- 因子值：相同历史轴、NaN mask、数值。
- 每个成员目标与组合目标：相同历史轴和数值，不允许靠 inner join 丢掉差异。
- 三情景 ledger：截断日之前完整记录逐项比较。
- 三情景 orders：日期、币种、方向、数量、价格、费用、状态全部比较。
- 三情景 positions：验证数量没有受到未来数据影响。
- 截断回测因尾部未到而 `incomplete` 是预期，不得为了消除该状态提前清仓；完整基准本身必须 complete。
- 对全量未来才出现的币，只有其历史值全 NaN、历史目标/持仓全零时允许移除该列，必须记录该事实。其他轴差异直接失败。

### B3.2 测试

- [ ] 新增测试，首次预期导入失败。

```python
# tests/portfolio_fixed/test_cutoff.py
import pandas as pd
import pytest

from portfolio.fixed_audit import audit_fixed, compare_matrix
from portfolio.fixed_config import FixedConfig


def test_real_provider_cutoffs_recompute_pipeline(config_dict):
    audit = audit_fixed(FixedConfig.from_dict(config_dict), ["2024-01-23", "2024-01-24"])
    assert audit["status"] == "verified"
    assert len(audit["cutoffs"]) == 2
    assert all(item["max_abs_diff"] <= 1e-10 for item in audit["cutoffs"])


def test_missing_nonzero_target_column_is_not_ignored():
    index = pd.date_range("2024-01-01", periods=1, name="date")
    full = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=index)
    cut = full[["A"]]
    with pytest.raises(AssertionError):
        compare_matrix(full, cut, index[0], absent_zero=True)


def test_nan_mask_change_is_not_ignored():
    index = pd.date_range("2024-01-01", periods=1, name="date")
    full = pd.DataFrame({"A": [float("nan")]}, index=index)
    cut = pd.DataFrame({"A": [0.0]}, index=index)
    with pytest.raises(AssertionError):
        compare_matrix(full, cut, index[0])


def test_empty_cutoff_list_cannot_claim_verified(config_dict):
    with pytest.raises(ValueError, match="cutoff"):
        audit_fixed(FixedConfig.from_dict(config_dict), [])
```

Run: `./.venv/bin/python -B -m pytest -q tests/portfolio_fixed/test_cutoff.py`。

### B3.3 完整审计实现

- [ ] 新增以下完整文件。

```python
# portfolio/fixed_audit.py
import numpy as np
import pandas as pd

from portfolio.fixed_config import daily_date
from portfolio.fixed_pipeline import run_fixed


def compare_matrix(full, cut, cutoff, *, absent_zero=False):
    prefix = full.loc[:cutoff].copy()
    excluded = []
    for column in prefix.columns.difference(cut.columns):
        series = prefix[column]
        acceptable = series.isna().all()
        if absent_zero:
            acceptable = acceptable or (series.fillna(0.0) == 0.0).all()
        assert acceptable, f"removed historical column {column}"
        excluded.append(str(column))
    assert set(cut.columns).issubset(prefix.columns), "cutoff introduced unexpected column"
    prefix = prefix.drop(columns=excluded)
    pd.testing.assert_index_equal(prefix.index, cut.index)
    pd.testing.assert_index_equal(prefix.columns, cut.columns)
    assert prefix.isna().equals(cut.isna()), "NaN mask changed"
    pd.testing.assert_frame_equal(prefix, cut, check_dtype=False, rtol=0.0, atol=1e-10)
    numeric = prefix.select_dtypes(include=[np.number]).columns
    delta = (prefix[numeric] - cut[numeric]).abs().to_numpy(dtype=float)
    finite = delta[np.isfinite(delta)]
    return {"max_abs_diff": float(finite.max()) if finite.size else 0.0,
            "excluded_future_only_columns": excluded, "rows": len(prefix)}


def audit_fixed(config, cutoffs):
    if not cutoffs:
        raise ValueError("at least one cutoff required")
    days = sorted({daily_date(value) for value in cutoffs})
    for day in days:
        if not daily_date(config.signal_start) + pd.Timedelta(days=1) <= day <= daily_date(config.signal_end):
            raise ValueError("cutoff must lie inside the active evaluation interval after first signal")
    full = run_fixed(config)
    if full["status"] != "complete":
        raise ValueError("full evaluation must be complete before cutoff certification")
    records = []
    for day in days:
        cut = run_fixed(config, as_of=day)
        if cut["source_stat"] != full["source_stat"]:
            raise RuntimeError("input source changed between cutoff replays")
        checks = {}
        for name in full["values"]:
            checks[f"values/{name}"] = compare_matrix(full["values"][name], cut["values"][name], day)
            checks[f"member_targets/{name}"] = compare_matrix(
                full["member_targets"][name], cut["member_targets"][name], day, absent_zero=True,
            )
        checks["targets"] = compare_matrix(full["targets"], cut["targets"], day, absent_zero=True)
        for name in ("gross", "trading_net", "all_costs"):
            first = full["accounting"]["scenarios"][name]
            second = cut["accounting"]["scenarios"][name]
            checks[f"{name}/ledger"] = compare_matrix(first["ledger"], second["ledger"], day)
            checks[f"{name}/positions"] = compare_matrix(
                first["positions"], second["positions"], day, absent_zero=True,
            )
            before = first["orders"].loc[first["orders"].date <= day]
            after = second["orders"].loc[second["orders"].date <= day]
            keys = ["date", "instrument", "side"]
            before = before.sort_values(keys, kind="mergesort").reset_index(drop=True)
            after = after.sort_values(keys, kind="mergesort").reset_index(drop=True)
            pd.testing.assert_frame_equal(before, after, check_dtype=False, rtol=0, atol=1e-10)
            numeric = before.select_dtypes(include=[np.number]).columns
            delta = (before[numeric] - after[numeric]).abs().to_numpy(dtype=float)
            finite = delta[np.isfinite(delta)]
            checks[f"{name}/orders"] = {
                "rows": len(before), "equal_within_tolerance": True,
                "max_abs_diff": float(finite.max()) if finite.size else 0.0,
            }
        records.append({"cutoff": day.isoformat(), "checks": checks,
                        "source_stat": cut["source_stat"], "input_reads": cut["input_reads"],
                        "max_abs_diff": max(item["max_abs_diff"] for item in checks.values())})
    return {
        "status": "verified", "cutoffs": records,
        "code_hashes": config.code_hashes,
        "configuration": full["configuration"],
        "source_stat": full["source_stat"],
        "input_reads": full["input_reads"],
        "limitations": "验证历史计算一致性；不验证历史数据实际发布时间或未来收益。",
    }
```

- [ ] 执行 cutoff 测试，预期 PASS，不允许空比较自动通过。记录实际数值列的最大差异并保留整表断言；`equal_within_tolerance` 不表示逐比特相等。
- [ ] 执行 `./.venv/bin/python -B -m pytest -q tests/portfolio_fixed tests/factor_common/test_target_backtest.py`，预期 PASS。
- [ ] 检查点提交信息：`test: certify fixed portfolio historical decisions with cutoff replay`。

## 10. Task B4：CLI、真实数据验收与使用文档

**Files:** Create `portfolio/main_fixed.py`, `tests/portfolio_fixed/test_cli.py`; Modify `portfolio/README.md`。

### B4.1 CLI 验收测试

- [ ] 新增测试文件，预期首次导入失败。

```python
# tests/portfolio_fixed/test_cli.py
import json

from portfolio.main_fixed import main


def test_run_cli_writes_report_and_returns_zero(config_dict, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict))
    output = tmp_path / "reports"
    assert main(["run", "--config", str(path), "--output-root", str(output)]) == 0
    assert len(list(output.glob("*/report.html"))) == 1


def test_run_cli_incomplete_returns_two(config_dict, tmp_path):
    config_dict["as_of"] = "2024-01-24"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict))
    output = tmp_path / "reports"
    assert main(["run", "--config", str(path), "--output-root", str(output)]) == 2
    assert len(list(output.glob("*/report.html"))) == 1


def test_freeze_does_not_overwrite_existing_config(fixed_h5, factor_paths, tmp_path):
    path = tmp_path / "config.json"
    args = ["freeze", "--h5", str(fixed_h5), "--factor", str(factor_paths[0]),
            "--allocation", "1", "--start", "2024-01-22", "--end", "2024-01-24",
            "--as-of", "2024-01-26", "--output", str(path)]
    assert main(args) == 0
    original = path.read_bytes()
    assert main(args) == 1
    assert path.read_bytes() == original


def test_audit_cli_writes_verified_receipt(config_dict, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_dict))
    receipt = tmp_path / "audit.json"
    assert main(["audit", "--config", str(path), "--cutoff", "2024-01-23",
                 "--output", str(receipt)]) == 0
    assert json.loads(receipt.read_text())["status"] == "verified"
```

Run: `./.venv/bin/python -B -m pytest -q tests/portfolio_fixed/test_cli.py`。

### B4.2 CLI 完整实现

- [ ] 新增以下完整文件。约定只支持 `python -m portfolio.main_fixed`，不重复添加 sys.path 修补。

```python
# portfolio/main_fixed.py
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from portfolio.fixed_artifacts import write_result
from portfolio.fixed_audit import audit_fixed
from portfolio.fixed_config import FactorAllocation, FixedConfig
from portfolio.fixed_pipeline import run_fixed
from portfolio.fixed_provenance import code_snapshot
from portfolio.factor_pool.module_loader import load_members


def parser():
    root = argparse.ArgumentParser(description="Fixed daily perpetual portfolio research")
    commands = root.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Freeze modules and research configuration")
    freeze.add_argument("--h5", required=True)
    freeze.add_argument("--factor", action="append", required=True)
    freeze.add_argument("--allocation", action="append", type=float, required=True)
    freeze.add_argument("--start", required=True)
    freeze.add_argument("--end", required=True)
    freeze.add_argument("--as-of", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--rebalance-days", type=int, default=1)
    freeze.add_argument("--anchor-date", default="2024-01-01")
    freeze.add_argument("--n-groups", type=int, default=5)
    freeze.add_argument("--min-valid-instruments", type=int, default=10)
    freeze.add_argument("--fee-rate", type=float, default=0.0005)
    freeze.add_argument("--slippage", type=float, default=0.001)
    for name, default in (("gross-limit", 0.5), ("long-limit", 0.25),
                          ("short-limit", 0.25), ("single-limit", 0.05), ("net-limit", 0.05)):
        freeze.add_argument("--" + name, type=float, default=default)
    run = commands.add_parser("run", help="Run a frozen research portfolio")
    run.add_argument("--config", required=True)
    run.add_argument("--output-root", default="reports/portfolio")
    audit = commands.add_parser("audit", help="Recompute historical prefixes from cutoff providers")
    audit.add_argument("--config", required=True)
    audit.add_argument("--cutoff", action="append", required=True)
    audit.add_argument("--output", required=True)
    return root


def write_new_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "freeze":
            if len(args.factor) != len(args.allocation):
                raise ValueError("each factor requires one allocation")
            paths = [str(Path(path).resolve(strict=True)) for path in args.factor]
            config = FixedConfig(
                h5_path=str(Path(args.h5).resolve(strict=True)),
                signal_start=args.start, signal_end=args.end, as_of=args.as_of,
                factors=tuple(FactorAllocation(path, weight)
                              for path, weight in zip(paths, args.allocation)),
                code_hashes=code_snapshot(paths),
                frozen_at=datetime.now(timezone.utc).isoformat(),
                rebalance_days=args.rebalance_days, anchor_date=args.anchor_date,
                n_groups=args.n_groups, min_valid_instruments=args.min_valid_instruments,
                fee_rate=args.fee_rate, slippage=args.slippage,
                gross_limit=args.gross_limit, long_limit=args.long_limit,
                short_limit=args.short_limit, single_limit=args.single_limit,
                net_limit=args.net_limit,
            )
            load_members(config)
            write_new_json(args.output, asdict(config))
            print(f"Frozen research config: {args.output}")
            return 0
        config = FixedConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
        if args.command == "audit":
            receipt = audit_fixed(config, args.cutoff)
            write_new_json(args.output, receipt)
            print(f"Cutoff audit verified: {args.output}")
            return 0
        result = run_fixed(config)
        output = write_result(result, args.output_root)
        print(f"Research status: {result['status']}")
        print(f"Report: {output / 'report.html'}")
        return 0 if result["status"] == "complete" else 2
    except (ValueError, TypeError, OSError, RuntimeError, AssertionError) as exc:
        print(f"Fixed portfolio failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

退出码：0=账本完成/冻结完成/审计完成（取决于子命令）；1=配置、I/O、来源变化或断言错误；2=结果已落盘但账本不完整或没有有效信号。不能把退出码 0 解释为策略通过实盘。

### B4.3 增补 README 的具体内容

- [ ] 在 `portfolio/README.md` 开头新增以下内容，不删除旧使用说明；将旧内容置于明确的 Legacy research 小节下。

```markdown
# Portfolio Research

日频固定组合的新入口是 `./.venv/bin/python -m portfolio.main_fixed`。
它按明确模块路径加载因子，合并各成员的方向多空目标，使用共享严格账本。
仅支持永续多空研究；不接实盘下单，也不模拟保证金和强平。
freeze、run、audit 均应在新 Python 进程执行；修改代码后不能沿用 notebook
已缓存的旧模块来宣称运行了新代码，必须重启解释器并重新冻结。

流程为 `freeze` → `run` → `audit`。freeze 记录代码版本和固定资本权重，
不执行因子选择；任何代码变化都需要生成新配置，旧配置不会被覆盖。
所有成员使用同一日频调仓日历，信号日 t 的目标在 t+1 open 基准执行。
缺失成员不沿用过期信号，也不把其预算重新分配给其他成员。

结果写入 `reports/portfolio/<run_id>/`。检查 `manifest.json` 的运行状态、
`metrics.json` 的各情景状态、订单/资金费/风险诊断，再查看 HTML。
`artifact_complete.json` 仅表示落盘完成；cutoff 审计以独立 audit JSON 为凭证，
必须核对其中 configuration、code_hashes、source_stat 与运行 manifest 一致。

旧 `main.py`、`main_hedged.py`、`main_pathB.py` 保留历史复现。
它们与新入口具有不同的持仓、计费、时间和异常数据规则，结果不可直接拼接。
旧测试通过不代表可用于实盘；旧入口不属于新固定组合的验收范围。

执行计划与完整命令见 `docs/superpowers/plans/2026-09-17-portfolio-fixed-ab-implementation.md`。

## Legacy research
```

README 的具体旧标题可调整为三级标题以保持结构；不改旧命令行为。若输出路径不存在或权限不足，CLI 必须报错，不能把报告写到意外目录。

### B4.4 真实数据 A 验收命令

- [ ] 先检查真实 H5 日期范围与读取能力，不更新 H5：

```bash
./.venv/bin/python -B - <<'PY'
from factor_common.data_provider import DataProvider
provider = DataProvider("data/crypto_quant.h5", as_of="2026-09-02")
print("market range:", provider.get_time_range())
print("symbol count:", len(provider.symbols))
PY
```

- [ ] 在全部代码修改完成后冻结一个单成员配置。以下 GP 仅作为迁移测试样本，不代表重新宣布该因子有效；费用与滑点为当前研究口径，不是交易所报价。

```bash
./.venv/bin/python -B -m portfolio.main_fixed freeze \
  --h5 data/crypto_quant.h5 \
  --factor factor_analyse/factor_mining/GP_Factor/GP_064185107a8f267e.py \
  --allocation 1 \
  --start 2024-01-01 --end 2026-08-31 --as-of 2026-09-02 \
  --n-groups 5 --min-valid-instruments 5 \
  --rebalance-days 1 --anchor-date 2024-01-01 \
  --fee-rate 0.0005 --slippage 0.001 \
  --gross-limit 1 --long-limit 1 --short-limit 1 --single-limit 1 --net-limit 1 \
  --output portfolio/output/ab_single_frozen.json

./.venv/bin/python -B -m portfolio.main_fixed run \
  --config portfolio/output/ab_single_frozen.json \
  --output-root reports/portfolio

./.venv/bin/python -B -m portfolio.main_fixed audit \
  --config portfolio/output/ab_single_frozen.json \
  --cutoff 2024-06-30 --cutoff 2025-06-30 --cutoff 2026-06-30 \
  --output portfolio/output/ab_single_cutoff.json
```

预期：数据完整时 run 返回 0 并写出新结果；审计全部轴/mask/历史账本一致，最大绝对差异在 1e-10 以内。真实数据存在缺价、未知资金费覆盖、末尾不足时允许得到诊断性的 incomplete，但这意味着**真实数据验收未通过**，不是可以跳过的测试。不得通过关闭资金费、放宽质量门槛或替换缺失价格来“修绿”。

若文件已存在，换一个新文件名，不能删掉旧配置/审计覆盖重跑。若数据截止日不足上述窗口，明确记录较短窗口及其原因，再生成不同版本配置；不静默缩短。

### B4.5 真实数据 B 验收命令

- [ ] 生成双成员固定等权研究配置，检查组合净额与风险缩放。两个成员是接线测试样本，不能根据本次报告重新优化其比例。

```bash
./.venv/bin/python -B -m portfolio.main_fixed freeze \
  --h5 data/crypto_quant.h5 \
  --factor factor_analyse/factor_mining/GP_Factor/GP_064185107a8f267e.py \
  --allocation 0.5 \
  --factor factor_analyse/factor_mining/Retail_Friction_Illiquidity_Factor.py \
  --allocation 0.5 \
  --start 2024-01-01 --end 2026-08-31 --as-of 2026-09-02 \
  --n-groups 5 --min-valid-instruments 10 \
  --rebalance-days 1 --anchor-date 2024-01-01 \
  --fee-rate 0.0005 --slippage 0.001 \
  --gross-limit 0.5 --long-limit 0.25 --short-limit 0.25 --single-limit 0.05 --net-limit 0.05 \
  --output portfolio/output/ab_pair_frozen.json

./.venv/bin/python -B -m portfolio.main_fixed run \
  --config portfolio/output/ab_pair_frozen.json \
  --output-root reports/portfolio

./.venv/bin/python -B -m portfolio.main_fixed audit \
  --config portfolio/output/ab_pair_frozen.json \
  --cutoff 2024-06-30 --cutoff 2025-06-30 --cutoff 2026-06-30 \
  --output portfolio/output/ab_pair_cutoff.json
```

必须读取 `risk.parquet` 检查所有目标上限；读取 `observed_exposure.parquet` 区分调仓目标与实际漂移；核对至少首次开仓、一次换仓、最后清仓的数量和现金流。组合毛敞口低于上限可能来自多空抵消或单币约束，不应当作需要自动补满的“资金闲置 bug”。

### B4.6 最终自动验证

- [ ] 执行新路径、共享模块与受共享修改影响的 GP 测试。现有 unrelated 失败必须与 A0 基线对照归因，不可直接删除测试。

```bash
./.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/portfolio_fixed tests/factor_common tests/genetic_algorithm tests/test_portfolio_*.py

./.venv/bin/python -B - <<'PY'
import ast
from pathlib import Path
paths = [Path("factor_common/backtest.py")]
paths += sorted(Path("portfolio").rglob("*.py"))
paths += sorted(Path("tests/portfolio_fixed").rglob("*.py"))
for path in paths:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"syntax checked: {len(paths)} files")
PY

git diff --check
git diff --stat
git status --short
```

- [ ] 检查点提交信息：`feat: deliver fixed portfolio CLI and A-B acceptance workflow`。

## 11. 补充必测边界与具体测试代码

本节不是可选清单；以下测试追加到指定文件，完成后才能勾选 B 总验收。

### 11.1 资金费正负号与退出后不再结算

追加至 `tests/factor_common/test_target_backtest.py`：

```python
def test_positive_funding_pays_long_and_credits_short():
    values, opens = basic_frames()
    profile = _profile(include_funding=True)
    quality = _quality(3, opens.columns)
    quality["has_complete_kline"] = True
    events = pd.DataFrame({
        "funding_time": pd.to_datetime(["2024-01-02T08:00:00Z"] * 2),
        "instrument": ["A", "B"], "funding_rate": [0.001, 0.002],
        "mark_price": [100.0, 100.0],
    })
    result = run_target_backtest(target_weights(values, profile)["long_short"],
        opens, events, quality, profile,
        signal_start="2024-01-01", signal_end="2024-01-01")
    account = result["scenarios"]["all_costs"]
    assert account["ledger"]["funding_cashflow"].sum() == pytest.approx(0.0005)
    assert account["diagnostics"]["final_quantities"] == {}


def test_unknown_funding_is_not_zero_cost_success():
    values, opens = basic_frames()
    profile = _profile(include_funding=True)
    quality = _quality(3, opens.columns)
    quality["has_complete_kline"] = True
    quality.loc[(pd.Timestamp("2024-01-02"), "A"), "funding_coverage_status"] = "unknown"
    account = run_targets(target_weights(values, profile)["long_short"],
                           opens, profile, quality)["scenarios"]["all_costs"]
    assert account["status"] == "incomplete"
    assert account["diagnostics"]["halt_reason"] == "unresolved_funding_coverage"
```

这里 `_quality` 明确构造的是合成 coverage 证据；真实数据严禁人工标记 complete。更多边界时点（入场前 00:00、退出当日 00:00、退出后 intraday）继续由现有 `tests/factor_common/test_funding.py` 和 `test_backtest.py` 验证；先阅读它们的实际断言，若执行接口变动导致不再覆盖新入口，应在同一测试内参数化两入口，而不是复制费用函数。

### 11.2 共享绩效必须保留首日损失

追加至 `tests/portfolio_fixed/test_artifacts.py`：

```python
def test_shared_metrics_include_initial_nav_and_real_compounding():
    import pytest
    from factor_common.metrics import summarize_returns
    first = summarize_returns(pd.Series([-0.1, 0.0]), periods_per_year=365)
    assert first["max_drawdown"] == pytest.approx(0.1)
    second = summarize_returns(pd.Series([0.1, -0.1]), periods_per_year=365)
    assert second["total_return"] == pytest.approx(-0.01)
    assert second["annual_return"] < 0
```

两天收益的年化只能用来检验公式，不能作为统计上有意义的年化预测。

### 11.3 运行过程中数据文件变化必须拒绝

追加至 `tests/portfolio_fixed/test_modules.py`：

```python
def test_changed_source_stat_is_rejected(tmp_path):
    from factor_common.storage import snapshot_source_stats
    from portfolio.fixed_provenance import assert_source_unchanged
    path = tmp_path / "source.bin"
    path.write_bytes(b"one")
    before = snapshot_source_stats([path])
    path.write_bytes(b"changed-size")
    with pytest.raises(RuntimeError, match="source changed"):
        assert_source_unchanged([path], before)
```

### 11.4 非调仓日保持数量，日历不依赖样本行号

追加至 `tests/factor_common/test_target_backtest.py`：

```python
def test_three_day_schedule_holds_quantities_between_rebalances():
    values, opens = _frames(8, {
        2: {"A": 2.0, "B": 1.0}, 5: {"A": 1.0, "B": 2.0},
    }, {i: {"A": 100.0 + i, "B": 100.0 - i} for i in range(8)})
    profile = _profile(rebalance_days=3, anchor_date="2024-01-01")
    quality = _quality(8, opens.columns)
    quality["has_complete_kline"] = True
    targets = target_weights(values, profile)["long_short"]
    result = run_target_backtest(targets, opens, EMPTY_EVENTS, quality, profile,
                                 signal_start="2024-01-01", signal_end="2024-01-06")
    account = result["scenarios"]["gross"]
    positions = account["positions"]
    pd.testing.assert_series_equal(positions.loc["2024-01-04"], positions.loc["2024-01-05"], check_names=False)
    pd.testing.assert_series_equal(positions.loc["2024-01-05"], positions.loc["2024-01-06"], check_names=False)
    assert set(account["orders"].date) == {pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-07")}
    assert account["diagnostics"]["halt_reason"] == "missing_tail"
```

## 12. 最终验收矩阵与实施交接

| 要求 | 实现任务 | 必须保留的证据 |
|---|---|---|
| GP 显式加载与方向 | A1/A2 | 负方向 fixture 与真实模块测试 |
| 不读取旧 CSV 别名猜方向 | A2/A4 | import/调用链检查 |
| 代码与数据版本记录 | A2/B2/B3 | frozen config、manifest、input reads、audit receipt |
| 单因子等价 | A3/A4 | orders/ledger/positions/funding 比较 |
| 严格缺价而非假清仓 | A3 | incomplete、保留数量、无虚构订单 |
| 无信号不满仓 | A4 | 全 NaN、全同值 → 全零目标 |
| 固定资本权重不调优 | B1 | config、手算目标、无标签依赖 |
| 净额成交与真实费用 | A3/B1 | 反向成员抵消、开平仓计费测试 |
| 目标风险约束 | B1/B2 | risk 表与实际漂移敞口表 |
| 资金费质量与方向 | A3/11.1 | 事件现金流、coverage、失败状态 |
| 日历和延迟 | A3/B3/11.4 | 固定锚点、数量保持、cutoff |
| 统一指标和报告 | B2/11.2 | 首日回撤、复利、null、不覆盖结果 |
| 可运行入口 | B4 | freeze/run/audit 三子命令与退出码 |
| 历史兼容 | A3/B4 | 旧测试基线对照与 legacy README |

- [ ] A 的合成数据等价测试全部通过。
- [ ] B 的合成组合、风险、资金费、异常状态、CLI、cutoff 测试全部通过。
- [ ] 真实 H5 完成至少一个单因子与一个双因子运行；若因质量不完整，列出具体日期/币种/原因并将该项保持未完成。
- [ ] 真实 cutoff 审计通过，且审计与运行的 configuration、code_hashes、source_stat 相符。
- [ ] 单独报告老入口与新入口的差异，不复用老 HTML 成绩作为新结果。
- [ ] 报告中明确统计窗口已经被研究查看过，冻结配置不恢复样本外独立性。
- [ ] 没有改动因子公式、factor_miner 契约、实盘权限或交易所账户。
- [ ] 最终回复列出：变更文件、测试命令/结果、A/B 输出目录、cutoff 最大差异、数据质量未完成项与实盘未覆盖项。

实施不是以“收益更好”为验收标准。A/B 的完成条件是同一策略可复现、组合目标可解释、现金流可核对、历史决策不随未来输入变化。若正确记账后收益下降，应保留结果并解释原因，不回调费用或筛选规则来维持原成绩。

### 文档自身的检查说明

本文为执行计划，文件创建/替换代码是实施时的目标内容；本轮写文档不等于已修改源码。实施者必须按任务执行红灯→实现→绿灯，不可把文档中的 Expected PASS 当作实际测试记录。审计、实盘执行、历史数据可用性分别属于不同证据层级。

**2026-09-17 文档自检结果：** 30 个 Python fenced 片段完成 AST 语法解析；将新模块和共享核心修改在内存中组合，使用临时 H5 与临时因子执行 45 项参考代码检查，结果 45 通过、0 失败。参数化的无效配置案例逐项调用，已计入 45 项；没有把未执行案例计为通过。检查发现并修正了临时因子源代码的多余括号、Parquet 往返不保留 pandas `freq` 缓存的断言，以及跨 cutoff 回放的数据源一致性检查。

自检没有把新模块写入项目源码目录，没有运行本计划的真实 H5 全期组合回测，没有执行实施后的完整 pytest 回归。临时检查的 CLI 通过 `main(argv)` 调用，实际新进程 `python -m portfolio.main_fixed` 仍须在 B4 验收。提交前仍须逐项完成任务和验收矩阵，不得仅引用这份文档的自检记录宣称 A/B 已交付。
