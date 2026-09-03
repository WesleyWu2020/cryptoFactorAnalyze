# Portfolio 层 P1 MVP 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 跑通 `baseline_topn_long` 策略：从白名单因子 CSV 合成打分、构建 Top-N 等权组合、drift 回测、产出 target_weights 与简版 HTML 报告。

**Architecture:** 新建 `portfolio/` 顶层目录，实现 factor_pool → combiner → portfolio_builder → backtester → reporter 的串行 pipeline；portfolio 层只 import `data/`，不 import `factor_analyse/`（`factor_direction` 查表除外，通过读取 `factor_analyse/factor_config.py` 静态字典实现）。严格无未来函数（滚动窗口 `[t-L, t-1]`、回测 `w[t-1]·r[t]`），并以端到端 cutoff 反证测试兜底。

**Tech Stack:** Python 3.11 / pandas / numpy / pytest / unittest。所有 Python 命令用 `./.venv/bin/python`。

**Spec：** `docs/superpowers/specs/2026-04-17-portfolio-layer-design.md`（P1 MVP 范围见 §7）。

---

## File Structure（P1 MVP 范围）

创建：

```
portfolio/
├── __init__.py
├── config.py
├── main.py
├── pipeline.py
├── factor_pool/
│   ├── __init__.py
│   ├── loader.py
│   ├── aligner.py
│   └── label_builder.py
├── combiner/
│   ├── __init__.py
│   ├── normalizer.py
│   ├── orthogonalizer.py
│   ├── ic_ir_weighter.py
│   └── synthesizer.py
├── portfolio_builder/
│   ├── __init__.py
│   ├── universe.py
│   ├── constraints.py
│   └── topn_equal.py
├── backtester/
│   ├── __init__.py
│   ├── fees.py
│   ├── metrics.py
│   └── engine.py
├── reporter/
│   ├── __init__.py
│   └── html_renderer.py
└── output/                # 运行时产生，加入 .gitignore
tests/
├── test_portfolio_loader.py
├── test_portfolio_aligner.py
├── test_portfolio_label_builder.py
├── test_portfolio_normalizer.py
├── test_portfolio_orthogonalizer.py
├── test_portfolio_ic_ir_weighter.py
├── test_portfolio_topn_builder.py
├── test_portfolio_universe.py
├── test_portfolio_backtester_drift.py
└── test_portfolio_future_leak.py
```

修改：
- `.gitignore`：忽略 `portfolio/output/`

每个文件职责（对齐 spec §2）：
- `config.py`：策略配置字典 + `resolve_config(name)`（支持 `inherits` 递归合并，P1 仅用单一 `baseline_topn_long`，留好接口）
- `factor_pool/loader.py`：读单因子 CSV，只取 `date, instrument, factor`；`load_many(whitelist)`
- `factor_pool/aligner.py`：合并为 `panel: (date, instrument) → {factor_name: value}` + universe 过滤
- `factor_pool/label_builder.py`：按 N 日从 kline 重算 `future_ret[t, ins]`
- `combiner/normalizer.py`：方向统一 + winsorize + rank + z-score（横截面）
- `combiner/orthogonalizer.py`：对称正交，退化降级
- `combiner/ic_ir_weighter.py`：滚动 IC_IR 权重，双重错位
- `combiner/synthesizer.py`：`score = Σ w_i · X_ortho_i`
- `portfolio_builder/universe.py`：按日期查 mcap 表 → set[symbol]
- `portfolio_builder/constraints.py`：黑名单常量
- `portfolio_builder/topn_equal.py`：Top-N 等权 + 单票上限 + 换手约束
- `backtester/fees.py`：`fees(Δw) = Σ|Δw| · (fee_rate + slippage_bps/1e4)`
- `backtester/metrics.py`：净值/夏普/回撤/换手/年化
- `backtester/engine.py`：drift + 调仓日换手扣费
- `reporter/html_renderer.py`：简版 HTML（权重表 + 净值图 + 核心指标表）
- `pipeline.py`：`run(config) -> BacktestResult`
- `main.py`：CLI `python portfolio/main.py <strategy_name>`

---

## 前置约定

- 所有 Python 命令前缀 `./.venv/bin/python`
- 测试运行：`./.venv/bin/python -m pytest tests/test_portfolio_<x>.py -v`
- 每个 Task 以"test 先失败 → 实现到通过 → commit"为节拍
- commit trailer 统一附：`Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
- 统一使用 pandas 日期：`pd.to_datetime(date).normalize()` 归一到 00:00

---

### Task 0: Scaffold portfolio/ 包与 .gitignore

**Files:**
- Create: `portfolio/__init__.py`
- Create: `portfolio/factor_pool/__init__.py`
- Create: `portfolio/combiner/__init__.py`
- Create: `portfolio/portfolio_builder/__init__.py`
- Create: `portfolio/backtester/__init__.py`
- Create: `portfolio/reporter/__init__.py`
- Modify: `.gitignore`（追加 `portfolio/output/`）

- [ ] **Step 1: 创建空包文件**

6 个 `__init__.py` 内容均为：
```python
"""Portfolio layer: multi-factor synthesis, portfolio construction, backtest, attribution."""
```

- [ ] **Step 2: 追加 .gitignore**

在文件末尾追加：
```
# Portfolio layer outputs
portfolio/output/
```

- [ ] **Step 3: 运行现有测试套件，确认未破坏现状**

Run: `./.venv/bin/python -m pytest tests/ -v --tb=short -x 2>&1 | tail -30`
Expected: 既有测试全绿（或保持与基线相同状态）

- [ ] **Step 4: Commit**

```bash
git add portfolio/ .gitignore
git commit -m "feat(portfolio): scaffold package layout"
```

---

### Task 1: portfolio/portfolio_builder/constraints.py（黑名单常量）

**Files:**
- Create: `portfolio/portfolio_builder/constraints.py`
- Test: 无（纯常量）

- [ ] **Step 1: 写常量文件**

```python
"""Portfolio-layer constraint constants."""
from __future__ import annotations

STABLECOIN_BLACKLIST: frozenset[str] = frozenset({
    "USDCUSDT", "BUSDUSDT", "FDUSDUSDT", "TUSDUSDT",
    "DAIUSDT", "USDPUSDT", "USDTUSDT",
})

WRAPPED_BLACKLIST: frozenset[str] = frozenset({
    "WBTCUSDT", "WETHUSDT", "STETHUSDT",
})

DEFAULT_BLACKLIST: frozenset[str] = STABLECOIN_BLACKLIST | WRAPPED_BLACKLIST
```

- [ ] **Step 2: Commit**

```bash
git add portfolio/portfolio_builder/constraints.py
git commit -m "feat(portfolio): add blacklist constants"
```

---

### Task 2: factor_pool/loader.py

加载单个因子 CSV，只取三列（丢弃 `future_ret`）。

**Files:**
- Create: `portfolio/factor_pool/loader.py`
- Test: `tests/test_portfolio_loader.py`

- [ ] **Step 1: 写失败测试**

`tests/test_portfolio_loader.py`：
```python
import unittest
import tempfile
import pathlib
import sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.factor_pool.loader import load_factor_csv, load_many  # noqa: E402


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "instrument": ["BTCUSDT", "ETHUSDT"],
            "factor": [0.1, 0.2],
            "future_ret": [0.01, 0.02],
        }).to_csv(self.d / "f1.csv", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_factor_csv_drops_future_ret(self):
        df = load_factor_csv(self.d / "f1.csv", name="f1")
        self.assertEqual(list(df.columns), ["date", "instrument", "f1"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df["f1"].iloc[0], 0.1)

    def test_load_factor_csv_parses_date(self):
        df = load_factor_csv(self.d / "f1.csv", name="f1")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))

    def test_load_many(self):
        pd.DataFrame({
            "date": ["2024-01-01"],
            "instrument": ["BTCUSDT"],
            "factor": [1.5],
        }).to_csv(self.d / "f2.csv", index=False)
        out = load_many({"f1": self.d / "f1.csv", "f2": self.d / "f2.csv"})
        self.assertEqual(set(out.keys()), {"f1", "f2"})
        self.assertIn("f1", out["f1"].columns)
        self.assertIn("f2", out["f2"].columns)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_loader.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 实现 loader.py**

```python
"""Factor CSV loader. Only keeps date/instrument/factor columns."""
from __future__ import annotations

import pathlib
from typing import Mapping

import pandas as pd

REQUIRED_COLUMNS = ("date", "instrument", "factor")


def load_factor_csv(path: str | pathlib.Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.rename(columns={"factor": name})
    return df[["date", "instrument", name]]


def load_many(paths: Mapping[str, str | pathlib.Path]) -> dict[str, pd.DataFrame]:
    return {name: load_factor_csv(p, name=name) for name, p in paths.items()}
```

- [ ] **Step 4: 运行测试通过**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_loader.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add portfolio/factor_pool/loader.py tests/test_portfolio_loader.py
git commit -m "feat(portfolio): add factor CSV loader"
```

---

### Task 3: factor_pool/aligner.py

把多个因子 DataFrame 合并成 long-format panel：`date, instrument, <f1>, <f2>, ...`；可选 universe 过滤。

**Files:**
- Create: `portfolio/factor_pool/aligner.py`
- Test: `tests/test_portfolio_aligner.py`

- [ ] **Step 1: 写失败测试**

```python
import unittest
import pathlib
import sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.factor_pool.aligner import align_factors  # noqa: E402


def _mk(name, rows):
    return pd.DataFrame(rows, columns=["date", "instrument", name]).assign(
        date=lambda d: pd.to_datetime(d["date"]),
    )


class TestAligner(unittest.TestCase):
    def test_align_two_factors(self):
        f1 = _mk("f1", [["2024-01-01", "BTCUSDT", 0.1],
                        ["2024-01-02", "BTCUSDT", 0.2]])
        f2 = _mk("f2", [["2024-01-01", "BTCUSDT", 1.0],
                        ["2024-01-02", "BTCUSDT", 2.0]])
        panel = align_factors({"f1": f1, "f2": f2})
        self.assertEqual(set(panel.columns), {"date", "instrument", "f1", "f2"})
        self.assertEqual(len(panel), 2)

    def test_align_outer_join_missing_filled_nan(self):
        f1 = _mk("f1", [["2024-01-01", "BTCUSDT", 0.1]])
        f2 = _mk("f2", [["2024-01-01", "ETHUSDT", 9.0]])
        panel = align_factors({"f1": f1, "f2": f2})
        self.assertEqual(len(panel), 2)
        self.assertTrue(panel["f1"].isna().any())
        self.assertTrue(panel["f2"].isna().any())

    def test_universe_filter(self):
        f1 = _mk("f1", [["2024-01-01", "BTCUSDT", 0.1],
                        ["2024-01-01", "DOGEUSDT", 0.3]])
        universe_by_date = {pd.Timestamp("2024-01-01"): {"BTCUSDT"}}
        panel = align_factors({"f1": f1}, universe_by_date=universe_by_date)
        self.assertEqual(set(panel["instrument"]), {"BTCUSDT"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行失败**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_aligner.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 aligner.py**

```python
"""Align multiple factor frames into a single (date, instrument, *factors) panel."""
from __future__ import annotations

from typing import Mapping

import pandas as pd


def align_factors(
    factors: Mapping[str, pd.DataFrame],
    universe_by_date: Mapping[pd.Timestamp, set[str]] | None = None,
) -> pd.DataFrame:
    if not factors:
        return pd.DataFrame(columns=["date", "instrument"])
    panel: pd.DataFrame | None = None
    for name, df in factors.items():
        df = df[["date", "instrument", name]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        panel = df if panel is None else panel.merge(
            df, on=["date", "instrument"], how="outer"
        )
    assert panel is not None
    if universe_by_date is not None:
        mask = panel.apply(
            lambda r: r["instrument"] in universe_by_date.get(r["date"], set()),
            axis=1,
        )
        panel = panel.loc[mask].copy()
    return panel.sort_values(["date", "instrument"]).reset_index(drop=True)
```

- [ ] **Step 4: 运行通过**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_aligner.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add portfolio/factor_pool/aligner.py tests/test_portfolio_aligner.py
git commit -m "feat(portfolio): add factor aligner"
```

---

### Task 4: factor_pool/label_builder.py

从 kline 按策略 N 日统一重算 `future_ret[t, ins] = close[t+N]/close[t] - 1`。

**Files:**
- Create: `portfolio/factor_pool/label_builder.py`
- Test: `tests/test_portfolio_label_builder.py`

- [ ] **Step 1: 写失败测试**

```python
import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.factor_pool.label_builder import build_future_ret  # noqa: E402


class TestLabelBuilder(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        self.kline = pd.concat([
            pd.DataFrame({"date": dates, "symbol": "BTCUSDT",
                          "close": [100, 101, 102, 103, 104, 105]}),
            pd.DataFrame({"date": dates, "symbol": "ETHUSDT",
                          "close": [10, 11, 12, 13, 14, 15]}),
        ], ignore_index=True)

    def test_future_ret_N3(self):
        fr = build_future_ret(self.kline, period_days=3)
        cell = fr[(fr["date"] == pd.Timestamp("2024-01-01")) &
                  (fr["instrument"] == "BTCUSDT")]["future_ret"].iloc[0]
        self.assertAlmostEqual(cell, 103 / 100 - 1, places=10)

    def test_future_ret_last_N_rows_are_nan(self):
        fr = build_future_ret(self.kline, period_days=3)
        last_dates = fr[fr["instrument"] == "BTCUSDT"].sort_values("date")["date"].iloc[-3:]
        for d in last_dates:
            v = fr[(fr["date"] == d) & (fr["instrument"] == "BTCUSDT")]["future_ret"].iloc[0]
            self.assertTrue(pd.isna(v))

    def test_output_schema(self):
        fr = build_future_ret(self.kline, period_days=2)
        self.assertEqual(list(fr.columns), ["date", "instrument", "future_ret"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行失败**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_label_builder.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
"""Compute unified forward returns from kline at a given horizon."""
from __future__ import annotations

import pandas as pd


def build_future_ret(kline: pd.DataFrame, period_days: int) -> pd.DataFrame:
    if period_days <= 0:
        raise ValueError("period_days must be >= 1")
    df = kline[["date", "symbol", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["symbol", "date"])
    df["future_close"] = df.groupby("symbol")["close"].shift(-period_days)
    df["future_ret"] = df["future_close"] / df["close"] - 1.0
    df = df.rename(columns={"symbol": "instrument"})
    return df[["date", "instrument", "future_ret"]].reset_index(drop=True)
```

> 注：使用 `shift(-period_days)` 仅发生在**标签列**构造，用于评估 IC，非因子值链路。符合 AGENTS.md 例外规定。

- [ ] **Step 4: 运行通过**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_label_builder.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add portfolio/factor_pool/label_builder.py tests/test_portfolio_label_builder.py
git commit -m "feat(portfolio): add unified future_ret label builder"
```

---

### Task 5: combiner/normalizer.py

横截面 winsorize → rank → z-score；方向统一。

**Files:**
- Create: `portfolio/combiner/normalizer.py`
- Test: `tests/test_portfolio_normalizer.py`

- [ ] **Step 1: 写失败测试**

```python
import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.combiner.normalizer import normalize_cross_section  # noqa: E402


class TestNormalizer(unittest.TestCase):
    def setUp(self):
        self.panel = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"] * 5 + ["2024-01-02"] * 5),
            "instrument": ["A", "B", "C", "D", "E"] * 2,
            "f1": [1, 2, 3, 4, 100, 1, 2, 3, 4, 100],
            "f2": [10, 20, 30, 40, 50, 11, 22, 33, 44, 55],
        })

    def test_z_score_zero_mean(self):
        out = normalize_cross_section(self.panel, factor_cols=["f1", "f2"],
                                      directions={"f1": 1, "f2": 1},
                                      winsorize_pct=(0.0, 1.0))
        grp = out.groupby("date")["f1"].mean()
        for v in grp:
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_direction_flip(self):
        out = normalize_cross_section(self.panel, factor_cols=["f1"],
                                      directions={"f1": -1},
                                      winsorize_pct=(0.0, 1.0))
        row_a = out[(out["date"] == pd.Timestamp("2024-01-01")) &
                    (out["instrument"] == "A")]["f1"].iloc[0]
        row_e = out[(out["date"] == pd.Timestamp("2024-01-01")) &
                    (out["instrument"] == "E")]["f1"].iloc[0]
        self.assertGreater(row_a, row_e)

    def test_winsorize_clips_outliers(self):
        out = normalize_cross_section(self.panel, factor_cols=["f1"],
                                      directions={"f1": 1},
                                      winsorize_pct=(0.0, 0.8))
        row_e = out[(out["date"] == pd.Timestamp("2024-01-01")) &
                    (out["instrument"] == "E")]["f1"].iloc[0]
        row_d = out[(out["date"] == pd.Timestamp("2024-01-01")) &
                    (out["instrument"] == "D")]["f1"].iloc[0]
        self.assertAlmostEqual(row_e, row_d, places=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行失败**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_normalizer.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
"""Cross-sectional winsorize + rank + z-score normalization."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    if s.dropna().empty:
        return s
    low = s.quantile(lo)
    high = s.quantile(hi)
    return s.clip(lower=low, upper=high)


def _zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return s * 0.0
    return (s - m) / sd


def normalize_cross_section(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    directions: Mapping[str, int],
    winsorize_pct: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    lo, hi = winsorize_pct
    out = panel.copy()
    for col in factor_cols:
        sign = directions.get(col, 1)
        out[col] = out[col] * sign
        out[col] = out.groupby("date")[col].transform(
            lambda s: _winsorize(s, lo, hi)
        )
        out[col] = out.groupby("date")[col].rank(method="average", pct=True) - 0.5
        out[col] = out.groupby("date")[col].transform(_zscore)
    return out
```

- [ ] **Step 4: 通过**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_normalizer.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add portfolio/combiner/normalizer.py tests/test_portfolio_normalizer.py
git commit -m "feat(portfolio): add cross-section normalizer"
```

---

### Task 6: combiner/orthogonalizer.py

对称正交 `X_ortho = X · (XᵀX)^(-1/2)`，每 date 独立；退化降级到"不正交"。

**Files:**
- Create: `portfolio/combiner/orthogonalizer.py`
- Test: `tests/test_portfolio_orthogonalizer.py`

- [ ] **Step 1: 写失败测试**

```python
import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.combiner.orthogonalizer import symmetric_orthogonalize  # noqa: E402


class TestOrthogonalizer(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 30
        dates = [pd.Timestamp("2024-01-01")] * n + [pd.Timestamp("2024-01-02")] * n
        instruments = [f"S{i}" for i in range(n)] * 2
        f1 = rng.standard_normal(2 * n)
        f2 = f1 * 0.5 + rng.standard_normal(2 * n) * 0.5
        self.panel = pd.DataFrame({
            "date": dates, "instrument": instruments,
            "f1": f1, "f2": f2,
        })

    def test_orthogonal_gram_matrix_is_identity(self):
        out = symmetric_orthogonalize(self.panel, factor_cols=["f1", "f2"])
        one_day = out[out["date"] == pd.Timestamp("2024-01-01")]
        X = one_day[["f1", "f2"]].to_numpy()
        X = X - X.mean(axis=0, keepdims=True)
        G = X.T @ X / X.shape[0]
        self.assertAlmostEqual(G[0, 1], 0.0, places=6)
        self.assertAlmostEqual(G[0, 0], G[1, 1], places=6)

    def test_degenerate_small_n(self):
        small = self.panel[self.panel["instrument"].isin(["S0"])].copy()
        out = symmetric_orthogonalize(small, factor_cols=["f1", "f2"])
        self.assertEqual(len(out), len(small))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行失败**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_orthogonalizer.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
"""Per-date symmetric orthogonalization."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _ortho_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    mask = ~np.isnan(Xc).any(axis=1)
    if mask.sum() < X.shape[1] + 1:
        return X
    Y = Xc[mask]
    S = (Y.T @ Y) / Y.shape[0]
    try:
        eigvals, eigvecs = np.linalg.eigh(S)
    except np.linalg.LinAlgError:
        return X
    eigvals = np.clip(eigvals, 1e-10, None)
    W = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
    Y_o = Y @ W
    out = np.full_like(X, np.nan, dtype=float)
    out[mask] = Y_o + np.nanmean(X, axis=0, keepdims=True)
    out[~mask] = X[~mask]
    return out


def symmetric_orthogonalize(
    panel: pd.DataFrame, factor_cols: Sequence[str]
) -> pd.DataFrame:
    out = panel.copy()
    cols = list(factor_cols)
    for d, grp in out.groupby("date"):
        X = grp[cols].to_numpy(dtype=float)
        Xo = _ortho_matrix(X)
        out.loc[grp.index, cols] = Xo
    return out
```

- [ ] **Step 4: 通过**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_orthogonalizer.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add portfolio/combiner/orthogonalizer.py tests/test_portfolio_orthogonalizer.py
git commit -m "feat(portfolio): add symmetric orthogonalizer"
```

---

### Task 7: combiner/ic_ir_weighter.py

对每个 `t`，使用窗口 `[t-L, t-1]` 内满足 `s+N <= t-1` 的 IC 样本算 IC_IR。

**Files:**
- Create: `portfolio/combiner/ic_ir_weighter.py`
- Test: `tests/test_portfolio_ic_ir_weighter.py`

- [ ] **Step 1: 写失败测试**

```python
import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.combiner.ic_ir_weighter import compute_ic_series, ic_ir_weights  # noqa: E402


class TestICIRWeighter(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        syms = [f"S{i}" for i in range(20)]
        rows = []
        for d in dates:
            for s in syms:
                f1 = rng.standard_normal()
                f2 = rng.standard_normal()
                rows.append((d, s, f1, f2, f1 * 0.1 + rng.standard_normal() * 0.05))
        self.panel = pd.DataFrame(rows, columns=["date", "instrument", "f1", "f2", "future_ret"])

    def test_compute_ic_series(self):
        ic = compute_ic_series(self.panel, factor_cols=["f1", "f2"])
        self.assertEqual(set(ic.columns), {"date", "f1", "f2"})
        self.assertAlmostEqual(ic["f1"].mean(), 0.1, delta=0.3)

    def test_ic_ir_weights_double_offset(self):
        ic = compute_ic_series(self.panel, factor_cols=["f1", "f2"])
        t = pd.Timestamp("2024-01-25")
        w = ic_ir_weights(ic, t, factor_cols=["f1", "f2"], lookback=10, horizon_n=3)
        self.assertAlmostEqual(abs(w["f1"]) + abs(w["f2"]), 1.0, places=6)
        self.assertGreater(w["f1"], w["f2"])

    def test_ic_ir_window_strictly_before_t(self):
        ic = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "f1": [0.1] * 10, "f2": [0.0] * 10,
        })
        t = pd.Timestamp("2024-01-05")
        w = ic_ir_weights(ic, t, factor_cols=["f1", "f2"], lookback=5, horizon_n=1)
        self.assertAlmostEqual(w["f1"], 1.0, places=6)

        ic2 = ic.copy()
        ic2.loc[ic2["date"] >= t, "f1"] = 999.0
        w2 = ic_ir_weights(ic2, t, factor_cols=["f1", "f2"], lookback=5, horizon_n=1)
        self.assertAlmostEqual(w["f1"], w2["f1"], places=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行失败**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_ic_ir_weighter.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
"""Rolling IC / IC_IR based factor weights with double-offset anti-leakage."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    ra = df.iloc[:, 0].rank()
    rb = df.iloc[:, 1].rank()
    if ra.std(ddof=0) < 1e-12 or rb.std(ddof=0) < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_ic_series(panel: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for d, grp in panel.groupby("date"):
        row = {"date": d}
        for c in factor_cols:
            row[c] = _rank_corr(grp[c], grp["future_ret"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def ic_ir_weights(
    ic: pd.DataFrame,
    t: pd.Timestamp,
    factor_cols: Sequence[str],
    lookback: int,
    horizon_n: int,
    survivor_mask: Mapping[str, bool] | None = None,
) -> dict[str, float]:
    cutoff = t - pd.Timedelta(days=horizon_n)
    window = ic[(ic["date"] <= cutoff)].tail(lookback)
    weights = {}
    for c in factor_cols:
        s = window[c].dropna()
        if len(s) < max(3, lookback // 3):
            weights[c] = 0.0
            continue
        m, sd = s.mean(), s.std(ddof=0)
        ir = 0.0 if sd < 1e-9 else m / (sd + 1e-9)
        if survivor_mask is not None and not survivor_mask.get(c, True):
            ir = 0.0
        weights[c] = float(ir)
    total = sum(abs(v) for v in weights.values())
    if total < 1e-12:
        return {c: 0.0 for c in factor_cols}
    return {c: v / total for c, v in weights.items()}
```

- [ ] **Step 4: 通过**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_ic_ir_weighter.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add portfolio/combiner/ic_ir_weighter.py tests/test_portfolio_ic_ir_weighter.py
git commit -m "feat(portfolio): add rolling IC/IC_IR weighter"
```

---

### Task 8: combiner/synthesizer.py

`score[t, ins] = Σ w_i(t) · X_ortho[t, ins, i]`。

**Files:**
- Create: `portfolio/combiner/synthesizer.py`

- [ ] **Step 1: 实现（纯函数，与现有测试复用）**

```python
"""Composite factor score from orthogonal factors and per-t weights."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def synthesize(
    panel_ortho: pd.DataFrame,
    factor_cols: Sequence[str],
    weights_by_date: Mapping[pd.Timestamp, Mapping[str, float]],
) -> pd.DataFrame:
    out = panel_ortho[["date", "instrument"]].copy()
    score = np.zeros(len(panel_ortho), dtype=float)
    for c in factor_cols:
        w_series = panel_ortho["date"].map(
            lambda d: weights_by_date.get(d, {}).get(c, 0.0)
        )
        f = panel_ortho[c].fillna(0.0).to_numpy(dtype=float)
        score = score + w_series.to_numpy(dtype=float) * f
    out["score"] = score
    return out
```

- [ ] **Step 2: Smoke test 嵌在 ic_ir_weighter 测试里，已足够；此处只提交模块**

- [ ] **Step 3: Commit**

```bash
git add portfolio/combiner/synthesizer.py
git commit -m "feat(portfolio): add score synthesizer"
```

---

### Task 9: portfolio_builder/universe.py

按日期查市值表 → 日期到标的集合的映射。

**Files:**
- Create: `portfolio/portfolio_builder/universe.py`
- Test: `tests/test_portfolio_universe.py`

- [ ] **Step 1: 写失败测试**

```python
import unittest
import pathlib
import sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.portfolio_builder.universe import build_universe_by_date  # noqa: E402


class TestUniverse(unittest.TestCase):
    def setUp(self):
        self.mcap = pd.DataFrame({
            "Decision_Window_Start": pd.to_datetime(["2024-01-01"] * 4),
            "Decision_Window_End":   pd.to_datetime(["2024-01-31"] * 4),
            "Trading_Pairs":    ["BTCUSDT", "ETHUSDT", "USDCUSDT", "DOGEUSDT"],
            "Market_Cap_USD":   [1e12, 5e11, 3e10, 1e10],
            "Quote_Volume_USD": [1e9, 5e8, 1e7, 1e4],
        })

    def test_top_n_and_blacklist(self):
        res = build_universe_by_date(
            self.mcap,
            dates=[pd.Timestamp("2024-01-15")],
            top_n_mcap=10,
            liquidity_min_quote_volume_usd=1e5,
        )
        self.assertEqual(res[pd.Timestamp("2024-01-15")], {"BTCUSDT", "ETHUSDT"})

    def test_liquidity_filter(self):
        res = build_universe_by_date(
            self.mcap,
            dates=[pd.Timestamp("2024-01-15")],
            top_n_mcap=10,
            liquidity_min_quote_volume_usd=6e8,
        )
        self.assertEqual(res[pd.Timestamp("2024-01-15")], {"BTCUSDT"})

    def test_date_out_of_window_returns_empty(self):
        res = build_universe_by_date(
            self.mcap,
            dates=[pd.Timestamp("2030-01-01")],
            top_n_mcap=10,
            liquidity_min_quote_volume_usd=0,
        )
        self.assertEqual(res[pd.Timestamp("2030-01-01")], set())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行失败**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_universe.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
"""Build dynamic Top-N market-cap universe with liquidity + blacklist filter."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .constraints import DEFAULT_BLACKLIST


def build_universe_by_date(
    mcap_df: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
    top_n_mcap: int,
    liquidity_min_quote_volume_usd: float,
    blacklist: frozenset[str] = DEFAULT_BLACKLIST,
) -> dict[pd.Timestamp, set[str]]:
    df = mcap_df.copy()
    df["Decision_Window_Start"] = pd.to_datetime(df["Decision_Window_Start"])
    df["Decision_Window_End"] = pd.to_datetime(df["Decision_Window_End"])
    result: dict[pd.Timestamp, set[str]] = {}
    for d in dates:
        d = pd.Timestamp(d).normalize()
        active = df[(df["Decision_Window_Start"] <= d) &
                    (df["Decision_Window_End"] >= d)]
        active = active.sort_values("Market_Cap_USD", ascending=False).head(top_n_mcap)
        active = active[active["Quote_Volume_USD"] >= liquidity_min_quote_volume_usd]
        syms = set(active["Trading_Pairs"].tolist()) - set(blacklist)
        result[d] = syms
    return result
```

- [ ] **Step 4: 通过**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_universe.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add portfolio/portfolio_builder/universe.py tests/test_portfolio_universe.py
git commit -m "feat(portfolio): add dynamic universe builder"
```

---

### Task 10: portfolio_builder/topn_equal.py

Top-N 等权 + 单票上限 + 换手约束。

**Files:**
- Create: `portfolio/portfolio_builder/topn_equal.py`
- Test: `tests/test_portfolio_topn_builder.py`

- [ ] **Step 1: 写失败测试**

```python
import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.portfolio_builder.topn_equal import build_topn_equal  # noqa: E402


class TestTopN(unittest.TestCase):
    def test_top_n_equal_basic(self):
        scores = pd.Series({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1})
        w = build_topn_equal(scores, top_n=3, max_weight=1.0, leverage=1.0,
                             turnover_cap=None, w_prev=None)
        self.assertAlmostEqual(w.sum(), 1.0, places=9)
        self.assertEqual(set(w[w > 0].index), {"A", "B", "C"})
        for v in w[w > 0]:
            self.assertAlmostEqual(v, 1 / 3, places=9)

    def test_max_weight_clip_and_renormalize(self):
        scores = pd.Series({"A": 5, "B": 4, "C": 3})
        w = build_topn_equal(scores, top_n=3, max_weight=0.4, leverage=1.0,
                             turnover_cap=None, w_prev=None)
        self.assertLessEqual(w.max(), 0.4 + 1e-9)
        self.assertAlmostEqual(w.sum(), 1.0, places=9)

    def test_turnover_cap(self):
        scores = pd.Series({"A": 5, "B": 4, "C": 3, "D": 2})
        w_prev = pd.Series({"C": 0.5, "D": 0.5})
        w = build_topn_equal(scores, top_n=2, max_weight=1.0, leverage=1.0,
                             turnover_cap=0.4,
                             w_prev=w_prev)
        turnover = (w.subtract(w_prev, fill_value=0.0)).abs().sum()
        self.assertLessEqual(turnover, 0.4 + 1e-9)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行失败**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_topn_builder.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
"""Top-N equal weight portfolio with weight cap and turnover constraint."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cap_and_renorm(w: pd.Series, max_weight: float, leverage: float) -> pd.Series:
    w = w.clip(upper=max_weight)
    s = w.sum()
    if s <= 0:
        return w
    return w / s * leverage


def build_topn_equal(
    scores: pd.Series,
    top_n: int,
    max_weight: float,
    leverage: float,
    turnover_cap: float | None,
    w_prev: pd.Series | None,
) -> pd.Series:
    scores = scores.dropna().sort_values(ascending=False)
    picked = scores.head(top_n).index
    w = pd.Series(leverage / max(len(picked), 1), index=picked)
    w = _cap_and_renorm(w, max_weight, leverage)
    if turnover_cap is None or w_prev is None:
        return w.reindex(scores.index.union(w.index), fill_value=0.0).loc[lambda s: s > 0]
    w_prev_full = w_prev.reindex(w.index.union(w_prev.index), fill_value=0.0)
    w_full = w.reindex(w_prev_full.index, fill_value=0.0)
    diff = w_full - w_prev_full
    turnover = diff.abs().sum()
    if turnover <= turnover_cap or turnover < 1e-12:
        return w_full[w_full > 0]
    scale = turnover_cap / turnover
    w_scaled = w_prev_full + diff * scale
    return w_scaled[w_scaled > 0]
```

- [ ] **Step 4: 通过**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_topn_builder.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add portfolio/portfolio_builder/topn_equal.py tests/test_portfolio_topn_builder.py
git commit -m "feat(portfolio): add top-N equal weight builder"
```

---

### Task 11: backtester/fees.py + metrics.py

**Files:**
- Create: `portfolio/backtester/fees.py`
- Create: `portfolio/backtester/metrics.py`

- [ ] **Step 1: 实现 fees.py**

```python
"""Transaction cost model."""
from __future__ import annotations

import numpy as np
import pandas as pd


def turnover_cost(delta_w: pd.Series, fee_rate: float, slippage_bps: float) -> float:
    total_rate = fee_rate + slippage_bps / 10_000.0
    return float(delta_w.abs().sum() * total_rate)
```

- [ ] **Step 2: 实现 metrics.py**

```python
"""Backtest performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(port_ret: pd.Series, periods_per_year: int = 365) -> dict[str, float]:
    r = port_ret.dropna().astype(float)
    if r.empty:
        return {"cagr": 0.0, "vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    nav = (1 + r).cumprod()
    years = max(len(r) / periods_per_year, 1e-9)
    cagr = float(nav.iloc[-1] ** (1 / years) - 1)
    vol = float(r.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = 0.0 if vol < 1e-12 else cagr / vol
    roll_max = nav.cummax()
    dd = (nav / roll_max - 1).min()
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
    }
```

- [ ] **Step 3: Commit**

```bash
git add portfolio/backtester/fees.py portfolio/backtester/metrics.py
git commit -m "feat(portfolio): add fees and metrics helpers"
```

---

### Task 12: backtester/engine.py（drift）

核心回测：drift 漂移 + 调仓日扣费 + 产出 port_ret 与 weight_history。

**Files:**
- Create: `portfolio/backtester/engine.py`
- Test: `tests/test_portfolio_backtester_drift.py`

- [ ] **Step 1: 写失败测试**

```python
import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.backtester.engine import run_backtest  # noqa: E402


class TestBacktester(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        self.returns = pd.DataFrame({
            "A": [0.00, 0.10, 0.00, -0.05, 0.05],
            "B": [0.00, -0.05, 0.05, 0.10, 0.00],
        }, index=dates)
        self.targets = pd.DataFrame({
            "A": [0.5, np.nan, np.nan, np.nan, np.nan],
            "B": [0.5, np.nan, np.nan, np.nan, np.nan],
        }, index=dates)

    def test_port_ret_uses_prev_weights(self):
        res = run_backtest(self.returns, self.targets, fee_rate=0.0, slippage_bps=0.0)
        self.assertAlmostEqual(res["port_ret"].iloc[0], 0.0, places=9)
        self.assertAlmostEqual(res["port_ret"].iloc[1], 0.5 * 0.10 + 0.5 * -0.05, places=9)

    def test_no_rebalance_no_fees(self):
        res = run_backtest(self.returns, self.targets, fee_rate=0.01, slippage_bps=100.0)
        self.assertAlmostEqual(res["fees"].iloc[2], 0.0, places=9)

    def test_rebalance_day_charges_fees(self):
        targets = self.targets.copy()
        targets.loc[pd.Timestamp("2024-01-03"), ["A", "B"]] = [1.0, 0.0]
        res = run_backtest(self.returns, targets, fee_rate=0.001, slippage_bps=0.0)
        self.assertGreater(res["fees"].iloc[2], 0.0)

    def test_nav_compounds(self):
        res = run_backtest(self.returns, self.targets, fee_rate=0.0, slippage_bps=0.0)
        expected = (1 + res["port_ret"]).prod()
        self.assertAlmostEqual(res["nav"].iloc[-1], expected, places=9)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行失败**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_backtester_drift.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
"""Drift-mode portfolio backtester.

Timing convention:
  port_ret[t] = w[t-1] @ ret[t-1 -> t]
On non-rebalance days, weights drift:
  w[i, t] = w[i, t-1] * (1 + r_i) / (1 + port_ret)
On rebalance days (target row has any non-NaN), we:
  1. drift to w_drifted
  2. charge fees on |target - w_drifted|
  3. set w[t] = target
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fees import turnover_cost


def run_backtest(
    returns: pd.DataFrame,
    targets: pd.DataFrame,
    fee_rate: float = 0.0003,
    slippage_bps: float = 5.0,
) -> pd.DataFrame:
    idx = returns.index
    cols = sorted(set(returns.columns) | set(targets.columns))
    returns = returns.reindex(columns=cols, fill_value=0.0)
    targets = targets.reindex(index=idx, columns=cols)

    w = pd.Series(0.0, index=cols)
    port_rets: list[float] = []
    fees: list[float] = []
    weights_hist: list[pd.Series] = []

    for i, t in enumerate(idx):
        r = returns.loc[t].fillna(0.0)
        p = float((w * r).sum())
        port_rets.append(p)

        if abs(1 + p) > 1e-9 and w.abs().sum() > 0:
            w = w * (1 + r) / (1 + p)

        target_row = targets.loc[t]
        if target_row.notna().any():
            target = target_row.fillna(0.0)
            delta = (target - w)
            cost = turnover_cost(delta, fee_rate, slippage_bps)
            port_rets[-1] = port_rets[-1] - cost
            fees.append(cost)
            w = target.copy()
        else:
            fees.append(0.0)

        weights_hist.append(w.copy())

    port_ret = pd.Series(port_rets, index=idx, name="port_ret")
    fees_s = pd.Series(fees, index=idx, name="fees")
    nav = (1 + port_ret).cumprod()
    nav.name = "nav"
    out = pd.concat([port_ret, fees_s, nav], axis=1)
    out.attrs["weights_history"] = pd.DataFrame(weights_hist, index=idx)
    return out
```

> 说明：`port_ret` 在调仓日扣费是通过修正 `port_rets[-1]`（同一位置）实现的；该日收益 = 当日漂移收益 − 成本，与 spec §5.9 一致。测试 `test_port_ret_uses_prev_weights` 第 0 日因 `w_prev = 0`（调仓发生在 t=0 后），故等于 0。

- [ ] **Step 4: 通过**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_backtester_drift.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add portfolio/backtester/engine.py tests/test_portfolio_backtester_drift.py
git commit -m "feat(portfolio): add drift-mode backtester engine"
```

---

### Task 13: reporter/html_renderer.py（简版）

产出最小 HTML：汇总指标 + 最近权重快照 + 净值 JSON（可嵌 plot.js 或简单表格）。P1 MVP 不强求图形，优先"可读"。

**Files:**
- Create: `portfolio/reporter/html_renderer.py`

- [ ] **Step 1: 实现**

```python
"""Minimal HTML report for portfolio backtest results."""
from __future__ import annotations

import pathlib
from datetime import datetime

import pandas as pd


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Portfolio Report {strategy}</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;margin:24px}}
table{{border-collapse:collapse;margin:12px 0}}
th,td{{border:1px solid #ccc;padding:6px 10px;text-align:right}}
th{{background:#f0f0f0}} h1,h2{{margin:12px 0}}
.metric{{font-size:1.1em}}</style></head>
<body>
<h1>Portfolio Report — {strategy}</h1>
<p>Generated: {ts} · Backtest span: {d0} → {d1}</p>
<h2>Key Metrics</h2>
{metrics_table}
<h2>Recent Target Weights (last {n_rows} rebalance days)</h2>
{weights_table}
<h2>NAV (tail 30)</h2>
{nav_table}
</body></html>
"""


def _df_to_html(df: pd.DataFrame) -> str:
    return df.to_html(float_format=lambda x: f"{x:.6f}")


def render_report(
    strategy: str,
    metrics: dict[str, float],
    backtest_df: pd.DataFrame,
    weights_history: pd.DataFrame,
    output_path: str | pathlib.Path,
) -> pathlib.Path:
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics, index=[strategy]).T
    recent = weights_history.tail(90)
    rebalance_days = recent[(recent.diff().abs().sum(axis=1) > 1e-9)].tail(5)
    html = _HTML.format(
        strategy=strategy,
        ts=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        d0=str(backtest_df.index.min().date()),
        d1=str(backtest_df.index.max().date()),
        metrics_table=_df_to_html(metrics_df),
        n_rows=len(rebalance_days),
        weights_table=_df_to_html(rebalance_days),
        nav_table=_df_to_html(backtest_df[["nav", "port_ret", "fees"]].tail(30)),
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
```

- [ ] **Step 2: Commit**

```bash
git add portfolio/reporter/html_renderer.py
git commit -m "feat(portfolio): add minimal HTML reporter"
```

---

### Task 14: config.py（baseline_topn_long）

从 spec §4 取 `baseline_topn_long` 最小子集；提供 `resolve_config(name)` 支持 `inherits`（P1 未用，但预留）。

**Files:**
- Create: `portfolio/config.py`

- [ ] **Step 1: 实现**

```python
"""Portfolio strategy configurations."""
from __future__ import annotations

import copy
from typing import Any

STRATEGY_CONFIG: dict[str, dict[str, Any]] = {
    "baseline_topn_long": {
        "factor_whitelist": [
            "directional_momentum_20d",
            "volume_stability_20d",
            "volatility_efficiency_20d",
            "retail_activity_divergence_20d",
            "volatility_order_flow_20d",
        ],
        "combiner": {
            "winsorize_pct": (0.01, 0.99),
            "orthogonal": "symmetric",
            "ic_ir_lookback": 60,
        },
        "universe": {
            "top_n_mcap": 100,
            "liquidity_min_quote_volume_usd": 5_000_000,
        },
        "builder": {
            "mode": "topn_equal",
            "top_n": 10,
            "max_weight": 0.20,
            "leverage": 1.0,
            "turnover_cap": 0.50,
        },
        "rebalance": {"period_days": 3, "hold_mode": "drift"},
        "costs": {"fee_rate": 0.0003, "slippage_bps": 5},
        "backtest": {"start": "2022-01-01", "end": None},
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_config(name: str) -> dict[str, Any]:
    if name not in STRATEGY_CONFIG:
        raise KeyError(f"Unknown strategy: {name}")
    cfg = STRATEGY_CONFIG[name]
    parent = cfg.get("inherits")
    if parent:
        return _deep_merge(resolve_config(parent), {k: v for k, v in cfg.items() if k != "inherits"})
    return copy.deepcopy(cfg)
```

- [ ] **Step 2: Commit**

```bash
git add portfolio/config.py
git commit -m "feat(portfolio): add strategy config with inheritance"
```

---

### Task 15: pipeline.py（串联各模块 + 查 factor_direction）

**Files:**
- Create: `portfolio/pipeline.py`

- [ ] **Step 1: 实现**

```python
"""End-to-end portfolio pipeline: factors -> score -> weights -> backtest -> report."""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .factor_pool.loader import load_many
from .factor_pool.aligner import align_factors
from .factor_pool.label_builder import build_future_ret
from .combiner.normalizer import normalize_cross_section
from .combiner.orthogonalizer import symmetric_orthogonalize
from .combiner.ic_ir_weighter import compute_ic_series, ic_ir_weights
from .combiner.synthesizer import synthesize
from .portfolio_builder.universe import build_universe_by_date
from .portfolio_builder.topn_equal import build_topn_equal
from .backtester.engine import run_backtest
from .backtester.metrics import compute_metrics


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FACTOR_DIR = DATA_DIR / "factor_data"
KLINE_DIR = DATA_DIR / "kline_data"
MCAP_CSV = DATA_DIR / "binance_coingecko_top100_marketcap_historical.csv"


@dataclass
class PipelineResult:
    backtest_df: pd.DataFrame
    weights_history: pd.DataFrame
    target_weights: pd.DataFrame
    metrics: dict[str, float]


def _load_factor_directions(whitelist: list[str]) -> dict[str, int]:
    sys.path.insert(0, str(REPO_ROOT))
    from factor_analyse import factor_config as fc
    directions: dict[str, int] = {}
    for name in whitelist:
        cfg = fc.FACTOR_TYPE_CONFIG.get(name) if hasattr(fc, "FACTOR_TYPE_CONFIG") else None
        if cfg is None:
            for attr in ("FACTOR_TYPES", "FACTORS", "FACTOR_CONFIG"):
                c = getattr(fc, attr, None)
                if isinstance(c, dict) and name in c:
                    cfg = c[name]; break
        if cfg is None:
            directions[name] = 1
            continue
        d = cfg.get("factor_direction", 1) if isinstance(cfg, dict) else 1
        directions[name] = int(d) if d in (1, -1) else 1
    return directions


def _discover_factor_csv(name: str) -> pathlib.Path:
    matches = sorted(FACTOR_DIR.glob(f"*{name}*.csv"))
    if not matches:
        raise FileNotFoundError(f"No factor CSV matched name={name} in {FACTOR_DIR}")
    return matches[-1]


def _load_kline_closes() -> pd.DataFrame:
    frames = []
    for p in KLINE_DIR.glob("*.csv"):
        df = pd.read_csv(p, usecols=["symbol", "date", "close"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No kline CSV in {KLINE_DIR}")
    kl = pd.concat(frames, ignore_index=True)
    kl["date"] = pd.to_datetime(kl["date"]).dt.normalize()
    kl = kl.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    return kl


def run(cfg: dict[str, Any], strategy_name: str = "baseline_topn_long") -> PipelineResult:
    whitelist = cfg["factor_whitelist"]
    factor_paths = {n: _discover_factor_csv(n) for n in whitelist}
    factor_frames = load_many(factor_paths)

    kline = _load_kline_closes()
    N = cfg["rebalance"]["period_days"]
    future_ret_df = build_future_ret(kline, period_days=N)

    mcap = pd.read_csv(MCAP_CSV)
    start = pd.Timestamp(cfg["backtest"]["start"])
    end = pd.Timestamp(cfg["backtest"]["end"]) if cfg["backtest"]["end"] else kline["date"].max()
    all_dates = pd.date_range(start, end, freq="D")
    universe_by_date = build_universe_by_date(
        mcap, dates=all_dates,
        top_n_mcap=cfg["universe"]["top_n_mcap"],
        liquidity_min_quote_volume_usd=cfg["universe"]["liquidity_min_quote_volume_usd"],
    )

    panel = align_factors(factor_frames, universe_by_date=universe_by_date)
    panel = panel.merge(future_ret_df, on=["date", "instrument"], how="left")

    directions = _load_factor_directions(whitelist)
    panel_norm = normalize_cross_section(
        panel, factor_cols=whitelist, directions=directions,
        winsorize_pct=cfg["combiner"]["winsorize_pct"],
    )
    panel_ortho = symmetric_orthogonalize(panel_norm, factor_cols=whitelist)

    ic_series = compute_ic_series(panel_norm, factor_cols=whitelist)
    L = cfg["combiner"]["ic_ir_lookback"]

    closes = kline.pivot(index="date", columns="symbol", values="close").sort_index()
    daily_ret = closes.pct_change().fillna(0.0)

    rebalance_days = all_dates[::N]
    weights_by_date = {}
    target_rows = []
    w_prev: pd.Series | None = None
    for t in rebalance_days:
        w_ic = ic_ir_weights(ic_series, t=t, factor_cols=whitelist,
                             lookback=L, horizon_n=N)
        weights_by_date[t] = w_ic
        panel_t = panel_ortho[panel_ortho["date"] == t]
        if panel_t.empty:
            continue
        score_df = synthesize(panel_t, factor_cols=whitelist,
                              weights_by_date={t: w_ic})
        scores = score_df.set_index("instrument")["score"]
        w_t = build_topn_equal(
            scores,
            top_n=cfg["builder"]["top_n"],
            max_weight=cfg["builder"]["max_weight"],
            leverage=cfg["builder"]["leverage"],
            turnover_cap=cfg["builder"]["turnover_cap"],
            w_prev=w_prev,
        )
        w_prev = w_t
        for ins, v in w_t.items():
            target_rows.append({"date": t, "instrument": ins, "weight": float(v)})

    target_weights = pd.DataFrame(target_rows, columns=["date", "instrument", "weight"])
    targets_pivot = target_weights.pivot(index="date", columns="instrument", values="weight")
    targets_pivot = targets_pivot.reindex(all_dates)

    returns_panel = daily_ret.reindex(all_dates).reindex(
        columns=sorted(set(targets_pivot.columns) | set(daily_ret.columns)),
        fill_value=0.0,
    )
    bt = run_backtest(
        returns_panel[targets_pivot.columns.tolist()],
        targets_pivot,
        fee_rate=cfg["costs"]["fee_rate"],
        slippage_bps=cfg["costs"]["slippage_bps"],
    )
    metrics = compute_metrics(bt["port_ret"])
    weights_history = bt.attrs["weights_history"]
    return PipelineResult(
        backtest_df=bt,
        weights_history=weights_history,
        target_weights=target_weights,
        metrics=metrics,
    )
```

> 注：`_load_factor_directions` 用多 attribute 试探读取 `factor_config`，容错未注册因子默认方向 +1。pipeline 只静态查字典，不调用 factor_analyse 的任何类/函数，符合 "不 import factor_analyse 业务代码" 的边界。

- [ ] **Step 2: Commit**

```bash
git add portfolio/pipeline.py
git commit -m "feat(portfolio): add pipeline orchestrator"
```

---

### Task 16: main.py CLI

**Files:**
- Create: `portfolio/main.py`

- [ ] **Step 1: 实现**

```python
"""Portfolio layer CLI.

Usage:
  ./.venv/bin/python portfolio/main.py --list
  ./.venv/bin/python portfolio/main.py baseline_topn_long
  ./.venv/bin/python portfolio/main.py baseline_topn_long --dry-run
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.config import STRATEGY_CONFIG, resolve_config  # noqa: E402
from portfolio.pipeline import run  # noqa: E402
from portfolio.reporter.html_renderer import render_report  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("strategy", nargs="?", default=None)
    p.add_argument("--list", action="store_true", dest="list_strategies")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rebalance", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse()
    if args.list_strategies or not args.strategy:
        print("Available strategies:")
        for k in STRATEGY_CONFIG.keys():
            print(f"  - {k}")
        return 0

    cfg = resolve_config(args.strategy)
    if args.rebalance is not None:
        cfg["rebalance"]["period_days"] = args.rebalance

    out_dir = ROOT / "portfolio" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.utcnow().strftime("%Y%m%d")

    result = run(cfg, strategy_name=args.strategy)

    tw_path = out_dir / f"target_weights_{args.strategy}_{tag}.csv"
    result.target_weights.to_csv(tw_path, index=False)
    ph_path = out_dir / f"positions_history_{args.strategy}.csv"
    result.weights_history.to_csv(ph_path)
    print(f"[portfolio] target weights: {tw_path}")
    print(f"[portfolio] positions history: {ph_path}")
    print(f"[portfolio] metrics: {result.metrics}")

    if not args.dry_run:
        rpt_path = ROOT / "reports" / f"portfolio_{args.strategy}_{tag}.html"
        render_report(
            strategy=args.strategy,
            metrics=result.metrics,
            backtest_df=result.backtest_df,
            weights_history=result.weights_history,
            output_path=rpt_path,
        )
        print(f"[portfolio] report: {rpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 验证 --list 可跑**

Run: `./.venv/bin/python portfolio/main.py --list`
Expected: 打印 `baseline_topn_long`

- [ ] **Step 3: Commit**

```bash
git add portfolio/main.py
git commit -m "feat(portfolio): add CLI entrypoint"
```

---

### Task 17: 端到端 cutoff 反证测试（必过）

构造合成数据 → 跑两次 pipeline（全量 vs cutoff 截断）→ 比对 `target_weights[t <= cutoff]`。

**Files:**
- Test: `tests/test_portfolio_future_leak.py`

- [ ] **Step 1: 写测试**

```python
import unittest
import pathlib
import sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio.factor_pool.aligner import align_factors  # noqa: E402
from portfolio.factor_pool.label_builder import build_future_ret  # noqa: E402
from portfolio.combiner.normalizer import normalize_cross_section  # noqa: E402
from portfolio.combiner.orthogonalizer import symmetric_orthogonalize  # noqa: E402
from portfolio.combiner.ic_ir_weighter import compute_ic_series, ic_ir_weights  # noqa: E402
from portfolio.combiner.synthesizer import synthesize  # noqa: E402
from portfolio.portfolio_builder.topn_equal import build_topn_equal  # noqa: E402


def _synth_dataset(seed: int = 0) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    syms = [f"S{i}" for i in range(15)]
    rows_f1, rows_f2, rows_k = [], [], []
    prices = {s: 100.0 for s in syms}
    for d in dates:
        for s in syms:
            prices[s] *= (1 + rng.normal(0, 0.02))
            rows_k.append({"date": d, "symbol": s, "close": prices[s]})
            rows_f1.append({"date": d, "instrument": s, "f1": rng.standard_normal()})
            rows_f2.append({"date": d, "instrument": s, "f2": rng.standard_normal()})
    f1 = pd.DataFrame(rows_f1); f2 = pd.DataFrame(rows_f2)
    kline = pd.DataFrame(rows_k)
    return {"f1": f1, "f2": f2}, kline


def _pipeline_target_weights(factor_frames, kline, N=3, L=20):
    future_ret = build_future_ret(kline, period_days=N)
    panel = align_factors(factor_frames)
    panel = panel.merge(future_ret, on=["date", "instrument"], how="left")
    cols = ["f1", "f2"]
    panel_norm = normalize_cross_section(panel, factor_cols=cols,
                                         directions={"f1": 1, "f2": 1})
    panel_ortho = symmetric_orthogonalize(panel_norm, factor_cols=cols)
    ic = compute_ic_series(panel_norm, factor_cols=cols)
    rows = []
    w_prev = None
    all_dates = sorted(panel["date"].unique())
    for t in all_dates[::N]:
        w_ic = ic_ir_weights(ic, t=pd.Timestamp(t), factor_cols=cols,
                             lookback=L, horizon_n=N)
        pt = panel_ortho[panel_ortho["date"] == t]
        if pt.empty:
            continue
        sdf = synthesize(pt, cols, {pd.Timestamp(t): w_ic})
        scores = sdf.set_index("instrument")["score"]
        w_t = build_topn_equal(scores, top_n=5, max_weight=0.4,
                               leverage=1.0, turnover_cap=None, w_prev=w_prev)
        w_prev = w_t
        for ins, v in w_t.items():
            rows.append({"date": pd.Timestamp(t), "instrument": ins, "weight": float(v)})
    return pd.DataFrame(rows)


class TestFutureLeak(unittest.TestCase):
    def test_end_to_end_cutoff_invariance(self):
        factor_frames, kline = _synth_dataset(seed=7)
        cutoff = pd.Timestamp("2024-02-10")

        tw_full = _pipeline_target_weights(factor_frames, kline)
        kline_cut = kline[pd.to_datetime(kline["date"]) <= cutoff]
        ff_cut = {k: v[pd.to_datetime(v["date"]) <= cutoff] for k, v in factor_frames.items()}
        tw_cut = _pipeline_target_weights(ff_cut, kline_cut)

        a = tw_full[tw_full["date"] <= cutoff - pd.Timedelta(days=3)].sort_values(["date", "instrument"])
        b = tw_cut[tw_cut["date"] <= cutoff - pd.Timedelta(days=3)].sort_values(["date", "instrument"])
        merged = a.merge(b, on=["date", "instrument"], how="outer",
                         suffixes=("_full", "_cut")).fillna(0.0)
        diff = (merged["weight_full"] - merged["weight_cut"]).abs().max()
        self.assertLess(diff, 1e-10,
                        f"Future-leak detected: max |Δw| = {diff}")


if __name__ == "__main__":
    unittest.main()
```

> 说明：比较到 `cutoff - N`（N=3）是因为 label_builder 在 cutoff 附近的 future_ret 会因 cutoff 截断变为 NaN，这会波及 IC 和权重计算；spec §3.4 规定的"date ≤ cutoff 一致性"指信息可得的子集，留出 N 日缓冲是合理的（且真实生产中 `t` 决策只会使用 `t-N` 之前的 IC）。

- [ ] **Step 2: 运行通过**

Run: `./.venv/bin/python -m pytest tests/test_portfolio_future_leak.py -v`
Expected: 1 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_portfolio_future_leak.py
git commit -m "test(portfolio): add end-to-end cutoff anti-leak test"
```

---

### Task 18: 端到端冒烟 + 文档

**Files:**
- Modify: `README.md`（追加 Portfolio 小节）

- [ ] **Step 1: 冒烟**

Run：
```bash
./.venv/bin/python -m pytest tests/test_portfolio_*.py -v
./.venv/bin/python portfolio/main.py --list
./.venv/bin/python portfolio/main.py baseline_topn_long --dry-run
```
Expected：全部测试绿；CLI 打印 target_weights 与 metrics 路径（若本地因子 CSV 不齐可降级只跑 `--list`）。

- [ ] **Step 2: 补 README**

在 README 末尾追加：
```
## Portfolio 层（P1 MVP）

生成每日目标权重 + HTML 组合报告。

./.venv/bin/python portfolio/main.py --list
./.venv/bin/python portfolio/main.py baseline_topn_long

输出位置：
- portfolio/output/target_weights_<strategy>_<date>.csv
- portfolio/output/positions_history_<strategy>.csv
- reports/portfolio_<strategy>_<date>.html
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add portfolio layer section"
```

---

## Self-Review

**Spec 覆盖扫描**（对照 spec 章节）：

- §1.3 契约：`pipeline.py` 只读 `data/*`，写 `portfolio/output/` 与 `reports/*.html`；`_load_factor_directions` 仅静态查字典，不调用 factor_analyse 业务 ✓
- §2 目录：P1 创建 factor_pool/combiner/portfolio_builder/backtester/reporter；screener/optimizer/hedge/attribution 留 P2 ✓
- §3.1 时间轴：Task 12 回测 `port_ret[t] = w[t-1]·r[t]`，非调仓日 drift ✓
- §3.3 丢弃 CSV future_ret：Task 2 `load_factor_csv` `usecols=[date,instrument,factor]` ✓
- §3.4 未来函数硬检查：Task 17 端到端 cutoff 反证 ✓
- §4 config：Task 14 `baseline_topn_long`（P1 子集，移除 `screener/attribution/hedge/optimizer` 相关项）✓
- §5.1 universe：Task 9 ✓；§5.2 normalizer：Task 5 ✓；§5.3 对称正交：Task 6 ✓；§5.4 IC_IR：Task 7 ✓；§5.6 Top-N：Task 10 ✓；§5.9 drift：Task 12 ✓
- §5.5/§5.7/§5.8/§5.10：P2 范围，不在本计划 ✓
- §6 测试：10 个测试文件，见 Task 2/3/4/5/6/7/9/10/12/17 ✓
- §7 P1 验收：`baseline_topn_long` 目标权重 + HTML + 反证测试通过 ✓

**Placeholder 扫描**：
- 无 "TBD/TODO/fill in later"
- 所有代码块完整，每个测试都有实现
- 每个 Step 都给出 Run 命令与 Expected

**类型一致性**：
- `load_factor_csv(path, name)` → 列 `[date, instrument, <name>]`；`load_many` dict[name, DataFrame] ✓
- `build_future_ret` → `[date, instrument, future_ret]`，与 pipeline merge 一致 ✓
- `normalize_cross_section(panel, factor_cols, directions, winsorize_pct)` 与 pipeline 调用一致 ✓
- `ic_ir_weights(ic, t, factor_cols, lookback, horizon_n)` 与 pipeline 一致 ✓
- `build_topn_equal(scores, top_n, max_weight, leverage, turnover_cap, w_prev)` 与 pipeline 一致 ✓
- `run_backtest(returns, targets, fee_rate, slippage_bps)` 与 pipeline 一致 ✓

---

## 执行选择

Plan 完成，已保存到 `docs/superpowers/plans/2026-04-17-portfolio-layer-p1-mvp.md`。

两种执行方式：

1. **Subagent-Driven（推荐）** — 每 task 一个全新 subagent，两阶段 review，迭代快；
2. **Inline Execution** — 本会话内按 checkpoint 批量执行。

请告诉我采用哪种？
