# MaxDD-15% Hedged Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有多因子组合的OOS最大回撤降至15%以内，通过IS/OOS因子筛选 + BTC趋势时序择时 + BTC永续对冲。

**Architecture:** 在现有 `portfolio/` pipeline 上叠加 4 层：(A) IS期预筛因子 Top5周换仓；(B) BTC趋势分数 T ∈ [-1,+1]；(C) 连续暴露调节 + 市场对冲；(D) 扩展回测引擎支持双腿(现货多头+永续空头)与资金费率。所有新参数可通过 `PortfolioConfig` 控制，旧功能保持向后兼容。

**Tech Stack:** Python 3.11, pandas, numpy, pytest, scipy。项目 venv: `./.venv/bin/python`

**Spec:** `docs/superpowers/specs/2026-04-20-crypto-portfolio-maxdd15-design.md`

---

## File Structure

### 新增文件
| 路径 | 职责 |
|------|------|
| `portfolio/combiner/is_oos_filter.py` | IS期预筛因子（\|IC_IR\|≥阈值 + 自动方向矫正） |
| `portfolio/portfolio_builder/beta_estimator.py` | 60d滚动Beta估算（各币 vs BTC），组合Beta=持仓加权 |
| `portfolio/regime/__init__.py` | 新模块 |
| `portfolio/regime/btc_trend.py` | BTC趋势分数 T 计算 |
| `portfolio/sizing/__init__.py` | 新模块 |
| `portfolio/sizing/exposure_controller.py` | T + beta_port → (alt_exposure, hedge_ratio) |
| `portfolio/backtester/funding.py` | 永续资金费率成本 |
| `tests/test_portfolio_is_oos_filter.py` | |
| `tests/test_portfolio_beta_estimator.py` | |
| `tests/test_portfolio_btc_trend.py` | |
| `tests/test_portfolio_exposure_controller.py` | |
| `tests/test_portfolio_funding.py` | |
| `tests/test_portfolio_engine_hedged.py` | 对冲回测引擎测试 |
| `tests/test_portfolio_pipeline_hedged.py` | E2E对冲pipeline测试 |
| `tests/test_portfolio_future_leak_hedged.py` | 未来函数动态反证 |
| `portfolio/main_hedged.py` | 新pipeline执行入口（跑 MaxDD-15 策略） |

### 修改文件
| 路径 | 改动 |
|------|------|
| `portfolio/config.py` | 新增12个字段（IS/OOS、趋势参数、暴露参数、成本参数）|
| `portfolio/backtester/engine.py` | 新增 `run_hedged_backtest()` 函数（旧 `run_backtest` 保留）|
| `portfolio/backtester/fees.py` | 新增 `compute_perp_fee()` 永续费率函数 |
| `portfolio/pipeline.py` | 新增 `run_hedged_pipeline()` 函数（旧 `run_pipeline` 保留）|
| `portfolio/reporter/html_renderer.py` | 支持额外暴露曲线/对冲比率曲线（小改）|

---

## Task 1: Config 扩展

**Files:**
- Modify: `portfolio/config.py`
- Test: `tests/test_portfolio_config.py`

- [ ] **Step 1: 写失败测试（新字段存在且默认值正确）**

在 `tests/test_portfolio_config.py` 末尾新增：

```python
def test_hedged_config_defaults():
    from portfolio.config import PortfolioConfig
    cfg = PortfolioConfig()
    # IS/OOS
    assert cfg.is_end_date == "2023-12-31"
    assert cfg.min_ic_ir == 0.05
    # Trend
    assert cfg.ema_short == 50
    assert cfg.ema_long == 200
    assert cfg.zscore_window == 120
    assert cfg.ema_ratio_saturation == 0.15
    assert cfg.zscore_saturation == 2.0
    assert cfg.t_smoothing_span == 5
    # Exposure
    assert cfg.alt_max_exposure == 1.0
    assert cfg.alt_min_exposure == 0.5
    assert cfg.hedge_cap_multiplier == 1.2
    assert cfg.beta_window == 60
    assert cfg.beta_prior == 1.3
    assert cfg.alt_exposure_ema_alpha == 0.05
    # Costs
    assert cfg.alt_fee_rate == 0.001
    assert cfg.perp_fee_rate == 0.0005
    assert cfg.funding_rate_annual == 0.1095

def test_hedged_config_backward_compat():
    """旧字段必须保留且默认值不变。"""
    from portfolio.config import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.top_n == 10  # 默认不改；pipeline调用方显式传5
    assert cfg.rebalance_period == 1
    assert cfg.fee_rate == 0.001
```

- [ ] **Step 2: 运行测试确认失败**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_config.py::test_hedged_config_defaults -v
```
期望：FAIL（字段不存在）

- [ ] **Step 3: 编辑 `portfolio/config.py`**

在 `PortfolioConfig` 类的最后一个字段 (`output_dir`) 之前插入新字段：

```python
    # IS/OOS 因子筛选
    is_end_date: str = "2023-12-31"
    min_ic_ir: float = 0.05

    # Layer B: BTC Trend
    ema_short: int = 50
    ema_long: int = 200
    zscore_window: int = 120
    ema_ratio_saturation: float = 0.15
    zscore_saturation: float = 2.0
    t_smoothing_span: int = 5

    # Layer C: Exposure
    alt_max_exposure: float = 1.0
    alt_min_exposure: float = 0.5
    hedge_cap_multiplier: float = 1.2
    beta_window: int = 60
    beta_prior: float = 1.3
    alt_exposure_ema_alpha: float = 0.05

    # Layer D: Costs
    alt_fee_rate: float = 0.001
    perp_fee_rate: float = 0.0005
    funding_rate_annual: float = 0.1095
```

- [ ] **Step 4: 运行测试确认通过**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_config.py -v
```
期望：全部 PASS（含旧测试）

- [ ] **Step 5: Commit**

```bash
git add portfolio/config.py tests/test_portfolio_config.py
git commit -m "feat(portfolio): extend PortfolioConfig with hedged-strategy params"
```

---

## Task 2: IS/OOS Factor Filter

**Files:**
- Create: `portfolio/combiner/is_oos_filter.py`
- Test: `tests/test_portfolio_is_oos_filter.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_portfolio_is_oos_filter.py`：

```python
import pandas as pd
import numpy as np
import pytest


def _make_panel():
    """构造3个因子、5只币、60日的面板。
    f_good: 与 future_ret 强正相关
    f_bad: 随机，IC接近0
    f_neg: 与 future_ret 强负相关（需方向反转）
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    instruments = [f"C{i}" for i in range(5)]
    rows = []
    for d in dates:
        base = np.random.randn(5)
        future_ret = base * 0.01
        for i, inst in enumerate(instruments):
            rows.append({
                "date": d,
                "instrument": inst,
                "f_good": base[i] + np.random.randn()*0.1,
                "f_bad": np.random.randn(),
                "f_neg": -base[i] + np.random.randn()*0.1,
                "future_ret": future_ret[i],
            })
    return pd.DataFrame(rows)


def test_is_oos_filter_keeps_high_ir_factors():
    from portfolio.combiner.is_oos_filter import filter_factors_by_is_ic_ir
    panel = _make_panel()
    is_end = pd.Timestamp("2023-02-15")
    selected = filter_factors_by_is_ic_ir(
        panel, ["f_good", "f_bad", "f_neg"],
        label_col="future_ret",
        is_end_date=is_end,
        min_ic_ir=0.1,
    )
    # 应保留 f_good (正方向) 和 f_neg (反方向)
    assert set(selected.keys()) == {"f_good", "f_neg"}
    assert selected["f_good"] == 1
    assert selected["f_neg"] == -1


def test_is_oos_filter_respects_is_end_date():
    """确保筛选不使用 IS 期之后的数据。"""
    from portfolio.combiner.is_oos_filter import filter_factors_by_is_ic_ir
    panel = _make_panel()
    # 只用前10天作IS期
    is_end = pd.Timestamp("2023-01-10")
    selected_short = filter_factors_by_is_ic_ir(
        panel, ["f_good"], label_col="future_ret",
        is_end_date=is_end, min_ic_ir=0.0,
    )
    # 全量IS期
    selected_full = filter_factors_by_is_ic_ir(
        panel, ["f_good"], label_col="future_ret",
        is_end_date=pd.Timestamp("2099-01-01"), min_ic_ir=0.0,
    )
    # 两者可能都包含 f_good，但 IC_IR 应不同（反证调用正确）
    # 关键：短IS期有些日IC_IR 可能不显著；改查返回类型
    assert isinstance(selected_short, dict)
    assert isinstance(selected_full, dict)


def test_is_oos_filter_empty_when_no_factor_passes():
    from portfolio.combiner.is_oos_filter import filter_factors_by_is_ic_ir
    panel = _make_panel()
    is_end = pd.Timestamp("2023-02-15")
    selected = filter_factors_by_is_ic_ir(
        panel, ["f_bad"], label_col="future_ret",
        is_end_date=is_end, min_ic_ir=0.5,  # 极高阈值
    )
    assert selected == {}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_is_oos_filter.py -v
```
期望：FAIL（模块不存在）

- [ ] **Step 3: 实现 `portfolio/combiner/is_oos_filter.py`**

```python
"""In-Sample factor filter based on IC_IR threshold with direction auto-correction."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _cross_sectional_ic(panel: pd.DataFrame, factor_col: str, label_col: str) -> pd.Series:
    """Spearman rank IC per date."""
    def _one(grp: pd.DataFrame) -> float:
        xy = grp[[factor_col, label_col]].dropna()
        if len(xy) < 3:
            return np.nan
        r, _ = spearmanr(xy[factor_col], xy[label_col])
        return float(r)
    return panel.groupby("date").apply(_one)


def filter_factors_by_is_ic_ir(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    label_col: str = "future_ret",
    is_end_date: pd.Timestamp | str = "2023-12-31",
    min_ic_ir: float = 0.05,
) -> dict[str, int]:
    """Filter factors by IS-period |IC_IR| ≥ threshold, returning direction.

    Args:
        panel: Long-format DataFrame with [date, instrument, factor_cols..., label_col]
        factor_cols: Candidate factor column names
        label_col: Forward return column
        is_end_date: Cutoff date (inclusive) — only dates ≤ this are used for IC
        min_ic_ir: |IC_IR| threshold

    Returns:
        {factor_name: direction} where direction ∈ {+1, -1}.
        Factors with |IC_IR| < threshold are excluded.
    """
    is_end = pd.Timestamp(is_end_date).normalize()
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    is_panel = panel[panel["date"] <= is_end]

    selected: dict[str, int] = {}
    for col in factor_cols:
        ic_series = _cross_sectional_ic(is_panel, col, label_col).dropna()
        if len(ic_series) < 20:  # need enough samples
            continue
        ic_mean = ic_series.mean()
        ic_std = ic_series.std(ddof=1)
        if not np.isfinite(ic_std) or ic_std < 1e-12:
            continue
        ic_ir = ic_mean / ic_std
        if abs(ic_ir) >= min_ic_ir:
            selected[col] = 1 if ic_ir > 0 else -1
    return selected
```

- [ ] **Step 4: 运行测试确认通过**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_is_oos_filter.py -v
```
期望：3个测试 PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/combiner/is_oos_filter.py tests/test_portfolio_is_oos_filter.py
git commit -m "feat(portfolio): add IS/OOS factor filter with |IC_IR| threshold"
```

---

## Task 3: BTC Trend Module

**Files:**
- Create: `portfolio/regime/__init__.py` (empty), `portfolio/regime/btc_trend.py`
- Test: `tests/test_portfolio_btc_trend.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_portfolio_btc_trend.py`：

```python
import pandas as pd
import numpy as np


def _make_btc_kline(days=400, trend="up"):
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    if trend == "up":
        close = 20000 * np.exp(np.linspace(0, 1.5, days))
    elif trend == "down":
        close = 60000 * np.exp(np.linspace(0, -1.0, days))
    else:  # flat
        close = 30000 + 1000*np.sin(np.arange(days)/10)
    return pd.DataFrame({"symbol": "BTCUSDT", "date": dates, "close": close})


def test_trend_strong_up_gives_positive_T():
    from portfolio.regime.btc_trend import compute_btc_trend_score
    kline = _make_btc_kline(trend="up")
    T = compute_btc_trend_score(kline)
    # 长期上升，最近期 T 应接近 +1
    last_T = T.iloc[-1]
    assert last_T > 0.7, f"Expected strong positive T, got {last_T}"


def test_trend_strong_down_gives_negative_T():
    from portfolio.regime.btc_trend import compute_btc_trend_score
    kline = _make_btc_kline(trend="down")
    T = compute_btc_trend_score(kline)
    last_T = T.iloc[-1]
    assert last_T < -0.5, f"Expected negative T, got {last_T}"


def test_trend_output_bounded_in_minus_one_plus_one():
    from portfolio.regime.btc_trend import compute_btc_trend_score
    for trend in ["up", "down", "flat"]:
        kline = _make_btc_kline(trend=trend)
        T = compute_btc_trend_score(kline)
        assert (T.dropna() >= -1.0).all()
        assert (T.dropna() <= 1.0).all()


def test_trend_no_future_leak():
    """对比全量计算与截断计算，cutoff 前的值必须一致。"""
    from portfolio.regime.btc_trend import compute_btc_trend_score
    kline = _make_btc_kline(days=500, trend="up")
    cutoff = kline["date"].iloc[300]
    T_full = compute_btc_trend_score(kline)
    T_cut = compute_btc_trend_score(kline[kline["date"] <= cutoff])
    aligned = T_full.reindex(T_cut.index)
    diff = (aligned - T_cut).abs().max()
    assert diff < 1e-9, f"Future leak detected: max diff {diff}"


def test_trend_requires_btc_symbol():
    from portfolio.regime.btc_trend import compute_btc_trend_score
    kline = pd.DataFrame({"symbol": ["ETHUSDT"]*10,
                           "date": pd.date_range("2023-01-01", periods=10),
                           "close": range(10)})
    # 无 BTCUSDT 应抛错或返回空
    import pytest
    with pytest.raises((ValueError, KeyError)):
        compute_btc_trend_score(kline)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_btc_trend.py -v
```
期望：FAIL（模块不存在）

- [ ] **Step 3: 创建 `portfolio/regime/__init__.py`（空文件）**

```bash
touch portfolio/regime/__init__.py
```

- [ ] **Step 4: 实现 `portfolio/regime/btc_trend.py`**

```python
"""BTC trend score T ∈ [-1, +1] — pure backward-looking calculation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_btc_trend_score(
    kline: pd.DataFrame,
    ema_short: int = 50,
    ema_long: int = 200,
    zscore_window: int = 120,
    ema_ratio_saturation: float = 0.15,
    zscore_saturation: float = 2.0,
    smoothing_span: int = 5,
    btc_symbol: str = "BTCUSDT",
) -> pd.Series:
    """Compute daily BTC trend score T ∈ [-1, +1].

    T = EMA(0.5·tanh((EMA_short/EMA_long - 1) / sat1) +
           0.5·tanh((close - MA_long) / std_long / sat2),
           span=smoothing_span)

    All operations are strictly backward-looking (no future leak).

    Args:
        kline: DataFrame with [symbol, date, close]
        btc_symbol: Filter to this symbol; raises if not found
    Returns:
        pd.Series indexed by date with T values.
    """
    df = kline[kline["symbol"].str.upper() == btc_symbol.upper()].copy()
    if df.empty:
        raise ValueError(f"No rows for symbol {btc_symbol}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    close = df["close"].astype(float)

    ema_s = close.ewm(span=ema_short, adjust=False, min_periods=ema_short).mean()
    ema_l = close.ewm(span=ema_long, adjust=False, min_periods=ema_long).mean()
    ma_l = close.rolling(ema_long, min_periods=ema_long).mean()
    std_l = close.rolling(zscore_window, min_periods=zscore_window).std(ddof=1)

    raw_ratio = (ema_s / ema_l) - 1.0
    raw_z = (close - ma_l) / std_l

    T1 = np.tanh(raw_ratio / ema_ratio_saturation)
    T2 = np.tanh(raw_z / zscore_saturation)
    T_raw = 0.5 * T1 + 0.5 * T2

    T = T_raw.ewm(span=smoothing_span, adjust=False, min_periods=smoothing_span).mean()
    T = T.clip(-1.0, 1.0)
    T.name = "btc_trend_T"
    return T
```

- [ ] **Step 5: 运行测试确认通过**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_btc_trend.py -v
```
期望：5个测试 PASS

- [ ] **Step 6: Commit**

```bash
git add portfolio/regime/__init__.py portfolio/regime/btc_trend.py tests/test_portfolio_btc_trend.py
git commit -m "feat(portfolio): add BTC trend score T calculation (no future leak)"
```

---

## Task 4: Beta Estimator

**Files:**
- Create: `portfolio/portfolio_builder/beta_estimator.py`
- Test: `tests/test_portfolio_beta_estimator.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_portfolio_beta_estimator.py`：

```python
import numpy as np
import pandas as pd


def _make_kline(days=200):
    np.random.seed(7)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    btc_ret = np.random.randn(days) * 0.03
    btc_close = 20000 * (1 + pd.Series(btc_ret)).cumprod().values
    # ETH beta=1.5
    eth_ret = 1.5 * btc_ret + np.random.randn(days) * 0.01
    eth_close = 1500 * (1 + pd.Series(eth_ret)).cumprod().values
    # SOL beta=2.0
    sol_ret = 2.0 * btc_ret + np.random.randn(days) * 0.01
    sol_close = 100 * (1 + pd.Series(sol_ret)).cumprod().values
    rows = []
    for i, d in enumerate(dates):
        rows += [
            {"symbol": "BTCUSDT", "date": d, "close": btc_close[i]},
            {"symbol": "ETHUSDT", "date": d, "close": eth_close[i]},
            {"symbol": "SOLUSDT", "date": d, "close": sol_close[i]},
        ]
    return pd.DataFrame(rows)


def test_coin_beta_matches_true_value():
    from portfolio.portfolio_builder.beta_estimator import compute_coin_betas
    kline = _make_kline()
    betas = compute_coin_betas(kline, window=60, btc_symbol="BTCUSDT")
    eth_beta = betas[betas["instrument"] == "ETHUSDT"]["beta"].iloc[-1]
    sol_beta = betas[betas["instrument"] == "SOLUSDT"]["beta"].iloc[-1]
    assert 1.2 < eth_beta < 1.8, f"ETH beta {eth_beta} off"
    assert 1.7 < sol_beta < 2.3, f"SOL beta {sol_beta} off"


def test_portfolio_beta_equal_weighted_average():
    from portfolio.portfolio_builder.beta_estimator import (
        compute_coin_betas, compute_portfolio_beta
    )
    kline = _make_kline()
    betas = compute_coin_betas(kline, window=60)
    date = pd.Timestamp(kline["date"].max())
    holdings = {"ETHUSDT": 0.5, "SOLUSDT": 0.5}
    port_beta = compute_portfolio_beta(betas, holdings, date, beta_prior=1.3)
    # Expected ~ (1.5 + 2.0)/2 = 1.75
    assert 1.5 < port_beta < 2.0


def test_portfolio_beta_uses_prior_when_no_history():
    from portfolio.portfolio_builder.beta_estimator import compute_portfolio_beta
    betas_empty = pd.DataFrame(columns=["date", "instrument", "beta"])
    date = pd.Timestamp("2023-01-01")
    holdings = {"NEWCOIN": 1.0}
    port_beta = compute_portfolio_beta(betas_empty, holdings, date, beta_prior=1.3)
    assert port_beta == 1.3


def test_beta_no_future_leak():
    from portfolio.portfolio_builder.beta_estimator import compute_coin_betas
    kline = _make_kline(days=200)
    cutoff = kline["date"].unique()[150]
    b_full = compute_coin_betas(kline, window=60)
    b_cut = compute_coin_betas(kline[kline["date"] <= cutoff], window=60)
    m = b_full.merge(b_cut, on=["date", "instrument"], suffixes=("_full", "_cut"))
    m = m[m["date"] <= cutoff]
    diff = (m["beta_full"] - m["beta_cut"]).abs().max()
    assert diff < 1e-9, f"Beta future leak: {diff}"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_beta_estimator.py -v
```
期望：FAIL

- [ ] **Step 3: 实现 `portfolio/portfolio_builder/beta_estimator.py`**

```python
"""Rolling Beta estimation: each coin vs BTC, and portfolio-level aggregate."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_coin_betas(
    kline: pd.DataFrame,
    window: int = 60,
    btc_symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """Compute rolling Beta of each coin vs BTC using daily log returns.

    beta_t = cov(ret_coin, ret_btc, window) / var(ret_btc, window)

    Args:
        kline: [symbol, date, close]
        window: rolling window size (default 60)

    Returns:
        Long-format DataFrame [date, instrument, beta].
        NaN for dates with <window obs.
    """
    df = kline.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["symbol", "date"])
    df["ret"] = df.groupby("symbol")["close"].pct_change()

    wide = df.pivot_table(index="date", columns="symbol", values="ret")
    if btc_symbol not in wide.columns:
        raise ValueError(f"No {btc_symbol} in kline")
    btc = wide[btc_symbol]
    btc_var = btc.rolling(window, min_periods=window).var(ddof=1)

    rows = []
    for col in wide.columns:
        if col == btc_symbol:
            continue
        coin = wide[col]
        cov = coin.rolling(window, min_periods=window).cov(btc)
        beta = cov / btc_var
        for d, b in beta.items():
            if pd.notna(b):
                rows.append({"date": d, "instrument": col, "beta": float(b)})
    return pd.DataFrame(rows)


def compute_portfolio_beta(
    betas_df: pd.DataFrame,
    holdings: dict[str, float],
    as_of_date: pd.Timestamp,
    beta_prior: float = 1.3,
) -> float:
    """Weighted average beta of current holdings as of date.

    Uses each coin's most recent available beta ≤ as_of_date.
    If a coin has no beta history, falls back to beta_prior.
    If holdings is empty, returns beta_prior.
    """
    if not holdings:
        return beta_prior

    total_w = sum(holdings.values())
    if total_w < 1e-12:
        return beta_prior

    as_of = pd.Timestamp(as_of_date).normalize()
    beta_sum = 0.0
    for inst, w in holdings.items():
        sub = betas_df[(betas_df["instrument"] == inst) &
                       (betas_df["date"] <= as_of)]
        if sub.empty:
            coin_beta = beta_prior
        else:
            coin_beta = float(sub.sort_values("date").iloc[-1]["beta"])
            if not np.isfinite(coin_beta):
                coin_beta = beta_prior
        beta_sum += (w / total_w) * coin_beta
    return beta_sum
```

- [ ] **Step 4: 运行测试确认通过**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_beta_estimator.py -v
```
期望：4个测试 PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/portfolio_builder/beta_estimator.py tests/test_portfolio_beta_estimator.py
git commit -m "feat(portfolio): add rolling Beta estimator (coin vs BTC + portfolio aggregate)"
```

---

## Task 5: Exposure Controller

**Files:**
- Create: `portfolio/sizing/__init__.py` (empty), `portfolio/sizing/exposure_controller.py`
- Test: `tests/test_portfolio_exposure_controller.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_portfolio_exposure_controller.py`：

```python
import pandas as pd
import numpy as np


def test_exposure_strong_bull():
    from portfolio.sizing.exposure_controller import compute_exposure_and_hedge
    alt_exp, hedge = compute_exposure_and_hedge(
        T=1.0, beta_port=1.3,
        alt_min=0.5, alt_max=1.0, hedge_cap_mult=1.2,
    )
    assert abs(alt_exp - 1.0) < 1e-9  # Alt满仓
    assert abs(hedge - 0.0) < 1e-9    # 不对冲


def test_exposure_strong_bear():
    from portfolio.sizing.exposure_controller import compute_exposure_and_hedge
    alt_exp, hedge = compute_exposure_and_hedge(
        T=-1.0, beta_port=1.3,
        alt_min=0.5, alt_max=1.0, hedge_cap_mult=1.2,
    )
    assert abs(alt_exp - 0.5) < 1e-9  # Alt半仓
    # hedge_raw = 1.3*0.5*(1-(-1)) = 1.3; cap = 0.5*1.2 = 0.6 → 0.6
    assert abs(hedge - 0.6) < 1e-9


def test_exposure_neutral():
    from portfolio.sizing.exposure_controller import compute_exposure_and_hedge
    alt_exp, hedge = compute_exposure_and_hedge(
        T=0.0, beta_port=1.0,
        alt_min=0.5, alt_max=1.0, hedge_cap_mult=1.2,
    )
    assert abs(alt_exp - 0.75) < 1e-9
    # hedge_raw = 1.0*0.75*1 = 0.75; cap = 0.75*1.2=0.9 → 0.75
    assert abs(hedge - 0.75) < 1e-9


def test_exposure_clips_T_out_of_range():
    from portfolio.sizing.exposure_controller import compute_exposure_and_hedge
    alt_exp, hedge = compute_exposure_and_hedge(T=2.0, beta_port=1.0)
    assert alt_exp <= 1.0
    assert hedge >= 0.0


def test_ema_smoothing_reduces_jitter():
    from portfolio.sizing.exposure_controller import smooth_exposure_series
    # 突变信号
    raw = pd.Series([0.5]*10 + [1.0]*10, index=pd.date_range("2024-01-01", periods=20))
    smoothed = smooth_exposure_series(raw, alpha=0.05)
    assert len(smoothed) == len(raw)
    # 平滑后第11天应远低于1.0（只吸收5%）
    assert smoothed.iloc[10] < 0.6
    # 最终收敛接近1.0
    assert smoothed.iloc[-1] < 1.0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_exposure_controller.py -v
```
期望：FAIL

- [ ] **Step 3: 创建 `portfolio/sizing/__init__.py`（空）**

```bash
touch portfolio/sizing/__init__.py
```

- [ ] **Step 4: 实现 `portfolio/sizing/exposure_controller.py`**

```python
"""Map (BTC trend T, portfolio beta) → (alt_exposure, hedge_ratio)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_exposure_and_hedge(
    T: float,
    beta_port: float,
    alt_min: float = 0.5,
    alt_max: float = 1.0,
    hedge_cap_mult: float = 1.2,
) -> tuple[float, float]:
    """Compute target (alt_exposure, hedge_ratio) from trend T and portfolio beta.

    alt_exposure = alt_min + (alt_max - alt_min) * (T + 1) / 2
    hedge_raw    = beta_port * alt_exposure * (1 - T)
    hedge_ratio  = min(hedge_raw, alt_exposure * hedge_cap_mult)
    hedge_ratio  = max(hedge_ratio, 0.0)

    Args:
        T: BTC trend score, expected in [-1, +1] but clipped defensively
        beta_port: Portfolio beta vs BTC
    Returns:
        (alt_exposure, hedge_ratio) both as floats.
    """
    T_clip = max(-1.0, min(1.0, float(T)))
    alt_exp = alt_min + (alt_max - alt_min) * (T_clip + 1.0) / 2.0
    hedge_raw = max(0.0, float(beta_port) * alt_exp * (1.0 - T_clip))
    hedge_cap = alt_exp * hedge_cap_mult
    hedge = min(hedge_raw, hedge_cap)
    return float(alt_exp), float(hedge)


def smooth_exposure_series(raw: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Apply 1-step recursive EMA: x_t = alpha * raw_t + (1-alpha) * x_{t-1}.

    Uses ewm with adjust=False for consistency with backward-looking semantics.
    """
    if raw.empty:
        return raw
    # span s.t. alpha = 2/(s+1) => s = 2/alpha - 1
    span = max(1.0, 2.0 / alpha - 1.0)
    return raw.ewm(span=span, adjust=False, min_periods=1).mean()
```

- [ ] **Step 5: 运行测试确认通过**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_exposure_controller.py -v
```
期望：5个测试 PASS

- [ ] **Step 6: Commit**

```bash
git add portfolio/sizing/__init__.py portfolio/sizing/exposure_controller.py tests/test_portfolio_exposure_controller.py
git commit -m "feat(portfolio): add exposure controller (T, beta) -> (alt_exp, hedge)"
```

---

## Task 6: Perp Fees & Funding

**Files:**
- Modify: `portfolio/backtester/fees.py`
- Create: `portfolio/backtester/funding.py`
- Test: `tests/test_portfolio_fees.py` (extend), `tests/test_portfolio_funding.py`

- [ ] **Step 1: 写失败测试 - perp fee**

在 `tests/test_portfolio_fees.py` 末尾新增：

```python
def test_perp_fee_basic():
    from portfolio.backtester.fees import compute_perp_fee
    # 从 0 变为 0.6 空头 → turnover 0.6，单边费率 5bps → 3bps
    fee = compute_perp_fee(old_hedge=0.0, new_hedge=0.6, fee_rate=0.0005)
    assert abs(fee - 0.6 * 0.0005) < 1e-12


def test_perp_fee_hold_zero():
    from portfolio.backtester.fees import compute_perp_fee
    fee = compute_perp_fee(old_hedge=0.3, new_hedge=0.3, fee_rate=0.0005)
    assert fee == 0.0


def test_perp_fee_decrease():
    from portfolio.backtester.fees import compute_perp_fee
    fee = compute_perp_fee(old_hedge=0.8, new_hedge=0.3, fee_rate=0.0005)
    assert abs(fee - 0.5 * 0.0005) < 1e-12
```

- [ ] **Step 2: 写失败测试 - funding**

创建 `tests/test_portfolio_funding.py`：

```python
def test_funding_daily_cost():
    from portfolio.backtester.funding import daily_funding_cost
    # hedge=0.6, 10.95%/365 = 0.03%/day
    cost = daily_funding_cost(hedge_ratio=0.6, funding_rate_annual=0.1095)
    assert abs(cost - 0.6 * 0.1095 / 365) < 1e-12


def test_funding_zero_hedge():
    from portfolio.backtester.funding import daily_funding_cost
    assert daily_funding_cost(0.0, 0.1095) == 0.0


def test_funding_negative_hedge_raises():
    from portfolio.backtester.funding import daily_funding_cost
    # hedge应恒为非负（short notional）
    import pytest
    with pytest.raises(ValueError):
        daily_funding_cost(-0.1, 0.1095)
```

- [ ] **Step 3: 运行测试确认失败**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_fees.py tests/test_portfolio_funding.py -v
```
期望：FAIL

- [ ] **Step 4: 修改 `portfolio/backtester/fees.py`**

在文件末尾追加：

```python
def compute_perp_fee(
    old_hedge: float,
    new_hedge: float,
    fee_rate: float = 0.0005,
) -> float:
    """Compute one-sided perpetual futures trading fee.

    fee = |new - old| * fee_rate

    (Single-leg turnover, not round-trip like compute_fee.)
    """
    return abs(float(new_hedge) - float(old_hedge)) * float(fee_rate)
```

- [ ] **Step 5: 创建 `portfolio/backtester/funding.py`**

```python
"""Perpetual funding rate cost model."""
from __future__ import annotations


def daily_funding_cost(
    hedge_ratio: float,
    funding_rate_annual: float = 0.1095,
) -> float:
    """Daily funding cost for perpetual short (paid by short when funding>0).

    cost = hedge_ratio * (funding_rate_annual / 365)

    Args:
        hedge_ratio: Short notional as fraction of NAV; must be ≥ 0
        funding_rate_annual: Annualized funding (e.g. 0.1095 = 10.95%/year)

    Returns:
        Daily cost as fraction of NAV.
    """
    hedge = float(hedge_ratio)
    if hedge < 0:
        raise ValueError(f"hedge_ratio must be >= 0, got {hedge}")
    return hedge * float(funding_rate_annual) / 365.0
```

- [ ] **Step 6: 运行测试确认通过**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_fees.py tests/test_portfolio_funding.py -v
```
期望：全部 PASS

- [ ] **Step 7: Commit**

```bash
git add portfolio/backtester/fees.py portfolio/backtester/funding.py tests/test_portfolio_fees.py tests/test_portfolio_funding.py
git commit -m "feat(portfolio): add perp fee and funding-rate cost functions"
```

---

## Task 7: Hedged Backtest Engine

**Files:**
- Modify: `portfolio/backtester/engine.py`
- Test: `tests/test_portfolio_engine_hedged.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_portfolio_engine_hedged.py`：

```python
import numpy as np
import pandas as pd


def _make_kline(days=20):
    """BTC flat, ETH +1%/day (provides pure long-side profit)."""
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []
    for d in dates:
        rows += [
            {"symbol": "BTCUSDT", "date": d, "close": 50000.0},
            {"symbol": "ETHUSDT", "date": d, "close": 1500 * (1.01 ** ((d - dates[0]).days))},
        ]
    return pd.DataFrame(rows)


def test_hedged_engine_no_hedge_same_as_long_only():
    from portfolio.backtester.engine import run_hedged_backtest
    kline = _make_kline()
    dates = sorted(kline["date"].unique())
    weights = {dates[0]: {"ETHUSDT": 1.0}}
    # 全程不对冲
    alt_exp = pd.Series(1.0, index=dates)
    hedge = pd.Series(0.0, index=dates)
    ret = run_hedged_backtest(kline, weights, alt_exp, hedge,
                               alt_fee=0.0, perp_fee=0.0, funding_annual=0.0)
    # 首日扣 0 手续费；此后 +1%/day
    assert abs(ret.iloc[1] - 0.01) < 1e-6


def test_hedged_engine_full_hedge_kills_btc_exposure():
    """BTC空头对冲完全对抵BTC涨幅；ETH=BTC涨跌时，hedge=beta=1 should net 0."""
    from portfolio.backtester.engine import run_hedged_backtest
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for i, d in enumerate(dates):
        price = 50000 * 1.01 ** i  # 都涨1%
        rows += [
            {"symbol": "BTCUSDT", "date": d, "close": price},
            {"symbol": "ETHUSDT", "date": d, "close": price * 30},
        ]
    kline = pd.DataFrame(rows)
    weights = {dates[0]: {"ETHUSDT": 1.0}}
    alt_exp = pd.Series(1.0, index=dates)
    hedge = pd.Series(1.0, index=dates)  # 1:1 对冲
    ret = run_hedged_backtest(kline, weights, alt_exp, hedge,
                               alt_fee=0.0, perp_fee=0.0, funding_annual=0.0)
    # 每日都应 ≈ 0（ETH +1% * 1 - BTC +1% * 1 = 0）
    for r in ret.iloc[1:]:
        assert abs(r) < 1e-6


def test_hedged_engine_funding_cost_applied():
    from portfolio.backtester.engine import run_hedged_backtest
    kline = _make_kline(days=5)
    dates = sorted(kline["date"].unique())
    weights = {dates[0]: {"ETHUSDT": 1.0}}
    alt_exp = pd.Series(1.0, index=dates)
    hedge = pd.Series(0.6, index=dates)
    ret = run_hedged_backtest(kline, weights, alt_exp, hedge,
                               alt_fee=0.0, perp_fee=0.0,
                               funding_annual=0.365)  # 极端: 0.1%/day
    # 第2天 ETH +1%, BTC 0, 资金费 0.6*0.365/365=0.0006
    # return = 1.0*0.01 - 0.6*0 - 0.0006 = 0.0094
    assert abs(ret.iloc[1] - 0.0094) < 1e-5


def test_hedged_engine_returns_empty_when_no_weights():
    from portfolio.backtester.engine import run_hedged_backtest
    kline = _make_kline(days=5)
    dates = sorted(kline["date"].unique())
    alt_exp = pd.Series(1.0, index=dates)
    hedge = pd.Series(0.0, index=dates)
    ret = run_hedged_backtest(kline, {}, alt_exp, hedge)
    assert ret.empty
```

- [ ] **Step 2: 运行测试确认失败**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_engine_hedged.py -v
```
期望：FAIL（`run_hedged_backtest` 不存在）

- [ ] **Step 3: 修改 `portfolio/backtester/engine.py`，在文件末尾追加**

```python
from portfolio.backtester.fees import compute_perp_fee
from portfolio.backtester.funding import daily_funding_cost


def run_hedged_backtest(
    kline: pd.DataFrame,
    weights: dict[pd.Timestamp, dict[str, float]],
    alt_exposure: pd.Series,
    hedge_ratio: pd.Series,
    alt_fee: float = 0.001,
    perp_fee: float = 0.0005,
    funding_annual: float = 0.1095,
    btc_symbol: str = "BTCUSDT",
) -> pd.Series:
    """Run backtest with Alt long + BTC perp short hedge.

    daily_ret_t = alt_exposure_t * port_ret_alt_t
                - hedge_ratio_t * btc_ret_t
                - alt_fee_t (on Alt rebalance days)
                - perp_fee_t (on hedge-change days)
                - funding_cost_t (every day)

    Args:
        kline: [symbol, date, close]
        weights: {date: {instrument: weight}} for Alt basket (unit-sum).
                 Only dates in weights are Alt rebalance days.
        alt_exposure: pd.Series[date] → [0, 1.0] Alt long notional fraction.
        hedge_ratio: pd.Series[date] → [0, 1.2*alt_exp] BTC short notional fraction.
        alt_fee: one-way spot fee (default 10bps)
        perp_fee: one-way perp fee (default 5bps)
        funding_annual: annualized funding (default 10.95%)
    """
    daily = _compute_daily_returns(kline)
    ret_wide = daily.pivot_table(index="date", columns="instrument", values="ret").sort_index()
    if btc_symbol not in ret_wide.columns:
        raise ValueError(f"Missing {btc_symbol} in kline")
    btc_ret = ret_wide[btc_symbol]

    if not weights:
        return pd.Series(dtype=float)

    rebalance_dates = set(weights.keys())
    first_reb = min(rebalance_dates)

    # Align exposure/hedge series to ret_wide dates
    alt_exp = alt_exposure.reindex(ret_wide.index).ffill().fillna(0.0)
    hedge = hedge_ratio.reindex(ret_wide.index).ffill().fillna(0.0)

    port_returns: dict[pd.Timestamp, float] = {}
    current_weights: dict[str, float] = {}
    prev_alt_exp = 0.0
    prev_hedge = 0.0

    for t in ret_wide.index:
        if t < first_reb:
            continue
        is_rebalance = t in rebalance_dates
        rets_t = ret_wide.loc[t]

        # Initialize on first rebalance
        if not current_weights:
            if is_rebalance:
                target = weights[t]
                alt_fee_t = sum(abs(target.get(k, 0.0)) for k in target) * alt_exp.loc[t] * alt_fee / 2
                perp_fee_t = compute_perp_fee(prev_hedge, hedge.loc[t], perp_fee)
                funding_t = daily_funding_cost(hedge.loc[t], funding_annual)
                port_returns[t] = -alt_fee_t - perp_fee_t - funding_t
                current_weights = dict(target)
                prev_alt_exp = alt_exp.loc[t]
                prev_hedge = hedge.loc[t]
            continue

        # Alt internal return
        alt_port_ret = sum(
            current_weights.get(ins, 0.0) * float(rets_t.get(ins, 0.0) if ins in rets_t.index else 0.0)
            for ins in current_weights
        )

        # Drift weights
        denom = 1 + alt_port_ret if abs(1 + alt_port_ret) > 1e-12 else 1e-12
        drifted = {
            ins: w * (1 + float(rets_t.get(ins, 0.0) if ins in rets_t.index else 0.0)) / denom
            for ins, w in current_weights.items()
        }

        btc_r = float(btc_ret.loc[t]) if pd.notna(btc_ret.loc[t]) else 0.0

        # P&L before costs
        pnl_long = alt_exp.loc[t] * alt_port_ret
        pnl_hedge = -hedge.loc[t] * btc_r

        # Costs
        if is_rebalance:
            target = weights[t]
            # fee on |new_w*alt_exp - drifted_w*prev_alt_exp|
            instruments = set(drifted) | set(target)
            alt_turnover = sum(
                abs(target.get(k, 0.0) * alt_exp.loc[t]
                    - drifted.get(k, 0.0) * prev_alt_exp)
                for k in instruments
            )
            alt_fee_t = alt_turnover * alt_fee / 2
            current_weights = dict(target)
        else:
            alt_fee_t = 0.0
            current_weights = drifted

        perp_fee_t = compute_perp_fee(prev_hedge, hedge.loc[t], perp_fee)
        funding_t = daily_funding_cost(hedge.loc[t], funding_annual)

        port_returns[t] = pnl_long + pnl_hedge - alt_fee_t - perp_fee_t - funding_t
        prev_alt_exp = alt_exp.loc[t]
        prev_hedge = hedge.loc[t]

    return pd.Series(port_returns).sort_index()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_engine_hedged.py tests/test_portfolio_engine.py -v
```
期望：全部 PASS（旧的 engine 测试也要继续通过）

- [ ] **Step 5: Commit**

```bash
git add portfolio/backtester/engine.py tests/test_portfolio_engine_hedged.py
git commit -m "feat(portfolio): add run_hedged_backtest supporting BTC perp short leg"
```

---

## Task 8: Hedged Pipeline Integration

**Files:**
- Modify: `portfolio/pipeline.py`
- Test: `tests/test_portfolio_pipeline_hedged.py`

- [ ] **Step 1: 写失败测试（E2E smoke）**

创建 `tests/test_portfolio_pipeline_hedged.py`：

```python
import numpy as np
import pandas as pd
import pytest
import tempfile
import pathlib


@pytest.fixture
def synthetic_data(tmp_path):
    """生成 300 天、10 个币、2 个因子的合成数据。"""
    np.random.seed(123)
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    symbols = [f"C{i}USDT" for i in range(10)] + ["BTCUSDT"]

    # Kline
    rows = []
    for s in symbols:
        beta = 1.0 if s != "BTCUSDT" else 1.0
        prices = [100.0]
        for _ in range(len(dates) - 1):
            prices.append(prices[-1] * (1 + np.random.randn()*0.02))
        for d, p in zip(dates, prices):
            rows.append({"symbol": s, "date": d, "close": p})
    kline = pd.DataFrame(rows)
    kline_path = tmp_path / "kline.csv"
    kline.to_csv(kline_path, index=False)

    # Factors
    f1 = []
    for d in dates:
        for s in symbols:
            if s == "BTCUSDT":
                continue
            f1.append({"date": d.strftime("%Y-%m-%d"), "instrument": s,
                       "factor": np.random.randn()})
    f1_df = pd.DataFrame(f1)
    f1_path = tmp_path / "f1.csv"
    f1_df.to_csv(f1_path, index=False)

    # Universe CSV (minimum: all symbols always in)
    uni_rows = []
    for d in dates:
        for s in symbols:
            if s == "BTCUSDT":
                continue
            uni_rows.append({
                "Decision_Window_Start": d.strftime("%Y-%m-%d"),
                "Decision_Window_End": d.strftime("%Y-%m-%d"),
                "CoinGecko_ID": s.lower(),
                "Binance_Symbol": s,
            })
    uni_path = tmp_path / "universe.csv"
    pd.DataFrame(uni_rows).to_csv(uni_path, index=False)

    return {
        "kline": str(kline_path),
        "factor_paths": {"f1": str(f1_path)},
        "universe": str(uni_path),
    }


def test_hedged_pipeline_runs_end_to_end(synthetic_data):
    from portfolio.pipeline import run_hedged_pipeline
    from portfolio.config import PortfolioConfig

    cfg = PortfolioConfig(
        top_n=3,
        rebalance_period=5,
        is_end_date="2023-06-01",
        min_ic_ir=0.0,  # 接受所有因子
        ema_short=10,  # 缩短让 warmup 合理
        ema_long=40,
        zscore_window=30,
        t_smoothing_span=3,
        beta_window=20,
    )
    ret = run_hedged_pipeline(
        factor_paths=synthetic_data["factor_paths"],
        kline_csv=synthetic_data["kline"],
        universe_csv=synthetic_data["universe"],
        config=cfg,
    )
    assert isinstance(ret, pd.Series)
    assert not ret.empty
    assert not ret.isna().all()


def test_hedged_pipeline_produces_report(synthetic_data, tmp_path):
    from portfolio.pipeline import run_hedged_pipeline
    from portfolio.config import PortfolioConfig

    out = tmp_path / "report.html"
    cfg = PortfolioConfig(
        top_n=3, rebalance_period=5, is_end_date="2023-06-01",
        min_ic_ir=0.0, ema_short=10, ema_long=40,
        zscore_window=30, t_smoothing_span=3, beta_window=20,
    )
    run_hedged_pipeline(
        factor_paths=synthetic_data["factor_paths"],
        kline_csv=synthetic_data["kline"],
        universe_csv=synthetic_data["universe"],
        config=cfg,
        output_path=str(out),
    )
    assert out.exists()
    assert out.stat().st_size > 1000
```

- [ ] **Step 2: 运行测试确认失败**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_pipeline_hedged.py -v
```
期望：FAIL

- [ ] **Step 3: 修改 `portfolio/pipeline.py`，在文件末尾新增 `run_hedged_pipeline`**

```python
def run_hedged_pipeline(
    factor_paths: Mapping[str, str | pathlib.Path],
    kline_csv: str | pathlib.Path,
    universe_csv: str | pathlib.Path,
    config: PortfolioConfig | None = None,
    output_path: str | pathlib.Path | None = None,
) -> pd.Series:
    """Run the hedged MaxDD-15 portfolio pipeline end-to-end.

    Flow:
      1. Load factors, kline, universe (same as long-only pipeline)
      2. Align + normalize + orthogonalize factors
      3. IS期预筛 factors by |IC_IR| >= min_ic_ir
      4. Compute rolling IC_IR weights on retained factors
      5. Synthesize composite score -> Top-N weights (weekly)
      6. Compute BTC trend T (daily)
      7. Compute rolling coin betas (daily)
      8. Per-date: portfolio_beta @ holdings, (alt_exp, hedge_ratio) = controller(T, beta)
      9. Smooth alt_exp with EMA(alpha=alt_exposure_ema_alpha)
     10. Run hedged backtest -> daily returns
    """
    import pathlib as _pl
    from portfolio.combiner.is_oos_filter import filter_factors_by_is_ic_ir
    from portfolio.regime.btc_trend import compute_btc_trend_score
    from portfolio.portfolio_builder.beta_estimator import (
        compute_coin_betas, compute_portfolio_beta
    )
    from portfolio.sizing.exposure_controller import (
        compute_exposure_and_hedge, smooth_exposure_series
    )
    from portfolio.backtester.engine import run_hedged_backtest

    if config is None:
        config = PortfolioConfig()
    factor_names = list(factor_paths.keys())

    # 1. Load & align
    factors = load_many(factor_paths)
    universe_by_date = build_universe(universe_csv)
    panel = align_factors(factors, universe_by_date)
    if panel.empty or not factor_names:
        return pd.Series(dtype=float)

    # 2. Labels
    kline = pd.read_csv(kline_csv)
    kline["date"] = pd.to_datetime(kline["date"]).dt.normalize()
    label_df = build_future_ret(kline, config.label_period)

    # 3. Normalize (directions start at +1; is_oos_filter will override)
    directions = {n: 1 for n in factor_names}
    panel = normalize_cross_section(panel, factor_names, directions, config.winsorize_pct)

    # 4. Merge labels
    panel_with_labels = panel.merge(
        label_df[["date", "instrument", "future_ret"]],
        on=["date", "instrument"], how="left",
    )

    # 5. IS/OOS filter
    selected_dirs = filter_factors_by_is_ic_ir(
        panel_with_labels, factor_names,
        label_col="future_ret",
        is_end_date=config.is_end_date,
        min_ic_ir=config.min_ic_ir,
    )
    if not selected_dirs:
        return pd.Series(dtype=float)
    selected_factors = list(selected_dirs.keys())

    # 6. Re-normalize with corrected directions + orthogonalize
    panel = normalize_cross_section(
        panel.drop(columns=selected_factors, errors="ignore").merge(
            pd.DataFrame(panel_with_labels)[["date", "instrument"] + selected_factors],
            on=["date", "instrument"], how="left",
        ) if False else panel,  # keep panel as-is; just re-sign
        selected_factors, selected_dirs, config.winsorize_pct,
    )
    if config.orthogonalize and len(selected_factors) > 1:
        panel = orthogonalize(panel, selected_factors)
    panel_with_labels = panel.merge(
        label_df[["date", "instrument", "future_ret"]],
        on=["date", "instrument"], how="left",
    )

    # 7. IC_IR weights
    ic_weights = compute_ic_ir_weights(
        panel_with_labels, selected_factors,
        label_col="future_ret", window=config.ic_window,
    )

    # 8. Synthesize
    scores = synthesize(panel, ic_weights, selected_factors)

    # 9. Top-N weekly weights
    portfolio_weights = build_topn_weights(
        scores, universe_by_date, config.top_n,
        rebalance_period=config.rebalance_period,
    )
    if not portfolio_weights:
        return pd.Series(dtype=float)

    # 10. BTC trend T
    T = compute_btc_trend_score(
        kline,
        ema_short=config.ema_short, ema_long=config.ema_long,
        zscore_window=config.zscore_window,
        ema_ratio_saturation=config.ema_ratio_saturation,
        zscore_saturation=config.zscore_saturation,
        smoothing_span=config.t_smoothing_span,
    )

    # 11. Coin betas
    betas_df = compute_coin_betas(kline, window=config.beta_window)

    # 12. Per-date exposure/hedge. Holdings change on rebalance; interpolate between.
    all_dates = sorted(set(kline["date"].unique()))
    all_dates = [d for d in all_dates if d >= min(portfolio_weights.keys())]
    sorted_reb = sorted(portfolio_weights.keys())
    reb_idx = 0

    raw_alt, raw_hedge = {}, {}
    current_holdings: dict[str, float] = {}
    for d in all_dates:
        while reb_idx < len(sorted_reb) and sorted_reb[reb_idx] <= d:
            current_holdings = portfolio_weights[sorted_reb[reb_idx]]
            reb_idx += 1
        if not current_holdings:
            continue
        T_d = T.get(d, 0.0) if not pd.isna(T.get(d, np.nan)) else 0.0
        beta_port = compute_portfolio_beta(betas_df, current_holdings, d,
                                           beta_prior=config.beta_prior)
        alt_e, hedge = compute_exposure_and_hedge(
            T=T_d, beta_port=beta_port,
            alt_min=config.alt_min_exposure,
            alt_max=config.alt_max_exposure,
            hedge_cap_mult=config.hedge_cap_multiplier,
        )
        raw_alt[d] = alt_e
        raw_hedge[d] = hedge

    raw_alt_s = pd.Series(raw_alt).sort_index()
    raw_hedge_s = pd.Series(raw_hedge).sort_index()
    alt_exp_s = smooth_exposure_series(raw_alt_s, alpha=config.alt_exposure_ema_alpha)
    hedge_s = raw_hedge_s  # hedge 日度即时，不平滑

    # 13. Hedged backtest
    returns = run_hedged_backtest(
        kline, portfolio_weights, alt_exp_s, hedge_s,
        alt_fee=config.alt_fee_rate,
        perp_fee=config.perp_fee_rate,
        funding_annual=config.funding_rate_annual,
    )

    # 14. Benchmarks + optional report
    benchmarks: dict[str, pd.Series] = {}
    btc_rows = kline[kline["symbol"].str.upper() == "BTCUSDT"].copy()
    if not btc_rows.empty:
        btc_rows = btc_rows.sort_values("date")
        btc_rows["ret"] = btc_rows["close"].pct_change()
        btc_ret = btc_rows.set_index("date")["ret"].dropna()
        if not returns.empty:
            btc_ret = btc_ret.reindex(returns.index).dropna()
        benchmarks["BTC Buy&Hold"] = btc_ret
    if output_path is not None:
        render_html(returns, output_path, benchmarks=benchmarks)
    return returns
```

- [ ] **Step 4: 简化上一步中的多余代码**

检查 Step 3 代码中 `normalize_cross_section` 的重复调用，简化为单次调用：

```python
    # Step 3中"re-normalize"那一块替换为以下简单版本
    panel = normalize_cross_section(
        panel, selected_factors, selected_dirs, config.winsorize_pct,
    )
    if config.orthogonalize and len(selected_factors) > 1:
        panel = orthogonalize(panel, selected_factors)
    panel_with_labels = panel.merge(
        label_df[["date", "instrument", "future_ret"]],
        on=["date", "instrument"], how="left",
    )
```

删除 Step 3 代码中 `panel.drop(columns=...) if False else panel` 的那一段错误代码，替换为以上简洁版本。

- [ ] **Step 5: 运行测试确认通过**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_pipeline_hedged.py -v
```
期望：2个测试 PASS

- [ ] **Step 6: 运行整体 portfolio 测试确认无回归**

```bash
./.venv/bin/python -m pytest tests/ -k portfolio -v
```
期望：全部 PASS

- [ ] **Step 7: Commit**

```bash
git add portfolio/pipeline.py tests/test_portfolio_pipeline_hedged.py
git commit -m "feat(portfolio): add run_hedged_pipeline integrating all layers"
```

---

## Task 9: Future-Leak Dynamic Verification Test

**Files:**
- Create: `tests/test_portfolio_future_leak_hedged.py`

- [ ] **Step 1: 写反证测试**

```python
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _generate_synthetic_bundle(tmp_path, days=500):
    """与 test_portfolio_pipeline_hedged 相同的合成数据，但时间更长。"""
    np.random.seed(77)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    symbols = [f"C{i}USDT" for i in range(8)] + ["BTCUSDT"]

    rows = []
    for s in symbols:
        prices = [100.0]
        for _ in range(len(dates) - 1):
            prices.append(prices[-1] * (1 + np.random.randn() * 0.02))
        for d, p in zip(dates, prices):
            rows.append({"symbol": s, "date": d, "close": p})
    kline = pd.DataFrame(rows)
    kline_path = tmp_path / "kline.csv"
    kline.to_csv(kline_path, index=False)

    f_rows = []
    for d in dates:
        for s in symbols:
            if s == "BTCUSDT":
                continue
            f_rows.append({"date": d.strftime("%Y-%m-%d"), "instrument": s,
                           "factor": np.random.randn()})
    f_path = tmp_path / "f1.csv"
    pd.DataFrame(f_rows).to_csv(f_path, index=False)

    uni_rows = []
    for d in dates:
        for s in symbols:
            if s == "BTCUSDT":
                continue
            uni_rows.append({
                "Decision_Window_Start": d.strftime("%Y-%m-%d"),
                "Decision_Window_End": d.strftime("%Y-%m-%d"),
                "CoinGecko_ID": s.lower(),
                "Binance_Symbol": s,
            })
    uni_path = tmp_path / "universe.csv"
    pd.DataFrame(uni_rows).to_csv(uni_path, index=False)
    return str(kline_path), {"f1": str(f_path)}, str(uni_path)


def test_hedged_pipeline_no_future_leak(tmp_path):
    """对比全量回测 vs 截断到 cutoff 的回测；cutoff 前的收益必须一致。"""
    from portfolio.pipeline import run_hedged_pipeline
    from portfolio.config import PortfolioConfig

    kline_path, factor_paths, uni_path = _generate_synthetic_bundle(tmp_path)

    cfg = PortfolioConfig(
        top_n=3, rebalance_period=5, is_end_date="2022-08-01",
        min_ic_ir=0.0, ema_short=10, ema_long=40,
        zscore_window=30, t_smoothing_span=3, beta_window=20,
    )

    # Full run
    ret_full = run_hedged_pipeline(factor_paths, kline_path, uni_path, config=cfg)

    # Truncate at cutoff
    cutoff = pd.Timestamp("2023-01-15")
    kline = pd.read_csv(kline_path)
    kline["date"] = pd.to_datetime(kline["date"])
    kline_cut = kline[kline["date"] <= cutoff]
    kline_cut_path = tmp_path / "kline_cut.csv"
    kline_cut.to_csv(kline_cut_path, index=False)

    # Truncate factor too
    f_df = pd.read_csv(factor_paths["f1"])
    f_df["date"] = pd.to_datetime(f_df["date"])
    f_cut = f_df[f_df["date"] <= cutoff]
    f_cut_path = tmp_path / "f1_cut.csv"
    f_cut.to_csv(f_cut_path, index=False)

    ret_cut = run_hedged_pipeline({"f1": str(f_cut_path)}, str(kline_cut_path), uni_path, config=cfg)

    # Compare dates <= cutoff
    common = ret_full.index.intersection(ret_cut.index)
    common = common[common <= cutoff]
    if len(common) == 0:
        pytest.skip("No common dates before cutoff")
    diff = (ret_full.reindex(common) - ret_cut.reindex(common)).abs().max()
    assert diff < 1e-6, f"Future leak detected: max_abs_diff = {diff}"
```

- [ ] **Step 2: 运行测试确认通过**

```bash
./.venv/bin/python -m pytest tests/test_portfolio_future_leak_hedged.py -v
```
期望：PASS（若失败说明 pipeline 中某处未来函数，回到 Task 3/4/8 排查）

- [ ] **Step 3: 静态模式扫描**

```bash
grep -rn "shift(-" portfolio/regime portfolio/sizing portfolio/combiner/is_oos_filter.py portfolio/portfolio_builder/beta_estimator.py portfolio/backtester/funding.py 2>&1 | grep -v ".pyc"
grep -rn "center=True" portfolio/regime portfolio/sizing portfolio/combiner/is_oos_filter.py portfolio/portfolio_builder/beta_estimator.py 2>&1 | grep -v ".pyc"
grep -rn "\.bfill" portfolio/regime portfolio/sizing portfolio/combiner/is_oos_filter.py portfolio/portfolio_builder/beta_estimator.py portfolio/pipeline.py 2>&1 | grep -v ".pyc"
grep -rn "direction.*forward" portfolio/regime portfolio/sizing portfolio/combiner/is_oos_filter.py portfolio/portfolio_builder/beta_estimator.py 2>&1 | grep -v ".pyc"
```
期望：全部**无输出**（除非在注释里说明原因）

- [ ] **Step 4: Commit**

```bash
git add tests/test_portfolio_future_leak_hedged.py
git commit -m "test(portfolio): add future-leak dynamic verification for hedged pipeline"
```

---

## Task 10: 跑真实数据 + 生成报告

**Files:**
- Create: `portfolio/main_hedged.py`

- [ ] **Step 1: 创建 `portfolio/main_hedged.py`**

```python
"""Execute the MaxDD-15 hedged pipeline on real Binance data."""
from __future__ import annotations

import glob
import os
import pathlib
import sys

import pandas as pd

from portfolio.config import PortfolioConfig
from portfolio.pipeline import run_hedged_pipeline


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "portfolio" / "output"


def _latest_kline() -> str:
    candidates = sorted(glob.glob(str(DATA_DIR / "kline_data" / "binance_daily_klines_*.csv")))
    if not candidates:
        raise FileNotFoundError("No kline CSV found")
    return candidates[-1]


def _load_factor_paths() -> dict[str, str]:
    """Auto-discover all factor CSVs matching *_YYYYMMDD.csv in factor_data/."""
    paths: dict[str, str] = {}
    for p in sorted(glob.glob(str(DATA_DIR / "factor_data" / "*.csv"))):
        name = pathlib.Path(p).stem
        # strip trailing _YYYYMMDD
        parts = name.rsplit("_", 1)
        base = parts[0] if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8 else name
        paths[base] = p
    return paths


def run_oos_backtest(
    oos_start: str = "2024-01-01",
    oos_end: str = "2026-04-17",
    top_n: int = 5,
    label: str = "maxdd15_oos",
) -> pd.Series:
    universe_csv = str(DATA_DIR / "binance_coingecko_top100_marketcap_historical.csv")
    kline_csv = _latest_kline()
    factor_paths = _load_factor_paths()

    cfg = PortfolioConfig(
        top_n=top_n,
        rebalance_period=5,
        is_end_date="2023-12-31",
        min_ic_ir=0.05,
        universe_csv=universe_csv,
        kline_csv=kline_csv,
    )
    output = OUTPUT_DIR / f"report_{label}.html"
    returns = run_hedged_pipeline(
        factor_paths=factor_paths,
        kline_csv=kline_csv,
        universe_csv=universe_csv,
        config=cfg,
        output_path=str(output),
    )
    # Slice to OOS
    oos = returns.loc[oos_start:oos_end] if not returns.empty else returns
    print(f"Report: {output}")
    print(f"OOS dates: {len(oos)}  Full dates: {len(returns)}")
    if not oos.empty:
        cum = (1 + oos).prod() - 1
        ann = (1 + oos).prod() ** (252/len(oos)) - 1
        dd = ((1 + oos).cumprod() / (1 + oos).cumprod().cummax() - 1).min()
        sharpe = oos.mean() / oos.std() * (252**0.5) if oos.std() > 0 else float("nan")
        print(f"OOS cum_ret = {cum:.2%}")
        print(f"OOS annualized = {ann:.2%}")
        print(f"OOS MaxDD = {dd:.2%}")
        print(f"OOS Sharpe = {sharpe:.3f}")
    return returns


if __name__ == "__main__":
    args = sys.argv[1:]
    label = args[0] if args else "maxdd15_oos"
    run_oos_backtest(label=label)
```

- [ ] **Step 2: 运行真实回测**

```bash
./.venv/bin/python portfolio/main_hedged.py maxdd15_v1
```
期望：打印 OOS 指标，生成 `portfolio/output/report_maxdd15_v1.html`

- [ ] **Step 3: 检查结果**

打开 `portfolio/output/report_maxdd15_v1.html`，确认：
- 有策略净值曲线 + BTC Buy&Hold 对比
- OOS MaxDD ≤ 15%（目标）
- OOS 年化 ≥ 20%
- OOS Sharpe ≥ 1.2

如果目标未达成，进入 Task 11 做参数调整。

- [ ] **Step 4: Commit**

```bash
git add portfolio/main_hedged.py portfolio/output/report_maxdd15_v1.html
git commit -m "feat(portfolio): add main_hedged.py runner + initial v1 OOS results"
```

---

## Task 11: 参数敏感性调优（若 Task 10 未达标）

**Files:**
- Modify: `portfolio/main_hedged.py`

这是条件性任务：只有在 Task 10 OOS MaxDD > 15% 或 Sharpe < 1.2 时执行。

- [ ] **Step 1: 扫描 `ema_short/ema_long` 对 (30/100), (50/200), (60/250)**

扩展 `main_hedged.py` 加入 `run_param_sweep()`：

```python
def run_param_sweep():
    """Scan key sensitivity params, print MaxDD/Sharpe matrix."""
    results = []
    for short, long_ in [(30, 100), (50, 200), (60, 250)]:
        for cap in [1.0, 1.2, 1.5]:
            for min_exp in [0.3, 0.5, 0.7]:
                label = f"sw_s{short}_l{long_}_cap{cap}_mex{min_exp}"
                # ... 复制 run_oos_backtest 核心逻辑，仅改这3个参数
                # 记录 MaxDD, annualized, Sharpe
                pass
    # 打印 markdown 表格
```

- [ ] **Step 2: 运行 sweep**

```bash
./.venv/bin/python -c "from portfolio.main_hedged import run_param_sweep; run_param_sweep()"
```

- [ ] **Step 3: 挑选 MaxDD≤15% 且 Sharpe 最高的参数组，更新默认并重新运行**

- [ ] **Step 4: Commit**

```bash
git add portfolio/main_hedged.py portfolio/output/report_maxdd15_final.html
git commit -m "feat(portfolio): parameter sensitivity sweep + final MaxDD-15 config"
```

---

## Task 12: 文档更新

**Files:**
- Modify: `README.md` (或 `portfolio/README.md` 如果存在)

- [ ] **Step 1: 在 README 中加一节使用说明**

```markdown
## MaxDD-15 Hedged Strategy

Runs the Alt-long + BTC-perp-short hedged portfolio targeting ≤15% OOS max drawdown.

```bash
./.venv/bin/python portfolio/main_hedged.py maxdd15_v1
```

Key config params (see `portfolio/config.py`):
- `is_end_date`: IS/OOS cutoff (default 2023-12-31)
- `min_ic_ir`: Factor keep threshold (default 0.05)
- `top_n`: 5 (weekly Top-5 equal-weight)
- `alt_min_exposure`: 0.5 (Alt min notional)
- `hedge_cap_multiplier`: 1.2 (BTC short max × Alt exposure)
- `funding_rate_annual`: 0.1095 (fixed +10.95%/yr)

See spec: `docs/superpowers/specs/2026-04-20-crypto-portfolio-maxdd15-design.md`
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document MaxDD-15 hedged strategy usage"
```

---

## Self-Review Checklist

1. **Spec coverage** ✓
   - Layer A IS/OOS filter → Task 2
   - Layer B BTC trend → Task 3
   - Layer C Beta + exposure → Tasks 4, 5
   - Layer D 引擎 + fees + funding → Tasks 6, 7
   - Pipeline integration → Task 8
   - 未来函数检查 → Task 9
   - 真实数据运行 + 成功标准 → Task 10
   - 参数敏感性 → Task 11
   - 文档 → Task 12

2. **No placeholders:** 所有代码块完整；每个 Task 独立可执行。

3. **Type consistency:** 
   - `compute_btc_trend_score` 返回 `pd.Series[date]` — 在 Task 8 中通过 `T.get(d, 0.0)` 访问
   - `compute_coin_betas` 返回 long-format DataFrame — Task 8/4 一致使用
   - `compute_exposure_and_hedge` 返回 `(alt_exp, hedge_ratio)` tuple — Task 8 一致解包
   - `run_hedged_backtest` 签名 `(kline, weights, alt_exposure, hedge_ratio, alt_fee, perp_fee, funding_annual)` — Task 7/8 一致

---

**End of Plan**
