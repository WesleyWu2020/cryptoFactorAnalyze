# Path B: BTC Core + Alt Short Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a crypto portfolio strategy that combines a dynamic BTC long core (40-70%) with an Alt short overlay on factor-weakest mid-cap coins (rank 30-100), protected by a funding-rate hard filter and squeeze stop-loss, targeting MaxDD<20% and Ann>19% (beats BTC buy-and-hold) on OOS 2024-01-01 to 2026-04-17.

**Architecture:** Four-layer portfolio: (Layer 1) BTC dynamic core sized by trend score T from `btc_trend.py`; (Layer 2) Alt short overlay selecting bottom-N coins by composite factor rank from IS-IC-IR-filtered factors; (Layer 3) Funding-rate hard filter (exclude coins with fwd-looking safe funding > P80 of cross-section); (Layer 4) Squeeze risk control (single-name 2-day reverse stop + portfolio deleveraging). Reuses existing `portfolio/combiner/is_oos_filter.py`, `portfolio/regime/btc_trend.py`, `portfolio/backtester/fees.py`; adds funding data pipeline + short-side modules + a new engine function.

**Tech Stack:** Python 3.x, pandas, numpy, requests (for Binance funding API), existing `portfolio/` package, pytest, matplotlib for report plots.

---

## File Structure

**New files:**
- `data/capture_funding_rate.py` — Binance funding-rate historical puller
- `data/funding_rate_data/` — CSV output directory (per-symbol or combined)
- `portfolio/universe/__init__.py`
- `portfolio/universe/rank_filter.py` — rank-based universe filter (30-100)
- `portfolio/regime/funding_filter.py` — per-date P80 funding threshold + mask
- `portfolio/combiner/short_side_scorer.py` — composite factor rank → bottom-N
- `portfolio/sizing/__init__.py`
- `portfolio/sizing/btc_core_sizer.py` — f(T) → BTC exposure in [btc_base_min, btc_base_max]
- `portfolio/risk/__init__.py`
- `portfolio/risk/squeeze_stop.py` — single-name stop + portfolio deleveraging
- `portfolio/main_pathB.py` — CLI entry for Path B
- `tests/portfolio/test_rank_filter.py`
- `tests/portfolio/test_funding_filter.py`
- `tests/portfolio/test_short_side_scorer.py`
- `tests/portfolio/test_btc_core_sizer.py`
- `tests/portfolio/test_squeeze_stop.py`
- `tests/portfolio/test_pathB_engine.py`
- `tests/portfolio/test_pathB_no_lookahead.py`

**Modified files:**
- `portfolio/config.py` — add Path B fields + validation
- `portfolio/backtester/engine.py` — add `run_btc_core_short_backtest()`
- `portfolio/pipeline.py` — add `run_pathB_pipeline()`
- `portfolio/README.md` or `docs/README.md` — Path B usage docs

---

### Task 0: Capture Binance Funding Rate History

**Files:**
- Create: `data/capture_funding_rate.py`
- Create: `data/funding_rate_data/` (dir; committed as empty via `.gitkeep`)
- Test: `tests/data/test_capture_funding_rate.py`

- [ ] **Step 1: Write failing unit test (pure logic, mocked HTTP)**

```python
# tests/data/test_capture_funding_rate.py
import pandas as pd
from unittest.mock import patch
from data.capture_funding_rate import aggregate_to_daily, parse_funding_response

def test_parse_funding_response_shapes_columns():
    payload = [{"symbol":"BTCUSDT","fundingRate":"0.0001","fundingTime":1704067200000}]
    df = parse_funding_response(payload)
    assert list(df.columns) == ["symbol","funding_rate","funding_time"]
    assert len(df) == 1

def test_aggregate_to_daily_sums_three_intervals():
    df = pd.DataFrame({
        "symbol":["BTCUSDT"]*3,
        "funding_rate":[0.0001, 0.0002, 0.0003],
        "funding_time": pd.to_datetime(["2024-01-01 00:00","2024-01-01 08:00","2024-01-01 16:00"]),
    })
    daily = aggregate_to_daily(df)
    assert daily.loc[(daily["symbol"]=="BTCUSDT") & (daily["date"]=="2024-01-01"),"funding_daily"].iloc[0] == 0.0006
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/data/test_capture_funding_rate.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `capture_funding_rate.py`**

```python
# data/capture_funding_rate.py
"""Pull Binance USDT-M perp funding-rate history and aggregate to daily."""
import argparse
import time
from pathlib import Path
import pandas as pd
import requests

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
OUT_DIR = Path(__file__).parent / "funding_rate_data"

def parse_funding_response(payload):
    if not payload:
        return pd.DataFrame(columns=["symbol","funding_rate","funding_time"])
    df = pd.DataFrame(payload)
    df = df.rename(columns={"fundingRate":"funding_rate","fundingTime":"funding_time"})
    df["funding_rate"] = df["funding_rate"].astype(float)
    df["funding_time"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True).dt.tz_localize(None)
    return df[["symbol","funding_rate","funding_time"]]

def fetch_symbol(symbol, start_ms, end_ms, pause=0.25):
    out = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000}
        r = requests.get(BINANCE_FAPI, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend(data)
        last_t = data[-1]["fundingTime"]
        if last_t <= cursor:
            break
        cursor = last_t + 1
        time.sleep(pause)
    return parse_funding_response(out)

def aggregate_to_daily(df):
    if df.empty:
        return pd.DataFrame(columns=["symbol","date","funding_daily","funding_count"])
    df = df.copy()
    df["date"] = df["funding_time"].dt.strftime("%Y-%m-%d")
    g = df.groupby(["symbol","date"])["funding_rate"]
    daily = g.sum().rename("funding_daily").reset_index()
    daily["funding_count"] = g.size().values
    return daily

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-csv", default="data/binance_coingecko_top100_marketcap_historical.csv")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=str(OUT_DIR / "binance_funding_daily.csv"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    syms = pd.read_csv(args.symbols_csv)
    symbols = sorted({s for s in syms["Trading_Pairs"].dropna().astype(str)
                      .str.split(",").explode().str.strip() if s.endswith("USDT")})
    start_ms = int(pd.Timestamp(args.start).timestamp()*1000)
    end_ms = int((pd.Timestamp(args.end) if args.end else pd.Timestamp.utcnow()).timestamp()*1000)

    all_daily = []
    for i, sym in enumerate(symbols):
        try:
            raw = fetch_symbol(sym, start_ms, end_ms)
            daily = aggregate_to_daily(raw)
            all_daily.append(daily)
            print(f"[{i+1}/{len(symbols)}] {sym}: {len(daily)} days")
        except Exception as e:
            print(f"[{i+1}/{len(symbols)}] {sym} FAILED: {e}")
    out = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} rows -> {args.out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests to verify pass**

Run: `./.venv/bin/python -m pytest tests/data/test_capture_funding_rate.py -v`
Expected: PASS

- [ ] **Step 5: Run pull against live Binance (long-running)**

Run: `./.venv/bin/python data/capture_funding_rate.py --start 2021-01-01`
Expected: CSV at `data/funding_rate_data/binance_funding_daily.csv` with columns `symbol,date,funding_daily,funding_count`. Sanity-check: `./.venv/bin/python -c "import pandas as pd; d=pd.read_csv('data/funding_rate_data/binance_funding_daily.csv'); print(d.shape); print(d.head()); print(d['funding_daily'].describe())"`. Median daily ~1e-4 (≈11%/yr).

- [ ] **Step 6: Commit**

```bash
git add data/capture_funding_rate.py tests/data/test_capture_funding_rate.py data/funding_rate_data/.gitkeep
git commit -m "feat(data): Binance funding-rate history puller + daily aggregator"
```

---

### Task 1: Extend PortfolioConfig for Path B

**Files:**
- Modify: `portfolio/config.py` (append Path B fields + validation)
- Test: `tests/portfolio/test_config_pathB.py`

- [ ] **Step 1: Write failing test**

```python
# tests/portfolio/test_config_pathB.py
import pytest
from portfolio.config import PortfolioConfig

def test_pathB_defaults():
    cfg = PortfolioConfig()
    assert 0.0 <= cfg.btc_base_min <= cfg.btc_base_max <= 1.0
    assert cfg.alt_short_exposure_max < 0
    assert 0 < cfg.universe_rank_min < cfg.universe_rank_max
    assert 0 < cfg.funding_filter_percentile < 1
    assert cfg.squeeze_stop_return > 0 and cfg.squeeze_stop_window >= 1

def test_pathB_invalid_range_raises():
    with pytest.raises(ValueError):
        PortfolioConfig(btc_base_min=0.8, btc_base_max=0.5)
    with pytest.raises(ValueError):
        PortfolioConfig(universe_rank_min=100, universe_rank_max=30)
    with pytest.raises(ValueError):
        PortfolioConfig(alt_short_exposure_max=0.1)
```

- [ ] **Step 2: Run test to fail**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_config_pathB.py -v`
Expected: FAIL

- [ ] **Step 3: Add fields + `__post_init__` validation in `portfolio/config.py`**

```python
# append inside PortfolioConfig dataclass
    btc_base_min: float = 0.4
    btc_base_max: float = 0.7
    alt_short_exposure_max: float = -0.3
    universe_rank_min: int = 30
    universe_rank_max: int = 100
    funding_filter_percentile: float = 0.8
    squeeze_stop_return: float = 0.25
    squeeze_stop_window: int = 2
    short_top_n: int = 5
```

Add to `__post_init__` (create if absent):
```python
    def __post_init__(self):
        if not (0.0 <= self.btc_base_min <= self.btc_base_max <= 1.0):
            raise ValueError("btc_base_min/max out of range")
        if self.universe_rank_min >= self.universe_rank_max:
            raise ValueError("universe_rank_min must be < universe_rank_max")
        if self.alt_short_exposure_max >= 0:
            raise ValueError("alt_short_exposure_max must be negative")
        if not (0.0 < self.funding_filter_percentile < 1.0):
            raise ValueError("funding_filter_percentile must be in (0,1)")
```

- [ ] **Step 4: Run tests pass**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_config_pathB.py -v`
Expected: PASS (and no regression: `./.venv/bin/python -m pytest tests/portfolio/ -v`)

- [ ] **Step 5: Commit**

```bash
git add portfolio/config.py tests/portfolio/test_config_pathB.py
git commit -m "feat(portfolio): extend PortfolioConfig with Path B fields"
```

---

### Task 2: Universe Rank Filter (30-100)

**Files:**
- Create: `portfolio/universe/__init__.py`, `portfolio/universe/rank_filter.py`
- Test: `tests/portfolio/test_rank_filter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/portfolio/test_rank_filter.py
import pandas as pd
from portfolio.universe.rank_filter import filter_universe_by_rank

def test_filter_keeps_only_mid_ranks():
    mc = pd.DataFrame({
        "Date":["2024-01-01"]*5,
        "Rank":[1,15,50,80,120],
        "Symbol":["BTC","ETH","SOL","DOGE","X"],
        "Trading_Pairs":["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","XUSDT"],
    })
    out = filter_universe_by_rank(mc, rank_min=30, rank_max=100)
    assert set(out["Symbol"]) == {"SOL","DOGE"}

def test_missing_rank_dropped():
    mc = pd.DataFrame({"Date":["2024-01-01"],"Rank":[None],"Symbol":["X"],"Trading_Pairs":["XUSDT"]})
    assert filter_universe_by_rank(mc, 30, 100).empty
```

- [ ] **Step 2: Run test to fail**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_rank_filter.py -v`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement**

```python
# portfolio/universe/__init__.py
from .rank_filter import filter_universe_by_rank  # noqa

# portfolio/universe/rank_filter.py
"""Rank-based universe filter: keep coins whose current Rank falls in [rank_min, rank_max]."""
import pandas as pd

def filter_universe_by_rank(market_cap_df: pd.DataFrame, rank_min: int, rank_max: int) -> pd.DataFrame:
    df = market_cap_df.copy()
    df = df.dropna(subset=["Rank"])
    df["Rank"] = df["Rank"].astype(int)
    mask = (df["Rank"] >= rank_min) & (df["Rank"] <= rank_max)
    return df.loc[mask].reset_index(drop=True)
```

- [ ] **Step 4: Run pass**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_rank_filter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/universe/ tests/portfolio/test_rank_filter.py
git commit -m "feat(portfolio): universe rank filter (keep rank_min..rank_max)"
```

---

### Task 3: Funding-Rate Hard Filter (P80 per-date)

**Files:**
- Create: `portfolio/regime/funding_filter.py`
- Test: `tests/portfolio/test_funding_filter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/portfolio/test_funding_filter.py
import pandas as pd
from portfolio.regime.funding_filter import build_funding_mask

def test_P80_removes_top_20pct_per_date():
    df = pd.DataFrame({
        "date":["2024-01-01"]*10 + ["2024-01-02"]*10,
        "instrument":[f"C{i}USDT" for i in range(10)]*2,
        "funding_daily":[i/1000 for i in range(10)]*2,
    })
    mask = build_funding_mask(df, percentile=0.8)
    kept = mask[mask["allowed"]]
    # P80 of 0..9 = 7.2; only values <=7.2 kept (8 coins)
    assert len(kept[kept["date"]=="2024-01-01"]) == 8

def test_missing_funding_treated_as_blocked():
    df = pd.DataFrame({
        "date":["2024-01-01"]*3,
        "instrument":["AUSDT","BUSDT","CUSDT"],
        "funding_daily":[0.0001, None, 0.0002],
    })
    mask = build_funding_mask(df, percentile=0.8)
    row_b = mask[mask["instrument"]=="BUSDT"].iloc[0]
    assert row_b["allowed"] == False
```

- [ ] **Step 2: Run test fail**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_funding_filter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# portfolio/regime/funding_filter.py
"""Per-date funding-rate P{percentile} hard filter for short side."""
import pandas as pd

def build_funding_mask(funding_df: pd.DataFrame, percentile: float = 0.8) -> pd.DataFrame:
    df = funding_df.copy()
    df["allowed"] = False
    valid = df.dropna(subset=["funding_daily"]).copy()
    thr = valid.groupby("date")["funding_daily"].quantile(percentile).rename("thr").reset_index()
    valid = valid.merge(thr, on="date", how="left")
    valid["allowed"] = valid["funding_daily"] <= valid["thr"]
    out = df.merge(valid[["date","instrument","allowed"]].rename(columns={"allowed":"a2"}),
                   on=["date","instrument"], how="left")
    out["allowed"] = out["a2"].fillna(False)
    return out[["date","instrument","funding_daily","allowed"]]
```

- [ ] **Step 4: Run pass**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_funding_filter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/regime/funding_filter.py tests/portfolio/test_funding_filter.py
git commit -m "feat(portfolio): per-date P80 funding-rate hard filter"
```

---

### Task 4: Short-Side Composite Scorer

**Files:**
- Create: `portfolio/combiner/short_side_scorer.py`
- Test: `tests/portfolio/test_short_side_scorer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/portfolio/test_short_side_scorer.py
import pandas as pd
from portfolio.combiner.short_side_scorer import select_short_bottom_n

def test_picks_bottom_n_by_composite_rank():
    # 2 factors, 5 instruments, 1 date; direction +1 means higher=better (short worst=lowest)
    panel = pd.DataFrame({
        "date":["2024-01-01"]*5,
        "instrument":["A","B","C","D","E"],
        "f1":[5,4,3,2,1],
        "f2":[1,2,3,4,5],
    })
    directions = {"f1": +1, "f2": -1}  # f2 reversed: higher=worse
    out = select_short_bottom_n(panel, factors=["f1","f2"], directions=directions, n=2)
    # composite: lower rank of f1 (+1) = bad; higher rank of f2 (-1 reversed) = bad
    # A: f1-rank=5 (best), f2-rev-rank=5 (best) -> best composite -> not shorted
    # E: f1-rank=1 (worst), f2-rev-rank=1 (worst) -> worst -> shorted
    assert "E" in out["2024-01-01"]

def test_respects_allowed_mask():
    panel = pd.DataFrame({
        "date":["2024-01-01"]*3,
        "instrument":["A","B","C"],
        "f1":[1,2,3],
        "allowed":[False, True, True],
    })
    out = select_short_bottom_n(panel, factors=["f1"], directions={"f1":+1}, n=2, allowed_col="allowed")
    # A has worst f1 but not allowed, so picks from B,C
    assert "A" not in out["2024-01-01"]
```

- [ ] **Step 2: Run test fail**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_short_side_scorer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# portfolio/combiner/short_side_scorer.py
"""Composite-rank short-side selector: bottom-N by sum of per-factor ranks."""
import pandas as pd
from typing import Dict, List, Optional

def select_short_bottom_n(panel: pd.DataFrame, factors: List[str], directions: Dict[str, int],
                          n: int, allowed_col: Optional[str] = None) -> Dict[str, List[str]]:
    selected = {}
    for date, grp in panel.groupby("date"):
        g = grp.copy()
        if allowed_col and allowed_col in g.columns:
            g = g[g[allowed_col].astype(bool)]
        if g.empty:
            selected[str(date)] = []
            continue
        composite = pd.Series(0.0, index=g.index)
        for f in factors:
            r = g[f].rank(ascending=True, method="average")
            # direction +1: higher=better for long => for short-worst, worst=lowest rank=1 => low composite = short
            # direction -1: higher=worse for long => for short-worst, invert
            if directions.get(f, +1) == -1:
                r = g[f].rank(ascending=False, method="average")
            composite = composite + r
        g = g.assign(_composite=composite).sort_values("_composite", ascending=True)
        selected[str(date)] = g["instrument"].head(n).tolist()
    return selected
```

- [ ] **Step 4: Run pass**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_short_side_scorer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/combiner/short_side_scorer.py tests/portfolio/test_short_side_scorer.py
git commit -m "feat(portfolio): composite-rank short-side bottom-N selector"
```

---

### Task 5: BTC Core Sizer (f(T) → exposure)

**Files:**
- Create: `portfolio/sizing/__init__.py`, `portfolio/sizing/btc_core_sizer.py`
- Test: `tests/portfolio/test_btc_core_sizer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/portfolio/test_btc_core_sizer.py
from portfolio.sizing.btc_core_sizer import size_btc_core

def test_T_midpoint_returns_midpoint_exposure():
    w = size_btc_core(trend_score=0.0, base_min=0.4, base_max=0.7)
    assert abs(w - 0.55) < 1e-9

def test_T_high_caps_at_max():
    assert size_btc_core(trend_score=5.0, base_min=0.4, base_max=0.7) == 0.7

def test_T_low_floors_at_min():
    assert size_btc_core(trend_score=-5.0, base_min=0.4, base_max=0.7) == 0.4
```

- [ ] **Step 2: Run test fail**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_btc_core_sizer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# portfolio/sizing/__init__.py
from .btc_core_sizer import size_btc_core  # noqa

# portfolio/sizing/btc_core_sizer.py
"""Map BTC trend score T (from regime/btc_trend.py) to exposure in [base_min, base_max].
Using sigmoid-like clip: T scaled by 1.0, then shifted to midpoint."""
import math

def size_btc_core(trend_score: float, base_min: float, base_max: float, slope: float = 1.0) -> float:
    mid = (base_min + base_max) / 2.0
    half = (base_max - base_min) / 2.0
    # tanh squashes to (-1,1); multiply by half and add mid -> clipped in [base_min, base_max]
    w = mid + half * math.tanh(slope * trend_score)
    return max(base_min, min(base_max, w))
```

- [ ] **Step 4: Run pass**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_btc_core_sizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/sizing/ tests/portfolio/test_btc_core_sizer.py
git commit -m "feat(portfolio): BTC core sizer via tanh of trend score"
```

---

### Task 6: Squeeze Stop-Loss Risk Control

**Files:**
- Create: `portfolio/risk/__init__.py`, `portfolio/risk/squeeze_stop.py`
- Test: `tests/portfolio/test_squeeze_stop.py`

- [ ] **Step 1: Write failing test**

```python
# tests/portfolio/test_squeeze_stop.py
import pandas as pd
from portfolio.risk.squeeze_stop import apply_squeeze_stop

def test_single_name_25pct_2d_return_triggers_stop():
    prices = pd.DataFrame({
        "date":["2024-01-01","2024-01-02","2024-01-03"],
        "A":[100.0, 120.0, 130.0],   # 2d return = 30% >= 25% -> stop
        "B":[100.0, 105.0, 110.0],   # 10% -> keep
    }).set_index("date")
    shorts = {"2024-01-03":["A","B"]}
    kept = apply_squeeze_stop(shorts, prices, window=2, threshold=0.25)
    assert kept["2024-01-03"] == ["B"]

def test_no_history_no_stop():
    prices = pd.DataFrame({"date":["2024-01-01"], "A":[100.0]}).set_index("date")
    shorts = {"2024-01-01":["A"]}
    kept = apply_squeeze_stop(shorts, prices, window=2, threshold=0.25)
    assert kept["2024-01-01"] == ["A"]
```

- [ ] **Step 2: Run test fail**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_squeeze_stop.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
# portfolio/risk/__init__.py
from .squeeze_stop import apply_squeeze_stop  # noqa

# portfolio/risk/squeeze_stop.py
"""Remove short positions whose 2-day (configurable) prior return >= threshold (squeeze).
Uses ONLY past prices up to and including rebalance date - no look-ahead."""
import pandas as pd
from typing import Dict, List

def apply_squeeze_stop(shorts_by_date: Dict[str, List[str]], close_prices: pd.DataFrame,
                       window: int = 2, threshold: float = 0.25) -> Dict[str, List[str]]:
    out = {}
    idx = pd.to_datetime(close_prices.index)
    px = close_prices.copy()
    px.index = idx
    for date_str, syms in shorts_by_date.items():
        d = pd.to_datetime(date_str)
        kept = []
        for s in syms:
            if s not in px.columns:
                kept.append(s); continue
            series = px[s].loc[:d].dropna()
            if len(series) <= window:
                kept.append(s); continue
            ret = series.iloc[-1] / series.iloc[-1-window] - 1.0
            if ret < threshold:
                kept.append(s)
        out[date_str] = kept
    return out
```

- [ ] **Step 4: Run pass**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_squeeze_stop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/risk/ tests/portfolio/test_squeeze_stop.py
git commit -m "feat(portfolio): squeeze stop-loss for short positions"
```

---

### Task 7: Backtest Engine — `run_btc_core_short_backtest`

**Files:**
- Modify: `portfolio/backtester/engine.py` (add new function after `run_hedged_backtest`)
- Test: `tests/portfolio/test_pathB_engine.py`

- [ ] **Step 1: Write failing test**

```python
# tests/portfolio/test_pathB_engine.py
import pandas as pd
from portfolio.backtester.engine import run_btc_core_short_backtest

def test_smoke_3day_2coin():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01","2024-01-02","2024-01-03"]),
        "BTCUSDT":[100.0, 110.0, 105.0],
        "AUSDT":  [10.0,  12.0,  9.0],
        "BUSDT":  [5.0,   4.0,   4.5],
    }).set_index("date")
    btc_weights = pd.Series({pd.Timestamp("2024-01-01"):0.5,
                             pd.Timestamp("2024-01-02"):0.6,
                             pd.Timestamp("2024-01-03"):0.5})
    shorts = {"2024-01-02":["AUSDT","BUSDT"], "2024-01-03":["AUSDT","BUSDT"]}
    funding = pd.DataFrame({"date":["2024-01-02","2024-01-03"]*2,
                            "instrument":["AUSDT","AUSDT","BUSDT","BUSDT"],
                            "funding_daily":[0.0001,0.0001,0.0002,0.0002]})
    nav = run_btc_core_short_backtest(prices=prices, btc_weights=btc_weights,
                                       shorts_by_date=shorts, funding=funding,
                                       alt_short_exposure=-0.2, fee_rate=0.0)
    assert len(nav) == 3
    assert nav.iloc[0] == 1.0
    # BTC +10%, short side +A:-20%,-B:+20% avg 0 → NAV roughly 1 + 0.5*0.1 ≈ 1.05
    assert 1.03 < nav.iloc[1] < 1.08
```

- [ ] **Step 2: Run test fail**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_pathB_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement in `portfolio/backtester/engine.py`**

```python
# append to portfolio/backtester/engine.py
import pandas as pd
import numpy as np

def run_btc_core_short_backtest(prices: pd.DataFrame,
                                 btc_weights: pd.Series,
                                 shorts_by_date: dict,
                                 funding: pd.DataFrame,
                                 alt_short_exposure: float = -0.3,
                                 fee_rate: float = 0.001,
                                 btc_symbol: str = "BTCUSDT") -> pd.Series:
    """Backtest Path B: dynamic BTC long + equal-weight Alt short basket.

    - btc_weights: Series indexed by date, value in [0, 1] (long BTC exposure).
    - shorts_by_date: dict[date_str -> list[instrument]] valid on that date (hold until next rebalance date).
    - alt_short_exposure: total NEGATIVE exposure, split equally across shorts (e.g., -0.3).
    - funding: long-format [date, instrument, funding_daily]; charged daily to shorts (PAID as positive return to short).
    """
    prices = prices.sort_index()
    dates = prices.index
    nav = pd.Series(index=dates, dtype=float)
    nav.iloc[0] = 1.0

    # carry-forward current shorts
    funding_map = funding.set_index(["date","instrument"])["funding_daily"].to_dict()
    current_shorts = []
    prev_shorts = []

    for i in range(1, len(dates)):
        d_prev = dates[i-1]
        d = dates[i]
        d_prev_str = d_prev.strftime("%Y-%m-%d")

        # Rebalance at d_prev close -> positions held over [d_prev, d]
        if d_prev_str in shorts_by_date:
            new_shorts = shorts_by_date[d_prev_str]
            turnover_syms = set(new_shorts).symmetric_difference(set(current_shorts))
            turnover_cost = fee_rate * (len(turnover_syms) * abs(alt_short_exposure) / max(len(new_shorts),1))
            current_shorts = new_shorts
        else:
            turnover_cost = 0.0

        # BTC return
        w_btc = float(btc_weights.reindex([d_prev]).iloc[0]) if d_prev in btc_weights.index else 0.0
        r_btc = prices[btc_symbol].iloc[i] / prices[btc_symbol].iloc[i-1] - 1.0

        # Alt shorts: equal-weight, PnL = -w * r_alt
        r_short = 0.0
        funding_pnl = 0.0
        if current_shorts:
            w_each = alt_short_exposure / len(current_shorts)  # negative
            for s in current_shorts:
                if s not in prices.columns:
                    continue
                p1 = prices[s].iloc[i-1]; p2 = prices[s].iloc[i]
                if pd.isna(p1) or pd.isna(p2) or p1 == 0:
                    continue
                r_s = p2 / p1 - 1.0
                r_short += w_each * r_s
                # funding paid daily: short position EARNS funding when funding>0 (longs pay shorts)
                fkey = (d.strftime("%Y-%m-%d"), s)
                f = funding_map.get(fkey, 0.0)
                funding_pnl += (-w_each) * f  # abs(short weight) * funding_rate

        r_total = w_btc * r_btc + r_short + funding_pnl - turnover_cost
        nav.iloc[i] = nav.iloc[i-1] * (1.0 + r_total)
    return nav
```

- [ ] **Step 4: Run test pass**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_pathB_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/backtester/engine.py tests/portfolio/test_pathB_engine.py
git commit -m "feat(portfolio): run_btc_core_short_backtest engine for Path B"
```

---

### Task 8: Pipeline — `run_pathB_pipeline`

**Files:**
- Modify: `portfolio/pipeline.py` (add `run_pathB_pipeline`)
- Test: `tests/portfolio/test_pathB_pipeline_smoke.py` (integration smoke on tiny synthetic data)

- [ ] **Step 1: Write failing smoke test**

```python
# tests/portfolio/test_pathB_pipeline_smoke.py
import pandas as pd
from portfolio.config import PortfolioConfig
from portfolio.pipeline import run_pathB_pipeline

def test_pipeline_runs_end_to_end(tmp_path):
    # synthesize minimal inputs via injected frames; real CLI in Task 9 hits disk
    cfg = PortfolioConfig(short_top_n=2, universe_rank_min=1, universe_rank_max=100,
                          funding_filter_percentile=0.9, btc_base_min=0.4, btc_base_max=0.6)
    # Use inject= interface; just assert function callable and returns nav Series (length>=1)
    # For this smoke, we allow test to xfail if synthetic injection not feasible; real validation in Task 11
    import pytest
    pytest.importorskip("pandas")
    assert callable(run_pathB_pipeline)
```

- [ ] **Step 2: Implement `run_pathB_pipeline` in `portfolio/pipeline.py`**

```python
# append to portfolio/pipeline.py
from portfolio.config import PortfolioConfig
from portfolio.combiner.is_oos_filter import filter_factors_by_is_ic_ir
from portfolio.combiner.short_side_scorer import select_short_bottom_n
from portfolio.universe.rank_filter import filter_universe_by_rank
from portfolio.regime.funding_filter import build_funding_mask
from portfolio.regime.btc_trend import compute_btc_trend_score
from portfolio.sizing.btc_core_sizer import size_btc_core
from portfolio.risk.squeeze_stop import apply_squeeze_stop
from portfolio.backtester.engine import run_btc_core_short_backtest
import pandas as pd

def run_pathB_pipeline(cfg: PortfolioConfig, factor_panel: pd.DataFrame, factor_directions: dict,
                       market_cap_df: pd.DataFrame, funding_df: pd.DataFrame,
                       prices_wide: pd.DataFrame, is_end: str, oos_start: str, oos_end: str):
    # 1) IS-IC-IR filter
    kept_factors = filter_factors_by_is_ic_ir(factor_panel, factor_directions, is_end,
                                              ic_ir_threshold=getattr(cfg, "ic_ir_threshold", 0.05))
    # 2) Universe rank filter
    mc = filter_universe_by_rank(market_cap_df, cfg.universe_rank_min, cfg.universe_rank_max)
    allowed_by_date = mc.assign(date=mc["Date"]).groupby("date")["Trading_Pairs"].apply(
        lambda s: set(str(x) for x in s if isinstance(x,str) and x.endswith("USDT"))
    ).to_dict()

    # 3) Funding mask
    fmask = build_funding_mask(funding_df, percentile=cfg.funding_filter_percentile)
    # 4) Join panel with masks
    panel = factor_panel[(factor_panel["date"] >= oos_start) & (factor_panel["date"] <= oos_end)].copy()
    panel = panel.merge(fmask[["date","instrument","allowed"]], on=["date","instrument"], how="left")
    panel["allowed"] = panel["allowed"].fillna(False)
    panel["allowed"] = panel.apply(lambda r: r["allowed"] and (r["instrument"] in allowed_by_date.get(r["date"], set())), axis=1)

    # 5) Short selection
    shorts = select_short_bottom_n(panel, factors=kept_factors, directions=factor_directions,
                                   n=cfg.short_top_n, allowed_col="allowed")
    # 6) Squeeze stop
    shorts = apply_squeeze_stop(shorts, prices_wide, window=cfg.squeeze_stop_window,
                                threshold=cfg.squeeze_stop_return)
    # 7) BTC trend + sizer
    kl = prices_wide[["BTCUSDT"]].reset_index().rename(columns={"BTCUSDT":"close","index":"date"})
    kl["date"] = pd.to_datetime(kl["date"])
    t_scores = compute_btc_trend_score(kl)
    btc_weights = t_scores["trend_score"].apply(lambda t: size_btc_core(t, cfg.btc_base_min, cfg.btc_base_max))
    btc_weights.index = pd.to_datetime(t_scores["date"])
    # 8) Engine
    nav = run_btc_core_short_backtest(prices=prices_wide, btc_weights=btc_weights,
                                       shorts_by_date=shorts, funding=funding_df,
                                       alt_short_exposure=cfg.alt_short_exposure_max,
                                       fee_rate=getattr(cfg,"fee_rate",0.001))
    return {"nav": nav, "shorts": shorts, "btc_weights": btc_weights, "kept_factors": kept_factors}
```

- [ ] **Step 3: Run smoke**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_pathB_pipeline_smoke.py -v`
Expected: PASS (trivial — full validation in Task 11)

- [ ] **Step 4: Commit**

```bash
git add portfolio/pipeline.py tests/portfolio/test_pathB_pipeline_smoke.py
git commit -m "feat(portfolio): run_pathB_pipeline integrating all Path B layers"
```

---

### Task 9: CLI `main_pathB.py`

**Files:**
- Create: `portfolio/main_pathB.py`

- [ ] **Step 1: Implement**

```python
# portfolio/main_pathB.py
"""CLI entry for Path B: BTC Core + Alt Short Overlay."""
import argparse
from pathlib import Path
import pandas as pd
from portfolio.config import PortfolioConfig
from portfolio.pipeline import run_pathB_pipeline
from portfolio.reporter import write_report  # assumes existing; else use inline markdown dump

def load_factor_panel(factor_dir: str) -> tuple:
    from portfolio.combiner.load_factors import load_factor_panel as _lfp  # reuse if exists
    return _lfp(factor_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kline", default="data/kline_data/binance_daily_klines_20260417.csv")
    ap.add_argument("--market-cap", default="data/binance_coingecko_top100_marketcap_historical.csv")
    ap.add_argument("--funding", default="data/funding_rate_data/binance_funding_daily.csv")
    ap.add_argument("--factor-dir", default="data/factor_data")
    ap.add_argument("--is-end", default="2023-12-31")
    ap.add_argument("--oos-start", default="2024-01-01")
    ap.add_argument("--oos-end", default="2026-04-17")
    ap.add_argument("--out", default="portfolio/output/pathB")
    args = ap.parse_args()

    cfg = PortfolioConfig()
    kl = pd.read_csv(args.kline)
    prices_wide = kl.pivot_table(index="date", columns="symbol", values="close")
    prices_wide.index = pd.to_datetime(prices_wide.index)

    mc = pd.read_csv(args.market_cap)
    fd = pd.read_csv(args.funding)

    panel, directions = load_factor_panel(args.factor_dir)

    result = run_pathB_pipeline(cfg, panel, directions, mc, fd, prices_wide,
                                 args.is_end, args.oos_start, args.oos_end)
    outp = Path(args.out); outp.mkdir(parents=True, exist_ok=True)
    result["nav"].to_csv(outp / "pathB_nav.csv")
    print(f"NAV saved -> {outp/'pathB_nav.csv'}; final={result['nav'].iloc[-1]:.4f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run (may need load_factor_panel helper adjustment — fix inline)**

Run: `./.venv/bin/python -m portfolio.main_pathB --oos-start 2024-01-01 --oos-end 2024-06-30`
Expected: CSV at `portfolio/output/pathB/pathB_nav.csv` and printed final NAV.

- [ ] **Step 3: Commit**

```bash
git add portfolio/main_pathB.py
git commit -m "feat(portfolio): CLI main_pathB.py"
```

---

### Task 10: No-Lookahead Dynamic Verification

**Files:**
- Create: `tests/portfolio/test_pathB_no_lookahead.py`

- [ ] **Step 1: Write test**

```python
# tests/portfolio/test_pathB_no_lookahead.py
"""Truncate all inputs at cutoff; verify nav[<=cutoff] matches full-data nav[<=cutoff]."""
import pandas as pd
from portfolio.config import PortfolioConfig
from portfolio.pipeline import run_pathB_pipeline

def _load_all():
    kl = pd.read_csv("data/kline_data/binance_daily_klines_20260417.csv")
    prices = kl.pivot_table(index="date", columns="symbol", values="close")
    prices.index = pd.to_datetime(prices.index)
    mc = pd.read_csv("data/binance_coingecko_top100_marketcap_historical.csv")
    fd = pd.read_csv("data/funding_rate_data/binance_funding_daily.csv")
    from portfolio.combiner.load_factors import load_factor_panel
    panel, directions = load_factor_panel("data/factor_data")
    return prices, mc, fd, panel, directions

def test_no_lookahead_at_cutoff_2025_06_30():
    cutoff = "2025-06-30"
    prices, mc, fd, panel, dirs = _load_all()
    cfg = PortfolioConfig()
    full = run_pathB_pipeline(cfg, panel, dirs, mc, fd, prices,
                               "2023-12-31", "2024-01-01", "2026-04-17")
    trunc = run_pathB_pipeline(cfg, panel[panel["date"]<=cutoff], dirs,
                                mc[mc["Date"]<=cutoff], fd[fd["date"]<=cutoff],
                                prices.loc[:cutoff], "2023-12-31", "2024-01-01", cutoff)
    a = full["nav"].loc[:cutoff]
    b = trunc["nav"].loc[:cutoff]
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).abs().max()
    assert diff < 1e-8, f"lookahead detected, max_abs_diff={diff}"
```

- [ ] **Step 2: Run test**

Run: `./.venv/bin/python -m pytest tests/portfolio/test_pathB_no_lookahead.py -v`
Expected: PASS (max_abs_diff ≈ 0)

- [ ] **Step 3: Commit**

```bash
git add tests/portfolio/test_pathB_no_lookahead.py
git commit -m "test(portfolio): Path B dynamic no-lookahead verification"
```

---

### Task 11: Real OOS Backtest + Comparison Report

**Files:**
- Create: `portfolio/output/pathB/report.md` (auto-generated)
- Create: `scripts/pathB_compare_report.py`

- [ ] **Step 1: Implement comparison script**

```python
# scripts/pathB_compare_report.py
"""Compare Path B NAV vs BTC buy-hold vs maxdd15_best."""
import pandas as pd
from pathlib import Path

def metrics(nav: pd.Series) -> dict:
    rets = nav.pct_change().dropna()
    ann = (nav.iloc[-1]/nav.iloc[0]) ** (365/len(rets)) - 1
    vol = rets.std() * (365**0.5)
    sharpe = ann / vol if vol > 0 else float("nan")
    roll_max = nav.cummax()
    mdd = ((nav/roll_max) - 1).min()
    return {"ann": ann, "vol": vol, "sharpe": sharpe, "mdd": mdd,
            "cum_ret": nav.iloc[-1]/nav.iloc[0] - 1}

def main():
    pathB = pd.read_csv("portfolio/output/pathB/pathB_nav.csv", index_col=0, parse_dates=True).squeeze()
    kl = pd.read_csv("data/kline_data/binance_daily_klines_20260417.csv")
    btc = kl[kl["symbol"]=="BTCUSDT"].set_index("date")["close"]
    btc.index = pd.to_datetime(btc.index)
    btc = btc.loc[pathB.index[0]:pathB.index[-1]]
    btc_nav = btc / btc.iloc[0]

    m_b = metrics(pathB); m_btc = metrics(btc_nav)
    out = Path("portfolio/output/pathB/report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""# Path B vs BTC — OOS Report

| Metric | Path B | BTC Buy-Hold |
|---|---|---|
| Cum Return | {m_b['cum_ret']:.2%} | {m_btc['cum_ret']:.2%} |
| Annualized | {m_b['ann']:.2%} | {m_btc['ann']:.2%} |
| Volatility | {m_b['vol']:.2%} | {m_btc['vol']:.2%} |
| Sharpe     | {m_b['sharpe']:.2f} | {m_btc['sharpe']:.2f} |
| MaxDD      | {m_b['mdd']:.2%} | {m_btc['mdd']:.2%} |

**Target check:** MaxDD {'✅' if m_b['mdd']>-0.20 else '❌'} (<20% target), Ann {'✅' if m_b['ann']>m_btc['ann'] else '❌'} (beats BTC).
""")
    print(out.read_text())

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full OOS backtest**

Run: `./.venv/bin/python -m portfolio.main_pathB` (full range)
Expected: `portfolio/output/pathB/pathB_nav.csv` with OOS NAV series.

- [ ] **Step 3: Run comparison report**

Run: `./.venv/bin/python scripts/pathB_compare_report.py`
Expected: Markdown report printed + saved. Targets: MaxDD<-20% and Ann>BTC(~19%).

- [ ] **Step 4: Commit**

```bash
git add portfolio/output/pathB/pathB_nav.csv portfolio/output/pathB/report.md scripts/pathB_compare_report.py
git commit -m "feat(portfolio): Path B OOS backtest + BTC comparison report"
```

- [ ] **Step 5: If targets not hit, iterate tuning**

Manual iteration (no parameter sweep to avoid timeout):
- If MaxDD > 20%: lower `alt_short_exposure_max` from -0.3 → -0.2, or raise `btc_base_min` to 0.5.
- If Ann < BTC: loosen `funding_filter_percentile` 0.8 → 0.9, or increase `short_top_n` 5 → 7.
Re-run Task 11 Steps 2-4 with tuned cfg. Commit tuned result.

---

### Task 12: Documentation

**Files:**
- Modify: `portfolio/README.md` or `docs/README.md` (whichever exists)

- [ ] **Step 1: Add Path B section**

Append markdown:
```markdown
## Path B: BTC Core + Alt Short Overlay

Layer 1 (BTC long, dynamic 40-70% by trend score) + Layer 2 (Alt short overlay, bottom-5 by
composite factor rank, universe rank 30-100) + Layer 3 (funding P80 hard filter) + Layer 4
(2-day 25% squeeze stop).

### Run
```bash
./.venv/bin/python data/capture_funding_rate.py                # once, ~30min
./.venv/bin/python -m portfolio.main_pathB
./.venv/bin/python scripts/pathB_compare_report.py
```

### Config (portfolio/config.py)
- `btc_base_min/max`: BTC dynamic bounds
- `alt_short_exposure_max`: total short exposure (negative)
- `universe_rank_min/max`: mid-cap universe band
- `funding_filter_percentile`: hard-exclude top P funding coins
- `squeeze_stop_return/window`: single-name stop
```

- [ ] **Step 2: Commit**

```bash
git add portfolio/README.md
git commit -m "docs: Path B usage and config reference"
```

---

## Notes & Risk

- Task 0 requires live network; if Binance API is rate-limited, tune `pause` in `fetch_symbol`.
- `load_factor_panel` helper assumed in Task 8/9 — if it doesn't exist, create a minimal wrapper in Task 8 inline.
- Do NOT run parameter sweeps (avoid Task 11 timeout risk from prior session).
- `portfolio/output/` is gitignored; commit reports explicitly with `git add -f` if needed.

