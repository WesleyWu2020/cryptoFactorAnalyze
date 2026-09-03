# Portfolio 层设计：多因子合成 + 组合构建 + 回测 + 归因

状态：Draft（待实施）
日期：2026-04-17
作者：基金经理视角 brainstorm（Copilot CLI / Claude Opus 4.7）
下一步：writing-plans → 实施计划

---

## 1. 目标与边界

### 1.1 目标

基于现有单因子库（`factor_analyse/factor_mining/` 约 35+ 规则因子 + Alpha101 + ML/GA），建设 portfolio 层，产出：

1. **每日目标权重** `target_weights.csv`（供未来 `execution/` 层消费下单）；
2. **完整组合回测 HTML 报告**（沿用 `board.html` 风格，含净值、夏普、回撤、换手、归因）；
3. 支持纯多头（默认）和"多头 + 市值加权指数对冲"两种模式；
4. 支持多策略配置 A/B 对比。

### 1.2 不做（明确 YAGNI）

- 实盘下单 / 订单管理 → 留 `execution/`
- 事前 / 事中风控拦截 → 留 `risk/`
- ML stacking 合成因子 → v2（本期只做 IC_IR 加权 + 对称正交）
- 另类数据（资金费率、持仓量、逐笔、深度）→ 留 `data/` 扩展
- `core/` 共享库抽取 → 延后（本期完成后评估）

### 1.3 契约（portfolio 层的单向门）

**输入（只读）**
- `data/factor_data/*.csv`：`date, instrument, factor[, future_ret]`，其中 `future_ret` 列**不使用**（见 §3.3）
- `data/kline_data/*.csv`：逐币价格与成交量
- `data/binance_coingecko_top100_marketcap_historical.csv`：动态市值池与流动性快照

**输出（写入）**
- `portfolio/output/target_weights_{strategy}_{YYYYMMDD}.csv`：列 `date, instrument, weight`
- `portfolio/output/positions_history_{strategy}.csv`：历史目标权重全量（回测/归因用）
- `reports/portfolio_{strategy}_{YYYYMMDD}.html`：组合报告

**依赖规则**
- `portfolio/` 可 import `data/`（读 CSV）和自身；**不允许** import `factor_analyse/`。
- 因子 CSV 为两层契约边界，研究端改因子实现不影响 portfolio。
- 未来 `execution/` 只消费 `portfolio/output/target_weights_*.csv`。

---

## 2. 目录结构

```
portfolio/
├── __init__.py
├── config.py                 # 策略配置字典（见 §4）
├── main.py                   # CLI 入口
├── pipeline.py               # 编排器：串联模块，传递 context
├── factor_pool/
│   ├── loader.py             # 读白名单因子 CSV（只取 date/instrument/factor）
│   ├── aligner.py            # 合并到 panel，对齐到 universe
│   ├── label_builder.py      # 从 kline 按策略 N 重算 future_ret
│   └── screener.py           # 滚动 IC_IR 门槛剔除失效因子
├── combiner/
│   ├── normalizer.py         # 横截面 rank + z-score + winsorize + 方向统一
│   ├── orthogonalizer.py     # 对称正交（Schmidt 可选）
│   ├── ic_ir_weighter.py     # 滚动 RankIC_IR 因子权重
│   └── synthesizer.py        # 合成打分 score
├── portfolio_builder/
│   ├── universe.py           # 动态 Top100 市值池 + 流动性 + 黑名单
│   ├── topn_equal.py         # Top-N 等权（基线）
│   ├── optimizer.py          # cvxpy 均值-方差（进阶）
│   ├── hedge.py              # 市值指数对冲
│   └── constraints.py        # 约束与黑名单常量
├── backtester/
│   ├── engine.py             # drift 模式组合回测
│   ├── fees.py               # 手续费 + 滑点
│   └── metrics.py            # 净值/夏普/回撤/换手/IR/beta
├── attribution/
│   ├── return_attr.py        # 收益归因：因子贡献
│   ├── risk_attr.py          # 风险贡献分解
│   └── factor_contrib.py     # 权重历史 + 滚动 IC
├── reporter/
│   ├── html_renderer.py
│   └── templates/
└── output/                   # CSV 产物（git-ignored）
```

`portfolio/` 为新顶层目录，与 `data/`、`factor_analyse/` 同级。

---

## 3. 核心数据流与时间口径

### 3.1 时间轴约定

- 因子 CSV 的 `date = t` 表示"截止 t 日收盘、仅用 `<= t` 数据"计算的信号（与 `AGENTS.md` Future-leak 规则一致）。
- portfolio 在 `date = t` 的决策仅使用 `date <= t` 的因子与价格。
- `target_weights[t]` 表示"t 收盘后发出、t+1 开盘执行"的目标仓位。
- 回测收益错位一期：`port_ret[t] = w[t-1] · ret[t-1 → t] - fees(Δw)`。
- 调仓周期 N 日：仅在 `t` 为调仓日重算权重；非调仓日按 `drift` 模式漂移（见 §5.8）。

### 3.2 Pipeline 数据流

```
1. factor_pool.loader        读白名单因子 CSV → dict{name: DataFrame[date,instrument,factor]}
2. factor_pool.aligner       合并为 panel[date × instrument × factor]；应用 universe 过滤
3. factor_pool.label_builder 从 kline 按策略 rebalance.period_days 统一重算 future_ret
4. factor_pool.screener      滚动窗口 [t-L, t-1] 检查 |mean(RankIC)| 与 IC_IR；产出 survivor_mask[t,i]
5. combiner.normalizer       方向统一 × winsorize × rank × z-score（仅横截面）
6. combiner.orthogonalizer   每 t 独立做对称正交 X_ortho = X · (XᵀX)^(-1/2)
7. combiner.ic_ir_weighter   每 t 用 [t-L, t-1] 且 s+N <= t-1 的 IC 序列算 IC_IR 权重
8. combiner.synthesizer      score[t, ins] = Σ_i w_i(t) · X_ortho[t, ins, i]
9. portfolio_builder         topn_equal 或 optimizer → target_weights[t]；可选 hedge
10. backtester.engine        drift 模式组合回测
11. attribution + reporter   HTML 报告
```

### 3.3 关键决策：统一 future_ret 口径

**问题**：不同因子 CSV 的 `future_ret` 是按各自 native 周期（如 `rebalance10d`、`rebalance3d`）算的，直接使用会导致 IC_IR 不可比。

**决策**：

1. `factor_pool.loader` 只取 `date, instrument, factor` 三列，**丢弃 CSV 中自带的 `future_ret`**。
2. `factor_pool.label_builder` 从 `data/kline_data/` 按策略 `rebalance.period_days = N` 统一计算：
   `future_ret[t, ins] = close[t+N, ins] / close[t, ins] - 1`
3. 所有因子共享同一 `future_ret` 序列，IC_IR 可公平比较与加权。
4. 因子的 "native 周期"仅作为 `factor_whitelist` 的可选 metadata 用于报告诊断，**不做筛选门槛**；是否保留由 screener 用事实说话。

### 3.4 未来函数硬检查（必过）

- 所有滚动窗口：`[t-L, t-1]`，**严禁包含 t**。
- IC 计算：标签使用 `ret[s → s+N]` 的 s 必须满足 `s+N <= t-1`（双重错位）。
- 回测：`port_ret[t]` 必须用 `w[t-1]`。
- **端到端 cutoff 反证测试**：全量 vs `data[<= cutoff]` 产出的 `target_weights[t <= cutoff]` 必须 `max_abs_diff < 1e-10`（见 §6）。

---

## 4. 配置 Schema（`portfolio/config.py`）

```python
STRATEGY_CONFIG = {
  "baseline_topn_long": {
    # ── 因子池 ─────────────────────────────────────
    "factor_whitelist": [
      # 可用 str（简写）或 dict（带 metadata）
      {"type": "directional_momentum_20d", "native_rebalance": 10},
      {"type": "cgo_strict_log_10d",       "native_rebalance": 3},
      "volume_stability_20d",
      "volatility_efficiency_20d",
      "retail_activity_divergence_20d",
      "volatility_order_flow_20d",
    ],
    "screener": {
      "enabled": True,
      "lookback_days": 60,
      "ic_threshold": 0.02,
      "icir_threshold": 0.3,
    },

    # ── 合成 ───────────────────────────────────────
    "combiner": {
      "winsorize_pct": (0.01, 0.99),
      "orthogonal": "symmetric",     # "symmetric" | "schmidt" | "none"
      "ic_ir_lookback": 60,
    },

    # ── Universe ──────────────────────────────────
    "universe": {
      "source": "coingecko_top100",
      "top_n_mcap": 100,
      "liquidity_min_quote_volume_usd": 5_000_000,
      "liquidity_lookback": 20,
      "stablecoin_blacklist": ["USDC","BUSD","FDUSD","TUSD","DAI","USDP"],
      "wrapped_blacklist":   ["WBTC","WETH","STETH"],
    },

    # ── 组合构建 ───────────────────────────────────
    "builder": {
      "mode": "topn_equal",           # "topn_equal" | "optimizer"
      "top_n": 10,
      "max_weight": 0.20,
      "leverage": 1.0,
      "turnover_cap": 0.50,
      "hedge": {
        "enabled": False,
        "instrument": "MCAP_INDEX",
        "beta_lookback": 60,
      },
      # optimizer only
      "risk_aversion": 5.0,
      "cov_lookback": 60,
      "cov_shrinkage": "ledoit_wolf",
    },

    # ── 调仓 ───────────────────────────────────────
    "rebalance": {
      "period_days": 3,
      "hold_mode": "drift",           # 固定为 drift
    },

    # ── 成本 ───────────────────────────────────────
    "costs": {
      "fee_rate": 0.0003,
      "slippage_bps": 5,
    },

    # ── 回测区间 ───────────────────────────────────
    "backtest": {
      "start": "2022-01-01",
      "end": None,
      "in_sample_end": "2024-06-30",
    },

    # ── 归因 ───────────────────────────────────────
    "attribution": {
      "benchmarks": ["BTCUSDT", "MCAP_INDEX"],
      "risk_decomp": True,
      "factor_contrib": True,
    },
  },

  "baseline_topn_hedged": {
    "inherits": "baseline_topn_long",
    "builder": {"hedge": {"enabled": True}},
  },

  "optimizer_mv": {
    "inherits": "baseline_topn_long",
    "builder": {"mode": "optimizer"},
  },
}
```

**CLI**

```bash
./.venv/bin/python portfolio/main.py --list
./.venv/bin/python portfolio/main.py baseline_topn_long
./.venv/bin/python portfolio/main.py baseline_topn_long --dry-run   # 只写权重，不出报告
./.venv/bin/python portfolio/main.py optimizer_mv    --rebalance 1  # 覆盖 rebalance.period_days
```

`inherits` 由 `main.py` 在加载时递归合并（子配置覆盖父配置）。

---

## 5. 关键算法细节

### 5.1 Universe 构建（`portfolio_builder/universe.py`）

```python
def build_universe(t, mcap_df, cfg) -> set[str]:
    row_t = mcap_df[(mcap_df.Decision_Window_Start <= t) &
                    (mcap_df.Decision_Window_End   >= t)]
    pool = set(row_t.nlargest(cfg.top_n_mcap, "Market_Cap_USD").Trading_Pairs)
    pool -= STABLE_BLACKLIST | WRAPPED_BLACKLIST
    liq  = row_t.set_index("Trading_Pairs")["Quote_Volume_USD"]
    pool = {s for s in pool if liq.get(s, 0) >= cfg.liquidity_min_quote_volume_usd}
    return pool
```

- 按日期预打表为 `dict[date, set]`。
- `Decision_Window_*` 来自 `data/` 层产出（由历史市值快照决定），天然无幸存者偏差。

### 5.2 横截面标准化（`combiner/normalizer.py`）

- 方向统一：因子方向从 `factor_analyse/factor_config.py` 的 `factor_direction` 读取（+1 保持，-1 乘 -1，统一为"越大越好"）。
- Winsorize `[1%, 99%]`；rank → `[0, 1]`；z-score；**全部在同 date 的横截面内**。
- NaN 不参与 rank，标准化后仍保留 NaN，合成时作 0 处理但不进入因子权重分母。

### 5.3 对称正交（`combiner/orthogonalizer.py`）

- 每 date 独立：`X_ortho = X · (XᵀX)^(-1/2)`，特征分解后对特征值 `clip(>=1e-10)` 防病态。
- 退化情形（n_ins < n_factor）：降级到 Schmidt 或 `none`（保留 z-score），记录 warning。

### 5.4 滚动 IC_IR 权重（`combiner/ic_ir_weighter.py`）

- 窗口 `[t-L, t-1]`，标签 `s+N <= t-1`。
- `IC_IR_i = mean(IC_i) / (std(IC_i) + 1e-9)`；`w_i(t) = IC_IR_i · survivor_mask[t, i]`；
- 归一化：`w = w / Σ|w|`。

### 5.5 Screener（`factor_pool/screener.py`）

- 滚动窗口计算每因子 `mean(RankIC)` 与 `IC_IR`；
- `survivor_mask[t, i] = (|mean(RankIC)| >= ic_threshold) AND (IC_IR >= icir_threshold)`；
- 剔除的因子在 `ic_ir_weighter` 中权重为 0，自动退出合成。

### 5.6 Top-N 等权（`portfolio_builder/topn_equal.py`）

1. 按 score 排序取 Top-N；
2. 初始权重 `leverage / N`；
3. 单票上限 `max_weight` clip；
4. 重归一化到 `leverage`；
5. 换手约束：若 `‖w - w_prev‖₁ > turnover_cap`，沿 `Δw` 方向等比回拉到上限。

### 5.7 优化器（`portfolio_builder/optimizer.py`，cvxpy）

```python
w = cp.Variable(n)
obj  = cp.Maximize(mu @ w - cfg.risk_aversion * cp.quad_form(w, Sigma))
cons = [w >= 0,
        w <= cfg.max_weight,
        cp.sum(w) <= cfg.leverage,
        cp.norm(w - w_prev, 1) <= cfg.turnover_cap]
cp.Problem(obj, cons).solve(solver=cp.ECOS)
```

- `mu` = score；`Sigma` 用 `[t-cov_lookback, t-1]` 的日收益样本，Ledoit-Wolf 收缩。
- 求解失败 → 降级到 Top-N 等权，记录 warning（不中断流程）。

### 5.8 对冲（`portfolio_builder/hedge.py`）

- 近 `beta_lookback` 日组合收益对指数收益回归，得 `beta_port`；
- 对冲头寸 `w_hedge = -beta_port`（面值占 leverage 的负头寸）；
- 输出两张表：现货多头 `w_long` + 对冲空头 `w_hedge`（以 `MCAP_INDEX` 为记号）。
- `execution/` 层未来决定映射到 BTC 永续或指数篮子。

### 5.9 回测引擎（`backtester/engine.py`，drift）

记 `r_i = ret[t-1 → t, i]`，`port_ret = Σ w_{i,t-1} · r_i`。

- 非调仓日漂移：`w_{i,t} = w_{i,t-1} · (1 + r_i) / (1 + port_ret)`（保持 `Σw = Σw_{t-1}`）。
- 调仓日：先按上式漂移到 `w_drifted`，再 `Δw = target[t] - w_drifted`，扣费 `fees(Δw)`，之后 `w_held = target[t]`。
- 费用：`fees = Σ|Δw_i| · (fee_rate + slippage_bps / 10000)`。
- 数值保护：当 `1 + port_ret < eps` 时保持 `w_{t-1}` 不做归一并记录 warning。

### 5.10 归因

- **收益归因**：`port_ret[t] ~ Σ β_i · ortho_factor_i[t]` 的横截面回归，β_i · f_i 作为因子贡献。
- **风险归因**：`Var(port) = Σ_i Σ_j w_i w_j Cov(f_i, f_j)`，报告按因子对角项累计。
- **单因子贡献面板**：`w_i(t)` 权重曲线 + 每因子累计贡献收益 + 滚动 RankIC。
- **基准对比**：相对 `BTCUSDT` 和 `MCAP_INDEX` 的超额、IR、beta。

---

## 6. 测试策略

每个模块一个 minimal test，全部放 `tests/`，沿用 pytest：

| 文件 | 验证 |
|---|---|
| `test_portfolio_universe.py` | t 日查表正确；黑名单剔除；流动性过滤 |
| `test_portfolio_aligner.py` | 多因子 CSV 对齐到 panel；缺失填充 |
| `test_portfolio_label_builder.py` | `future_ret[t, N]` 与 kline 实算一致；边界 NaN 正确 |
| `test_portfolio_normalizer.py` | rank/z-score 只横截面；winsorize 对称；方向翻转 |
| `test_portfolio_orthogonalizer.py` | 正交后 `XᵀX ≈ I`；退化情形降级；数值稳定性 |
| `test_portfolio_ic_ir_weighter.py` | 窗口严格 `< t`；标签 `s+N <= t-1`；survivor_mask 生效 |
| `test_portfolio_topn_builder.py` | 权重上限、换手约束生效；归一化正确 |
| `test_portfolio_optimizer.py` | cvxpy 约束满足；求解失败降级到 Top-N |
| `test_portfolio_backtester_drift.py` | drift 再归一无收益损失；调仓日费用正确 |
| `test_portfolio_future_leak.py` | **端到端 cutoff 反证**（必过） |

端到端反证示例：

```python
def test_target_weights_no_lookahead():
    cutoff = "2024-06-30"
    w_full = run_pipeline(all_data, "baseline_topn_long").target_weights
    w_cut  = run_pipeline(all_data[all_data.date <= cutoff],
                          "baseline_topn_long").target_weights
    diff = (w_full.loc[:cutoff] - w_cut.loc[:cutoff]).abs().max().max()
    assert diff < 1e-10
```

---

## 7. 分期实施

| 期 | 目标 | 范围 | 产出 | 验收 |
|---|---|---|---|---|
| **P1 MVP** | 跑通 long-only Top-N 基线 | `factor_pool`（loader/aligner/label_builder）；`combiner`（normalizer/orthogonalizer/ic_ir_weighter/synthesizer）；`portfolio_builder.topn_equal`；`backtester.engine`；简版 `reporter` | `baseline_topn_long` 策略的 `target_weights.csv` + HTML 报告 | `test_portfolio_future_leak.py` 通过；合成因子 IC > 白名单单因子平均 IC |
| **P2 增强** | 对冲 + 优化器 + 完整归因 | `portfolio_builder.optimizer`；`hedge.py`；`attribution/*`；完整 `reporter` | `baseline_topn_hedged`、`optimizer_mv` 两条新策略线 | 三策略 HTML 报告可对比；归因面板可用 |
| **P3 打磨** | 自动筛选 + A/B + 调度集成 | `factor_pool.screener`；config `inherits` 机制；接入 `daily_feishu_scheduler` | 每日自动生成权重 + 飞书推送多策略对比 | 调度稳定运行 7 日；权重可重现 |

**不做**：参见 §1.2。

---

## 8. 风险与开放问题

1. **cvxpy 依赖**：`requirements.txt` 当前未含，P2 时需新增并确认版本（`cvxpy>=1.4`）。
2. **universe 覆盖**：`binance_coingecko_top100_marketcap_historical.csv` 当前覆盖 `2021-01-01 ~ 2026-03-01`；若数据截止早于回测 end，需让 `universe.py` 回退到最近可用窗口并记录 warning。
3. **因子 CSV 的 `date` 覆盖差异**：不同因子可能起止日期不一致；`aligner` 以 "所有因子都存在" 的交集为默认 panel 区间，并在策略 config 可选 `min_coverage_ratio` 放宽。
4. **Drift 归一的数值稳定**：`port_ret` 接近 -1 时归一化退化；实现里做 `eps` 保护。
5. **对冲指数的可交易性**：`MCAP_INDEX` 为合成概念，`execution/` 落地时必须映射到具体合约（BTC/ETH 永续或篮子），portfolio 层只承诺输出"指数对冲头寸"。
6. **ML 因子与规则因子混合**：本期假设所有 whitelist 因子 CSV 格式同构，`factor_direction` 从 `factor_analyse/factor_config.py` 查表；若 ML 因子未注册 direction，screener 可先用全量 `|IC|` 替代，并告警。

---

## 9. 与其他层的关系

```
┌────────────┐    factor CSVs      ┌──────────────────┐
│factor_analyse│ ──────────────►   │   portfolio/     │
└────────────┘                     │                  │
┌────────────┐    kline / mcap     │  (本 spec)       │
│   data/    │ ──────────────►     │                  │
└────────────┘                     │  target_weights  │
                                   │  HTML report     │
                                   └────────┬─────────┘
                                            │ target_weights.csv
                                            ▼
                                   ┌──────────────────┐
                                   │  execution/ (未来) │
                                   └──────────────────┘
```

- 研究端（`factor_analyse/`）与执行端（`execution/`）的**唯一桥梁**就是 `portfolio/`。
- 因子 CSV 与 target_weights CSV 是两条**单向门**契约。
