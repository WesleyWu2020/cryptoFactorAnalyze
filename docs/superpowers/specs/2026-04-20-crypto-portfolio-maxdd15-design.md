# 加密货币多因子组合：最大回撤 ≤15% 风控策略设计

**日期**：2026-04-20
**作者**：Quant PM + Copilot CLI
**目标**：在现有 Top-N 多因子组合的基础上，通过时序（BTC 趋势择时）+ 截面（因子筛选）叠加市场对冲，将 OOS 期间最大回撤降至 15% 以内。

---

## 1. 背景与当前问题

### 1.1 现有策略表现
基于 `portfolio/output/cmp_*.html` 的现有 8 个策略变体，全部对比 BTC Buy&Hold：

| 指标 | 现有策略范围 | BTC Buy&Hold |
|------|------------|--------------|
| 年化收益 | 17% ~ 39% | 33.77% |
| Sharpe | 0.23 ~ 0.55 | 0.693 |
| MaxDD | **−85% ~ −94%** | −76.63% |
| Calmar | 0.19 ~ 0.44 | 0.441 |

**结论**：0/8 策略在 Sharpe 或 MaxDD 上超过 BTC，1/8 超过绝对收益。

### 1.2 根因分析（三大结构性问题）

1. **高 Beta 暴露**：Alt 组合相对 BTC 的 Beta 均值 ≈ 1.28（范围 0.67-1.98），组合年化波动率 69-75% >> BTC 49%，熊市系统性高频爆仓
2. **因子信号极弱**：全样本 35 因子中，24 个 IC 为负反向拖累；最强因子 IC_IR 仅 0.19
3. **高频换仓的摩擦成本**：daily_top3 年化手续费估算 44-73%，直接吃掉 alpha

### 1.3 改造目标

**OOS 期间（2024-01 ~ 2026-04）硬性要求**：
- MaxDD ≤ 15%
- 年化收益 ≥ 20%
- Sharpe ≥ 1.2

**工具箱约束**：
- 可用：Alt 现货多头、USDT 现金、BTC 永续合约（可空）
- 对冲上限：BTC 空头名义 ≤ Alt 现货名义 × 1.2x
- 不使用杠杆做 Alt 多头（Alt 多头名义 ≤ 100%）

---

## 2. 总体架构

```
┌────────────────────────────────────────────────────────────────┐
│                    每日驱动 Pipeline                             │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer A: Alt 选股层（每周换仓，Top 5 等权）                    │
│  ┌──────────────────────────────────────────────────┐           │
│  │ IS期筛选IC_IR>0.05因子 → 正交化 → IC_IR加权合成  │           │
│  │   → Top 5 等权 → Alt篮子                         │           │
│  └──────────────────────────────────────────────────┘           │
│                          │                                       │
│                          ▼                                       │
│  Layer B: BTC 趋势分数 T（每日更新）                            │
│  ┌──────────────────────────────────────────────────┐           │
│  │ T = 0.5·tanh(EMA50/EMA200 斜率) + 0.5·tanh(Z120) │           │
│  │ T ← EMA(T_raw, span=5) 平滑                       │           │
│  └──────────────────────────────────────────────────┘           │
│                          │                                       │
│                          ▼                                       │
│  Layer C: 仓位调节器（每日更新）                                │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Alt_exposure = 0.5 + 0.5·T          [50%~100%]   │           │
│  │ beta_port    = 成分加权估计 (B2方法)             │           │
│  │ hedge_ratio  = beta_port × (1−T) × Alt_exposure  │           │
│  │ hedge_ratio  = min(hedge_ratio, Alt_exposure×1.2)│           │
│  └──────────────────────────────────────────────────┘           │
│                          │                                       │
│                          ▼                                       │
│  Layer D: 回测/执行层                                           │
│  ┌──────────────────────────────────────────────────┐           │
│  │ P&L = Alt_exp·ret_alt − hedge·ret_btc            │           │
│  │     − 手续费 − 资金费率 + USDT闲置(0)            │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

### 2.1 设计核心原则

| 原则 | 说明 |
|------|------|
| **截面选股** | 用因子做 Alt 之间的相对强弱选股（Layer A） |
| **时序择时** | 用 BTC 趋势信号调节整体暴露与对冲（Layer B+C） |
| **Beta 中性化** | 对冲层把 Alt 组合的系统性 Beta 衰减到低水平 |
| **IS/OOS 隔离** | 因子筛选只用 IS 期数据，回测结果只看 OOS |
| **零未来函数** | 所有计算只依赖 `≤ t` 的数据 |

---

## 3. Layer A：Alt 选股层（截面）

### 3.1 IS/OOS 切分
| 阶段 | 区间 | 用途 |
|------|------|------|
| In-Sample | 2020-10-03 ~ 2023-12-31 | 筛选因子集、参数调优 |
| Out-of-Sample | 2024-01-01 ~ 2026-04-17 | 最终回测评估 |

### 3.2 因子筛选逻辑

```python
# 伪代码（在 IS 期间执行一次，固化结果）
for factor in all_35_factors:
    ic_series = cross_sectional_spearman_ic(factor, future_ret_1d, IS_dates)
    ic_ir = ic_series.mean() / ic_series.std()
    if abs(ic_ir) >= 0.05:
        # 若 IC_IR 为负，说明因子方向应反转
        direction = sign(ic_ir)
        selected.append((factor, direction))
```

预期保留约 **11-15 个因子**（基于之前探索性分析，有方向矫正后可能更多）。

### 3.3 合成流程（复用现有 Pipeline）
1. 加载筛选后因子 CSV
2. 横截面标准化（winsorize → rank → z-score）
3. 对称正交化（`orthogonalize=True`）
4. IC_IR 滚动加权（现有 `ic_ir_weighter.py`，`ic_window=20`）
5. 合成复合分数 `composite_score`
6. 每周（每 5 交易日）按 `composite_score` 取 Top 5 等权

### 3.4 持仓约束
- 持仓数：**5 只**
- 权重：等权（每只 20%）
- 换仓频率：每 5 交易日（周一）
- 成分必须在当日 universe 内（来自 `binance_coingecko_top100_marketcap_historical.csv`）

---

## 4. Layer B：BTC 趋势分数 T

### 4.1 公式定义

```python
# 每日 t 只使用 close[:t+1]
ema_short = EMA(close, span=50)
ema_long  = EMA(close, span=200)
ma_long   = MA(close, window=200)
std_long  = STD(close, window=120)

raw_ema_ratio = ema_short / ema_long - 1.0
raw_zscore    = (close - ma_long) / std_long

T1 = tanh(raw_ema_ratio / 0.15)   # 15% 偏离即饱和
T2 = tanh(raw_zscore / 2.0)        # 2σ 即饱和

T_raw = 0.5 * T1 + 0.5 * T2
T     = EMA(T_raw, span=5)         # 5 日 EMA 平滑
T     = clip(T, -1.0, +1.0)
```

### 4.2 参数表
| 参数 | 值 | 说明 |
|------|-----|------|
| `ema_short` | 50 | "快线" |
| `ema_long` | 200 | "慢线"，crypto 行业共识 |
| `zscore_window` | 120 | 约 6 个月 |
| `ema_ratio_saturation` | 0.15 | T1 饱和点 |
| `zscore_saturation` | 2.0 | T2 饱和点 |
| `smoothing_span` | 5 | 日线级平滑 |

### 4.3 未来函数检查（必做）
- ✅ `EMA`、`MA`、`STD` 默认后向窗口
- ✅ `tanh`、`clip` 为逐元素函数
- ✅ 禁止 `shift(-k)`、`rolling(center=True)`、`bfill`
- ✅ 提交前做 cutoff 动态反证：截取 `close[:cutoff]` 计算 T 与全量计算的差异应为 0

---

## 5. Layer C：仓位调节器

### 5.1 Alt 多头仓位
```
Alt_exposure_t = 0.5 + 0.5 × T_t          # 范围 [0.5, 1.0]
```
- T=+1（强牛）：100% Alt 多头
- T= 0（震荡）：75% Alt 多头
- T=−1（强熊）：50% Alt 多头，剩余 50% USDT

### 5.2 组合 Beta 估算（方法 B2：成分加权）
```python
# 每次换仓时计算（每周一次）
beta_port = mean([
    beta_60d(coin, btc)    # 各持仓币的 60 日历史 Beta
    for coin in current_holdings
])
```
- 未持仓的币不参与 Beta 计算
- 启动期（< 60 日历史）使用固定 `beta = 1.3` 作为先验

### 5.3 BTC 对冲仓位
```
hedge_ratio_raw = beta_port × Alt_exposure × (1.0 − T)
hedge_ratio     = min(hedge_ratio_raw, Alt_exposure × 1.2)
hedge_ratio     = max(hedge_ratio, 0)     # 不可转多
```
- T=+1：`hedge = 0`，不对冲
- T= 0：`hedge = beta_port × Alt_exposure`，市场中性
- T=−1：`hedge = min(2·beta × Alt_exp, 1.2·Alt_exp) = 1.2·Alt_exp`，过度对冲到上限

### 5.4 每日再平衡规则
- `Alt_exposure` 每日按信号连续调整（软调仓，当日平滑到目标，例如用 5% 的 EMA 平滑因子避免抖动，**避免每日 0.1% 的信号噪声产生大量交易**）
- `hedge_ratio` 每日按信号即时调整（合约再平衡成本低）
- `Alt 持仓明细`只在每周换仓日变更

---

## 6. Layer D：回测与成本建模

### 6.1 每日 P&L 公式
```
daily_return_t =
    Alt_exposure_t        × port_ret_t
  − hedge_ratio_t         × btc_ret_t
  − trading_fee_t         (仅换仓日/调仓日)
  − funding_fee_t
```

其中：
```
port_ret_t = Σ_i w_i × ret_i_t       # Alt 持仓当日加权收益
trading_fee_t = turnover_alt × 0.001 + turnover_btc_hedge × 0.0005
funding_fee_t = hedge_ratio_t × (0.1095 / 365)     # 每日资金费率
```

### 6.2 成本参数
| 项目 | 值 | 备注 |
|------|-----|------|
| Alt 现货手续费 | 0.10% / 边 | Binance Taker |
| BTC 永续手续费 | 0.05% / 边 | 永续费率 |
| 资金费率 | +10.95% / 年 = +0.03% / 天 | 固定假设，保守 |

### 6.3 手续费结算逻辑
- **Alt 换仓日**（每周一）：按 Alt 总换手率（含 `Alt_exposure` 变化和成分变化）收 10bps 单边
- **对冲调整日**（每天）：按 `|hedge_ratio_t − hedge_ratio_{t−1}|` 收 5bps 单边
- **资金费率**：每日按当天 `hedge_ratio_t` 线性扣除 `0.03%`

---

## 7. 模块与代码改动

### 7.1 新增/修改文件清单
```
portfolio/
├── combiner/
│   └── is_oos_filter.py           [NEW]
├── portfolio_builder/
│   ├── topn_equal.py              (保持不变)
│   └── beta_estimator.py          [NEW]
├── regime/                         [NEW 模块]
│   ├── __init__.py
│   └── btc_trend.py               [NEW]
├── sizing/                         [NEW 模块]
│   ├── __init__.py
│   └── exposure_controller.py     [NEW]
├── backtester/
│   ├── engine.py                  [MODIFY]
│   ├── fees.py                    [MODIFY]
│   └── funding.py                 [NEW]
└── config.py                      [MODIFY]
```

### 7.2 新增配置字段（`PortfolioConfig`）
```python
@dataclass
class PortfolioConfig:
    # 原有字段保持不变
    top_n: int = 5                          # [CHANGED default 10 -> 5]
    rebalance_period: int = 5               # [CHANGED default 1 -> 5, weekly]

    # IS/OOS
    is_end_date: str = "2023-12-31"         # [NEW]
    min_ic_ir: float = 0.05                 # [NEW]

    # Layer B: BTC Trend
    ema_short: int = 50                     # [NEW]
    ema_long: int = 200                     # [NEW]
    zscore_window: int = 120                # [NEW]
    ema_ratio_saturation: float = 0.15      # [NEW]
    zscore_saturation: float = 2.0          # [NEW]
    t_smoothing_span: int = 5               # [NEW]

    # Layer C: Exposure
    alt_max_exposure: float = 1.0           # [NEW]
    alt_min_exposure: float = 0.5           # [NEW]
    hedge_cap_multiplier: float = 1.2       # [NEW]
    beta_window: int = 60                   # [NEW]
    beta_prior: float = 1.3                 # [NEW]

    # Layer C: Exposure smoothing
    alt_exposure_ema_alpha: float = 0.05    # [NEW; Alt暴露每日向目标平滑的EMA alpha]

    # Layer D: Costs
    alt_fee_rate: float = 0.001             # [NEW; 0.1%/边 现货]
    perp_fee_rate: float = 0.0005           # [NEW; 0.05%/边 永续]
    funding_rate_annual: float = 0.1095     # [NEW; 资金费率年化]
```

### 7.3 模块职责
| 模块 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `is_oos_filter.py` | 在 IS 期间筛选 IC_IR ≥ 阈值的因子 + 自动方向矫正 | 所有因子面板 + IS 期 label | 因子名清单 + 方向 |
| `beta_estimator.py` | 60d 滚动计算各币 vs BTC 的 Beta；组合 Beta = 持仓等权平均 | kline + holdings | `beta_port` 时间序列 |
| `btc_trend.py` | 计算 BTC 趋势分数 T | BTC close | `T[date]` 时间序列 |
| `exposure_controller.py` | 把 T + Beta 转为 `(alt_exposure, hedge_ratio)` | T, beta_port | 每日仓位目标 |
| `funding.py` | 计算每日资金费率成本 | `hedge_ratio` | daily cost |
| `engine.py` | 修改：支持 hedge 分项、funding 扣减、双层换仓 | 以上全部 | daily portfolio return |

---

## 8. 验证方案（Definition of Done）

### 8.1 未来函数检查
- **静态**：grep 禁用模式（`shift(-`、`center=True`、`bfill`、`merge_asof.*forward`）
- **动态反证**：选取 2025-01-01 作为 cutoff，对比
  - 全量数据计算 `T`、`alt_exposure`、`hedge_ratio`、`composite_score`
  - 截断 `close[:cutoff]` 重新计算
  - 要求：`date ≤ cutoff` 区间，`max_abs_diff` ≤ 1e-9

### 8.2 性能目标
OOS 期间（2024-01-01 ~ 2026-04-17）：
- ✅ MaxDD ≤ 15%
- ✅ 年化收益 ≥ 20%
- ✅ Sharpe ≥ 1.2

### 8.3 稳健性检查（Bonus）
- 同期 BTC Buy&Hold 最大回撤（~35-40%）时，策略回撤 ≤ 15%
- 全样本（IS+OOS）MaxDD ≤ 20%
- 换手率：Alt 年化换手 ≤ 1500%（每周换仓，60% 换仓率上限）
- 策略在 IS 期的 Sharpe 不显著优于 OOS（避免样本内过拟合）

### 8.4 报告产出
- OOS 期：`portfolio/output/report_maxdd15_oos.html`
- 全样本：`portfolio/output/report_maxdd15_full.html`
- 对比图：策略净值 vs BTC 净值 vs 市场中性基准

---

## 9. 风险与限制

| 风险 | 缓解措施 |
|------|---------|
| **趋势信号滞后** | 牛转熊时 T 下降有延迟，可能造成 5-10% 额外回撤。实测若超阈则加入第二层熔断 |
| **假信号频繁切换** | 5 日 EMA 平滑 + 连续暴露映射已降噪，但震荡市仍可能 whipsaw |
| **资金费率波动** | 固定 10.95% 假设在极端牛市可能偏低（实际 50%+）；OOS 实测若偏差大，改用动态 funding API |
| **Top 5 集中风险** | 单币暴雷（如 LUNA 事件）对 20% 权重冲击大；后续可加 `single_name_cap = 25%` + 流动性筛选 |
| **OOS 样本较短** | 仅 2.3 年 OOS，未完整覆盖一轮牛熊；若后续积累更多数据，重新验证 |
| **IS 筛选可能过拟合** | IS 用 3.25 年筛 11-15 个因子，过拟合风险存在；做因子 bootstrap 稳健性测试 |

---

## 10. 交付阶段与顺序

1. **阶段 1**：IS/OOS 因子筛选（`is_oos_filter.py` + 探索脚本）→ 固化因子清单
2. **阶段 2**：Layer B（`btc_trend.py`）→ 单元测试 + 未来函数反证
3. **阶段 3**：Layer C（`beta_estimator.py` + `exposure_controller.py`）
4. **阶段 4**：Layer D 引擎扩展（`engine.py` + `funding.py` + `fees.py`）
5. **阶段 5**：端到端 Pipeline 整合 + OOS 回测
6. **阶段 6**：参数敏感性测试（EMA窗口、T平滑、对冲上限）
7. **阶段 7**：若 MaxDD 未达标，叠加二层熔断（drawdown circuit breaker）

---

## 附录 A：IS 期因子预筛（预估）

基于探索性分析（存在全样本偏差，仅供参考量级）：
```
IC_IR ≥ 0.05 的因子（预估 11 个）：
  apb_factor_3d                            0.189
  return_skew_reversal_20d                 0.135
  intraday_accumulation_distribution_20d   0.116
  taker_buy_quote_extreme_efficiency_20d   0.057
  momentum_takerbuyquote_extreme_20d       0.055
  ats_price_divergence_20d                 0.052
  smart_money_accum_divergence_20d         0.049
  momentum_volume_extreme_20d              0.041
  volume_stability_20d                     0.040

|IC_IR| ≥ 0.05 含方向反转的因子（新增 18 个）：
  retail_friction_illiquidity_20d         -0.238 → 方向反转
  price_path_efficiency_10d               -0.169 → 方向反转
  n_day_momentum_1d                       -0.157 → 方向反转
  ... (共 18 个反向因子)

实际 IS 期重新计算后清单会略有不同。
```

## 附录 B：参数稳健性测试计划

| 参数 | 默认 | 敏感性扫描范围 |
|------|------|--------------|
| `ema_short / ema_long` | 50/200 | (30/100), (50/200), (60/250) |
| `zscore_window` | 120 | 60, 120, 180 |
| `hedge_cap_multiplier` | 1.2 | 1.0, 1.2, 1.5 |
| `top_n` | 5 | 3, 5, 10 |
| `alt_min_exposure` | 0.5 | 0.3, 0.5, 0.7 |

扫描规则：每次只改一个参数，观察 OOS MaxDD 和 Sharpe 是否对该参数敏感；若某参数敏感性过高，说明过拟合风险大。

---

**End of Design**
