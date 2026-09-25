# Daily Operators — Genetic_Algorithm 日频算子目录与先验

> 本仓库无分钟数据，原 skill 的日内算子目录不适用。本文件改为日频口径，
> 来源：`Genetic_Algorithm/operators.py` + `features.py`（日频 GP 搜索的
> 算子与终端特征集）。
>
> **定位：假设生成的参考词表，不是免评估通行证。** GA 侧有自己的
> train 2024 / validation 2025 / frozen test 2026 三段管线与质量门
> （见 `Genetic_Algorithm/README.md`）；GA 跑出来的幸存公式只能说明
> "这个模式在日频数据上活过了成本门"，任何手工因子假设照常走 7 步流程。

## 1. 终端特征（GP 的叶子节点，可直接当因子子信号）

原始字段（`RAW_FIELDS`，与 signal-contract.md 可得字段一致）：
`open / high / low / close / volume / quote_volume / trade_count /
taker_buy_base_volume / taker_buy_quote_volume`

衍生字段（`DERIVED_FIELDS`，因果构造，括号内为依赖窗口）：

| 字段 | 定义 | 语义 |
|---|---|---|
| `return_1d` | 日收益（1d lag） | 反转/动量载体 |
| `range_relative` | (H-L)/C | 日内振幅（range_level 语义） |
| `body_relative` | (C-O)/O | K 线实体方向 |
| `volume_relative_20` / `quote_volume_relative_20` | 量/额对 20 日均值比 | 放量/缩量（volume_surge/dryup） |
| `taker_base_ratio` / `taker_quote_ratio` | 主动买占比（币/额口径） | taker_imbalance |
| `quote_per_trade` | quote_volume / trade_count | 单笔均额（trade_size_shift） |

特征族（`FEATURE_FAMILIES`，组合/去重时的正交性参考）：
`price` / `volatility` / `activity` / `buy_flow`。

## 2. 算子目录（作用于 date × instrument 面板）

| 类别 | 算子 |
|---|---|
| 逐元素算术 | `add / subtract / multiply / safe_div / negate / absolute` |
| 时序 | `lag(a, w)`、`delta(a, w)` |
| 滚动窗口（trailing，无未来） | `rolling_mean / rolling_std / rolling_min / rolling_max(a, w)` |
| 双变量 | `rolling_correlation(a, b, w)` |
| 截面 | `cross_sectional_rank(a)`、`cross_sectional_zscore(a)` |

扩展算子（2026-09 新增 20 个，全部 trailing 窗口、min_periods=window，按族分组）：

| 族 | 算子 | 说明 |
|---|---|---|
| 稳健统计 | `rolling_median / rolling_mad / rolling_iqr / rolling_downside_std` | 中位数、绝对中位差、四分位距、下行波动（负值 <2 个时 NaN）；输出量纲=preserve |
| 分布形状 | `rolling_skew`、`rolling_kurt`、`rolling_autocorr` | **最小窗口 20**（`MIN_OPERATOR_WINDOW`，小于 20 直接被校验拒绝）；输出 ratio |
| 窗口位置/时间 | `ts_rank / ts_zscore / ts_argmax / ts_argmin` | 当前值在窗口内的分位、时序 z-score、距窗口极值的归一化时间（0=今天即极值）；输出 ratio |
| 趋势质量 | `rolling_slope_norm / rolling_r2 / efficiency_ratio` | 归一化斜率、趋势 R²、|净位移|/路径长度；输出 ratio |
| 路径符号 | `rolling_hit_rate` | 窗口内正值占比；输出 ratio |
| 波动结构与极值路径 | `rolling_cv / rolling_max_drawdown` | 变异系数、窗口最大回撤（running max ≤0 时 NaN，语义上针对价格类正数序列）；输出 ratio |
| 非线性变换 | `signed_sqrt / safe_log` | sign 保持的 sqrt / log1p 压缩；输出量纲 mixed（只能再接 preserve 类一元算子或 rank/zscore 输出层，不能再进二元算子） |
| 截面 | `cross_sectional_zscore(a)` | 与 rank 同款掩码逻辑，截面 (x−μ)/σ，当日有效截面数 ≤1 时 NaN；输出 ratio |

这套算子刻意保持小：GP 经验是**简单可解释的窗口统计 + 截面 rank** 足以覆盖
大多数日频模式；复杂结构（条件筛选、分组压缩）在日频 top50 截面上的自由度
不值得过拟合风险。扩展族仍遵循同一原则：全部为无状态的窗口统计/压缩，
不引入条件分支。

## 2b. 量纲纪律（dimension discipline）

GP 层在 `expression.validate_tree` 自底向上强制量纲代数（实现见
`Genetic_Algorithm/operators.py` 的 `TERMINAL_DIMENSIONS` /
`combine_dimensions`），**不同量纲字段禁止直接加减乘除**，非法树在生成、
交叉、变异三条路径都会被拒绝。手工因子同理。

终端字段量纲：
`price`(open/high/low/close)、`base_volume`(volume/taker_buy_base_volume)、
`quote_volume`(quote_volume/taker_buy_quote_volume)、`count`(trade_count)、
`money_per_trade`(quote_per_trade)、`ratio`(全部比率类衍生字段)。

组合规则：
- `add / subtract`：仅同量纲合法（price+volume 直接拒绝）
- `multiply`：`ratio × X → X`；其余 → mixed
- `safe_div`：同量纲相除 → ratio；`X / ratio → X`；仅在商有明确经济含义时
  保留具名量纲：`quote_volume / base_volume = price`、
  `quote_volume / count = money_per_trade`（单笔均额）、
  `base_volume / count = base_per_trade`；其余 → mixed
- mixed 只能进 preserve 类一元算子（negate/abs/lag/delta/rolling_* 统计）或
  rank/zscore 输出层，**禁止再进任何二元算子**

手工因子的推论：**跨字段组合必须先各自无量纲化**（`ts_zscore` / `ts_rank` /
截面 rank / 比率化），再组合；除法仅在商有明确经济含义时允许
（quote/base=price、quote/count=单笔均额、同量纲相比=ratio）。

## 3. 常见模式（日频手工因子的高产结构）

| 模式 | 例子 | 对应语义 |
|---|---|---|
| 比率水平 | `safe_div(rolling_std(return_1d, 20), rolling_mean(quote_volume, 20))` | illiquidity_level / 单位量能波动 |
| 短长窗对比 | `rolling_mean(x, 5) / rolling_mean(x, 20)` | 变化类 event（spike/surge） |
| 符号化流量 | `rolling_mean(taker_quote_ratio * sign(return_1d), N)` | chase_flow / taker 方向确认 |
| 截面化 | `cross_sectional_rank(任意上述)` | 输出层，框架也有 `mad_rank` preprocessing |
| 时序平滑 | `rolling_mean(signal, 5~20)` | 压换手的第一手段（trading_net 口径下成本敏感） |

## 4. 使用纪律

- ITER_NOTE 引用 GA 侧证据时写清来源（run 目录、阶段），但 GA 的
  validation 段（2025）属于本 skill 的 IS 段，可以自由引用；GA 的
  frozen test（2026+）= 本 skill 的 OOS，**冻结段结果不可引用**
- GA 导出的因子（`export` 落到 `factor_analyse/factor_mining/`）入本流程照样
  建档：ITER_NOTE + SEMANTIC_PLAN + 相关性门控，一步不少
- 同一模式的换皮（换字段对但机制不变）触发规则 6/7 审查：semantic_key 不变
  = 同一假设，不允许当新因子刷轮次
- GA 幸存公式普遍带平滑外壳——本 skill 单点假设原则要求**拆开试**：
  先裸信号验证方向，再逐轮加平滑，不要一轮全上
