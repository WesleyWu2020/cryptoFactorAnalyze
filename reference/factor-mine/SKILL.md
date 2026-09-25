---
name: factor-mine
description: "Use when iterating on crypto quant factors in this repo - creating a new factor, modifying an existing factor's parameters/windows, or running disciplined factor-mining experiments. Enforces one-hypothesis-per-iteration (ITER_NOTE), the acceptance gates defined in this skill (no separate EVALUATION.md exists here), train/test split discipline (mine on IS 2024-2025 only, one-shot OOS 2026+ validation gate before accepting a new best, no OOS peeking), HTML report diagnosis, correlation gating against the factor_mining factor base, accept/rollback decisions, identity-drift re-registration, SEMANTIC_PLAN with a controlled schema vocabulary for new/composed factors, failure classification (implementation vs statistical vs redundancy vs OOS), and guards against degenerate convergence to mono-field beta signals (prefer composing orthogonal fields)."
---

# Factor Mine（本仓库适配版）

> 改编自 [quantskills/skill-factor-mine](https://github.com/quantskills/skill-factor-mine)，
> 已映射到本仓库的 `factor_common` 基础设施：Binance USDT 永续**日频**面板
> （`data/crypto_quant.h5`，2023-07-05 起），universe 为 point-in-time CMC top50
> （`historical_top50`）。本仓库无分钟数据、无 OI/基差/标记价信号字段，
> 无 `eval_factor.py` / `EVALUATION.md` / `--prod-corr`——对应能力由本 skill
> 内置口径替代（见"接口映射"）。

## 核心规则

1. **单点假设原则**：每轮只改一件事（见 `references/op-types.md`）
2. **必先读口径**：本 skill 的"接受门槛"一节是评分公式、固定口径与接受门槛的
   唯一真源（本仓库没有独立 EVALUATION.md）——没读就不许提假设；
   指标口径细节（复合收益、null 语义、样本切分 purge 规则）以
   `docs/factor_common.md` 为准
3. **ITER_NOTE 强制**：`op_type / hypothesis / change / expected` 四字段必填，缺一不跑
   （模板：`references/iter-note.md`）
4. **信号契约**：因子文件必须符合 `factor_common` loader 规范（TYPE/META/SETTING/
   calc_factor，`META.factor_name` 等于文件名），无未来函数——静态扫描 +
   `check_cutoff` 动态回放双门禁（细则：`references/signal-contract.md`）
5. **相关性门控**：新因子/改因子必须与 `factor_analyse/factor_mining/` 已有因子的
   缓存值（`data/factor_results/*.parquet`）算逐日截面 rank 相关，
   max |ρ| ≥ 0.85 直接拒（细则：`references/correlation-gate.md`）
6. **身份漂移即重建档**：累计改动若改变了输入字段集 / category / 经济假设
   （如从价格动量一路迭代成 funding carry），它已是另一个因子——必须改名、按
   `add_factor` 重新建档（新 ITER_NOTE + 重跑相关性门控），禁止以"迭代"名义偷渡身份。
   有 SEMANTIC_PLAN 的因子以 semantic_key 为结构化判据：任一五维字段实质变化
   = semantic mutation（ITER_NOTE 记录），累计 mutation 改变核心机制即按本条重建档
7. **拒绝退化收敛**：迭代轨迹若是一路"删成分 → 全样本指标变好"，终点收敛成
   "单一原始字段 + 平滑"的裸信号，这就是全样本选择过拟合。单字段终点必须满足：
   ITER_NOTE 显式论证经济机制 + 与库内同字段因子相关性有增量；否则优先走
   `compose_factor`（组合 ≥2 个正交字段/族的子信号，见 `references/op-types.md`）。
   有 SEMANTIC_PLAN 的因子："删掉某个 context/quality 成分"也算 semantic diff，
   触发本条审查
8. **语义计划前置**：`add_factor` / `compose_factor` 在写代码前必须先建
   SEMANTIC_PLAN（五维 id 只能取自 `references/schema-vocabulary.md`，模板与
   继承规则见 `references/iter-note.md`）；`modify_factor` 默认继承原 plan，
   仅当五维字段实质变化时在 ITER_NOTE 记 `semantic_mutation`
9. **训练/测试集隔离**：所有挖掘决策（假设、改参、accept/reject）只在 IS 段
   （2024-01-01 ~ 2025-12-31）上做；IS 过门槛后才跑一次 OOS 段
   （2026-01-01 ~ 最新存储日期）做验证，通过才成为新 best、才允许进入下一轮。
   OOS 结果禁止用于设计下一轮假设或调参；同一谱系 OOS_REJECT ≥3 次即冻结
   （细则与红线：`references/is-oos-protocol.md`）。切分与
   `Genetic_Algorithm` 的 train 2024 / validation 2025 / frozen test 2026 约定一致

## 接受门槛（本仓库唯一口径真源）

主决策口径：**`trading_net` 场景**（fee 0.0005 + slippage 0.001，每边，固定不可调低）。
`all_costs` 场景当前已可认证（funding 覆盖由 `build_funding_schedule` 的
数据派生结算时刻表支撑，panel 全量 `complete`），作为**第二道成本参考门**：
trading_net 全过但 all_costs 年化 ≤0 的因子视为成本敏感，须在 ITER_NOTE 论证。
注意该时刻表是从观测事件反推的（非交易所官方日程），强度有限；若未来面板
退回 `unknown` 覆盖，all_costs 指标变 null 时不进门槛、只看 trading_net。
执行口径固定：信号日 t → t+1 开盘成交，`rebalance_days=1`，**分组数 `n_groups=5`**
（2026-09-13 起由 10 改为 5：top50 宇宙下 10 分组每组仅 ~5 币，尾部噪声大；
5 分组更贴近可交易持仓数。门槛表不变，历史 10 分组结论不作 retroactive 重判，
新口径下接受的因子以 5 分组 IS+OOS 为准）。

### IS 硬门槛（任一 FAIL 即 REJECTED）

| 指标 | 路径（`result["factor_performance"]` 下） | 门槛 |
|---|---|---|
| RankIC | `samples.<is>.ic.rank_ic_mean` | \|·\| ≥ 0.02，且符号与 `factor_direction` 一致 |
| RankICIR | `samples.<is>.ic.icir`（未年化 mean/std） | \|·\| ≥ 0.10 |
| t 统计 | `samples.<is>.ic.t_stat` | \|·\| ≥ 2 |
| 多空年化 | `scenarios.trading_net.<is>.annual_return` | > 0 |
| 多空 Sharpe | `scenarios.trading_net.<is>.sharpe` | ≥ 1 |
| 最大回撤 | `scenarios.trading_net.<is>.max_drawdown` | ≤ 0.25 |
| 换手 | `scenarios.trading_net.<is>.turnover` | ≤ 1.0 |
| 覆盖率 | `samples.<is>.coverage` | ≥ 0.6 |
| 相关性 | 与库内因子 max \|ρ\|（correlation-gate.md） | < 0.85（0.6~0.85 需 ITER_NOTE 论证） |

`<is>` 指 IS 窗口评估的 `full` 样本块（IS-only 运行时 `full` 即 IS 段）。
指标缺失（null）一律视为 FAIL，不许当 0 处理。OOS 验证门见 is-oos-protocol.md §3。

### 成本磨损分支（调仓周期 horizon）

主口径仍是 `rebalance_days=1`。但当 IS 诊断显示**信号有毛 alpha、被换手成本
磨损**——gross 年化 > 0 而 trading_net 年化 ≤ 0 或 net sharpe < 1，且
turnover > 1.0（或成本拖累吃掉 gross 大头）——允许对**同一因子文件（不改
代码）**在 `rebalance_days ∈ {3, 5}` 下重做 IS 评估：

- horizon 只能取先验档位 {1, 3, 5}，禁止扫连续值挑最优（防口径过拟合）；
- 某 horizon 的 IS 硬门槛（同一张表，trading_net 口径）全部通过，才算该
  horizon 通过；OOS 验证必须在同一 horizon 做，且与主口径**共享同一谱系的
  OOS 预算**（horizon 变体不重置计数）；
- 换手门槛随 horizon 自然放松（持仓期拉长日换手下降），其余门槛不动；
- 每个 horizon 的评估独立记入语义日志，`conclusion` 注明口径（如 "rb=3"）；
- 接受后因子档案需注明可交易 horizon（ITER_NOTE 补 `accepted_rebalance_days`）；
- {1, 3, 5} 全失败 → 该信号在当前成本口径下不可交易，关闭分支，禁止再拉更长。

## 接口映射（已确认，直接用）

| skill 概念 | 本地实现 |
|---|---|
| 评分口径真源 | 本 skill"接受门槛"一节 + `docs/factor_common.md`（指标定义） |
| 评估命令（IS，每轮迭代决策用） | `./.venv/bin/python factor_analyse/main.py <factor_path> 1 --groups 5 --start 2024-01-01 --end 2025-12-31`（成本磨损分支下 `1` 换成 `3` 或 `5`） |
| 评估命令（OOS 验证 + 全样本归档，IS 过后一次性） | `FactorManager().evaluate(path, params={"start": "2024-01-01", "end": <最新存储日期>, "rebalance_days": 1, "n_groups": 5, "split_date": "2026-01-01"})`，读 `samples.out_of_sample` / `scenarios.trading_net.out_of_sample`；成本磨损分支下 `rebalance_days` 用被验证的同一 horizon |
| IS/OOS 切分与验证门 | `references/is-oos-protocol.md`（切分固定、OOS 验证门、防污染红线） |
| 交互式评估 | `factor_analyse/factor_common_usage.ipynb`，或 `FactorManager().evaluate(..., plot=True)`，口径与 CLI 一致 |
| 因子文件接口 | 标准 `.py`：`TYPE="regular"` / `META` / `SETTING` / `calc_factor(data_ctx)`，仅日频（见 `docs/factor_common.md`"Factor module contract"） |
| 因子脚手架 | `FactorManager().create_template('<name>')`（不覆盖已有文件），参考 `factor_analyse/factor_mining/example_momentum.py` |
| 相关性门控 | 自算：`data/factor_results/<factor_id>.parquet` 缓存值逐日截面 rank 相关（细则与代码：`references/correlation-gate.md`）；聚类选代表可用 `portfolio/factor_pool/cluster_selector.py` |
| 批量基线 | `scripts/batch_evaluate_factors.py`（全库统一参数批评，结果落 `outputs/`） |
| 实验/语义日志 | `outputs/factor_mine_semantic_log.jsonl`（手工追加，格式：`references/semantic-memory.md`） |
| 因子存放 | 研究中：`factor_analyse/factor_mining/`；存量批量研究因子在 `factor_analyse/unsubmit/` |
| 图表诊断 | 自包含 HTML 报告（`reports/<factor>.html`）：NAV/分组收益/IC 衰减/换手/覆盖率逐块诊断 |
| 数据真源 | `data/crypto_quant.h5`；可用字段：`open, high, low, close, volume, quote_volume, trade_count, taker_buy_base_volume, taker_buy_quote_volume, funding`（细则：`references/factor-families.md`） |

## 工作流（每轮 7 步）

```
1. 读本 skill"接受门槛" + 当前因子文件全文
2. 形成单点假设 → 写 ITER_NOTE（写进因子文件顶部，四字段必填；
   hypothesis/reasoning 的证据只能来自 IS 段评估或先验机制）
3. 改代码（≤ 一个 op_type）
4. 跑 IS 评估（main.py --start 2024-01-01 --end 2025-12-31；add/modify/compose
   必做相关性门控；保留 HTML 报告）
   → IS 任一硬门槛 FAIL：本轮 REJECTED，直接进第 6 步，不看 OOS
   → IS 全过：跑 OOS 验证（带 split_date="2026-01-01" 的全窗口 evaluate），
     按 is-oos-protocol.md §3 第二级验证门判定（同号 / 衰减 ≤50% / OOS 年化为正）
5. 读评估结果：对照门槛表 + 逐块诊断 HTML 报告
   （决策用 IS 报告；OOS 切片只回答"要不要成为新 best"）
6. ACCEPTED（IS 过 + OOS 验证过）→ 保留改动，
   ITER_NOTE 补记 oos_result，可进入下一轮
   REJECTED / CRASH → 回滚到上一最优版本（git checkout 或恢复备份），记录失败原因，
   并按结果归类失败类型：
   - IMPLEMENTATION_INVALID：crash / 信号契约违反 / 未来函数（静态扫描或
     check_cutoff 不通过）/ NaN-inf
     → 只说明本次实现不合格，**不等于 hypothesis 无 alpha**，修实现可重试
   - STATISTICAL_REJECT：实现有效但 IS 统计门槛 FAIL（RankIC/ICIR/sharpe/
     turnover/maxdd/coverage）
     → hypothesis 被 IS 数据与口径否定
   - REDUNDANCY_REJECT：max |ρ| ≥ 0.85 → 有 alpha 但无增量，与统计否定分开记
   - OOS_REJECT：IS 过门槛但 OOS 验证失败 → IS 段过拟合信号，回滚；
     同谱系累计 ≥3 次冻结该方向，转 explore
7. 追加语义日志：向 outputs/factor_mine_semantic_log.jsonl 追加一行结构化记录
   （semantic_key / search_mode / decision_class / conclusion，格式见
   references/semantic-memory.md）——ACCEPTED 和 REJECTED 都要记，缺一不算完成
```

### 轨迹审查（每 ~5 轮、或每次想"删一个成分"时必做）

- 因子身份还与起点一致吗（输入字段集 / category / 经济假设）？不一致 → 按规则 6
  改名重建档，不要继续假装在"迭代"原因子
- 近几轮是否全是"删成分 → 指标变好"？是 → 警惕全样本贪心选择：用 IS 段
  （2024~2025）复核删减决策，并保留多成分版本作对照评估，再决定接受哪个终点
- 终点是否已是单字段裸信号？是 → 按规则 7 补机制论证 + 同字段因子相关性，
  或改走 `compose_factor` 把正交字段组合回来
- 翻 outputs/factor_mine_semantic_log.jsonl 回答 semantic-memory.md 的四问：
  研究区域是否过度集中？词表哪些区域没碰过？同一 semantic_key 是否已失败
  ≥2 次（是则降权）？哪些 schema 反复触发相关性 WARN？
- 当前谱系的 OOS_REJECT 累计几次了？≥3 次 → 按规则 9 冻结该谱系，
  不要再用"微调再过一次 OOS"消耗测试集

## 按需加载（references/）

| 何时读 | 文件 |
|---|---|
| 跑评估前确认 IS/OOS 口径、OOS 验证门、防污染红线 | `references/is-oos-protocol.md` |
| 准备改代码、不确定单点假设范围 | `references/op-types.md` |
| 写 ITER_NOTE 不知道字段含义 | `references/iter-note.md` |
| 因子文件规范 / 未来函数 / 可用字段 | `references/signal-contract.md` |
| 相关性门控触发警告或拒绝 | `references/correlation-gate.md` |
| 想挖新因子、看本地可用数据和因子族分布 | `references/factor-families.md` |
| 找假设灵感：日频算子目录、GA 幸存公式先验、落地映射 | `references/daily-operators.md` |
| 建 SEMANTIC_PLAN 选词、算 semantic_key、加新词条 | `references/schema-vocabulary.md` |
| 写语义日志、轨迹审查四问、decision_class 定义 | `references/semantic-memory.md` |
| 常见坑（未来函数、过拟合、换皮、身份漂移、退化收敛） | `references/anti-patterns.md` |

## QA 检查清单（提交前）

- [ ] 只改了一件事（一个 op_type）？
- [ ] add/compose 已建 SEMANTIC_PLAN，五维 id 均取自 schema-vocabulary.md？
- [ ] semantic_key 未变；或变化已记 semantic_mutation，且未构成身份漂移（构成则已改名重建档）？
- [ ] ITER_NOTE 四个必填字段都有内容，且写进了因子文件顶部？
- [ ] hypothesis 有具体经济/统计含义，不是"试试看"？
- [ ] expected 给了具体指标区间（不是"提升"）？
- [ ] 因子身份未漂移？（字段集 / category / 经济假设与起点一致；已漂移则已改名并按 add_factor 重建档、重跑相关性门控）
- [ ] 终点不是无论证的单字段裸信号？（是单字段则 ITER_NOTE 有机制论证 + 与库内同字段因子相关性增量）
- [ ] 删成分的决策已用 IS 段复核，且有多成分对照版本，不是全样本贪心？
- [ ] 本轮假设与决策只用了 IS（2024~2025）证据，没有引用 OOS 段结果？
- [ ] IS 过门槛后才跑 OOS 验证，且 OOS 只做了一次（不是调到过为止）？
- [ ] ACCEPTED 判定 = IS 全门槛 PASS + OOS 验证门通过？
- [ ] OOS_REJECT 已回滚、已计入该谱系验证预算（≥3 次则冻结方向）？
- [ ] 评估 status 不是 crash；未来函数静态扫描 + check_cutoff 回放均通过？
- [ ] 相关性门控 max |ρ| < 0.85（0.6~0.85 已在 ITER_NOTE 论证）？
- [ ] 已逐块诊断 HTML 报告（NAV / 分组单调性 / IC 衰减 / 换手 / 覆盖率）？
- [ ] 成本口径未动（fee 0.0005 + slippage 0.001），决策用 trading_net？
- [ ] REJECTED 已归类失败类型（IMPLEMENTATION_INVALID / STATISTICAL_REJECT / REDUNDANCY_REJECT / OOS_REJECT），未把 crash 当作 hypothesis 被否定？
- [ ] 语义日志已追加到 outputs/factor_mine_semantic_log.jsonl，且 conclusion 写清了"学到什么"（不是只写 REJECTED）？

## 边界与合规

- 本 skill 只管**研究迭代纪律**；因子入组合池另走 `portfolio/` 流程
  （cluster_selector / combiner 等），不在本 skill 范围
- 评估分数仅反映历史数据 + 固定口径下的统计表现，不代表未来表现
- 成本口径固定 fee=0.0005、slippage=0.001，禁止调低成本美化结果；IS/OOS 切分
  与验证纪律按 `references/is-oos-protocol.md`（切分点先验固定，禁止按结果挪动）
- 本仓库数据为日频（2023-07-05 起），样本期短于原 skill；对 2024 年才有的
  完整截面年份之外的结论保持保守
