# Daily genetic-algorithm workflow

This package searches causal daily cross-sectional formulas.  A formula at day
`t` may use only data available at `t` or earlier: rolling windows are trailing,
membership is point-in-time, and labels are never inputs to the formula.
Training is 2024, validation is 2025, and test is the frozen 2026 holdout.

Run each stage explicitly; `search` and `validate` never invoke `test`.

```bash
./.venv/bin/python -m Genetic_Algorithm audit --stage train --h5 data/crypto_quant.h5 --output Genetic_Algorithm/runs/audit_train.json
./.venv/bin/python -m Genetic_Algorithm search --config Genetic_Algorithm/configs/default.json --h5 data/crypto_quant.h5 --run-dir Genetic_Algorithm/runs/daily_001
./.venv/bin/python -m Genetic_Algorithm validate --run-dir Genetic_Algorithm/runs/daily_001 --h5 data/crypto_quant.h5
./.venv/bin/python -m Genetic_Algorithm freeze --run-dir Genetic_Algorithm/runs/daily_001
./.venv/bin/python -m Genetic_Algorithm test --manifest Genetic_Algorithm/runs/daily_001/frozen.json --h5 data/crypto_quant.h5
./.venv/bin/python -m Genetic_Algorithm export --manifest Genetic_Algorithm/runs/daily_001/frozen.json --output-dir factor_analyse/factor_mining
```

`configs/smoke.json` is a deliberately small smoke configuration; it confirms
the pipeline, not factor quality.  A successful pipeline is not evidence of a
useful factor.  `no_candidates` is a successful, informative outcome when all
candidates fail quality, novelty, or validation gates.

## Data, costs, and quality

Prices and volumes are daily Binance panels.  Quality eligibility excludes
unknown membership, placeholders, and incomplete klines.  Replay uses the
standard `perp_1d` profile: costs are fractions of gross notional (fee `0.0005`
and slippage `0.001` per side) and funding is enabled in strict mode.  Metrics
come from the certified `all_costs` ledger; an incomplete ledger is a failure,
not a zero return.

## Artifacts and reproduction

The run directory contains `config.json`, `audit_train.json`, `provenance.json`,
`training_candidates.json`, `generations.jsonl`, `deduplication.json`,
`validation.json`, and `frozen.json`.  Replay wrappers and their reports stay
under that run directory.  Exports alone are written to `factor_analyse/factor_mining`.
Test receipts are immutable evidence of holdout access; do not repeatedly use a
holdout as an iterative tuning target.

`training_candidates.json` holds training-only ASTs and diagnostics;
`deduplication.json` records novelty comparisons; `validation.json` holds stage
results and gate decisions; `frozen.json` binds selected ASTs, stage
fingerprints, configuration, profile, selection record, and runtime hashes.
To reproduce a frozen run, retain its H5 snapshot and run `test` or `export`
against that exact verified `frozen.json`; changed runtime hashes intentionally
block a test replay.

Exported factor modules depend on `Genetic_Algorithm.evaluator`,
`Genetic_Algorithm.expression`, and the normal `factor_common` point-in-time
context (including `__eligible__`).

## 稳健训练模式

`configs/daily_5groups_no_reports.json` 默认启用 `fitness_mode=robust_ic`；
`default.json`、`smoke.json` 及未指定该字段的旧配置仍使用 `legacy_ic`。
现有 `tmp/run_full_daily_gp.sh` 默认读取五分组配置，因此下次新运行会启用新评分。

稳健模式仅使用 2024 年：Q1 的平均 Rank IC 确定一次方向，Q2–Q4
沿用该方向。每个季度剔除标签退出日期超过季度末的样本；默认下一日
开盘交易、持有一天，因此剔除季度最后两天的标签。

设 Q2–Q4 的有向季度平均 Rank IC 为 `q`，节点数为 `n`：

```text
robust_ic = median(q) - 0.5 * std(q)
            - 1.0 * max(0, -min(q)) - 0.001 * n
score = (robust_ic, min(q), -n)
```

三个权重分别为 `stability_penalty`、`worst_quarter_penalty`、
`complexity_penalty`，是初始研究参数，并未根据 2026 收益拟合。
仍采用 Pareto 选择及相关性去重，保留覆盖率和每季度有效天数门槛；
另外要求稳健评分大于零、Q2–Q4 至少两个季度 IC 为正。
`score[0]` 在该模式下是稳健 IC，而非全年平均 IC；`score[1]` 是 Q2–Q4 最差 IC。

这是训练期内的时间分段稳定性目标，三个后续季度会反复参与进化，
不属于独立样本外或完整的逐折重新训练 walk-forward。
训练数据没有完整资金费率与持仓账本，因此训练评分不是净收益 Sharpe。
2025 正式回测仍用完整 `all_costs` 账本要求 `total_return > 0` 且
`Sharpe > 1`；2026 仅在显式请求时测试。已经查看的 2026 不能用于宣称
新方案通过未见数据检验。此修改不会自动重跑或修改历史 run。

因子表达式仅处理当日及历史特征；未来收益只用于评分。
相关防前视回归测试包括季度标签尾部污染、方向固定和表达式截断一致性。

## 全成本 Sharpe 训练与逐折重新训练

新配置 `configs/daily_5groups_all_costs.json` 使用 `all_costs_sharpe`。
它替代 IC 作为进化主目标，旧配置仍可用于复现 IC 实验。
五分组、每日调仓、手续费 0.0005、滑点 0.001、严格资金费率、
单位总敞口与正式 `replay` 一致。直接复用 `factor_common.run_backtest`
及账本指标计算；资金费率只用于评分，绝不加入表达式特征。

训练窗口必须由连续四个完整季度组成，可跨自然年。首季度的 Rank IC
固定方向；后三季度分别回测，在各季度内部清仓。IC 仅用于方向校准及
覆盖率初筛，不再要求季度 IC 为正。对每个通过初筛的唯一公式执行完整
成本评估（由搜索缓存避免重复公式回测），结果直接参与当代和后续亲代选择。

```text
S = 后三个季度的 all_costs Sharpe
fitness = median(S) - 0.5 * std(S) - 1.0 * max(0, -min(S))
          - 1.0 * max(季度最大回撤) - 0.001 * 节点数
```

权重是可配置的初始研究参数，尚未证明最优。合格输出还要求全年训练
`all_costs total_return > 0`、`Sharpe > 1`、后三季度多数净盈利、fitness > 0。
全年指标包含方向校准季度，只是训练内验收，不能被称为样本外证据。
不完整资金费率账本、空账本、未清仓或无效 Sharpe 不允许按零成本或零收益替代。
有限但未过门槛的评分用于探索；输出只包含达标候选。优先按是否达标、
fitness、最差季度 Sharpe 排序，并保留相关性去重。

`walk-forward` 默认执行以下四折，每折从头独立搜索，缓存不跨折复用：

| 训练窗口 | 独立验证 | 冻结后测试窗口 |
|---|---|---|
| 2023 Q4–2024 Q3 | 2024 Q4 | 2025 Q1 |
| 2024 Q1–Q4 | 2025 Q1 | 2025 Q2 |
| 2024 Q2–2025 Q1 | 2025 Q2 | 2025 Q3 |
| 2024 Q3–2025 Q2 | 2025 Q3 | 2025 Q4 |

训练去重后的至多 20 个候选先写入不可覆盖的 `shortlist.json`，再读取独立
验证季度。验证仅作 all_costs 收益 > 0、Sharpe > 1 的门槛，不反馈给进化、
方向或排名。通过者仍按训练排名选择，并检查训练期净收益 Pearson 相关性
及同方向持仓重叠，默认达到 0.8 或 0.7 即拒绝重复候选。
持仓重叠按每日同方向共同持仓数的两倍除以双方持仓总数计算，再对活跃日
取均值；不用不同币种的原始数量比较。检查仅使用训练数据。

允许下一折使用已经发生的上一折行情重新训练；当前折不能读取随后测试
季度的数据来选公式、方向或权重。每折在测试数据加载前写入不可覆盖的
`frozen.json`，训练/验证/测试均由独立 DataProvider 截止时间约束。最多取
通过验证和交易去重的 `frozen_limit` 个候选，等初始资金分配给各自独立持仓账本，
季度内不做额外的子策略资金再平衡，季度末清仓。没有候选的折记为空仓；
某候选评估失败则该折失败，不能删除失败候选后重新平均或拼出成功总收益。

每折输出 `training.json`（含成本评分及逐季度指标）、`shortlist.json`、
`validation.json`（含逐候选门槛及交易去重证据）、`frozen.json`、
`test.json`（含每日净收益）；根目录 `experiment.json` 固定配置与代码哈希，
`summary.json` 汇总净值、Sharpe、回撤、盈利折比例、失败及空仓折数。
整体通过条件为：无失败折、合并全成本收益 > 0、Sharpe > 1、多数折盈利。
这些历史年份已被研究过，结果属于历史向前验证，不宣称全新未见样本。
涉及 2026 的测试折必须传入 `--allow-2026-test`，只允许完整 Q1、Q2。
Q1 使用 2024 Q4–2025 Q3 训练、2025 Q4 验证；Q2 使用 2025 全年训练、
2026 Q1 验证。已查看的 2026 结果只能作为重复研究证据。

```bash
./.venv/bin/python -m Genetic_Algorithm walk-forward \
  --config /Users/dmiwu/work/PythonProject/cryptoFactorAnalyze/Genetic_Algorithm/configs/daily_5groups_all_costs.json \
  --h5 /Users/dmiwu/work/PythonProject/cryptoFactorAnalyze/data/crypto_quant.h5 \
  --run-dir /Users/dmiwu/work/PythonProject/cryptoFactorAnalyze/Genetic_Algorithm/runs/walk_forward_costs_001
```

`tmp/run_full_daily_gp.sh` 默认 `GP_WORKFLOW=single-split`：2024 全成本训练、
2025 全年验证筛选并冻结，然后测试 2026-01-01 至 2026-09-01，输出
`test_summary.json`。全部阶段成功后删除本次运行的 `factor_results`；失败时保留诊断。
此模式下 `all_costs_sharpe` 验证保留覆盖率门槛，要求全年净收益 > 0、
Sharpe > 1；IC 仅作诊断，不再要求正 IC 和三个正 IC 季度。
默认配置现在启用 `family_diversity`：随机生成/变异先均匀抽取价格、波动、
量额、主买家族，再在家族内构造子树；交叉仍可产生混合公式。种群按纯家族及
混合桶轮流保留候选，各桶内按原全成本适应度排序，为其他家族保留探索机会。
训练候选清单每家族最多 ceil(validation_limit / 4) 个，最终每家族最多
`family_candidate_limit=2` 个。混合公式计入每个引用家族；价格的标准差也计入
波动家族。结构分类是保守代理，不能替代收益相关性和持仓检查；数据覆盖不足
或适应度不合格者不会因家族配额被送入验证。
默认配置启用 `validation_stability`：2025 四个季度分别在季度内清仓回测，
至少三个季度净收益为正，最差季度收益 >= -10%，最大正季度收益除以所有
正季度收益之和 <= 60%。这些独立季度账本不是全年连续净值的收益归因。
再以手续费、滑点各 1.5 倍回测 2025 全年，要求净收益和 Sharpe 均 > 0，
资金费保持真实值。季度指标、集中度、压力指标、原因和家族标签写入
`validation.json`。这些阈值为事先设定的研究参数，不是已证明最优值。
候选按 2025 净 Sharpe 排序，使用 2024 的净收益和持仓去重后最多保留五个。
2026 只报告所有冻结候选的测试结果，不反馈筛选；已查看的数据属于重复研究。
零候选时记录空冻结清单，不把没有交易称为因子通过。
原始价格、成交量、成交金额、笔数、主动买入量/金额与八个衍生字段均保留。
设 `GP_WORKFLOW=walk-forward` 运行上述 2025 四折；`GP_WORKFLOW=holdout-2026`
运行 2026 两折，均为可选流程。
家族与全年稳定性规则目前仅接入固定年份流程；运行旧滚动流程需要使用
关闭 `family_diversity`、`validation_stability` 的配置，否则入口会明确拒绝。
walk-forward 的训练及测试账本在内存中计算，不产生 HTML 或 `factor_results`。
全成本回测比 IC 评分昂贵，四折需四次独立搜索；先用小种群配置验证运行耗时。

验证命令：`./.venv/bin/python -m pytest tests/genetic_algorithm -q`。
新增验证覆盖正式回放指标一致性、资金费率缺失拒绝、季度标签截断、
净 Sharpe 排序与门槛、逐折重新训练及先冻结后读取、失败/空仓折不被遗漏。

### 固定基准增量研究与邻近参数检验

`daily_5groups_all_costs.json` 现默认启用 `reference_factor=064185107a8f267e`
及 `validation_parameter_stability=true`，原 bash 入口自动读取此配置。
基准公式与方向（-1）固定在 `incremental.py`，不从旧测试收益文件加载。
训练只用 2024：原扣费后 Sharpe 目标减去
`reference_similarity_penalty * max(0, 收益相关性, 同向持仓重合度)`；
相关性达到 0.8 或持仓重合度达到 0.7 时拒绝候选。基准账本每阶段缓存一次。

2025 保留原全年、季度、成本压力及家族筛选，再要求与基准不过度相似，
且固定组合的扣费后 Sharpe 严格高于基准（默认增量阈值 0）。组合为初始
资本各占 50% 的两个独立账户，不跨账户再平衡，不调整候选方向或优化权重；
组合收益从两个已扣费账户的净值合成，不代表合并持仓净额后的执行模拟。

逐个窗口做 ±20% 扰动（四舍五入、去重、保持正窗口），默认至少 75% 的
变体净收益和 Sharpe 均 > 0。保持原公式及方向，不选择表现最好的变体。
无可变窗口时记录不适用；超出预热预算的变体按失败计入分母，不静默丢弃。
每个变体的 AST、指标及失败原因写入 `validation.json` 的 `parameter_stability`；
基准、组合及增量指标写入 `incremental`。缺失新增证据时拒绝冻结候选。

新功能仅支持固定年份流程；旧滚动流程还需关闭这两个新开关。
2026 不参与上述计算或门槛调整。由于基准是查看过 2026 后选出的，整个流程
仍属于重复研究，新增筛选不能恢复历史测试的独立性，也可能筛出零个候选。
邻近参数每个窗口最多增加两次验证回测；本次代码修改不自动启动完整训练。
