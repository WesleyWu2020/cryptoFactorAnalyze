# Daily genetic-algorithm workflow

## 第三批：预测期限与执行周期

`configs/daily_5groups_all_costs.json` 已启用 `horizon_diagnostics`。2025 验证
结果的 `horizon_execution` 分开记录两组预先确定的对照：

- `prediction_horizons=[1,3,7,14]`：信号日 `t` 在 `t+1` 开盘入场、持有
  指定天数的 Rank IC。按退出日剔除验证尾部标签，方向沿用训练结果。
- `execution_intervals=[1,3,7]`：按固定锚点每隔指定天数调仓，保留相同的
  下一日开盘执行、五分组、手续费、滑点和资金费。每条账本都在阶段末日
  之前平仓，空仓日补零后按同一验证日历计算可比 Sharpe。

两组结果只作诊断；正式回放和入选仍固定每天调仓、`hold_days=1`。
多日标签彼此重叠，其 IC 均值不作为独立样本显著性检验。测试多个期限
不能据此挑选 2025 表现最好的周期再声称它是未调参的结果。
可在新运行配置中调整两个有序周期列表，且都必须包含 1 日基线。
本批变更：`Genetic_Algorithm/horizon.py`、`cost_fitness.py`、`config.py`、
`cli.py`、`search.py`、`configs/daily_5groups_all_costs.json`、本说明及
`tests/genetic_algorithm/test_horizon_execution.py`、`test_cli.py`。

## 第二批：风格暴露与残差信号验证

新运行的配置可设 `"exposure_residual_mode": "diagnostic"`，在 2025 验证阶段
把 Barra 九类当日风格暴露及残差写入 `validation.json` 每个候选的
`exposure_residual`。改为 `"gate"` 时，残差必须有足够的共同样本，且扣除
手续费、滑点和资金费后总收益及 Sharpe 均为正，候选才可入选；默认 `off`。
`configs/daily_5groups_all_costs.json` 已显式开启诊断模式，不改变原有
入选门槛；确认数据覆盖后可按需切换为 `gate`。

风格数据按验证阶段末日截断，一次加载并供该阶段所有候选复用。原信号和残差
只在残差有效的同一批日期、币种上比较，沿用训练确定的方向、五分组等执行
设置；记录联合回归 R²、有效天数、同样本 IC 与两条完整费用账本指标。
样本或回归不足会写明原因，`gate` 模式下拒绝候选。该检查增加验证耗时，
并改变验证判据，需在新运行目录重新完成搜索及验证，不能复用旧冻结结果。
标准化后日残差标准差不高于 `1e-8` 时按数值零处理，避免纯浮点误差
形成可交易排序。
本批主要变更：`Genetic_Algorithm/exposure.py`、`config.py`、`cli.py`、
`selection.py`、`search.py`、`configs/daily_5groups_all_costs.json`、本说明，
以及 `tests/genetic_algorithm/test_exposure_residual.py`、`test_cli.py`。

## GP 计算语义 v2

新运行统一使用 `symmetric_fractional` 同值分组：跨分组边界的同值标的
平均分配对应名额；全部同值时多空目标为零，抵消后不重新放大敞口。
训练、正式回放、导出因子的默认设置和报告采用相同口径。通用回测默认仍为
`legacy_instrument`，旧 `assign_groups` 整数分组接口保持不变。

`rolling_hit_rate` 将缺失与非有限输入保留为 NaN，窗口必须完整；真实零值
按未命中处理。表达式校验保守推导值域，拒绝对严格正数排名或恒非正输入
计算正值占比等确定退化组合；非负输入仍可通过零值频率表达信息。

导出记录语义版本及运行时哈希，加载与调用时检查兼容性。无版本的旧 GP
导出由标准 loader 拒绝；旧运行应使用原代码环境复现，或在新的输出目录
重新导出、评估。不要覆盖旧结果，也不要复用旧适应度或冻结结论。
训练归档版本升至 `daily-gp-search-v2`；冻结和多种子实验指纹包含共享分组、
执行 profile 与相关回放代码。源码版本变化后需新建实验目录。

验证：`./.venv/bin/python -m pytest tests/genetic_algorithm/test_signal_semantics.py -q`。

本批变更文件（路径相对仓库根目录）：

- 计算与诊断：`Genetic_Algorithm/operators.py`、`expression.py`、`cost_fitness.py`。
- 分组与报告：`factor_common/grouping.py`、`profiles.py`、`loader.py`、`manager.py`、`reporting.py`。
- 运行与版本：`Genetic_Algorithm/export.py`、`replay.py`、`search.py`、`cli.py`、`experiment.py` 及本说明。
- 回归：`tests/genetic_algorithm/test_signal_semantics.py`、`test_cli.py`、`test_end_to_end.py`，以及 `tests/factor_common/test_profiles.py`。

回放同时检查市场数据末日覆盖指定阶段末日，避免缺少执行尾部数据时被标为完成。

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

`daily_5groups_all_costs.json` 现默认固定三个已通过候选作为基准：
`064185107a8f267e + a37c60cf4492e9f1 + 0006a47b612eaf9b`。
及 `validation_parameter_stability=true`，原 bash 入口自动读取此配置。
基准公式与方向（-1）固定在 `incremental.py`，不从旧测试收益文件加载。
训练只用 2024。默认训练目标为原扣费后 Sharpe 目标减去
`reference_similarity_penalty * max(0, 收益相关性, 同向持仓重合度)`；
设置 `training_reference_objective=portfolio_incremental_sharpe` 时，保留原有单因子
收益、Sharpe、季度稳定性、回撤和覆盖率门槛，并改为按候选加入固定参考组合后的
2024 年扣费后 Sharpe 增量排序。默认值 `similarity_penalty` 保持旧行为不变；
相关性达到 0.8 或持仓重合度达到 0.7 时拒绝候选。基准账本每阶段缓存一次。

2025 保留原全年、季度、成本压力及家族筛选，再要求与基准不过度相似，
且固定组合的扣费后 Sharpe 严格高于基准（默认增量阈值 0）。三个基准按
初始资本等权分别运行，候选加入后四个账户初始各占 25%；不跨账户再平衡，
不调整候选方向或优化权重。组合收益从各自已扣费净值合成，不代表合并持仓
净额后的执行模拟。训练时候选分别与三个基准计算收益相关和持仓重合度。

逐个窗口做 ±20% 扰动（四舍五入、去重、保持正窗口），默认至少 75% 的
变体净收益和 Sharpe 均 > 0。保持原公式及方向，不选择表现最好的变体。
无可变窗口时记录不适用；超出预热预算的变体按失败计入分母，不静默丢弃。
每个变体的 AST、指标及失败原因写入 `validation.json` 的 `parameter_stability`；
基准、组合及增量指标写入 `incremental`。缺失新增证据时拒绝冻结候选。

新功能仅支持固定年份流程；旧滚动流程还需关闭这两个新开关。
2026 不参与上述计算或门槛调整。由于基准是查看过 2026 后选出的，整个流程
仍属于重复研究，新增筛选不能恢复历史测试的独立性，也可能筛出零个候选。
邻近参数每个窗口最多增加两次验证回测；本次代码修改不自动启动完整训练。

### 连续账本稳定性与固定多种子实验

配置 `stability_mode=continuous_leave_best_out` 后，2024 Q1 仅用于冻结方向，
Q2–Q4 使用一条连续的全成本账本评分。2025 也从全年连续账本计算季度收益、
财富利润贡献和集中度。集中度超过配置值只产生适应度惩罚；验证新增硬门槛：
删除利润贡献最大季度的日收益后，剩余净收益和 Sharpe 必须分别超过
`validation_min_remaining_return` 与 `validation_min_remaining_sharpe`。
组合诊断用同一组删除日期比较参考组合与加入候选后的 Sharpe，避免区间错位。

`tmp/run_multi_seed_daily_gp.sh` 默认预先冻结种子 47–51、搜索预算、代码哈希、
2024 数据指纹和历史归档哈希。各种子独立读取同一个历史归档，不串行继承新结果；
全部完成后仅合并各自 `deduplication.accepted`，按 2024 因子值跨种子去重，
最多选 20 个候选统一进行一次 2025 验证。候选来源记录在
`pooled/candidate_sources.json`，实时代数进度记录在各 seed 目录的
`search_progress.json`，汇总状态为 `experiment_progress.json`。

```bash
bash /Users/dmiwu/work/PythonProject/cryptoFactorAnalyze/tmp/run_multi_seed_daily_gp.sh
```

可通过 `GP_SEEDS`、`GP_EXPERIMENT_DIR`、`GP_CONFIG` 和 `GP_ARCHIVE` 覆盖默认值。
同一实验目录允许跳过已完成种子并续跑后续阶段；清单冻结后源码变化会终止合并，
需要新建实验目录。

### 训练期邻近参数稳定性 A/B

配置 `training_parameter_stability=true` 后，每代先按原适应度选择尚未检查的前
`training_parameter_stability_top_k` 个含窗口公式。每个窗口分别做一次约 -20%
和 +20% 的单点扰动，只读取固定 2024 训练阶段，并沿用原公式已经冻结的方向。
变体使用相同全成本账本；中位净收益退化、中位 Sharpe 退化和正收益变体比例不足
共同形成只减不加的适应度惩罚。公式 hash 的结果在整个搜索内缓存，不重复回测。
每代检查公式数和实际新增扰动回测数分别写入 `generations.jsonl` 与进度文件。

`daily_5groups_continuous_multiseed_parameter_stability.json` 是实验组配置；原
`daily_5groups_continuous_multiseed.json` 保持对照组行为。下面的入口固定用相同
两个 seed（默认 52、53）、种群、代数、历史归档和 2025 验证规则依次运行两组：

```bash
bash /Users/dmiwu/work/PythonProject/cryptoFactorAnalyze/tmp/run_parameter_stability_ab.sh
```

两组使用相同种群、代数和尝试上限，但缓存命中和无效公式比例可能不同，因此实际
唯一公式评估数不保证相等。实验组还会额外执行扰动回测；这些调用被单独计数。

### 可选实验：局部变异、行为分区、分层精英与 DSR 诊断

`configs/daily_5groups_behavior_elites.json` 开启下面所有功能；默认配置的开关均为
关闭，保留原算法。该配置沿用原 2025 验证与 2026 报告规则，不自动启动训练。
各开关可单独消融；分层精英必须同时开启训练参数稳定性。

* `structured_mutation`：在原变异概率之内，用 `mutation_weights` 选择窗口、字段、
  剪枝、子树四类变异（默认 35%/25%/15%/25%）。窗口取相邻的合法配置值；字段
  必须同家族且同量纲；剪枝只移除一层同量纲的一元节点，再重新评估，不能视为
  数学等价化简。无合法修改时返回父代，日志记录 attempts/changed，不冒充新公式。
* `semantic_generation`：复用现有 `expression_dimension`，在生成和交叉阶段检查
  类型，禁止任何子表达式为 mixed。`dimensionless_probability=0.75` 是生成目标
  偏好，不保证最终种群恰好占 75%。最多尝试 24 次后回退到合法字段。该约束也会
  排除原量纲系统将输出标为 mixed 的 safe_log/signed_sqrt 等算子，这是实验限制。
* `behavior_diversity`：用 2024 评分账本的换手率、与固定参考因子的最大有符号
  收益相关、最大同侧持仓重叠、市场 beta 划分固定格子。市场代理是前一日已知
  成分的等权开盘到开盘收益；不用未来成分、不填充缺价。无参考时相关与重叠进入
  缺失格子，不当成零。分区边界固定在 `research.BEHAVIOR_BINS`，不能按2025/2026
  调整。保留时优先各格最优者，每格最多 `behavior_cell_capacity=2`，父代先均匀
  抽格子再在格内竞赛。不是完整 MAP-Elites；是有容量上限的行为分区精英实现。
  行为分区开启时替代训练种群的字段家族轮流配额，最终家族筛选仍然生效。
* `layered_elites`：探索池按未扣扰动惩罚的基础分排序，每代至多检查 top_k 个
  尚未检查的合格候选（按行为分区分配名额）。精英池只比较已检查后的分数；
  扣分后不再大于零的个体不能成为精英。无窗口公式记录不适用、零扰动成本。
  每个公式只检查一次，最终仅输出已检查精英。探索保留至少30%的种群名额，
  每代先生成 ceil(population*0.30) 个缓存中不存在的合法随机公式，再生成其余
  子代；小种群至少保留一个精英位置。新结构名额不足时明确报错。早期精英不足
  时探索池补位；分区容量可能使实际保留人数小于 population，但每代评估池仍
  以 population 为目标。旧精英也参与下一代分区竞争。
* `dsr_diagnostics`：只记录诊断，不改适应度、方向、候选门槛或验证规则。
  `search_research.json` 对 `dsr_effective_trials=[1,10,100]` 分别报告条件 DSR。
  用日频 Sharpe、偏度、非超额峰度和所有有账本试验（含拒绝候选）的 Sharpe
  离散度，假设零均值原假设。有效试验次数是外部假设，绝非公式数的替代名字；
  超过可用试验数量的情景标为 unavailable。没有估计真实独立试验数，也没有
  校正完整自适应搜索、收益序列自相关或覆盖率过滤后的存活偏差。因此不能解释
  为未来盈利概率，不能用单次公式统计宣称整个搜索已经消除过拟合。

新增诊断只接受固定 2024 训练阶段。2025 不参与分区、繁殖或 DSR 统计。
`cost_fitness.json` 保留每个公式的 behavior、return_moments 和扰动前后分数；
`search_research.json` 保存最终分区及 DSR 假设。`generations.jsonl` 和进度文件
新增累计公式评估/扰动检查/扰动回测数、已选合格数、精英/探索人数、分区数和
局部变异计数。原 unique_evaluations / parameter_stability_* 字段仍是单代值。
这些新增源码也纳入训练与冻结清单的源码指纹。

验证命令：

```bash
./.venv/bin/python -m pytest tests/genetic_algorithm/test_research_search.py -q
```

已有多种子入口可通过 GP_CONFIG 指向此新配置；应另建实验目录。要判断具体机制
的贡献，分别关闭其余新开关进行单变量对照，并同时报告实际公式数与扰动回测成本。
