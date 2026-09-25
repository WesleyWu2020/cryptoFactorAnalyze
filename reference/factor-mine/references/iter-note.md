# ITER_NOTE — 实验便签强制模板（本地版）

每次改因子文件，同步更新文件顶层的 `ITER_NOTE: dict`。缺一不跑。

## 必填四字段

```python
ITER_NOTE: dict = {
    "op_type":    "modify_factor",  # 见 op-types.md
    "hypothesis": "...",            # 为什么觉得这个改动会改善指标（经济 or 统计含义）
    "change":     "...",            # 具体改了什么（一句话能说清）
    "expected":   "...",            # 预期指标区间 + 副作用（turnover / maxdd / coverage）
}
```

## 推荐字段（强烈建议）

```python
    "parent_iter": 7,               # 当前 best 来自第几次实验（对应语义日志行号）
    "reasoning":   "...",           # 更深的因果链：上一轮报告诊断发现了什么 → 这一轮做什么
```

> **IS/OOS 纪律**：`hypothesis` / `reasoning` 引用的证据只能来自 IS 段
> （2024~2025）评估或先验机制，禁止引用 OOS 段结果作为改动依据
> （见 `is-oos-protocol.md` §4）。OOS 验证结果在当轮实验结束后补记到
> ITER_NOTE 的 `"oos_result"` 字段（如 `"pass: rankic 0.021->0.014"` 或
> `"reject: 衰减 71%"`），供轨迹审查追溯。

## op_type 特定字段

| op_type | 推荐字段 |
|---|---|
| `add_factor` | `factor_family`: "liq"（所属因子族，见 factor-families.md）；且必须先建 `SEMANTIC_PLAN`（见下节） |
| `compose_factor` | 同 `add_factor`，必须先建 `SEMANTIC_PLAN` |
| `modify_factor` | `old_param`、`new_param` |
| `transform_factor` | `transform`: "mad_rank" / "none"（preprocessing 切换）或截面处理说明 |
| `preprocess` | 改了哪个预处理项（warmup_bars / 缺失值策略） |

## SEMANTIC_PLAN（add_factor / compose_factor 强制）

新建或组合因子时，写代码前先在因子文件顶部建立 `SEMANTIC_PLAN: dict`。
五维 id **只能取自 `schema-vocabulary.md`**；modify_factor 默认继承原 plan 不重写。

```python
SEMANTIC_PLAN: dict = {
    "event":     {"id": "vol_spike", "statement": "波动率骤升，恐慌/分歧加剧"},
    "context":   {"id": "funding_crowding", "statement": "多头拥挤时波动放大更倾向踩踏"},
    "qualities": [{"id": "volume_confirm", "statement": "需放量确认，过滤无量假波动"}],
    "direction": {"id": "reversal", "statement": "拥挤方向反转"},
    "output":    {"id": "rank", "statement": "截面 rank 打分"},
    "mechanism": "拥挤多头 + 波动骤升 + 放量 → 强制平仓踩踏 → 短期反转",
    "fields":    ["close", "quote_volume", "funding"],
    "expected_horizon": "1d",
    "invalidation": "RankIC 不显著为负、或去掉 funding_crowding 后 IC 不变（机制不成立）",
    "search_mode": "explore",          # explore / exploit / mutate
    "semantic_key": "vol_spike|funding_crowding|volume_confirm|reversal|rank",
}
```

规则：

- `semantic_key` 格式 `event|context|qualities(排序)|direction|output`，无 context/quality 记 `-`
- `fields` 只能填 signal-contract.md 列出的本地可得字段
- `modify_factor` 只改窗口/参数/平滑 → key 不变，plan 原样保留
- 改动导致任一五维字段实质变化 → 本轮 ITER_NOTE 加 `"semantic_mutation": "before_key -> after_key"`
- 累计 mutation 改变核心经济机制 → 按 SKILL.md 规则 6 身份漂移处理：改名 + add_factor 重建档
- "删掉某个 quality/context 成分"也算 semantic diff（结构性简化），触发轨迹审查的退化收敛检查

## 完整示例

```python
ITER_NOTE: dict = {
    "op_type":    "modify_factor",
    "hypothesis": "IS 报告显示 turnover=1.8 超过门槛且换手 spikes 与 trading_net 回撤对齐；"
                  "把动量窗口从 20 日拉到 60 日应能压换手持平 gross 收益。",
    "change":     "SETTING['params']['window'] 20 -> 60，warmup_bars 同步 20 -> 70；其它不动。",
    "expected":   "turnover 1.8 -> 1.0 以下；RankIC 微降（0.028 -> 0.022±0.004）；trading_net sharpe 升。",
    "parent_iter": 7,
    "reasoning":  "第 7 轮 gross 好 net 差（gross-to-net gap 大），成本是主失败模式，先压换手再谈信号增强。",
    "old_param":  20,
    "new_param":  60,
}
```

## 为什么强制

没写 ITER_NOTE 的实验 = 没有假设的实验 = 蒙对了也无法复用。
ACCEPTED 时 ITER_NOTE 随因子文件归档；REJECTED 时把失败原因补进便签再回滚，
让 `outputs/factor_mine_semantic_log.jsonl` + git 历史构成完整思考链。

## 反模式

| 反模式 | 修复 |
|---|---|
| `hypothesis: "试试看"` | 写出经济/统计因果（"流动性溢价" / "taker 失衡的知情交易含义"） |
| `expected: "提升"` | 给具体区间（"RankIC 0.02 -> 0.03±0.005，turnover 略升"） |
| `change: "优化代码"` | 说清楚改了哪一行 / 哪个参数 |
| 把"改窗口 + 换 preprocessing"挤一个 ITER_NOTE | 拆两次实验，两个 ITER_NOTE |
