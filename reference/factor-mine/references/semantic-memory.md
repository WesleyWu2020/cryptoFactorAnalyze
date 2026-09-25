# Semantic Memory — 语义研究日志契约（本地版）

> 每轮实验结束后（无论 ACCEPTED / REJECTED / CRASH），向
> `outputs/factor_mine_semantic_log.jsonl` **追加一行** JSON。
> 本仓库没有 eval 自动落 jsonl 的机制，这份日志同时承担实验记录职能——
> 因子文件可能回滚删除，但这份日志必须留住"研究过什么、结论是什么"。

## 记录格式（一行一个 JSON，字段全必填）

```json
{
  "ts": "2026-09-13T08:00:00+00:00",
  "factor_id": "funding_extreme_revert_20d",
  "parent_factor_id": null,
  "op_type": "add_factor",
  "search_mode": "explore",
  "semantic_key": "funding_extreme|-|persistence|mean_reversion|rank",
  "semantic_mutation": null,
  "decision_class": "STATISTICAL_REJECT",
  "rankic": -0.0100,
  "rankicir": -0.11,
  "ls_sharpe": -0.4,
  "ls_annual_return": -0.129,
  "turnover": 0.9,
  "coverage": 0.89,
  "maxdd": 0.41,
  "corr_max": 0.62,
  "corr_factor": "Volume_Stability_Factor",
  "conclusion": "IC 稳定为负，数据支持 continuation 而非 mean_reversion；翻号后收益仍不达门槛，分支关闭"
}
```

## 字段规则

- `parent_factor_id`：本轮基于哪个因子/哪轮实验；全新方向填 `null`
- `search_mode`：`explore`（新维度）/ `exploit`（深挖已验证区域）/ `mutate`（单点语义修改）
- `semantic_mutation`：五维变化时记 `before_key -> after_key`，否则 `null`
- `decision_class`：`ACCEPTED` / `STATISTICAL_REJECT` / `REDUNDANCY_REJECT` /
  `IMPLEMENTATION_INVALID` / `OOS_REJECT`（实现有效、IS 过门槛、OOS 验证失败），
  与工作流第 6 步分类一致
- metrics 字段口径（全部 IS 段、trading_net 场景）：
  - `rankic` / `rankicir`：`samples.<is>.ic.rank_ic_mean` / `icir`
  - `ls_sharpe` / `ls_annual_return`：`scenarios.trading_net.<is>.sharpe` / `annual_return`
  - `turnover` / `maxdd`：同 scenario 块；`coverage`：`samples.<is>.coverage`
  - `corr_max` / `corr_factor`：相关性门控结果（未做填 null）
- `OOS_REJECT` 记录时：metrics 字段抄 IS 段结果，`conclusion` 必须写 OOS 关键数字
  与衰减幅度（例："IS RankIC 0.021 → OOS 0.006，衰减 71%，谱系 OOS 预算剩 1 次"）
- `conclusion`：**一两句话写清这轮学到什么**（分支关闭、方向证伪、成本不可行……），
  这是整份日志最有价值的字段，禁止只写 "REJECTED"
- 指标缺失（评估 null）就如实写 null，禁止填 0

## 轨迹审查时的用法（每 ~5 轮，配合 SKILL.md 轨迹审查）

回答四个问题再决定下一轮：

1. 最近都在研究哪些 event / direction？是否一直在同一 semantic region 打转？
2. 哪些词表区域从未覆盖？
3. 失败集中在哪些 semantic_key？同一 key 失败 ≥2 次 → 该区域降权
4. 哪些 schema 容易触发相关性 WARN/FAIL？（与库内因子的语义距离比想象近）

## 边界

- 本日志只用于**研究决策参考**（选下一轮研究什么），不构成 accept/reject 依据——
  最终门槛永远以 SKILL.md"接受门槛" + 当轮评估结果为准
- 样本量不足前禁止用本日志训练任何排序模型；先做统计观察
