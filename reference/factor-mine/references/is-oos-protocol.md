# IS/OOS 协议 — 训练集挖掘 / 测试集验证（本地版）

> 核心思想：**所有挖掘决策只在训练集（IS）上做；测试集（OOS）只用于验证，
> 验证通过才允许进入下一轮迭代。** 防的就是全样本选择过拟合——
> 在全样本上边看指标边调参，等于把测试集当训练集用。

## 1. 样本切分（固定，不可按结果挪动）

| 段 | 区间 | 用途 |
|---|---|---|
| IS（训练集） | 2024-01-01 ~ 2025-12-31 | 提出假设、改参、accept/reject 迭代决策、报告诊断 |
| OOS（测试集） | 2026-01-01 ~ 最新存储日期 | **仅**对 IS 已通过的候选版本做一次性验证 |
| 全样本 | 2024-01-01 ~ 最新存储日期 | 归档核对（门槛不变），不做迭代选择依据 |

切分点 2025-12-31 是**先验固定**的，与 `Genetic_Algorithm` 的
train 2024 / validation 2025 / frozen test 2026 约定一致（该 GA 的 test 段同样
冻结，两边不互相偷看）。数据从 2023-07-05 开始，2023H2 只作 warmup 前缀，
不进评估窗口。禁止"OOS 不好看就挪切分点"——挪切分点 = 用 OOS 信息做决策 =
测试集污染。

## 2. 评估命令（两种口径）

```bash
# IS（每轮必跑，迭代决策只看这个；samples.full 即 IS 段；分组固定 5）
./.venv/bin/python factor_analyse/main.py <factor_path> 1 --groups 5 \
    --start 2024-01-01 --end 2025-12-31

# 成本磨损分支的 horizon 变体（触发条件与纪律见 SKILL.md「成本磨损分支」）
./.venv/bin/python factor_analyse/main.py <factor_path> 3 --groups 5 --start 2024-01-01 --end 2025-12-31
./.venv/bin/python factor_analyse/main.py <factor_path> 5 --groups 5 --start 2024-01-01 --end 2025-12-31
```

```python
# OOS 验证 + 全样本归档（仅当 IS 全门槛 PASS 后跑一次）：
# 用 split_date 让框架切出 in_sample / out_of_sample 切片
from factor_common import FactorManager

result = FactorManager().evaluate(
    "factor_analyse/factor_mining/<factor>.py",
    params={"start": "2024-01-01", "end": "<最新存储日期>",
            "rebalance_days": 1, "n_groups": 5, "split_date": "2026-01-01"},
)
perf = result["factor_performance"]
oos_ic = perf["samples"]["out_of_sample"]["ic"]
oos_ls = perf["scenarios"]["trading_net"]["out_of_sample"]
```

- 切分语义以 `docs/factor_common.md`"Sample definitions"为准：跨切分点的持仓期
  信号从两个切片同时剔除（purge），scenario 切片是认证账本的纯日期切片，
  不会重跑组合
- IS-only 运行时 main.py 默认的 `out_of_sample_days=180` 会在 IS 窗口内部再切
  一刀——那个内部切片无意义，IS 决策只看 `full` 块
- 相关性门控在 IS 阶段做即可（换皮判断与样本段无关）；全样本归档时复核一次
- 三段评估在同一轮迭代里共用一条 ITER_NOTE

## 3. 两级接受判定

### 第一级：IS 门槛（挖掘阶段的 accept/reject）

IS 评估必须过 SKILL.md"接受门槛"全部硬门槛（`trading_net` 口径）。
任一 FAIL → 本轮 REJECTED，回滚，**不看 OOS**。

### 第二级：OOS 验证门（IS 通过后一次性验证）

| 检查 | 门槛 | 含义 |
|---|---|---|
| 方向一致 | sign(rank_ic_mean_oos) == sign(rank_ic_mean_is) | 信号没有翻号 |
| 衰减幅度 | \|rank_ic_mean_oos\| ≥ 0.5 × \|rank_ic_mean_is\| | OOS 衰减 ≤50%（真实因子常见衰减 30~70%，超 50% 视为 IS 过拟合） |
| 可盈利 | `scenarios.trading_net.out_of_sample.annual_return` > 0 | 扣费后多空年化为正 |

- 三项全过 → OOS 验证通过 → **ACCEPTED**，成为新 best
- 任一不过 → **OOS_REJECT**：回滚到上一 best，记录失败，**不成为新 best**
- OOS 段短（<1 年），不设 ICIR / maxdd 硬门，但报告诊断仍要做，
  出现危险信号（anti-patterns.md 清单）照样拒
- OOS 指标缺失（null）视为不过

### 为什么不直接拿全样本门槛迭代

全样本门槛照旧是**归档标准**（不动）；但逐轮选择若看全样本，
每一轮"指标变好就接受"都在向全样本过拟合。IS 决策 + OOS 验证 +
全样本归档核对，三层各管一段。

## 4. 禁止事项（测试集污染红线）

1. **禁止用 OOS 结果设计下一轮假设**：OOS 诊断只能回答"这个版本要不要
   成为新 best"，不能回答"下一轮改什么"。下一轮的 hypothesis 必须来自
   IS 证据或先验机制
2. **禁止调参让 OOS 变好**：OOS fail 后的合法动作只有"回滚该改动，回 IS
   重新挖"，不是"微调一下再跑 OOS 看看"
3. **OOS 验证预算**：同一因子谱系（同一 semantic_key 及其 mutate 后代）
   OOS_REJECT 累计 ≥3 次 → 该谱系冻结，转 explore 新方向。OOS 每多看一次，
   它的信息就漏进决策一次，预算必须有限
4. **禁止挪切分点 / 换 OOS 区间**：包括"2026 年太特殊，换 2025H2~2026H1 试试"
5. **禁止偷看 GA 的 frozen test**：`Genetic_Algorithm` 的 test 段（2026+）与本
   skill 的 OOS 是同一段；那边 freeze 之前的任何 test 结果都不能拿来做本
   skill 的决策依据
6. IS 段内的子区间（如 2024 vs 2025）可以自由看，那属于训练集内部分析
7. **horizon 变体共享谱系 OOS 预算**：成本磨损分支下 {1,3,5} 各档位的评估
   属同一谱系，禁止"每个 horizon 都跑一次 OOS 挑一个过的"——只有 IS 全
   门槛通过的 horizon 才允许进 OOS 验证

## 5. 日志记录约定

- 本仓库没有 eval 自动落 jsonl 的机制；每轮把 IS（及可能的 OOS）关键指标手工
  抄进 `outputs/factor_mine_semantic_log.jsonl`（格式见 semantic-memory.md）
- 语义日志的 `decision_class` 含 `OOS_REJECT`（实现有效、IS 过门槛、
  OOS 验证失败）；metrics 字段抄 IS 段的，`conclusion` 里写 OOS 关键数字
  （如 "IS RankIC 0.021 → OOS 0.006，衰减 71%，谱系预算剩 1 次"）
