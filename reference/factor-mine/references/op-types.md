# op_types — 单点假设的合法改动（本地版）

> 每次实验只改一件事。这份清单定义"一件事"的边界。
> 本地每个因子是独立 `.py` 文件（`factor_analyse/factor_mining/`），
> op_type 围绕"一个因子文件"定义。

| op_type | 含义 | 触发相关性门控 | 典型影响 |
|---|---|---|---|
| `add_factor` | 新建 1 个因子文件（`factor_analyse/factor_mining/<name>.py`） | ✅ 强制 | 全新信号，看重 RankIC / coverage |
| `modify_factor` | 改已有因子的窗口 / 参数 / 实现细节（如 `SETTING["params"]` 里的回看天数、分位数、平滑 span） | ✅ 与库内因子算 ρ，且与原版本对比 | 指标 ±小幅变化 |
| `transform_factor` | 只改截面处理：`preprocessing` none ↔ mad_rank、因子侧的去量纲/rank 方式 | ✅ 建议 | net/gross 差、maxdd 变化 |
| `compose_factor` | 在同一因子文件内组合 ≥2 个正交字段/族的子信号（如 动量 + funding + taker 失衡），组合方式限简单可解释形式（rank 加权、符号对齐求和）；成分应各自方向已验证，一轮只动组合方式或成分集，不连带改成分内部参数 | ✅ 强制 | RankIC/IR 增量、鲁棒性（OOS 衰减） |
| `preprocess` | 改 warmup_bars、缺失值填充策略（不动公式与截面处理） | ❌ | 主要影响覆盖率和前期段 |
| `other` | 重构 / 兼容性变更，逻辑不变 | ❌ | 指标应 ≈ 0 变化（可用来验证重构正确性） |

## 跨多个 op_type 的改动 = 违反单点假设

```
❌ 一次性"窗口 20→60 + 换 mad_rank + 加 funding 过滤"
   即使指标变好也无法归因，无法复用

✅ 拆三次：
   Iter N+1: modify_factor    窗口 20→60        [看 RankIC/turnover 变化]
   Iter N+2: transform_factor preprocessing → mad_rank [看 net IR 变化]
   Iter N+3: compose_factor   加 funding 过滤    [看 RankIC 增量]
   每一步都可独立验证、独立回滚
```

## 边界情况

- **改一个参数同时改 factor_direction** → 算一次 `modify_factor`，但 ITER_NOTE
  里必须写清两个改动点，并解释为什么不可分割
- **删因子** → 本地因子库以文件为单位，删除文件即删除；若是替换（删旧加新），
  拆两轮：先评新的，确认优于旧的再删
- **重构不改逻辑** → `other`，预期指标完全不变；变了说明重构引入了 bug
- **改 rebalance_days / n_groups 等评估参数** → 不算因子改动，不建档；
  主口径固定 rebalance_days=1；当 IS 诊断显示"有毛 alpha 被成本磨损"
  （gross 正、net 负、turnover 超标）时，可按 SKILL.md「成本磨损分支」
  在 {3, 5} 档重评——同一因子文件不改代码，每个 horizon 独立记语义日志，
  只接受先验档位、禁止连续扫参
- **给已有因子加时序平滑**（ewm/rolling/ts_decay 类）→ `modify_factor`，
  ITER_NOTE 注明 smoothing 类改动及它针对的报告问题（通常是 turnover 过高）
- **改 universe / eligibility** → 本仓库 universe 固定 `historical_top50`，
  不存在合法改动；任何变相收窄/扩张宇宙的做法都按身份漂移审查
- **输入字段集变化**（如从纯 OHLCV 变成加 funding）→ 先过规则 6 身份漂移审查：
  构成漂移则改名按 `add_factor` 重建档，不构成才按 `modify_factor`
