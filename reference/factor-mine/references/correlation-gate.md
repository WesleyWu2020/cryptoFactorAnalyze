# 相关性门控 — 防因子换皮重复（本地版）

crypto 量价因子族容易换皮重复（5 日反转 vs 1 日反转、波动率 vs 振幅、
taker_buy_base_volume vs taker_buy_quote_volume）。高相关因子入库只带共线性，
不带新信息。

## 本地机制（没有 --prod-corr，自己算）

本仓库没有 FactorComparator。相关性用**因子值缓存**自算：
`FactorManager.evaluate` 会把因子值落到 `data/factor_results/<factor_id>.parquet`
（long 格式：`date / instrument / factor`），库内已有因子跑过评估就有缓存。

口径：**逐日截面 Spearman 相关，再对重叠日期取均值**（不是把面板拍平算 pooled
Pearson——pooled 口径会被宇宙构成变化污染；逐日均值口径也更接近"两个信号每天
排序是否一样"的换皮判断）。

```python
import pandas as pd
from pathlib import Path

RESULTS = Path("data/factor_results")

def load_values(factor_id: str) -> pd.DataFrame:
    df = pd.read_parquet(RESULTS / f"{factor_id}.parquet")
    return df.pivot(index="date", columns="instrument", values="factor")

def daily_rank_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    dates = a.index.intersection(b.index)
    corr = a.loc[dates].corrwith(b.loc[dates], axis=1, method="spearman")
    return float(corr.mean())

new_values = load_values("<new_factor_id>")
for cached in sorted(RESULTS.glob("*.parquet")):
    fid = cached.stem
    if fid == "<new_factor_id>":
        continue
    rho = daily_rank_corr(new_values, load_values(fid))
    print(f"{fid}: {rho:+.3f}")
```

前提：被对比的因子已在同一数据快照下跑过评估（缓存按源哈希+数据指纹失效，
过期缓存会自动重算，不用手工管）。没跑过的先用 main.py / batch 脚本补跑。

| max \|ρ\|（与任一库内因子） | 判定 |
|---|---|
| ≥ **0.85** | **直接 REJECTED**，回滚 |
| 0.60 ~ 0.85 | 警告：允许继续，但必须在 ITER_NOTE 里论证独立信息 |
| < 0.60 | 通过 |

## 命中 ≥ 0.85 怎么办

三个改设计方向（不要硬加）：

1. **换计算方式**：如绝对波动已存在，改成相对波动 `vol_20 / vol_120`
2. **换输入**：close 反转已存在，改用 taker 流 / trade_count 结构 / funding 等
   未被占用的信息源（可用字段见 factor-families.md）
3. **换截面定义**：全宇宙已有，改成对 BTC 残差化、或条件子集内 rank

## 何时绕过

如果确信高相关因子有独立价值（如作为组合层的对冲腿而非 alpha），不要作为
新 alpha 建档，转 `portfolio/` 组合层做残差化/正交化。

## 补充自查

- 门控覆盖 `factor_analyse/factor_mining/` 已有因子；`factor_analyse/unsubmit/`
  里的在研因子互相也可能撞车——批量基线用
  `./.venv/bin/python scripts/batch_evaluate_factors.py` 统一跑一遍后按上面对比
- 组合入库阶段的聚类去重用 `portfolio/factor_pool/cluster_selector.py`
  （pooled Pearson、|ρ|>0.6 聚类取覆盖最好者）——那是组合层口径，与本门控的
  逐日截面口径不同，别混用阈值结论

## 阈值说明

0.85 严格门槛 / 0.60 软警告沿用原 skill 经验值。本地是逐日截面 rank 相关口径：
同一信号不同实现的 ρ 在该口径下通常很高（>0.9），0.85 是强换皮证据，不建议下调；
而跨族但同向的因子（如低波 vs 低换手）常在 0.3~0.6，属正常邻域。
