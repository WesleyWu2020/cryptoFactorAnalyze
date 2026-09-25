# Minute GP System

**Single-path 24h-bar crypto factor mining.** 一套管道、一套默认、一个目标：持续挖出训练期和 OOS 上 ls_netret 都正的因子，通过统一自动链路进入正式因子库。

## 目录布局

```
engines/unified_v2/
  run.py                    # [1] GP search (NSGA-III, 5 objectives + archive novelty)
  evaluate_with_framework.py # [2] OOS formal replay
  publish_pipeline.py       # [3] gate + push to review queue (writes archive)
  driver_once.py            # orchestrator: [1][2][3] sequentially per cron tick
  archive.py                # persistent phenotype archive (cross-run novelty)
  config.py                 # all knobs (defaults固化)
  data_loader.py, fitness.py, evolution.py, evaluator.py, operators.py,
  indicators.py, backend.py # engine internals
registry.py                 # source specs (unified_v2 + framework_evaluator)
common/data/factor_platform/strategy_factory_runs/minute_gp_system/
  review_queue.json         # central queue snapshot
  phenotype_archive.json    # cross-run memory: params of past-published factors
  driver_state.json         # seed counter + run stats for driver_once
  driver.log                # START/OK/FAIL per cron tick
  runs_auto/                # auto-mined experiment outputs per seed
```

## 数据字段（26 个 indicator）

```
slots 0..19  : OHLCV-derived (open/high/low/close/volume + 15 派生)
slots 20..25 : crypto-native raw (funding, open_interest, premium_close,
                                   mark_close, index_close, turnover)
```

crypto-native 字段是唯一与 OHLCV 正交的信号源（derivatives pricing / OI flow / basis），**绝对不要在 refactor 中删掉**（见 memory note）。h5 `bybit_linear_1m_unified.h5` 原生包含这 6 个字段。

## 管线

```
                       ┌──────────────────┐
                       │ driver_once.py   │ ← cron xx:15 每小时触发
                       └────────┬─────────┘
                                │
      ┌─────────────────────────┼─────────────────────────┐
      ▼                         ▼                         ▼
 [1] run.py              [2] evaluate_with_           [3] publish_pipeline
    GP search               framework.py                 IS+OOS gate
    (+ archive novelty)     OOS 2026-01-01..2026-05-01 write queue + archive
      │                         │                         │
      ▼                         ▼                         ▼
 pareto_results.json     formal_replay.json         review_queue.json
                                                   phenotype_archive.json
                                                         │
                                                         ▼
                                      [4] backfill_gp_platform_sync.py
                                          + auto_triage.py
                                                         │
                                                ┌────────┴────────┐
                                                ▼                 ▼
                                         materialize         archive
                                              │
                                              ▼
                             factor_base/factor_values/<name>.parquet
                             factor_base/factor_registry.json
```

## 固化默认值

| 维度 | 值 | 说明 |
|---|---|---|
| MINUTES_PER_PERIOD | 1440 | 24h bar，daily rebalance |
| TOP_QUANTILE | 0.2 | L top 20% / S bot 20%，对称 |
| TRADING_COST | 0.0015 | 15bp round-trip |
| N_INDICATORS | 26 | 20 OHLCV + 6 crypto-native |
| min_presence | 30% | coin pool filter（`data_loader.py:83`），336 coin pool |
| train window | 2022-01-02 → 2025-12-31 | 4 个完整训练年 |
| OOS window | 2026-01-01 → 2026-05-01 | 样本外正式回放 |
| profile | perp_1d | OOS replay 的评估 profile |
| composition_mode | base_only | 禁用 ts/cs composition 层 |
| population × generations | 2000 × 15 | 每轮 30K evaluations |
| publish gate | IS_netret > 0 AND OOS_netret > 0 AND yearly train netret > 0 | `publish_pipeline.MIN_OOS_NETRET` |
| archive capacity | 2000 entries | FIFO，`archive.ARCHIVE_CAP` |

## NSGA-III 5 个目标

`['ls_netret', 'ls_net_sharpe', 'neg_turnover', 'novelty', 'ls_1-maxdd']`

novelty 现在是 **pop novelty + archive novelty** 的 max corr 合并值：
- pop novelty：当代种群内 phenotype corr 距离
- archive novelty：与 `phenotype_archive.json` 里所有已发布因子的 phenotype corr 距离
- 两者取 max corr → novelty = 1 - max corr

含义：GP 被迫避开已发现的 basin，朝 phenotype 空间未开垦区域搜。

## Cron 自动化

```cron
# 每小时挖一轮
15 * * * * cd /root/crypto-research/common && LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda/lib64 \
           flock -n /tmp/minute_gp_driver.lock timeout --kill-after=60s 180m \
           /root/.pyenv/versions/3.9.0/bin/python -m gp.minute_gp_system.engines.unified_v2.driver_once \
           >> /root/crypto-research/common/data/factor_platform/strategy_factory_runs/minute_gp_system/driver.log 2>&1

# 每日数据刷新
0 2 * * * flock -n /tmp/bybit_daily_refresh.lock \
          /bin/bash /root/crypto-research/common/scripts/run_bybit_daily_refresh.sh
```

## CLI（手动单步）

```bash
cd /root/crypto-research/common

# 一键跑一轮 GP→OOS→publish
python -m gp.minute_gp_system.engines.unified_v2.driver_once

# 或者手动分三步
python -m gp.minute_gp_system.engines.unified_v2.run --output_dir /tmp/exp_NAME/raw
python -m gp.minute_gp_system.engines.unified_v2.evaluate_with_framework \
    --pareto /tmp/exp_NAME/raw/pareto_results.json \
    --json_output /tmp/exp_NAME/formal_replay.json
python -m gp.minute_gp_system.engines.unified_v2.publish_pipeline \
    --pareto /tmp/exp_NAME/raw/pareto_results.json \
    --formal /tmp/exp_NAME/formal_replay.json \
    --experiment exp_NAME

# 自动收口到 platform + factor_system
python common/scripts/backfill_gp_platform_sync.py

# 手动数据刷新
bash scripts/run_bybit_daily_refresh.sh
```

## 监控与状态

```bash
# 队列状态
python3 -c "
import json
q = json.load(open('/root/crypto-research/common/data/factor_platform/strategy_factory_runs/minute_gp_system/review_queue.json'))
from collections import Counter
print(f'total={len(q)}  ', Counter(c['status'] for c in q))
"

# 驱动状态
cat /root/crypto-research/common/data/factor_platform/strategy_factory_runs/minute_gp_system/driver_state.json

# 每轮 START/OK/FAIL
tail -30 /root/crypto-research/common/data/factor_platform/strategy_factory_runs/minute_gp_system/driver.log

# Archive 规模
python3 -c "
from gp.minute_gp_system.engines.unified_v2.archive import load_archive_params
print(f'archived factors: {len(load_archive_params())}')
"
```

## 关键约定

- **只有 auto_triage 最终保留在 approved pool 的因子文件才留盘**：parquet + registry 写入 `factor_base/`。pending/rejected 只保留 queue / sqlite 轨迹。
- **archive 是持久化的跨 run 记忆**：publish 时自动 append；新 GP 启动时 reconstruct phenotype 后作为 novelty 的对照物。
- **publish 是 append-only**，按 `candidate_id = sha1(experiment::formula)[:16]` 去重。review 决策幂等。
- **OOS profile (`perp_1d`) 必须与 GP bar 对齐**：改 `MINUTES_PER_PERIOD` 时同步改 `evaluate_with_framework` 的默认 profile。
- **bar 内操作无时间穿越**：所有 window / slice 仅限当前 bar 内分钟，跨 bar 只用 backward-looking；`build_lagged_returns` 保证因子在 t 配对 t+1 return。
- **每轮 cron 用新 seed**：`driver_state.json` 的 `next_seed` 单调递增，重启不影响。

## 数据流

```
bybit API ─┬→ core 1m parquet ─→ core h5 ─┐
           ├→ funding parquet ──────────────┤
           ├→ open_interest parquet ────────┼→ unified h5 ─→ GP data_loader
           └→ price_state parquet ──────────┘   (原子 rename)    (26 indicators)
                                               build 脚本默认含全部字段
                                               ALWAYS preserve 6 extra fields
```
