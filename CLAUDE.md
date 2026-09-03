# CLAUDE.md

## 项目概述

加密货币因子研究与回测分析系统。基于 Binance Top100 市值历史数据，围绕 USDT 交易对进行因子构建、挖掘与 HTML 报告生成。

## 目录结构

```
data/                        # 数据获取与预处理
  capture_data_binance.py    # 增量抓取 Binance K 线（主数据入口）
  list-from-binance.py       # 获取 Top50 交易对列表
  market_cap_50_index.py     # 构建市值加权指数
  pipeline_time_utils.py     # 时间工具
  factor_data/               # 因子 CSV 输出目录（date/instrument/factor）
  kline_data/                # K 线 CSV 主文件（增量复用，不每日新建）
  indicator/                 # 指标中间产物

factor_analyse/              # 因子分析主流程
  main.py                    # 报告生成入口：python factor_analyse/main.py <factor_type>
  factor_config.py           # 所有因子注册表（必须在此注册才能运行）
  factor_analyse_custom.py   # factor_miner 核心类
  factor_mining/             # 规则因子脚本（复用 util_factor.py + operator_utils.py）
  ML_factor_mining/          # ML 因子脚本（XGBoost / LightGBM / GRU）
  Alpha101/                  # WorldQuant Alpha101 因子实现
  Genetic_Algorithm/         # 遗传算法因子挖掘模块

reports/                     # HTML 因子分析报告输出目录
logs/                        # 调度器日志
tests/                       # 测试脚本
daily_feishu_scheduler.py    # 每日 08:00 BJT 自动运行 + 飞书推送
```

## 运行命令

```bash
# 使用项目 venv（必须，避免解释器路径问题）
./.venv/bin/python data/list-from-binance.py          # Step 1: 获取 Top50 列表
./.venv/bin/python data/capture_data_binance.py       # Step 2: 增量更新 K 线数据
./.venv/bin/python factor_analyse/main.py --list      # 查看所有已注册因子
./.venv/bin/python factor_analyse/<factor_script>.py  # Step 3: 生成因子数据
./.venv/bin/python factor_analyse/main.py <factor_type>  # Step 4: 生成分析报告

# 安装依赖
pip install -r requirements.txt

# 日常调度（飞书推送需配置 .env）
./.venv/bin/python daily_feishu_scheduler.py --once   # 手动触发一次
./.venv/bin/python daily_feishu_scheduler.py          # 持续运行（每日 08:00）
```

## 新增因子的固定流程（严格顺序，不跳步）

1. 在 `factor_analyse/factor_mining/` 新建因子脚本，复用 `util_factor.py` 与 `operator_utils.py`
2. 在 `factor_analyse/factor_config.py` 注册配置，必须包含：`file_prefix`、`factor_name`、`factor_direction`、`factor_desc`、`rebalance_period`
3. 运行因子脚本，生成 CSV 到 `data/factor_data/`（格式：`date` / `instrument` / `factor` 三列，date 为 `yyyy-mm-dd`）
4. 运行 `python factor_analyse/main.py <factor_type>` 生成 HTML 报告到 `reports/`
5. 返回结果必须包含：变更文件列表、执行命令、核心输出路径、**未来函数检查结论**

## 未来函数（Look-ahead Bias）强制规则

`factor` 列只能由 `t` 时点及历史数据计算，严禁以下模式：

| 禁止模式 | 说明 |
|---------|------|
| `shift(-k)` (k>0) | 前瞻位移 |
| `rolling(..., center=True)` | 引入未来窗口 |
| `bfill` / 反向填充 | 特征列上禁用 |
| `merge_asof(..., direction="forward")` | 向前匹配未来时间 |

**提交前必须提供**：
- 静态检查：是否存在上述禁止模式及位置
- 动态反证：`max_abs_diff` 在 `date <= cutoff` 区间应为 0 或在浮点容忍范围内

## 数据格式规范

- 因子 CSV：至少包含 `date`（yyyy-mm-dd）、`instrument`（Binance 交易对符号）、`factor`（数值）三列
- K 线 CSV：`symbol`、`date`、`open`、`high`、`low`、`close`、`volume`、`quote_volume`、`trades_count`、`taker_buy_base`、`taker_buy_quote`
- K 线文件增量复用已有主文件，不每日新建（`resolve_incremental_output_path` 逻辑）

## 工程约束

- **Python 解释器**：始终使用 `./.venv/bin/python`，在项目根目录执行
- **禁止**修改 `factor_miner` 的输入输出契约（列约定、核心调用签名）
- **禁止**引入新的大型框架或重型依赖
- **禁止**为"看起来更好"做大规模重构（影响历史脚本可复现性）
- 优先复用 `factor_analyse/factor_mining/util_factor.py` 与 `operator_utils.py`，不新增重复 util

## 测试与验证

当前无统一测试入口。新增逻辑的验收标准：
1. 因子脚本可独立运行并输出正确格式的 CSV
2. `main.py` 能成功读取并生成 HTML 报告
3. 通过基础语法检查（无 import 错误、无运行时崩溃）
4. 完成未来函数静态 + 动态检查

## 输出规范

每次完成任务后说明：
1. 改了什么文件
2. 为什么这么改
3. 验证步骤与风险点
