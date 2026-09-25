# 信号契约 — factor_common 因子文件硬门禁（本地版）

契约 = **factor_common loader 规范 + 无未来函数双门禁**。违反 = 评估直接抛错
（loader 拒绝 / 静态扫描中止 / check_cutoff 回放不一致），或结果不可信。

## 文件结构契约

```python
TYPE = "regular"                 # 本仓库只有 regular

META = {
    "factor_name": "my_factor",  # 必须等于文件名（不含 .py），全局唯一
    "author": "local",
    "level": "daily",            # 本仓库仅支持日频
    "category": "momentum",      # 因子族，见 factor-families.md
    "description": "...",        # 一句话经济含义
}

SETTING = {
    "data_needed": ["close", "quote_volume"],  # 只声明真实用到的字段
    "universe": "historical_top50",            # 唯一合法 universe，不要自创
    "warmup_bars": 20,                         # ≥ 最长滚动窗口 + 余量
    "preprocessing": "mad_rank",               # "none" 或 "mad_rank"
    "params": {"window": 20},                  # 公式参数全放这里
    "factor_direction": 1,                     # 1=因子值大预测跑赢；-1 反之
}

def calc_factor(data_ctx):  # data_ctx[field] 是 date × instrument 的 DataFrame
    ...
    return factor             # 同为 date × instrument 的 float DataFrame
```

- 因子脚本放 `factor_analyse/factor_mining/`；脚手架：
  `FactorManager().create_template('<name>')`；参考实现
  `factor_analyse/factor_mining/example_momentum.py`
- import 模块必须无副作用；可执行入口放 `if __name__ == "__main__":`
- 评估入参只允许 profile 字段（start/end/rebalance_days/n_groups/fee_rate/
  slippage/split_date 等）；公式参数一律走 `SETTING["params"]`

## 无未来函数双门禁

1. **静态扫描**（`validation.scan_future_leaks`，读数据前执行，命中即中止）：
   禁止 `shift(-k)`（k>0）及任何负位移、`rolling(..., center=True)`、
   `bfill`/`backfill`（含 `fillna(method=...)`）、`merge_asof(direction="forward")`
2. **动态回放**（`check_cutoff`）：把行情历史与 universe 成员决策都截断到若干
   内部 cutoff，重算因子并要求与全历史前缀完全一致（`max_abs_diff ≤ 1e-12`，
   轴与 NaN 掩码都要一致）

静态扫描只是证据不是证明——任意 Python 可以藏前视；回放通过才是硬保证。
唯一允许含未来数据的模块是 `factor_common/labels.py`（next-open forward returns），
它只喂 IC/分组统计，永远不进因子构造、分组或交易。

## 数值/时序约束

| 约束 | 要求 | 违反后果 |
|---|---|---|
| 因果性 | 因子值只能由 ≤ t 的数据计算 | 双门禁直接拒 |
| 执行口径 | 信号日 t → t+1 开盘成交（signal_delay_days=1 固定），`rebalance_days` 非重叠持有 | 口径误以为 trade-at-close 会高估 IC |
| 全样本统计量 | 禁止用全段 mean/std 做时序 z-score、禁止全样本拟合参数 | 未来函数 |
| warmup | `warmup_bars` ≥ 最长滚动窗口 + 余量，前期 NaN 段不进评估 | 前期截面不足，分组失真 |
| universe | 固定 `historical_top50`（point-in-time CMC top50 成员，决策日已知） | coverage 失真；自创宇宙 = 违规 |
| NaN 处理 | 明确策略；ffill 只能沿历史方向 | 隐蔽未来函数 |
| 数据质量 | `has_placeholder_kline`（零量平价 K 线）由框架处理：D-1 占位 bar 只挡 D 日新开仓；因子侧不需要也不许自己猜下架日 | 脏数据进信号 |

## 可用数据（data_needed 可声明）

唯一数据真源：`data/crypto_quant.h5`（Binance USDT 永续**日频**，2023-07-05 起）。
`DataProvider.list_datas()` 返回的字段：

| 字段 | 内容 |
|---|---|
| `open` / `high` / `low` / `close` | 日 K OHLC |
| `volume` | 成交量（币单位） |
| `quote_volume` | 成交额（USDT） |
| `trade_count` | 成交笔数 |
| `taker_buy_base_volume` / `taker_buy_quote_volume` | 主动买入量（币）/ 额（USDT） |
| `funding` | 日频资金费率（当日结算事件 `funding_rate_mean`，来自 research_panel_daily） |

**没有**分钟数据、open_interest、mark/index/premium、跨所字段——这些词只存在于
`factor_analyse/unsubmit/` 里迁移来的旧因子的 SEMANTIC_PLAN 中，新因子不可用。
新字段接入前先跑 `DataProvider().list_datas()` 确认，别信记忆。

资金费质量注意：funding 结算覆盖由 `build_funding_schedule` 的数据派生时刻表
支撑（当前 panel 全量 `complete`，all_costs 可认证），但该时刻表是从观测事件
反推的、非交易所官方历史日程，强度有限；费率字段做信号没问题，涉及"按数量实算
资金费"的回测口径以 `docs/crypto_quant_data.md` 的质量标记为准。

## 快速自检（跑评估前）

1. `./.venv/bin/python -c "from factor_common.loader import load_factor; load_factor('<path>')"` 不报错、META/SETTING 字段齐全、factor_name == 文件名
2. `./.venv/bin/python factor_analyse/main.py --list` 能看到该因子且 direction 正确
3. 截面每日非 NaN 数量足够（coverage 门槛 0.6，对应 samples 块的 coverage）
4. 若指标好得离谱（|RankIC| > 0.15 或 trading_net sharpe > 5），**先假设有未来函数**，逐行查时间戳——静态扫描不是证明
5. 指标好但不离谱也要记住执行口径：信号 t → t+1 开盘成交已是框架内置，
   不需要也不许自己在因子里 shift 对齐"可交易时刻"
