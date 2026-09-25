# 因子族清单 — crypto perp 日频本地版

不要"为了挖而挖"。先想清楚要补哪个**信息维度**，并先看 `factor_analyse/factor_mining/`
已有因子的族分布（`./.venv/bin/python factor_analyse/main.py --list` 看 META.category /
description）。

## H5 字段清单（唯一数据真源）

文件：`data/crypto_quant.h5`（管线见 `docs/crypto_quant_data.md`）。
Binance USDT 永续**日频** K 线，2023-07-05 起每日更新，全库 ~150+ 交易对；
universe 为 point-in-time CMC top50（`historical_top50`，决策日已知成员，无未来成分）。
`data_needed` 可声明字段（`DataProvider.list_datas()` 实测）：

| 字段 | 内容 | 数据层 |
|---|---|---|
| `open` / `high` / `low` / `close` | 日 K OHLC | T1 |
| `volume` | 成交量（币单位） | T1 |
| `quote_volume` | 成交额（USDT） | T1 |
| `trade_count` | 成交笔数 | T2 |
| `taker_buy_base_volume` / `taker_buy_quote_volume` | 主动买入量（币）/ 额（USDT） | T2 |
| `funding` | 日频资金费率（当日结算事件均值） | T3 |

数据层含义：T1=OHLCV 可自给；T2=成交结构字段；T3=资金费（质量注意：结算覆盖
由数据派生时刻表支撑而非官方日程，费率可做信号，回测中 all_costs 已可认证
但强度有限）。

**明确没有的**（旧仓库迁移过来的 unsubmit 因子里出现过，新因子不可用）：
分钟级数据、open_interest、mark/index/premium 价格、基差、跨所字段、市值字段
（市值只在 universe 选币时用，不作信号）。

## 因子族（按本地数据可得性排序）

### Tier 1：仅 OHLCV + quote_volume（随时可用）

| 因子族 | category | 典型代表 | 经济含义 | 推荐窗口 |
|---|---|---|---|---|
| **动量 / 反转** | momentum/reversal | `-pct_change(N)`、N 日动量 | 过度反应 / 趋势延续 | 反转 1~10d，动量 20~120d |
| **波动率 / 低波** | volatility | `-std(ret, N)`、`-(high-low)/close`、MAX 因子 | 低波异象 / lottery 偏好 | 5~60d |
| **流动性** | liquidity | amihud `mean(\|ret\|/quote_volume)` | 流动性溢价 | 5~60d |
| **量价相关 / 量能** | pvc | `corr(close, volume, N)`、`-vol_cv`、放量滞涨 | 信息含量 / 筹码结构 | 5~60d |
| **路径 / 形态** | path | 路径效率、上下影线比、偏度 | 微观结构 / lottery | 5~60d |

### Tier 2：成交结构字段（trade_count / taker）

| 因子族 | 典型代表 | 经济含义 |
|---|---|---|
| **taker 流 / 主动买卖** | `taker_buy_quote_volume / quote_volume` 失衡及其时序变化 | 知情交易 / 情绪 |
| **交易频率 / 单笔均额** | `quote_volume / trade_count`、trade_count 与 volume 背离 | 散户/机构结构 |

已有参考：`factor_analyse/factor_mining/` 下多个 taker/trade_count 因子
（Momentum_TakerBuyQuote_Extreme、TopBottom_Trades_Ratio_Factor 等）。

### Tier 3：资金费（质量受限，谨慎用）

| 因子族 | 典型代表 | 经济含义 |
|---|---|---|
| **资金费率** | funding 水平/均值/极端化 | 多空拥挤 / 持仓成本 |

funding 结算覆盖由数据派生时刻表支撑（panel 全量 `complete`，all_costs 可认证），
但时刻表非交易所官方历史日程、强度有限；做信号可用，ITER_NOTE 需注明该不确定性；
回测主决策口径仍是 trading_net，all_costs 作第二道成本参考。

## 经验门槛

- 新因子与库内任一因子 max |ρ|（逐日截面 rank 相关）< 0.6 才值得继续
- 理想因子库：覆盖 4~6 个族，避免单族堆叠

## 决策树：下一个该挖什么

```
库内因子的 category 分布？（main.py --list / 各文件 META.category）
├─ 集中在 1~2 个族 → 换族（最缺哪个挖哪个，优先 Tier 2/3 未占用信息源）
├─ 覆盖 3+ 族但互相 |ρ| > 0.4 → 先删同质/做正交，不要继续加
└─ 各族都有但单因子都弱 → 回头做 modify_factor / transform_factor 提质，不是加数量
```

## 反偏好

| ❌ 别做 | ✅ 改做 |
|---|---|
| 同族堆 5 个（1d/3d/5d/10d 反转全保留） | 选 1 个代表，换族 |
| 50 行复杂特征工程换 0.002 RankIC | 物理含义清晰的简单因子 |
| 纯 grid search 幸存者 | 有经济/行为解释的因子 |
| 不看库内已有就开工 | 先 `--list` + 相关性门控摸清族分布 |
| 给 unsubmit 旧因子的 mark/basis/OI 字段找替代数据源 | 承认数据边界，换 Tier 1/2 假设 |
