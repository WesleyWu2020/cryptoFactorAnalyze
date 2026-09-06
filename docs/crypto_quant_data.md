# Crypto Market-Cap Top50 Data Pipeline

This is the point-in-time CoinMarketCap/Binance workflow. It writes the normalized research store to `data/crypto_quant.h5`; the existing CSV workflow remains separate.

## Daily commands

```bash
./.venv/bin/python data/update_crypto_quant.py backfill
./.venv/bin/python data/update_crypto_quant.py update
./.venv/bin/python data/update_crypto_quant.py validate
./.venv/bin/python data/update_crypto_quant.py inspect
./.venv/bin/python data/update_crypto_quant.py rebuild-derived
```

Mutation commands accept `--as-of 2026-09-03T00:20:00Z`, `--store PATH`, and `--reset-staging`. `--as-of` must include a timezone and is normalized to UTC; without it, the current UTC time is used.

## HDF5 layout

The store keys are `/cmc100_daily`, `/cmc100_constituents`, `/futures_contracts`, `/klines_daily`, `/funding_events`, `/universe_monthly`, `/research_panel_daily`, and `/_metadata`.

The daily kline and research-panel schemas use the canonical `quote_volume` name for Binance's quote-asset volume. `open_time` is not stored; `date` is derived from Binance's candle-open timestamp and `close_time` is retained. Existing stores must be backfilled or updated once after this schema change so that historical `quote_volume` values are fetched again.

Example query:

```python
import pandas as pd

with pd.HDFStore("data/crypto_quant.h5", mode="r") as store:
    print(store.keys())
    panel = store.select(
        "research_panel_daily",
        where="date >= Timestamp('2026-08-01')",
    )
```

## Cron

The daily update runs at 08:20 Asia/Shanghai:

```cron
20 8 * * * cd /Users/dmiwu/work/PythonProject/cryptoFactorAnalyze && ./.venv/bin/python data/update_crypto_quant.py update >> logs/crypto_quant_update.log 2>&1
```

Cron inherits a minimal environment. The entry therefore uses an absolute `cd` and the repository's absolute virtual-environment path. Set `CMC_PRO_API_KEY` (or `CMC_API_KEY`) in the cron environment if the CMC endpoint requires authentication, and ensure `logs/` exists.

## Recovery and validation

On a non-zero run, inspect `logs/crypto_quant_update.log` and rerun the same command; the staged HDF5 is designed to resume from checkpoints. Use `--reset-staging` only when configuration, mapping rules, or the requested `--as-of` target changed and the existing staging state should be discarded. Run `validate` and require a zero exit code before consuming data. `validate` returns 2 for validation errors; source, store, and runtime failures return 1.

### 资金费与占位行情质量（2026-09-05）

原始 `funding_events` 的费率、缺失价格和零价格保持原样，不填补、不删除。
`panel.annotate_funding_prices(events)` 返回副本并增加 `mark_price_valid`：仅有限且大于零的价格为有效。
日频面板新增 `funding_invalid_price_count`，价格无效不影响有效费率聚合；需要按数量计算实际资金费时，必须检查价格标记，不能将缺失金额视为零。
费率自身缺失或非有限值时，该日聚合费率为 NaN，事件计数仍保留。

`funding_coverage_status` 按交易对和 UTC 日期标记：

- `unknown`：有事件，但缺少可靠的应结算时刻表，不能证明完整。
- `no_events`：无事件且没有应结算时刻表；不等于零资金费，也不直接断言漏采。
- `complete`：可信时刻表与实际事件一一匹配，且费率有效。
- `missing`：已知应结算时刻表下事件数量不足。
- `not_applicable`：时刻表明确该日无需结算，且没有事件。
- `schedule_mismatch`：事件数或时间与时刻表冲突。
- `invalid_rate`：实际费率缺失或非有限值。

`has_complete_funding` 仅在 `complete` 时为 True。旧的 `funding_complete_through` 参数保留调用兼容，但不再用于证明完整性。
采集 checkpoint 与 `last_successful_funding_time` 仍描述请求覆盖，不能作为结算完整性的证明。
现有数据没有历史结算周期来源，因此标准管线不会自动产生已验证的 `complete`。不能用当前周期或从已有事件猜测出的周期冒充历史规则。

`build_research_panel(..., funding_schedule=...)` 接受 `date, symbol, expected_times` 三列的可信日程，每日每标的一行，`expected_times` 为该日全部 UTC 应结算时刻；空列表明确表示不适用，缺行表示未知。允许一秒以内的历史时间戳漂移，一对一匹配；拒绝重复及跨日时刻。该可选输入目前用于显式核验，标准 HDF 管线尚无可信历史日程数据源接入。
这些状态描述该日结束后的数据质量，不能提前用于当日开仓决策。

`has_placeholder_kline` 标记成交量为零且 OHLC 全部相等的行情。这是无交易行情识别，不是推定下架日期。
月度选币在原有有效期约束之外，排除决策日前一天的此类完整 K 线；下一次出现有效行情时可以重新入选。不会依据今天的合约状态删除其全部历史。
持有期内出现占位行情时保留成员与行情行，将 `has_complete_kline` 设为 False，不追溯删除持仓。

重建仅处理派生表：`./.venv/bin/python data/update_crypto_quant.py rebuild-derived`。
验证使用 `./.venv/bin/python -m pytest tests/test_crypto_quant*.py -q`，并比较重建前后原始表哈希与按 cutoff 截断后的选币、面板结果。
