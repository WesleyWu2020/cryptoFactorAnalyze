# Unsubmit 因子两两组合推荐（5-5 / 3-7 / 7-3）

> 基于 `factor_cluster_catalog.csv`、`factor_pair_similarity.csv` 和 `FACTOR_CLUSTER_REPORT.md` 生成。
> 本推荐仅用于**缩小组合搜索范围**，最终绩效仍需按六图协议逐案复核。

## 评判标准

1. **单因子门槛**（必须同时满足）
   - `cache_status == valid`
   - `coverage_warning == False`
   - `mean_train_year_net_sharpe > 0`
   - `rankic_stability > 0`
   - `mean_turnover < 1.0`

2. **单因子强度分**
   - 对 `mean_train_year_net_sharpe`、`rankic_stability` 分别做 min-max 归一化。
   - `strength_score = 0.6 * sharpe_norm + 0.4 * rankic_stability_norm`。

3. **组合对门槛**
   - 两个因子必须来自**不同 primary_cluster**（跨簇分散）。
   - 不是 `high_redundancy`，也不是 `unstable_low_correlation`。
   - 训练期 `|rank_similarity| < 0.5`，`holding_overlap < 0.5`；`|return_similarity|` 若可计算也必须 `< 0.5`。
   - 验证/测试期最大 `|rank_similarity|` 不超过 0.5，否则额外扣分。

4. **组合综合分**
   - `div_score`：训练期 rank/return/holding 相似度的平均“分散度”（1 - |sim|，NaN 按中性 0.5 处理）。
   - `avg_strength`：两因子 strength_score 的均值。
   - `combined_score = 0.45 * div_score + 0.45 * avg_strength - 0.10 * turnover_penalty - oos_penalty`。

5. **权重分配**
   - `5-5`：等权。
   - `3-7` 与 `7-3`：将 70% 权重分配给 strength_score 更高的因子，30% 给较低者。

满足单因子门槛的因子数：**40** / 121 个有效缓存因子。
满足全部门槛的组合对数：**433**。
下表列出综合分最高的前 **20** 对。

## 推荐组合清单

### crypto_alpha_main_force_volatility_USA_session + crypto_alpha_realized_kurtosis

- **source**: `crypto_alpha_main_force_volatility` + `crypto_alpha_realized_kurtosis`
- **综合分**: 0.663  (分散度 0.817, 平均强度 0.702)
- **语义簇**: 日内时段与跨时区效应 × 波动率、尾部风险与高阶矩
- **经济逻辑**: Uses close, return, volume with (diff * diff).sum, (row_ss[valid_count] / count[valid_count]).astype, (row_sum[valid_count] / count[valid_count]).astype, (session_pos[block_ends - 1] + 1).astype, _calc_one_asset_state_returns_np to express session conditioned signal. / Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal.
- **训练期相似度**: rank=0.044, return=NaN, holding=0.006
- **样本外最大 |rank|**: 0.237
- **强度分**: crypto_alpha_main_force_volatility=0.767, crypto_alpha_realized_kurtosis=0.636 (强者: `crypto_alpha_main_force_volatility`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_main_force_volatility` | 50% | `crypto_alpha_realized_kurtosis` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_realized_kurtosis` | 30% | `crypto_alpha_main_force_volatility` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_main_force_volatility` | 70% | `crypto_alpha_realized_kurtosis` | 30% | 偏重强者 |

### crypto_alpha_realized_kurtosis + crypto_alpha_volume_up_continuous_down_volatility_USA

- **source**: `crypto_alpha_realized_kurtosis` + `crypto_alpha_volume_up_continuous_down_volatility_USA`
- **综合分**: 0.643  (分散度 0.810, 平均强度 0.665)
- **语义簇**: 波动率、尾部风险与高阶矩 × 日内时段与跨时区效应
- **经济逻辑**: Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal. / Uses close, return, volume with (diff * diff).sum, (row_ss[valid_count] / count[valid_count]).astype, (row_sum[valid_count] / count[valid_count]).astype, (session_pos[block_ends - 1] + 1).astype, _calc_one_asset_volume_up_continuous_down_return_np to express session conditioned signal.
- **训练期相似度**: rank=0.059, return=NaN, holding=0.012
- **样本外最大 |rank|**: 0.243
- **强度分**: crypto_alpha_realized_kurtosis=0.636, crypto_alpha_volume_up_continuous_down_volatility_USA=0.694 (强者: `crypto_alpha_volume_up_continuous_down_volatility_USA`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_realized_kurtosis` | 50% | `crypto_alpha_volume_up_continuous_down_volatility_USA` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_realized_kurtosis` | 30% | `crypto_alpha_volume_up_continuous_down_volatility_USA` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_volume_up_continuous_down_volatility_USA` | 70% | `crypto_alpha_realized_kurtosis` | 30% | 偏重强者 |

### crypto_alpha_csk_xyy_up + crypto_alpha_realized_kurtosis

- **source**: `crypto_alpha_csk_xyy_up` + `crypto_alpha_realized_kurtosis`
- **综合分**: 0.631  (分散度 0.781, 平均强度 0.645)
- **语义簇**: 量价关系与资金流 × 波动率、尾部风险与高阶矩
- **经济逻辑**: Uses close, dollar_volume, return, volume with (daily_ret * weights).sum, TARGET_TZ.upper, _as_market_series, _build_market_ret, _calc_chunk to express volume price interaction. / Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal.
- **训练期相似度**: rank=0.121, return=NaN, holding=0.035
- **样本外最大 |rank|**: 0.182
- **强度分**: crypto_alpha_csk_xyy_up=0.654, crypto_alpha_realized_kurtosis=0.636 (强者: `crypto_alpha_csk_xyy_up`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_csk_xyy_up` | 50% | `crypto_alpha_realized_kurtosis` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_realized_kurtosis` | 30% | `crypto_alpha_csk_xyy_up` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_csk_xyy_up` | 70% | `crypto_alpha_realized_kurtosis` | 30% | 偏重强者 |

### crypto_alpha_price_valley_vmacd_volume + crypto_alpha_realized_kurtosis

- **source**: `crypto_alpha_price_valley_vmacd_volume` + `crypto_alpha_realized_kurtosis`
- **综合分**: 0.620  (分散度 0.815, 平均强度 0.613)
- **语义簇**: 量价关系与资金流 × 波动率、尾部风险与高阶矩
- **经济逻辑**: Uses close, dollar_volume, high, low, return with ((high_tz - low_tz) / close_tz).where, (diff - dea).shift, (diff - dea).shift(1).replace, (diff - dea).shift(1).replace([np.inf, -np.inf], np.nan).astype, (dollar_volume / volume).where to express volume price interaction. / Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal.
- **训练期相似度**: rank=0.046, return=NaN, holding=0.010
- **样本外最大 |rank|**: 0.149
- **强度分**: crypto_alpha_price_valley_vmacd_volume=0.589, crypto_alpha_realized_kurtosis=0.636 (强者: `crypto_alpha_realized_kurtosis`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_price_valley_vmacd_volume` | 50% | `crypto_alpha_realized_kurtosis` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_price_valley_vmacd_volume` | 30% | `crypto_alpha_realized_kurtosis` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_realized_kurtosis` | 70% | `crypto_alpha_price_valley_vmacd_volume` | 30% | 偏重强者 |

### crypto_alpha_realized_kurtosis + crypto_alpha_volume_convergence_factor

- **source**: `crypto_alpha_realized_kurtosis` + `crypto_alpha_volume_convergence_factor`
- **综合分**: 0.610  (分散度 0.807, 平均强度 0.577)
- **语义簇**: 波动率、尾部风险与高阶矩 × 量价关系与资金流
- **经济逻辑**: Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal. / Uses return, volume with _calc_chunk, _chunk_columns, _convergence_factor, _convergence_factor(daily_volume).shift, daily_value.rolling to express volume price interaction.
- **训练期相似度**: rank=0.059, return=NaN, holding=0.019
- **样本外最大 |rank|**: 0.018
- **强度分**: crypto_alpha_realized_kurtosis=0.636, crypto_alpha_volume_convergence_factor=0.517 (强者: `crypto_alpha_realized_kurtosis`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_realized_kurtosis` | 50% | `crypto_alpha_volume_convergence_factor` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_volume_convergence_factor` | 30% | `crypto_alpha_realized_kurtosis` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_realized_kurtosis` | 70% | `crypto_alpha_volume_convergence_factor` | 30% | 偏重强者 |

### crypto_alpha_main_force_volatility_USA_session + crypto_alpha_volume_convergence_factor

- **source**: `crypto_alpha_main_force_volatility` + `crypto_alpha_volume_convergence_factor`
- **综合分**: 0.604  (分散度 0.740, 平均强度 0.642)
- **语义簇**: 日内时段与跨时区效应 × 量价关系与资金流
- **经济逻辑**: Uses close, return, volume with (diff * diff).sum, (row_ss[valid_count] / count[valid_count]).astype, (row_sum[valid_count] / count[valid_count]).astype, (session_pos[block_ends - 1] + 1).astype, _calc_one_asset_state_returns_np to express session conditioned signal. / Uses return, volume with _calc_chunk, _chunk_columns, _convergence_factor, _convergence_factor(daily_volume).shift, daily_value.rolling to express volume price interaction.
- **训练期相似度**: rank=0.254, return=NaN, holding=0.025
- **样本外最大 |rank|**: 0.279
- **强度分**: crypto_alpha_main_force_volatility=0.767, crypto_alpha_volume_convergence_factor=0.517 (强者: `crypto_alpha_main_force_volatility`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_main_force_volatility` | 50% | `crypto_alpha_volume_convergence_factor` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_volume_convergence_factor` | 30% | `crypto_alpha_main_force_volatility` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_main_force_volatility` | 70% | `crypto_alpha_volume_convergence_factor` | 30% | 偏重强者 |

### crypto_alpha_realized_kurtosis + crypto_alpha_volume_down_fall_volatility_ASIA_session

- **source**: `crypto_alpha_realized_kurtosis` + `crypto_alpha_volume_down_fall_volatility_ASIA`
- **综合分**: 0.603  (分散度 0.816, 平均强度 0.565)
- **语义簇**: 波动率、尾部风险与高阶矩 × 日内时段与跨时区效应
- **经济逻辑**: Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal. / Uses close, return, volume with (diff * diff).sum, (row_ss[valid_count] / count[valid_count]).astype, (row_sum[valid_count] / count[valid_count]).astype, (session_pos[block_ends - 1] + 1).astype, _calc_one_asset_volume_down_fall_return_np to express session conditioned signal.
- **训练期相似度**: rank=0.042, return=NaN, holding=0.011
- **样本外最大 |rank|**: 0.273
- **强度分**: crypto_alpha_realized_kurtosis=0.636, crypto_alpha_volume_down_fall_volatility_ASIA=0.493 (强者: `crypto_alpha_realized_kurtosis`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_realized_kurtosis` | 50% | `crypto_alpha_volume_down_fall_volatility_ASIA` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_volume_down_fall_volatility_ASIA` | 30% | `crypto_alpha_realized_kurtosis` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_realized_kurtosis` | 70% | `crypto_alpha_volume_down_fall_volatility_ASIA` | 30% | 偏重强者 |

### crypto_alpha_realized_kurtosis + crypto_alpha_skewed_deviation_return

- **source**: `crypto_alpha_realized_kurtosis` + `crypto_alpha_skewed_deviation_return`
- **综合分**: 0.594  (分散度 0.778, 平均强度 0.580)
- **语义簇**: 波动率、尾部风险与高阶矩 × 日内时段与跨时区效应
- **经济逻辑**: Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal. / Uses close, return with KeyError, _align_daily_to_original_index, _calc_daily_deviation_sum_np, _get_frame, _get_target_day_info to express session conditioned signal.
- **训练期相似度**: rank=0.151, return=NaN, holding=0.014
- **样本外最大 |rank|**: 0.222
- **强度分**: crypto_alpha_realized_kurtosis=0.636, crypto_alpha_skewed_deviation_return=0.523 (强者: `crypto_alpha_realized_kurtosis`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_realized_kurtosis` | 50% | `crypto_alpha_skewed_deviation_return` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_skewed_deviation_return` | 30% | `crypto_alpha_realized_kurtosis` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_realized_kurtosis` | 70% | `crypto_alpha_skewed_deviation_return` | 30% | 偏重强者 |

### crypto_alpha_price_valley_relative_vwap + crypto_alpha_realized_kurtosis

- **source**: `crypto_alpha_price_valley_relative_vwap` + `crypto_alpha_realized_kurtosis`
- **综合分**: 0.591  (分散度 0.814, 平均强度 0.528)
- **语义簇**: 量价关系与资金流 × 波动率、尾部风险与高阶矩
- **经济逻辑**: Uses close, dollar_volume, high, low, return with ((high_tz - low_tz) / close_tz).where, (dollar_volume / volume).where, (minute_vwap * volume_tz).where, (valley_vwap / total_vwap).where, TARGET_TZ.upper to express volume price interaction. / Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal.
- **训练期相似度**: rank=0.049, return=NaN, holding=0.009
- **样本外最大 |rank|**: 0.150
- **强度分**: crypto_alpha_price_valley_relative_vwap=0.419, crypto_alpha_realized_kurtosis=0.636 (强者: `crypto_alpha_realized_kurtosis`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_price_valley_relative_vwap` | 50% | `crypto_alpha_realized_kurtosis` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_price_valley_relative_vwap` | 30% | `crypto_alpha_realized_kurtosis` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_realized_kurtosis` | 70% | `crypto_alpha_price_valley_relative_vwap` | 30% | 偏重强者 |

### crypto_alpha_new_momentum_day_night_volume + crypto_alpha_realized_kurtosis

- **source**: `crypto_alpha_new_momentum_day_night_volume` + `crypto_alpha_realized_kurtosis`
- **综合分**: 0.588  (分散度 0.829, 平均强度 0.542)
- **语义簇**: 日内时段与跨时区效应 × 波动率、尾部风险与高阶矩
- **经济逻辑**: Uses close, dollar_volume, open, return, returns with KeyError, TARGET_TZ.upper, _align_daily_to_original_index, _calc_chunk, _calc_daily_inputs to express session conditioned signal. / Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal.
- **训练期相似度**: rank=-0.008, return=NaN, holding=0.006
- **样本外最大 |rank|**: 0.019
- **强度分**: crypto_alpha_new_momentum_day_night_volume=0.447, crypto_alpha_realized_kurtosis=0.636 (强者: `crypto_alpha_realized_kurtosis`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_new_momentum_day_night_volume` | 50% | `crypto_alpha_realized_kurtosis` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_new_momentum_day_night_volume` | 30% | `crypto_alpha_realized_kurtosis` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_realized_kurtosis` | 70% | `crypto_alpha_new_momentum_day_night_volume` | 30% | 偏重强者 |

### crypto_alpha_realized_kurtosis + crypto_alpha_volume_surge_volatility

- **source**: `crypto_alpha_realized_kurtosis` + `crypto_alpha_volume_surge_volatility`
- **综合分**: 0.587  (分散度 0.761, 平均强度 0.580)
- **语义簇**: 波动率、尾部风险与高阶矩 × 日内时段与跨时区效应
- **经济逻辑**: Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal. / Uses close, dollar_volume, open, return, returns with KeyError, _align_daily_to_original_index, _calc_daily_raw_chunk_np, _chunk_columns, _cs_zscore to express session conditioned signal.
- **训练期相似度**: rank=0.186, return=NaN, holding=0.029
- **样本外最大 |rank|**: 0.300
- **强度分**: crypto_alpha_realized_kurtosis=0.636, crypto_alpha_volume_surge_volatility=0.524 (强者: `crypto_alpha_realized_kurtosis`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_realized_kurtosis` | 50% | `crypto_alpha_volume_surge_volatility` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_volume_surge_volatility` | 30% | `crypto_alpha_realized_kurtosis` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_realized_kurtosis` | 70% | `crypto_alpha_volume_surge_volatility` | 30% | 偏重强者 |

### crypto_alpha_volume_convergence_factor + crypto_alpha_volume_up_continuous_down_volatility_USA

- **source**: `crypto_alpha_volume_convergence_factor` + `crypto_alpha_volume_up_continuous_down_volatility_USA`
- **综合分**: 0.586  (分散度 0.739, 平均强度 0.605)
- **语义簇**: 量价关系与资金流 × 日内时段与跨时区效应
- **经济逻辑**: Uses return, volume with _calc_chunk, _chunk_columns, _convergence_factor, _convergence_factor(daily_volume).shift, daily_value.rolling to express volume price interaction. / Uses close, return, volume with (diff * diff).sum, (row_ss[valid_count] / count[valid_count]).astype, (row_sum[valid_count] / count[valid_count]).astype, (session_pos[block_ends - 1] + 1).astype, _calc_one_asset_volume_up_continuous_down_return_np to express session conditioned signal.
- **训练期相似度**: rank=0.237, return=NaN, holding=0.048
- **样本外最大 |rank|**: 0.259
- **强度分**: crypto_alpha_volume_convergence_factor=0.517, crypto_alpha_volume_up_continuous_down_volatility_USA=0.694 (强者: `crypto_alpha_volume_up_continuous_down_volatility_USA`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_volume_convergence_factor` | 50% | `crypto_alpha_volume_up_continuous_down_volatility_USA` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_volume_convergence_factor` | 30% | `crypto_alpha_volume_up_continuous_down_volatility_USA` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_volume_up_continuous_down_volatility_USA` | 70% | `crypto_alpha_volume_convergence_factor` | 30% | 偏重强者 |

### crypto_alpha_csk_xyy_up + crypto_alpha_new_momentum_day_night_volume

- **source**: `crypto_alpha_csk_xyy_up` + `crypto_alpha_new_momentum_day_night_volume`
- **综合分**: 0.581  (分散度 0.798, 平均强度 0.551)
- **语义簇**: 量价关系与资金流 × 日内时段与跨时区效应
- **经济逻辑**: Uses close, dollar_volume, return, volume with (daily_ret * weights).sum, TARGET_TZ.upper, _as_market_series, _build_market_ret, _calc_chunk to express volume price interaction. / Uses close, dollar_volume, open, return, returns with KeyError, TARGET_TZ.upper, _align_daily_to_original_index, _calc_chunk, _calc_daily_inputs to express session conditioned signal.
- **训练期相似度**: rank=0.101, return=NaN, holding=0.004
- **样本外最大 |rank|**: 0.182
- **强度分**: crypto_alpha_csk_xyy_up=0.654, crypto_alpha_new_momentum_day_night_volume=0.447 (强者: `crypto_alpha_csk_xyy_up`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_csk_xyy_up` | 50% | `crypto_alpha_new_momentum_day_night_volume` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_new_momentum_day_night_volume` | 30% | `crypto_alpha_csk_xyy_up` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_csk_xyy_up` | 70% | `crypto_alpha_new_momentum_day_night_volume` | 30% | 偏重强者 |

### crypto_alpha_realized_kurtosis + crypto_alpha_us_volume_dominance_USA

- **source**: `crypto_alpha_realized_kurtosis` + `crypto_alpha_us_volume_dominance_USA`
- **综合分**: 0.581  (分散度 0.824, 平均强度 0.508)
- **语义簇**: 波动率、尾部风险与高阶矩 × 日内时段与跨时区效应
- **经济逻辑**: Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal. / Uses return, volume with _calc_chunk, _chunk_columns, _daily_us_session_volume, _to_et_index, _to_et_index(volume).clip to express session conditioned signal.
- **训练期相似度**: rank=0.017, return=NaN, holding=0.011
- **样本外最大 |rank|**: 0.174
- **强度分**: crypto_alpha_realized_kurtosis=0.636, crypto_alpha_us_volume_dominance_USA=0.379 (强者: `crypto_alpha_realized_kurtosis`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_realized_kurtosis` | 50% | `crypto_alpha_us_volume_dominance_USA` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_us_volume_dominance_USA` | 30% | `crypto_alpha_realized_kurtosis` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_realized_kurtosis` | 70% | `crypto_alpha_us_volume_dominance_USA` | 30% | 偏重强者 |

### crypto_alpha_realized_kurtosis + crypto_alpha_us_volume_breakout_momentum_USA

- **source**: `crypto_alpha_realized_kurtosis` + `crypto_alpha_us_volume_breakout_momentum_USA`
- **综合分**: 0.576  (分散度 0.828, 平均强度 0.484)
- **语义簇**: 波动率、尾部风险与高阶矩 × 日内时段与跨时区效应
- **经济逻辑**: Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal. / Uses close, return, volume with _calc_chunk, _chunk_columns, _daily_us_session_return, _daily_us_session_volume, _to_et_index to express session conditioned signal.
- **训练期相似度**: rank=0.007, return=NaN, holding=0.009
- **样本外最大 |rank|**: 0.031
- **强度分**: crypto_alpha_realized_kurtosis=0.636, crypto_alpha_us_volume_breakout_momentum_USA=0.332 (强者: `crypto_alpha_realized_kurtosis`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_realized_kurtosis` | 50% | `crypto_alpha_us_volume_breakout_momentum_USA` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_us_volume_breakout_momentum_USA` | 30% | `crypto_alpha_realized_kurtosis` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_realized_kurtosis` | 70% | `crypto_alpha_us_volume_breakout_momentum_USA` | 30% | 偏重强者 |

### crypto_alpha_main_force_volatility_USA_session + crypto_alpha_price_jump_amount_corr

- **source**: `crypto_alpha_main_force_volatility` + `crypto_alpha_price_jump_amount_corr`
- **综合分**: 0.576  (分散度 0.745, 平均强度 0.590)
- **语义簇**: 日内时段与跨时区效应 × 量价关系与资金流
- **经济逻辑**: Uses close, return, volume with (diff * diff).sum, (row_ss[valid_count] / count[valid_count]).astype, (row_sum[valid_count] / count[valid_count]).astype, (session_pos[block_ends - 1] + 1).astype, _calc_one_asset_state_returns_np to express session conditioned signal. / Uses close, dollar_volume, high, low, return with (close_arr * vol_arr).astype, (~nan_mask).astype, TARGET_TZ.upper, _calc_column, _daily_jump_amount_corr_vec to express volume price interaction.
- **训练期相似度**: rank=0.222, return=NaN, holding=0.044
- **样本外最大 |rank|**: 0.402
- **强度分**: crypto_alpha_main_force_volatility=0.767, crypto_alpha_price_jump_amount_corr=0.413 (强者: `crypto_alpha_main_force_volatility`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_main_force_volatility` | 50% | `crypto_alpha_price_jump_amount_corr` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_price_jump_amount_corr` | 30% | `crypto_alpha_main_force_volatility` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_main_force_volatility` | 70% | `crypto_alpha_price_jump_amount_corr` | 30% | 偏重强者 |

### crypto_alpha_gk_vol_normalized_volume_convergence_factor_v2 + crypto_alpha_main_force_volatility_USA_session

- **source**: `crypto_alpha_dollar_volume_convergence_factor` + `crypto_alpha_main_force_volatility`
- **综合分**: 0.574  (分散度 0.827, 平均强度 0.507)
- **语义簇**: 量价关系与资金流 × 日内时段与跨时区效应
- **经济逻辑**: Uses close, dollar_volume, high, low, open with _calc_chunk_gk, _chunk_columns, _convergence_factor, _convergence_factor(daily_vnl).shift, daily_total_vol.rolling to express volume price interaction. / Uses close, return, volume with (diff * diff).sum, (row_ss[valid_count] / count[valid_count]).astype, (row_sum[valid_count] / count[valid_count]).astype, (session_pos[block_ends - 1] + 1).astype, _calc_one_asset_state_returns_np to express session conditioned signal.
- **训练期相似度**: rank=0.019, return=NaN, holding=0.001
- **样本外最大 |rank|**: 0.142
- **强度分**: crypto_alpha_dollar_volume_convergence_factor=0.247, crypto_alpha_main_force_volatility=0.767 (强者: `crypto_alpha_main_force_volatility`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_dollar_volume_convergence_factor` | 50% | `crypto_alpha_main_force_volatility` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_dollar_volume_convergence_factor` | 30% | `crypto_alpha_main_force_volatility` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_main_force_volatility` | 70% | `crypto_alpha_dollar_volume_convergence_factor` | 30% | 偏重强者 |

### crypto_alpha_gp_trade_count_mark_spread_corr_001 + crypto_alpha_realized_kurtosis

- **source**: `crypto_alpha_gp_trade_count_mark_spread_corr_001` + `crypto_alpha_realized_kurtosis`
- **综合分**: 0.568  (分散度 0.827, 平均强度 0.488)
- **语义簇**: 日内时段与跨时区效应 × 波动率、尾部风险与高阶矩
- **经济逻辑**: Uses close, funding, high, low, return with (cov / denom.where(denom > EPS)).where, (ranks - 1.0).div, (xc * xc).sum, (xc * yc).sum, (yc * yc).sum to express session conditioned signal. / Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal.
- **训练期相似度**: rank=-0.018, return=NaN, holding=0.000
- **样本外最大 |rank|**: 0.126
- **强度分**: crypto_alpha_gp_trade_count_mark_spread_corr_001=0.340, crypto_alpha_realized_kurtosis=0.636 (强者: `crypto_alpha_realized_kurtosis`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_gp_trade_count_mark_spread_corr_001` | 50% | `crypto_alpha_realized_kurtosis` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_gp_trade_count_mark_spread_corr_001` | 30% | `crypto_alpha_realized_kurtosis` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_realized_kurtosis` | 70% | `crypto_alpha_gp_trade_count_mark_spread_corr_001` | 30% | 偏重强者 |

### crypto_alpha_csk_xyy_up + crypto_alpha_gp_trade_count_mark_spread_corr_001

- **source**: `crypto_alpha_csk_xyy_up` + `crypto_alpha_gp_trade_count_mark_spread_corr_001`
- **综合分**: 0.568  (分散度 0.812, 平均强度 0.497)
- **语义簇**: 量价关系与资金流 × 日内时段与跨时区效应
- **经济逻辑**: Uses close, dollar_volume, return, volume with (daily_ret * weights).sum, TARGET_TZ.upper, _as_market_series, _build_market_ret, _calc_chunk to express volume price interaction. / Uses close, funding, high, low, return with (cov / denom.where(denom > EPS)).where, (ranks - 1.0).div, (xc * xc).sum, (xc * yc).sum, (yc * yc).sum to express session conditioned signal.
- **训练期相似度**: rank=0.063, return=NaN, holding=0.002
- **样本外最大 |rank|**: 0.122
- **强度分**: crypto_alpha_csk_xyy_up=0.654, crypto_alpha_gp_trade_count_mark_spread_corr_001=0.340 (强者: `crypto_alpha_csk_xyy_up`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_csk_xyy_up` | 50% | `crypto_alpha_gp_trade_count_mark_spread_corr_001` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_gp_trade_count_mark_spread_corr_001` | 30% | `crypto_alpha_csk_xyy_up` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_csk_xyy_up` | 70% | `crypto_alpha_gp_trade_count_mark_spread_corr_001` | 30% | 偏重强者 |

### crypto_alpha_gp_barra_094 + crypto_alpha_main_force_volatility_USA_session

- **source**: `crypto_alpha_gp_barra_094` + `crypto_alpha_main_force_volatility`
- **综合分**: 0.567  (分散度 0.784, 平均强度 0.514)
- **语义簇**: 波动率、尾部风险与高阶矩 × 日内时段与跨时区效应
- **经济逻辑**: Uses close, funding, high, low, open with (cov / (var_y + EPS)).where, (ranks - 1.0).div, (x * y).sum, (xy_mean - x_mean * y_mean).where, (y * y).sum to express session conditioned signal. / Uses close, return, volume with (diff * diff).sum, (row_ss[valid_count] / count[valid_count]).astype, (row_sum[valid_count] / count[valid_count]).astype, (session_pos[block_ends - 1] + 1).astype, _calc_one_asset_state_returns_np to express session conditioned signal.
- **训练期相似度**: rank=0.148, return=NaN, holding=0.000
- **样本外最大 |rank|**: 0.251
- **强度分**: crypto_alpha_gp_barra_094=0.261, crypto_alpha_main_force_volatility=0.767 (强者: `crypto_alpha_main_force_volatility`)

| 比例 | 因子 A | 权重 | 因子 B | 权重 | 说明 |
|------|--------|------|--------|------|------|
| 5-5 | `crypto_alpha_gp_barra_094` | 50% | `crypto_alpha_main_force_volatility` | 50% | 等权分散 |
| 3-7 | `crypto_alpha_gp_barra_094` | 30% | `crypto_alpha_main_force_volatility` | 70% | 偏重强者 |
| 7-3 | `crypto_alpha_main_force_volatility` | 70% | `crypto_alpha_gp_barra_094` | 30% | 偏重强者 |

## 使用建议

1. 对上表每一对，用 `fm.evaluate(DataFrame, profile_id="perp_1d", plot=True, 
   params={"start":"2022-01-01","end":"2026-07-01","cost":0.0015})` 跑六图绩效。
2. 把两个因子的缓存信号按 `locked_direction` 对齐后加权：
   ```python
   combined = (df_a * dir_a * w_a + df_b * dir_b * w_b) / (w_a + w_b)
   ```
3. 优先关注 `combined_score > 0.5` 且样本外漂移不大的组合。
4. 最终是否入库仍需人工复核经济逻辑、换手率和过拟合风险。
