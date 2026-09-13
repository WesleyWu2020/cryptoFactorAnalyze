# Unsubmit 因子分类与相关性地图

## 1. 范围、窗口与方法

覆盖 alphas/unsubmit 顶层 135 个 Python 因子；不递归读取 alpha191/ 与 ppo/。
训练期为 2022–2023，验证期为 2024–2025，测试观察期截至 2026-07-01。
高重复阈值：|横截面 Spearman| ≥ 0.80、|净收益相关性| ≥ 0.80、持仓重合度 ≥ 0.70。
低相似阈值：三项均低于 0.30；缓存无效或样本不足时不以 0 代替。

## 2. 语义簇概览

| primary_cluster | primary_cluster_label | factor_count |
| --- | --- | --- |
| behavioral_information | 行为与信息分布 | 0 |
| derivatives_carry_oi_basis | Funding、持仓量与基差信号 | 0 |
| intraday_session_timezone | 日内时段与跨时区效应 | 48 |
| liquidity_activity_capacity | 流动性、成交活跃度与容量 | 3 |
| market_relative_style_neutral | 市场相对、BTC 残差与风格中性 | 3 |
| mean_reversion_reversal | 均值回归与反转 | 6 |
| multi_mechanism_composite | 多机制复合信号 | 13 |
| order_flow_microstructure | 订单流与市场微观结构 | 8 |
| price_path_structure | 价格路径与形态结构 | 6 |
| price_volume_money_flow | 量价关系与资金流 | 33 |
| trend_momentum | 趋势与动量 | 6 |
| volatility_tail_higher_moments | 波动率、尾部风险与高阶矩 | 9 |

## 3. 135 个因子逐簇明细

### 趋势与动量

| factor_name | subcluster | economic_logic | data_needed | lookback_parameters | signal_level | locked_direction | cache_status | empirical_cluster | role | suggested_cross_cluster_factors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_alpha_001 | return_persistence | Uses close, return, volume with close.pct_change, float, raw.ewm, raw.ewm(span=1440, min_periods=240).mean, raw.replace to express volume price interaction. | close/volume |  | minutes | 1 | valid | E001 | independent |  |
| crypto_alpha_004 | directional_trend | Uses close, return, volume with close.pct_change, close.pct_change(fill_method=None).rolling, close.pct_change(fill_method=None).rolling(1440, min_periods=240).std, float, np.sqrt to express volume price interaction. | close/volume |  | minutes | -1 | valid | E003 | independent |  |
| crypto_alpha_acd_accumulation_distribution | directional_momentum | Uses close, high, low, return with (acd_short / LONG_WINDOW).shift, _calc_chunk, _chunk_columns, _daily_ohlc, close.index.normalize to express directional momentum. |  | LONG_WINDOW=20 | minutes | -1 | valid | E006 | independent |  |
| crypto_alpha_directional_momentum | directional_momentum | Uses close, return with (-signal_return).clip, (-signal_return).clip(lower=0).rolling, (-signal_return).clip(lower=0).rolling(WINDOW_BARS, min_periods=WINDOW_BARS).sum, close.resample, close.resample(SIGNAL_FREQ, origin='epoch', label='right', closed='right').last to express directional momentum. |  | WINDOW_DAYS=60 | minutes | 1 | valid | E019 | independent |  |
| crypto_alpha_gp_barra_042 | return_path | Uses close, high, low, open, return with (cov / (var_y + EPS)).where, (ranks - 1.0).div, (x * y).sum, (xy_mean - x_mean * y_mean).where, (y * y).sum to express session conditioned signal. |  |  | minutes | -1 | valid | E022 | independent |  |
| crypto_alpha_ppo_3_123_20260709150104 | price_peak_ema | Uses close, high, low, open, return with (block[:, valid] * weights[:, None]).sum, (centered * centered).sum, _align_frame, _daily_price_peak_minutes, _daily_tradable_mask to express session conditioned signal. |  |  | minutes | -1 | valid | E042 | independent |  |

### 均值回归与反转

| factor_name | subcluster | economic_logic | data_needed | lookback_parameters | signal_level | locked_direction | cache_status | empirical_cluster | role | suggested_cross_cluster_factors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_alpha_002 | short_horizon_reversal | Uses close, return, volume with close.pct_change, float, raw.replace, volume.rolling, volume.rolling(240, min_periods=60).mean to express volume price interaction. | close/volume |  | minutes | 1 | valid | E002 | independent |  |
| crypto_alpha_003 | cross_sectional_reversal | Uses close, return, volume with close.pct_change, float, raw.ewm, raw.ewm(span=4320, min_periods=720).mean, raw.replace to express volume price interaction. | close/volume |  | minutes | 1 | valid | E001 | independent |  |
| crypto_alpha_005 | return_reversal | Uses close, return, volume with close.pct_change, float, raw.ewm, raw.ewm(span=1440, min_periods=480).mean, raw.replace to express volume price interaction. | close/volume |  | minutes | 1 | valid | E004 | independent |  |
| crypto_alpha_max_single_day_return | cross_sectional_reversal | Uses close, return with TARGET_TZ.upper, _calc_chunk, _chunk_columns, _to_target_tz, _to_target_tz(close).astype to express cross sectional reversal. |  |  | minutes | -1 | valid | E037 | independent |  |
| crypto_alpha_price_ridge_minute_return | cross_sectional_reversal | Uses close, high, low, return, returns with (~nan_mask).astype, TARGET_TZ.upper, ThreadPoolExecutor, _calc_column, _day_codes to express cross sectional reversal. |  |  | minutes | -1 | valid | E048 | independent |  |
| crypto_alpha_reversal_maxret_mix | cross_sectional_reversal | Uses close, return with TARGET_TZ.upper, _chunk_columns, _cross_sectional_zscore, _daily_components, _daily_mix_from_components to express cross sectional reversal. |  |  | minutes | 1 | valid | E059 | independent |  |

### 波动率、尾部风险与高阶矩

| factor_name | subcluster | economic_logic | data_needed | lookback_parameters | signal_level | locked_direction | cache_status | empirical_cluster | role | suggested_cross_cluster_factors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_alpha_directional_volatility_efficiency | realized_risk_moments | Uses high, low, open, return with _calc_chunk, _calc_chunk_nb, data_ctx['high'].astype, data_ctx['low'].astype, data_ctx['open'].astype to express realized risk moments. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E020 | independent |  |
| crypto_alpha_gp_barra_001 | intraday_realized_volatility_path | Uses close, high, low, open, return with (cov / (var_y + EPS)).where, (ranks - 1.0).div, (x * y).sum, (xy_mean - x_mean * y_mean).where, (y * y).sum to express intraday realized volatility path. |  |  | minutes | -1 | valid | E022 | independent |  |
| crypto_alpha_gp_barra_002 | realized_volatility | Uses close, dollar_volume, high, low, open with (cov / (var_y + EPS)).where, (total / (count + EPS)).where, (x * y).sum, (xy_mean - x_mean * y_mean).where, (y * y).sum to express session conditioned signal. |  |  | minutes | 1 | valid | E022 | independent |  |
| crypto_alpha_gp_barra_094 | tail_risk | Uses close, funding, high, low, open with (cov / (var_y + EPS)).where, (ranks - 1.0).div, (x * y).sum, (xy_mean - x_mean * y_mean).where, (y * y).sum to express session conditioned signal. |  |  | minutes | 1 | valid | E028 | independent |  |
| crypto_alpha_price_ridge_interval_skew | realized_risk_moments | Uses close, high, low, return with (~nan_mask).astype, TARGET_TZ.upper, ThreadPoolExecutor, _calc_column, _daily_interval_skew_vec to express realized risk moments. |  |  | minutes | -1 | valid | E047 | independent |  |
| crypto_alpha_realized_kurtosis | realized_kurtosis | Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E053 | independent |  |
| crypto_alpha_realized_skewness | realized_skewness | Uses close, return with KeyError, _calc_chunk, _chunk_columns, close.index.normalize, close.where to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E054 | independent |  |
| crypto_alpha_return_skew_reversal | realized_risk_moments | Uses close, return with _calc_chunk, _calc_chunk_nb, close_chunk.to_numpy, data_ctx['close'].astype, int to express realized risk moments. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E054 | independent |  |
| crypto_alpha_volatility_efficiency | realized_risk_moments | Uses close, high, low, return with ((high_chunk - low_chunk) / prev_close).rolling, ((high_chunk - low_chunk) / prev_close).rolling(WINDOW_BARS, min_periods=WINDOW_BARS).mean, _calc_chunk, _chunk_columns, close_chunk.astype to express realized risk moments. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E020 | independent |  |

### 流动性、成交活跃度与容量

| factor_name | subcluster | economic_logic | data_needed | lookback_parameters | signal_level | locked_direction | cache_status | empirical_cluster | role | suggested_cross_cluster_factors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_alpha_dollar_volume_ma | dollar_volume_trend | Uses dollar_volume, return with (-dollar_volume_ma).rank, (-dollar_volume_ma).rank(axis=1, pct=True).rolling, (-dollar_volume_ma).rank(axis=1, pct=True).rolling(RANK_SMOOTH_DAYS, min_periods=RANK_SMOOTH_DAYS).mean, (-dollar_volume_ma).rank(axis=1, pct=True).rolling(RANK_SMOOTH_DAYS, min_periods=RANK_SMOOTH_DAYS).mean().shift, _calc_factor to express dollar volume trend. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E022 | independent |  |
| crypto_alpha_logvolume_intraday_std | intraday_volume_dispersion | Uses return, volume with _calc_chunk, _chunk_columns, daily_std.rolling, daily_std.rolling(WINDOW_DAYS, min_periods=WINDOW_DAYS).mean, daily_std.rolling(WINDOW_DAYS, min_periods=WINDOW_DAYS).mean().shift to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E022 | independent |  |
| crypto_alpha_resiliency_hp_fft | liquidity_capacity | Uses close, return, returns with KeyError, OUTPUT_ALPHA_DIRECTION.lower, _calc_chunk, _chunk_columns, _hp_cycle to express liquidity capacity. |  |  | minutes | -1 | valid | E056 | independent |  |

### 量价关系与资金流

| factor_name | subcluster | economic_logic | data_needed | lookback_parameters | signal_level | locked_direction | cache_status | empirical_cluster | role | suggested_cross_cluster_factors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_alpha_acceleration_short_reversal | volume_price_interaction | Uses close, dollar_volume, realized_vol, return with _apply_tail_filter, _chunk_columns, _daily_raw_signal, close.groupby, close.groupby(day_index).last to express volume price interaction. |  |  | minutes | 1 | valid | E005 | independent |  |
| crypto_alpha_ad_volume_momentum | volume_price_interaction | Uses close, high, low, return, volume with (daily_close - daily_low - (daily_high - daily_close)).div, _calc_chunk, _chunk_columns, _daily, close.index.normalize to express volume price interaction. |  | WINDOW_DAYS=5 | minutes | 1 | valid | E007 | independent |  |
| crypto_alpha_bias_neutralized_zscore | volume_price_interaction | Uses close, dollar_volume, return with (dev_x * dev_y).mean, (dev_x ** 2).mean, (dev_x ** 2).mean(axis=1).replace, X_chunk.mean, X_chunk.sub to express volume price interaction. |  | WINDOW_DAYS=120 | minutes | 1 | valid | E011 | independent |  |
| crypto_alpha_cgo_pos_neg_tail_split_groupfix | volume_price_interaction | Uses close, return, volume with ValueError, _calc_cgo_chunk_nb, _resample_15m, _resample_15m(data_ctx['close'], 'last').astype, _resample_15m(data_ctx['volume'], 'sum').astype to express volume price interaction. |  | TURNOVER_WINDOW_DAYS=60 | minutes | 1 | valid | E013 | independent |  |
| crypto_alpha_cgo_volume_ratio_turnover_conservative | volume_price_interaction | Uses close, return, volume with ValueError, _calc_cgo_chunk_nb, _map_daily_turnover, _resample_15m, _resample_15m(data_ctx['close'], 'last').astype to express volume price interaction. |  | TURNOVER_WINDOW_DAYS=60 | minutes | 1 | valid | E013 | independent |  |
| crypto_alpha_close_ultramem_10d_20260703 | volume_price_interaction | Uses close, high, low, return, volume with (vals * vals).sum, _cs_zscore_inplace, _daily_ret, _daily_score, _day_boundaries to express volume price interaction. |  | WINDOW_DAYS=10 | minutes | 1 | valid | E014 | independent |  |
| crypto_alpha_csk_xyy_down | volume_price_interaction | Uses close, dollar_volume, return, volume with (-result).shift, (-result).shift(1).replace, (-result).shift(1).replace([np.inf, -np.inf], np.nan).astype, (daily_ret * weights).sum, TARGET_TZ.upper to express volume price interaction. |  |  | minutes | -1 | valid | E016 | independent |  |
| crypto_alpha_csk_xyy_up | volume_price_interaction | Uses close, dollar_volume, return, volume with (daily_ret * weights).sum, TARGET_TZ.upper, _as_market_series, _build_market_ret, _calc_chunk to express volume price interaction. |  |  | minutes | 1 | valid | E017 | independent |  |
| crypto_alpha_csk_xyy_up_down | volume_price_interaction | Uses close, dollar_volume, return, volume with (daily_ret * weights).sum, (dy - beta * dx).astype, TARGET_TZ.upper, _as_market_series, _build_market_ret to express volume price interaction. |  |  | minutes | 1 | valid | E016 | independent |  |
| crypto_alpha_daily_rebalance_fit | volume_price_interaction | Uses close, dollar_volume, funding, high, low with ThreadPoolExecutor, calc_single_asset_numba, data_ctx['close'].astype, data_ctx['dollar_volume'].iloc[r:r_end].rank, data_ctx['dollar_volume'].iloc[r:r_end].rank(axis=1, pct=True).values.astype to express volume price interaction. | high/low/close/volume/funding/dollar_volume |  | minutes | -1 | valid | E035 | independent |  |
| crypto_alpha_gk_vol_normalized_volume_convergence_factor_v2 | volume_price_interaction | Uses close, dollar_volume, high, low, open with _calc_chunk_gk, _chunk_columns, _convergence_factor, _convergence_factor(daily_vnl).shift, daily_total_vol.rolling to express volume price interaction. |  |  | minutes | 1 | valid | E021 | independent |  |
| crypto_alpha_gp_barra_000 | volume_price_interaction | Uses close, high, low, open_interest, return with _as_aligned, _calc_daily_signal, _calc_daily_signal_chunk, _chunk_columns, _daily_rv20 to express volume price interaction. |  | RV_WINDOW_DAYS=20/TS_DECAY_WINDOW=20/WINDOW=240 | minutes | -1 | valid | E022 | independent |  |
| crypto_alpha_group_path_length_low_mem | volume_price_interaction | Uses close, funding, high, low, return with (minutes_of_day <= int(1440 * 0.3)).astype, ThreadPoolExecutor, abs, calc_single_asset_numba, data_ctx['close'][col_name].values.astype to express volume price interaction. | high/low/close/volume/funding |  | minutes | -1 | valid | E031 | independent |  |
| crypto_alpha_high_amplitude_excess_momentum_2 | volume_price_interaction | Uses close, dollar_volume, high, low, return with (valid_return * valid_weight).sum, (x_demean * x_demean).sum, (x_demean * x_demean).sum(axis=1).replace, (x_demean * y_demean).sum, _calc_chunk to express volume price interaction. |  |  | minutes | 1 | valid | E033 | independent |  |
| crypto_alpha_klinger_volume_oscillator | volume_price_interaction | Uses close, high, low, return, volume with _calc_chunk, _chunk_columns, _daily, close.index.normalize, data.resample to express volume price interaction. |  |  | minutes | -1 | valid | E034 | independent |  |
| crypto_alpha_mfi_dollar_volume | volume_price_interaction | Uses close, dollar_volume, high, low, return with (100.0 * positive_sum / total_sum.replace(0.0, np.nan)).shift, _calc_chunk, _chunk_columns, _daily, close.index.normalize to express volume price interaction. |  | WINDOW_DAYS=120 | minutes | 1 | valid | E038 | independent |  |
| crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | volume_price_interaction | Uses close, dollar_volume, high, low, return with _cross_section_rank, close.index.normalize, close.to_numpy, daily_dv.shift, daily_dv.shift(1).rolling to express volume price interaction. |  | DV_LOOKBACK_DAYS=30 | days | -1 | valid | E022 | independent |  |
| crypto_alpha_price_jump_amount_corr | volume_price_interaction | Uses close, dollar_volume, high, low, return with (close_arr * vol_arr).astype, (~nan_mask).astype, TARGET_TZ.upper, _calc_column, _daily_jump_amount_corr_vec to express volume price interaction. |  |  | minutes | 1 | valid | E044 | independent |  |
| crypto_alpha_price_valley_relative_vwap | volume_price_interaction | Uses close, dollar_volume, high, low, return with ((high_tz - low_tz) / close_tz).where, (dollar_volume / volume).where, (minute_vwap * volume_tz).where, (valley_vwap / total_vwap).where, TARGET_TZ.upper to express volume price interaction. |  |  | minutes | 1 | valid | E050 | independent |  |
| crypto_alpha_price_valley_vmacd_volume | volume_price_interaction | Uses close, dollar_volume, high, low, return with ((high_tz - low_tz) / close_tz).where, (diff - dea).shift, (diff - dea).shift(1).replace, (diff - dea).shift(1).replace([np.inf, -np.inf], np.nan).astype, (dollar_volume / volume).where to express volume price interaction. |  |  | minutes | -1 | valid | E050 | independent |  |
| crypto_alpha_price_valley_vwap_percentile | volume_price_interaction | Uses close, dollar_volume, high, low, return with ((high_tz - low_tz) / close_tz).where, (dollar_volume / volume).where, (minute_vwap * volume_tz).where, TARGET_TZ.upper, _calc_chunk to express volume price interaction. |  |  | minutes | -1 | valid | E051 | independent |  |
| crypto_alpha_price_volume_convergence_factor | volume_price_interaction | Uses close, return, volume with (_cross_sectional_zscore(pcf_daily) + _cross_sectional_zscore(vcf_daily)).shift, _chunk_columns, _convergence_factor, _cross_sectional_zscore, _daily_convergence_by_chunk to express volume price interaction. |  |  | minutes | -1 | valid | E052 | independent |  |
| crypto_alpha_rv_20d_resid_style_wma_rank | volume_price_interaction | Uses close, dollar_volume, return with (y_arr[row, idx] - x @ beta).astype, _calc_chunk, _calc_daily_resid_style, _chunk_columns, _cross_section_rank to express volume price interaction. |  |  | minutes | -1 | valid | E061 | independent |  |
| crypto_alpha_sig_up_p_v_ratio_USA | volume_price_interaction | Uses close, return, volume with _calc_chunk, _chunk_columns, _daily_sig_up_ratio, _to_et_index, _to_et_index(close).astype to express volume price interaction. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E064 | independent |  |
| crypto_alpha_sig_up_p_v_ratio_stability_USA | volume_price_interaction | Uses close, return, volume with _calc_chunk, _chunk_columns, _daily_sig_up_ratio, _to_et_index, _to_et_index(close).astype to express volume price interaction. |  | WINDOW_DAYS=10 | minutes | -1 | valid | E065 | independent |  |
| crypto_alpha_vmacd_volume | volume_price_interaction | Uses return, volume with (diff - dea).shift, _calc_chunk, _chunk_columns, daily_volume.ewm, daily_volume.ewm(span=FAST_SPAN, min_periods=FAST_SPAN, adjust=False).mean to express volume price interaction. |  |  | minutes | -1 | valid | E079 | independent |  |
| crypto_alpha_volatility_efficiency_mfi_dollar_volume | volume_price_interaction | Uses close, dollar_volume, high, low, return with ((daily_high - daily_low) / daily_close.shift(1)).rolling, ((daily_high - daily_low) / daily_close.shift(1)).rolling(VOL_WINDOW_DAYS, min_periods=VOL_WINDOW_DAYS).mean, (100.0 * positive_sum / total_sum.replace(0.0, np.nan)).shift, (flow_score * agreement_gate).replace, (flow_score * agreement_gate).replace([np.inf, -np.inf], np.nan).astype to express volume price interaction. |  |  | minutes | 1 | valid | E038 | independent |  |
| crypto_alpha_volume_amplitude_divergence_corr | volume_price_interaction | Uses close, high, low, return, volume with _daily_factor_from_arrays, _process_one_column, _rolling_sum_1d, c.index.equals, c.reindex to express volume price interaction. |  | WINDOW_DAYS=10 | minutes | 1 | valid | E081 | independent |  |
| crypto_alpha_volume_convergence_factor | volume_price_interaction | Uses return, volume with _calc_chunk, _chunk_columns, _convergence_factor, _convergence_factor(daily_volume).shift, daily_value.rolling to express volume price interaction. |  |  | minutes | 1 | valid | E082 | independent |  |
| crypto_alpha_volume_ratio_1_5_USA | volume_price_interaction | Uses return, volume with _calc_chunk, _calc_ratio_daily_col_nb, _prepare_time_arrays, day_ids.astype, enumerate to express volume price interaction. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E088 | independent |  |
| crypto_alpha_volume_short_mean_anomaly | volume_price_interaction | Uses return, volume with (np.log1p(daily_volume) - np.log1p(ref_mean)).shift, _calc_chunk, _chunk_columns, daily_volume.shift, daily_volume.shift(1).rolling to express volume price interaction. |  | WINDOW_DAYS=3 | minutes | -1 | valid | E094 | independent |  |
| crypto_alpha_volume_sorted_volatility_efficiency_ratio | volume_price_interaction | Uses close, high, low, return, volume with _calc_chunk, _calc_chunk_nb, close.index.normalize, close_chunk.to_numpy, day_ids.astype to express volume price interaction. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E097 | independent |  |
| crypto_alpha_vroc_dollar_volume | volume_price_interaction | Uses dollar_volume, return with ((daily_dollar_volume - ref) / ref.replace(0.0, np.nan) * 100.0).shift, _calc_chunk, _chunk_columns, daily_dollar_volume.shift, dollar_volume.clip to express volume price interaction. |  | LOOKBACK_DAYS=20 | minutes | 1 | valid | E099 | independent |  |

### 订单流与市场微观结构

| factor_name | subcluster | economic_logic | data_needed | lookback_parameters | signal_level | locked_direction | cache_status | empirical_cluster | role | suggested_cross_cluster_factors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpv_preemption_cdpdv_v | trade_flow_structure | Uses close, return, volume with (X_m * Y_m).rolling, (X_m * Y_m).rolling(window, min_periods=min_obs).sum, (X_m ** 2).rolling, (X_m ** 2).rolling(window, min_periods=min_obs).sum, (Y_m ** 2).rolling to express trade flow structure. | close/volume |  | minutes | 1 | valid | E015 | independent |  |
| crypto_alpha_price_retail_correlation_divergence | trade_flow_structure | Uses close, quote_volume, return, trade_count with _calc_chunk, _calc_chunk_nb, close_chunk.to_numpy, data_ctx['close'].astype, data_ctx['xbinance_quote_volume'].astype to express trade flow structure. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E046 | independent |  |
| crypto_alpha_reconstructed_cumulative_delta_order_flow | trade_flow_structure | Uses close, high, low, open, return with ValueError, _as_float_array, _calc_daily_delta_np, _calc_output_chunk_np, _chunk_columns to express trade flow structure. |  | WINDOW=50 | days | 1 | valid | E055 | independent |  |
| crypto_alpha_retail_activity_divergence | trade_flow_structure | Uses quote_volume, return, trade_count with _rank_divergence_nb, _rank_pct_row, _rolling_mean_chunk, _rolling_mean_chunk_nb, data_ctx['xbinance_quote_volume'].astype to express trade flow structure. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E057 | independent |  |
| crypto_alpha_retail_fomo_ratio | trade_flow_structure | Uses close, return, trade_count with _calc_chunk, _calc_chunk_nb, close_chunk.to_numpy, data_ctx['close'].astype, data_ctx['xbinance_trade_count'].astype to express trade flow structure. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E049 | independent |  |
| crypto_alpha_retail_friction_illiquidity | trade_flow_structure | Uses close, quote_volume, return, trade_count with _calc_chunk, _calc_chunk_nb, abs, close_chunk.to_numpy, data_ctx['close'].astype to express trade flow structure. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E058 | independent |  |
| crypto_alpha_volatility_order_flow_efficiency | trade_flow_structure | Uses close, high, low, quote_volume, return with _calc_chunk, _calc_chunk_nb, close_chunk.to_numpy, data_ctx['close'].astype, data_ctx['high'].astype to express trade flow structure. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E080 | independent |  |
| ms_siphon_effect | trade_flow_structure | Uses close, dollar_volume, return, volume with ((xv - x_mean) * (yv - y_mean)).sum, ((xv - x_mean) ** 2).sum, ((y - ym) ** 2).sum, _calc_daily_raw, _chunk_columns to express trade flow structure. |  |  | minutes | -1 | valid | E066 | independent |  |

### 价格路径与形态结构

| factor_name | subcluster | economic_logic | data_needed | lookback_parameters | signal_level | locked_direction | cache_status | empirical_cluster | role | suggested_cross_cluster_factors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_alpha_amplitude_sorted_momentum_ratio | price_path_geometry | Uses close, high, low, return with _calc_chunk, _calc_chunk_nb, close.index.normalize, close_chunk.to_numpy, day_ids.astype to express price path geometry. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E008 | independent |  |
| crypto_alpha_cum_return_ratio | price_path_geometry | Uses close, cum_return, return, returns with close.index.normalize, close.resample, close.resample('D').last, close_d.to_numpy, factor_d.reindex to express price path geometry. |  | INNER_WINDOW_DAYS=5/NUM_WINDOW_DAYS=40/OUTER_WINDOW_DAYS=10 | minutes | -1 | valid | E018 | independent |  |
| crypto_alpha_gp_barra_099 | price_path_shape | Uses close, high, low, open, return with (cov / (var_y + EPS)).where, (ranks - 1.0).div, (x * y).sum, (xy_mean - x_mean * y_mean).where, (y * y).sum to express session conditioned signal. |  |  | minutes | 1 | valid | E029 | independent |  |
| crypto_alpha_high_amplitude_decisive_momentum | price_path_geometry | Uses close, high, low, return with _calc_chunk, _calc_chunk_nb, _chunk_columns, close.index.normalize, close.to_numpy to express price path geometry. |  |  | minutes | 1 | valid | E032 | independent |  |
| crypto_alpha_price_convergence_factor | price_path_geometry | Uses close, return with _calc_chunk, _chunk_columns, _convergence_factor, _convergence_factor(daily_close).shift, close.index.normalize to express price path geometry. |  |  | minutes | 1 | valid | E043 | independent |  |
| crypto_alpha_price_peak_minutes_120d | price_path_geometry | Uses close, high, low, return with ((high - low) / close).where, (prev_valid & next_valid & prev_low.le(next_high) & next_low.le(prev_high)).fillna, _align_frame, _calc_chunk, _calc_daily_peak_count_chunk to express price path geometry. |  | WINDOW_DAYS=120 | minutes | -1 | valid | E045 | independent |  |

### 日内时段与跨时区效应

| factor_name | subcluster | economic_logic | data_needed | lookback_parameters | signal_level | locked_direction | cache_status | empirical_cluster | role | suggested_cross_cluster_factors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_alpha_asia_reversal_absorption_USA | session_conditioned_signal | Uses close, return, volume with (asia_minute >= ASIA_START_MINUTE).astype, _asia_session_day, _calc_chunk, _chunk_columns, _daily_asia_session_return to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E009 | independent |  |
| crypto_alpha_asia_session_reversal_USA | session_conditioned_signal | Uses close, return with _calc_chunk, _chunk_columns, _daily_asia_session_return, _to_et_index, _to_et_index(close).astype to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E010 | independent |  |
| crypto_alpha_body_ratio_sorted_momentum_ratio | session_conditioned_signal | Uses close, high, low, open, return with _calc_chunk, _calc_chunk_nb, _resample_us_session_chunk, _set_index_fast, _to_et_datetime_index to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E012 | independent |  |
| crypto_alpha_elastic_potential_gap | session_conditioned_signal | Uses close, dollar_volume, high, low, return with KeyError, _align_daily_to_original_index, _calc_daily_uptrend_totals_chunk_nb, _calc_daily_uptrend_totals_chunk_np, _chunk_columns to express session conditioned signal. |  |  | minutes | -1 | valid | E023 | independent |  |
| crypto_alpha_gp_barra_003 | session_conditioned_signal | Uses close, dollar_volume, high, low, open with (cov / (var_y + EPS)).where, (dx * dx).sum, (dx * dy).sum, (total / (count + EPS)).where, (x * y).sum to express session conditioned signal. |  |  | minutes | -1 | valid | E022 | independent |  |
| crypto_alpha_gp_barra_004 | session_conditioned_signal | Uses close, high, low, open, return with (ranks - 1.0).div, KeyError, ValueError, _align_to_utc_frame, _calc_daily_base to express session conditioned signal. |  |  | minutes | 1 | valid | E024 | independent |  |
| crypto_alpha_gp_barra_004 | session_conditioned_signal | Uses close, high, low, open, return with (ranks - 1.0).div, KeyError, ValueError, _align_to_utc_frame, _calc_daily_base to express session conditioned signal. |  |  | minutes | 1 | valid | E024 | independent |  |
| crypto_alpha_gp_barra_004 | session_conditioned_signal | Uses close, high, low, open, return with (ranks - 1.0).div, KeyError, ValueError, _align_to_utc_frame, _calc_daily_base to express session conditioned signal. |  |  | minutes | 1 | valid | E024 | independent |  |
| crypto_alpha_gp_barra_004_2 | session_conditioned_signal | Uses close, funding, high, low, return with (ranks - 1.0).div, KeyError, ValueError, _align_to_utc_frame, _calc_daily_base to express session conditioned signal. |  |  | minutes | 1 | valid | E025 | independent |  |
| crypto_alpha_gp_barra_006 | session_conditioned_signal | Uses close, funding, high, low, open with (close.index - day_index).total_seconds, (ranks - 1.0).div, (vol_window / (vol_mean + EPS)).reindex, KeyError, ValueError to express session conditioned signal. |  |  | minutes | 1 | valid | E027 | independent |  |
| crypto_alpha_gp_quote_volume_decay_001 | session_conditioned_signal | Uses close, open, quote_volume, return with _align_to_utc_frame, _intraday_window_mean, _replace_inf, _replace_inf(factor).astype, _ts_decay to express session conditioned signal. |  |  | minutes | -1 | valid | E022 | independent |  |
| crypto_alpha_gp_trade_count_mark_spread_corr_001 | session_conditioned_signal | Uses close, funding, high, low, return with (cov / denom.where(denom > EPS)).where, (ranks - 1.0).div, (xc * xc).sum, (xc * yc).sum, (yc * yc).sum to express session conditioned signal. |  |  | minutes | 1 | valid | E030 | independent |  |
| crypto_alpha_main_force_volatility_USA_session | session_conditioned_signal | Uses close, return, volume with (diff * diff).sum, (row_ss[valid_count] / count[valid_count]).astype, (row_sum[valid_count] / count[valid_count]).astype, (session_pos[block_ends - 1] + 1).astype, _calc_one_asset_state_returns_np to express session conditioned signal. |  |  | minutes | 1 | valid | E036 | independent |  |
| crypto_alpha_new_intraday_volume_adjusted | session_conditioned_signal | Uses close, dollar_volume, open, return, volume with KeyError, TARGET_TZ.upper, _align_daily_to_original_index, _all_days_from_target_index, _calc_chunk to express session conditioned signal. |  |  | minutes | 1 | valid | E039 | independent |  |
| crypto_alpha_new_momentum_day_night_volume | session_conditioned_signal | Uses close, dollar_volume, open, return, returns with KeyError, TARGET_TZ.upper, _align_daily_to_original_index, _calc_chunk, _calc_daily_inputs to express session conditioned signal. |  |  | minutes | 1 | valid | E040 | independent |  |
| crypto_alpha_old_overnight_momentum | session_conditioned_signal | Uses close, open, return with (~np.isfinite(log_r)).astype, TARGET_TZ.upper, ThreadPoolExecutor, _build_day_ids, _calc_chunk to express session conditioned signal. |  |  | minutes | 1 | valid | E041 | independent |  |
| crypto_alpha_ppo_6_123_20260706165517 | session_conditioned_signal | Uses close, cum_return, dollar_volume, high, low with ((row_values - mean) / std).astype, (block[:, valid] * weights[:, None]).sum, (centered * centered).sum, (centered ** 3).sum, (sq_diff * w).sum to express session conditioned signal. |  |  | minutes |  | not_cached |  | ensemble |  |
| crypto_alpha_price_ridge_minutes_USA | session_conditioned_signal | Uses close, high, low, return with TARGET_TZ.upper, _calc_column, _chunk_columns, _daily_std, _day_codes to express session conditioned signal. |  |  | minutes | -1 | valid | E049 | independent |  |
| crypto_alpha_rpv_stability_USA | session_conditioned_signal | Uses close, return, volume with (cum[window:] - cum[:-window]).astype, _to_et_index, close_df.values.astype, cs_zscore_numpy, et_times.normalize to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E060 | independent |  |
| crypto_alpha_sig_up_p_v_intraday_std_USA | session_conditioned_signal | Uses close, return, volume with (sig_volume_std_daily / daily_avg_volume).rank, _calc_chunk, _chunk_columns, _daily_sig_up_intraday_std, _to_et_index to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E062 | independent |  |
| crypto_alpha_sig_up_p_v_intraday_std_stability_USA | session_conditioned_signal | Uses close, return, volume with _calc_chunk, _chunk_columns, _daily_sig_up_intraday_std_series, _to_et_index, close.astype to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E063 | independent |  |
| crypto_alpha_skewed_deviation_return | session_conditioned_signal | Uses close, return with KeyError, _align_daily_to_original_index, _calc_daily_deviation_sum_np, _get_frame, _get_target_day_info to express session conditioned signal. |  |  | minutes | 1 | valid | E067 | independent |  |
| crypto_alpha_structured_reversal_USA | session_conditioned_signal | Uses close, return, volume with (-structured_return).rolling, (-structured_return).rolling(WINDOW_DAYS, min_periods=WINDOW_DAYS).mean, (-structured_return).rolling(WINDOW_DAYS, min_periods=WINDOW_DAYS).mean().shift, _calc_chunk, _chunk_columns to express session conditioned signal. |  |  | minutes | 1 | valid | E068 | independent |  |
| crypto_alpha_uid_devol | session_conditioned_signal | Uses close, return with (-residual).shift, (-residual).shift(1).astype, (yv - fit).astype, TARGET_TZ.upper, _align_daily_to_original_index to express session conditioned signal. |  |  | minutes | 1 | valid | E069 | independent |  |
| crypto_alpha_uid_information_uniformity | session_conditioned_signal | Uses close, return with TARGET_TZ.upper, _align_daily_to_original_index, _calc_chunk, _calc_daily_uid, _chunk_columns to express session conditioned signal. |  |  | minutes | 1 | valid | E069 | independent |  |
| crypto_alpha_us_asia_mom_reversal_spread_USA | session_conditioned_signal | Uses close, return with _calc_chunk, _chunk_columns, _daily_asia_session_return, _daily_us_session_return, _to_et_index to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E070 | independent |  |
| crypto_alpha_us_session_momentum_USA | session_conditioned_signal | Uses close, return with _calc_chunk, _chunk_columns, _daily_us_session_return, _to_et_index, _to_et_index(close).astype to express session conditioned signal. |  | WINDOW_DAYS=10 | minutes | 1 | valid | E071 | independent |  |
| crypto_alpha_us_trend_persistence_USA | session_conditioned_signal | Uses close, return with _calc_chunk, _chunk_columns, _daily_us_session_return, _to_et_index, _to_et_index(close).astype to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E072 | independent |  |
| crypto_alpha_us_up_asia_down_consistency_USA | session_conditioned_signal | Uses close, return with ((us_return > 0) & (asia_return.shift(-1) < 0)).astype, _calc_chunk, _chunk_columns, _daily_asia_session_return, _daily_us_session_return to express session conditioned signal. |  | WINDOW_DAYS=30 | minutes | 1 | valid | E073 | independent |  |
| crypto_alpha_us_volume_breakout_momentum_USA | session_conditioned_signal | Uses close, return, volume with _calc_chunk, _chunk_columns, _daily_us_session_return, _daily_us_session_volume, _to_et_index to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E074 | independent |  |
| crypto_alpha_us_volume_dominance_USA | session_conditioned_signal | Uses return, volume with _calc_chunk, _chunk_columns, _daily_us_session_volume, _to_et_index, _to_et_index(volume).clip to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E075 | independent |  |
| crypto_alpha_us_volume_weighted_momentum_USA | session_conditioned_signal | Uses close, return, volume with _calc_chunk, _chunk_columns, _daily_us_session_return, _daily_us_session_volume, _to_et_index to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E076 | independent |  |
| crypto_alpha_v_p_reversal_USA | session_conditioned_signal | Uses close, return, volume with _calc_chunk, _chunk_columns, _daily_reversal_value, _daily_v_p_reversal_series, _resample_intraday_series to express session conditioned signal. |  | WINDOW_DAYS=30 | minutes | 1 | valid | E077 | independent |  |
| crypto_alpha_v_p_skewness_USA | session_conditioned_signal | Uses close, return, volume with _calc_chunk, _chunk_columns, _daily_v_p_skewness, _pearson_median_skew, _resample_intraday to express session conditioned signal. |  | WINDOW_DAYS=30 | minutes | 1 | valid | E078 | independent |  |
| crypto_alpha_volume_down_fall_volatility_ASIA_session | session_conditioned_signal | Uses close, return, volume with (diff * diff).sum, (row_ss[valid_count] / count[valid_count]).astype, (row_sum[valid_count] / count[valid_count]).astype, (session_pos[block_ends - 1] + 1).astype, _calc_one_asset_volume_down_fall_return_np to express session conditioned signal. |  |  | minutes | -1 | valid | E083 | independent |  |
| crypto_alpha_volume_energy_divergence | session_conditioned_signal | Uses close, dollar_volume, return, volume, vwap with KeyError, _align_daily_to_original_index, _calc_daily_raw_chunk_np, _chunk_columns, _cs_zscore to express session conditioned signal. |  |  | minutes | 1 | valid | E084 | independent |  |
| crypto_alpha_volume_intraday_std_1_USA | session_conditioned_signal | Uses return, volume with _calc_chunk, _chunk_columns, _daily_active_std, _to_et_index, _to_et_index(volume).clip to express session conditioned signal. |  | WINDOW_DAYS=60 | minutes | -1 | valid | E085 | independent |  |
| crypto_alpha_volume_intraday_std_1_stability_USA | session_conditioned_signal | Uses return, volume with _calc_chunk, _chunk_columns, _daily_active_std, _to_et_index, _to_et_index(volume).clip to express session conditioned signal. |  | WINDOW_DAYS=30 | minutes | 1 | valid | E086 | independent |  |
| crypto_alpha_volume_price_intraday_corr_USA | session_conditioned_signal | Uses close, return, volume with (x * x).groupby, (x * x).groupby(day_index).mean, (x * y).groupby, (x * y).groupby(day_index).mean, (x.notna() & y.notna()).groupby to express session conditioned signal. |  |  | minutes | -1 | valid | E087 | independent |  |
| crypto_alpha_volume_ratio_1_5_USA | session_conditioned_signal | Uses return, volume with _calc_chunk, _chunk_columns, _daily_dual_session_ratio, _to_local_index, _to_local_index(volume, ASIA_TZ).clip to express session conditioned signal. |  | WINDOW_DAYS=10 | minutes | 1 | valid | E092 | independent |  |
| crypto_alpha_volume_ratio_1_5_stability_USA | session_conditioned_signal | Uses return, volume with (-rank_daily - TAIL_THRESHOLD).clip, (rank_daily - TAIL_THRESHOLD).clip, _calc_chunk, _chunk_columns, _cs_rank_scale to express session conditioned signal. |  | WINDOW_DAYS=30 | minutes | 1 | valid | E089 | independent |  |
| crypto_alpha_volume_ratio_8_4_USA | session_conditioned_signal | Uses return, volume with _calc_chunk, _chunk_columns, _daily_dual_session_ratio, _to_et_index, _to_et_index(volume).clip to express session conditioned signal. |  | WINDOW_DAYS=10 | minutes | 1 | valid | E090 | independent |  |
| crypto_alpha_volume_ratio_USA | session_conditioned_signal | Uses return, volume with _calc_chunk, _chunk_columns, _daily_active_ratio, _to_et_index, _to_et_index(volume).clip to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E091 | independent |  |
| crypto_alpha_volume_ratio_stability_USA | session_conditioned_signal | Uses return, volume with _calc_chunk, _chunk_columns, _daily_active_ratio, _to_et_index, _to_et_index(volume).clip to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | 1 | valid | E093 | independent |  |
| crypto_alpha_volume_sorted_amplitude_ratio | session_conditioned_signal | Uses close, high, low, return, volume with _calc_chunk, _calc_chunk_nb, _resample_us_session_chunk, _set_index_fast, _to_et_datetime_index to express session conditioned signal. |  | WINDOW_DAYS=20 | minutes | -1 | valid | E095 | independent |  |
| crypto_alpha_volume_sorted_momentum_ratio | session_conditioned_signal | Uses close, return, volume with _calc_chunk, _calc_chunk_nb, _resample_us_session_chunk, _set_index_fast, _to_et_datetime_index to express session conditioned signal. |  | WINDOW_DAYS=30 | minutes | 1 | valid | E096 | independent |  |
| crypto_alpha_volume_surge_volatility | session_conditioned_signal | Uses close, dollar_volume, open, return, returns with KeyError, _align_daily_to_original_index, _calc_daily_raw_chunk_np, _chunk_columns, _cs_zscore to express session conditioned signal. |  |  | minutes | 1 | valid | E098 | independent |  |
| crypto_alpha_volume_up_continuous_down_volatility_USA | session_conditioned_signal | Uses close, return, volume with (diff * diff).sum, (row_ss[valid_count] / count[valid_count]).astype, (row_sum[valid_count] / count[valid_count]).astype, (session_pos[block_ends - 1] + 1).astype, _calc_one_asset_volume_up_continuous_down_return_np to express session conditioned signal. |  |  | minutes | -1 | valid | E036 | independent |  |

### 市场相对、BTC 残差与风格中性

| factor_name | subcluster | economic_logic | data_needed | lookback_parameters | signal_level | locked_direction | cache_status | empirical_cluster | role | suggested_cross_cluster_factors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_alpha_gp_barra_005 | btc_relative_residual | Uses close, high, low, open, return with (ranks - 1.0).div, KeyError, ValueError, _calc_daily_base, _cs_zscore to express btc relative residual. |  |  | minutes | -1 | valid | E026 | independent |  |
| crypto_alpha_gp_barra_money_flow_btc_strength_dispersion | btc_relative_residual | Uses close, dollar_volume, high, low, open with (cov / (var_y + EPS)).where, (total / (count + EPS)).where, (x * y).sum, (xy_mean - x_mean * y_mean).where, (y * y).sum to express btc relative residual. |  |  | minutes | -1 | valid | E022 | independent |  |
| crypto_alpha_ppo_3_123_20260709150104_resid_return_btc | btc_relative_residual | Uses close, high, low, open, return with (block[:, valid] * weights[:, None]).sum, _align_frame, _daily_tradable_mask, _find_btc_column, _optional_frame to express btc relative residual. |  |  | minutes | 1 | valid | E018 | independent |  |

### 多机制复合信号

| factor_name | subcluster | economic_logic | data_needed | lookback_parameters | signal_level | locked_direction | cache_status | empirical_cluster | role | suggested_cross_cluster_factors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_alpha_ensemble_trajectory_rv_closeposition_csk_mark_rv_20260703 | weighted_component_ensemble | Uses close, dollar_volume, high, low, return with ((row_values - mean) / std).astype, (daily_close.notna() & daily_tradable).to_numpy, (num[ok] / denom[ok]).astype, (safe_dr * dm2[:, None]).sum, (safe_dr * safe_dr).sum to express weighted component ensemble. |  | DOLLAR_VOLUME_LOOKBACK_DAYS=30/TRAJECTORY_WINDOW_DAYS=20/WINDOW_DAYS=40 | minutes |  | not_cached |  | ensemble |  |
| crypto_alpha_ppo_10_0_20260608_ensemble | weighted_component_ensemble | Uses close, dollar_volume, funding, high, low with FileNotFoundError, ValueError, _add_weighted, _as_aligned, _calc_factor_core to express weighted component ensemble. |  | MAX_WINDOW_DAYS=40 | days |  | not_cached |  | ensemble |  |
| crypto_alpha_ppo_10_0_20260613160816_ensemble | weighted_component_ensemble | Uses close, cum_return, return with _calc_daily_cum_return, _resample_last, close.index.normalize, cum_return.rolling, cum_return.rolling(STD_WINDOW_DAYS, min_periods=STD_WINDOW_DAYS).std to express weighted component ensemble. |  | STD_WINDOW_DAYS=5 | minutes |  | not_cached |  | ensemble |  |
| crypto_alpha_ppo_12_0_20260626130557_ensemble | weighted_component_ensemble | Uses close, cum_return, high, low, return with (centered * centered).sum, _calc_daily_cum_return, _calc_daily_logvolume_intraday_std_20d, _calc_daily_price_peak_minutes_30d, _daily_tradable_mask to express weighted component ensemble. |  | LOGVOLUME_WINDOW_DAYS=20/PRICE_PEAK_WINDOW_DAYS=30 | days |  | not_cached |  | ensemble |  |
| crypto_alpha_ppo_12_0_20260628160526_ensemble | weighted_component_ensemble | Uses close, cum_return, dollar_volume, funding, high with ((row_values - mean) / std).astype, (-oi_z20).astype, (centered * centered).sum, (centered ** 2).sum, (centered ** 3).sum to express weighted component ensemble. |  | DOLLAR_VOLUME_LOOKBACK_DAYS=30/FUNDING_CUMUL_WINDOW_DAYS=5/LOGVOLUME_WINDOW_DAYS=20/PRICE_PEAK_WINDOW_DAYS=30/RIGHT_TAIL_WINDOW_DAYS=20 | days |  | not_cached |  | ensemble |  |
| crypto_alpha_ppo_12_17_20260628221245_ensemble | weighted_component_ensemble | Uses close, cum_return, dollar_volume, funding, high with ((daily_high - daily_low) / (daily_close + EPS)).shift, ((row_values - mean) / std).astype, (-oi_z20).astype, (centered * centered).sum, (centered ** 2).sum to express weighted component ensemble. |  | DOLLAR_VOLUME_LOOKBACK_DAYS=30/FUNDING_CUMUL_WINDOW_DAYS=5/LOGVOLUME_WINDOW_DAYS=20/PRICE_PEAK_WINDOW_DAYS=30/RIGHT_TAIL_WINDOW_DAYS=20 | days |  | not_cached |  | ensemble |  |
| crypto_alpha_ppo_12_1_20260628163318_ensemble | weighted_component_ensemble | Uses close, cum_return, dollar_volume, funding, high with ((row_values - mean) / std).astype, (-oi_z20).astype, (centered * centered).sum, (centered ** 2).sum, (centered ** 3).sum to express weighted component ensemble. |  | DOLLAR_VOLUME_LOOKBACK_DAYS=30/FUNDING_CUMUL_WINDOW_DAYS=5/LOGVOLUME_WINDOW_DAYS=20/PRICE_PEAK_WINDOW_DAYS=30/RIGHT_TAIL_WINDOW_DAYS=20 | days |  | not_cached |  | ensemble |  |
| crypto_alpha_ppo_12_2_20260628170032_ensemble | weighted_component_ensemble | Uses close, cum_return, dollar_volume, funding, high with ((row_values - mean) / std).astype, (-oi_z20).astype, (centered * centered).sum, (centered ** 2).sum, (centered ** 3).sum to express weighted component ensemble. |  | DOLLAR_VOLUME_LOOKBACK_DAYS=30/FUNDING_CUMUL_WINDOW_DAYS=5/LOGVOLUME_WINDOW_DAYS=20/PRICE_PEAK_WINDOW_DAYS=30/RIGHT_TAIL_WINDOW_DAYS=20 | days |  | not_cached |  | ensemble |  |
| crypto_alpha_ppo_12_3_20260628173014_ensemble | weighted_component_ensemble | Uses close, cum_return, dollar_volume, funding, high with ((row_values - mean) / std).astype, (-oi_z20).astype, (centered * centered).sum, (centered ** 2).sum, (centered ** 3).sum to express weighted component ensemble. |  | DOLLAR_VOLUME_LOOKBACK_DAYS=30/FUNDING_CUMUL_WINDOW_DAYS=5/LOGVOLUME_WINDOW_DAYS=20/PRICE_PEAK_WINDOW_DAYS=30/RIGHT_TAIL_WINDOW_DAYS=20 | days |  | not_cached |  | ensemble |  |
| crypto_alpha_ppo_12_42_20260628103854_ensemble | weighted_component_ensemble | Uses close, cum_return, dollar_volume, funding, high with ((row_values - mean) / std).astype, (-oi_z20).astype, (centered * centered).sum, (centered ** 3).sum, (safe_dr * dm2[:, None]).sum to express weighted component ensemble. |  | DOLLAR_VOLUME_LOOKBACK_DAYS=30/FUNDING_CUMUL_WINDOW_DAYS=5/LOGVOLUME_WINDOW_DAYS=20/PRICE_PEAK_WINDOW_DAYS=30/RIGHT_TAIL_WINDOW_DAYS=20 | days |  | not_cached |  | ensemble |  |
| crypto_alpha_ppo_12_4_20260628201213_ensemble | weighted_component_ensemble | Uses close, cum_return, dollar_volume, funding, high with ((daily_high - daily_low) / (daily_close + EPS)).shift, ((row_values - mean) / std).astype, (-oi_z20).astype, (centered * centered).sum, (centered ** 2).sum to express weighted component ensemble. |  | DOLLAR_VOLUME_LOOKBACK_DAYS=30/FUNDING_CUMUL_WINDOW_DAYS=5/LOGVOLUME_WINDOW_DAYS=20/PRICE_PEAK_WINDOW_DAYS=30/RIGHT_TAIL_WINDOW_DAYS=20 | days |  | not_cached |  | ensemble |  |
| crypto_alpha_rv_cum_var_liquidity_ensemble | weighted_component_ensemble | Uses close, cum_return, dollar_volume, return with (y_arr[row, idx] - x @ beta).astype, _calc_daily_cum_return, _calc_factor_core, _cross_section_rank, _daily_base_features to express weighted component ensemble. |  | BETA_WINDOW_DAYS=60/CORR_WINDOW_DAYS=20/DOLLAR_VOLUME_LOOKBACK_DAYS=30/MEAN_WINDOW_DAYS=10/RANK_WINDOW_DAYS=40/VAR_LONG_WINDOW_DAYS=10/VAR_SHORT_WINDOW_DAYS=5 | days |  | not_cached |  | ensemble |  |
| crypto_alpha_rv_resid_btc_cum_liquidity_ensemble | weighted_component_ensemble | Uses close, cum_return, dollar_volume, resid_btc, return with (y_arr[row, idx] - x @ beta).astype, _calc_daily_cum_return, _calc_factor_core, _cross_section_rank, _daily_base_features to express weighted component ensemble. |  | BETA_WINDOW_DAYS=60/CORR_WINDOW_DAYS=20/DOLLAR_VOLUME_LOOKBACK_DAYS=30/MAD_WINDOW_DAYS=10/MEAN_WINDOW_DAYS=10/RANK_WINDOW_DAYS=40 | days |  | not_cached |  | ensemble |  |

## 4. 实证相关性簇

### E001

导航代表：crypto_alpha_001。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_001 | 趋势与动量 | crypto_alpha_003 |
| crypto_alpha_003 | 均值回归与反转 | crypto_alpha_001 |

### E002

导航代表：crypto_alpha_002。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_002 | 均值回归与反转 |  |

### E003

导航代表：crypto_alpha_004。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_004 | 趋势与动量 |  |

### E004

导航代表：crypto_alpha_005。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_005 | 均值回归与反转 |  |

### E005

导航代表：crypto_alpha_acceleration_short_reversal。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_acceleration_short_reversal | 量价关系与资金流 |  |

### E006

导航代表：crypto_alpha_acd_accumulation_distribution。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_acd_accumulation_distribution | 趋势与动量 |  |

### E007

导航代表：crypto_alpha_ad_volume_momentum。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_ad_volume_momentum | 量价关系与资金流 |  |

### E008

导航代表：crypto_alpha_amplitude_sorted_momentum_ratio。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_amplitude_sorted_momentum_ratio | 价格路径与形态结构 |  |

### E009

导航代表：crypto_alpha_asia_reversal_absorption_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_asia_reversal_absorption_USA | 日内时段与跨时区效应 |  |

### E010

导航代表：crypto_alpha_asia_session_reversal_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_asia_session_reversal_USA | 日内时段与跨时区效应 |  |

### E011

导航代表：crypto_alpha_bias_neutralized_zscore。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_bias_neutralized_zscore | 量价关系与资金流 |  |

### E012

导航代表：crypto_alpha_body_ratio_sorted_momentum_ratio。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_body_ratio_sorted_momentum_ratio | 日内时段与跨时区效应 |  |

### E013

导航代表：crypto_alpha_cgo_volume_ratio_turnover_conservative。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_cgo_pos_neg_tail_split_groupfix | 量价关系与资金流 | crypto_alpha_cgo_volume_ratio_turnover_conservative |
| crypto_alpha_cgo_volume_ratio_turnover_conservative | 量价关系与资金流 | crypto_alpha_cgo_pos_neg_tail_split_groupfix |

### E014

导航代表：crypto_alpha_close_ultramem_10d_20260703。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_close_ultramem_10d_20260703 | 量价关系与资金流 |  |

### E015

导航代表：cpv_preemption_cdpdv_v。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| cpv_preemption_cdpdv_v | 订单流与市场微观结构 |  |

### E016

导航代表：crypto_alpha_csk_xyy_up_down。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_csk_xyy_down | 量价关系与资金流 | crypto_alpha_csk_xyy_up_down |
| crypto_alpha_csk_xyy_up_down | 量价关系与资金流 | crypto_alpha_csk_xyy_down |

### E017

导航代表：crypto_alpha_csk_xyy_up。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_csk_xyy_up | 量价关系与资金流 |  |

### E018

导航代表：crypto_alpha_ppo_3_123_20260709150104_resid_return_btc。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_cum_return_ratio | 价格路径与形态结构 | crypto_alpha_ppo_3_123_20260709150104_resid_return_btc |
| crypto_alpha_ppo_3_123_20260709150104_resid_return_btc | 市场相对、BTC 残差与风格中性 | crypto_alpha_cum_return_ratio |

### E019

导航代表：crypto_alpha_directional_momentum。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_directional_momentum | 趋势与动量 |  |

### E020

导航代表：crypto_alpha_volatility_efficiency。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_directional_volatility_efficiency | 波动率、尾部风险与高阶矩 | crypto_alpha_volatility_efficiency |
| crypto_alpha_volatility_efficiency | 波动率、尾部风险与高阶矩 | crypto_alpha_directional_volatility_efficiency |

### E021

导航代表：crypto_alpha_gk_vol_normalized_volume_convergence_factor_v2。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_gk_vol_normalized_volume_convergence_factor_v2 | 量价关系与资金流 |  |

### E022

导航代表：crypto_alpha_gp_quote_volume_decay_001。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_dollar_volume_ma | 流动性、成交活跃度与容量 | crypto_alpha_gp_barra_000/crypto_alpha_gp_barra_001/crypto_alpha_gp_barra_002/crypto_alpha_gp_barra_003/crypto_alpha_gp_barra_042/crypto_alpha_gp_barra_money_flow_btc_strength_dispersion/crypto_alpha_gp_quote_volume_decay_001/crypto_alpha_logvolume_intraday_std/crypto_alpha_ppo_8_0_20260625_dollar_volume_rank |
| crypto_alpha_gp_barra_000 | 量价关系与资金流 | crypto_alpha_dollar_volume_ma/crypto_alpha_gp_barra_001/crypto_alpha_gp_barra_002/crypto_alpha_gp_barra_003/crypto_alpha_gp_barra_042/crypto_alpha_gp_barra_money_flow_btc_strength_dispersion/crypto_alpha_gp_quote_volume_decay_001/crypto_alpha_ppo_8_0_20260625_dollar_volume_rank |
| crypto_alpha_gp_barra_001 | 波动率、尾部风险与高阶矩 | crypto_alpha_dollar_volume_ma/crypto_alpha_gp_barra_000/crypto_alpha_gp_barra_002/crypto_alpha_gp_barra_003/crypto_alpha_gp_barra_042/crypto_alpha_gp_barra_money_flow_btc_strength_dispersion/crypto_alpha_logvolume_intraday_std/crypto_alpha_ppo_8_0_20260625_dollar_volume_rank |
| crypto_alpha_gp_barra_002 | 波动率、尾部风险与高阶矩 | crypto_alpha_dollar_volume_ma/crypto_alpha_gp_barra_000/crypto_alpha_gp_barra_001/crypto_alpha_gp_barra_003/crypto_alpha_gp_barra_042/crypto_alpha_gp_barra_money_flow_btc_strength_dispersion/crypto_alpha_gp_quote_volume_decay_001/crypto_alpha_ppo_8_0_20260625_dollar_volume_rank |
| crypto_alpha_gp_barra_003 | 日内时段与跨时区效应 | crypto_alpha_dollar_volume_ma/crypto_alpha_gp_barra_000/crypto_alpha_gp_barra_001/crypto_alpha_gp_barra_002/crypto_alpha_gp_barra_042/crypto_alpha_gp_barra_money_flow_btc_strength_dispersion/crypto_alpha_gp_quote_volume_decay_001/crypto_alpha_logvolume_intraday_std/crypto_alpha_ppo_8_0_20260625_dollar_volume_rank |
| crypto_alpha_gp_barra_042 | 趋势与动量 | crypto_alpha_dollar_volume_ma/crypto_alpha_gp_barra_000/crypto_alpha_gp_barra_001/crypto_alpha_gp_barra_002/crypto_alpha_gp_barra_003/crypto_alpha_gp_barra_money_flow_btc_strength_dispersion/crypto_alpha_gp_quote_volume_decay_001/crypto_alpha_logvolume_intraday_std/crypto_alpha_ppo_8_0_20260625_dollar_volume_rank |
| crypto_alpha_gp_barra_money_flow_btc_strength_dispersion | 市场相对、BTC 残差与风格中性 | crypto_alpha_dollar_volume_ma/crypto_alpha_gp_barra_000/crypto_alpha_gp_barra_001/crypto_alpha_gp_barra_002/crypto_alpha_gp_barra_003/crypto_alpha_gp_barra_042/crypto_alpha_gp_quote_volume_decay_001/crypto_alpha_ppo_8_0_20260625_dollar_volume_rank |
| crypto_alpha_gp_quote_volume_decay_001 | 日内时段与跨时区效应 | crypto_alpha_dollar_volume_ma/crypto_alpha_gp_barra_000/crypto_alpha_gp_barra_002/crypto_alpha_gp_barra_003/crypto_alpha_gp_barra_042/crypto_alpha_gp_barra_money_flow_btc_strength_dispersion/crypto_alpha_ppo_8_0_20260625_dollar_volume_rank |
| crypto_alpha_logvolume_intraday_std | 流动性、成交活跃度与容量 | crypto_alpha_dollar_volume_ma/crypto_alpha_gp_barra_001/crypto_alpha_gp_barra_003/crypto_alpha_gp_barra_042/crypto_alpha_ppo_8_0_20260625_dollar_volume_rank |
| crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | 量价关系与资金流 | crypto_alpha_dollar_volume_ma/crypto_alpha_gp_barra_000/crypto_alpha_gp_barra_001/crypto_alpha_gp_barra_002/crypto_alpha_gp_barra_003/crypto_alpha_gp_barra_042/crypto_alpha_gp_barra_money_flow_btc_strength_dispersion/crypto_alpha_gp_quote_volume_decay_001/crypto_alpha_logvolume_intraday_std |

### E023

导航代表：crypto_alpha_elastic_potential_gap。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_elastic_potential_gap | 日内时段与跨时区效应 |  |

### E024

导航代表：crypto_alpha_gp_barra_004。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_gp_barra_004 | 日内时段与跨时区效应 | crypto_alpha_gp_barra_004_1/crypto_alpha_gp_barra_004_3 |
| crypto_alpha_gp_barra_004 | 日内时段与跨时区效应 | crypto_alpha_gp_barra_004/crypto_alpha_gp_barra_004_1 |
| crypto_alpha_gp_barra_004 | 日内时段与跨时区效应 | crypto_alpha_gp_barra_004/crypto_alpha_gp_barra_004_3 |

### E025

导航代表：crypto_alpha_gp_barra_004_2。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_gp_barra_004_2 | 日内时段与跨时区效应 |  |

### E026

导航代表：crypto_alpha_gp_barra_005。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_gp_barra_005 | 市场相对、BTC 残差与风格中性 |  |

### E027

导航代表：crypto_alpha_gp_barra_006。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_gp_barra_006 | 日内时段与跨时区效应 |  |

### E028

导航代表：crypto_alpha_gp_barra_094。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_gp_barra_094 | 波动率、尾部风险与高阶矩 |  |

### E029

导航代表：crypto_alpha_gp_barra_099。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_gp_barra_099 | 价格路径与形态结构 |  |

### E030

导航代表：crypto_alpha_gp_trade_count_mark_spread_corr_001。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_gp_trade_count_mark_spread_corr_001 | 日内时段与跨时区效应 |  |

### E031

导航代表：crypto_alpha_group_path_length_low_mem。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_group_path_length_low_mem | 量价关系与资金流 |  |

### E032

导航代表：crypto_alpha_high_amplitude_decisive_momentum。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_high_amplitude_decisive_momentum | 价格路径与形态结构 |  |

### E033

导航代表：crypto_alpha_high_amplitude_excess_momentum_2。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_high_amplitude_excess_momentum_2 | 量价关系与资金流 |  |

### E034

导航代表：crypto_alpha_klinger_volume_oscillator。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_klinger_volume_oscillator | 量价关系与资金流 |  |

### E035

导航代表：crypto_alpha_daily_rebalance_fit。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_daily_rebalance_fit | 量价关系与资金流 |  |

### E036

导航代表：crypto_alpha_main_force_volatility_USA_session。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_main_force_volatility_USA_session | 日内时段与跨时区效应 | crypto_alpha_volume_up_continuous_down_volatility_USA |
| crypto_alpha_volume_up_continuous_down_volatility_USA | 日内时段与跨时区效应 | crypto_alpha_main_force_volatility |

### E037

导航代表：crypto_alpha_max_single_day_return。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_max_single_day_return | 均值回归与反转 |  |

### E038

导航代表：crypto_alpha_mfi_dollar_volume。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_mfi_dollar_volume | 量价关系与资金流 | crypto_alpha_volatility_efficiency_mfi_dollar_volume |
| crypto_alpha_volatility_efficiency_mfi_dollar_volume | 量价关系与资金流 | crypto_alpha_mfi_dollar_volume |

### E039

导航代表：crypto_alpha_new_intraday_volume_adjusted。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_new_intraday_volume_adjusted | 日内时段与跨时区效应 |  |

### E040

导航代表：crypto_alpha_new_momentum_day_night_volume。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_new_momentum_day_night_volume | 日内时段与跨时区效应 |  |

### E041

导航代表：crypto_alpha_old_overnight_momentum。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_old_overnight_momentum | 日内时段与跨时区效应 |  |

### E042

导航代表：crypto_alpha_ppo_3_123_20260709150104。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_ppo_3_123_20260709150104 | 趋势与动量 |  |

### E043

导航代表：crypto_alpha_price_convergence_factor。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_price_convergence_factor | 价格路径与形态结构 |  |

### E044

导航代表：crypto_alpha_price_jump_amount_corr。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_price_jump_amount_corr | 量价关系与资金流 |  |

### E045

导航代表：crypto_alpha_price_peak_minutes_120d。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_price_peak_minutes_120d | 价格路径与形态结构 |  |

### E046

导航代表：crypto_alpha_price_retail_correlation_divergence。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_price_retail_correlation_divergence | 订单流与市场微观结构 |  |

### E047

导航代表：crypto_alpha_price_ridge_interval_skew。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_price_ridge_interval_skew | 波动率、尾部风险与高阶矩 |  |

### E048

导航代表：crypto_alpha_price_ridge_minute_return。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_price_ridge_minute_return | 均值回归与反转 |  |

### E049

导航代表：crypto_alpha_price_ridge_minutes_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_price_ridge_minutes_USA | 日内时段与跨时区效应 | crypto_alpha_retail_fomo_ratio |
| crypto_alpha_retail_fomo_ratio | 订单流与市场微观结构 | crypto_alpha_price_ridge_minutes |

### E050

导航代表：crypto_alpha_price_valley_vmacd_volume。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_price_valley_relative_vwap | 量价关系与资金流 | crypto_alpha_price_valley_vmacd_volume |
| crypto_alpha_price_valley_vmacd_volume | 量价关系与资金流 | crypto_alpha_price_valley_relative_vwap |

### E051

导航代表：crypto_alpha_price_valley_vwap_percentile。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_price_valley_vwap_percentile | 量价关系与资金流 |  |

### E052

导航代表：crypto_alpha_price_volume_convergence_factor。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_price_volume_convergence_factor | 量价关系与资金流 |  |

### E053

导航代表：crypto_alpha_realized_kurtosis。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_realized_kurtosis | 波动率、尾部风险与高阶矩 |  |

### E054

导航代表：crypto_alpha_realized_skewness。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_realized_skewness | 波动率、尾部风险与高阶矩 | crypto_alpha_return_skew_reversal |
| crypto_alpha_return_skew_reversal | 波动率、尾部风险与高阶矩 | crypto_alpha_realized_skewness |

### E055

导航代表：crypto_alpha_reconstructed_cumulative_delta_order_flow。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_reconstructed_cumulative_delta_order_flow | 订单流与市场微观结构 |  |

### E056

导航代表：crypto_alpha_resiliency_hp_fft。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_resiliency_hp_fft | 流动性、成交活跃度与容量 |  |

### E057

导航代表：crypto_alpha_retail_activity_divergence。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_retail_activity_divergence | 订单流与市场微观结构 |  |

### E058

导航代表：crypto_alpha_retail_friction_illiquidity。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_retail_friction_illiquidity | 订单流与市场微观结构 |  |

### E059

导航代表：crypto_alpha_reversal_maxret_mix。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_reversal_maxret_mix | 均值回归与反转 |  |

### E060

导航代表：crypto_alpha_rpv_stability_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_rpv_stability_USA | 日内时段与跨时区效应 |  |

### E061

导航代表：crypto_alpha_rv_20d_resid_style_wma_rank。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_rv_20d_resid_style_wma_rank | 量价关系与资金流 |  |

### E062

导航代表：crypto_alpha_sig_up_p_v_intraday_std_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_sig_up_p_v_intraday_std_USA | 日内时段与跨时区效应 |  |

### E063

导航代表：crypto_alpha_sig_up_p_v_intraday_std_stability_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_sig_up_p_v_intraday_std_stability_USA | 日内时段与跨时区效应 |  |

### E064

导航代表：crypto_alpha_sig_up_p_v_ratio_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_sig_up_p_v_ratio_USA | 量价关系与资金流 |  |

### E065

导航代表：crypto_alpha_sig_up_p_v_ratio_stability_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_sig_up_p_v_ratio_stability_USA | 量价关系与资金流 |  |

### E066

导航代表：ms_siphon_effect。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| ms_siphon_effect | 订单流与市场微观结构 |  |

### E067

导航代表：crypto_alpha_skewed_deviation_return。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_skewed_deviation_return | 日内时段与跨时区效应 |  |

### E068

导航代表：crypto_alpha_structured_reversal_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_structured_reversal_USA | 日内时段与跨时区效应 |  |

### E069

导航代表：crypto_alpha_uid_information_uniformity。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_uid_devol | 日内时段与跨时区效应 | crypto_alpha_uid_information_uniformity |
| crypto_alpha_uid_information_uniformity | 日内时段与跨时区效应 | crypto_alpha_uid_devol |

### E070

导航代表：crypto_alpha_us_asia_mom_reversal_spread_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_us_asia_mom_reversal_spread_USA | 日内时段与跨时区效应 |  |

### E071

导航代表：crypto_alpha_us_session_momentum_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_us_session_momentum_USA | 日内时段与跨时区效应 |  |

### E072

导航代表：crypto_alpha_us_trend_persistence_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_us_trend_persistence_USA | 日内时段与跨时区效应 |  |

### E073

导航代表：crypto_alpha_us_up_asia_down_consistency_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_us_up_asia_down_consistency_USA | 日内时段与跨时区效应 |  |

### E074

导航代表：crypto_alpha_us_volume_breakout_momentum_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_us_volume_breakout_momentum_USA | 日内时段与跨时区效应 |  |

### E075

导航代表：crypto_alpha_us_volume_dominance_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_us_volume_dominance_USA | 日内时段与跨时区效应 |  |

### E076

导航代表：crypto_alpha_us_volume_weighted_momentum_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_us_volume_weighted_momentum_USA | 日内时段与跨时区效应 |  |

### E077

导航代表：crypto_alpha_v_p_reversal_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_v_p_reversal_USA | 日内时段与跨时区效应 |  |

### E078

导航代表：crypto_alpha_v_p_skewness_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_v_p_skewness_USA | 日内时段与跨时区效应 |  |

### E079

导航代表：crypto_alpha_vmacd_volume。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_vmacd_volume | 量价关系与资金流 |  |

### E080

导航代表：crypto_alpha_volatility_order_flow_efficiency。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volatility_order_flow_efficiency | 订单流与市场微观结构 |  |

### E081

导航代表：crypto_alpha_volume_amplitude_divergence_corr。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_amplitude_divergence_corr | 量价关系与资金流 |  |

### E082

导航代表：crypto_alpha_volume_convergence_factor。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_convergence_factor | 量价关系与资金流 |  |

### E083

导航代表：crypto_alpha_volume_down_fall_volatility_ASIA_session。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_down_fall_volatility_ASIA_session | 日内时段与跨时区效应 |  |

### E084

导航代表：crypto_alpha_volume_energy_divergence。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_energy_divergence | 日内时段与跨时区效应 |  |

### E085

导航代表：crypto_alpha_volume_intraday_std_1_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_intraday_std_1_USA | 日内时段与跨时区效应 |  |

### E086

导航代表：crypto_alpha_volume_intraday_std_1_stability_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_intraday_std_1_stability_USA | 日内时段与跨时区效应 |  |

### E087

导航代表：crypto_alpha_volume_price_intraday_corr_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_price_intraday_corr_USA | 日内时段与跨时区效应 |  |

### E088

导航代表：crypto_alpha_volume_ratio_1_5_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_ratio_1_5_USA | 量价关系与资金流 |  |

### E089

导航代表：crypto_alpha_volume_ratio_1_5_stability_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_ratio_1_5_stability_USA | 日内时段与跨时区效应 |  |

### E090

导航代表：crypto_alpha_volume_ratio_8_4_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_ratio_8_4_USA | 日内时段与跨时区效应 |  |

### E091

导航代表：crypto_alpha_volume_ratio_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_ratio_USA | 日内时段与跨时区效应 |  |

### E092

导航代表：crypto_alpha_volume_ratio_1_5_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_ratio_1_5_USA | 日内时段与跨时区效应 |  |

### E093

导航代表：crypto_alpha_volume_ratio_stability_USA。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_ratio_stability_USA | 日内时段与跨时区效应 |  |

### E094

导航代表：crypto_alpha_volume_short_mean_anomaly。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_short_mean_anomaly | 量价关系与资金流 |  |

### E095

导航代表：crypto_alpha_volume_sorted_amplitude_ratio。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_sorted_amplitude_ratio | 日内时段与跨时区效应 |  |

### E096

导航代表：crypto_alpha_volume_sorted_momentum_ratio。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_sorted_momentum_ratio | 日内时段与跨时区效应 |  |

### E097

导航代表：crypto_alpha_volume_sorted_volatility_efficiency_ratio。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_sorted_volatility_efficiency_ratio | 量价关系与资金流 |  |

### E098

导航代表：crypto_alpha_volume_surge_volatility。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_volume_surge_volatility | 日内时段与跨时区效应 |  |

### E099

导航代表：crypto_alpha_vroc_dollar_volume。簇由高重复边的连通分量形成，簇内任意两项不一定直接超过阈值。

| factor_name | primary_cluster_label | direct_high_neighbors |
| --- | --- | --- |
| crypto_alpha_vroc_dollar_volume | 量价关系与资金流 |  |

## 5. 高重复因子对

| left | right | train_rank_similarity | train_return_similarity | train_holding_overlap | trigger_metrics |
| --- | --- | --- | --- | --- | --- |
| crypto_alpha_001 | crypto_alpha_003 | 0.8318850858542989 |  | 0.5274348422496571 | rank |
| crypto_alpha_cgo_pos_neg_tail_split_groupfix | crypto_alpha_cgo_volume_ratio_turnover_conservative | 0.878141567503806 |  | 0.5902777777777778 | rank |
| crypto_alpha_csk_xyy_down | crypto_alpha_csk_xyy_up_down | -0.9816648824341231 |  | 0.0 | rank |
| crypto_alpha_cum_return_ratio | crypto_alpha_ppo_3_123_20260709150104_resid_return_btc | -0.8050843544560431 |  | 0.0 | rank |
| crypto_alpha_directional_volatility_efficiency | crypto_alpha_volatility_efficiency | 0.9994122832794592 |  | 0.9725352112676057 | rank/holding |
| crypto_alpha_dollar_volume_ma | crypto_alpha_gp_barra_000 | 0.9210343112542514 |  | 0.6070921985815603 | rank |
| crypto_alpha_dollar_volume_ma | crypto_alpha_gp_barra_001 | 0.9324038486267875 |  | 0.6127659574468085 | rank |
| crypto_alpha_dollar_volume_ma | crypto_alpha_gp_barra_002 | -0.916718077057636 |  | 0.0 | rank |
| crypto_alpha_dollar_volume_ma | crypto_alpha_gp_barra_003 | 0.9513786690040852 |  | 0.6382978723404256 | rank |
| crypto_alpha_dollar_volume_ma | crypto_alpha_gp_barra_042 | 0.9638810792360468 |  | 0.6659574468085107 | rank |
| crypto_alpha_dollar_volume_ma | crypto_alpha_gp_barra_money_flow_btc_strength_dispersion | 0.919045343964157 |  | 0.6141843971631206 | rank |
| crypto_alpha_dollar_volume_ma | crypto_alpha_gp_quote_volume_decay_001 | 0.811947312413786 |  | 0.4524822695035461 | rank |
| crypto_alpha_dollar_volume_ma | crypto_alpha_logvolume_intraday_std | 0.8525176300077885 |  | 0.004964539007092199 | rank |
| crypto_alpha_dollar_volume_ma | crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | 0.9800617631907892 |  | 0.6560283687943262 | rank |
| crypto_alpha_gp_barra_000 | crypto_alpha_gp_barra_001 | 0.952978507860309 |  | 0.6890581717451524 | rank |
| crypto_alpha_gp_barra_000 | crypto_alpha_gp_barra_002 | -0.9101816972535288 |  | 0.0 | rank |
| crypto_alpha_gp_barra_000 | crypto_alpha_gp_barra_003 | 0.9178824944468961 |  | 0.6627423822714681 | rank |
| crypto_alpha_gp_barra_000 | crypto_alpha_gp_barra_042 | 0.9336298784110313 |  | 0.6731301939058172 | rank |
| crypto_alpha_gp_barra_000 | crypto_alpha_gp_barra_money_flow_btc_strength_dispersion | 0.9144593053777258 |  | 0.6980609418282548 | rank |
| crypto_alpha_gp_barra_000 | crypto_alpha_gp_quote_volume_decay_001 | 0.811930732123621 |  | 0.45637119113573404 | rank |
| crypto_alpha_gp_barra_000 | crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | 0.9301807801844093 |  | 0.6876731301939059 | rank |
| crypto_alpha_gp_barra_001 | crypto_alpha_gp_barra_002 | -0.9103410254420423 |  | 0.0 | rank |
| crypto_alpha_gp_barra_001 | crypto_alpha_gp_barra_003 | 0.9224372048704386 |  | 0.7001385041551247 | rank/holding |
| crypto_alpha_gp_barra_001 | crypto_alpha_gp_barra_042 | 0.9349249221754258 |  | 0.668724279835391 | rank |
| crypto_alpha_gp_barra_001 | crypto_alpha_gp_barra_money_flow_btc_strength_dispersion | 0.9143503888767074 |  | 0.6995884773662552 | rank |
| crypto_alpha_gp_barra_001 | crypto_alpha_logvolume_intraday_std | 0.8406603080620848 |  | 0.014788732394366197 | rank |
| crypto_alpha_gp_barra_001 | crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | 0.9359466221851542 |  | 0.6879286694101509 | rank |
| crypto_alpha_gp_barra_002 | crypto_alpha_gp_barra_003 | -0.9673198400740324 |  | 0.0 | rank |
| crypto_alpha_gp_barra_002 | crypto_alpha_gp_barra_042 | -0.9574860337980591 |  | 0.0 | rank |
| crypto_alpha_gp_barra_002 | crypto_alpha_gp_barra_money_flow_btc_strength_dispersion | -0.987708073081098 |  | 0.0 | rank |
| crypto_alpha_gp_barra_002 | crypto_alpha_gp_quote_volume_decay_001 | -0.8451395893484553 |  | 0.0 | rank |
| crypto_alpha_gp_barra_002 | crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | -0.9389109215461937 |  | 0.0 | rank |
| crypto_alpha_gp_barra_003 | crypto_alpha_gp_barra_042 | 0.9804132268607522 |  | 0.7707756232686981 | rank/holding |
| crypto_alpha_gp_barra_003 | crypto_alpha_gp_barra_money_flow_btc_strength_dispersion | 0.9670449720670683 |  | 0.7361495844875346 | rank/holding |
| crypto_alpha_gp_barra_003 | crypto_alpha_gp_quote_volume_decay_001 | 0.8543655596861501 |  | 0.453601108033241 | rank |
| crypto_alpha_gp_barra_003 | crypto_alpha_logvolume_intraday_std | 0.8248774577297147 |  | 0.027464788732394368 | rank |
| crypto_alpha_gp_barra_003 | crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | 0.9688084554441371 |  | 0.8116343490304709 | rank/holding |
| crypto_alpha_gp_barra_004 | crypto_alpha_gp_barra_004_1 | 0.9559988398698076 |  | 0.7407407407407407 | rank/holding |
| crypto_alpha_gp_barra_004 | crypto_alpha_gp_barra_004_3 | 1.0 |  | 1.0 | rank/holding |
| crypto_alpha_gp_barra_004_1 | crypto_alpha_gp_barra_004_3 | 0.9559988398698076 |  | 0.7407407407407407 | rank/holding |
| crypto_alpha_gp_barra_042 | crypto_alpha_gp_barra_money_flow_btc_strength_dispersion | 0.9596116292725559 |  | 0.7181069958847737 | rank/holding |
| crypto_alpha_gp_barra_042 | crypto_alpha_gp_quote_volume_decay_001 | 0.8591051720653022 |  | 0.4540466392318244 | rank |
| crypto_alpha_gp_barra_042 | crypto_alpha_logvolume_intraday_std | 0.8376994695237632 |  | 0.014788732394366197 | rank |
| crypto_alpha_gp_barra_042 | crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | 0.9834774920232191 |  | 0.7654320987654321 | rank/holding |
| crypto_alpha_gp_barra_money_flow_btc_strength_dispersion | crypto_alpha_gp_quote_volume_decay_001 | 0.8475841155705292 |  | 0.4547325102880658 | rank |
| crypto_alpha_gp_barra_money_flow_btc_strength_dispersion | crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | 0.942493963845271 |  | 0.7037037037037037 | rank/holding |
| crypto_alpha_gp_quote_volume_decay_001 | crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | 0.8325441324045223 |  | 0.4540466392318244 | rank |
| crypto_alpha_logvolume_intraday_std | crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | 0.8545624567799456 |  | 0.009154929577464789 | rank |
| crypto_alpha_main_force_volatility | crypto_alpha_volume_up_continuous_down_volatility_USA | 0.8956004140786749 |  | 0.422425952045134 | rank |
| crypto_alpha_mfi_dollar_volume | crypto_alpha_volatility_efficiency_mfi_dollar_volume | 0.9915006349943328 |  | 0.7011494252873564 | rank/holding |
| crypto_alpha_price_ridge_minutes | crypto_alpha_retail_fomo_ratio | -0.8388777333553772 |  | 0.0 | rank |
| crypto_alpha_price_valley_relative_vwap | crypto_alpha_price_valley_vmacd_volume | 0.9925839020185954 |  | 0.6148648648648649 | rank |
| crypto_alpha_realized_skewness | crypto_alpha_return_skew_reversal | 0.8003277180628998 |  | 0.18591549295774648 | rank |
| crypto_alpha_uid_devol | crypto_alpha_uid_information_uniformity | 0.9299827521559805 |  | 0.5754583921015515 | rank |

## 6. 跨簇低相似搜索地图

| left | right | train_rank_similarity | train_return_similarity | train_holding_overlap |
| --- | --- | --- | --- | --- |

## 7. 同簇但实证低相似

| left | right | train_rank_similarity | train_return_similarity | train_holding_overlap |
| --- | --- | --- | --- | --- |

## 8. Ensemble 成分与重复风险

| factor_name | primary_cluster_label | components | tags | cache_status |
| --- | --- | --- | --- | --- |
| crypto_alpha_ensemble_trajectory_rv_closeposition_csk_mark_rv_20260703 | 多机制复合信号 | CSK Up 20d + Return Abs/Mark-Index Spread / Relative Strength BTC/Mean Close Position 5d/Residualized RV Ratio minus RV 20d/Sign Rolling Return 20d/Trajectory Illiquidity & EMA RV | ensemble/manual/minutes/price/volume | not_cached |
| crypto_alpha_ppo_10_0_20260608_ensemble | 多机制复合信号 | component_1/component_2 | ensemble/manual/minutes/ppo/price/volume | not_cached |
| crypto_alpha_ppo_10_0_20260613160816_ensemble | 多机制复合信号 | component_1/component_2 | ensemble/manual/minutes/ppo/price | not_cached |
| crypto_alpha_ppo_12_0_20260626130557_ensemble | 多机制复合信号 | component_1/component_2 | ensemble/manual/minutes/ppo/price/volume | not_cached |
| crypto_alpha_ppo_12_0_20260628160526_ensemble | 多机制复合信号 | cum_return/logvolume_intraday_std_20d/price_peak_minutes_30d/funding_cumul_5d/dollar_volume_convergence_20d/csk_up_20d/rv_ratio_5_20/right_tail_cvar_logvolume_intraday_std_20d | ensemble/manual/minutes/ppo/price/volume | not_cached |
| crypto_alpha_ppo_12_17_20260628221245_ensemble | 多机制复合信号 | cum_return/logvolume_intraday_std_20d/price_peak_minutes_30d/max_ret_60d/trajectory_illiquidity_20d | ensemble/manual/minutes/ppo/price/volume | not_cached |
| crypto_alpha_ppo_12_1_20260628163318_ensemble | 多机制复合信号 | cum_return/logvolume_intraday_std_20d/price_peak_minutes_30d/funding_cumul_5d/sub_neg5_volume_intensity/directional_ratio_20d/realized_skew_liquidity_gated_20d/dollar_volume_convergence_20d/volume_intensity/log_oi_z20/oi_z20_resid_style/trajectory_illiquidity_20d | ensemble/manual/minutes/ppo/price/volume | not_cached |
| crypto_alpha_ppo_12_2_20260628170032_ensemble | 多机制复合信号 | cum_return/logvolume_intraday_std_20d/price_peak_minutes_30d/oi_z20_resid_style/funding_cumul_5d/csk_up_20d/right_tail_cvar_logvolume_intraday_std_20d/funding_oi_joint_5d/money_flow/dollar_volume_convergence_20d/dollar_volume_rank | ensemble/manual/minutes/ppo/price/volume | not_cached |
| crypto_alpha_ppo_12_3_20260628173014_ensemble | 多机制复合信号 | cum_return/logvolume_intraday_std_20d/price_peak_minutes_30d/directional_ratio_20d/turnover_resid_style | ensemble/manual/minutes/ppo/price/volume | not_cached |
| crypto_alpha_ppo_12_42_20260628103854_ensemble | 多机制复合信号 | cum_return/logvolume_intraday_std_20d/price_peak_minutes_30d/funding_cumul_5d/volume_intensity/csk_up_20d/funding_oi_joint_5d/trajectory_illiquidity_20d/neg_oi_z20/turnover_resid_style/right_tail_cvar_logvolume_intraday_std_20d/oi_z20_resid_style | ensemble/manual/minutes/ppo/price/volume | not_cached |
| crypto_alpha_ppo_12_4_20260628201213_ensemble | 多机制复合信号 | cum_return/logvolume_intraday_std_20d/price_peak_minutes_30d/rv_20d_resid_style/roll_skew_20d/dollar_volume_rank/money_flow/turnover_resid_style/funding_oi_joint_5d/directional_ratio_20d/rv_ratio_5_20/funding_cumul_5d | ensemble/manual/minutes/ppo/price/volume | not_cached |
| crypto_alpha_ppo_6_123_20260706165517 | 日内时段与跨时区效应 | component_1/component_2/component_3 | ensemble/manual/minutes/ppo/price/volume | not_cached |
| crypto_alpha_rv_cum_var_liquidity_ensemble | 多机制复合信号 | component_1/component_2 | ensemble/manual/minutes/price/volume | not_cached |
| crypto_alpha_rv_resid_btc_cum_liquidity_ensemble | 多机制复合信号 | component_1/component_2 | ensemble/manual/minutes/price/volume | not_cached |

## 9. 缓存、证据不足与漂移警告

### 未缓存或缓存无效

| factor_name | cache_status | cache_reason | evidence_quality |
| --- | --- | --- | --- |
| crypto_alpha_ensemble_trajectory_rv_closeposition_csk_mark_rv_20260703 | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_ppo_10_0_20260608_ensemble | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_ppo_10_0_20260613160816_ensemble | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_ppo_12_0_20260626130557_ensemble | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_ppo_12_0_20260628160526_ensemble | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_ppo_12_17_20260628221245_ensemble | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_ppo_12_1_20260628163318_ensemble | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_ppo_12_2_20260628170032_ensemble | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_ppo_12_3_20260628173014_ensemble | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_ppo_12_42_20260628103854_ensemble | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_ppo_12_4_20260628201213_ensemble | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_ppo_6_123_20260706165517 | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_rv_cum_var_liquidity_ensemble | not_cached | manifest_record_missing | semantic_only |
| crypto_alpha_rv_resid_btc_cum_liquidity_ensemble | not_cached | manifest_record_missing | semantic_only |

### 覆盖率不足或漂移

| factor_name | train_coverage | validation_coverage | test_coverage | validation_minus_train_coverage |
| --- | --- | --- | --- | --- |
| crypto_alpha_acceleration_short_reversal | 0.24177015082671333 | 0.24169249832076864 | 0.23785316668417184 | -7.765250594468642e-05 |
| crypto_alpha_bias_neutralized_zscore | 0.6339394903508685 | 0.5337327331211175 | 0.4399327801701502 | -0.10020675722975103 |
| crypto_alpha_directional_momentum | 0.3259527336720389 | 0.31894548443216947 | 0.33211847495011027 | -0.007007249239869451 |
| crypto_alpha_directional_volatility_efficiency | 0.7793177502261903 | 0.5681726208107871 | 0.6178132549101986 | -0.2111451294154032 |
| crypto_alpha_high_amplitude_excess_momentum_2 | 0.5037830958748845 | 0.8678225897293621 | 0.9637012918811049 | 0.36403949385447765 |
| crypto_alpha_price_retail_correlation_divergence | 0.19945515465455016 | 0.16898699586554466 | 0.06567587438294296 | -0.030468158789005495 |
| crypto_alpha_retail_activity_divergence | 0.630191192992573 | 0.4841025812184753 | 0.30351853796870076 | -0.14608861177409765 |
| crypto_alpha_retail_fomo_ratio | 0.6234104534743833 | 0.4770018315094725 | 0.3033714945909043 | -0.14640862196491078 |
| crypto_alpha_retail_friction_illiquidity | 0.630191192992573 | 0.4841025812184753 | 0.30351853796870076 | -0.14608861177409765 |
| crypto_alpha_return_skew_reversal | 0.7793177502261903 | 0.5681726208107871 | 0.6178132549101986 | -0.2111451294154032 |
| crypto_alpha_rv_20d_resid_style_wma_rank | 0.263862237644041 | 0.8404834517486952 | 0.9837412036550782 | 0.5766212141046543 |
| crypto_alpha_volatility_efficiency | 0.7793177502261903 | 0.5681726208107871 | 0.6178132549101986 | -0.2111451294154032 |
| crypto_alpha_volatility_order_flow_efficiency | 0.19945515465455016 | 0.16900785588231612 | 0.06567587438294296 | -0.030447298772234038 |

### 样本外相关性上升

| left | right | validation_rank_similarity | test_rank_similarity |
| --- | --- | --- | --- |

### 证据不足的因子对

| left | right | train_rank_valid_dates | train_return_valid_dates |
| --- | --- | --- | --- |
| crypto_alpha_price_retail_correlation_divergence | crypto_alpha_rv_20d_resid_style_wma_rank | 49 | 1 |
| crypto_alpha_rv_20d_resid_style_wma_rank | crypto_alpha_volatility_order_flow_efficiency | 49 | 1 |

### 命名或成分依赖警告

| factor_name | source_name | duplicate_factor_name | dependency_warning |
| --- | --- | --- | --- |
| crypto_alpha_gp_barra_004 | crypto_alpha_gp_barra_004 | True |  |
| crypto_alpha_gp_barra_004 | crypto_alpha_gp_barra_004_1 | True |  |
| crypto_alpha_gp_barra_004 | crypto_alpha_gp_barra_004_3 | True |  |
| crypto_alpha_volume_ratio_1_5_USA | crypto_alpha_volume_ratio_1_5_USA | True |  |
| crypto_alpha_volume_ratio_1_5_USA | crypto_alpha_volume_ratio_USA_ASIA | True |  |
