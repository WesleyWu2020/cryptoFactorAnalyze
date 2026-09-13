# Unsubmit 因子库筛查报告（crypto_factor_library_rules0709 标准）

## 评估口径与方法

- 标准来源: `tmp/crypto_factor_library_rules0709.html`（规则名与 L2 阈值 = `common/core/contracts/research_acceptance.py` 的 `research_gp_daily_acceptance_v1`）
- 口径: profile=`perp_1d`, cost=0.0015, universe=tradable_mask, H5=`bybit_linear_1m_unified_gp_research.h5`
- IS 窗口: 2022-01-01 ~ 2023-12-31；OOS 窗口: 2024-01-01 ~ 2025-12-31；逐年: 2022~2025；tail-drop 与 seg 指标在 OOS 窗口计算
- 计算路径: 离线复刻官方 replay（`simulate_py_file` + `evaluate_research_gp_daily_acceptance_bundle` + approved overlap cache），不写 platform DB；prod_corr 不属于规则集，未计算
- teacher H5 为本地代理实现: direction_consistency=全样本 rankic 与 ls_netret 同号；full_sample=全样本 ls_netret>0；annual=2022~2025 每年 ls_netret>0
- 判定: **L2_PASS** = replay 全规则 + teacher_h5 三项全过（triple_pass，可 submit）；**L1_PASS** = [H] 全过 + [H-L1] 达 L1 线（coverage≥0.65、全样本盈利）+ 非双弱 + EVR 信号下限（IS≥0.65 或 OOS≥0.80），软失败打标签；**FAIL** = 其余

## 总览

- 因子总数: **229**
- **L2_PASS（符合二级库/submit 标准）: 1**
- **L1_PASS（符合一级库标准，待升级）: 4**
- **FAIL（不符合入库标准）: 224**

### FAIL 原因分布

| 失败类别 | 数量 |
|---|---|
| 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) | 80 |
| [H-L1] min_yearly_coverage | 41 |
| preflight/replay crash (worker killed/OOM) | 27 |
| [H-L1] 全样本 ls_netret < | 27 |
| [S] IS Sharpe | 13 |
| [H] overlap 未过 | 11 |
| [S] IS netret | 10 |
| preflight/replay crash | 7 |
| preflight/replay timeout | 3 |
| [S] OOS netret | 3 |
| [S] OOS seg_min | 2 |

## 通过 - 二级库（L2_PASS, n=1）

| 因子 | IS Sharpe | OOS netret | OOS Sharpe | min覆盖 | maxValCorr | 标签 |
|---|---|---|---|---|---|---|

## 通过 - 一级库（L1_PASS, n=4）

| 因子 | IS Sharpe | OOS netret | OOS Sharpe | min覆盖 | maxValCorr | L2 未过规则 | 标签 |
|---|---|---|---|---|---|---|---|
| crypto_alpha_gp_barra_004 | 1.24 | 0.34 | 2.14 | 0.99 | 0.49 | min_yearly_sharpe | regime_dependent, sign_flip_candidate |
| crypto_alpha_gp_barra_004_3 | 1.24 | 0.34 | 2.14 | 0.99 | 0.49 | min_yearly_sharpe | regime_dependent, sign_flip_candidate |
| crypto_alpha_gp_barra_004_1 | 1.00 | 0.29 | 1.55 | 0.99 | 0.56 | min_yearly_netret, min_yearly_sharpe | regime_dependent, sign_flip_candidate |
| crypto_alpha_vclock_group_slope | 0.82 | 0.07 | 0.68 | 0.93 | 0.14 | min_oos_net_sharpe, min_oos_netret | h5_ov_only |

## 未通过（FAIL, n=224）

| 因子 | 状态 | 失败原因 |
|---|---|---|
| crypto_alpha_001 | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_002 | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_003 | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_004 | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_005 | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_acceleration_short_reversal | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_acd_accumulation_distribution | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_activity_burst_premium_20d | ok | [S] OOS seg_min=-0.16800903138404133 < -0.15 |
| crypto_alpha_ad_volume_momentum | ok | [S] IS Sharpe=0.11291618838331098 < 0.65 |
| crypto_alpha_amplitude_sorted_momentum_ratio | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_asia_reversal_absorption_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_asia_session_reversal_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_avg_trade_size | ok | [H] overlap 未过: ['min_value_residual_norm', 'min_holding_residual_norm'] |
| crypto_alpha_basis_meanrev_1d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_bias_neutralized_zscore | ok | [H-L1] min_yearly_coverage=0.2671824500941608 < 0.65 |
| crypto_alpha_body_ratio_sorted_momentum_ratio | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_breakout_funding_dv_confirm_20d | ok | [S] IS netret=-0.0950199530173812 <= -0.05 |
| crypto_alpha_breakout_funding_dv_confirm_20d_btcgate | ok | [H-L1] min_yearly_coverage=0.43484678059187476 < 0.65 |
| crypto_alpha_breakout_funding_dv_confirm_20d_btcgate_hyst | ok | [H-L1] min_yearly_coverage=0.4545862658771661 < 0.65 |
| crypto_alpha_breakout_funding_dv_confirm_20d_btcgate_hyst_asym | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_breakout_funding_dv_confirm_20d_btcgate_hyst_damp | ok | [S] IS netret=-0.0950199530173812 <= -0.05 |
| crypto_alpha_breakout_funding_dv_confirm_20d_regsplit | ok | [S] IS netret=-0.10545363640517724 <= -0.05 |
| crypto_alpha_breakout_funding_dv_confirm_20d_resid | ok | [H-L1] min_yearly_coverage=0.6316785717908592 < 0.65 |
| crypto_alpha_breakout_funding_dv_confirm_20d_sqfil | ok | [H-L1] min_yearly_coverage=0.39750482926503977 < 0.65 |
| crypto_alpha_breakout_quality_20d | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_bybit_heat_premium_20_120 | ok | [H-L1] min_yearly_coverage=0.4931506849315068 < 0.65 |
| crypto_alpha_bybit_heat_standard_tier_20_120 | ok | [H-L1] min_yearly_coverage=0.4931506849315068 < 0.65 |
| crypto_alpha_cgo_pos_neg_tail_split_groupfix | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_cgo_volume_ratio_turnover_conservative | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_chase_flow_premium_20d | ok | [S] IS Sharpe=0.5649404803366597 < 0.65 |
| crypto_alpha_close_location_extremeshort_20d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_close_location_premium_20d | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_comp_attention_burst_oi | ok | [H-L1] min_yearly_coverage=0.45295617015386713 < 0.65 |
| crypto_alpha_comp_attention_burst_premium | ok | [H-L1] min_yearly_coverage=0.4931506849315068 < 0.65 |
| crypto_alpha_comp_attention_oi_buildup | ok | [H-L1] min_yearly_coverage=0.45295617015386713 < 0.65 |
| crypto_alpha_comp_breaktail_squeeze_mid | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_comp_oi_buildup_chase_int | ok | [H-L1] min_yearly_coverage=0.45295617015386713 < 0.65 |
| crypto_alpha_cpv | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_csk_xyy_down | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_csk_xyy_up | ok | [H-L1] min_yearly_coverage=0.48347001134340034 < 0.65 |
| crypto_alpha_csk_xyy_up_down | ok | [H-L1] min_yearly_coverage=0.41435902593692747 < 0.65 |
| crypto_alpha_cum_return_ratio | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_directional_volatility_efficiency | ok | [H-L1] min_yearly_coverage=0.5183844380452809 < 0.65 |
| crypto_alpha_dollar_volume_convergence_factor | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_dollar_volume_ma | ok | [H] overlap 未过: ['min_value_residual_norm', 'min_holding_residual_norm'] |
| crypto_alpha_elastic_potential_gap | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_ensemble_trajectory_rv_closeposition_csk_mark_rv_20260703 | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_flow_resid_mkt_intraday | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_flow_resid_noise_intraday | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_funding_carry | ok | [H] overlap 未过: ['max_value_overlap_corr', 'min_value_residual_norm'] |
| crypto_alpha_funding_deviation_revert_20_60 | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_gp_barra_000 | ok | [H] overlap 未过: ['min_value_residual_norm', 'min_holding_residual_norm'] |
| crypto_alpha_gp_barra_001 | ok | [H] overlap 未过: ['max_holding_overlap_corr', 'min_holding_residual_norm'] |
| crypto_alpha_gp_barra_002 | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_gp_barra_003 | timeout | preflight/replay timeout |
| crypto_alpha_gp_barra_004_2 | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_gp_barra_005 | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_gp_barra_006 | timeout | preflight/replay timeout |
| crypto_alpha_gp_barra_042 | ok | [H] overlap 未过: ['max_value_overlap_corr', 'min_value_residual_norm', 'min_holding_residual_norm'] |
| crypto_alpha_gp_barra_094 | timeout | preflight/replay timeout |
| crypto_alpha_gp_barra_099 | ok | [S] OOS seg_min=-0.19505462220195435 < -0.15 |
| crypto_alpha_gp_barra_money_flow_btc_strength_dispersion | ok | [H] overlap 未过: ['min_holding_residual_norm'] |
| crypto_alpha_gp_corr_index_close_close_amihud_high08_liqneut_w90_lag2_decay20 | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_gp_corr_mktturnover_turnover_amihud_high08_liqneut_w90_decay20 | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_gp_cov_index_close_close_amihud_high08_liqneut_w90_lag2_decay20 | ok | [S] IS Sharpe=0.352695800175262 < 0.65 |
| crypto_alpha_gp_cov_index_close_close_amihud_high08_liqneut_w90_lag2_decay5 | ok | [S] IS Sharpe=0.12283470279240867 < 0.65 |
| crypto_alpha_gp_cov_index_close_close_dollar_volume_rank_004 | crash | preflight/replay crash |
| crypto_alpha_gp_cov_mark_close_typical_price_dollar_volume_rank_005 | crash | preflight/replay crash |
| crypto_alpha_gp_cov_typical_price_dollar_volume_rank_007 | crash | preflight/replay crash |
| crypto_alpha_gp_quote_volume_decay_001 | ok | [H] overlap 未过: ['min_holding_residual_norm'] |
| crypto_alpha_gp_quote_volume_decay_002 | ok | [H] overlap 未过: ['min_holding_residual_norm'] |
| crypto_alpha_gp_ratio_mean_index_close_dollar_volume_rank_mark_index_spread_003 | crash | preflight/replay crash |
| crypto_alpha_gp_ratio_mean_typical_price_amihud_illiq_mark_index_spread_001 | crash | preflight/replay crash |
| crypto_alpha_gp_ratio_std_index_close_down_volume_pressure_funding_oi_joint_5d_002 | crash | preflight/replay crash |
| crypto_alpha_gp_residual_std_typical_price_close_open_006 | crash | preflight/replay crash |
| crypto_alpha_gp_residual_std_typical_price_money_flow_dvranklow03_w180_lag1 | ok | [S] IS Sharpe=0.15926596930280906 < 0.65 |
| crypto_alpha_gp_trade_count_mark_spread_corr_001 | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_group_path_length | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_high_amplitude_decisive_momentum | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_high_amplitude_excess_momentum_2 | ok | [H-L1] min_yearly_coverage=0.2871006843380829 < 0.65 |
| crypto_alpha_high_volume_distribution | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_illiq_premium_20d | ok | [H] overlap 未过: ['min_value_residual_norm', 'min_holding_residual_norm'] |
| crypto_alpha_informed_minute_ret_5d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_intraday_path_noise_10d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_klinger_volume_oscillator | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_large_move_liqbucket_20d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_large_move_liqbucket_rev_20d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_logvolume_intraday_std | ok | [H-L1] min_yearly_coverage=0.623969077508179 < 0.65 |
| crypto_alpha_low_turnover_pro | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_low_volume_absorption | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_main_force_volatility | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_max_single_day_return | ok | [H-L1] min_yearly_coverage=0.6197042749253718 < 0.65 |
| crypto_alpha_mfi_dollar_volume | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_new_intraday_volume_adjusted | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_new_momentum_day_night_volume | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_oi_buildup_premium_20_120 | ok | [H-L1] min_yearly_coverage=0.45295617015386713 < 0.65 |
| crypto_alpha_oi_buildup_premium_20_120_longfil | ok | [H-L1] min_yearly_coverage=0.4117339779801776 < 0.65 |
| crypto_alpha_oi_chase_intensity_premium_5d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_oi_chase_reversal_5d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_old_overnight_momentum | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_path_efficiency_volume_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_path_efficiency_volume_down_event_ewm20_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_path_efficiency_volume_early_volume_exhaustion_ewm20_USA | ok | [S] IS netret=-0.06940662290087984 <= -0.05 |
| crypto_alpha_path_efficiency_volume_historical_move_efficiency_surprise_ewm20_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_path_efficiency_volume_historical_move_surprise_ewm20_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_path_efficiency_volume_history_reliability_60d_ewm20_USA | ok | [S] IS Sharpe=0.03481221863071642 < 0.65 |
| crypto_alpha_path_efficiency_volume_idiosyncratic_coherence_ewm20_USA | ok | [S] IS netret=-0.13935083270937942 <= -0.05 |
| crypto_alpha_path_efficiency_volume_idiosyncratic_ewm20_USA | ok | [S] IS netret=-0.11156982565222484 <= -0.05 |
| crypto_alpha_path_efficiency_volume_late_volume_confirmation_ewm20_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_path_efficiency_volume_magnitude_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_path_efficiency_volume_magnitude_ewm10_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_path_efficiency_volume_magnitude_ewm20_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_path_efficiency_volume_magnitude_reversal_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_path_efficiency_volume_realized_variance_adjusted_ewm20_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_path_efficiency_volume_robust_mad_normalized_ewm20_USA | ok | [S] IS netret=-0.1287025958222 <= -0.05 |
| crypto_alpha_path_efficiency_volume_up_event_ewm20_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_ppo_10_0_20260608_ensemble | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_ppo_10_0_20260613160816_ensemble | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_ppo_12_0_20260626130557_ensemble | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_ppo_12_0_20260628160526_ensemble | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_ppo_12_17_20260628221245_ensemble | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_ppo_12_1_20260628163318_ensemble | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_ppo_12_2_20260628170032_ensemble | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_ppo_12_3_20260628173014_ensemble | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_ppo_12_42_20260628103854_ensemble | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_ppo_12_4_20260628201213_ensemble | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_ppo_3_123_20260709150104 | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_ppo_3_123_20260709150104_resid_return_btc | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_ppo_6_123_20260706165517 | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_ppo_8_0_20260625_dollar_volume_rank | ok | [H] overlap 未过: ['max_value_overlap_corr', 'min_value_residual_norm', 'min_holding_residual_norm'] |
| crypto_alpha_price_convergence_factor | ok | [H-L1] min_yearly_coverage=0.6252338341580427 < 0.65 |
| crypto_alpha_price_jump_amount_corr | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_price_peak_minutes_120d | ok | [H-L1] min_yearly_coverage=0.395047390307174 < 0.65 |
| crypto_alpha_price_retail_correlation_divergence | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_price_ridge_interval_skew | ok | [S] OOS netret=-0.05218036755095312 <= 0 |
| crypto_alpha_price_ridge_minute_return | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_price_ridge_minutes | ok | [H-L1] min_yearly_coverage=0.6194924270399517 < 0.65 |
| crypto_alpha_price_valley_relative_vwap | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_price_valley_vmacd_volume | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_price_valley_vwap_percentile | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_price_volume_convergence_factor | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_pv_corr_premium_10d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_pv_corr_regimesplit_10d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_pv_diverge_exhaustion_10d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_range_level_premium_20d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_realized_kurtosis | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_realized_skewness | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_reconstructed_cumulative_delta_order_flow | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_resiliency_hp_fft | ok | [H-L1] min_yearly_coverage=0.44235816652292964 < 0.65 |
| crypto_alpha_retail_activity_divergence | ok | [H-L1] min_yearly_coverage=0.446082946156534 < 0.65 |
| crypto_alpha_retail_fomo_ratio | ok | [H-L1] min_yearly_coverage=0.44095952837701774 < 0.65 |
| crypto_alpha_retail_friction_illiquidity | ok | [H-L1] min_yearly_coverage=0.446082946156534 < 0.65 |
| crypto_alpha_return_skew_reversal | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_reversal_maxret_mix | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_rpv_stability_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_rv_20d_resid_style_wma_rank | ok | [H-L1] min_yearly_coverage=0.16810051546022475 < 0.65 |
| crypto_alpha_rv_cum_var_liquidity_ensemble | ok | [H-L1] min_yearly_coverage=0.5868433182547091 < 0.65 |
| crypto_alpha_rv_resid_btc_cum_liquidity_ensemble | ok | [H-L1] min_yearly_coverage=0.5868433182547091 < 0.65 |
| crypto_alpha_session_signed_momentum_10d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_sig_up_p_v_intraday_std_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_sig_up_p_v_intraday_std_stability_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_sig_up_p_v_ratio_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_sig_up_p_v_ratio_stability_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_siphon_effect | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_skewed_deviation_return | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_structured_reversal_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_taker_attention_premium_20_120 | ok | [H-L1] min_yearly_coverage=0.4931506849315068 < 0.65 |
| crypto_alpha_taker_attention_premium_20_120_retneut | ok | [H-L1] min_yearly_coverage=0.4893775737638495 < 0.65 |
| crypto_alpha_taker_buy_ratio_persist10 | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_taker_fomo_reversal_z20 | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_ticket_shift_premium_20_120 | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_ticket_shift_premium_20_120_rev | ok | [H-L1] min_yearly_coverage=0.4931506849315068 < 0.65 |
| crypto_alpha_trend_money_composite_5d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_trend_money_net_support_5d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_trend_money_rel_vwap_5d | ok | [S] IS Sharpe=0.20336698403893091 < 0.65 |
| crypto_alpha_uid_devol | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_uid_information_uniformity | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_us_asia_mom_reversal_spread_USA | ok | [S] OOS netret=-0.11960913954471541 <= 0 |
| crypto_alpha_us_session_momentum_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_us_trend_persistence_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_us_up_asia_down_consistency_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_us_volume_breakout_momentum_USA | ok | [S] IS Sharpe=0.4234497344113276 < 0.65 |
| crypto_alpha_us_volume_dominance_USA | ok | [S] IS Sharpe=-0.17892934099086066 < 0.65 |
| crypto_alpha_us_volume_weighted_momentum_USA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_utd_turnover_uniformity_20d | ok | [S] OOS netret=-0.03954714626268829 <= 0 |
| crypto_alpha_v_p_reversal_USA | ok | [S] IS netret=-0.07229195677979483 <= -0.05 |
| crypto_alpha_v_p_skewness_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_venue_share_shift_20_120 | ok | [H-L1] min_yearly_coverage=0.3163930296121165 < 0.65 |
| crypto_alpha_vmacd_volume | ok | [H-L1] min_yearly_coverage=0.6493150684931507 < 0.65 |
| crypto_alpha_vol_squeeze_position_20_120 | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_vol_squeeze_position_20_120_liqfil | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_vol_squeeze_premium_20_120_tailfil | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_vol_volume_elasticity_intraday | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volatility_efficiency | ok | [H-L1] min_yearly_coverage=0.5183844380452809 < 0.65 |
| crypto_alpha_volatility_efficiency_mfi_dollar_volume | ok | [H-L1] min_yearly_coverage=0.38717510726233145 < 0.65 |
| crypto_alpha_volatility_order_flow_efficiency | ok | [H-L1] min_yearly_coverage=0.12598162121877515 < 0.65 |
| crypto_alpha_volspike_breadth_regime_20d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volspike_breadth_regime_20d_s20 | ok | [H-L1] min_yearly_coverage=0.5666237767188382 < 0.65 |
| crypto_alpha_volspike_breadth_regime_20d_tail | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_amplitude_divergence_corr | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_volume_convergence_factor | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_down_fall_volatility_ASIA | ok | [S] IS netret=-0.09541460317753114 <= -0.05 |
| crypto_alpha_volume_energy_divergence | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_instability_resid_20d | crash | preflight/replay crash (worker killed/OOM) |
| crypto_alpha_volume_instability_resid_60d | ok | [S] IS Sharpe=0.11754012136120615 < 0.65 |
| crypto_alpha_volume_instability_signed_20d | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_intraday_std_1_USA | ok | [H-L1] min_yearly_coverage=0.6194924270399517 < 0.65 |
| crypto_alpha_volume_intraday_std_1_stability_USA | ok | [S] IS Sharpe=0.3657232106011439 < 0.65 |
| crypto_alpha_volume_price_intraday_corr_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_ratio_1_5_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_ratio_1_5_stability_USA | ok | [S] IS Sharpe=0.4502179607909506 < 0.65 |
| crypto_alpha_volume_ratio_8_4_USA | ok | [S] IS Sharpe=-0.30136463585497697 < 0.65 |
| crypto_alpha_volume_ratio_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_ratio_USA_ASIA | ok | [H-L1] 全样本 ls_netret <= 0 |
| crypto_alpha_volume_ratio_stability_USA | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_short_mean_anomaly | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_sorted_amplitude_ratio | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_sorted_momentum_ratio | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_sorted_volatility_efficiency_ratio | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_surge_trend_decay_h10 | ok | [H-L1] min_yearly_coverage=0.5114264640947638 < 0.65 |
| crypto_alpha_volume_surge_trend_decay_h10_fullvote | ok | [H-L1] min_yearly_coverage=0.4535914630593262 < 0.65 |
| crypto_alpha_volume_surge_volatility | ok | 双弱 (OOS Sharpe<0 且 IS Sharpe<0.65) |
| crypto_alpha_volume_up_continuous_down_volatility_USA | ok | [S] IS netret=-0.05479189213683744 <= -0.05 |
| crypto_alpha_vroc_dollar_volume | ok | [H-L1] 全样本 ls_netret <= 0 |
