# factor_analyse/factor_config.py
# 因子配置字典，包含所有支持的因子

FACTOR_CONFIG = {
    "price_momentum_60d": {
        "file_prefix": "price_momentum_60d_rebalance10d_",
        "factor_name": "Price_Momentum_60D",
        "factor_direction": 1,
        "factor_desc": "价格动量因子(60日中期趋势)",
        "rebalance_period": 10
    },
    "price_momentum_120d": {
        "file_prefix": "price_momentum_120d_rebalance20d_",
        "factor_name": "Price_Momentum_120D",
        "factor_direction": 1,
        "factor_desc": "价格动量因子(120日长期趋势)",
        "rebalance_period": 20
    },
    "ma_cross_trend_20_60d": {
        "file_prefix": "ma_cross_trend_20_60d_rebalance5d_",
        "factor_name": "MA_Cross_Trend_20_60D",
        "factor_direction": 1,
        "factor_desc": "均线多头排列强度因子(20MA/60MA)",
        "rebalance_period": 5
    },
    "Retail_Activity_Divergence_20d": {
        "file_prefix": "retail_activity_divergence_20d_rebalance10d_",
        "factor_name": "Retail_Activity_Divergence_20D_Rebalance10D",
        "factor_direction": -1,
        "factor_desc": "散户活跃度背离因子(20日)",
        "rebalance_period": 10
    },
    "Retail_Friction_Illiquidity_Factor": {
        "file_prefix": "retail_friction_illiquidity_20d_rebalance10d_",
        "factor_name": "Retail_Friction_Illiquidity_20D_Rebalance10D",
        "factor_direction": -1,
        "factor_desc": "散户摩擦-流动性因子(20日)",
        "rebalance_period": 10
    },
    "directional_momentum_20d": {
        "file_prefix": "directional_momentum_20d_",
        "factor_name": "Directional_Momentum_20D",
        "factor_direction": 1,
        "factor_desc": "方向动量因子(20日)",
        "rebalance_period": 10
    },
    "volatility_order_flow_20d": {
        "file_prefix": "volatility_order_flow_20d_",
        "factor_name": "Volatility_Order_Flow_20D",
        "factor_direction": 1,
        "factor_desc": "波动率-订单流因子(20日)",
        "rebalance_period": 10
    },
    "volatility_efficiency_20d": {
        "file_prefix": "volatility_efficiency_20d_rebalance10d_",
        "factor_name": "Volatility_Efficiency_20D_Rebalance10D",
        "factor_direction": 1,
        "factor_desc": "波动率效率因子(20日)",
        "rebalance_period": 10
    },
    "volatility_efficiency": {
        "file_prefix": "volatility_efficiency_20d_rebalance1d_",
        "factor_name": "Volatility_Efficiency_20D_Rebalance1D",
        "factor_direction": 1,
        "factor_desc": "波动率效率因子(20日)",
        "rebalance_period": 1
    },
    "volume_stability_20d": {
        "file_prefix": "volume_stability_20d_rebalance10d_",
        "factor_name": "Volume_Stability_20D_Rebalance10D",
        "factor_direction": 1,
        "factor_desc": "成交量稳定性因子(20日)",
        "rebalance_period": 10
    },
    "retail_activity_divergence_20d": {
        "file_prefix": "retail_activity_divergence_20d_rebalance10d_",
        "factor_name": "Retail_Activity_Divergence_20D_Rebalance10D",
        "factor_direction": -1,
        "factor_desc": "散户活跃度背离因子(20日)",
        "rebalance_period": 10
    },
    "ats_price_divergence_20d": {
        "file_prefix": "ats_price_divergence_20d_rebalance10d_",
        "factor_name": "ATS_Price_Divergence_20D_Rebalance10D",
        "factor_direction": 1,
        "factor_desc": "笔均成交额-价格增速背离因子(20日)",
        "rebalance_period": 10
    },
    "return_skew_reversal_20d": {
        "file_prefix": "return_skew_reversal_20d_rebalance10d_",
        "factor_name": "Return_Skew_Reversal_20D_Rebalance10D",
        "factor_direction": 1,
        "factor_desc": "收益率偏度反转因子(20日)",
        "rebalance_period": 10
    },
    "intraday_accumulation_distribution_20d": {
        "file_prefix": "intraday_accumulation_distribution_20d_rebalance10d_",
        "factor_name": "Intraday_Accumulation_Distribution_20D_Rebalance10D",
        "factor_direction": 1,
        "factor_desc": "日内买卖压强分布因子(20日)",
        "rebalance_period": 10
    },
    "price_retail_corr_divergence_20d": {
        "file_prefix": "price_retail_corr_divergence_20d_rebalance10d_",
        "factor_name": "Price_Retail_Corr_Divergence_20D_Rebalance10D",
        "factor_direction": -1,
        "factor_desc": "价量与散户相关性背离因子(20日)",
        "rebalance_period": 10
    },
    "retail_fomo_ratio_20d": {
        "file_prefix": "retail_fomo_ratio_20d_rebalance10d_",
        "factor_name": "Retail_FOMO_Ratio_20D_Rebalance10D",
        "factor_direction": -1,
        "factor_desc": "散户追涨FOMO情绪指数(20日)",
        "rebalance_period": 10
    },
    "smart_money_accum_divergence_20d": {
        "file_prefix": "smart_money_accum_divergence_20d_rebalance10d_",
        "factor_name": "Smart_Money_Accum_Divergence_20D_Rebalance10D",
        "factor_direction": 1,
        "factor_desc": "聪明钱潜伏背离因子(20日)",
        "rebalance_period": 10
    },
    "retail_friction_illiquidity_20d": {
        "file_prefix": "retail_friction_illiquidity_20d_rebalance10d_",
        "factor_name": "Retail_Friction_Illiquidity_20D_Rebalance10D",
        "factor_direction": -1,
        "factor_desc": "散户拥挤下的流动性耗散因子(20日)",
        "rebalance_period": 10
    },
    "xgboost_1d": {
        "file_prefix": "xgboost_10d_rebalance1d_",
        "factor_name": "XGBoost_Enhanced",
        "factor_direction": 1,
        "factor_desc": "XGBoost 增强因子",
        "rebalance_period": 1
    },
    "xgboost_10d": {
        "file_prefix": "xgboost_10d_rebalance10d_",
        "factor_name": "XGBoost_Enhanced",
        "factor_direction": 1,
        "factor_desc": "XGBoost 增强因子",
        "rebalance_period": 10
    },
    "xgboost_7d": {
        "file_prefix": "xgboost_14d_rebalance7d_",
        "factor_name": "XGBoost_Enhanced",
        "factor_direction": 1,
        "factor_desc": "XGBoost 增强因子",
        "rebalance_period": 7
    },
    "xgboost_5d": {
        "file_prefix": "xgboost_10d_rebalance5d_",
        "factor_name": "XGBoost_Enhanced",
        "factor_direction": 1,
        "factor_desc": "XGBoost 增强因子",
        "rebalance_period": 5
    },
    "CGO_Factor": {
        "file_prefix": "cgo_strict_log_10d_",
        "factor_name": "CGO_Factor",
        "factor_direction": 1,
        "factor_desc": "资本利得突出量因子",
        "rebalance_period": 3
    },
    "vwap_vol_mom_cap_20d": {
        "file_prefix": "vwap_vol_mom_cap_20d_rebalance10d_",
        "factor_name": "VWAP_Vol_Mom_Cap_Factor",
        "factor_direction": 1,
        "factor_desc": "VWAP-波动率-动量-市值因子",
        "rebalance_period": 10
    },  
    "directional_volatility_20d": {
        "file_prefix": "directional_volatility_20d_rebalance10d_",
        "factor_name": "Directional_Volatility_20D_Rebalance10D",
        "factor_direction": 1,
        "factor_desc": "方向波动率因子(10日)",
        "rebalance_period": 10
    },
    "example_momentum": {
        "file_prefix": "example_momentum_",
        "factor_name": "example_momentum",
        "factor_direction": 1,
        "factor_desc": "示例动量因子(N日对数动量, common框架迁移示例)",
        "rebalance_period": 1,
        "module_path": "factor_analyse/factor_mining/example_momentum.py"
    },

}

# 辅助函数
def get_factor_config(factor_id):
    """获取指定因子的配置"""
    return FACTOR_CONFIG.get(factor_id)

def list_all_factors():
    """列出所有支持的因子"""
    return list(FACTOR_CONFIG.keys())

def get_factor_description(factor_id):
    """获取指定因子的描述"""
    config = get_factor_config(factor_id)
    return config.get("factor_desc") if config else None

def get_rebalance_period(factor_id):
    """获取指定因子的调仓周期"""
    config = get_factor_config(factor_id)
    return config.get("rebalance_period") if config else None

def get_module_path(factor_id):
    """获取指定因子的 common 框架模块路径（未迁移因子返回 None）"""
    config = get_factor_config(factor_id)
    return config.get("module_path") if config else None
