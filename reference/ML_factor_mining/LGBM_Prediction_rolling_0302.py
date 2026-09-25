# factor_analyse/ML_Factor/LGBM_Prediction_Factor_Enhanced.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 可视化相关导入
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    print("✅ 可视化库可用")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("警告: 可视化库未安装，相关性分析将只显示文本")

try:
    import lightgbm as lgb
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    from sklearn.decomposition import PCA
    LGBM_AVAILABLE = True
    print("✅ LightGBM可用")
except ImportError:
    print("警告: LightGBM未安装，请安装: pip install lightgbm")
    LGBM_AVAILABLE = False

# 贝叶斯优化相关导入
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
    print("✅ Scikit-optimize可用")
except ImportError:
    print("警告: Scikit-optimize未安装，请安装: pip install scikit-optimize")
    SKOPT_AVAILABLE = False

from util_factor import (
    # 基础工具函数
    load_kline_df,
    build_available_tokens_by_date_from_kline,
    rank_to_unit_by_date, save_factor_df,
    print_factor_summary,
    # 数据预处理函数
    prepare_robust_data, prepare_xgboost_data_fast as prepare_lgbm_data_fast,
    # 特征选择函数
    adaptive_feature_selection, conservative_feature_selection,
    robust_feature_selection,
    ic_stability_feature_selection,
    # 分析函数
    analyze_selected_features_correlation, analyze_feature_importance,
    get_feature_names,
    # 时间序列分割
    create_strict_time_split, create_rolling_time_split, create_expanding_rolling_time_split,
    create_quarterly_expanding_rolling_time_split,
    create_monthly_sliding_window_time_split,
    # 模型训练函数
    # 工具函数
    print_latest_daily_groups_live,
    _artifact_paths,
    apply_label_purge_to_split,
    dedupe_predictions_keep_latest_window,
)

from scipy import stats
import os
from pathlib import Path
import joblib


def _daily_cross_section_corr(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    min_cross_section: int = 5,
) -> pd.Series:
    """Compute daily cross-sectional correlation series (one value per date)."""
    daily_corrs = {}
    for date, group in df.groupby('date'):
        if len(group) < min_cross_section:
            continue
        corr = group[pred_col].corr(group[actual_col])
        if np.isnan(corr):
            continue
        daily_corrs[pd.to_datetime(date)] = float(corr)
    if not daily_corrs:
        return pd.Series(dtype=float)
    return pd.Series(daily_corrs).sort_index()


def _subsample_non_overlapping(series: pd.Series, step: int) -> pd.Series:
    """Keep every `step`-th observation to reduce overlap (crude but robust)."""
    if series.empty or step is None or step <= 1:
        return series
    return series.iloc[::step]


def _newey_west_tstat(series: pd.Series, lag: int) -> tuple[float, float]:
    """Newey-West (Bartlett kernel) t-stat for mean of a possibly autocorrelated series.

    Returns (t_stat, se_mean). If insufficient data, returns (nan, nan).
    """
    x = series.dropna().to_numpy(dtype=float)
    n = x.shape[0]
    if n < 3:
        return (float('nan'), float('nan'))

    x = x - x.mean()
    lag = int(max(0, min(lag, n - 1)))

    gamma0 = np.dot(x, x) / n
    var = gamma0

    # Bartlett weights
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = np.dot(x[k:], x[:-k]) / n
        var += 2.0 * w * gamma_k

    var_mean = var / n
    if var_mean <= 0 or not np.isfinite(var_mean):
        return (float('nan'), float('nan'))

    se_mean = float(np.sqrt(var_mean))
    t_stat = float(series.mean() / se_mean)
    return (t_stat, se_mean)

def _carve_tail_val_from_train(train_data, min_val_days: int = 10, min_val_samples: int = 200):
    """当外部 val 集不足时，从 train 尾部按日期切一段作为伪验证集（仅用于 early-stopping，
    不用于调参评估/IC 报告）。若无法构造满足条件的伪 val 则原样返回。
    返回 (new_train_data, pseudo_val_data)。
    """
    if len(train_data) == 0:
        return train_data, []
    sorted_data = sorted(train_data, key=lambda x: x[4])
    dates_sorted = pd.to_datetime([x[4] for x in sorted_data])
    uniq_dates = sorted(set(dates_sorted.tolist()))
    if len(uniq_dates) < min_val_days + 5:
        return train_data, []
    cut_set = set(uniq_dates[-min_val_days:])
    pseudo_val = [x for x in sorted_data if pd.to_datetime(x[4]) in cut_set]
    new_train  = [x for x in sorted_data if pd.to_datetime(x[4]) not in cut_set]
    if len(pseudo_val) < min_val_samples or len(new_train) < 500:
        return train_data, []
    return new_train, pseudo_val


def _default_lgbm_params(regularization_strength: str) -> dict:
    """Return a conservative LightGBM parameter set by regularization strength."""
    base = {
        'objective': 'regression',
        'metric': 'l2',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 60,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 1,
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        'verbosity': -1,
        'seed': 42,
    }

    if regularization_strength == 'ultra':
        base.update({
            'num_leaves': 15,
            'min_data_in_leaf': 120,
            'feature_fraction': 0.4,
            'bagging_fraction': 0.4,
            'lambda_l1': 1.0,
            'lambda_l2': 2.0,
        })
    elif regularization_strength == 'enhanced':
        base.update({
            'num_leaves': 23,
            'min_data_in_leaf': 90,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'lambda_l1': 0.5,
            'lambda_l2': 1.0,
        })

    return base


def train_enhanced_lgbm_model(
    X_train,
    y_train,
    X_val,
    y_val,
    use_bayesian_opt: bool,
    regularization_strength: str,
    train_epochs: int = 2000,
):
    """Train a LightGBM model with early stopping. Bayesian optimization is not used here."""
    if use_bayesian_opt:
        print("⚠️ LightGBM版本未启用贝叶斯优化，使用稳健默认参数")

    params = _default_lgbm_params(regularization_strength)
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=min(int(train_epochs), 2000),
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ],
    )

    val_pred = model.predict(X_val)
    return model, val_pred


# All utility functions have been moved to util_factor.py
def create_lgbm_prediction_factor(window=20, rebalance_period=5, train_epochs=3000,
                                    min_history_days=60, min_samples_per_symbol=30,
                                    use_bayesian_opt=True,
                                    use_orthogonalization=True, correlation_threshold=0.95, use_pca=False,
                                    use_improved_features=False, use_robust_training=False,
                                    use_conservative_selection=False, max_features=40,
                                    use_robust_preprocessing=False, use_rank_target=False,
                                    regularization_strength='ultra',  # 'default', 'enhanced', 'ultra'
                                    refresh_feature_selection_each_window=True,
                                    # 集成稳健特征选择参数
                                    use_robust_selection=False,  # 是否使用集成稳健特征选择
                                    use_rolling_window=True,     # 稳健选择中是否使用滚动窗口
                                    use_cv_selection=True,       # 稳健选择中是否使用交叉验证
                                    rolling_window_size=1000,    # 滚动窗口大小
                                    rolling_step_size=500,       # 滚动窗口步长
                                    rolling_min_stability=0.7,   # 滚动窗口最小稳定性
                                    cv_splits=5,                 # CV折数
                                    cv_min_stability=0.8,        # CV最小稳定性
                                    # 贝叶斯优化目标函数选择:
                                    # 'ic_ir'    (推荐) 单指标非重叠IC_IR，无手工权重，杜绝权重泄露
                                    # 'weighted' (兼容) 加权和，仅当权重是在独立holdout集上确定时才无偏
                                    bayes_objective='ic_ir',
                                    # 以下四个权重仅在 bayes_objective='weighted' 时生效
                                    # 默认均为 1.0（等权），如需自定义请确保权重是在独立样本上确定的
                                    ic_ir_weight=1.0, rankIC=1.0, ir_weight=1.0, monotonicity_weight=1.0,
                                    huber_delta=1.0,  # Huber Loss的阈值参数（默认1.0，较小的值对异常值更鲁棒）
                                    # IC 稳定性特征选择参数
                                    use_ic_stability_selection=False,
                                    ic_n_sub_periods=4,
                                    ic_min_positive_ratio=0.6,
                                    ic_recent_days=60,
                                    ic_min_recent_ic=0.0,
                                    # 滚动训练参数
                                    use_rolling_training=False,  # 是否使用滚动训练
                                    use_sliding_window=False,    # True=固定长度滑动窗口 False=扩展式训练
                                    train_window_days=180,       # 训练窗口长度（天数）
                                    val_window_days=180,         # 验证窗口长度（天数）
                                    test_window_days=180,        # 测试窗口长度（天数）
                                    roll_step_days=30,           # 滑动窗口步长（天数）
                                    roll_step_months=6):         # 扩展式滚动步长（月数，已废弃）
    """
    基于LightGBM的收益率预测因子
    使用基于feature.txt的特征定义或改进的特征工程
    支持贝叶斯优化和特征正交化处理
    
    Parameters:
    - use_orthogonalization: 是否使用正交化处理（相关性过滤）
    - correlation_threshold: 相关性阈值，超过此值的特征对将被去除其中一个（默认0.95）
    - use_pca: 是否使用PCA进行正交化（会进一步降低特征维度，但可能丢失可解释性）
    - use_improved_features: 是否使用改进的特征工程（减少过拟合风险）
    - use_robust_training: 是否使用稳健训练策略（增强正则化，减少过拟合）
    - use_conservative_selection: 是否使用保守特征选择策略（更严格的特征筛选）
    - max_features: 保守特征选择的最大特征数量（默认40）
    - use_robust_preprocessing: 是否使用稳健数据预处理（使用中位数和MAD标准化，保守的目标变量处理）
    - use_rank_target: 是否使用排名目标变量（False=预测收益率，True=预测排名）
    - regularization_strength: 正则化强度选择 ('default'=默认, 'enhanced'=增强, 'ultra'=超强)
    - use_robust_selection: 是否使用集成稳健特征选择（结合滚动窗口和CV）
    - use_rolling_window: 稳健选择中是否使用滚动窗口方法
    - use_cv_selection: 稳健选择中是否使用交叉验证方法
    - rolling_window_size: 滚动窗口大小（默认1000）
    - rolling_step_size: 滚动窗口步长（默认500）
    - rolling_min_stability: 滚动窗口特征最小稳定性阈值（默认0.7）
    - cv_splits: 交叉验证折数（默认5）
    - cv_min_stability: CV特征最小稳定性阈值（默认0.8）
    - ic_ir_weight: IC_IR的权重（默认0.4），用于贝叶斯优化目标函数
    - rankIC: Rank IC的权重（默认0.3），用于贝叶斯优化目标函数
    - ir_weight: IR的权重（默认0.2），用于贝叶斯优化目标函数
    - monotonicity_weight: 分组单调性的权重（默认0.2），用于贝叶斯优化目标函数
    - huber_delta: Huber Loss的阈值参数（默认1.0）
                  较小的delta（如0.5）对异常值更鲁棒，但可能降低对正常样本的拟合精度
                  较大的delta（如2.0）更接近MSE，对正常样本拟合更好，但对异常值敏感
    - use_rolling_training: 是否使用滚动训练（默认False）
    - train_window_days: 初始训练窗口长度（天数，默认360）
    - val_window_days: 验证窗口长度（天数，默认180）
    - test_window_days: 测试窗口长度（天数，默认180）
    - roll_step_months: 滚动步长（月数，默认6）
    """
    # 设置全局Huber Loss的delta值（用于全局函数_huber_obj_global和_huber_metric_global）
    global _huber_delta_global
    _huber_delta_global = huber_delta
    
    # 根据正则化强度选择参数
    reg_type_map = {
        'ultra': '超强正则化',
        'enhanced': '增强正则化',
        'default': '默认正则化'
    }
    reg_type = reg_type_map.get(regularization_strength, '默认正则化')

    opt_type = "贝叶斯优化" if use_bayesian_opt else "默认参数"
    ortho_type = "正交化" if use_orthogonalization else "无正交化"
    feature_type = "改进特征" if use_improved_features else "标准特征"
    training_type = "稳健训练" if use_robust_training else "标准训练"

    # 确定特征选择类型
    if use_ic_stability_selection:
        selection_type = "IC稳定性选择"
    elif use_robust_selection:
        selection_type = "集成稳健选择"
    elif use_conservative_selection:
        selection_type = "保守选择"
    else:
        selection_type = "自适应选择"

    preprocessing_type = "稳健预处理" if use_robust_preprocessing else "标准预处理"
    target_type = "预测排名" if use_rank_target else "预测收益率"
    if use_rolling_training and use_sliding_window:
        training_mode = "滑动窗口训练"
    elif use_rolling_training:
        training_mode = "扩展式滚动训练"
    else:
        training_mode = "固定分割"

    print(f"🚀 开始构建LightGBM预测因子 ({opt_type}, {ortho_type}, {feature_type}, {training_type}, {selection_type}, {preprocessing_type}, {target_type}, {reg_type}, {training_mode})")
    print(f"参数: window={window}, rebalance_period={rebalance_period}")
    print(f"      min_history_days={min_history_days}, min_samples_per_symbol={min_samples_per_symbol}")
    print(f"      use_bayesian_opt={use_bayesian_opt}")
    print(f"      use_orthogonalization={use_orthogonalization}, correlation_threshold={correlation_threshold}, use_pca={use_pca}")
    print(f"      use_improved_features={use_improved_features}")
    print(f"      use_robust_training={use_robust_training}")
    print(f"      use_conservative_selection={use_conservative_selection}, max_features={max_features}")
    print(f"      use_robust_preprocessing={use_robust_preprocessing}")
    print(f"      use_rank_target={use_rank_target} ({target_type})")
    print(f"      🔥 使用Huber Loss损失函数 (delta={huber_delta})，增强对异常值的鲁棒性")
    if use_rolling_training and use_sliding_window:
        print(f"      🔄 滑动窗口训练: 训练{train_window_days}天，验证{val_window_days}天，测试{test_window_days}天，步长{roll_step_days}天")
    elif use_rolling_training:
        print(f"      🔄 扩展式滚动训练: 初始{train_window_days}天，验证{val_window_days}天，测试{test_window_days}天，按季度重训")
    else:
        print(f"      📊 固定分割: 训练集、验证集、测试集固定长度")
    if use_ic_stability_selection:
        print(f"      IC稳定性选择: {ic_n_sub_periods}子时段, 最低正IC比例={ic_min_positive_ratio}, 近期IC天数={ic_recent_days}, 最低近期IC={ic_min_recent_ic}")
    elif use_robust_selection:
        print(f"      集成稳健选择: 滚动窗口={use_rolling_window}, CV={use_cv_selection}")
        print(f"      滚动参数: 窗口大小={rolling_window_size}, 步长={rolling_step_size}, 稳定性阈值={rolling_min_stability}")
        print(f"      CV参数: 折数={cv_splits}, 稳定性阈值={cv_min_stability}")

    if not LGBM_AVAILABLE:
        print("LightGBM不可用，无法进行训练")
        return pd.DataFrame()

    # K线数据
    df = load_kline_df()
    
    # 🔥 关键：优先从指数缓存文件加载成分股列表，如果没有则从K线数据计算
    print(f"📊 加载每日前{50}排名，只使用前50个币种...")
    
    # 尝试从指数缓存文件加载
    from util_factor import load_available_tokens_from_index_cache
    # 获取所有日期，用于将调仓日的成分股扩展到所有交易日
    all_dates = sorted(df['date'].unique())
    available_tokens_by_date = load_available_tokens_from_index_cache(
        cache_file=None,  # 自动查找最新的index_cache文件
        top_n=50,
        all_dates=all_dates  # 传入所有日期，将调仓日的成分股扩展到所有交易日
    )
    
    # 如果缓存文件不存在或为空，回退到从K线数据计算
    if not available_tokens_by_date:
        print(f"📊 指数缓存文件不存在或为空，从K线数据构建每日前{50}排名...")
        available_tokens_by_date = build_available_tokens_by_date_from_kline(
            kline_df=df,
            top_n=50,
            ranking_method='quote_volume',
            rebalance_period=rebalance_period,
            strict_top_n=True  # 严格限制为50个币种
        )
        print(f"✅ 排名构建完成: {len(available_tokens_by_date)} 个交易日")
    else:
        print(f"✅ 从指数缓存文件加载完成: {len(available_tokens_by_date)} 个交易日")
    
    # 🔥 关键修复：先计算特征（在全量连续数据上），再过滤（只保留Top50期间的样本）
    # 这样可以避免K线数据不连续导致的特征计算错误（如pct_change计算3个月的涨跌幅被当成1天）
    print("🔄 准备全量特征（先计算后过滤，保证K线连续性）...")
    
    # 准备训练数据（返回原始目标变量用于IC计算）
    # 注意：这里传入原始的df（未过滤的），函数内部会先计算特征，然后再过滤
    if use_robust_preprocessing:
        all_features, all_targets, all_targets_original, all_symbols, all_dates = prepare_robust_data(
            df, rebalance_period, min_history_days, n_jobs=40, 
            use_improved_features=use_improved_features,
            use_rank_target=use_rank_target,
            available_tokens_by_date=available_tokens_by_date  # 传入过滤条件
        )
    else:
        all_features, all_targets, all_targets_original, all_symbols, all_dates = prepare_lgbm_data_fast(
            df, rebalance_period, min_history_days, n_jobs=40, 
            use_improved_features=use_improved_features,
            use_rank_target=use_rank_target,
            available_tokens_by_date=available_tokens_by_date  # 传入过滤条件
    )

    # 路径准备
    model_path, selector_path, scaler_path = _artifact_paths(window, rebalance_period, model_type='lgbm')

    # 不再需要scaler_path，因为已经移除了RobustScaler
    use_cached = model_path.exists() and selector_path.exists()
    if use_cached:
        print(f"📦 检测到已训练模型与预处理器，直接加载: {model_path.name}")
    else:
        print("🧪 未发现已训练模型，将进行训练并保存。")
    
    if len(all_features) < 1000:
        print("警告: 训练数据不足，需要至少1000个样本")
        return pd.DataFrame()
    
    # 时间序列分割（包含原始目标变量）
    data_with_dates = list(zip(all_features, all_targets, all_targets_original, all_symbols, all_dates))

    if use_rolling_training and use_sliding_window:
        rolling_windows = create_monthly_sliding_window_time_split(
            data_with_dates,
            train_window_days=train_window_days,
            val_window_days=val_window_days,
            test_window_days=test_window_days,
            roll_step_days=roll_step_days,
            min_train_samples=1000
        )
        print(f"🎯 固定窗口滑动训练模式：生成 {len(rolling_windows)} 个训练窗口")
    elif use_rolling_training:
        rolling_windows = create_quarterly_expanding_rolling_time_split(
            data_with_dates,
            initial_train_days=train_window_days,
            val_window_days=val_window_days,
            test_window_days=test_window_days,
            min_train_samples=1000
        )
        print(f"🎯 扩展式滚动训练模式：生成 {len(rolling_windows)} 个训练窗口")
    else:
        # 固定分割：返回单个窗口（包装成列表以统一处理）
        train_data, val_data, test_data = create_strict_time_split(
            data_with_dates, test_days=180, validation_days=150
        )
        # 包装成与滚动窗口相同的格式
        window_info = {
            'window_start': min(all_dates),
            'train_end': max([d for d in all_dates if d < min([item[4] for item in val_data])]),
            'val_end': max([d for d in all_dates if d < min([item[4] for item in test_data])]),
            'test_end': max(all_dates),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data)
        }
        rolling_windows = [(train_data, val_data, test_data, window_info)]
        print(f"📊 固定分割模式：使用传统的时间序列分割")

    # 初始化用于收集所有窗口结果的列表
    all_models = []
    all_selectors = []
    all_window_results = []
    all_predictions_data = []

    # 对每个滚动窗口进行训练
    for window_idx, (train_data, val_data, test_data, window_info) in enumerate(rolling_windows):
        print(f"\n🏁 处理窗口 {window_idx + 1}/{len(rolling_windows)} (idx={window_idx}): {window_info['window_start'].date()} ~ {window_info['test_end'].date()}")

        # 标签purge：避免标签窗口穿越 train/val/test 边界
        train_data, val_data, test_data = apply_label_purge_to_split(
            train_data,
            val_data,
            test_data,
            purge_days=rebalance_period,
            date_index=4,
        )
        if len(train_data) < 1000 or len(val_data) == 0 or len(test_data) == 0:
            print(
                f"⚠️ 窗口 {window_idx + 1} 在purge后样本不足，跳过。"
                f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
            )
            continue

        # 分离数据（使用排名目标变量用于训练，原始目标变量用于IC计算）
        print(f"   原始数据规模: 训练 {len(train_data)} 验证 {len(val_data)} 测试 {len(test_data)}")
        print(f"   window_idx判断: {window_idx == 0}")
        X_train = np.array([item[0] for item in train_data])
        y_train = np.array([item[1] for item in train_data])
        y_train_original = np.array([item[2] for item in train_data])
        symbols_train = [item[3] for item in train_data]
        dates_train = [item[4] for item in train_data]

        X_val = np.array([item[0] for item in val_data])
        y_val = np.array([item[1] for item in val_data])
        y_val_original = np.array([item[2] for item in val_data])
        symbols_val = [item[3] for item in val_data]
        dates_val = [item[4] for item in val_data]

        X_test = np.array([item[0] for item in test_data])
        y_test = np.array([item[1] for item in test_data])
        y_test_original = np.array([item[2] for item in test_data])
        symbols_test = [item[3] for item in test_data]
        dates_test = [item[4] for item in test_data]

        print(f"   处理后数据规模: 训练 {X_train.shape[0]} 验证 {X_val.shape[0]} 测试 {X_test.shape[0]}")

        # ── 验证集兜底：验证集过短时关闭贝叶斯优化，并尝试从训练集尾部切伪验证集 ──
        _min_val_days    = max(int(rebalance_period) * 2, 10)   # 至少覆盖 2 个换仓周期
        _min_val_samples = max(200, int(rebalance_period) * 20)
        val_uniq_days = int(pd.Series(dates_val).drop_duplicates().shape[0]) if len(dates_val) > 0 else 0
        val_is_short  = (len(val_data) < _min_val_samples) or (val_uniq_days < _min_val_days)
        use_bayes_this_window = use_bayesian_opt

        if val_is_short:
            print(f"⚠️ 窗口 {window_idx + 1}: 验证集不足 "
                  f"(samples={len(val_data)}, days={val_uniq_days}, "
                  f"要求 samples≥{_min_val_samples} & days≥{_min_val_days})")
            use_bayes_this_window = False
            print("   → 已关闭贝叶斯优化，改用稳健默认参数")

            pseudo_train, pseudo_val = _carve_tail_val_from_train(
                train_data,
                min_val_days=_min_val_days,
                min_val_samples=_min_val_samples,
            )
            if len(pseudo_val) > 0:
                train_data = pseudo_train
                val_data   = pseudo_val
                # 重建 numpy 数组和列表
                X_train = np.array([item[0] for item in train_data])
                y_train = np.array([item[1] for item in train_data])
                y_train_original = np.array([item[2] for item in train_data])
                symbols_train = [item[3] for item in train_data]
                dates_train   = [item[4] for item in train_data]
                X_val = np.array([item[0] for item in val_data])
                y_val = np.array([item[1] for item in val_data])
                y_val_original = np.array([item[2] for item in val_data])
                symbols_val = [item[3] for item in val_data]
                dates_val   = [item[4] for item in val_data]
                print(f"   → 已从训练集尾部切出伪验证集: "
                      f"train={len(train_data)}, pseudo_val={len(val_data)}")
            else:
                print("   → 无法构造伪验证集，沿用原有短验证集（仅保证 early-stopping）")

        # 为每个窗口生成独立的模型和选择器路径
        window_model_path = Path(model_path.parent) / f"{model_path.stem}_window{window_idx}{model_path.suffix}"
        window_selector_path = Path(selector_path.parent) / f"{selector_path.stem}_window{window_idx}{selector_path.suffix}"

        # 检查是否已有缓存的窗口模型
        use_cached_window = window_model_path.exists() and window_selector_path.exists()
        if refresh_feature_selection_each_window:
            use_cached_window = False
            print("🔁 已启用每窗口重做特征选择，跳过缓存模型与选择器")
        if use_cached_window:
            print(f"📦 检测到已训练的窗口模型: {window_model_path.name}")
        else:
            print("🧪 未发现已训练的窗口模型，将进行训练。")

        # 特征选择 - 支持多种策略
        print("🔄 进行特征选择...")
        if use_cached_window:
            selector = joblib.load(window_selector_path)
            X_train_selected = selector.transform(X_train)
            X_val_selected   = selector.transform(X_val)
            X_test_selected  = selector.transform(X_test)
        else:
            if use_ic_stability_selection:
                selector, X_train_selected, X_val_selected, X_test_selected = ic_stability_feature_selection(
                    X_train, y_train, dates_train, symbols_train,
                    X_val, y_val, dates_val, symbols_val,
                    X_test,
                    max_features=max_features,
                    correlation_threshold=correlation_threshold,
                    n_sub_periods=ic_n_sub_periods,
                    min_positive_ratio=ic_min_positive_ratio,
                    recent_ic_days=ic_recent_days,
                    min_recent_ic=ic_min_recent_ic,
                )
            elif use_robust_selection:
                rolling_params = {
                    'window_size': rolling_window_size,
                    'step_size': rolling_step_size,
                    'min_stability': rolling_min_stability
                } if use_rolling_window else None

                cv_params = {
                    'n_splits': cv_splits,
                    'min_cv_stability': cv_min_stability
                } if use_cv_selection else None

                selector, X_train_selected, X_val_selected, X_test_selected = robust_feature_selection(
                    X_train, y_train, X_val, y_val, X_test,
                    use_rolling=use_rolling_window,
                    use_cv=use_cv_selection,
                    rolling_params=rolling_params,
                    cv_params=cv_params,
                    max_features=max_features,
                    correlation_threshold=correlation_threshold
                )
            elif use_conservative_selection:
                # 使用保守特征选择策略
                selector, X_train_selected, X_val_selected, X_test_selected = conservative_feature_selection(
                    X_train, y_train, X_val, y_val, X_test,
                    max_features=max_features,
                    correlation_threshold=correlation_threshold
                )
            else:
                # 使用自适应特征选择策略
                selector, X_train_selected, X_val_selected, X_test_selected = adaptive_feature_selection(
                    X_train, y_train, X_val, y_val, X_test,
                    target_features=100,
                    importance_threshold=0.001,
                    correlation_threshold=correlation_threshold,
                    use_orthogonalization=use_orthogonalization,
                    use_pca=use_pca
                )
            # 保存特征选择器（无论哪种选择方法）
            joblib.dump(selector, window_selector_path)
            print(f"特征选择: {X_train.shape[1]} -> {X_train_selected.shape[1]}")

            # 特征相关性分析和可视化
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            # 获取实际的特征名称
            all_feature_names = get_feature_names(use_improved_features)

            corr_save_path = reports_dir / f"lgbm_features_correlation_w{window}_r{rebalance_period}_w{window_idx}.png"
            analyze_selected_features_correlation(
                X_train_selected, X_train_selected.shape[1],
                title=f"LightGBM选定特征相关性分析 (w{window}_r{rebalance_period}_window{window_idx})",
                save_path=str(corr_save_path),
                feature_names=all_feature_names[:X_train_selected.shape[1]]  # 只取选定特征的数量
            )

            # 特征重要性分析
            importance_save_path = reports_dir / f"lgbm_features_importance_w{window}_r{rebalance_period}_w{window_idx}.png"
            analyze_feature_importance(
                X_train_selected, y_train, X_train_selected.shape[1],
                title=f"LightGBM特征重要性分析 (w{window}_r{rebalance_period}_window{window_idx})",
                save_path=str(importance_save_path),
                feature_names=all_feature_names[:X_train_selected.shape[1]]  # 只取选定特征的数量
            )

        # 标准化 - 已移除RobustScaler，因为已经在prepare_lgbm_data_fast中按日期进行了横截面标准化
        # 横截面标准化已经将每个日期的特征标准化为均值0、标准差1，不需要再做全局标准化
        # 这样可以保持不同季度间的一致性，避免因训练集统计量导致的分布偏移
        print("✅ 特征已在prepare_lgbm_data_fast中按日期进行横截面标准化，跳过RobustScaler")
        # 不需要RobustScaler，因为已经做了横截面标准化
        X_train_scaled = X_train_selected
        X_val_scaled   = X_val_selected
        X_test_scaled  = X_test_selected

        # 模型：加载或训练
        if use_cached_window:
            model = lgb.Booster(model_file=str(window_model_path))
            print("✅ 已加载缓存模型与预处理器")
        else:
            model, val_pred = train_enhanced_lgbm_model(
                X_train_scaled, y_train, X_val_scaled, y_val,
                use_bayesian_opt=use_bayes_this_window,   # 窗口级开关（短验证集时自动关闭）
                regularization_strength=regularization_strength,
                train_epochs=train_epochs,
            )
            # 保存模型
            model.save_model(str(window_model_path))
            print(f"✅ 模型已保存到: {window_model_path}")
        
        # 验证集预测（用于IC计算）
        print(f"🔄 窗口 {window_idx + 1} 开始验证集预测...")
        val_pred = model.predict(X_val_scaled)

        # 计算验证集IC（统一口径：rawIC始终对原始未来收益；如use_rank_target=True，额外给出rankIC）
        val_df = pd.DataFrame({
            'date': dates_val,
            'pred': val_pred,
            'actual_raw': y_val_original,
            'actual_rank': y_val,
        })

        lag = max(0, int(rebalance_period) - 1)

        val_ic_raw = _daily_cross_section_corr(val_df, 'pred', 'actual_raw')
        val_avg_ic_raw = float(val_ic_raw.mean()) if not val_ic_raw.empty else 0.0
        val_ic_raw_no = _subsample_non_overlapping(val_ic_raw, step=int(rebalance_period))
        val_avg_ic_raw_no = float(val_ic_raw_no.mean()) if not val_ic_raw_no.empty else 0.0
        val_t_raw, val_se_raw = _newey_west_tstat(val_ic_raw, lag=lag)
        print(
            f"[窗口 {window_idx + 1} 验证集] rawIC均值={val_avg_ic_raw:.4f} (NW t={val_t_raw:.2f}, se={val_se_raw:.4f}, n={val_ic_raw.shape[0]}) "
            f"| 非重叠均值(step={rebalance_period})={val_avg_ic_raw_no:.4f} (n={val_ic_raw_no.shape[0]})"
        )

        if use_rank_target:
            val_ic_rank = _daily_cross_section_corr(val_df, 'pred', 'actual_rank')
            val_avg_ic_rank = float(val_ic_rank.mean()) if not val_ic_rank.empty else 0.0
            val_ic_rank_no = _subsample_non_overlapping(val_ic_rank, step=int(rebalance_period))
            val_avg_ic_rank_no = float(val_ic_rank_no.mean()) if not val_ic_rank_no.empty else 0.0
            val_t_rank, val_se_rank = _newey_west_tstat(val_ic_rank, lag=lag)
            print(
                f"[窗口 {window_idx + 1} 验证集] rankIC均值={val_avg_ic_rank:.4f} (NW t={val_t_rank:.2f}, se={val_se_rank:.4f}, n={val_ic_rank.shape[0]}) "
                f"| 非重叠均值(step={rebalance_period})={val_avg_ic_rank_no:.4f} (n={val_ic_rank_no.shape[0]})"
            )
        
        # 样本外预测和评估
        print(f"🔄 窗口 {window_idx + 1} 开始样本外预测...")
        test_pred = model.predict(X_test_scaled)
        
        # 异常值处理
        def smart_clip_values(y, clip_percentile=98):
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = max(lower_bound, np.percentile(y, 100 - clip_percentile))
            upper_bound = min(upper_bound, np.percentile(y, clip_percentile))
            return np.clip(y, lower_bound, upper_bound)
        
        # 根据use_rank_target选择用于评估的目标变量
        if use_rank_target:
            y_test_for_eval = smart_clip_values(y_test, clip_percentile=98)
        else:
            y_test_for_eval = smart_clip_values(y_test_original, clip_percentile=98)
        
        # 样本外指标
        test_rmse = np.sqrt(mean_squared_error(y_test_for_eval, test_pred))
        test_mae = mean_absolute_error(y_test_for_eval, test_pred)
        test_r2 = r2_score(y_test_for_eval, test_pred)

        # 计算样本外IC（统一口径：rawIC始终对原始未来收益；如use_rank_target=True，额外给出rankIC）
        test_df = pd.DataFrame({
            'date': dates_test,
            'pred': test_pred,
            'actual_raw': y_test_original,
            'actual_rank': y_test,
        })

        test_ic_raw = _daily_cross_section_corr(test_df, 'pred', 'actual_raw')
        test_avg_ic_raw = float(test_ic_raw.mean()) if not test_ic_raw.empty else 0.0
        test_ic_raw_no = _subsample_non_overlapping(test_ic_raw, step=int(rebalance_period))
        test_avg_ic_raw_no = float(test_ic_raw_no.mean()) if not test_ic_raw_no.empty else 0.0
        test_t_raw, test_se_raw = _newey_west_tstat(test_ic_raw, lag=lag)

        test_avg_ic = test_avg_ic_raw
        print(f"[窗口 {window_idx + 1} 样本外-Test]  RMSE={test_rmse:.6f}, MAE={test_mae:.6f}, R2={test_r2:.4f}")
        print(
            f"[窗口 {window_idx + 1} 样本外-Test]  rawIC均值={test_avg_ic_raw:.4f} (NW t={test_t_raw:.2f}, se={test_se_raw:.4f}, n={test_ic_raw.shape[0]}) "
            f"| 非重叠均值(step={rebalance_period})={test_avg_ic_raw_no:.4f} (n={test_ic_raw_no.shape[0]})"
        )

        test_avg_ic_rank = None
        if use_rank_target:
            test_ic_rank = _daily_cross_section_corr(test_df, 'pred', 'actual_rank')
            test_avg_ic_rank = float(test_ic_rank.mean()) if not test_ic_rank.empty else 0.0
            test_ic_rank_no = _subsample_non_overlapping(test_ic_rank, step=int(rebalance_period))
            test_avg_ic_rank_no = float(test_ic_rank_no.mean()) if not test_ic_rank_no.empty else 0.0
            test_t_rank, test_se_rank = _newey_west_tstat(test_ic_rank, lag=lag)
            print(
                f"[窗口 {window_idx + 1} 样本外-Test]  rankIC均值={test_avg_ic_rank:.4f} (NW t={test_t_rank:.2f}, se={test_se_rank:.4f}, n={test_ic_rank.shape[0]}) "
                f"| 非重叠均值(step={rebalance_period})={test_avg_ic_rank_no:.4f} (n={test_ic_rank_no.shape[0]})"
            )

        # 收集窗口结果用于最终汇总
        window_result = {
            'window_idx': window_idx,
            'window_info': window_info,
            'test_metrics': {
                'rmse': test_rmse,
                'mae': test_mae,
                'r2': test_r2,
                'avg_ic': test_avg_ic,  # rawIC均值（对future_ret），保持兼容
                'avg_rank_ic': test_avg_ic_rank,
                'ic_n_dates': int(test_ic_raw.shape[0]),
                'ic_nw_lag': int(lag),
            },
            'model': model,
            'selector': selector
        }
        all_window_results.append(window_result)

        # 生成当前窗口的因子数据（所有窗口都只保存测试集数据）
        window_result_data = []

        # 所有窗口都只保存测试集数据（样本外表现）
        sets = [
            ("test",  X_test_scaled,  symbols_test,  dates_test,  y_test_original),
        ]

        for split, Xs, syms, dts, ys_orig in sets:
            if len(Xs) == 0:
                continue
            preds = model.predict(Xs)
            for pred, s, d, y_true_orig in zip(preds, syms, dts, ys_orig):
                window_result_data.append({
                    'date': d,
                    'symbol': s,
                    'lgbm_prediction': float(pred),
                    'future_ret': float(y_true_orig),  # 使用原始目标变量
                    'split': split,
                    'window_idx': window_idx,  # 添加窗口索引用于跟踪
                    'window_start': window_info['window_start'],  # 添加窗口开始时间
                    'window_end': window_info['test_end']  # 添加窗口结束时间
                })

        all_predictions_data.extend(window_result_data)

        # 存储模型和选择器用于后续使用
        all_models.append(model)
        all_selectors.append(selector)

        # 调试信息：统计当前窗口收集的数据
        window_train_count = sum(1 for item in window_result_data if item['split'] == 'train')
        window_val_count = sum(1 for item in window_result_data if item['split'] == 'val')
        window_test_count = sum(1 for item in window_result_data if item['split'] == 'test')

        print(f"✅ 窗口 {window_idx + 1} 处理完成，获得 {len(window_result_data)} 个预测样本")
        print(f"   窗口数据分布: 训练集 {window_train_count} 个, 验证集 {window_val_count} 个, 测试集 {window_test_count} 个")
        print(f"   累积数据总量: {len(all_predictions_data)} 个")

    # 循环结束：汇总所有窗口的结果（所有窗口的测试集数据）
    if len(all_predictions_data) == 0:
        print("警告: 无可用预测样本，返回空 DataFrame")
        return pd.DataFrame()

    print(f"\n🎯 滚动训练完成，共处理 {len(rolling_windows)} 个窗口")

    # 统计各类型数据
    train_count = sum(1 for item in all_predictions_data if item['split'] == 'train')
    val_count = sum(1 for item in all_predictions_data if item['split'] == 'val')
    test_count = sum(1 for item in all_predictions_data if item['split'] == 'test')

    print(f"📊 数据汇总: 训练集 {train_count} 个, 验证集 {val_count} 个, 测试集 {test_count} 个")
    print(f"📊 总计 {len(all_predictions_data)} 个预测样本")

    # 汇总所有预测结果（所有窗口的测试集数据）
    factor_df = pd.DataFrame(all_predictions_data)

    # 对所有测试集数据进行rank归一化
    if len(factor_df) > 0:
        factor_df, removed_dups = dedupe_predictions_keep_latest_window(
            factor_df,
            date_col="date",
            symbol_col="symbol",
            window_col="window_idx",
        )
        if removed_dups > 0:
            print(f"🧹 去除重叠窗口重复样本: {removed_dups} 条")
        factor_df = rank_to_unit_by_date(factor_df, col="lgbm_prediction", out_col="factor")
    else:
        print("⚠️ 警告：没有预测数据！")
        factor_df = pd.DataFrame()

    # 保存因子数据
    final_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret", "split", "window_idx"]]

    if len(final_df) > 0:
        out_path = save_factor_df(final_df, file_prefix=f"lgbm_{window}d_rebalance{rebalance_period}d_")
        print_factor_summary(final_df, out_path)
    else:
        print("⚠️ 没有因子数据可以保存")

    # 新增：如果因"目标位移"导致factor_df不是最新自然日，则进行一次Live推理并打印真正最新日
    print_latest_daily_groups_live(
        window=window, 
        rebalance_period=rebalance_period, 
        min_history_days=min_history_days, 
        groups=5,
        use_improved_features=use_improved_features,
        use_robust_preprocessing=use_robust_preprocessing,
        model_type='lgbm'
    )
    
    return factor_df

if __name__ == "__main__":
    create_lgbm_prediction_factor(
        window=10,
        rebalance_period=10,
        train_epochs=3000,
        min_history_days=30,
        min_samples_per_symbol=50,
        use_bayesian_opt=True,
        use_orthogonalization=True,
        correlation_threshold=0.50,
        use_pca=False,
        use_improved_features=False,
        use_robust_training=True,
        # 特征选择：使用 IC 稳定性选择
        use_ic_stability_selection=True,
        ic_n_sub_periods=4,
        ic_min_positive_ratio=0.6,
        ic_recent_days=60,
        ic_min_recent_ic=0.0,
        max_features=30,
        use_robust_preprocessing=True,
        use_rank_target=True,
        regularization_strength='ultra',
        huber_delta=0.3,
        # 贝叶斯优化目标：单指标IC_IR（推荐，无权重泄露）
        bayes_objective='ic_ir',
        # 滑动窗口训练（固定长度，非扩展式）
        use_rolling_training=True,
        use_sliding_window=True,
        train_window_days=180,
        val_window_days=60,
        test_window_days=60,
        roll_step_days=60,
    )
