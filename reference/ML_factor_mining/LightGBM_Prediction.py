# factor_analyse/ML_Factor/LightGBM_Prediction_Factor_Enhanced.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    LGB_AVAILABLE = True
    print("✅ LightGBM可用")
except ImportError:
    print("警告: LightGBM未安装，请安装: pip install lightgbm")
    LGB_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("✅ Optuna可用")
except ImportError:
    print("警告: Optuna未安装，请安装: pip install optuna")
    OPTUNA_AVAILABLE = False

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

from collections import defaultdict
from tqdm import tqdm
from scipy import stats
import os
from pathlib import Path
import joblib

def _artifact_paths(window: int, rebalance_period: int):
    base_dir = Path(os.path.dirname(__file__)) / "models"
    base_dir.mkdir(parents=True, exist_ok=True)
    tag = f"lgb_enhanced_w{window}_r{rebalance_period}"
    # tag = f"lgb_w{window}_r{rebalance_period}"
    model_path = base_dir / f"{tag}.txt"
    selector_path = base_dir / f"{tag}_selector.pkl"
    scaler_path = base_dir / f"{tag}_scaler.pkl"
    return model_path, selector_path, scaler_path

def create_features_symbol_fast(gp: pd.DataFrame, rebalance_period: int, min_history_days: int = 60):
    """
    向量化为单个symbol一次性生成全部特征与未来收益；避免逐点for循环.
    仅使用历史滚动值（rolling/ewm/shift），无未来信息泄露。
    """
    eps = 1e-8
    gp = gp.sort_values('date').reset_index(drop=True).copy()

    if len(gp) < min_history_days + rebalance_period + 20:
        return pd.DataFrame()

    close = gp['close']
    high  = gp['high']
    low   = gp['low']
    vol   = gp['volume']
    qv    = gp['quote_volume'] if 'quote_volume' in gp.columns else None
    tbq   = gp['taker_buy_quote'] if 'taker_buy_quote' in gp.columns else None

    feat = pd.DataFrame({
        'date': gp['date'],
        'symbol': gp['symbol']
    })

    # 基础
    feat['open']   = gp['open']
    feat['high']   = high
    feat['low']    = low
    feat['close']  = close
    feat['volume'] = vol

    if qv is not None:
        feat['quote_volume'] = qv
        feat['vwap'] = qv / (vol + eps)
    else:
        feat['vwap'] = close

    # 价格动量
    for n in (1,3,5,10,20,30):
        feat[f'ret_{n}']     = close.pct_change(n)
        feat[f'logret_{n}']  = np.log(close / close.shift(n))
        feat[f'volume_ret_{n}'] = vol.pct_change(n)

    # 均线（滚动/指数）
    for n in (5,10,20,30,60):
        sma = close.rolling(n).mean()
        ema = close.ewm(span=n, adjust=False).mean()
        feat[f'sma_{n}'] = sma
        feat[f'ema_{n}'] = ema
        feat[f'price_sma_ratio_{n}'] = close / (sma + eps)
        feat[f'price_ema_ratio_{n}'] = close / (ema + eps)

    # 波动率与区间位置
    logret1 = np.log(close / close.shift(1))
    for n in (5,10,20,30):
        voln = logret1.rolling(n).std()
        feat[f'volatility_{n}'] = voln
        feat[f'high_low_ratio_{n}'] = (high - low).rolling(n).mean() / (close + eps)
        rmin = low.rolling(n).min()
        rmax = high.rolling(n).max()
        feat[f'close_position_{n}'] = (close - rmin) / (rmax - rmin + eps)

    # 成交量特征
    for n in (5,10,20,30):
        vma = vol.rolling(n).mean()
        feat[f'volume_sma_{n}'] = vma
        feat[f'volume_ratio_{n}'] = vol / (vma + eps)
        if qv is not None:
            qvma = qv.rolling(n).mean()
            feat[f'quote_volume_sma_{n}'] = qvma
            feat[f'quote_volume_ratio_{n}'] = qv / (qvma + eps)

    # RSI
    for period in (14,21,30):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + eps)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat['macd'] = macd
    feat['macd_signal'] = signal
    feat['macd_histogram'] = macd - signal

    # 布林带
    for period in (20,30,60):
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + 2*std
        lower = sma - 2*std
        feat[f'bb_upper_{period}'] = upper
        feat[f'bb_lower_{period}'] = lower
        feat[f'bb_width_{period}'] = (upper - lower) / (sma + eps)
        feat[f'bb_position_{period}'] = (close - lower) / (upper - lower + eps)

    # 主动买盘
    if tbq is not None:
        feat['taker_buy_quote'] = tbq
        if qv is not None:
            buy_ratio = tbq / (qv + eps)
            feat['buy_ratio'] = buy_ratio
            for n in (5,10,20,30):
                feat[f"buy_ratio_sma_{n}"] = buy_ratio.rolling(n).mean()
                feat[f"buy_ratio_std_{n}"] = buy_ratio.rolling(n).std()
                feat[f"buy_ratio_price_corr_{n}"] = close.pct_change().rolling(n).corr(buy_ratio)

        for n in (5,10,20,30):
            feat[f"buyq_sma_{n}"] = tbq.rolling(n).mean()
            feat[f"buyq_std_{n}"] = tbq.rolling(n).std()
            feat[f"buyq_roc_{n}"] = tbq.pct_change(n)

    # Alpha构件
    feat["alpha41_geo_vwap"] = np.sqrt(high * low) - feat["vwap"]
    if qv is not None:
        vwap10 = (qv / (vol + eps)).rolling(10).mean()
    else:
        vwap10 = close.rolling(10).mean()
    feat["alpha5_open_minus_vwap10"] = gp["open"] - vwap10
    feat["alpha5_neg_abs_close_minus_vwap"] = -(close - feat["vwap"]).abs()

    # 目标：未来对数收益（仅shift，不参与特征构造）
    feat["future_ret"] = np.log(close.shift(-rebalance_period) / close)

    # 丢掉前期不完整/末尾无目标的行
    feat = feat.iloc[min_history_days : -rebalance_period].dropna().reset_index(drop=True)
    return feat

def prepare_lightgbm_data_fast(df, rebalance_period=5, min_history_days=60, n_jobs=-1):
    """
    向量化 + 并行 的数据准备。返回与原接口兼容的四元组。
    """
    from joblib import Parallel, delayed

    symbols = df['symbol'].unique().tolist()

    def run_one(sym):
        gp = df[df['symbol'] == sym].copy()
        if gp.empty: return pd.DataFrame()
        return create_features_symbol_fast(gp, rebalance_period, min_history_days)

    dfs = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(run_one)(s) for s in symbols)
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return np.array([]), np.array([]), [], []

    out = pd.concat(dfs, ignore_index=True)

    X = out.drop(columns=['date','symbol','future_ret']).to_numpy(dtype=np.float32)
    y = out['future_ret'].to_numpy(dtype=np.float32)
    syms = out['symbol'].tolist()
    dates = pd.to_datetime(out['date']).tolist()

    print(f"✅ 向量化+并行 数据样本: {len(out)}，特征维度: {X.shape[1]}")
    return X, y, syms, dates

def create_strict_time_split(data_with_dates, test_days=180, validation_days=90):
    """严格的时间序列分割"""
    print("�� 严格时间序列分割...")
    
    data_with_dates.sort(key=lambda x: x[3])
    dates = [item[3] for item in data_with_dates]
    min_date, max_date = min(dates), max(dates)
    
    # 测试集：最后test_days天
    test_cutoff = max_date - pd.Timedelta(days=test_days - 1)
    test_data = [it for it in data_with_dates if it[3] >= test_cutoff]
    
    # 验证集：测试集之前的validation_days天
    val_cutoff = test_cutoff - pd.Timedelta(days=validation_days)
    val_data = [it for it in data_with_dates if val_cutoff <= it[3] < test_cutoff]
    
    # 训练集：验证集之前的所有数据
    train_data = [it for it in data_with_dates if it[3] < val_cutoff]
    
    print(f"数据时间范围: {min_date} ~ {max_date}")
    print(f"训练集: {min_date} ~ {val_cutoff.date()} ({len(train_data)} 样本)")
    print(f"验证集: {val_cutoff.date()} ~ {test_cutoff.date()} ({len(val_data)} 样本)")
    print(f"测试集: {test_cutoff.date()} ~ {max_date} ({len(test_data)} 样本)")
    
    return train_data, val_data, test_data

def train_enhanced_lightgbm_model(X_train, y_train, X_val, y_val, symbols_train, symbols_val, dates_train, dates_val, 
                                  params=None, verbose=True):
    """训练增强版LightGBM模型
    
    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        symbols_train, symbols_val: 训练和验证集的symbol列表
        dates_train, dates_val: 训练和验证集的日期列表
        params: LightGBM参数字典，如果为None则使用默认参数
        verbose: 是否打印详细信息
    """
    if verbose:
        print("🚀 开始增强版LightGBM训练...")
    
    # 更智能的异常值处理
    def smart_clip_values(y, clip_percentile=98):
        """使用更智能的异常值处理"""
        # 使用IQR方法检测异常值
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 结合分位数方法
        lower_bound = max(lower_bound, np.percentile(y, 100 - clip_percentile))
        upper_bound = min(upper_bound, np.percentile(y, clip_percentile))
        
        return np.clip(y, lower_bound, upper_bound)
    
    y_train_clipped = smart_clip_values(y_train, clip_percentile=98)
    y_val_clipped = smart_clip_values(y_val, clip_percentile=98)
    
    if verbose:
        print(f"目标收益率统计:")
        print(f"  训练集: 均值={y_train_clipped.mean():.6f}, 标准差={y_train_clipped.std():.6f}")
        print(f"  验证集: 均值={y_val_clipped.mean():.6f}, 标准差={y_val_clipped.std():.6f}")
    
    # 默认参数
    if params is None:
        params = {
            'objective': 'mae',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'max_depth': 6,
            'num_leaves': 63,
            'min_child_samples': 10,
            'min_split_gain': 0.01,
            'bagging_fraction': 0.5,
            'bagging_freq': 1,
            'feature_fraction': 0.5,
            'feature_fraction_bynode': 0.8,
            'lambda_l1': 0.5,
            'lambda_l2': 0.8,
            'num_threads': -1,
            'seed': 42,
            'verbose': -1 if not verbose else 1,
        }
    else:
        # 确保必要的参数存在
        if 'verbose' not in params:
            params['verbose'] = -1 if not verbose else 1
        if 'num_threads' not in params:
            params['num_threads'] = -1
        if 'seed' not in params:
            params['seed'] = 42
    
    # 创建Dataset
    if verbose:
        print("📊 创建训练数据集...")
    dtrain = lgb.Dataset(X_train, label=y_train_clipped)
    dval = lgb.Dataset(X_val, label=y_val_clipped, reference=dtrain)
    
    # 训练模型
    # 使用更保守的早停：从150轮增加到300轮，给模型更多学习机会
    stopping_rounds = 300
    if verbose:
        print(f"🏋️ 开始训练模型（最多3000轮，早停{stopping_rounds}轮）...")
        print("   每20轮输出一次进度，请耐心等待...")
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False),
    ]
    if verbose:
        callbacks.append(lgb.log_evaluation(period=20))
    else:
        callbacks.append(lgb.log_evaluation(period=1000))  # 几乎不输出
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    if verbose:
        print(f"✅ 训练完成！最佳迭代轮次: {model.best_iteration}")
    
    # 验证集预测
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    if verbose:
        val_rmse = np.sqrt(mean_squared_error(y_val_clipped, val_pred))
        val_mae = mean_absolute_error(y_val_clipped, val_pred)
        val_r2 = r2_score(y_val_clipped, val_pred)
        print(f"[验证集] RMSE={val_rmse:.6f}, MAE={val_mae:.6f}, R2={val_r2:.4f}")
    
    # 计算横截面IC
    val_df = pd.DataFrame({
        'date': dates_val,
        'pred': val_pred,
        'actual': y_val_clipped
    })
    daily_corrs = []
    for date, group in val_df.groupby('date'):
        if len(group) >= 5:
            corr = group['pred'].corr(group['actual'])
            if not np.isnan(corr):
                daily_corrs.append(corr)
    
    avg_ic = np.mean(daily_corrs) if daily_corrs else 0
    if verbose:
        print(f"[验证集] 平均IC={avg_ic:.4f}, 有效日期数={len(daily_corrs)}")
    
    return model, val_pred, avg_ic

def bayesian_optimize_lightgbm(X_train, y_train, X_val, y_val, symbols_train, symbols_val, 
                                dates_train, dates_val, n_trials=50):
    """使用贝叶斯优化（Optuna）寻找最佳LightGBM参数
    
    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        symbols_train, symbols_val: 训练和验证集的symbol列表
        dates_train, dates_val: 训练和验证集的日期列表
        n_trials: 优化试验次数
    
    Returns:
        best_params: 最佳参数字典
        best_ic: 最佳IC值
    """
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna未安装，无法进行贝叶斯优化")
        return None, None
    
    print(f"🔍 开始贝叶斯优化，共{n_trials}次试验...")
    print("   优化目标：稳健IC指标（验证集IC中位数 + 均值加权）")
    print("   参数搜索空间：更保守的设置（限制learning_rate，提高min_split_gain，增强正则化）")
    
    def objective(trial):
        """优化目标函数：最大化稳健的IC指标"""
        # 定义参数搜索空间（更保守的设置）
        max_depth = trial.suggest_int('max_depth', 4, 7)  # 限制深度范围，避免过深
        # num_leaves应该不超过2^max_depth
        max_num_leaves = min(2 ** max_depth, 127)
        # 确保下限不超过上限
        min_num_leaves = min(15, max_num_leaves)
        num_leaves = trial.suggest_int('num_leaves', min_num_leaves, max_num_leaves)
        
        params = {
            'objective': 'mae',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            # 限制learning_rate上限到0.05，避免过高的学习率导致不稳定
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),  # 提高下限
            # 提高min_split_gain下限到0.01，避免噪声分裂
            'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 0.3),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.6, 1.0),
            # 增加正则化强度：L1下限提高
            'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 2.0),
            # 增加正则化强度：L2下限提高
            'lambda_l2': trial.suggest_float('lambda_l2', 0.5, 3.0),
            'num_threads': -1,
            'seed': 42,
            'verbose': -1,
        }
        
        # 训练模型并获取IC
        try:
            model, val_pred, ic_mean = train_enhanced_lightgbm_model(
                X_train, y_train, X_val, y_val,
                symbols_train, symbols_val, dates_train, dates_val,
                params=params, verbose=False
            )
            
            # 计算更稳健的IC指标：使用中位数和均值的加权组合
            # 注意：train_enhanced_lightgbm_model内部已经对y_val进行了clipping处理
            # 我们需要重新计算clipped的y_val来匹配预测值
            def smart_clip_values(y, clip_percentile=98):
                Q1 = np.percentile(y, 25)
                Q3 = np.percentile(y, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                lower_bound = max(lower_bound, np.percentile(y, 100 - clip_percentile))
                upper_bound = min(upper_bound, np.percentile(y, clip_percentile))
                return np.clip(y, lower_bound, upper_bound)
            
            y_val_clipped = smart_clip_values(y_val, clip_percentile=98)
            val_df = pd.DataFrame({
                'date': dates_val,
                'pred': val_pred,
                'actual': y_val_clipped
            })
            daily_corrs = []
            for date, group in val_df.groupby('date'):
                if len(group) >= 5:
                    corr = group['pred'].corr(group['actual'])
                    if not np.isnan(corr):
                        daily_corrs.append(corr)
            
            if len(daily_corrs) < 5:
                return -1.0  # 有效日期太少，返回差值
            
            # 使用中位数和均值的加权组合（中位数更稳健，均值更敏感）
            ic_median = np.median(daily_corrs)
            ic_mean_calc = np.mean(daily_corrs)
            # 加权组合：70%中位数 + 30%均值（更重视稳健性）
            robust_ic = 0.7 * ic_median + 0.3 * ic_mean_calc
            
            return robust_ic
        except Exception as e:
            print(f"试验失败: {e}")
            return -1.0  # 返回一个很差的IC值
    
    # 创建study并优化
    study = optuna.create_study(
        direction='maximize',  # 最大化IC
        study_name='lightgbm_ic_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)  # 使用TPE采样器
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 输出最佳结果
    print(f"\n✅ 贝叶斯优化完成！")
    print(f"   最佳稳健IC: {study.best_value:.4f} (70%中位数 + 30%均值)")
    print(f"   最佳参数:")
    for key, value in study.best_params.items():
        print(f"     {key}: {value}")
    
    return study.best_params, study.best_value

def create_enhanced_lightgbm_prediction_factor(window=20, rebalance_period=5, train_epochs=3000,
                                            min_history_days=60, min_samples_per_symbol=30,
                                            use_bayesian_optimization=False, n_trials=30):
    """
    基于增强版LightGBM的收益率预测因子
    
    Args:
        window: 窗口大小
        rebalance_period: 调仓周期
        train_epochs: 训练轮数（已弃用，由早停控制）
        min_history_days: 最小历史天数
        min_samples_per_symbol: 每个symbol的最小样本数（已弃用）
        use_bayesian_optimization: 是否使用贝叶斯优化寻找最佳参数
        n_trials: 贝叶斯优化的试验次数（仅在use_bayesian_optimization=True时有效）
    """
    print(f"开始构建增强版LightGBM预测因子")
    print(f"参数: window={window}, rebalance_period={rebalance_period}")
    print(f"      min_history_days={min_history_days}, min_samples_per_symbol={min_samples_per_symbol}")
    if use_bayesian_optimization:
        print(f"      🔍 使用贝叶斯优化，试验次数={n_trials}")
    
    if not LGB_AVAILABLE:
        print("LightGBM不可用，无法进行训练")
        return pd.DataFrame()
    
    # K线数据
    df = load_kline_df()
    
    # 准备增强版训练数据
    all_features, all_targets, all_symbols, all_dates = prepare_lightgbm_data_fast(
        df, rebalance_period, min_history_days, n_jobs=40   # 根据CPU核心数设置
    )

    # 路径准备
    model_path, selector_path, scaler_path = _artifact_paths(window, rebalance_period)

    use_cached = model_path.exists() and selector_path.exists() and scaler_path.exists()
    if use_cached:
        print(f"📦 检测到已训练模型与预处理器，直接加载: {model_path.name}")
    else:
        print("🧪 未发现已训练模型，将进行训练并保存。")
    
    if len(all_features) < 1000:
        print("警告: 训练数据不足，需要至少1000个样本")
        return pd.DataFrame()
    
    # 严格的时间序列分割
    data_with_dates = list(zip(all_features, all_targets, all_symbols, all_dates))
    train_data, val_data, test_data = create_strict_time_split(
        data_with_dates, test_days=180, validation_days=90
    )

    # 分离数据
    X_train = np.array([item[0] for item in train_data])
    y_train = np.array([item[1] for item in train_data])
    symbols_train = [item[2] for item in train_data]
    dates_train = [item[3] for item in train_data]

    X_val = np.array([item[0] for item in val_data])
    y_val = np.array([item[1] for item in val_data])
    symbols_val = [item[2] for item in val_data]
    dates_val = [item[3] for item in val_data]

    X_test = np.array([item[0] for item in test_data])
    y_test = np.array([item[1] for item in test_data])
    symbols_test = [item[2] for item in test_data]
    dates_test = [item[3] for item in test_data]
    
    # 特征选择
    print("🔄 进行特征选择...")
    if use_cached:
        selector = joblib.load(selector_path)
        X_train_selected = selector.transform(X_train)
        X_val_selected   = selector.transform(X_val)
        X_test_selected  = selector.transform(X_test)
    else:
        selector = SelectKBest(score_func=mutual_info_regression, k=min(100, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected   = selector.transform(X_val)
        X_test_selected  = selector.transform(X_test)
        joblib.dump(selector, selector_path)
    print(f"特征选择: {X_train.shape[1]} -> {X_train_selected.shape[1]}")

    # 标准化
    print("�� 标准化特征（使用RobustScaler）...")
    if use_cached:
        scaler = joblib.load(scaler_path)
        X_train_scaled = scaler.transform(X_train_selected)
        X_val_scaled   = scaler.transform(X_val_selected)
        X_test_scaled  = scaler.transform(X_test_selected)
    else:
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled   = scaler.transform(X_val_selected)
        X_test_scaled  = scaler.transform(X_test_selected)
        joblib.dump(scaler, scaler_path)

    # 模型：加载或训练
    if use_cached:
        model = lgb.Booster(model_file=str(model_path))
        print("✅ 已加载缓存模型与预处理器")
    else:
        # 贝叶斯优化或使用默认参数
        if use_bayesian_optimization and OPTUNA_AVAILABLE:
            best_params, best_ic = bayesian_optimize_lightgbm(
                X_train_scaled, y_train, X_val_scaled, y_val,
                symbols_train, symbols_val, dates_train, dates_val,
                n_trials=n_trials
            )
            if best_params is not None:
                print(f"\n🎯 使用优化后的参数重新训练最终模型...")
                model, val_pred, _ = train_enhanced_lightgbm_model(
                    X_train_scaled, y_train, X_val_scaled, y_val,
                    symbols_train, symbols_val, dates_train, dates_val,
                    params=best_params, verbose=True
                )
            else:
                print("⚠️ 贝叶斯优化失败，使用默认参数")
                model, val_pred, _ = train_enhanced_lightgbm_model(
                    X_train_scaled, y_train, X_val_scaled, y_val,
                    symbols_train, symbols_val, dates_train, dates_val
                )
        else:
            if use_bayesian_optimization and not OPTUNA_AVAILABLE:
                print("⚠️ Optuna未安装，使用默认参数训练")
            model, val_pred, _ = train_enhanced_lightgbm_model(
                X_train_scaled, y_train, X_val_scaled, y_val,
                symbols_train, symbols_val, dates_train, dates_val
            )
        # 保存模型（包含best_iteration信息）
        model.save_model(str(model_path))
        print(f"✅ 模型已保存到: {model_path}")
    
    # 样本外预测和评估
    print("🔮 开始样本外预测...")
    test_pred = model.predict(X_test_scaled, num_iteration=model.best_iteration)
    
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
    
    y_test_clipped = smart_clip_values(y_test, clip_percentile=98)
    
    # 样本外指标
    test_rmse = np.sqrt(mean_squared_error(y_test_clipped, test_pred))
    test_mae = mean_absolute_error(y_test_clipped, test_pred)
    test_r2 = r2_score(y_test_clipped, test_pred)
    
    # 计算横截面相关性（按日期）
    test_df = pd.DataFrame({
        'date': dates_test,
        'pred': test_pred,
        'actual': y_test_clipped
    })
    daily_corrs = []
    for date, group in test_df.groupby('date'):
        if len(group) >= 5:
            corr = group['pred'].corr(group['actual'])
            if not np.isnan(corr):
                daily_corrs.append(corr)
    
    avg_ic = np.mean(daily_corrs) if daily_corrs else 0
    print(f"[样本外-Test]  RMSE={test_rmse:.6f}, MAE={test_mae:.6f}, R2={test_r2:.4f}")
    print(f"[样本外-Test]  平均IC={avg_ic:.4f}, 有效日期数={len(daily_corrs)}")

    # 统一生成因子数据
    result_data = []
    sets = [
        ("train", X_train_scaled, symbols_train, dates_train, y_train),
        ("val",   X_val_scaled,   symbols_val,   dates_val,   y_val),
        ("test",  X_test_scaled,  symbols_test,  dates_test,  y_test),
    ]
    
    for split, Xs, syms, dts, ys in sets:
        if len(Xs) == 0:
            continue
        preds = model.predict(Xs, num_iteration=model.best_iteration)
        for pred, s, d, y_true in zip(preds, syms, dts, ys):
            result_data.append({
                'date': d,
                'symbol': s,
                'lgb_prediction': float(pred),
                'future_ret': float(y_true),
                'split': split
            })

    if len(result_data) == 0:
        print("警告: 无可用预测样本，返回空 DataFrame")
        return pd.DataFrame()

    # 构造DataFrame
    factor_df = pd.DataFrame(result_data)

    # 直接使用预测收益率作为因子，按日做rank归一化
    factor_df = rank_to_unit_by_date(factor_df, col="lgb_prediction", out_col="factor")

    # 重命名并保存
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret", "split"]]
    out_path = save_factor_df(factor_df, file_prefix=f"lightgbm_enhanced_{window}d_rebalance{rebalance_period}d_")
    print_factor_summary(factor_df, out_path)

    # 新增：如果因“目标位移”导致factor_df不是最新自然日，则进行一次Live推理并打印真正最新日
    print_latest_daily_groups_live(window=window, rebalance_period=rebalance_period, min_history_days=min_history_days, groups=5)
    return factor_df

def create_features_symbol_inference_latest(gp: pd.DataFrame, min_history_days: int = 60) -> pd.DataFrame:
    """
    仅用于推理：构造与训练一致的特征，但不生成future_ret，也不丢弃末尾。
    返回该symbol在其各自最新日期的一行特征。
    """
    eps = 1e-8
    gp = gp.sort_values('date').reset_index(drop=True).copy()
    if len(gp) < min_history_days + 20:
        return pd.DataFrame()

    close = gp['close']
    high  = gp['high']
    low   = gp['low']
    vol   = gp['volume']
    qv    = gp['quote_volume'] if 'quote_volume' in gp.columns else None
    tbq   = gp['taker_buy_quote'] if 'taker_buy_quote' in gp.columns else None

    feat = pd.DataFrame({'date': gp['date'], 'symbol': gp['symbol']})
    feat['open']   = gp['open']
    feat['high']   = high
    feat['low']    = low
    feat['close']  = close
    feat['volume'] = vol

    if qv is not None:
        feat['quote_volume'] = qv
        feat['vwap'] = qv / (vol + eps)
    else:
        feat['vwap'] = close

    for n in (1,3,5,10,20,30):
        feat[f'ret_{n}'] = close.pct_change(n)
        feat[f'logret_{n}'] = np.log(close / close.shift(n))
        feat[f'volume_ret_{n}'] = vol.pct_change(n)

    for n in (5,10,20,30,60):
        sma = close.rolling(n).mean()
        ema = close.ewm(span=n, adjust=False).mean()
        feat[f'sma_{n}'] = sma
        feat[f'ema_{n}'] = ema
        feat[f'price_sma_ratio_{n}'] = close / (sma + eps)
        feat[f'price_ema_ratio_{n}'] = close / (ema + eps)

    logret1 = np.log(close / close.shift(1))
    for n in (5,10,20,30):
        voln = logret1.rolling(n).std()
        feat[f'volatility_{n}'] = voln
        feat[f'high_low_ratio_{n}'] = (high - low).rolling(n).mean() / (close + eps)
        rmin = low.rolling(n).min(); rmax = high.rolling(n).max()
        feat[f'close_position_{n}'] = (close - rmin) / (rmax - rmin + eps)

    for n in (5,10,20,30):
        vma = vol.rolling(n).mean()
        feat[f'volume_sma_{n}'] = vma
        feat[f'volume_ratio_{n}'] = vol / (vma + eps)
        if qv is not None:
            qvma = qv.rolling(n).mean()
            feat[f'quote_volume_sma_{n}'] = qvma
            feat[f'quote_volume_ratio_{n}'] = qv / (qvma + eps)

    for period in (14,21,30):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + eps)
        feat[f'rsi_{period}'] = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat['macd'] = macd
    feat['macd_signal'] = signal
    feat['macd_histogram'] = macd - signal

    for period in (20,30,60):
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = sma + 2*std; lower = sma - 2*std
        feat[f'bb_upper_{period}'] = upper
        feat[f'bb_lower_{period}'] = lower
        feat[f'bb_width_{period}'] = (upper - lower) / (sma + eps)
        feat[f'bb_position_{period}'] = (close - lower) / (upper - lower + eps)

    if tbq is not None:
        feat['taker_buy_quote'] = tbq
        if qv is not None:
            buy_ratio = tbq / (qv + eps)
            feat['buy_ratio'] = buy_ratio
            for n in (5,10,20,30):
                feat[f"buy_ratio_sma_{n}"] = buy_ratio.rolling(n).mean()
                feat[f"buy_ratio_std_{n}"] = buy_ratio.rolling(n).std()
                feat[f"buy_ratio_price_corr_{n}"] = close.pct_change().rolling(n).corr(buy_ratio)
        for n in (5,10,20,30):
            feat[f"buyq_sma_{n}"] = tbq.rolling(n).mean()
            feat[f"buyq_std_{n}"] = tbq.rolling(n).std()
            feat[f"buyq_roc_{n}"] = tbq.pct_change(n)

    feat["alpha41_geo_vwap"] = np.sqrt(high * low) - feat["vwap"]
    vwap10 = (qv / (vol + eps)).rolling(10).mean() if qv is not None else close.rolling(10).mean()
    feat["alpha5_open_minus_vwap10"] = gp["open"] - vwap10
    feat["alpha5_neg_abs_close_minus_vwap"] = -(close - feat["vwap"]).abs()

    feat = feat.iloc[min_history_days:].dropna().reset_index(drop=True)
    if feat.empty:
        return pd.DataFrame()
    return feat.iloc[[-1]]  # 仅取各自最新一行

def print_latest_daily_groups_live(window: int, rebalance_period: int, min_history_days: int = 60, groups: int = 5):
    """
    使用已训练(可能是10日目标)的模型，对原始K线的“全局最新日期”做一次前向预测并分组打印。
    """
    # 读取数据与工件
    df = load_kline_df()
    model_path, selector_path, scaler_path = _artifact_paths(window, rebalance_period)
    if not (model_path.exists() and selector_path.exists() and scaler_path.exists()):
        print("未找到已训练模型/预处理器，跳过live日度分组打印。")
        return

    model = lgb.Booster(model_file=str(model_path))
    selector = joblib.load(selector_path)
    scaler = joblib.load(scaler_path)

    # 为每个symbol构造"最新一行"特征
    feats = []
    for sym, gp in df.groupby('symbol'):
        row = create_features_symbol_inference_latest(gp.copy(), min_history_days=min_history_days)
        if row is not None and not row.empty:
            feats.append(row)
    if not feats:
        print("无法构造最新日度特征，跳过live分组打印。")
        return

    feat_df = pd.concat(feats, ignore_index=True)
    latest_date = pd.to_datetime(feat_df['date']).max()
    day = feat_df[pd.to_datetime(feat_df['date']) == latest_date].copy()
    if day.empty:
        print("最新日期无可推理样本。")
        return

    X = day.drop(columns=['date','symbol']).to_numpy(dtype=np.float32)
    X_sel = selector.transform(X)
    X_scl = scaler.transform(X_sel)
    preds = model.predict(X_scl, num_iteration=model.best_iteration)

    out = pd.DataFrame({'instrument': day['symbol'].values, 'pred': preds})
    # 横截面秩值归一化到[-1,1]，并按百分位分组，展示更直观
    out['pct'] = out['pred'].rank(pct=True, method='average')
    out['score'] = 2 * out['pct'] - 1  # [-1, 1]

    try:
        out['group'] = pd.qcut(out['pct'], q=groups, labels=list(range(groups)))
    except Exception:
        bins = np.linspace(out['pct'].min() - 1e-9, out['pct'].max() + 1e-9, groups + 1)
        out['group'] = pd.cut(out['pct'], bins=bins, labels=list(range(groups)), include_lowest=True)
    out['group'] = out['group'].astype(int)

    print(f"\n==== 1日调仓视角 | 最新日期: {latest_date.date()} (Live 推理) ====")

    # CSV表头
    print("date,group,instrument,score,pred")

    for g in range(groups):
        sub = out[out['group'] == g].sort_values('score')
        for _, r in sub.iterrows():
            print(f"{latest_date.date()},{g},{r['instrument']},{r['score']:.4f},{r['pred']:.6f}")

if __name__ == "__main__":
    # 使用增强版参数调用
    # 设置 use_bayesian_optimization=True 来启用贝叶斯优化
    create_enhanced_lightgbm_prediction_factor(
        window=30, 
        rebalance_period=10, 
        train_epochs=3000,
        min_history_days=60,
        min_samples_per_symbol=30,
        use_bayesian_optimization=True,  # 设置为True启用贝叶斯优化
        n_trials=30  # 贝叶斯优化的试验次数（建议30-50次）
    )