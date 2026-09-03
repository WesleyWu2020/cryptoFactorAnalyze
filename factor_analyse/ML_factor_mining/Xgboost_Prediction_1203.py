# factor_analyse/ML_Factor/XGBoost_Prediction_Factor_Enhanced.py
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
    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    from sklearn.decomposition import PCA
    XGB_AVAILABLE = True
    print("✅ XGBoost可用")
except ImportError:
    print("警告: XGBoost未安装，请安装: pip install xgboost")
    XGB_AVAILABLE = False

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
    load_kline_df,
    build_available_tokens_by_date_from_kline,
    filter_group_by_availability,
    rank_to_unit_by_date, save_factor_df,
    print_factor_summary
)

from scipy import stats
import os
from pathlib import Path
import joblib

class AdaptiveSelector:
    """自适应特征选择器，支持pickle序列化"""
    def __init__(self, pre_selector, selected_pre_indices):
        self.pre_selector = pre_selector
        self.selected_pre_indices = selected_pre_indices

    def fit(self, X, y):
        return self

    def transform(self, X):
        X_pre = self.pre_selector.transform(X)
        return X_pre[:, self.selected_pre_indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

class CombinedSelector:
    """组合选择器：先应用正交化，再应用互信息选择，支持pickle序列化"""
    def __init__(self, orthogonalizer, pre_selector, selected_pre_indices):
        self.orthogonalizer = orthogonalizer
        self.pre_selector = pre_selector
        self.selected_pre_indices = selected_pre_indices
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        X_ortho = self.orthogonalizer.transform(X)
        if self.pre_selector is not None:
            X_pre = self.pre_selector.transform(X_ortho)
            return X_pre[:, self.selected_pre_indices]
        else:
            return X_ortho[:, self.selected_pre_indices]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

class ConservativeSelector:
    """保守特征选择器：先正交化，再基于重要性选择，支持pickle序列化"""
    def __init__(self, orthogonalizer, selected_indices):
        self.orthogonalizer = orthogonalizer
        self.selected_indices = selected_indices
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        X_ortho = self.orthogonalizer.transform(X)
        return X_ortho[:, self.selected_indices]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

class FeatureOrthogonalizer:
    """
    特征正交化处理器：去除高度相关的特征并进行正交化
    支持两种模式：
    1. 相关性过滤：去除高度相关的冗余特征
    2. PCA正交化：将特征转换为正交主成分（可选）
    """
    def __init__(self, correlation_threshold=0.95, use_pca=False, pca_components=None, pca_variance_ratio=0.95):
        """
        Parameters:
        - correlation_threshold: 相关性阈值，超过此值的特征对将被去除其中一个
        - use_pca: 是否使用PCA进行正交化
        - pca_components: PCA主成分数量（None表示自动选择）
        - pca_variance_ratio: PCA保留的方差比例（当pca_components为None时使用）
        """
        self.correlation_threshold = correlation_threshold
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca_variance_ratio = pca_variance_ratio
        self.selected_features_ = None
        self.pca_ = None
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        """
        拟合正交化器
        
        Parameters:
        - X: 特征矩阵 (n_samples, n_features)
        - y: 目标变量（可选，用于基于重要性的特征选择）
        """
        X = np.array(X)
        n_features = X.shape[1]
        
        # 1. 相关性过滤：去除高度相关的特征
        print(f"🔄 特征相关性分析（阈值={self.correlation_threshold}）...")
        
        # 计算相关性矩阵（使用pandas DataFrame更高效）
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().values
        np.fill_diagonal(corr_matrix, 0)  # 对角线设为0
        
        # 如果提供了y，计算每个特征与y的相关性
        if y is not None:
            y_corrs = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(n_features)])
            y_corrs = np.nan_to_num(y_corrs, nan=0)
        else:
            # 如果没有y，使用特征的方差作为重要性指标
            y_corrs = np.var(X, axis=0)
            y_corrs = np.nan_to_num(y_corrs, nan=0)
        
        # 找出高度相关的特征对，并选择要移除的特征
        to_remove = set()
        for i in range(n_features):
            if i in to_remove:
                continue
            for j in range(i+1, n_features):
                if j in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    # 优先保留与y相关性更高（或方差更大）的特征
                    if y_corrs[i] >= y_corrs[j]:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
        
        # 保留的特征索引
        self.selected_features_ = np.array([i for i in range(n_features) if i not in to_remove])
        
        removed_count = len(to_remove)
        print(f"   去除 {removed_count} 个高度相关特征，保留 {len(self.selected_features_)} 个特征")
        
        # 应用相关性过滤
        X_filtered = X[:, self.selected_features_]
        
        # 2. PCA正交化（可选）
        if self.use_pca:
            print(f"🔄 PCA正交化处理...")
            self.pca_ = PCA(n_components=self.pca_components, 
                           svd_solver='full' if self.pca_components is None else 'auto')
            self.pca_.fit(X_filtered)
            
            # 如果未指定主成分数量，根据方差比例自动选择
            if self.pca_components is None:
                cumsum_variance = np.cumsum(self.pca_.explained_variance_ratio_)
                n_components = np.argmax(cumsum_variance >= self.pca_variance_ratio) + 1
                n_components = min(n_components, X_filtered.shape[1])
                self.pca_ = PCA(n_components=n_components, svd_solver='full')
                self.pca_.fit(X_filtered)
            
            explained_variance = np.sum(self.pca_.explained_variance_ratio_)
            print(f"   PCA主成分数: {self.pca_.n_components_}, 解释方差: {explained_variance:.4f}")
        else:
            self.pca_ = None
        
        return self
    
    def transform(self, X):
        """转换特征矩阵"""
        X = np.array(X)
        
        # 1. 应用相关性过滤
        X_filtered = X[:, self.selected_features_]
        
        # 2. 应用PCA正交化（如果启用）
        if self.pca_ is not None:
            X_transformed = self.pca_.transform(X_filtered)
        else:
            X_transformed = X_filtered
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """拟合并转换"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_importance_mapping(self):
        """
        获取PCA主成分到原始特征的映射（如果使用PCA）
        返回主成分的载荷矩阵
        """
        if self.pca_ is not None:
            # 载荷矩阵：每个主成分对原始特征的贡献
            return self.pca_.components_.T  # (n_features, n_components)
        else:
            return np.eye(len(self.selected_features_))  # 单位矩阵（无变换）

def bayesian_optimize_xgboost(X_train, y_train, X_val, y_val, dates_val=None, symbols_val=None,
                              n_calls=50, n_random_starts=10,
                              ic_ir_weight=0.1, rankIC=1.8, ir_weight=0.0, monotonicity_weight=0.1):
    """
    使用贝叶斯优化寻找最优的XGBoost参数
    目标函数：最大化 IC_IR、IC、IR、分组单调性 的加权和

    Parameters:
    - X_train, y_train: 训练数据
    - X_val, y_val: 验证数据
    - dates_val: 验证集日期列表（用于计算按日期的IC）
    - symbols_val: 验证集符号列表（用于计算按日期的IC）
    - n_calls: 总的优化迭代次数
    - n_random_starts: 随机初始化的次数
    - ic_ir_weight: IC_IR的权重（默认0.4）
    - rankIC: Rank IC的权重（默认0.3）
    - ir_weight: IR的权重（默认0.2）
    - monotonicity_weight: 分组单调性的权重（默认0.2）

    Returns:
    - best_params: 最优参数字典
    - best_score: 最优验证得分（加权IC指标）
    """
    if not SKOPT_AVAILABLE:
        print("⚠️ Scikit-optimize不可用，使用默认参数")
        return get_default_xgb_params(), None

    print("🚀 开始贝叶斯优化XGBoost参数（目标：最大化IC_IR、IC、IR、分组单调性加权和）...")
    print(f"   优化迭代次数: {n_calls}, 随机启动: {n_random_starts}")
    print(f"   权重设置: IC_IR={ic_ir_weight}, RankIC={rankIC}, IR={ir_weight}, 单调性={monotonicity_weight}")

    # 检查是否有日期和符号信息用于计算IC
    use_ic_objective = dates_val is not None and symbols_val is not None and len(dates_val) == len(y_val)
    if not use_ic_objective:
        print("⚠️ 警告: 缺少dates_val或symbols_val，将使用MAE作为目标函数")
        use_ic_objective = False

    # 定义参数搜索空间
    param_space = [
        # 学习率
        Real(0.05, 0.3, name='learning_rate'),

        # 树的最大深度
        Integer(4, 8, name='max_depth'),

        # 最小叶子节点样本数
        Integer(5, 50, name='min_child_weight'),

        # gamma参数，用于控制是否后剪枝
        Real(0.0, 2.0, name='gamma'),

        # 训练样本的子采样比例
        Real(0.5, 1.0, name='subsample'),

        # 每棵树列的子采样比例
        Real(0.5, 1.0, name='colsample_bytree'),

        # 每级树列的子采样比例
        Real(0.5, 1.0, name='colsample_bynode'),

        # L1正则化参数（扩大搜索范围）
        Real(5.0, 30.0, name='reg_alpha'),   # 从(2.0, 10.0)扩大到(5.0, 30.0)

        # L2正则化参数（扩大搜索范围）
        Real(10.0, 50.0, name='reg_lambda'), # 从(5.0, 20.0)扩大到(10.0, 50.0)

        # 树的数量 (固定为较大的值，由early stopping控制)
        Integer(3000, 5000, name='n_estimators'),
    ]

    # 全局变量用于存储最佳分数
    global_best_score = [-float('inf')]  # 改为负无穷，因为我们要最大化

    @use_named_args(param_space)
    def objective(**params):
        """目标函数：最大化IC_IR、IC、IR的加权和（返回负值以便gp_minimize最小化）"""
        try:
            # 提取参数
            learning_rate = params['learning_rate']
            max_depth = params['max_depth']
            min_child_weight = params['min_child_weight']
            gamma = params['gamma']
            subsample = params['subsample']
            colsample_bytree = params['colsample_bytree']
            colsample_bynode = params['colsample_bynode']
            reg_alpha = params['reg_alpha']
            reg_lambda = params['reg_lambda']
            n_estimators = params['n_estimators']

            # 设置XGBoost参数
            xgb_params = {
                'objective': 'reg:absoluteerror',
                'eval_metric': 'mae',
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_child_weight': min_child_weight,
                'gamma': gamma,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'colsample_bynode': colsample_bynode,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'tree_method': 'hist',
                'random_state': 42,
                'n_jobs': -1,
            }

            # 创建DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            # 训练模型（带early stopping）
            evals_result = {}
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=100,
                verbose_eval=False,
                evals_result=evals_result
            )

            # 获取验证集预测
            val_pred = model.predict(dval)

            if use_ic_objective:
                # 计算IC相关指标
                val_df = pd.DataFrame({
                    'date': dates_val,
                    'symbol': symbols_val,
                    'pred': val_pred,
                    'actual': y_val
                })
                
                # 按日期分组计算每日IC
                daily_ics = []
                for date, group in val_df.groupby('date'):
                    if len(group) >= 5:  # 至少5个样本才计算IC
                        ic = group['pred'].corr(group['actual'])
                        if not np.isnan(ic):
                            daily_ics.append(ic)
                
                if len(daily_ics) > 0:
                    # 计算IC统计量
                    ic_mean = np.mean(daily_ics)
                    ic_std = np.std(daily_ics)
                    
                    # IC_IR = IC均值 / IC标准差
                    ic_ir = ic_mean / ic_std if ic_std > 1e-8 else ic_mean
                    
                    # 计算分组单调性：将预测值分为5组，计算每组平均收益，然后计算斯皮尔曼相关系数
                    val_df_sorted = val_df.sort_values('pred', ascending=False)
                    n_groups = 5
                    group_size = len(val_df_sorted) // n_groups

                    group_returns = []
                    for i in range(n_groups):
                        start_idx = i * group_size
                        end_idx = (i + 1) * group_size if i < n_groups - 1 else len(val_df_sorted)
                        group_data = val_df_sorted.iloc[start_idx:end_idx]
                        group_return = group_data['actual'].mean()
                        group_returns.append(group_return)

                    # 计算分组单调性（斯皮尔曼相关系数：组序号 vs 组收益）
                    from scipy import stats as scipy_stats
                    group_ranks = list(range(1, n_groups + 1))  # [1, 2, 3, 4, 5]
                    if len(group_returns) == n_groups:
                        monotonicity_corr, _ = scipy_stats.spearmanr(group_ranks, group_returns)
                        monotonicity = abs(monotonicity_corr)  # 取绝对值，确保为正
                    else:
                        monotonicity = 0

                    # IR = 年化收益率 / 年化波动率
                    # 简化计算：使用多空组合收益（前20% - 后20%）
                    n_top = max(1, int(len(val_df_sorted) * 0.2))
                    n_bottom = max(1, int(len(val_df_sorted) * 0.2))

                    top_returns = val_df_sorted.head(n_top)['actual'].mean()
                    bottom_returns = val_df_sorted.tail(n_bottom)['actual'].mean()
                    long_short_return = top_returns - bottom_returns

                    # 计算多空组合的波动率（按日期分组）
                    daily_ls_returns = []
                    for date, group in val_df.groupby('date'):
                        if len(group) >= 10:  # 至少10个样本才计算
                            group_sorted = group.sort_values('pred', ascending=False)
                            n_top_daily = max(1, int(len(group_sorted) * 0.2))
                            n_bottom_daily = max(1, int(len(group_sorted) * 0.2))
                            top_ret = group_sorted.head(n_top_daily)['actual'].mean()
                            bottom_ret = group_sorted.tail(n_bottom_daily)['actual'].mean()
                            daily_ls_returns.append(top_ret - bottom_ret)

                    if len(daily_ls_returns) > 1:
                        ls_return_mean = np.mean(daily_ls_returns)
                        ls_return_std = np.std(daily_ls_returns)
                        # 年化IR（假设252个交易日）
                        ir = (ls_return_mean / ls_return_std) * np.sqrt(252) if ls_return_std > 1e-8 else 0
                    else:
                        ir = 0

                    # 加权和（使用绝对值确保为正）
                    weighted_score = (ic_ir_weight * abs(ic_ir) +
                                     rankIC * abs(ic_mean) +
                                     ir_weight * abs(ir) +
                                     monotonicity_weight * monotonicity)
                    
                    # 更新全局最佳分数（取负值，因为gp_minimize是最小化）
                    if weighted_score > global_best_score[0]:
                        global_best_score[0] = weighted_score
                    
                    # 返回负值，因为gp_minimize是最小化，我们要最大化weighted_score
                    return -weighted_score
                else:
                    # 没有有效的IC数据，返回一个很大的正值（表示很差）
                    return 1e6
            else:
                # 回退到MAE
                val_mae = evals_result['val']['mae'][-1]
                if val_mae < abs(global_best_score[0]):
                    global_best_score[0] = -val_mae  # 存储负值以保持一致性
            return val_mae

        except Exception as e:
            print(f"参数组合训练失败: {e}")
            return float('inf')  # 返回无穷大表示失败

    # 执行贝叶斯优化
    try:
        result = gp_minimize(
            objective,
            param_space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            random_state=42,
            verbose=True
        )

        # 提取最优参数
        best_params = {
            'learning_rate': result.x[0],
            'max_depth': result.x[1],
            'min_child_weight': result.x[2],
            'gamma': result.x[3],
            'subsample': result.x[4],
            'colsample_bytree': result.x[5],
            'colsample_bynode': result.x[6],
            'reg_alpha': result.x[7],
            'reg_lambda': result.x[8],
            'n_estimators': result.x[9],
        }

        # 添加固定参数
        best_params.update({
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'early_stopping_rounds': 300,
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
        })

        print("✅ 贝叶斯优化完成!")
        if use_ic_objective:
            # result.fun是负值（因为我们返回的是-weighted_score），需要取绝对值
            best_score = abs(result.fun)
            print(f"   最优加权IC指标: {best_score:.6f}")
            print(f"   (IC_IR权重={ic_ir_weight}, RankIC权重={rankIC}, IR权重={ir_weight}, 单调性权重={monotonicity_weight})")
        else:
            best_score = result.fun
            print(f"   最优验证MAE: {best_score:.6f}")
        print(f"   最优参数: learning_rate={best_params['learning_rate']:.4f}, "
              f"max_depth={best_params['max_depth']}, "
              f"min_child_weight={best_params['min_child_weight']}")

        return best_params, best_score

    except Exception as e:
        print(f"❌ 贝叶斯优化失败: {e}")
        print("   使用默认参数")
        return get_default_xgb_params(), None

def get_default_xgb_params():
    """获取默认的XGBoost参数（强正则化版本）"""
    return {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'learning_rate': 0.01,      # 从0.03进一步降低到0.01，增强正则化
        'max_depth': 3,             # 从4降低到3，减少过拟合
        'min_child_weight': 25,     # 从12增加到25，增强正则化
        'gamma': 2.0,              # 从0.8增加到2.0，增强后剪枝
        'subsample': 0.6,          # 从0.7降低到0.6，增强正则化
        'colsample_bytree': 0.4,   # 从0.5降低到0.4，增强正则化
        'colsample_bynode': 0.6,   # 从0.8降低到0.6，增强正则化
        'reg_alpha': 8.0,          # 从3.0增加到8.0，增强L1正则化
        'reg_lambda': 20.0,        # 从10.0增加到20.0，增强L2正则化
        'n_estimators': 2500,      # 从3000降低到2500，减少过拟合
        'early_stopping_rounds': 200,  # 从300降低到200，更严格的早停
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
    }

def get_enhanced_xgb_params():
    """增强的正则化参数（超强正则化版本）"""
    return {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'learning_rate': 0.008,     # 从0.015进一步降低到0.008，极强正则化
        'max_depth': 2,             # 从3降低到2，极度限制树深度
        'min_child_weight': 35,     # 从20增加到35，极强最小叶子权重
        'gamma': 4.0,              # 从2.0增加到4.0，极强后剪枝
        'subsample': 0.4,          # 从0.5降低到0.4，极低子采样比例
        'colsample_bytree': 0.25,  # 从0.3降低到0.25，极低特征采样比例
        'colsample_bynode': 0.35,  # 从0.5降低到0.35，极低节点特征采样
        'reg_alpha': 12.0,         # 从5.0增加到12.0，极强L1正则化
        'reg_lambda': 30.0,        # 从15.0增加到30.0，极强L2正则化
        'n_estimators': 2000,      # 从3000降低到2000，减少迭代次数
        'early_stopping_rounds': 100,  # 从150降低到100，极严格早停
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
    }

def get_ultra_regularized_xgb_params():
    """超强正则化参数（最大限度防止过拟合）"""
    return {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'learning_rate': 0.005,     # 极低学习率，最大限度正则化
        'max_depth': 2,             # 极浅的树
        'min_child_weight': 50,     # 极高的最小叶子权重
        'gamma': 8.0,              # 极强的后剪枝
        'subsample': 0.3,          # 极低的子采样比例
        'colsample_bytree': 0.15,  # 极低的特征采样比例
        'colsample_bynode': 0.2,   # 极低的节点特征采样
        'reg_alpha': 20.0,         # 极强的L1正则化
        'reg_lambda': 50.0,        # 极强的L2正则化
        'n_estimators': 1500,      # 极少的迭代次数
        'early_stopping_rounds': 50,   # 极严格的早停
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
    }

def _artifact_paths(window: int, rebalance_period: int):
    base_dir = Path(os.path.dirname(__file__)) / "models"
    base_dir.mkdir(parents=True, exist_ok=True)
    # tag = f"xgb_enhanced_w{window}_r5"
    tag = f"xgb_enhanced_w{window}_r{rebalance_period}"
    # tag = f"xgb_w{window}_r{rebalance_period}"
    model_path = base_dir / f"{tag}.json"
    selector_path = base_dir / f"{tag}_selector.pkl"
    scaler_path = base_dir / f"{tag}_scaler.pkl"
    return model_path, selector_path, scaler_path

def prepare_robust_data(df, rebalance_period=5, min_history_days=60, n_jobs=-1, 
                        cross_sectional_standardize=True, use_rank_target=True, use_improved_features=False):
    """
    更稳健的数据预处理
    关键改进点：
    1. 对特征进行更严格的异常值处理
    2. 使用中位数和MAD进行稳健标准化
    3. 对目标变量使用更保守的处理方法
    """
    from joblib import Parallel, delayed
    from scipy import stats

    symbols = df['symbol'].unique().tolist()

    def run_one(sym):
        gp = df[df['symbol'] == sym].copy()
        if gp.empty: return pd.DataFrame()
        # 使用统一的特征生成函数
        return create_features_symbol_fast_based_on_feature_txt(gp, rebalance_period, min_history_days, use_improved_features)

    dfs = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(run_one)(s) for s in symbols)
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return np.array([]), np.array([]), np.array([]), [], []

    out = pd.concat(dfs, ignore_index=True)

    # 保存原始目标变量（用于IC计算）
    y_original = out['future_ret'].copy().to_numpy(dtype=np.float32)

    # 横截面标准化：按日期分组，使用中位数和MAD进行稳健标准化
    if cross_sectional_standardize:
        print("🔄 进行稳健横截面标准化（按日期分组，使用中位数和MAD）...")
        feature_cols = [col for col in out.columns if col not in ['date', 'symbol', 'future_ret']]
        
        def robust_standardize_by_date(group):
            """对每个日期的特征进行稳健标准化（使用中位数和MAD）"""
            group = group.copy()
            for col in feature_cols:
                # 使用中位数和MAD进行稳健标准化
                median_val = group[col].median()
                mad_val = (group[col] - median_val).abs().median()
                if mad_val > 1e-8:
                    # MAD标准化：MAD = median(|x - median(x)|)
                    group[col] = (group[col] - median_val) / mad_val
                else:
                    group[col] = 0.0  # 如果MAD为0，设为0
            return group
        
        out = out.groupby('date', group_keys=False).apply(robust_standardize_by_date)
        print(f"   已完成稳健横截面标准化，处理了 {out['date'].nunique()} 个交易日")

    # 目标变量排名转换：按日期分组，使用更保守的处理方法
    if use_rank_target:
        print("🔄 将目标变量转换为横截面排名（按日期分组，保守处理）...")
        def conservative_target_processing(group):
            """对每个日期的目标变量进行保守的排名转换"""
            group = group.copy()
            # 使用更稳健的排名方法
            rank_pct = group['future_ret'].rank(pct=True, method='average')
            # 避免极端值，使用更窄的clip范围（0.01-0.99）
            rank_pct_clipped = rank_pct.clip(0.01, 0.99)
            group['future_ret'] = stats.norm.ppf(rank_pct_clipped)
            return group
        
        out = out.groupby('date', group_keys=False).apply(conservative_target_processing)
        print(f"   已完成保守目标变量排名转换，处理了 {out['date'].nunique()} 个交易日")

    X = out.drop(columns=['date','symbol','future_ret']).to_numpy(dtype=np.float32)
    y = out['future_ret'].to_numpy(dtype=np.float32)
    syms = out['symbol'].tolist()
    dates = pd.to_datetime(out['date']).tolist()

    target_type = "排名" if use_rank_target else "绝对值"
    print(f"✅ 稳健特征 向量化+并行 数据样本: {len(out)}，特征维度: {X.shape[1]}，目标变量类型: {target_type}")
    return X, y, y_original, syms, dates

def prepare_xgboost_data_fast(df, rebalance_period=5, min_history_days=60, n_jobs=-1, 
                              cross_sectional_standardize=True, use_rank_target=True, use_improved_features=False):
    """
    向量化 + 并行 的数据准备。返回与原接口兼容的五元组（增加原始目标变量）。
    使用基于feature.txt的特征定义或改进的特征。
    
    Parameters:
    - cross_sectional_standardize: 是否对特征进行横截面标准化（按日期分组）
    - use_rank_target: 是否将目标变量转换为横截面排名（预测排名而非绝对值）
    - use_improved_features: 是否使用改进的特征工程（减少过拟合风险）
    
    Returns:
    - X: 特征矩阵
    - y: 目标变量（如果use_rank_target=True，则为排名；否则为原始值）
    - y_original: 原始目标变量（用于IC计算）
    - syms: 符号列表
    - dates: 日期列表
    """
    from joblib import Parallel, delayed

    symbols = df['symbol'].unique().tolist()

    def run_one(sym):
        gp = df[df['symbol'] == sym].copy()
        if gp.empty: return pd.DataFrame()
        # 使用统一的特征生成函数
        return create_features_symbol_fast_based_on_feature_txt(gp, rebalance_period, min_history_days, use_improved_features)

    dfs = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(run_one)(s) for s in symbols)
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return np.array([]), np.array([]), np.array([]), [], []

    out = pd.concat(dfs, ignore_index=True)

    # 保存原始目标变量（用于IC计算）
    y_original = out['future_ret'].copy().to_numpy(dtype=np.float32)

    # 横截面标准化：按日期分组，对每个日期的特征进行标准化
    if cross_sectional_standardize:
        print("🔄 进行横截面标准化（按日期分组）...")
        feature_cols = [col for col in out.columns if col not in ['date', 'symbol', 'future_ret']]
        
        def standardize_by_date(group):
            """对每个日期的特征进行横截面标准化"""
            group = group.copy()
            for col in feature_cols:
                mean_val = group[col].mean()
                std_val = group[col].std()
                if std_val > 1e-8:  # 避免除零
                    group[col] = (group[col] - mean_val) / std_val
                else:
                    group[col] = 0.0  # 如果标准差为0，设为0
            return group
        
        out = out.groupby('date', group_keys=False).apply(standardize_by_date)
        print(f"   已完成横截面标准化，处理了 {out['date'].nunique()} 个交易日")

    # 目标变量排名转换：按日期分组，将future_ret转换为横截面排名
    if use_rank_target:
        print("🔄 将目标变量转换为横截面排名（按日期分组）...")
        def rank_target_by_date(group):
            """对每个日期的目标变量进行排名转换"""
            group = group.copy()
            # 使用百分比排名（0-1之间），然后转换为标准正态分布的分位数
            # 这样可以保持相对顺序，同时使分布更接近正态分布
            rank_pct = group['future_ret'].rank(pct=True, method='average')
            # 将百分比排名转换为标准正态分布的分位数（使用scipy.stats.norm.ppf）
            from scipy import stats
            # 避免边界值（0和1）导致无穷大，使用clip
            rank_pct_clipped = rank_pct.clip(0.001, 0.999)
            group['future_ret'] = stats.norm.ppf(rank_pct_clipped)
            return group
        
        out = out.groupby('date', group_keys=False).apply(rank_target_by_date)
        print(f"   已完成目标变量排名转换，处理了 {out['date'].nunique()} 个交易日")

    X = out.drop(columns=['date','symbol','future_ret']).to_numpy(dtype=np.float32)
    y = out['future_ret'].to_numpy(dtype=np.float32)
    syms = out['symbol'].tolist()
    dates = pd.to_datetime(out['date']).tolist()

    target_type = "排名" if use_rank_target else "绝对值"
    print(f"✅ 特征 向量化+并行 数据样本: {len(out)}，特征维度: {X.shape[1]}，目标变量类型: {target_type}")
    return X, y, y_original, syms, dates

def adaptive_feature_selection(X_train, y_train, X_val, X_test,
                             target_features=50, importance_threshold=0.001,
                             correlation_threshold=0.95, use_orthogonalization=True, use_pca=False):
    """
    自适应特征选择：结合多种方法，动态调整选择数量
    新增：特征正交化处理，去除高度相关的特征
    
    Parameters:
    - correlation_threshold: 相关性阈值，超过此值的特征对将被去除其中一个
    - use_orthogonalization: 是否使用正交化处理（相关性过滤）
    - use_pca: 是否使用PCA进行正交化（会进一步降低特征维度）
    """
    print("🔄 自适应特征选择...")
    
    # 0. 特征正交化处理（去除高度相关的特征）
    orthogonalizer = None
    if use_orthogonalization:
        orthogonalizer = FeatureOrthogonalizer(
            correlation_threshold=correlation_threshold,
            use_pca=use_pca,
            pca_components=None,  # 自动选择主成分数量
            pca_variance_ratio=0.95
        )
        X_train_ortho = orthogonalizer.fit_transform(X_train, y_train)
        X_val_ortho = orthogonalizer.transform(X_val)
        X_test_ortho = orthogonalizer.transform(X_test)
        
        print(f"正交化后特征维度: {X_train.shape[1]} -> {X_train_ortho.shape[1]}")
        
        # 使用正交化后的特征继续后续处理
        X_train_for_selection = X_train_ortho
        X_val_for_selection = X_val_ortho
        X_test_for_selection = X_test_ortho
    else:
        X_train_for_selection = X_train
        X_val_for_selection = X_val
        X_test_for_selection = X_test

    # 1. 初步筛选（互信息）
    n_preselect = min(120, X_train_for_selection.shape[1])
    pre_selector = SelectKBest(score_func=mutual_info_regression, k=n_preselect)
    X_train_pre = pre_selector.fit_transform(X_train_for_selection, y_train)

    # 2. 训练临时模型评估特征重要性
    temp_model = xgb.XGBRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        random_state=42, n_jobs=-1
    )
    temp_model.fit(X_train_pre, y_train)

    # 3. 获取原始特征的重要性（通过pre_selector的反向映射）
    feature_importance = temp_model.feature_importances_

    # 4. 根据重要性阈值和目标数量动态选择
    sorted_indices = np.argsort(feature_importance)[::-1]
    cumulative_importance = np.cumsum(feature_importance[sorted_indices])
    
    # 确保不超过实际可用特征数
    max_available = len(sorted_indices)

    # 选择重要性累积到80%的特征，或者至少target_features个特征
    target_cumulative = 0.8
    n_selected = np.where(cumulative_importance >= target_cumulative)[0]
    n_selected = n_selected[0] + 1 if len(n_selected) > 0 else max_available
    # 确保不超过可用特征数，且至少选择target_features个（如果可用的话）
    n_selected = min(max(target_features, n_selected), max_available)

    # 映射回原始特征索引
    selected_pre_indices = sorted_indices[:n_selected]
    
    # 创建组合选择器
    if orthogonalizer is not None:
        # 使用模块级别的CombinedSelector类（支持pickle序列化）
        selector = CombinedSelector(orthogonalizer, pre_selector, selected_pre_indices)
    else:
        selector = AdaptiveSelector(pre_selector, selected_pre_indices)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    print(f"最终特征选择: {X_train.shape[1]} -> {X_train_selected.shape[1]}")
    if n_selected > 0:
        print(f"累积重要性覆盖: {cumulative_importance[n_selected-1]:.3f}")
    else:
        print(f"累积重要性覆盖: 0.000")

    return selector, X_train_selected, X_val_selected, X_test_selected

def conservative_feature_selection(X_train, y_train, X_val, X_test, 
                                 max_features=40, correlation_threshold=0.9):
    """
    保守的特征选择策略
    更严格的特征筛选，减少过拟合风险
    
    Parameters:
    - max_features: 最大特征数量（默认40）
    - correlation_threshold: 相关性阈值（默认0.9，更严格）
    """
    print("🔄 保守特征选择...")
    
    # 1. 先去除高相关性特征
    orthogonalizer = FeatureOrthogonalizer(
        correlation_threshold=correlation_threshold,
        use_pca=False  # 不使用PCA，保持可解释性
    )
    X_train_ortho = orthogonalizer.fit_transform(X_train, y_train)
    X_val_ortho = orthogonalizer.transform(X_val)
    X_test_ortho = orthogonalizer.transform(X_test)
    
    print(f"正交化后特征维度: {X_train.shape[1]} -> {X_train_ortho.shape[1]}")
    
    # 2. 基于XGBoost重要性进行选择
    temp_model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42, 
        n_jobs=-1
    )
    temp_model.fit(X_train_ortho, y_train)
    
    # 3. 更严格的重要性筛选
    importance = temp_model.feature_importances_
    # 只考虑重要性大于0的特征，然后取30分位数作为阈值
    positive_importance = importance[importance > 0]
    if len(positive_importance) > 0:
        importance_threshold = np.percentile(positive_importance, 30)
    else:
        importance_threshold = 0
    
    selected_indices = np.where(importance >= importance_threshold)[0]
    
    # 4. 限制最大特征数量
    if len(selected_indices) > max_features:
        top_indices = np.argsort(importance[selected_indices])[-max_features:]
        selected_indices = selected_indices[top_indices]
    
    # 创建选择器
    selector = ConservativeSelector(orthogonalizer, selected_indices)
    
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    print(f"保守特征选择: {X_train.shape[1]} -> {X_train_selected.shape[1]}")
    if len(selected_indices) > 0:
        print(f"   重要性阈值: {importance_threshold:.6f}, 选择特征数: {len(selected_indices)}")
    
    return selector, X_train_selected, X_val_selected, X_test_selected

def get_feature_names(use_improved_features: bool = False):
    """
    获取特征名称列表（与特征生成函数保持一致）

    Parameters:
    - use_improved_features: 是否使用改进的特征工程

    Returns:
    - feature_names: 特征名称列表
    """
    if use_improved_features:
        # 精简版特征名称
        feature_names = [
            # 价格动量特征
            'momentum_5', 'momentum_vol_adj_5', 'range_efficiency_5',
            'momentum_10', 'momentum_vol_adj_10', 'range_efficiency_10',
            'momentum_20', 'momentum_vol_adj_20', 'range_efficiency_20',
            # 量价关系特征
            'price_volume_divergence_5', 'vwap_premium_5',
            'price_volume_divergence_10', 'vwap_premium_10',
            # 均值回复特征
            'ma_deviation_10', 'rsi_10',
            'ma_deviation_20', 'rsi_20',
            # 市场微观结构特征（如果可用）
            'money_flow_5', 'large_order_ratio_5',
            'money_flow_10', 'large_order_ratio_10',
            # 技术指标特征
            'macd_signal', 'bollinger_position_20'
        ]
    else:
        # 完整版特征名称
        feature_names = [
            # 基础价格特征
            'open', 'high', 'low', 'close', 'volume', 'vwap',
            # 累积买卖比率因子
            'cum_buy_sell_ratio',
            # 买入稳定因子
            'buy_stability_5', 'buy_stability_10', 'buy_stability_20', 'buy_stability_30',
            # 方向动量因子
            'directional_momentum_5', 'directional_momentum_10', 'directional_momentum_20', 'directional_momentum_30',
            # 定向波动率因子
            'directional_volatility_5', 'directional_volatility_10', 'directional_volatility_20', 'directional_volatility_30',
            # 高波动性动量
            'hv_momentum_product_5', 'hv_momentum_ratio_5',
            'hv_momentum_product_10', 'hv_momentum_ratio_10',
            'hv_momentum_product_20', 'hv_momentum_ratio_20',
            'hv_momentum_product_30', 'hv_momentum_ratio_30',
            # 动量-主动买入极值
            'momentum_tbq_extreme_10', 'momentum_tbq_extreme_20', 'momentum_tbq_extreme_30',
            # 动量-成交量极值分组因子
            'momentum_volume_extreme_10', 'momentum_volume_extreme_20', 'momentum_volume_extreme_30',
            # 净订单流
            'net_order_flow_5', 'net_order_flow_10', 'net_order_flow_20', 'net_order_flow_30',
            # 涨跌距离/路径效率
            'price_path_efficiency_5', 'price_path_efficiency_10', 'price_path_efficiency_20', 'price_path_efficiency_30',
            # 滚动相关性
            'rolling_corr_buyquote_5', 'rolling_corr_buyquote_10', 'rolling_corr_buyquote_20', 'rolling_corr_buyquote_30',
            # RSI截面排序因子
            'rsi_14', 'rsi_21', 'rsi_30',
            # TBQ极值效率
            'tbq_extreme_efficiency_10', 'tbq_extreme_efficiency_20', 'tbq_extreme_efficiency_30',
            # 交易比
            'top_bottom_trades_ratio_10', 'top_bottom_trades_ratio_20', 'top_bottom_trades_ratio_30',
            # 波动效率
            'volatility_efficiency_5', 'volatility_efficiency_10', 'volatility_efficiency_20', 'volatility_efficiency_30',
            # 波动率-订单流
            'volatility_order_flow_5', 'volatility_order_flow_10', 'volatility_order_flow_20', 'volatility_order_flow_30',
            # 量稳因子
            'volume_stability_10', 'volume_stability_20', 'volume_stability_30',
            # 成交量切分
            'volume_tbb_extreme_10', 'volume_tbb_extreme_20', 'volume_tbb_extreme_30'
        ]

        # 条件特征（如果有相关数据才添加）
        conditional_features = ['quote_volume']  # 基础价格特征已包含
        # 如果有taker_buy_quote数据，会添加相关的特征（已在上面包含）

    return feature_names

def create_features_symbol_fast_based_on_feature_txt(gp: pd.DataFrame, rebalance_period: int, min_history_days: int = 60, use_improved_features: bool = False):
    """
    统一的特征生成函数，基于feature.txt的特征定义创建特征
    仅使用历史滚动值（rolling/ewm/shift），无未来信息泄露
    
    **重要**: 为避免未来函数，所有基础价格特征都使用T-1日的数据
    即：在T日做决策时，只能使用截至T-1日收盘的所有数据
    
    Parameters:
    - gp: 单个symbol的K线数据
    - rebalance_period: 重平衡周期
    - min_history_days: 最小历史天数
    - use_improved_features: 如果True，使用精简版特征（23个特征，减少过拟合风险）
                           如果False，使用完整版特征（100+个特征，基于feature.txt）
    """
    eps = 1e-8
    gp = gp.sort_values('date').reset_index(drop=True).copy()

    if len(gp) < min_history_days + rebalance_period + 20:
        return pd.DataFrame()

    # 🔥 关键修改：使用T-1日的数据作为特征，避免未来函数
    # 在T日做决策时，只能使用截至T-1日的所有数据
    close = gp['close'].shift(1)
    high  = gp['high'].shift(1)
    low   = gp['low'].shift(1)
    open = gp['open'].shift(1)
    vol   = gp['volume'].shift(1)
    qv    = gp['quote_volume'].shift(1) if 'quote_volume' in gp.columns else None
    tbq   = gp['taker_buy_quote'].shift(1) if 'taker_buy_quote' in gp.columns else None
    tbb   = gp['taker_buy_base'].shift(1) if 'taker_buy_base' in gp.columns else None
    trades_count = gp['trades_count'].shift(1) if 'trades_count' in gp.columns else None

    feat = pd.DataFrame({
        'date': gp['date'],
        'symbol': gp['symbol']
    })

    # ========== 精简版特征（use_improved_features=True）==========
    if use_improved_features:
        # 1. 基础价格动量特征（减少窗口数量）
        for window in [5, 10, 20]:
            # 价格动量
            feat[f'momentum_{window}'] = close.pct_change(window)
            # 波动率调整动量
            volatility = close.pct_change().rolling(window).std()
            feat[f'momentum_vol_adj_{window}'] = feat[f'momentum_{window}'] / (volatility + eps)
            
            # 高低波动特征
            high_low_ratio = (high / low - 1).rolling(window).mean()
            feat[f'range_efficiency_{window}'] = feat[f'momentum_{window}'] / (high_low_ratio + eps)
        
        # 2. 量价关系特征
        for window in [5, 10]:
            # 量价背离
            price_change = close.pct_change(window)
            volume_change = vol.pct_change(window)
            feat[f'price_volume_divergence_{window}'] = price_change - volume_change
            
            # 成交量加权价格
            vwap = (vol * (high + low + close) / 3).rolling(window).sum() / (vol.rolling(window).sum() + eps)
            feat[f'vwap_premium_{window}'] = (close - vwap) / close
        
        # 3. 均值回复特征
        for window in [10, 20]:
            # 价格与均线的偏离
            ma = close.rolling(window).mean()
            feat[f'ma_deviation_{window}'] = (close - ma) / ma
            
            # RSI改进版本
            returns = close.pct_change()
            gain = returns.clip(lower=0).rolling(window).mean()
            loss = (-returns.clip(upper=0)).rolling(window).mean()
            rs = gain / (loss + eps)
            feat[f'rsi_{window}'] = 100 - 100 / (1 + rs)
        
        # 4. 市场微观结构特征
        if tbq is not None and qv is not None:
            for window in [5, 10]:
                # 资金流强度
                money_flow = (tbq - (qv - tbq)) / (qv + eps)
                feat[f'money_flow_{window}'] = money_flow.rolling(window).mean()
                
                # 大单净流入
                large_order_ratio = (tbq / (qv + eps)).rolling(window).mean()
                feat[f'large_order_ratio_{window}'] = large_order_ratio
        
        # 5. 技术指标特征
        # MACD信号
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        feat['macd_signal'] = macd.ewm(span=9).mean()
        
        # 布林带位置
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        feat['bollinger_position_20'] = (close - bb_middle) / (2 * bb_std + eps)
        
        # 目标变量: 注意close已经是shift(1)，所以这里相当于预测从T日到T+rebalance_period日的收益
        # 即：用T-1日及之前的特征，预测T日到T+rebalance_period日的收益
        feat["future_ret"] = np.log(gp['close'].shift(-rebalance_period) / gp['close'])
        
        # 清理数据
        feat = feat.iloc[min_history_days:-rebalance_period].dropna().reset_index(drop=True)
        return feat
    
    # ========== 完整版特征（use_improved_features=False，基于feature.txt）==========

    # 基础价格特征（已经通过shift(1)处理，使用T-1日数据）
    feat['open']   = open
    feat['high']   = high
    feat['low']    = low
    feat['close']  = close
    feat['volume'] = vol

    if qv is not None:
        feat['quote_volume'] = qv
        feat['vwap'] = qv / (vol + eps)
    else:
        feat['vwap'] = close

    # ===== 特征1: 累积主买/累积主卖比率因子 =====
    if tbq is not None and qv is not None:
        tbs = qv - tbq  # 主动卖盘
        # 收盘为涨的日期：累积主动买盘
        up_days = close > close.shift(1)
        cum_buy_up = (tbq * up_days).rolling(window=20, min_periods=1).sum()
        cum_sell_down = (tbs * (~up_days)).rolling(window=20, min_periods=1).sum()
        feat['cum_buy_sell_ratio'] = cum_buy_up / (cum_sell_down + eps)

    # ===== 特征2: 买入稳定因子 =====
    if tbq is not None and qv is not None:
        buy_ratio = tbq / (qv + eps)
        for n in (5, 10, 20, 30):
            buy_ratio_mean = buy_ratio.rolling(n).mean()
            buy_ratio_std = buy_ratio.rolling(n).std()
            feat[f'buy_stability_{n}'] = buy_ratio_mean / (buy_ratio_std + eps)

    # ===== 特征3: 方向动量因子 =====
    for window in (5, 10, 20, 30):
        up_momentum = (close.diff().clip(lower=0)).rolling(window).sum()
        down_momentum = ((-close.diff()).clip(lower=0)).rolling(window).sum()
        feat[f'directional_momentum_{window}'] = up_momentum / (down_momentum + eps)

    # ===== 特征4: 定向波动率因子 =====
    for window in (5, 10, 20, 30):
        up_volatility = ((high - open) / (open + eps)).rolling(window).mean()
        down_volatility = ((open - low) / (open + eps)).rolling(window).mean()
        up_roc = (high - open.shift(window)) / (open.shift(window) + eps)
        down_roc = (open.shift(window) - low) / (open.shift(window) + eps)
        feat[f'directional_volatility_{window}'] = (up_roc / (up_volatility + eps)) - (down_roc / (down_volatility + eps))

    # ===== 特征5: 高波动性动量 =====
    for window in (5, 10, 20, 30):
        roc = close.pct_change(window)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        range_pct = (high - low) / (close + eps)
        feat[f'hv_momentum_product_{window}'] = roc * atr
        feat[f'hv_momentum_ratio_{window}'] = roc / (range_pct + eps)

    # ===== 特征6: 动量-主动买入极值 =====
    if tbq is not None:
        for window in (10, 20, 30):
            momentum = close.pct_change(window)
            # 简化实现：使用滚动窗口内的相关性或分位数
            tbq_rank = tbq.rolling(window).rank(pct=True)
            feat[f'momentum_tbq_extreme_{window}'] = momentum * tbq_rank

    # ===== 特征7: 动量-成交量极值分组因子 =====
    for window in (10, 20, 30):
        momentum = close.pct_change(window)
        vol_rank = vol.rolling(window).rank(pct=True)
        feat[f'momentum_volume_extreme_{window}'] = momentum * vol_rank

    # ===== 特征8: net_order_flow =====
    if tbq is not None and qv is not None:
        for window in (5, 10, 20, 30):
            daily_flow = tbq - (qv - tbq)  # taker_buy_quote - taker_sell_quote
            net_flow = daily_flow.rolling(window).sum() / (qv.rolling(window).sum() + eps)
            feat[f'net_order_flow_{window}'] = net_flow

    # ===== 特征9: 涨跌距离/路径（价格路径效率） =====
    for window in (5, 10, 20, 30):
        net_ret = close.pct_change(window)
        path_len = close.pct_change().abs().rolling(window).sum()
        feat[f'price_path_efficiency_{window}'] = net_ret / (path_len + eps)

    # ===== 特征10: rolling_corr_buyquote =====
    if tbq is not None:
        for window in (5, 10, 20, 30):
            ret = np.log(close / close.shift(1))
            corr = ret.rolling(window).corr(tbq)
            vol_mean = vol.rolling(window).mean()
            feat[f'rolling_corr_buyquote_{window}'] = corr * np.log(1 + vol_mean)

    # ===== 特征11: RSI 截面排序因子 =====
    for period in (14, 21, 30):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + eps)
        rsi = 100 - 100 / (1 + rs)
        feat[f'rsi_{period}'] = rsi

    # ===== 特征12: TBQ极值效率 =====
    if tbq is not None:
        for window in (10, 20, 30):
            roc = close.pct_change(window)
            range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
            vol_eff = roc / (range_pct + eps)
            tbq_rank = tbq.rolling(window).rank(pct=True)
            feat[f'tbq_extreme_efficiency_{window}'] = vol_eff * tbq_rank

    # ===== 特征13: Top-Bottom Trades Ratio =====
    if trades_count is not None and tbq is not None and qv is not None:
        for window in (10, 20, 30):
            buy_ratio = tbq / (qv + eps)
            trades_rank = trades_count.rolling(window).rank(pct=True)
            feat[f'top_bottom_trades_ratio_{window}'] = buy_ratio * trades_rank

    # ===== 特征14: 波动效率 =====
    for window in (5, 10, 20, 30):
        roc = close.pct_change(window)
        range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
        feat[f'volatility_efficiency_{window}'] = roc / (range_pct + eps)

    # ===== 特征15: 波动率-订单流 =====
    if trades_count is not None and qv is not None:
        for window in (5, 10, 20, 30):
            roc = close.pct_change(window)
            range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
            vol_eff = roc / (range_pct + eps)
            avg_trade_size = qv / (trades_count + eps)
            trades_ratio = trades_count / (trades_count.rolling(window).mean() + eps)
            avg_trade_size_ratio = avg_trade_size / (avg_trade_size.rolling(window).mean() + eps)
            order_flow = trades_ratio / (avg_trade_size_ratio + eps)
            feat[f'volatility_order_flow_{window}'] = vol_eff * order_flow

    # ===== 特征16: 量稳因子（成交量稳定性） =====
    if trades_count is not None:
        for lookback_days in (10, 20, 30):
            volume_mean = vol.rolling(lookback_days).mean()
            volume_std = vol.rolling(lookback_days).std()
            feat[f'volume_stability_{lookback_days}'] = volume_mean / (volume_std + eps)

    # ===== 特征17: 成交量切分-主动买入Base 极值 =====
    if tbb is not None:
        for window in (10, 20, 30):
            tb_base_mom = tbb.rolling(10).sum()
            vol_rank = vol.rolling(window).rank(pct=True)
            feat[f'volume_tbb_extreme_{window}'] = tb_base_mom * vol_rank

    # 目标：未来对数收益（仅shift，不参与特征构造）
    # 注意：close已经是shift(1)，所以这里相当于预测从T日到T+rebalance_period日的收益
    # 即：用T-1日及之前的特征，预测T日到T+rebalance_period日的收益
    feat["future_ret"] = np.log(gp['close'].shift(-rebalance_period) / gp['close'])

    # 丢掉前期不完整/末尾无目标的行
    feat = feat.iloc[min_history_days : -rebalance_period].dropna().reset_index(drop=True)
    return feat

def create_strict_time_split(data_with_dates, test_days=180, validation_days=180):
    """
    严格的时间序列分割
    支持4元组 (X, y, symbol, date) 或 5元组 (X, y, y_original, symbol, date)
    """
    print("🔄 严格时间序列分割...")
    
    # 检测数据格式：4元组还是5元组
    is_5tuple = len(data_with_dates[0]) == 5
    
    if is_5tuple:
        # 5元组: (X, y, y_original, symbol, date)
        data_with_dates.sort(key=lambda x: x[4])
        dates = [item[4] for item in data_with_dates]
    else:
        # 4元组: (X, y, symbol, date)
        data_with_dates.sort(key=lambda x: x[3])
        dates = [item[3] for item in data_with_dates]
    
    min_date, max_date = min(dates), max(dates)
    
    # 测试集：最后test_days天
    test_cutoff = max_date - pd.Timedelta(days=test_days - 1)
    if is_5tuple:
        test_data = [it for it in data_with_dates if it[4] >= test_cutoff]
    else:
        test_data = [it for it in data_with_dates if it[3] >= test_cutoff]
    
    # 验证集：测试集之前的validation_days天
    val_cutoff = test_cutoff - pd.Timedelta(days=validation_days)
    if is_5tuple:
        val_data = [it for it in data_with_dates if val_cutoff <= it[4] < test_cutoff]
        train_data = [it for it in data_with_dates if it[4] < val_cutoff]
    else:
        val_data = [it for it in data_with_dates if val_cutoff <= it[3] < test_cutoff]
        train_data = [it for it in data_with_dates if it[3] < val_cutoff]
    
    print(f"数据时间范围: {min_date} ~ {max_date}")
    print(f"训练集: {min_date} ~ {val_cutoff.date()} ({len(train_data)} 样本)")
    print(f"验证集: {val_cutoff.date()} ~ {test_cutoff.date()} ({len(val_data)} 样本)")
    print(f"测试集: {test_cutoff.date()} ~ {max_date} ({len(test_data)} 样本)")
    
    return train_data, val_data, test_data

def train_robust_xgboost(X_train, y_train, X_val, y_val, dates_val=None, symbols_val=None,
                        use_bayesian_opt=True, ic_ir_weight=0.1, rankIC=1.8, ir_weight=0.0, monotonicity_weight=0.1):
    """更稳健的训练过程"""
    print("🛡️ 使用稳健训练策略...")
    
    # 1. 更严格的异常值处理和数据增强
    def robust_winsorize(y, lower_percentile=2, upper_percentile=98):
        lower_bound = np.percentile(y, lower_percentile)
        upper_bound = np.percentile(y, upper_percentile)
        return np.clip(y, lower_bound, upper_bound)

    # 添加数据增强正则化（轻微噪声注入）
    def add_noise_regularization(X, noise_level=0.01):
        """添加轻微噪声作为正则化"""
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise

    y_train_processed = robust_winsorize(y_train)
    y_val_processed = robust_winsorize(y_val)

    # 应用数据增强正则化
    X_train_augmented = add_noise_regularization(X_train, noise_level=0.005)
    
    print(f"目标收益率统计（稳健处理）:")
    print(f"  训练集: 均值={y_train_processed.mean():.6f}, 标准差={y_train_processed.std():.6f}")
    print(f"  验证集: 均值={y_val_processed.mean():.6f}, 标准差={y_val_processed.std():.6f}")
    
    # 2. 参数优化（如果启用）
    if use_bayesian_opt and SKOPT_AVAILABLE:
        print("🎯 使用贝叶斯优化寻找最优参数...")
        params, best_score = bayesian_optimize_xgboost(
            X_train, y_train_processed, X_val, y_val_processed,
            dates_val=dates_val, symbols_val=symbols_val,
            n_calls=25, n_random_starts=5,
            ic_ir_weight=ic_ir_weight, rankIC=rankIC, ir_weight=ir_weight, monotonicity_weight=monotonicity_weight
        )
        if best_score is not None:
            if dates_val is not None and symbols_val is not None:
                print(f"✅ 贝叶斯优化完成，最优加权IC指标: {best_score:.6f}")
            else:
                print(f"✅ 贝叶斯优化完成，最优验证MAE: {best_score:.6f}")
    else:
        print("📋 使用增强的正则化参数")
        params = get_enhanced_xgb_params()
    
    # 3. 增强正则化参数（如果未指定则使用更强的默认值）
    if 'reg_alpha' not in params:
        params['reg_alpha'] = 15.0  # 增强L1正则化
    if 'reg_lambda' not in params:
        params['reg_lambda'] = 25.0  # 增强L2正则化
    if 'gamma' not in params:
        params['gamma'] = 5.0  # 增强后剪枝
    if 'min_child_weight' not in params:
        params['min_child_weight'] = 30  # 增强最小叶子权重
    if 'subsample' not in params:
        params['subsample'] = 0.35  # 降低子采样
    if 'colsample_bytree' not in params:
        params['colsample_bytree'] = 0.2  # 降低特征采样

    dtrain = xgb.DMatrix(X_train_augmented, label=y_train_processed)
    dval = xgb.DMatrix(X_val, label=y_val_processed)
    
    # 4. 使用更严格的训练策略
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=min(params.get('n_estimators', 2000), 2000),  # 限制最大轮数
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=min(params.get('early_stopping_rounds', 80), 80),  # 更严格的早停
    verbose_eval=100
    )

    # 验证集预测
    val_pred = model.predict(dval)
    val_rmse = np.sqrt(mean_squared_error(y_val_processed, val_pred))
    val_mae = mean_absolute_error(y_val_processed, val_pred)
    val_r2 = r2_score(y_val_processed, val_pred)
    
    print(f"[验证集-稳健训练] RMSE={val_rmse:.6f}, MAE={val_mae:.6f}, R2={val_r2:.4f}")
    
    return model, val_pred

def train_enhanced_xgboost_model(X_train, y_train, X_val, y_val, symbols_train, symbols_val, dates_train, dates_val,
                                 use_bayesian_opt=True, use_robust_training=False,
                                 ic_ir_weight=0.1, rankIC=1.8, ir_weight=0.0, monotonicity_weight=0.1):
    """训练增强版XGBoost模型，支持贝叶斯优化和稳健训练"""
    if use_robust_training:
        return train_robust_xgboost(X_train, y_train, X_val, y_val,
                                   dates_val=dates_val, symbols_val=symbols_val,
                                   use_bayesian_opt=use_bayesian_opt,
                                   ic_ir_weight=ic_ir_weight, rankIC=rankIC, ir_weight=ir_weight, monotonicity_weight=monotonicity_weight)
    
    print("🚀 开始增强版XGBoost训练...")

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

    print(f"目标收益率统计:")
    print(f"  训练集: 均值={y_train_clipped.mean():.6f}, 标准差={y_train_clipped.std():.6f}")
    print(f"  验证集: 均值={y_val_clipped.mean():.6f}, 标准差={y_val_clipped.std():.6f}")

    # 贝叶斯优化参数（可选）
    if use_bayesian_opt and SKOPT_AVAILABLE:
        print("🎯 使用贝叶斯优化寻找最优参数...")
        params, best_score = bayesian_optimize_xgboost(
            X_train, y_train_clipped, X_val, y_val_clipped,
            dates_val=dates_val, symbols_val=symbols_val,
            n_calls=30, n_random_starts=5,  # 减少迭代次数以加快速度
            ic_ir_weight=ic_ir_weight, rankIC=rankIC, ir_weight=ir_weight, monotonicity_weight=monotonicity_weight
        )
        if best_score is not None:
            if dates_val is not None and symbols_val is not None:
                print(f"✅ 贝叶斯优化完成，最优加权IC指标: {best_score:.6f}")
            else:
                print(f"✅ 贝叶斯优化完成，最优验证MAE: {best_score:.6f}")
        else:
            print(f"📋 使用{reg_type}XGBoost参数")
            if regularization_strength == 'ultra':
                params = get_ultra_regularized_xgb_params()
            elif regularization_strength == 'enhanced':
                params = get_enhanced_xgb_params()
            else:  # 'default'
                params = get_default_xgb_params()
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train_clipped)
    dval = xgb.DMatrix(X_val, label=y_val_clipped)
    
    # 训练模型
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=150,
        verbose_eval=100
    )
    
    # 验证集预测
    val_pred = model.predict(dval)
    val_rmse = np.sqrt(mean_squared_error(y_val_clipped, val_pred))
    val_mae = mean_absolute_error(y_val_clipped, val_pred)
    val_r2 = r2_score(y_val_clipped, val_pred)
    
    print(f"[验证集] RMSE={val_rmse:.6f}, MAE={val_mae:.6f}, R2={val_r2:.4f}")
    
    return model, val_pred

def create_xgboost_prediction_factor(window=20, rebalance_period=5, train_epochs=3000,
                                            min_history_days=60, min_samples_per_symbol=30,
                                    use_bayesian_opt=True,
                                    use_orthogonalization=True, correlation_threshold=0.95, use_pca=False,
                                    use_improved_features=False, use_robust_training=False,
                                    use_conservative_selection=False, max_features=40,
                                    use_robust_preprocessing=False, use_rank_target=False,
                                    regularization_strength='ultra',  # 'default', 'enhanced', 'ultra'
                                    ic_ir_weight=0.1, rankIC=1.8, ir_weight=0.0, monotonicity_weight=0.1):
    """
    基于XGBoost的收益率预测因子
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
    - ic_ir_weight: IC_IR的权重（默认0.4），用于贝叶斯优化目标函数
    - rankIC: Rank IC的权重（默认0.3），用于贝叶斯优化目标函数
    - ir_weight: IR的权重（默认0.2），用于贝叶斯优化目标函数
    - monotonicity_weight: 分组单调性的权重（默认0.2），用于贝叶斯优化目标函数
    - regularization_strength: 正则化强度选择 ('default'=默认, 'enhanced'=增强, 'ultra'=超强)
    """
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
    selection_type = "保守选择" if use_conservative_selection else "自适应选择"
    preprocessing_type = "稳健预处理" if use_robust_preprocessing else "标准预处理"
    target_type = "预测排名" if use_rank_target else "预测收益率"

    print(f"🚀 开始构建XGBoost预测因子 ({opt_type}, {ortho_type}, {feature_type}, {training_type}, {selection_type}, {preprocessing_type}, {target_type}, {reg_type})")
    print(f"参数: window={window}, rebalance_period={rebalance_period}")
    print(f"      min_history_days={min_history_days}, min_samples_per_symbol={min_samples_per_symbol}")
    print(f"      use_bayesian_opt={use_bayesian_opt}")
    print(f"      use_orthogonalization={use_orthogonalization}, correlation_threshold={correlation_threshold}, use_pca={use_pca}")
    print(f"      use_improved_features={use_improved_features}")
    print(f"      use_robust_training={use_robust_training}")
    print(f"      use_conservative_selection={use_conservative_selection}, max_features={max_features}")
    print(f"      use_robust_preprocessing={use_robust_preprocessing}")
    print(f"      use_rank_target={use_rank_target} ({target_type})")

    if not XGB_AVAILABLE:
        print("XGBoost不可用，无法进行训练")
        return pd.DataFrame()

    # K线数据
    df = load_kline_df()
    
    # 🔥 关键：基于K线数据构建每日前50排名，只使用前50个币种
    print(f"📊 构建每日前{50}排名，只使用前50个币种...")
    available_tokens_by_date = build_available_tokens_by_date_from_kline(
        kline_df=df,
        top_n=50,
        ranking_method='quote_volume',
        rebalance_period=rebalance_period,
        strict_top_n=True  # 严格限制为50个币种
    )
    
    print(f"✅ 排名构建完成: {len(available_tokens_by_date)} 个交易日")
    
    # 过滤数据：只保留每日前50的币种（但保留它们的完整历史数据）
    print("🔄 过滤数据，只保留每日前50的币种...")
    filtered_data = []
    for symbol, group in df.groupby('symbol'):
        filtered_group = filter_group_by_availability(group, symbol, available_tokens_by_date)
        if not filtered_group.empty:
            filtered_data.append(filtered_group)
    
    if not filtered_data:
        print("警告: 过滤后没有可用数据")
        return pd.DataFrame()
    
    df = pd.concat(filtered_data, ignore_index=True)
    print(f"✅ 数据过滤完成: {df['symbol'].nunique()} 个币种, {len(df):,} 条记录")

    # 准备训练数据（返回原始目标变量用于IC计算）
    if use_robust_preprocessing:
        all_features, all_targets, all_targets_original, all_symbols, all_dates = prepare_robust_data(
            df, rebalance_period, min_history_days, n_jobs=40, 
            use_improved_features=use_improved_features,
            use_rank_target=use_rank_target
        )
    else:
        all_features, all_targets, all_targets_original, all_symbols, all_dates = prepare_xgboost_data_fast(
            df, rebalance_period, min_history_days, n_jobs=40, 
            use_improved_features=use_improved_features,
            use_rank_target=use_rank_target
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
    
    # 严格的时间序列分割（包含原始目标变量）
    data_with_dates = list(zip(all_features, all_targets, all_targets_original, all_symbols, all_dates))
    train_data, val_data, test_data = create_strict_time_split(
        data_with_dates, test_days=180, validation_days=180
    )

    # 分离数据（使用排名目标变量用于训练，原始目标变量用于IC计算）
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
    
    # 特征选择 - 使用自适应特征选择或保守特征选择
    print("🔄 进行特征选择...")
    if use_cached:
        selector = joblib.load(selector_path)
        X_train_selected = selector.transform(X_train)
        X_val_selected   = selector.transform(X_val)
        X_test_selected  = selector.transform(X_test)
    else:
        if use_conservative_selection:
            # 使用保守特征选择策略
            selector, X_train_selected, X_val_selected, X_test_selected = conservative_feature_selection(
                X_train, y_train, X_val, X_test,
                max_features=max_features,
                correlation_threshold=correlation_threshold
            )
        else:
            # 使用自适应特征选择策略
            selector, X_train_selected, X_val_selected, X_test_selected = adaptive_feature_selection(
                X_train, y_train, X_val, X_test,
                target_features=100,
                importance_threshold=0.001,
                correlation_threshold=correlation_threshold,
                use_orthogonalization=use_orthogonalization,
                use_pca=use_pca
            )
        # 保存特征选择器（无论哪种选择方法）
        joblib.dump(selector, selector_path)
        print(f"特征选择: {X_train.shape[1]} -> {X_train_selected.shape[1]}")

        # 特征相关性分析和可视化
        from pathlib import Path
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        # 获取实际的特征名称
        all_feature_names = get_feature_names(use_improved_features)

        corr_save_path = reports_dir / f"xgb_features_correlation_w{window}_r{rebalance_period}.png"
        analyze_selected_features_correlation(
            X_train_selected, X_train_selected.shape[1],
            title=f"XGBoost选定特征相关性分析 (w{window}_r{rebalance_period})",
            save_path=str(corr_save_path),
            feature_names=all_feature_names[:X_train_selected.shape[1]]  # 只取选定特征的数量
        )

        # 特征重要性分析
        importance_save_path = reports_dir / f"xgb_features_importance_w{window}_r{rebalance_period}.png"
        analyze_feature_importance(
            X_train_selected, y_train, X_train_selected.shape[1],
            title=f"XGBoost特征重要性分析 (w{window}_r{rebalance_period})",
            save_path=str(importance_save_path),
            feature_names=all_feature_names[:X_train_selected.shape[1]]  # 只取选定特征的数量
        )

    # 标准化
    print("🔄 标准化特征（使用RobustScaler）...")
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
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        model = booster
        print("✅ 已加载缓存模型与预处理器")
    else:
        model, val_pred = train_enhanced_xgboost_model(
            X_train_scaled, y_train, X_val_scaled, y_val,
            symbols_train, symbols_val, dates_train, dates_val,
            use_bayesian_opt=use_bayesian_opt,
            use_robust_training=use_robust_training,
            ic_ir_weight=ic_ir_weight, rankIC=rankIC, ir_weight=ir_weight, monotonicity_weight=monotonicity_weight
        )
        # 保存模型（包含best_iteration信息）
        model.save_model(str(model_path))
        print(f"✅ 模型已保存到: {model_path}")
    
    # 验证集预测（用于IC计算）
    dval = xgb.DMatrix(X_val_scaled)
    val_pred = model.predict(dval)
    
    # 根据use_rank_target选择用于IC计算的目标变量
    if use_rank_target:
        # 使用排名目标变量计算IC
        val_actual = y_val
        test_actual = y_test
    else:
        # 使用原始收益率计算IC
        val_actual = y_val_original
        test_actual = y_test_original
    
    # 计算验证集IC
    val_df = pd.DataFrame({
        'date': dates_val,
        'pred': val_pred,
        'actual': val_actual
    })
    val_daily_corrs = []
    for date, group in val_df.groupby('date'):
        if len(group) >= 5:
            corr = group['pred'].corr(group['actual'])
            if not np.isnan(corr):
                val_daily_corrs.append(corr)
    
    val_avg_ic = np.mean(val_daily_corrs) if val_daily_corrs else 0
    print(f"[验证集] 平均IC={val_avg_ic:.4f}, 有效日期数={len(val_daily_corrs)}")
    
    # 样本外预测和评估
    print("🔄 开始样本外预测...")
    dtest = xgb.DMatrix(X_test_scaled)
    test_pred = model.predict(dtest)
    
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
    
    # 计算横截面相关性（按日期）
    test_df = pd.DataFrame({
        'date': dates_test,
        'pred': test_pred,
        'actual': test_actual
    })
    test_daily_corrs = []
    for date, group in test_df.groupby('date'):
        if len(group) >= 5:
            corr = group['pred'].corr(group['actual'])
            if not np.isnan(corr):
                test_daily_corrs.append(corr)
    
    test_avg_ic = np.mean(test_daily_corrs) if test_daily_corrs else 0
    print(f"[样本外-Test]  RMSE={test_rmse:.6f}, MAE={test_mae:.6f}, R2={test_r2:.4f}")
    print(f"[样本外-Test]  平均IC={test_avg_ic:.4f}, 有效日期数={len(test_daily_corrs)}")

    # 统一生成因子数据（使用原始目标变量）
    result_data = []
    sets = [
        ("train", X_train_scaled, symbols_train, dates_train, y_train_original),
        ("val",   X_val_scaled,   symbols_val,   dates_val,   y_val_original),
        ("test",  X_test_scaled,  symbols_test,  dates_test,  y_test_original),
    ]
    
    for split, Xs, syms, dts, ys_orig in sets:
        if len(Xs) == 0:
            continue
        dmatrix = xgb.DMatrix(Xs)
        preds = model.predict(dmatrix)
        for pred, s, d, y_true_orig in zip(preds, syms, dts, ys_orig):
            result_data.append({
                'date': d,
                'symbol': s,
                'xgb_prediction': float(pred),
                'future_ret': float(y_true_orig),  # 使用原始目标变量
                'split': split
            })

    if len(result_data) == 0:
        print("警告: 无可用预测样本，返回空 DataFrame")
        return pd.DataFrame()

    # 构造DataFrame
    factor_df = pd.DataFrame(result_data)

    # 直接使用预测收益率作为因子，按日做rank归一化
    factor_df = rank_to_unit_by_date(factor_df, col="xgb_prediction", out_col="factor")

    # 重命名并保存
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret", "split"]]
    out_path = save_factor_df(factor_df, file_prefix=f"xgboost_{window}d_rebalance{rebalance_period}d_")
    print_factor_summary(factor_df, out_path)

    # 新增：如果因"目标位移"导致factor_df不是最新自然日，则进行一次Live推理并打印真正最新日
    print_latest_daily_groups_live(
        window=window, 
        rebalance_period=rebalance_period, 
        min_history_days=min_history_days, 
        groups=5,
        use_improved_features=use_improved_features,
        use_robust_preprocessing=use_robust_preprocessing
    )
    
    return factor_df

def create_features_symbol_inference_latest(gp: pd.DataFrame, min_history_days: int = 60, 
                                           use_improved_features: bool = False) -> pd.DataFrame:
    """
    仅用于推理：构造与训练一致的特征，但不生成future_ret，也不丢弃末尾。
    返回该symbol在其各自最新日期的一行特征。
    使用基于feature.txt的特征定义。
    
    **重要**: 为避免未来函数，所有基础价格特征都使用T-1日的数据
    推理时：在最新日期T做预测时，使用截至T-1日收盘的所有数据
    
    Parameters:
    - gp: 单个symbol的K线数据
    - min_history_days: 最小历史天数
    - use_improved_features: 是否使用改进的特征工程（需与训练时一致）
    """
    eps = 1e-8
    gp = gp.sort_values('date').reset_index(drop=True).copy()
    
    # 推理时只需要min_history_days的历史数据，不需要rebalance_period的未来数据
    if len(gp) < min_history_days + 20:
        return pd.DataFrame()

    # 🔥 关键修改：使用T-1日的数据作为特征，避免未来函数（与训练时保持一致）
    close = gp['close'].shift(1)
    high  = gp['high'].shift(1)
    low   = gp['low'].shift(1)
    open = gp['open'].shift(1)
    vol   = gp['volume'].shift(1)
    qv    = gp['quote_volume'].shift(1) if 'quote_volume' in gp.columns else None
    tbq   = gp['taker_buy_quote'].shift(1) if 'taker_buy_quote' in gp.columns else None
    tbb   = gp['taker_buy_base'].shift(1) if 'taker_buy_base' in gp.columns else None
    trades_count = gp['trades_count'].shift(1) if 'trades_count' in gp.columns else None

    feat = pd.DataFrame({
        'date': gp['date'],
        'symbol': gp['symbol']
    })
    
    # ========== 精简版特征（use_improved_features=True）==========
    if use_improved_features:
        # 1. 基础价格动量特征（减少窗口数量）
        for window in [5, 10, 20]:
            # 价格动量
            feat[f'momentum_{window}'] = close.pct_change(window)
            # 波动率调整动量
            volatility = close.pct_change().rolling(window).std()
            feat[f'momentum_vol_adj_{window}'] = feat[f'momentum_{window}'] / (volatility + eps)
            
            # 高低波动特征
            high_low_ratio = (high / low - 1).rolling(window).mean()
            feat[f'range_efficiency_{window}'] = feat[f'momentum_{window}'] / (high_low_ratio + eps)
        
        # 2. 量价关系特征
        for window in [5, 10]:
            # 量价背离
            price_change = close.pct_change(window)
            volume_change = vol.pct_change(window)
            feat[f'price_volume_divergence_{window}'] = price_change - volume_change
            
            # 成交量加权价格
            vwap = (vol * (high + low + close) / 3).rolling(window).sum() / (vol.rolling(window).sum() + eps)
            feat[f'vwap_premium_{window}'] = (close - vwap) / close
        
        # 3. 均值回复特征
        for window in [10, 20]:
            # 价格与均线的偏离
            ma = close.rolling(window).mean()
            feat[f'ma_deviation_{window}'] = (close - ma) / ma
            
            # RSI改进版本
            returns = close.pct_change()
            gain = returns.clip(lower=0).rolling(window).mean()
            loss = (-returns.clip(upper=0)).rolling(window).mean()
            rs = gain / (loss + eps)
            feat[f'rsi_{window}'] = 100 - 100 / (1 + rs)
        
        # 4. 市场微观结构特征
        if tbq is not None and qv is not None:
            for window in [5, 10]:
                # 资金流强度
                money_flow = (tbq - (qv - tbq)) / (qv + eps)
                feat[f'money_flow_{window}'] = money_flow.rolling(window).mean()
                
                # 大单净流入
                large_order_ratio = (tbq / (qv + eps)).rolling(window).mean()
                feat[f'large_order_ratio_{window}'] = large_order_ratio
        
        # 5. 技术指标特征
        # MACD信号
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        feat['macd_signal'] = macd.ewm(span=9).mean()
        
        # 布林带位置
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        feat['bollinger_position_20'] = (close - bb_middle) / (2 * bb_std + eps)
        
        # 推理时不需要future_ret，也不丢弃最后几行
        # 只丢弃前面不完整的行
    feat = feat.iloc[min_history_days:].dropna().reset_index(drop=True)
    if feat.empty:
        return pd.DataFrame()
        return feat.iloc[[-1]]  # 返回最新一行
    
    # ========== 完整版特征（use_improved_features=False，基于feature.txt）==========
    
    # 基础价格特征（已经通过shift(1)处理，使用T-1日数据）
    feat['open']   = open
    feat['high']   = high
    feat['low']    = low
    feat['close']  = close
    feat['volume'] = vol

    if qv is not None:
        feat['quote_volume'] = qv
        feat['vwap'] = qv / (vol + eps)
    else:
        feat['vwap'] = close

    # ===== 特征1: 累积主买/累积主卖比率因子 =====
    if tbq is not None and qv is not None:
        tbs = qv - tbq  # 主动卖盘
        # 收盘为涨的日期：累积主动买盘
        up_days = close > close.shift(1)
        cum_buy_up = (tbq * up_days).rolling(window=20, min_periods=1).sum()
        cum_sell_down = (tbs * (~up_days)).rolling(window=20, min_periods=1).sum()
        feat['cum_buy_sell_ratio'] = cum_buy_up / (cum_sell_down + eps)

    # ===== 特征2: 买入稳定因子 =====
    if tbq is not None and qv is not None:
        buy_ratio = tbq / (qv + eps)
        for n in (5, 10, 20, 30):
            buy_ratio_mean = buy_ratio.rolling(n).mean()
            buy_ratio_std = buy_ratio.rolling(n).std()
            feat[f'buy_stability_{n}'] = buy_ratio_mean / (buy_ratio_std + eps)

    # ===== 特征3: 方向动量因子 =====
    for window in (5, 10, 20, 30):
        up_momentum = (close.diff().clip(lower=0)).rolling(window).sum()
        down_momentum = ((-close.diff()).clip(lower=0)).rolling(window).sum()
        feat[f'directional_momentum_{window}'] = up_momentum / (down_momentum + eps)

    # ===== 特征4: 定向波动率因子 =====
    for window in (5, 10, 20, 30):
        up_volatility = ((high - open) / (open + eps)).rolling(window).mean()
        down_volatility = ((open - low) / (open + eps)).rolling(window).mean()
        up_roc = (high - open.shift(window)) / (open.shift(window) + eps)
        down_roc = (open.shift(window) - low) / (open.shift(window) + eps)
        feat[f'directional_volatility_{window}'] = (up_roc / (up_volatility + eps)) - (down_roc / (down_volatility + eps))

    # ===== 特征5: 高波动性动量 =====
    for window in (5, 10, 20, 30):
        roc = close.pct_change(window)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        range_pct = (high - low) / (close + eps)
        feat[f'hv_momentum_product_{window}'] = roc * atr
        feat[f'hv_momentum_ratio_{window}'] = roc / (range_pct + eps)

    # ===== 特征6: 动量-主动买入极值 =====
    if tbq is not None:
        for window in (10, 20, 30):
            momentum = close.pct_change(window)
            # 简化实现：使用滚动窗口内的相关性或分位数
            tbq_rank = tbq.rolling(window).rank(pct=True)
            feat[f'momentum_tbq_extreme_{window}'] = momentum * tbq_rank

    # ===== 特征7: 动量-成交量极值分组因子 =====
    for window in (10, 20, 30):
        momentum = close.pct_change(window)
        vol_rank = vol.rolling(window).rank(pct=True)
        feat[f'momentum_volume_extreme_{window}'] = momentum * vol_rank

    # ===== 特征8: net_order_flow =====
    if tbq is not None and qv is not None:
        for window in (5, 10, 20, 30):
            daily_flow = tbq - (qv - tbq)  # taker_buy_quote - taker_sell_quote
            net_flow = daily_flow.rolling(window).sum() / (qv.rolling(window).sum() + eps)
            feat[f'net_order_flow_{window}'] = net_flow

    # ===== 特征9: 涨跌距离/路径（价格路径效率） =====
    for window in (5, 10, 20, 30):
        net_ret = close.pct_change(window)
        path_len = close.pct_change().abs().rolling(window).sum()
        feat[f'price_path_efficiency_{window}'] = net_ret / (path_len + eps)

    # ===== 特征10: rolling_corr_buyquote =====
    if tbq is not None:
        for window in (5, 10, 20, 30):
            ret = np.log(close / close.shift(1))
            corr = ret.rolling(window).corr(tbq)
            vol_mean = vol.rolling(window).mean()
            feat[f'rolling_corr_buyquote_{window}'] = corr * np.log(1 + vol_mean)

    # ===== 特征11: RSI 截面排序因子 =====
    for period in (14, 21, 30):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + eps)
        rsi = 100 - 100 / (1 + rs)
        feat[f'rsi_{period}'] = rsi

    # ===== 特征12: TBQ极值效率 =====
    if tbq is not None:
        for window in (10, 20, 30):
            roc = close.pct_change(window)
            range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
            vol_eff = roc / (range_pct + eps)
            tbq_rank = tbq.rolling(window).rank(pct=True)
            feat[f'tbq_extreme_efficiency_{window}'] = vol_eff * tbq_rank

    # ===== 特征13: Top-Bottom Trades Ratio =====
    if trades_count is not None and tbq is not None and qv is not None:
        for window in (10, 20, 30):
            buy_ratio = tbq / (qv + eps)
            trades_rank = trades_count.rolling(window).rank(pct=True)
            feat[f'top_bottom_trades_ratio_{window}'] = buy_ratio * trades_rank

    # ===== 特征14: 波动效率 =====
    for window in (5, 10, 20, 30):
        roc = close.pct_change(window)
        range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
        feat[f'volatility_efficiency_{window}'] = roc / (range_pct + eps)

    # ===== 特征15: 波动率-订单流 =====
    if trades_count is not None and qv is not None:
        for window in (5, 10, 20, 30):
            roc = close.pct_change(window)
            range_pct = ((high - low) / (close.shift(1) + eps)).rolling(window).mean()
            vol_eff = roc / (range_pct + eps)
            avg_trade_size = qv / (trades_count + eps)
            trades_ratio = trades_count / (trades_count.rolling(window).mean() + eps)
            avg_trade_size_ratio = avg_trade_size / (avg_trade_size.rolling(window).mean() + eps)
            order_flow = trades_ratio / (avg_trade_size_ratio + eps)
            feat[f'volatility_order_flow_{window}'] = vol_eff * order_flow

    # ===== 特征16: 量稳因子（成交量稳定性） =====
    if trades_count is not None:
        for lookback_days in (10, 20, 30):
            volume_mean = vol.rolling(lookback_days).mean()
            volume_std = vol.rolling(lookback_days).std()
            feat[f'volume_stability_{lookback_days}'] = volume_mean / (volume_std + eps)

    # ===== 特征17: 成交量切分-主动买入Base 极值 =====
    if tbb is not None:
        for window in (10, 20, 30):
            tb_base_mom = tbb.rolling(10).sum()
            vol_rank = vol.rolling(window).rank(pct=True)
            feat[f'volume_tbb_extreme_{window}'] = tb_base_mom * vol_rank

    # 推理时不需要future_ret，也不丢弃最后几行
    # 只丢弃前面不完整的行
    feat = feat.iloc[min_history_days:].dropna().reset_index(drop=True)
    if feat.empty:
        return pd.DataFrame()
    return feat.iloc[[-1]]  # 返回最新一行

def print_latest_daily_groups_live(window: int, rebalance_period: int, min_history_days: int = 60, 
                                   groups: int = 5, use_improved_features: bool = False,
                                   use_robust_preprocessing: bool = False):
    """
    使用已训练的模型，对原始K线的"全局最新日期"做一次前向预测并分组打印。
    类似 Xgboost_Prediction_original.py 中的实现，但适配新的代码结构。
    只使用每日前50个币种。
    """
    from util_factor import load_kline_df, build_available_tokens_by_date_from_kline
    
    # 读取数据与工件
    df = load_kline_df()
    
    # 🔥 关键：基于K线数据构建每日前50排名，只使用前50个币种
    print(f"📊 Live推理：构建每日前50排名，只使用前50个币种...")
    available_tokens_by_date = build_available_tokens_by_date_from_kline(
        kline_df=df,
        top_n=50,
        ranking_method='quote_volume',
        rebalance_period=rebalance_period,
        strict_top_n=True  # 严格限制为50个币种
    )
    
    model_path, selector_path, scaler_path = _artifact_paths(window, rebalance_period)
    if not (model_path.exists() and selector_path.exists() and scaler_path.exists()):
        print("未找到已训练模型/预处理器，跳过live日度分组打印。")
        return

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    selector = joblib.load(selector_path)
    scaler = joblib.load(scaler_path)

    # 获取最新日期（确保类型一致）
    latest_date = df['date'].max()
    
    # 获取最新日期的前50个币种（尝试精确匹配，如果失败则找最接近的日期）
    latest_top50 = available_tokens_by_date.get(latest_date, [])
    if not latest_top50:
        # 如果精确匹配失败，找最接近的日期
        available_dates = sorted(available_tokens_by_date.keys())
        if available_dates:
            # 找到小于等于latest_date的最大日期
            closest_date = None
            for d in reversed(available_dates):
                if d <= latest_date:
                    closest_date = d
                    break
            if closest_date:
                latest_top50 = available_tokens_by_date[closest_date]
                latest_date = closest_date  # 使用找到的日期
                print(f"注意: 使用最接近的日期 {latest_date.date()} 的前50排名")
    
    if not latest_top50:
        print(f"最新日期 {latest_date.date()} 没有前50排名数据。")
        return
    
    print(f"最新日期 {latest_date.date()} 的前50个币种: {len(latest_top50)} 个")

    # 只为前50个币种构造"最新一行"特征
    feats = []
    for sym in latest_top50:
        gp = df[df['symbol'] == sym]
        if gp.empty:
            continue
        row = create_features_symbol_inference_latest(
            gp.copy(), 
            min_history_days=min_history_days,
            use_improved_features=use_improved_features
        )
        if row is not None and not row.empty:
            feats.append(row)
    if not feats:
        print("无法构造最新日度特征，跳过live分组打印。")
        return

    feat_df = pd.concat(feats, ignore_index=True)
    day = feat_df[pd.to_datetime(feat_df['date']) == latest_date].copy()
    if day.empty:
        print("最新日期无可推理样本。")
        return

    # 横截面标准化（与训练时保持一致）
    feature_cols = [col for col in day.columns if col not in ['date', 'symbol']]
    
    if use_robust_preprocessing:
        # 使用稳健标准化（中位数和MAD）
        def robust_standardize_by_date(group):
            """对每个日期的特征进行稳健标准化（使用中位数和MAD）"""
            group = group.copy()
            for col in feature_cols:
                median_val = group[col].median()
                mad_val = (group[col] - median_val).abs().median()
                if mad_val > 1e-8:
                    group[col] = (group[col] - median_val) / mad_val
                else:
                    group[col] = 0.0
            return group
        day = day.groupby('date', group_keys=False).apply(robust_standardize_by_date)
    else:
        # 使用标准标准化（均值和标准差）
        def standardize_by_date(group):
            """对每个日期的特征进行横截面标准化"""
            group = group.copy()
            for col in feature_cols:
                mean_val = group[col].mean()
                std_val = group[col].std()
                if std_val > 1e-8:
                    group[col] = (group[col] - mean_val) / std_val
                else:
                    group[col] = 0.0
            return group
        day = day.groupby('date', group_keys=False).apply(standardize_by_date)

    X = day.drop(columns=['date','symbol']).to_numpy(dtype=np.float32)
    X_sel = selector.transform(X)
    X_scl = scaler.transform(X_sel)
    preds = booster.predict(xgb.DMatrix(X_scl))

    out = pd.DataFrame({'instrument': day['symbol'].values, 'pred': preds})
    # 横截面秩值归一化到[-1,1]，并按百分位分组，展示更直观
    out['pct'] = out['pred'].rank(pct=True, method='average')
    out['score'] = 2 * out['pct'] - 1  # [-1, 1]

    try:
        out['group'] = pd.qcut(out['pct'], q=groups, labels=list(range(groups)), duplicates='drop')
    except Exception:
        bins = np.linspace(out['pct'].min() - 1e-9, out['pct'].max() + 1e-9, groups + 1)
        out['group'] = pd.cut(out['pct'], bins=bins, labels=list(range(groups)), include_lowest=True)
    out['group'] = out['group'].astype(int)

    print(f"\n==== {rebalance_period}日调仓视角 | 最新日期: {latest_date.date()} (Live 推理) ====")

    # CSV表头
    print("date,group,instrument,score,pred")

    for g in range(groups):
        sub = out[out['group'] == g].sort_values('score')
        for _, r in sub.iterrows():
            print(f"{latest_date.date()},{g},{r['instrument']},{r['score']:.4f},{r['pred']:.6f}")

def analyze_selected_features_correlation(X_selected, n_features, title="特征相关性分析",
                                         save_path=None, show_plot=True, feature_names=None):
    """
    分析并可视化选定特征的相关性

    Parameters:
    - X_selected: 选定的特征矩阵
    - n_features: 特征数量
    - title: 图表标题
    - save_path: 保存路径（可选）
    - show_plot: 是否显示图形
    - feature_names: 特征名称列表（可选）
    """
    print(f"\n🔍 {title}")
    print("=" * 60)

    # 创建特征名称
    if feature_names is None or len(feature_names) != n_features:
        # 如果没有提供特征名称或数量不匹配，使用默认编号
        if n_features <= 20:
            feature_names = [f'feature_{i+1}' for i in range(n_features)]
        else:
            # 对于大量特征，使用编号
            feature_names = [f'f{i+1}' for i in range(n_features)]

    # 计算相关性矩阵
    corr_matrix = np.corrcoef(X_selected.T)

    # 相关性统计信息
    corr_flat = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    corr_abs = np.abs(corr_flat)

    print("📊 相关性统计:")
    print(f"   特征数量: {n_features}")
    print(f"   平均相关性: {corr_abs.mean():.6f}")
    print(f"   中位数相关性: {np.median(corr_abs):.6f}")
    print(f"   最大相关性: {corr_abs.max():.6f}")
    print(f"   最小相关性: {corr_abs.min():.6f}")
    print(f"   相关性标准差: {corr_abs.std():.6f}")
    # 找出高度相关的特征对
    high_corr_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(corr_matrix[i, j]) >= 0.7:  # 相关性阈值
                high_corr_pairs.append((i, j, corr_matrix[i, j]))

    if high_corr_pairs:
        print(f"\n⚠️  高度相关特征对 (|corr| >= 0.7): {len(high_corr_pairs)} 对")
        for i, j, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:  # 只显示前10个
            print(f"      f{i+1} ↔ f{j+1}: {corr:.4f}")
    else:
        print("\n✅ 无高度相关特征对 (|corr| >= 0.7)")

    # 可视化相关性矩阵
    if VISUALIZATION_AVAILABLE and show_plot:
        try:
            if n_features <= 30:  # 小型特征集：完整热力图
                plt.figure(figsize=(max(10, min(20, n_features * 0.5)), max(8, min(18, n_features * 0.45))))

                # 创建遮罩，只显示上三角
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

                # 绘制热力图
                sns.heatmap(corr_matrix,
                           mask=mask,
                           annot=n_features <= 12,  # 只有很少特征时显示数值
                           cmap='coolwarm',
                           vmin=-1, vmax=1,
                           center=0,
                           square=True,
                           xticklabels=feature_names,
                           yticklabels=feature_names,
                           cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                           linewidths=0.5 if n_features <= 15 else 0)

                plt.title(title, fontsize=14, pad=20)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()

            elif n_features <= 100:  # 中型特征集：简化热力图
                plt.figure(figsize=(12, 10))

                # 只显示相关性绝对值大于阈值的部分
                high_corr_mask = np.abs(corr_matrix) >= 0.5
                corr_display = corr_matrix * high_corr_mask

                sns.heatmap(corr_display,
                           cmap='coolwarm',
                           vmin=-1, vmax=1,
                           center=0,
                           square=False,
                           xticklabels=False,
                           yticklabels=False,
                           cbar_kws={'shrink': 0.8, 'label': 'Correlation (|corr| >= 0.5)'})

                plt.title(f"{title} (|corr| >= 0.5)", fontsize=14, pad=20)
                plt.tight_layout()

                print("ℹ️  特征数量较多，只显示高相关性部分 (|corr| >= 0.5)")

            else:  # 大型特征集：相关性分布图
                plt.figure(figsize=(12, 8))

                # 相关性绝对值的分布
                plt.hist(corr_abs, bins=50, alpha=0.7, edgecolor='black', density=True)
                plt.xlabel('Absolute Correlation')
                plt.ylabel('Density')
                plt.title(f"{title} - Correlation Distribution", fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.axvline(0.7, color='red', linestyle='--', label='High correlation threshold (0.7)')
                plt.axvline(corr_abs.mean(), color='blue', linestyle='--',
                           label='.3f')
                plt.legend()

                print("ℹ️  特征数量过多，显示相关性分布而非热力图")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 相关性分析图表已保存: {save_path}")

            plt.show()

        except Exception as e:
            print(f"⚠️  绘图失败: {e}")
            print("   将显示文本形式的相关性摘要")

    # 对于大型特征集，提供文本摘要
    if n_features > 30 and not (VISUALIZATION_AVAILABLE and show_plot):
        print("ℹ️  特征数量较多，提供相关性分布摘要:")

        # 相关性分布统计
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        hist, _ = np.histogram(corr_abs, bins=bins)
        print("   相关性分布:")
        for i in range(len(bins)-1):
            pct = hist[i] / len(corr_abs) * 100
            print(".1f")
        # 显示最相关的特征对
        if len(high_corr_pairs) > 0:
            print(f"\n🔗 最相关的10个特征对:")
            for i, j, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
                print(".4f")

    print("=" * 60)

def analyze_feature_importance(X_train, y_train, n_features, title="特征重要性分析",
                              save_path=None, show_plot=True, top_n=20, feature_names=None):
    """
    分析并可视化特征重要性

    Parameters:
    - X_train: 训练特征
    - y_train: 训练目标
    - n_features: 特征数量
    - title: 图表标题
    - save_path: 保存路径（可选）
    - show_plot: 是否显示图形
    - top_n: 显示前N个最重要的特征
    - feature_names: 特征名称列表（可选）
    """
    print(f"\n📈 {title}")
    print("=" * 60)

    # 创建特征名称
    if feature_names is None or len(feature_names) != n_features:
        # 如果没有提供特征名称或数量不匹配，使用默认编号
        if n_features <= top_n:
            feature_names = [f'feature_{i+1}' for i in range(n_features)]
        else:
            feature_names = [f'f{i+1}' for i in range(n_features)]

    try:
        # 训练临时模型获取特征重要性
        temp_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )

        temp_model.fit(X_train, y_train)
        importances = temp_model.feature_importances_

        # 创建重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # 重要性统计
        print("📊 特征重要性统计:")
        print(f"   总特征数: {n_features}")
        print(f"   平均重要性: {importances.mean():.6f}")
        print(f"   最大重要性: {importances.max():.6f}")
        print(f"   最小重要性: {importances.min():.6f}")
        print(f"   重要性标准差: {importances.std():.6f}")
        # 显示前N个重要特征
        display_n = min(top_n, n_features)
        print(f"\n🏆 前{display_n}个最重要的特征:")
        for i, row in importance_df.head(display_n).iterrows():
            rank = i + 1
            print(f"  #{rank:2d} {row['feature']}: {row['importance']:.6f}")
        # 重要性分布分析
        importance_df['importance_pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100
        cumulative_pct = importance_df['importance_pct'].cumsum()

        # 找到解释80%重要性的特征数量
        n_80pct = (cumulative_pct >= 80).idxmax() + 1
        print(f"   80%重要性所需特征数: {n_80pct} (占总数的{n_80pct/n_features*100:.1f}%)")
        # 可视化
        if VISUALIZATION_AVAILABLE and show_plot:
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # 特征重要性条形图
                top_features = importance_df.head(display_n)
                ax1.barh(range(len(top_features)), top_features['importance'][::-1])
                ax1.set_yticks(range(len(top_features)))
                ax1.set_yticklabels(top_features['feature'][::-1])
                ax1.set_xlabel('Importance')
                ax1.set_title(f'Top {display_n} Feature Importances')
                ax1.grid(True, alpha=0.3)

                # 重要性分布直方图
                ax2.hist(importances, bins=20, alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Importance')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Feature Importance Distribution')
                ax2.grid(True, alpha=0.3)
                ax2.axvline(importances.mean(), color='red', linestyle='--',
                           label='.4f')
                ax2.legend()

                plt.suptitle(title, fontsize=14)
                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"💾 特征重要性图表已保存: {save_path}")

                plt.show()

            except Exception as e:
                print(f"⚠️  绘图失败: {e}")

        return importance_df

    except Exception as e:
        print(f"❌ 特征重要性分析失败: {e}")
        return None

if __name__ == "__main__":
    # 使用基于feature.txt的特征和贝叶斯优化，并启用特征正交化
    # 直接预测收益率（而非排名）
    create_xgboost_prediction_factor(
        window=10,
        rebalance_period=10,
        train_epochs=3000,
        min_history_days=60,
        min_samples_per_symbol=50,
        use_bayesian_opt=True,              # 使用贝叶斯优化
        use_orthogonalization=True,         # 启用特征正交化（去除高度相关的特征）
        correlation_threshold=0.60,        # 相关性阈值
        use_pca=False,                     # 不使用PCA（保持特征可解释性）
        # 可选：启用以下改进功能
        use_improved_features=False,       # 使用改进的特征工程
        use_robust_training=True,        # 启用稳健训练策略
        use_conservative_selection=True, # 启用保守特征选择
        max_features=30,                 # 保守特征选择的最大特征数量
        use_robust_preprocessing=True,    # 启用稳健数据预处理
        use_rank_target=True,            # False=预测收益率（MAE），True=预测排名
        regularization_strength='ultra'  # 正则化强度：'ultra'=超强, 'enhanced'=增强, 'default'=默认
    )