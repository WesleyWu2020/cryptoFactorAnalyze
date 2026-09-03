# factor_analyse/factor_mining/GRU_Prediction_Factor.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 深度学习相关导入
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    TORCH_AVAILABLE = True
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        DEVICE = torch.device('cpu')
        print("⚠️ GPU不可用，使用CPU训练")
        
except ImportError:
    print("警告: PyTorch未安装，将使用简化的线性模型替代")
    TORCH_AVAILABLE = False
    DEVICE = None
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

from util_factor import (
    load_historical_marketcap, build_available_tokens_by_date, load_kline_df,
    filter_group_by_availability, group_apply_with_progress, winsorize_by_date,
    rank_to_unit_by_date, save_factor_df, future_return, print_availability_sample,
    print_factor_summary
)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict
from tqdm import tqdm # Added for progress bar

class MultiSymbolGRUDataset(Dataset):
    """多symbol联合训练的GRU数据集类"""
    def __init__(self, sequences, targets, symbols, dates, weights=None):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.symbols = symbols
        self.dates = dates
        self.weights = None if weights is None else torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.weights is None:
            return self.sequences[idx], self.targets[idx], self.symbols[idx], self.dates[idx]
        return self.sequences[idx], self.targets[idx], self.symbols[idx], self.dates[idx], self.weights[idx]

def collate_fn(batch):
    # 支持 (X,y,symbol,date) 或 (X,y,symbol,date,weight)
    if len(batch[0]) == 5:
        X, y, symbols, dates, weights = zip(*batch)
        X = torch.stack(X)          # [B, T, F]
        y = torch.stack(y)          # [B]
        weights = torch.stack(weights)
        return X, y, list(symbols), list(dates), weights
    else:
        X, y, symbols, dates = zip(*batch)
        X = torch.stack(X)
        y = torch.stack(y)
        return X, y, list(symbols), list(dates)

# 替换原 MultiSymbolGRUPredictor
class ImprovedGRUPredictor(nn.Module):
    """BiGRU + Temporal Attention + LN + Residual MLP + Symbol Embedding Gating"""
    def __init__(self, input_size, hidden_size=96, num_layers=2, dropout=0.2, num_symbols=None):
        super().__init__()
        self.num_symbols = num_symbols
        self.hidden_size = hidden_size

        # 双向GRU，增强时序表征
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout, bidirectional=True)

        # 时间注意力池化（对所有step加权平均，而非只取最后一步）
        self.attn_vec = nn.Linear(hidden_size * 2, 1, bias=False)

        # 符号embedding + 规范化
        if num_symbols is not None:
            self.symbol_embedding = nn.Embedding(num_symbols, hidden_size // 2)
            self.sym_ln = nn.LayerNorm(hidden_size // 2)
            # 门控融合符号信息
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2 + hidden_size // 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            fc_in = hidden_size
        else:
            fc_in = hidden_size * 2

        # LayerNorm + 残差MLP头
        self.pre_ln = nn.LayerNorm(fc_in)
        self.mlp = nn.Sequential(
            nn.Linear(fc_in, fc_in),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(fc_in // 2, 1)

    def temporal_attention(self, H):
        # H: [B, T, 2H]; attn: [B, T, 1]; 加权平均 -> [B, 2H]
        attn = torch.softmax(self.attn_vec(H).squeeze(-1), dim=1)  # [B, T]
        pooled = torch.bmm(attn.unsqueeze(1), H).squeeze(1)        # [B, 2H]
        return pooled

    def forward(self, x, symbol_ids=None):
        # x: [B, T, F]
        H, _ = self.gru(x)                    # [B, T, 2H]
        pooled = self.temporal_attention(H)   # [B, 2H]

        if self.num_symbols is not None and symbol_ids is not None:
            sym = self.symbol_embedding(symbol_ids)                 # [B, H/2]
            sym = self.sym_ln(sym)
            fused = torch.cat([pooled, sym], dim=1)                # [B, 2H + H/2]
            pooled = self.gate(fused)                               # [B, H]
        # 规范化 + 残差
        z = self.pre_ln(pooled)
        h = self.mlp(z)
        out = self.out(h).squeeze(-1)
        return out

def create_technical_features(df):
    """创建技术指标特征 - 专注收益率预测"""
    eps = 1e-8
    features = pd.DataFrame(index=df.index)

    # 基础价量
    features["open"] = df["open"]
    features["high"] = df["high"]
    features["low"] = df["low"]
    features["close"] = df["close"]
    features["volume"] = df["volume"]
    if "quote_volume" in df.columns:
        features["quote_volume"] = df["quote_volume"]

    # 计算VWAP
    if "vwap" in df.columns:
        vwap = df["vwap"]
    else:
        vwap = df.get("quote_volume", pd.Series(index=df.index, dtype=float)) / (df["volume"] + eps)
    features["vwap"] = vwap

    # 1) 价格动量特征
    for n in [1, 3, 5, 10, 20]:
        features[f"ret_{n}"] = df["close"].pct_change(n)  # 简单收益率
        features[f"logret_{n}"] = np.log(df["close"] / df["close"].shift(n))  # 对数收益率
        features[f"volume_ret_{n}"] = df["volume"].pct_change(n)  # 成交量变化率

    # 2) 移动平均线特征
    for n in [5, 10, 20, 30]:
        features[f"sma_{n}"] = df["close"].rolling(n).mean()
        features[f"ema_{n}"] = df["close"].ewm(span=n).mean()
        features[f"price_sma_ratio_{n}"] = df["close"] / (features[f"sma_{n}"] + eps)
        features[f"price_ema_ratio_{n}"] = df["close"] / (features[f"ema_{n}"] + eps)

    # 3) 波动率特征
    for n in [5, 10, 20]:
        features[f"volatility_{n}"] = features["logret_1"].rolling(n).std()
        features[f"high_low_ratio_{n}"] = (df["high"] - df["low"]) / (df["close"] + eps)
        features[f"close_position_{n}"] = (df["close"] - df["low"].rolling(n).min()) / (df["high"].rolling(n).max() - df["low"].rolling(n).min() + eps)

    # 4) 成交量特征
    for n in [5, 10, 20]:
        features[f"volume_sma_{n}"] = df["volume"].rolling(n).mean()
        features[f"volume_ratio_{n}"] = df["volume"] / (features[f"volume_sma_{n}"] + eps)
        if "quote_volume" in df.columns:
            features[f"quote_volume_sma_{n}"] = df["quote_volume"].rolling(n).mean()
            features[f"quote_volume_ratio_{n}"] = df["quote_volume"] / (features[f"quote_volume_sma_{n}"] + eps)

    # 5) RSI指标
    for period in [14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + eps)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # 6) MACD指标
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    features['macd'] = ema_12 - ema_26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']

    # 7) 布林带
    for period in [20, 30]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        features[f'bb_upper_{period}'] = sma + (std * 2)
        features[f'bb_lower_{period}'] = sma - (std * 2)
        features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / (sma + eps)
        features[f'bb_position_{period}'] = (df['close'] - features[f'bb_lower_{period}']) / ((features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) + eps)

    # 8) 主动买入相关特征
    if "taker_buy_quote" in df.columns:
        buyq = df["taker_buy_quote"]
        if "quote_volume" in df.columns:
            buy_ratio = buyq / (df["quote_volume"] + eps)
            features["buy_ratio"] = buy_ratio
            for n in [5, 10, 20]:
                features[f"buy_ratio_sma_{n}"] = buy_ratio.rolling(n).mean()
                features[f"buy_ratio_std_{n}"] = buy_ratio.rolling(n).std()

        for n in [5, 10, 20]:
            features[f"buyq_sma_{n}"] = buyq.rolling(n).mean()
            features[f"buyq_std_{n}"] = buyq.rolling(n).std()
            features[f"buyq_roc_{n}"] = buyq.pct_change(n)

    # 4) Alpha参考构件（取核心构件，便于模型学习）
    # 4.1 Alpha41：((high * low)^0.5 - vwap)
    features["alpha41_geo_vwap"] = np.sqrt(df["high"] * df["low"]) - vwap

    # 4.2 Alpha5 的两个核心分量（不做横截面rank与乘积，在模型里学习组合）
    vwap_10 = vwap.rolling(10).mean()
    features["alpha5_open_minus_vwap10"] = df["open"] - vwap_10
    features["alpha5_neg_abs_close_minus_vwap"] = -np.abs(df["close"] - vwap)

    # # 5) 现有你已有的一些基础特征（保留原来）
    # # 5.1 价格变化率
    # for period in [1, 3, 5, 10, 15]:
    #     features[f'roc_{period}'] = df['close'].pct_change(period)
    #     features[f'volume_roc_{period}'] = df['volume'].pct_change(period)

    # # 5.2 均线与价格相对位置
    # for period in [5, 10, 20, 30]:
    #     features[f'sma_{period}'] = df['close'].rolling(period).mean()
    #     features[f'price_sma_ratio_{period}'] = df['close'] / (features[f'sma_{period}'] + eps)
    #     features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    #     features[f'price_ema_ratio_{period}'] = df['close'] / (features[f'ema_{period}'] + eps)

    # # 5.3 价格位置（区间归一）
    # for period in [10, 20, 30]:
    #     hh = df['high'].rolling(period).max()
    #     ll = df['low'].rolling(period).min()
    #     rng = (hh - ll).replace(0, np.nan)
    #     features[f'high_low_ratio_{period}'] = (hh - ll) / (df['close'] + eps)
    #     features[f'close_position_{period}'] = (df['close'] - ll) / (rng + eps)
    #     features[f'high_position_{period}'] = (df['high'] - ll) / (rng + eps)

    # # 5.4 成交量派生
    # features['volume_sma_5'] = df['volume'].rolling(5).mean()
    # features['volume_sma_20'] = df['volume'].rolling(20).mean()
    # features['volume_ratio_5'] = df['volume'] / (features['volume_sma_5'] + eps)
    # features['volume_ratio_20'] = df['volume'] / (features['volume_sma_20'] + eps)

    # # 5.5 RSI
    # for period in [14, 21]:
    #     delta = df['close'].diff()
    #     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    #     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    #     rs = gain / (loss + eps)
    #     features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # # 5.6 MACD
    # ema_12 = df['close'].ewm(span=12).mean()
    # ema_26 = df['close'].ewm(span=26).mean()
    # features['macd'] = ema_12 - ema_26
    # features['macd_signal'] = features['macd'].ewm(span=9).mean()
    # features['macd_histogram'] = features['macd'] - features['macd_signal']

    # # 5.7 布林带
    # for period in [20, 30]:
    #     sma = df['close'].rolling(period).mean()
    #     std = df['close'].rolling(period).std()
    #     features[f'bb_upper_{period}'] = sma + (std * 2)
    #     features[f'bb_lower_{period}'] = sma - (std * 2)
    #     features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / (sma + eps)
    #     features[f'bb_position_{period}'] = (df['close'] - features[f'bb_lower_{period}']) / ((features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) + eps)

    return features

def prepare_multi_symbol_data(df, sequence_length=30, rebalance_period=1):
    """准备多symbol联合训练数据 - 按时间分割"""
    print("🔄 准备多symbol联合训练数据...")
    
    all_sequences = []
    all_targets = []
    all_symbols = []
    all_dates = []
    
    # 获取所有symbol
    symbols = df['symbol'].unique()
    print(f"处理 {len(symbols)} 个symbols...")
    
    # 为每个symbol准备数据
    for symbol in symbols:
        symbol_data = df[df['symbol'] == symbol].copy().reset_index(drop=True)
        
        if len(symbol_data) < sequence_length + rebalance_period + 50:
            continue
            
        # 创建技术特征
        features_df = create_technical_features(symbol_data)
        
        # 计算未来收益率
        future_ret = future_return(symbol_data["close"], rebalance_period, method="log")
        
        # 准备训练数据
        valid_idx = ~(features_df.isnull().any(axis=1) | future_ret.isnull())
        features_clean = features_df[valid_idx]
        targets_clean = future_ret[valid_idx]
        
        if len(features_clean) < sequence_length + 20:
            continue
        
        # 准备序列数据
        sequences, target_values = prepare_sequences(features_clean, targets_clean, sequence_length)
        
        if len(sequences) < 50:
            continue
        
        # 添加symbol和date信息
        for i in range(len(sequences)):
            date_idx = features_clean.index[i + sequence_length]
            if date_idx in symbol_data.index:
                all_sequences.append(sequences[i])
                all_targets.append(target_values[i])
                all_symbols.append(symbol)
                all_dates.append(symbol_data.loc[date_idx, 'date'])
    
    print(f"✅ 总共收集到 {len(all_sequences)} 个训练样本")
    return np.array(all_sequences), np.array(all_targets), all_symbols, all_dates

def prepare_sequences(features, targets, sequence_length=20):
    """准备时间序列数据"""
    sequences = []
    target_values = []
    
    for i in range(sequence_length, len(features)):
        # 获取序列
        seq = features.iloc[i-sequence_length:i].values
        target = targets.iloc[i]
        
        # 检查是否有缺失值
        if not np.isnan(seq).any() and not np.isnan(target):
            sequences.append(seq)
            target_values.append(target)
    
    return np.array(sequences), np.array(target_values)

def fit_symbol_target_scaler(symbols, y):
    stats = {}
    df = pd.DataFrame({"symbol": symbols, "y": y})
    for s, g in df.groupby("symbol"):
        mu = g["y"].mean()
        sigma = g["y"].std()
        stats[s] = {"mu": float(mu), "sigma": float(sigma if sigma and sigma > 1e-8 else 1.0)}
    return stats

def transform_y(symbols, y, stats):
    mu = np.array([stats.get(s, {"mu": 0.0})["mu"] for s in symbols], dtype=np.float32)
    sg = np.array([stats.get(s, {"sigma": 1.0})["sigma"] for s in symbols], dtype=np.float32)
    return (y - mu) / sg, mu, sg

def inverse_transform(pred, mu, sg):
    return pred * sg + mu

def zscore_by_date(tensor, dates):
    # tensor: [B] on device; dates: list of python date
    out = tensor.clone()
    from collections import defaultdict
    idx_by_date = defaultdict(list)
    for i, d in enumerate(dates):
        idx_by_date[d].append(i)
    for _, idxs in idx_by_date.items():
        if len(idxs) < 2: 
            continue
        idx = torch.tensor(idxs, device=tensor.device, dtype=torch.long)
        mu = tensor[idx].mean()
        sg = tensor[idx].std().clamp_min(1e-6)
        out[idx] = (tensor[idx] - mu) / sg
    return out

def datewise_mean(values, dates):
    # values: [B]
    from collections import defaultdict
    vals = []
    by = defaultdict(list)
    for i, d in enumerate(dates):
        by[d].append(values[i])
    for _, v in by.items():
        v = torch.stack(v)
        vals.append(v.mean())
    if len(vals) == 0:
        return torch.tensor(0.0, device=values.device)
    return torch.stack(vals).mean()

def datewise_corr_loss(pred, target, dates, eps=1e-8):
    # 1 - Pearson corr per date, then均值
    from collections import defaultdict
    by = defaultdict(list)
    for i, d in enumerate(dates):
        by[d].append(i)
    losses = []
    for _, idxs in by.items():
        if len(idxs) < 2: 
            continue
        idx = torch.tensor(idxs, device=pred.device)
        x = pred[idx]; y = target[idx]
        x = x - x.mean(); y = y - y.mean()
        num = (x * y).sum()
        den = torch.sqrt((x.pow(2)).sum() + eps) * torch.sqrt((y.pow(2)).sum() + eps)
        losses.append(1.0 - num / den)
    if not losses:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(losses).mean()

def compute_sample_weights(symbols, stats, eps: float = 1e-6):
    w = np.array([1.0 / (eps + stats.get(s, {"sigma": 1.0})["sigma"]) for s in symbols], dtype=np.float32)
    return w / w.mean()

def train_multi_symbol_gru_model(X_train, y_train, X_val, y_val,
                                 symbols_train, symbols_val, dates_train, dates_val,
                                 input_size, epochs=100, batch_size=128,
                                 weights_train=None, weights_val=None,
                                 symbol_to_id=None):

    if symbol_to_id is None:
        unique_symbols = sorted(set(symbols_train + symbols_val))
        symbol_to_id = {"UNK": 0}
        symbol_to_id.update({s: i + 1 for i, s in enumerate(unique_symbols)})

    train_dataset = MultiSymbolGRUDataset(X_train, y_train, symbols_train, dates_train, weights=weights_train)
    val_dataset   = MultiSymbolGRUDataset(X_val,   y_val,   symbols_val,   dates_val,   weights=weights_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)

    model = ImprovedGRUPredictor(input_size=input_size, hidden_size=128, num_layers=2,
                                 dropout=0.2, num_symbols=len(symbol_to_id)).to(DEVICE)
    # 2) 换稳健损失 + 温和的优化与调度（训练函数里）
    criterion = nn.SmoothL1Loss(reduction='none')  # 替代 nn.MSELoss
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=5e-5)

    # 线性 warmup 5 个 epoch，再用 CosineAnnealing
    total_epochs = epochs
    warmup_epochs = 5
    def lr_lambda(cur_epoch):
        if cur_epoch < warmup_epochs:
            return float(cur_epoch + 1) / float(warmup_epochs)
        t = cur_epoch - warmup_epochs
        T = max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * t / T))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    early_stopping_patience = 40

    # 训练参数
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    drop_prob = 0.1  # 随机将部分symbol置为UNK，提升未知symbol鲁棒性

    print(f"🚀 开始多symbol联合GPU训练，设备: {DEVICE}")
    print(f"训练样本: {len(X_train)}, 验证样本: {len(X_val)}")
    print(f"批次大小: {batch_size}, 训练轮数: {epochs}")
    print(f"Symbol词表大小: {len(symbol_to_id)}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 5:
                batch_X, batch_y, batch_symbols, batch_dates, batch_w = batch
            else:
                batch_X, batch_y, batch_symbols, batch_dates = batch
                batch_w = torch.ones_like(batch_y)

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_w = batch_w.to(DEVICE)

            batch_ids = torch.LongTensor([symbol_to_id.get(s, 0) for s in batch_symbols]).to(DEVICE)
            if drop_prob > 0:
                mask = (torch.rand_like(batch_ids.float()) < drop_prob)
                batch_ids = torch.where(mask, torch.zeros_like(batch_ids), batch_ids)

            # 模型前向
            outputs = model(batch_X, batch_ids).squeeze()  # [B]
            
            # 简化损失：直接预测收益率，强化排序能力
            # 1) 基础回归损失
            mse_loss = torch.nn.functional.mse_loss(outputs, batch_y)
            
            # 2) 横截面排序损失（按日期分组）
            from collections import defaultdict
            idx_by_date = defaultdict(list)
            for i, d in enumerate(batch_dates):
                idx_by_date[d].append(i)
            
            rank_losses = []
            for _, idxs in idx_by_date.items():
                if len(idxs) < 3:
                    continue
                idx = torch.tensor(idxs, device=outputs.device)
                pred_slice = outputs[idx]
                target_slice = batch_y[idx]
                
                # 计算排序损失：预测排序 vs 真实排序
                pred_ranks = torch.argsort(torch.argsort(pred_slice, descending=True))
                target_ranks = torch.argsort(torch.argsort(target_slice, descending=True))
                rank_losses.append(torch.nn.functional.mse_loss(pred_ranks.float(), target_ranks.float()))
            
            rank_loss = torch.stack(rank_losses).mean() if rank_losses else torch.tensor(0.0, device=outputs.device)
            
            # 3) 相关性损失（按日期）
            corr_losses = []
            for _, idxs in idx_by_date.items():
                if len(idxs) < 2:
                    continue
                idx = torch.tensor(idxs, device=outputs.device)
                pred_slice = outputs[idx]
                target_slice = batch_y[idx]
                
                # 计算Pearson相关系数
                pred_centered = pred_slice - pred_slice.mean()
                target_centered = target_slice - target_slice.mean()
                corr = (pred_centered * target_centered).sum() / (torch.sqrt((pred_centered**2).sum() + 1e-8) * torch.sqrt((target_centered**2).sum() + 1e-8))
                corr_losses.append(1.0 - corr)  # 1 - corr，越小越好
            
            corr_loss = torch.stack(corr_losses).mean() if corr_losses else torch.tensor(0.0, device=outputs.device)
            
            # 组合损失：回归 + 排序 + 相关
            loss = 0.4 * mse_loss + 0.4 * rank_loss + 0.2 * corr_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 5:
                    batch_X, batch_y, batch_symbols, _, batch_w = batch
                else:
                    batch_X, batch_y, batch_symbols, _ = batch
                    batch_w = torch.ones_like(batch_y)
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                batch_w = batch_w.to(DEVICE)
                batch_ids = torch.LongTensor([symbol_to_id.get(s, 0) for s in batch_symbols]).to(DEVICE)
                outputs = model(batch_X, batch_ids).squeeze()
                loss_vec = criterion(outputs, batch_y)
                loss = (loss_vec * batch_w).mean()
                val_loss += loss.item()

        train_losses.append(train_loss / max(1, len(train_loader)))
        val_losses.append(val_loss / max(1, len(val_loader)))
        scheduler.step()
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= early_stopping_patience:
            print(f"早停触发，在第{epoch+1}轮停止训练")
            break
    # 训练结束后，若暂不做EMA，直接返回当前最优模型
    import copy
    ema_model = copy.deepcopy(model)  # 深拷贝，结构和参数完全一致
    model = ema_model
    return model, train_losses, val_losses, symbol_to_id

def analyze_symbol_availability_over_time(df):
    """分析币种数量随时间的变化"""
    print("📊 分析币种数量随时间变化...")
    
    # 按日期统计可用币种数量
    daily_symbols = df.groupby('date')['symbol'].nunique().reset_index()
    daily_symbols.columns = ['date', 'symbol_count']
    
    # 找出每个币种的上市日期
    symbol_start_dates = df.groupby('symbol')['date'].min().reset_index()
    symbol_start_dates.columns = ['symbol', 'start_date']
    
    print(f"币种数量统计:")
    print(f"  总币种数: {df['symbol'].nunique()}")
    print(f"  最少日币种数: {daily_symbols['symbol_count'].min()}")
    print(f"  最多日币种数: {daily_symbols['symbol_count'].max()}")
    print(f"  平均日币种数: {daily_symbols['symbol_count'].mean():.1f}")
    
    # 显示币种数量变化趋势
    print(f"\n币种数量变化趋势 (前10个日期):")
    print(daily_symbols.head(10).to_string(index=False))
    
    return daily_symbols, symbol_start_dates

def create_dynamic_availability_filter(df, symbol_start_dates, min_history_days=60):
    """创建动态可用性过滤器，确保币种有足够历史数据"""
    print(f"🔧 创建动态可用性过滤器 (最少历史: {min_history_days}天)...")
    
    # 为每个日期创建可用币种列表
    availability_by_date = {}
    unique_dates = sorted(df['date'].unique())
    
    for current_date in unique_dates:
        available_symbols = []
        for symbol in df['symbol'].unique():
            symbol_start = symbol_start_dates[symbol_start_dates['symbol'] == symbol]['start_date'].iloc[0]
            days_since_start = (current_date - symbol_start).days
            
            # 只有当币种有足够历史数据时才可用
            if days_since_start >= min_history_days:
                available_symbols.append(symbol)
        
        availability_by_date[current_date] = available_symbols
    
    # 统计可用币种数量变化
    date_counts = [(date, len(symbols)) for date, symbols in availability_by_date.items()]
    date_counts_df = pd.DataFrame(date_counts, columns=['date', 'available_count'])
    
    print(f"动态可用性统计:")
    print(f"  最少可用币种: {date_counts_df['available_count'].min()}")
    print(f"  最多可用币种: {date_counts_df['available_count'].max()}")
    print(f"  平均可用币种: {date_counts_df['available_count'].mean():.1f}")
    
    return availability_by_date

def prepare_multi_symbol_data_with_dynamic_filter(df, sequence_length=30, rebalance_period=1, 
                                                 min_history_days=60, min_samples_per_symbol=30):
    """准备多symbol联合训练数据 - 支持动态币种数量"""
    print("🔄 准备多symbol联合训练数据 (支持动态币种数量)...")
    
    # 分析币种可用性
    daily_symbols, symbol_start_dates = analyze_symbol_availability_over_time(df)
    availability_by_date = create_dynamic_availability_filter(df, symbol_start_dates, min_history_days)
    
    all_sequences = []
    all_targets = []
    all_symbols = []
    all_dates = []
    all_availability_counts = []  # 记录每个样本对应日期的可用币种数
    
    # 获取所有symbol
    symbols = df['symbol'].unique()
    print(f"处理 {len(symbols)} 个symbols...")
    
    # 为每个symbol准备数据
    for symbol in tqdm(symbols, desc="处理symbols"):
        symbol_data = df[df['symbol'] == symbol].copy().reset_index(drop=True)
        
        if len(symbol_data) < sequence_length + rebalance_period + 50:
            continue
            
        # 创建技术特征
        features_df = create_technical_features(symbol_data)
        
        # 计算未来收益率
        future_ret = future_return(symbol_data["close"], rebalance_period, method="log")
        
        # 准备训练数据
        valid_idx = ~(features_df.isnull().any(axis=1) | future_ret.isnull())
        features_clean = features_df[valid_idx]
        targets_clean = future_ret[valid_idx]
        
        if len(features_clean) < sequence_length + 20:
            continue
        
        # 准备序列数据
        sequences, target_values = prepare_sequences(features_clean, targets_clean, sequence_length)
        
        if len(sequences) < min_samples_per_symbol:
            continue
        
        # 添加symbol和date信息，同时检查可用性
        for i in range(len(sequences)):
            date_idx = features_clean.index[i + sequence_length]
            if date_idx in symbol_data.index:
                current_date = symbol_data.loc[date_idx, 'date']
                
                # 检查该日期该symbol是否可用
                if current_date in availability_by_date and symbol in availability_by_date[current_date]:
                    all_sequences.append(sequences[i])
                    all_targets.append(target_values[i])
                    all_symbols.append(symbol)
                    all_dates.append(current_date)
                    all_availability_counts.append(len(availability_by_date[current_date]))
    
    print(f"✅ 总共收集到 {len(all_sequences)} 个训练样本")
    print(f"  平均每个样本对应 {np.mean(all_availability_counts):.1f} 个可用币种")
    
    return np.array(all_sequences), np.array(all_targets), all_symbols, all_dates, all_availability_counts

def create_time_aware_split_with_last_days(data_with_dates, test_days=180, validation_ratio=0.15):
    """按时间划分：最后 test_days 天为测试集，之前样本按比例切分为训练/验证"""
    print("🔄 按时间分割数据（测试集=最近N天）...")
    # data_with_dates: list of (X, y, symbol, date)
    data_with_dates.sort(key=lambda x: x[3])
    dates = [item[3] for item in data_with_dates]
    min_date, max_date = min(dates), max(dates)
    cutoff_date = max_date - pd.Timedelta(days=test_days - 1)

    test_data = [it for it in data_with_dates if it[3] >= cutoff_date]
    earlier_data = [it for it in data_with_dates if it[3] < cutoff_date]

    total_earlier = len(earlier_data)
    val_split_idx = int(total_earlier * (1 - validation_ratio))
    train_data = earlier_data[:val_split_idx]
    val_data = earlier_data[val_split_idx:]

    print(f"数据时间范围: {min_date} ~ {max_date}")
    print(f"测试集使用最近 {test_days} 天（cutoff={cutoff_date.date()}）")
    print(f"📊 数据分割完成:")
    print(f"  训练集: {len(train_data)} 样本")
    print(f"  验证集: {len(val_data)} 样本")
    print(f"  测试集: {len(test_data)} 样本")
    return train_data, val_data, test_data

def create_adaptive_symbol_embedding(symbols_list, embedding_dim=32):
    """创建自适应的symbol embedding映射"""
    print("🔧 创建自适应symbol embedding...")
    
    # 统计symbol频次，用于确定重要性
    from collections import Counter
    symbol_counts = Counter(symbols_list)
    
    # 按频次排序，频次高的symbol获得更小的ID（更重要）
    sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 创建映射: UNK=0, 然后按重要性分配ID
    symbol_to_id = {"UNK": 0}
    for i, (symbol, count) in enumerate(sorted_symbols):
        symbol_to_id[symbol] = i + 1
    
    print(f"  总symbol数: {len(symbol_to_id)-1}")
    print(f"  Top 10 symbols: {[s for s, _ in sorted_symbols[:10]]}")
    
    return symbol_to_id, len(symbol_to_id)

def create_multi_symbol_gru_prediction_factor(window=20, rebalance_period=1, sequence_length=30, 
                                             availability_lookback_days=90, train_epochs=100,
                                             min_history_days=60, min_samples_per_symbol=30):
    """
    基于多symbol联合训练的GRU收益率预测因子 - 适配动态币种数量版本
    
    新增参数:
    - min_history_days: 币种需要的最少历史数据天数
    - min_samples_per_symbol: 每个币种需要的最少样本数
    """
    print(f"开始构建多symbol联合GRU预测因子 - 动态币种数量适配版本")
    print(f"参数: window={window}, rebalance_period={rebalance_period}, sequence_length={sequence_length}")
    print(f"      min_history_days={min_history_days}, min_samples_per_symbol={min_samples_per_symbol}")
    
    if not TORCH_AVAILABLE:
        print("PyTorch不可用，无法进行深度学习训练")
        return pd.DataFrame()
    
    # K线数据
    df = load_kline_df()
    
    # 准备多symbol联合训练数据 - 支持动态币种数量
    all_sequences, all_targets, all_symbols, all_dates, all_availability_counts = prepare_multi_symbol_data_with_dynamic_filter(
        df, sequence_length, rebalance_period, min_history_days, min_samples_per_symbol
    )
    
    if len(all_sequences) < 1000:
        print("警告: 训练数据不足，需要至少1000个样本")
        return pd.DataFrame()
    
    # 时间感知的数据分割
    data_with_dates = list(zip(all_sequences, all_targets, all_symbols, all_dates))
    train_data, val_data, test_data = create_time_aware_split_with_last_days(
        data_with_dates, test_days=180, validation_ratio=0.15
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
    
    # 标准化特征 - 只使用训练数据
    print("🔄 标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # 创建自适应symbol embedding
    symbol_to_id, num_symbols = create_adaptive_symbol_embedding(symbols_train + symbols_val + symbols_test)
    
    # 简化目标处理：直接使用原始收益率，不做复杂的标准化
    # 只做简单的异常值裁剪
    def clip_extreme_values(y, clip_percentile=99):
        """裁剪极端值"""
        lower_bound = np.percentile(y, 100 - clip_percentile)
        upper_bound = np.percentile(y, clip_percentile)
        return np.clip(y, lower_bound, upper_bound)
    
    y_train_clipped = clip_extreme_values(y_train, clip_percentile=99)
    y_val_clipped = clip_extreme_values(y_val, clip_percentile=99)
    y_test_clipped = clip_extreme_values(y_test, clip_percentile=99)
    
    print(f"目标收益率统计:")
    print(f"  训练集: 均值={y_train_clipped.mean():.6f}, 标准差={y_train_clipped.std():.6f}")
    print(f"  验证集: 均值={y_val_clipped.mean():.6f}, 标准差={y_val_clipped.std():.6f}")
    print(f"  测试集: 均值={y_test_clipped.mean():.6f}, 标准差={y_test_clipped.std():.6f}")
        
    def time_decay_weights(dates, tau_days=360):
        if len(dates) == 0:
            return np.array([], dtype=np.float32)
        max_d = max(dates)
        deltas = np.array([(max_d - d).days for d in dates], dtype=np.float32)
        w = np.exp(-deltas / float(tau_days))
        return (w / (w.mean() + 1e-8)).astype(np.float32)

    # 简化权重：只使用时间衰减权重
    w_time_train = time_decay_weights(dates_train, tau_days=360)
    w_time_val   = time_decay_weights(dates_val, tau_days=360)
    
    # 训练模型
    input_size = X_train.shape[2]
    model, train_losses, val_losses, _ = train_multi_symbol_gru_model(
        X_train_scaled, y_train_clipped, X_val_scaled, y_val_clipped,
        symbols_train, symbols_val, dates_train, dates_val,
        input_size, epochs=train_epochs, batch_size=256,
        weights_train=w_time_train, weights_val=w_time_val, symbol_to_id=symbol_to_id
    )
    
    # 样本外预测和评估
    print("🔮 开始样本外预测...")
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_test_scaled), 128):
            bx = torch.FloatTensor(X_test_scaled[i:i+128]).to(DEVICE)
            bs = torch.LongTensor([symbol_to_id.get(s, 0) for s in symbols_test[i:i+128]]).to(DEVICE)
            pred = model(bx, bs).squeeze().cpu().numpy()
            predictions.extend(pred)
    
    predictions = np.array(predictions)
    
    # 样本外指标
    test_rmse = np.sqrt(mean_squared_error(y_test_clipped, predictions))
    test_mae = mean_absolute_error(y_test_clipped, predictions)
    test_r2 = r2_score(y_test_clipped, predictions)
    
    # 计算横截面相关性（按日期）
    test_df = pd.DataFrame({
        'date': dates_test,
        'pred': predictions,
        'actual': y_test_clipped
    })
    daily_corrs = []
    for date, group in test_df.groupby('date'):
        if len(group) >= 3:
            corr = group['pred'].corr(group['actual'])
            if not np.isnan(corr):
                daily_corrs.append(corr)
    
    avg_ic = np.mean(daily_corrs) if daily_corrs else 0
    print(f"[样本外-Test]  RMSE={test_rmse:.6f}, MAE={test_mae:.6f}, R2={test_r2:.4f}")
    print(f"[样本外-Test]  平均IC={avg_ic:.4f}, 有效日期数={len(daily_corrs)}")

    # 统一生成因子数据（包含 train/val/test）
    result_data = []
    sets = [
        ("train", X_train_scaled, symbols_train, dates_train, y_train_clipped),
        ("val",   X_val_scaled,   symbols_val,   dates_val,   y_val_clipped),
        ("test",  X_test_scaled,  symbols_test,  dates_test,  y_test_clipped),
    ]
    with torch.no_grad():
        for split, Xs, syms, dts, ys in sets:
            if len(Xs) == 0:
                continue
            preds = []
            for i in range(0, len(Xs), 128):
                bx = torch.FloatTensor(Xs[i:i+128]).to(DEVICE)
                bs = torch.LongTensor([symbol_to_id.get(s, 0) for s in syms[i:i+128]]).to(DEVICE)
                preds.extend(model(bx, bs).squeeze().cpu().numpy())
            for pred, s, d, y_true in zip(preds, syms, dts, ys):
                result_data.append({
                    'date': d,
                    'symbol': s,
                    'gru_prediction': float(pred),
                    'future_ret': float(y_true),
                    'split': split
                })

    # 安全保护：若没有任何样本，直接返回空
    if len(result_data) == 0:
        print("警告: 无可用预测样本，返回空 DataFrame")
        return pd.DataFrame()

    # 1) 先构造 DataFrame
    factor_df = pd.DataFrame(result_data)

    # 2) 直接使用预测收益率作为因子，按日做rank归一化
    # 不做z-score，直接rank到[0,1]，保持预测收益率的原始含义
    factor_df = rank_to_unit_by_date(factor_df, col="gru_prediction", out_col="factor")

    # 3) 重命名并保存
    factor_df = factor_df.rename(columns={"symbol": "instrument"})[["date", "instrument", "factor", "future_ret", "split"]]
    out_path = save_factor_df(factor_df, file_prefix=f"adaptive_multi_symbol_gru_{window}d_rebalance{rebalance_period}d_")
    print_factor_summary(factor_df, out_path)
    return factor_df

if __name__ == "__main__":
    # 使用新的参数调用
    create_multi_symbol_gru_prediction_factor(
        window=20, 
        rebalance_period=5, 
        sequence_length=75, 
        train_epochs=200,
        min_history_days=60,
        min_samples_per_symbol=40
    )