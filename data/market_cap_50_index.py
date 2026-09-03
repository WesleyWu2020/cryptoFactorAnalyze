"""
Binance 可交易流动性 Top50 指数编制系统

核心规则:
1. 排名指标: 30 日中位数成交额 `median_quote_volume_30d`
2. 样本过滤: 上市满 90 天，且近 30 天有效成交天数 >= 20
3. 成分稳定: 使用 40/60 buffer，降低成分抖动
4. 调仓口径: 固定频率调仓，T 日收盘选股，T+1 生效

注意:
1. 这里的排序与权重本质上是“流动性”口径，不是真实流通市值。
2. 如果要做真实 market cap 指数，应接入可回放的历史流通市值数据源。
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import warnings
import logging
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXCLUDED_FIAT_BASE_ASSETS = {
    "U", "EUR", "TRY", "RUB", "BRL", "UAH", "BIDR", "IDRT", "NGN", "PLN", "RON",
    "JPY", "GBP", "AUD", "CAD", "CHF", "NZD", "HKD", "SGD", "ZAR", "MXN",
    "ARS", "CLP", "COP", "PEN", "KES", "GHS", "MAD", "EGP", "SAR", "AED",
    "QAR", "KWD", "BHD", "OMR", "JOD", "THB", "VND", "MYR", "IDR", "PHP", "TWD"
}


def _is_excluded_fiat_symbol(symbol: str) -> bool:
    sym = str(symbol).upper()
    if not sym.endswith("USDT"):
        return False
    return sym[:-4] in EXCLUDED_FIAT_BASE_ASSETS


def build_daily_index_cache(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    基于日频指数结果生成日频缓存：
    - 保留真实 is_rebalance_day 标识
    - components/weights 从最近调仓日向前填充到非调仓日
    """
    if index_df.empty:
        return index_df

    df = index_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if 'components' in df.columns:
        comp = df['components'].astype('string')
        comp_non_empty = comp.notna() & comp.str.strip().ne('')
        if 'is_rebalance_day' in df.columns and df['is_rebalance_day'].notna().any():
            source_mask = df['is_rebalance_day'].fillna(False) & comp_non_empty
        else:
            source_mask = comp_non_empty
        df['components'] = comp.where(source_mask).ffill()
        df['components_count'] = df['components'].fillna('').apply(
            lambda x: len([s for s in str(x).split(',') if s.strip()]) if x else 0
        )

    if 'weights' in df.columns:
        weights = df['weights'].astype('string')
        weights_non_empty = weights.notna() & weights.str.strip().ne('')
        if 'is_rebalance_day' in df.columns and df['is_rebalance_day'].notna().any():
            w_source_mask = df['is_rebalance_day'].fillna(False) & weights_non_empty
        else:
            w_source_mask = weights_non_empty
        df['weights'] = weights.where(w_source_mask).ffill()

    return df


def write_index_metadata(csv_path: str, source: str, schema_version: str = "v1") -> None:
    """写入最小元信息。"""
    if not os.path.exists(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return
        date_series = pd.to_datetime(df.get('date'), format='mixed', errors='coerce')
        rebalance_count = 0
        if 'is_rebalance_day' in df.columns:
            rebalance_count = int(pd.to_numeric(df['is_rebalance_day'], errors='coerce').fillna(0).astype(bool).sum())
        meta = {
            "dataset": os.path.basename(csv_path),
            "schema_version": schema_version,
            "source": source,
            "row_count": int(len(df)),
            "date_min": str(date_series.min().date()) if date_series.notna().any() else None,
            "date_max": str(date_series.max().date()) if date_series.notna().any() else None,
            "rebalance_days": rebalance_count,
            "updated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "columns": list(df.columns),
        }
        with open(f"{csv_path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"写入指数元信息失败: {e}")

# 添加factor_mining目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
factor_mining_dir = os.path.join(current_dir, '..', 'factor_analyse', 'factor_mining')
if factor_mining_dir not in sys.path:
    sys.path.insert(0, factor_mining_dir)

try:
    from util_factor import build_daily_top50_ranking_from_kline
except ImportError as e:
    # 本地备用函数
    def build_daily_top50_ranking_from_kline(kline_df, ranking_method='quote_volume', top_n=50):
        daily_top50 = {}
        for date, group in kline_df.groupby('date'):
            if ranking_method == 'quote_volume':
                ranking_df = group.groupby('symbol').agg({'quote_volume': 'sum'}).reset_index()
                ranking_df['rank_score'] = ranking_df['quote_volume']
            else:
                ranking_df = group.groupby('symbol').agg({'volume': 'sum'}).reset_index()
                ranking_df['rank_score'] = ranking_df['volume']

            ranking_df = ranking_df[
                ~ranking_df['symbol'].astype(str).map(_is_excluded_fiat_symbol)
            ].copy()
            
            ranking_df = ranking_df.sort_values('rank_score', ascending=False)
            top_n_symbols = ranking_df.head(top_n)['symbol'].tolist()
            daily_top50[date] = top_n_symbols
        return daily_top50

class IndexWeightMethod(Enum):
    """指数加权方法"""
    MARKET_CAP = "market_cap"           # 市值加权
    EQUAL_WEIGHT = "equal_weight"       # 等权重
    LIQUIDITY_ADJUSTED = "liquidity_adjusted"  # 流动性调整

class RebalanceFrequency(Enum):
    """调仓频率"""
    DAILY = 1
    BIWEEKLY = 14
    WEEKLY = 7
    MONTHLY = 30
    QUARTERLY = 90

@dataclass
class IndexConfig:
    """指数配置参数"""
    index_name: str = "Binance Tradable Liquidity Top 50"
    base_date: str = None
    base_value: float = 1.0          # 基期指数值通常设为1
    sample_size: int = 50
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    max_component_weight: float = 0.10
    weight_method: IndexWeightMethod = IndexWeightMethod.MARKET_CAP
    use_csv_ranking: bool = True
    ranking_csv_path: str = None
    enable_cache: bool = True
    cache_validity_hours: int = 24
    lookback_days: int = 30
    min_listed_days: int = 90
    min_valid_quote_days: int = 20
    entry_rank: int = 40
    exit_rank: int = 60

class BinanceMarketCap50Index:
    """Binance加密货币市值50指数 (修正版)"""
    
    def __init__(self, config: IndexConfig = None, data_dir: str = None):
        self.config = config or IndexConfig()
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        self.klines_data: Optional[pd.DataFrame] = None
        
        logger.info(f"初始化 {self.config.index_name}")
        logger.info(
            f"配置: Top{self.config.sample_size}, "
            f"调仓={self.config.rebalance_frequency.name}, "
            f"加权={self.config.weight_method.value}, "
            f"排名=30d成交额中位数, "
            f"过滤=上市{self.config.min_listed_days}天+近30天有效成交{self.config.min_valid_quote_days}天, "
            f"buffer={self.config.entry_rank}/{self.config.exit_rank}"
        )
    
    def load_market_data(self) -> pd.DataFrame:
        if self.klines_data is not None:
            return self.klines_data
            
        logger.info("📈 加载K线数据...")
        kline_dir = os.path.join(self.data_dir, "kline_data")
        if not os.path.exists(kline_dir):
            raise FileNotFoundError(f"未找到K线数据目录: {kline_dir}")
        
        kline_files = [f for f in os.listdir(kline_dir) if f.startswith("binance_daily_klines_") and f.endswith(".csv")]
        if not kline_files:
            raise FileNotFoundError("未找到K线数据文件")
        
        latest_file = sorted(kline_files)[-1]
        file_path = os.path.join(kline_dir, latest_file)
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 数据清洗
        df = df.dropna(subset=['close', 'quote_volume'])
        df = df[df['close'] > 0]
        df = df[df['symbol'].str.endswith('USDT')]
        df = df[~df['symbol'].astype(str).map(_is_excluded_fiat_symbol)]
        
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        grouped = df.groupby('symbol', group_keys=False)

        # 真实口径是流动性分数，而不是市值
        df['is_valid_quote_day'] = (df['quote_volume'] > 0).astype(int)
        df['listed_days'] = grouped.cumcount() + 1
        df['median_quote_volume_30d'] = grouped['quote_volume'].transform(
            lambda x: x.rolling(self.config.lookback_days, min_periods=1).median()
        )
        df['valid_quote_days_30d'] = grouped['is_valid_quote_day'].transform(
            lambda x: x.rolling(self.config.lookback_days, min_periods=1).sum()
        )
        df['avg_quote_volume_30d'] = grouped['quote_volume'].transform(
            lambda x: x.rolling(self.config.lookback_days, min_periods=1).mean()
        )
        df['selection_score'] = df['median_quote_volume_30d']
        
        self.klines_data = df
        return df

    def _build_rebalance_dates(self, trading_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
        """生成决策日列表，调仓在下一交易日生效。"""
        if not trading_dates:
            return []

        if self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            rebalance_dates = []
            for idx, current_date in enumerate(trading_dates):
                is_last = idx == len(trading_dates) - 1
                next_month_changed = (
                    not is_last and trading_dates[idx + 1].month != current_date.month
                )
                if is_last or next_month_changed:
                    rebalance_dates.append(current_date)
            return rebalance_dates

        if self.config.rebalance_frequency == RebalanceFrequency.BIWEEKLY:
            rebalance_dates = [trading_dates[0]]
            last_rebalance = trading_dates[0]
            for current_date in trading_dates[1:]:
                if current_date - last_rebalance >= pd.Timedelta(days=14):
                    rebalance_dates.append(current_date)
                    last_rebalance = current_date
            return rebalance_dates

        step = max(1, int(self.config.rebalance_frequency.value))
        return [trading_dates[idx] for idx in range(0, len(trading_dates), step)]

    def _select_components_with_buffer(
        self,
        date: pd.Timestamp,
        df: pd.DataFrame,
        previous_components: List[str],
    ) -> Tuple[List[str], pd.DataFrame]:
        """按过滤条件和 40/60 buffer 选择成分股。"""
        day_data = df[df['date'] == date].copy()
        if day_data.empty:
            return [], pd.DataFrame()

        eligible = day_data[
            (day_data['listed_days'] >= self.config.min_listed_days) &
            (day_data['valid_quote_days_30d'] >= self.config.min_valid_quote_days) &
            (day_data['selection_score'] > 0)
        ].copy()

        if eligible.empty:
            logger.warning(f"{date.date()} 无满足上市/成交过滤条件的候选币种")
            return [], pd.DataFrame()

        ranked = eligible.sort_values(
            ['selection_score', 'quote_volume', 'symbol'],
            ascending=[False, False, True]
        ).reset_index(drop=True)
        ranked['rank'] = np.arange(1, len(ranked) + 1)

        entry_cut = min(self.config.entry_rank, len(ranked))
        exit_cut = min(self.config.exit_rank, len(ranked))

        keep_components = ranked[
            ranked['symbol'].isin(previous_components) & (ranked['rank'] <= exit_cut)
        ]['symbol'].tolist()

        new_components = []
        for symbol in ranked[ranked['rank'] <= entry_cut]['symbol'].tolist():
            if symbol not in keep_components:
                new_components.append(symbol)

        selected = keep_components + new_components
        if len(selected) < self.config.sample_size:
            for symbol in ranked['symbol'].tolist():
                if symbol not in selected:
                    selected.append(symbol)
                if len(selected) >= self.config.sample_size:
                    break

        selected = selected[:self.config.sample_size]
        selected_ranked = ranked[ranked['symbol'].isin(selected)].copy()
        selected_ranked['selection_order'] = selected_ranked['symbol'].map(
            {symbol: idx for idx, symbol in enumerate(selected)}
        )
        selected_ranked = selected_ranked.sort_values('selection_order')
        return selected, selected_ranked

    def _build_rebalance_plan(
        self,
        df: pd.DataFrame,
        trading_dates: List[pd.Timestamp],
    ) -> Dict[pd.Timestamp, Dict]:
        """构建 T 日决策、T+1 生效的调仓计划。"""
        decision_dates = self._build_rebalance_dates(trading_dates)
        plan: Dict[pd.Timestamp, Dict] = {}
        previous_components: List[str] = []

        for decision_date in decision_dates:
            try:
                decision_idx = trading_dates.index(decision_date)
            except ValueError:
                continue

            if decision_idx + 1 >= len(trading_dates):
                continue

            effective_date = trading_dates[decision_idx + 1]
            components, selected_ranked = self._select_components_with_buffer(
                decision_date, df, previous_components
            )
            if not components:
                continue

            previous_components = components
            plan[effective_date] = {
                'decision_date': decision_date,
                'effective_date': effective_date,
                'components': components,
                'selected_ranked': selected_ranked
            }

        logger.info(f"生成 {len(plan)} 个调仓生效日计划")
        return plan

    def _calculate_target_weights(self, ranked_df: pd.DataFrame) -> Dict[str, float]:
        """计算目标权重，使用决策日的流动性分数。"""
        if ranked_df.empty:
            return {}

        ranked_df = ranked_df.copy()
        valid_symbols = ranked_df['symbol'].tolist()
        weights = {}

        if self.config.weight_method == IndexWeightMethod.EQUAL_WEIGHT:
            w = 1.0 / len(valid_symbols)
            weights = {s: w for s in valid_symbols}
        else:
            score_col = 'selection_score'
            raw_scores = ranked_df.set_index('symbol')[score_col].clip(lower=0.0)
            if self.config.weight_method == IndexWeightMethod.LIQUIDITY_ADJUSTED:
                raw_scores = np.sqrt(raw_scores)

            total_score = raw_scores.sum()
            if total_score > 0:
                weights = (raw_scores / total_score).clip(
                    upper=self.config.max_component_weight
                ).to_dict()

        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        return weights

    def calculate_index(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        修正后的指数计算引擎：使用持仓量法 (Holdings Method)
        """
        df = self.load_market_data()
        logger.info("构建价格查询索引...")
        price_lookup = df.pivot(index='date', columns='symbol', values='close').sort_index()

        if not start_date:
            start_date = df['date'].min()
        else:
            start_date = pd.to_datetime(start_date)
        if not end_date:
            end_date = df['date'].max()
        else:
            end_date = pd.to_datetime(end_date)

        trading_dates = sorted([d for d in price_lookup.index if d >= start_date and d <= end_date])
        rebalance_plan = self._build_rebalance_plan(df, trading_dates)

        current_holdings: Dict[str, float] = {}
        current_weights: Dict[str, float] = {}
        current_components: List[str] = []
        index_value = self.config.base_value
        results = []

        logger.info("开始逐日计算指数...")

        for date in trading_dates:
            try:
                current_prices = price_lookup.loc[date].dropna()
            except KeyError:
                logger.warning(f"{date} 无法获取价格数据，跳过")
                continue

            plan_item = rebalance_plan.get(date)
            if plan_item is not None:
                target_weights = self._calculate_target_weights(plan_item['selected_ranked'])
                new_holdings = {}
                for symbol, weight in target_weights.items():
                    if symbol in current_prices and current_prices[symbol] > 0:
                        qty = (index_value * weight) / current_prices[symbol]
                        new_holdings[symbol] = qty

                if new_holdings:
                    current_holdings = new_holdings
                    current_weights = target_weights
                    current_components = list(target_weights.keys())

            if not current_holdings:
                continue

            portfolio_value = 0.0
            valid_value = False
            for symbol, qty in current_holdings.items():
                if symbol in current_prices:
                    portfolio_value += qty * current_prices[symbol]
                    valid_value = True

            if valid_value:
                prev_value = results[-1]['index_value'] if results else self.config.base_value
                daily_return = (portfolio_value / prev_value) - 1 if prev_value > 0 else 0.0
                index_value = portfolio_value
            else:
                daily_return = 0.0

            res = {
                'date': date,
                'index_value': index_value,
                'daily_return': daily_return,
                'components_count': len(current_components),
                'is_rebalance_day': plan_item is not None
            }
            if plan_item is not None:
                res['decision_date'] = plan_item['decision_date']
                res['effective_date'] = plan_item['effective_date']
                res['components'] = ','.join(current_components)
                res['weights'] = json.dumps(current_weights, ensure_ascii=True, sort_keys=True)
            results.append(res)

        return pd.DataFrame(results)

    def generate_daily_from_monthly(self, monthly_cache_path: str, output_path: str = None) -> pd.DataFrame:
        """
        基于月度指数数据生成每日指数 (修正版：使用持仓插值法)
        
        原理：
        1. 读入月度指数（包含了每个月调仓时的点位和成分股）。
        2. 在月初，根据月度指数点位和成分股权重，反算出持仓量(Holdings)。
        3. 在月内，保持持仓量不变，根据每日价格计算指数值。
        这样生成的每日曲线是精确平滑的。
        """
        if not os.path.exists(monthly_cache_path):
            logger.error("月度文件不存在")
            return pd.DataFrame()
            
        monthly_df = pd.read_csv(monthly_cache_path)
        monthly_df['date'] = pd.to_datetime(monthly_df['date'])
        
        # 加载价格
        df = self.load_market_data()
        price_lookup = df.pivot(index='date', columns='symbol', values='close')
        
        # 收集所有交易日
        start_date = monthly_df['date'].min()
        end_date = df['date'].max()
        all_dates = sorted([d for d in price_lookup.index if d >= start_date and d <= end_date])
        
        # 索引月度数据
        monthly_map = monthly_df.set_index('date')
        
        daily_results = []
        current_holdings = {}
        current_index = 1.0
        
        for date in all_dates:
            # 标准化日期比较
            date_norm = pd.Timestamp(date).normalize()
            is_rebalance = date_norm in monthly_map.index
            
            # 获取价格
            try:
                prices = price_lookup.loc[date].dropna()
            except:
                continue

            # 如果是调仓日（月初）
            if is_rebalance:
                row = monthly_map.loc[date_norm]
                current_index = row['index_value'] # 强制校准为月度文件的值
                daily_return = row.get('daily_return', 0.0)
                
                # 解析成分股和权重
                if pd.notna(row.get('components')):
                    components = str(row['components']).split(',')
                    weight_map = {}
                    if pd.notna(row.get('weights')):
                        try:
                            weight_map = json.loads(row['weights'])
                        except Exception:
                            weight_map = {}

                    if not weight_map and components:
                        fallback_weight = 1.0 / len(components)
                        weight_map = {sym.strip(): fallback_weight for sym in components if sym.strip()}
                    
                    # 重新计算持仓
                    # Q = (Index * W) / P
                    new_holdings = {}
                    for sym in components:
                        sym = sym.strip()
                        weight = weight_map.get(sym, 0.0)
                        if sym in prices and prices[sym] > 0 and weight > 0:
                            qty = (current_index * weight) / prices[sym]
                            new_holdings[sym] = qty
                    current_holdings = new_holdings
                
            else:
                # 非调仓日：计算市值
                if not current_holdings:
                    # 如果还没有初始化持仓（在第一个调仓日之前），跳过
                    continue
                    
                new_value = 0.0
                for sym, qty in current_holdings.items():
                    if sym in prices:
                        new_value += qty * prices[sym]
                
                # 计算收益率
                if current_index > 0 and new_value > 0:
                    daily_return = (new_value / current_index) - 1
                    current_index = new_value
                else:
                    # 如果无法计算（价格缺失等），保持前一日指数值
                    daily_return = 0.0
                    # current_index 保持不变

            # 记录结果
            result_row = {
                'date': date,
                'index_value': current_index,
                'daily_return': daily_return,
                'components_count': len(current_holdings),
                'is_rebalance_day': is_rebalance
            }
            
            # 只在调仓日保存components信息
            if is_rebalance and current_holdings:
                result_row['components'] = ','.join(sorted(current_holdings.keys()))
            
            daily_results.append(result_row)
            
        result_df = pd.DataFrame(daily_results)
        
        if output_path and not result_df.empty:
            result_df.to_csv(output_path, index=False)
            logger.info(f"💾 每日指数已保存: {output_path}")
            
        return result_df

def main():
    print("🚀 启动修正版指数计算...")
    
    # 获取当前脚本所在目录作为数据目录
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 示例配置: 等权、月度、Top50
    config = IndexConfig(
        index_name="Binance Tradable Liquidity Top 50",
        weight_method=IndexWeightMethod.EQUAL_WEIGHT,
        rebalance_frequency=RebalanceFrequency.MONTHLY,
        sample_size=50,
        lookback_days=30,
        min_listed_days=90,
        min_valid_quote_days=20,
        entry_rank=40,
        exit_rank=60,
    )
    
    idx = BinanceMarketCap50Index(config, data_dir=data_dir)
    
    try:
        # 1. 计算指数
        df = idx.calculate_index()
        print(f"计算完成! 最终指数: {df.iloc[-1]['index_value']:.2f}")
        
        # 2. 保存主要缓存文件 (index_cache_equal_weight_30_50.csv)
        # 这里的命名规则参考原文件: {method}_{freq}_{size}
        # Monthly=30 (approx), Equal=equal_weight, Size=50
        cache_filename = "index_cache_equal_weight_30_50.csv"
        cache_path = os.path.join(data_dir, cache_filename)
        df.to_csv(cache_path, index=False)
        print(f"主缓存文件已保存至: {cache_filename}")
        
        # 3. 基于主缓存生成日频文件（保留真实调仓日标识，成分按调仓日向前填充）
        daily_filename = "index_cache_equal_weight_30_50_daily.csv"
        daily_path = os.path.join(data_dir, daily_filename)
        
        daily_df = build_daily_index_cache(df)
        daily_df.to_csv(daily_path, index=False)
        write_index_metadata(cache_path, source="liquidity_top50_index_cache")
        write_index_metadata(daily_path, source="liquidity_top50_index_daily_cache")
        print(f"每日指数文件已保存至: {daily_filename}")
        
        # 打印部分数据
        print("\n生成数据预览:")
        print(daily_df.tail())

    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
