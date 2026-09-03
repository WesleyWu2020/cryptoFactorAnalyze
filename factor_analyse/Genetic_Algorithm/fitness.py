from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import threading

import numpy as np
import pandas as pd

# 让本目录能 import util_factor（在 factor_mining 目录下）
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FACTOR_ANALYSE_DIR = os.path.dirname(THIS_DIR)
FACTOR_MINING_DIR = os.path.join(FACTOR_ANALYSE_DIR, "factor_mining")
if FACTOR_MINING_DIR not in sys.path:
    sys.path.insert(0, FACTOR_MINING_DIR)

from util_factor import (
    future_return,
    rank_to_unit_by_date,
    winsorize_by_date,
)

from expr import Node
from metrics import daily_group_long_short, daily_rank_ic, series_ir


_AVAIL_DF_CACHE: dict[int, pd.DataFrame] = {}
_AVAIL_DF_LOCK = threading.Lock()


def _normalize_ts(x) -> pd.Timestamp:
    return pd.to_datetime(x).normalize()


def _get_availability_pairs_df(available_tokens_by_date: Dict) -> pd.DataFrame:
    """Build (date,symbol) pairs DataFrame. Cached by object id."""
    key = id(available_tokens_by_date)
    cached = _AVAIL_DF_CACHE.get(key)
    if cached is not None:
        return cached

    with _AVAIL_DF_LOCK:
        # double-check inside lock
        cached = _AVAIL_DF_CACHE.get(key)
        if cached is not None:
            return cached

        rows = []
        for d, symbols in available_tokens_by_date.items():
            dt = _normalize_ts(d)
            for sym in symbols:
                rows.append((dt, sym))
        avail = pd.DataFrame(rows, columns=["date", "symbol"])
        _AVAIL_DF_CACHE[key] = avail
        return avail


@dataclass
class FitnessResult:
    fitness: float
    ic_ir: float
    ls_ir: float
    ic_mean: float
    ls_mean: float
    n_rows: int
    expr: str


def _prepare_derived_features(gp: pd.DataFrame) -> pd.DataFrame:
    g = gp.copy()
    # 所有 shift/rolling 必须按 symbol 分组，避免不同币种之间串行
    close_lag1 = g.groupby("symbol", sort=False)["close"].shift(1)
    open_lag1 = g.groupby("symbol", sort=False)["open"].shift(1)

    g["ret1"] = g["close"] / (close_lag1 + 1e-12) - 1
    g["hl"] = g["high"] - g["low"]
    g["hl_pct"] = (g["high"] - g["low"]) / (close_lag1 + 1e-12)
    g["oc_ret"] = g["open"] / (close_lag1 + 1e-12) - 1
    g["co_pct"] = g["close"] / (g["open"] + 1e-12) - 1
    return g


def build_factor_df_for_expr(
    df: pd.DataFrame,
    expr: Node,
    available_tokens_by_date: Dict,
    rebalance_period: int,
    min_symbol_rows: int = 40,
    ret_clip: float = 10.0,
) -> pd.DataFrame:
    # 预处理：全表派生特征（必须保证按 symbol,date 排序后 rolling/shift 正确）
    g = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = _prepare_derived_features(g)

    # 表达式在全表计算：支持 cross-sectional rank/scale/neutralize
    try:
        raw = expr.eval(g)
    except Exception:
        return pd.DataFrame(columns=["date", "symbol", "raw_factor", "future_ret"])

    g = g.copy()
    # raw 可能带有重复 index（或与 g.index 不一致）；按位置对齐写入，避免 reindex 报错
    if isinstance(raw, pd.Series):
        raw_values = raw.to_numpy()
    else:
        raw_values = np.asarray(raw)
    if raw_values.shape[0] != len(g):
        return pd.DataFrame(columns=["date", "symbol", "raw_factor", "future_ret"])
    g["raw_factor"] = pd.Series(raw_values, index=g.index).replace([np.inf, -np.inf], np.nan)

    # 未来收益：必须按 symbol 分组计算，避免跨 symbol shift
    g["future_ret"] = (
        g.groupby("symbol", sort=False)["close"]
        .apply(lambda s: future_return(s, rebalance_period, method="log"))
        .reset_index(level=0, drop=True)
    )
    # 清洗未来收益，避免极端值/inf 导致后续 std/IR 计算 overflow
    g["future_ret"] = g["future_ret"].replace([np.inf, -np.inf], np.nan)
    if ret_clip is not None and ret_clip > 0:
        g["future_ret"] = g["future_ret"].clip(lower=-ret_clip, upper=ret_clip)

    # 先做最小行过滤：只保留符合成分股可用性的 (date,symbol)
    avail = _get_availability_pairs_df(available_tokens_by_date)
    g2 = g[["date", "symbol", "raw_factor", "future_ret"]].copy()
    g2["date"] = g2["date"].dt.normalize()
    fac = g2.merge(avail, on=["date", "symbol"], how="inner")
    fac = fac.dropna()

    # 可选：过滤特别短的 symbol（避免极少数点）
    if min_symbol_rows > 0:
        counts = fac.groupby("symbol", sort=False).size()
        keep = counts[counts >= min_symbol_rows].index
        fac = fac[fac["symbol"].isin(keep)]

    if fac.empty:
        return pd.DataFrame(columns=["date", "symbol", "raw_factor", "future_ret"])

    fac = winsorize_by_date(fac, col="raw_factor", n_std=3.0)
    fac = rank_to_unit_by_date(fac, col="raw_factor", out_col="factor")
    return fac[["date", "symbol", "factor", "future_ret"]]


def evaluate_expression(
    df: pd.DataFrame,
    expr: Node,
    available_tokens_by_date: Dict,
    rebalance_period: int,
    train_date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    group_n: int = 5,
    ic_weight: float = 0.6,
    ls_weight: float = 0.4,
    complexity_penalty: float = 0.002,
    missing_penalty: float = 1.0,
    min_rows: int = 8000,
    min_days: int = 60,
) -> FitnessResult:
    expr_str = expr.to_str()

    fac = build_factor_df_for_expr(
        df=df,
        expr=expr,
        available_tokens_by_date=available_tokens_by_date,
        rebalance_period=rebalance_period,
    )

    if fac.empty:
        return FitnessResult(
            fitness=-1e9,
            ic_ir=np.nan,
            ls_ir=np.nan,
            ic_mean=np.nan,
            ls_mean=np.nan,
            n_rows=0,
            expr=expr_str,
        )

    if train_date_range is not None:
        start, end = train_date_range
        fac = fac[(fac["date"] >= start) & (fac["date"] <= end)].copy()

    if len(fac) < min_rows:
        # 缺失/有效样本太少，直接判差
        miss = 1.0 - (len(fac) / max(min_rows, 1))
        return FitnessResult(
            fitness=-1e9 - missing_penalty * miss,
            ic_ir=np.nan,
            ls_ir=np.nan,
            ic_mean=np.nan,
            ls_mean=np.nan,
            n_rows=int(len(fac)),
            expr=expr_str,
        )

    ic_s = daily_rank_ic(fac)
    ls_s = daily_group_long_short(fac, group_n=group_n)

    ic_stat = series_ir(ic_s)
    ls_stat = series_ir(ls_s)

    if ic_stat.n < min_days or ls_stat.n < min_days:
        return FitnessResult(
            fitness=-1e9,
            ic_ir=ic_stat.ir,
            ls_ir=ls_stat.ir,
            ic_mean=ic_stat.mean,
            ls_mean=ls_stat.mean,
            n_rows=int(len(fac)),
            expr=expr_str,
        )

    score = ic_weight * ic_stat.ir + ls_weight * ls_stat.ir
    score -= complexity_penalty * expr.size()

    # 轻微惩罚：样本量越少越差（相对 min_rows）
    miss = max(0.0, 1.0 - (len(fac) / max(min_rows, 1)))
    score -= missing_penalty * miss

    return FitnessResult(
        fitness=float(score),
        ic_ir=float(ic_stat.ir),
        ls_ir=float(ls_stat.ir),
        ic_mean=float(ic_stat.mean),
        ls_mean=float(ls_stat.mean),
        n_rows=int(len(fac)),
        expr=expr_str,
    )
