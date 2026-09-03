from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


def daily_rank_ic(df: pd.DataFrame, factor_col: str = "factor", ret_col: str = "future_ret") -> pd.Series:
    def one_day(g: pd.DataFrame) -> float:
        if g[factor_col].nunique() < 3:
            return np.nan
        return g[[factor_col, ret_col]].corr(method="spearman").iloc[0, 1]

    # Avoid FutureWarning about apply operating on grouping columns by grouping on a Series
    # while applying to a DataFrame that doesn't contain the grouping column.
    x = df[[factor_col, ret_col]]
    return x.groupby(df["date"], sort=False).apply(one_day)


def daily_group_long_short(
    df: pd.DataFrame,
    factor_col: str = "factor",
    ret_col: str = "future_ret",
    group_n: int = 5,
) -> pd.Series:
    def one_day(g: pd.DataFrame) -> float:
        if len(g) < group_n * 3:
            return np.nan
        x = g[[factor_col, ret_col]].dropna()
        if len(x) < group_n * 3:
            return np.nan

        # 用 rank 再 qcut，减少重复值导致的分箱失败
        r = x[factor_col].rank(method="first")
        try:
            bins = pd.qcut(r, group_n, labels=False, duplicates="drop")
        except Exception:
            return np.nan
        if bins is None:
            return np.nan

        x = x.assign(_bin=bins)
        top = x[x["_bin"] == x["_bin"].max()][ret_col].mean()
        bot = x[x["_bin"] == x["_bin"].min()][ret_col].mean()
        return float(top - bot)

    x = df[[factor_col, ret_col]]
    return x.groupby(df["date"], sort=False).apply(one_day)


@dataclass
class StatIR:
    mean: float
    std: float
    ir: float
    n: int


def series_ir(s: pd.Series) -> StatIR:
    s = s.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(s) == 0:
        return StatIR(mean=np.nan, std=np.nan, ir=np.nan, n=0)

    # 防止极端值导致方差计算溢出（对 LS 序列尤其常见）
    # 这里做一个宽松裁剪，不改变正常范围内的信号。
    s = s.clip(lower=-10.0, upper=10.0)

    m = float(s.mean())
    with np.errstate(over="ignore", invalid="ignore"):
        sd = float(s.std(ddof=1) + 1e-12)
    return StatIR(mean=m, std=sd, ir=m / sd, n=int(len(s)))
