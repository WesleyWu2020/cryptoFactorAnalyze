"""Reusable same-day cross-sectional preprocessing for long-form factors."""

import numpy as np
import pandas as pd


def winsorize_by_date(df: pd.DataFrame, col: str, n_std: float = 3.0) -> pd.DataFrame:
    """使用 median ± n * 1.4826 * MAD 去极值（向量化实现，避免 groupby.apply 丢列问题）。"""
    out = df.copy()
    if out.empty:
        return out

    # MAD 与 σ 的换算：对正态分布 σ ≈ 1.4826 * MAD
    scale = n_std * 1.4826

    med = out.groupby("date")[col].transform("median")
    abs_dev = (out[col] - med).abs()
    mad = abs_dev.groupby(out["date"]).transform("median")

    # 当 MAD 非常小时，保持原值
    valid = mad >= 1e-12
    lower = med - scale * mad
    upper = med + scale * mad
    out.loc[valid, col] = out.loc[valid, col].clip(lower=lower[valid], upper=upper[valid])
    return out


def rank_to_unit_by_date(df: pd.DataFrame, col: str, out_col: str = "factor") -> pd.DataFrame:
    """按日截面将 rank 映射到 [-1, 1]（向量化实现，避免 groupby.apply 丢列问题）。"""
    out = df.copy()
    if out.empty:
        out[out_col] = []
        return out

    rank_ = out.groupby("date")[col].rank(method="average")
    n = out.groupby("date")[col].transform("count")
    out[out_col] = np.where(n <= 1, 0.0, 2 * (rank_ - 1) / (n - 1) - 1)
    return out
