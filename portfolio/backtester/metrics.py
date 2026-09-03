"""Portfolio performance metrics from daily return series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def annual_return(ret: pd.Series) -> float:
    mean_daily = ret.mean()
    return float((1 + mean_daily) ** 252 - 1)


def annual_volatility(ret: pd.Series) -> float:
    return float(ret.std(ddof=1) * np.sqrt(252))


def sharpe(ret: pd.Series, rf: float = 0.0) -> float:
    vol = annual_volatility(ret)
    if not np.isfinite(vol) or vol < 1e-12:
        return 0.0
    return float((annual_return(ret) - rf) / vol)


def max_drawdown(ret: pd.Series) -> float:
    cum = (1 + ret).cumprod()
    running_max = cum.cummax()
    dd = (cum / running_max - 1)
    return float(dd.min())


def calmar(ret: pd.Series) -> float:
    mdd = max_drawdown(ret)
    if not np.isfinite(mdd) or abs(mdd) < 1e-12:
        return 0.0
    return float(annual_return(ret) / abs(mdd))
