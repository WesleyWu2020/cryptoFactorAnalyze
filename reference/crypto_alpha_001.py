import numpy as np
import pandas as pd


TYPE = "regular"

META = {
    "factor_name":"crypto_alpha_001",
    "author":"root",
    "level":"minutes",
    "tag":"crypto_demo",
    "category":"momentum",
}

SETTING = {
    "data_needed":["close","volume"],
    "universe":"tradable_mask",
    "pasteurization":True,
    "warmup_bars":1440,
}


def calc_factor(data_ctx:dict[str,pd.DataFrame])->pd.DataFrame:
    close = data_ctx["close"]
    volume = data_ctx["volume"]

    ret_60 = close.pct_change(60,fill_method=None)
    ret_240 = close.pct_change(240,fill_method=None)
    volume_ratio = volume / volume.rolling(240,min_periods=60).mean()
    raw = (ret_240 - ret_60) * (volume_ratio - 1.0)
    raw = raw.replace([float("inf"),float("-inf")],np.nan)
    return -raw.ewm(span=1440,min_periods=240).mean()
