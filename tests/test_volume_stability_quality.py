import numpy as np
import pandas as pd

from factor_analyse.factor_mining.Volume_Stability_Factor import calc_factor


def test_zero_trades_invalidate_window_and_recover_without_future_data():
    dates = pd.date_range('2024-01-01', periods=65)
    quote = pd.DataFrame({'A': np.arange(65) + 100.0}, index=dates)
    trades = pd.DataFrame({'A': 10.0}, index=dates)
    quote.iloc[25] = 0
    trades.iloc[25] = 0
    context = {'quote_volume': quote, 'trade_count': trades}
    full = calc_factor(context)
    assert full.iloc[24].notna().all()
    assert full.iloc[25:45].isna().all().all()
    assert full.iloc[45].notna().all()
    for cutoff in [24, 30, 50]:
        short = calc_factor({key: value.iloc[:cutoff + 1] for key, value in context.items()})
        pd.testing.assert_frame_equal(short, full.iloc[:cutoff + 1])
