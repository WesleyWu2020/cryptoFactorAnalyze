"""Read-only missing-signal attribution and optional direction comparison.

Run from the repository root with ./.venv/bin/python.
Outputs preserve missing cells and explicitly separate structural history gaps.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factor_common.data_provider import DataProvider
from factor_common.manager import FactorManager
from factor_analyse.factor_mining.Volume_Stability_Factor import calc_factor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', default='data/crypto_quant.h5')
    parser.add_argument('--start', default='2024-01-01')
    parser.add_argument('--end', default='2026-03-15')
    parser.add_argument('--output', default='reports/volume_stability_quality_20260906')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    provider = DataProvider(args.store)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    context = {field: provider.get_single_data(field, start=start-pd.Timedelta(days=20), end=end)
               for field in ['quote_volume', 'trade_count']}
    quote, trades = context.values()
    universe = provider.get_universe(start=quote.index[0], end=end)
    factor = calc_factor(context).loc[start:end]
    eligible = universe.loc[start:end]
    missing = eligible & factor.isna()
    # Attribution uses only past observations; never fill the signal matrix.
    no_trade = (quote <= 0) | (trades <= 0)
    input_gap = quote.isna() | trades.isna()
    observed_count = (~input_gap & ~no_trade).rolling(20, min_periods=1).sum()
    zero_window = no_trade.rolling(20, min_periods=1).sum().gt(0)
    rows = []
    for i, j in zip(*np.nonzero(missing.to_numpy())):
        day, symbol = missing.index[i], missing.columns[j]
        if no_trade.loc[day, symbol]:
            reason = 'no_trade_current_day'
        elif input_gap.loc[day, symbol]:
            reason = 'missing_current_input'
        elif zero_window.loc[day, symbol]:
            reason = 'no_trade_in_window'
        elif observed_count.loc[day, symbol] < 20:
            reason = 'insufficient_observed_history'
        else:
            reason = 'nonfinite_formula'
        rows.append({'date': day, 'instrument': symbol, 'reason': reason,
                     'observed_bars_in_window': observed_count.loc[day, symbol]})
    detail = pd.DataFrame(rows, columns=['date', 'instrument', 'reason', 'observed_bars_in_window'])
    detail.to_csv(output/'missing_rows.csv', index=False)
    daily = pd.DataFrame({'eligible': eligible.sum(axis=1), 'valid': (eligible & factor.notna()).sum(axis=1),
                          'missing': missing.sum(axis=1)})
    daily.to_csv(output/'daily_coverage.csv', index_label='date')
    cutoff_checks = []
    for cutoff in [pd.Timestamp('2025-02-06'), end-pd.Timedelta(days=1)]:
        if cutoff < start:
            continue
        short = calc_factor({key: value.loc[:cutoff] for key, value in context.items()}).loc[start:cutoff]
        full = factor.loc[:cutoff]
        pd.testing.assert_frame_equal(short, full)
        cutoff_checks.append({'cutoff': str(cutoff.date()), 'max_abs_diff': 0.0})
    summary = {'store': args.store, 'start': args.start, 'end': args.end,
               'eligible_count': int(eligible.to_numpy().sum()),
               'valid_count': int((eligible & factor.notna()).to_numpy().sum()),
               'missing_count': len(detail), 'missing_by_reason': detail.reason.value_counts().to_dict(),
               'missing_by_symbol': detail.instrument.value_counts().to_dict(),
               'empty_universe_dates': [str(d.date()) for d in daily.index[daily.eligible.eq(0)]],
               'cutoff_checks': cutoff_checks}
    if args.evaluate:
        manager = FactorManager(h5_path=args.store, reports_dir=output)
        results = []
        for direction in [1, -1]:
            # Distinct report folders prevent the same factor-value run_id
            # overwriting reports with different evaluation profiles.
            manager.reports_dir = output / ('positive' if direction == 1 else 'negative')
            result = manager.evaluate('factor_analyse/factor_mining/Volume_Stability_Factor.py',
                params={'start': args.start, 'end': args.end, 'factor_direction': direction,
                        'rebalance_days': 1, 'n_groups': 10, 'include_funding': True}, plot=True)
            results.append({'direction': direction, 'status': result['status'], 'paths': result['paths'],
                            'diagnostics': result['diagnostics'], 'performance': result['factor_performance']})
            print('evaluated', direction, result['status'], result['paths']['report_path'], flush=True)
        summary['evaluations'] = results
    (output/'summary.json').write_text(json.dumps(summary, indent=2, default=str)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'evaluations'}, indent=2))


if __name__ == '__main__':
    main()
