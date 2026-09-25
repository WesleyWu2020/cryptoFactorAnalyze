"""Crypto continuous scanning: multiple rounds, accumulate top 100 factors."""
import sys, os, time, json
sys.path.insert(0, '/root/crypto-research/common/gp')
os.environ['NUMBA_NUM_THREADS'] = '8'

from parametric_crypto.evolution import run_evolution

BASE_CFG = {
    'h5_path': '/root/crypto-research/common/data/raw/h5/binance_perp_1m.h5',
    'start_date': '20230301',
    'end_date': '20260331',
    'train_end_date': '20250630',
    'minutes_per_period': 240,
    'chunk_periods': 300,
    'population_size': 2000,
    'n_generations': 10,
    'backend': 'cpu',
    'lag_periods': 1,
    'top_quantile': 0.2,
}

N_ROUNDS = 50
all_factors = []
seen_formulas = set()  # deduplicate
output_base = '/root/crypto-research/common/gp/crypto_output'
os.makedirs(output_base, exist_ok=True)

for rd in range(N_ROUNDS):
    seed = 42 + rd * 1000
    output_dir = f'{output_base}/round_{rd:03d}'
    cfg = dict(BASE_CFG, seed=seed, output_dir=output_dir)

    sep = "=" * 80
    print(f'\n{sep}', flush=True)
    print(f'[ROUND {rd}] seed={seed}', flush=True)
    print(sep, flush=True)

    t0 = time.time()
    try:
        result = run_evolution(cfg)
        elapsed = time.time() - t0
        factors = result.get('results', [])
        n_new = 0

        # Keep ALL OOB-validated factors (deduplicated)
        for r in factors:
            oob = r.get('oob_objectives', {})
            formula = r.get('formula', '')
            if (oob.get('abs_ic', 0) > 0.02
                and oob.get('ls_sharpe', 0) > 0.0
                and formula not in seen_formulas):
                r['round'] = rd
                r['seed'] = seed
                all_factors.append(r)
                seen_formulas.add(formula)
                n_new += 1

        # Sort by OOB sharpe
        all_factors.sort(key=lambda x: x.get('oob_objectives', {}).get('ls_sharpe', -999), reverse=True)

        print(f'[ROUND {rd}] Done in {elapsed:.0f}s. '
              f'{len(factors)} Pareto, +{n_new} new, {len(all_factors)} total', flush=True)

        # Save ALL accumulated
        with open(f'{output_base}/all_validated_factors.json', 'w') as f:
            json.dump(all_factors, f, indent=2, default=str)

        # Print current top 5
        for i, r in enumerate(all_factors[:5]):
            oob = r.get('oob_objectives', {})
            print(f'  Top{i+1}: |IC|={oob.get("abs_ic",0):.4f} Shp={oob.get("ls_sharpe",0):.2f} '
                  f'Ret={oob.get("ls_return",0):.4f} | {r["formula"]}', flush=True)

    except Exception as e:
        print(f'[ROUND {rd}] ERROR: {e}', flush=True)
        import traceback
        traceback.print_exc()
        continue

print(f'\n[FINAL] {len(all_factors)} unique OOB-validated factors saved', flush=True)
