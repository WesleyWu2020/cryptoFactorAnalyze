"""CLI entry point for crypto parametric factor mining v2."""
import argparse, json, os, sys

from .config import (
    CACHE_DTYPE,
    DEFAULT_FUNDAMENTAL_PANEL_PATH,
    EVAL_BATCH_SIZE,
    FINAL_CANDIDATE_POOL_SIZE,
    N_GENERATIONS,
    POPULATION_SIZE,
    RESULT_TARGET_MAX,
    RESULT_TARGET_MIN,
    TOP_QUANTILE,
    TRADING_COST,
)


_DEPRECATED_CONFIG_KEYS = {
    'composition_mode',
    'search_space_profile',
    'search_objective_profile',
    'oob_interval',
    'oob_eval_interval',
}

DIRECT_RUN_CHUNK_PERIODS = 30


def main():
    parser = argparse.ArgumentParser(description='Crypto parametric factor mining v2')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--h5_path', type=str, default=None)
    parser.add_argument('--start_date', type=str, default='20220102')
    parser.add_argument('--end_date', type=str, default='20251231',
                        help='Train/search window end. Formal OOS uses 2026-01-01..2026-05-01.')
    parser.add_argument('--train_end_date', type=str, default='20251231')
    parser.add_argument('--period_minutes', type=int, default=1440)
    parser.add_argument('--chunk_periods', type=int, default=DIRECT_RUN_CHUNK_PERIODS,
                        help='Periods per indicator chunk. Keep small for 24h GPU runs.')
    parser.add_argument('--population', type=int, default=POPULATION_SIZE)
    parser.add_argument('--generations', type=int, default=N_GENERATIONS)
    parser.add_argument('--backend', type=str, default='gpu', choices=['cpu', 'gpu'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--cpu_workers', type=int, default=None,
                        help='Override CPU worker count for evaluator/fitness threads')
    parser.add_argument('--gpu_eval_workers', type=int, default=None,
                        help='Override GPU-side concurrent evaluator workers')
    parser.add_argument('--top_quantile', type=float, default=TOP_QUANTILE,
                        help='Long/short quantile aligned with formal trading profile')
    parser.add_argument('--trading_cost', type=float, default=TRADING_COST,
                        help='Per-side trading cost used during fitness evaluation')
    parser.add_argument('--cache_dtype', type=str, default=CACHE_DTYPE, choices=['float16', 'float32'])
    parser.add_argument('--tradable_mask_overlay', type=str, default=None,
                        help='Optional .npy tradable mask overlay; use "none" to disable loader default.')
    parser.add_argument('--fundamental_panel', type=str, default=None,
                        help=f'Optional CoinMetrics daily panel .npz; default driver path is {DEFAULT_FUNDAMENTAL_PANEL_PATH}.')
    parser.add_argument('--eval_batch_size', type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument('--result_target_min', type=int, default=RESULT_TARGET_MIN,
                        help='Minimum number of curated candidates to save after final selection')
    parser.add_argument('--result_target_max', type=int, default=RESULT_TARGET_MAX,
                        help='Maximum number of curated candidates to save after final selection')
    parser.add_argument('--candidate_pool_size', type=int, default=FINAL_CANDIDATE_POOL_SIZE,
                        help='Number of near-front candidates to evaluate before final curation')
    parser.add_argument('--output_dir', type=str, default='crypto_parametric_v2_output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed_from', type=str, nargs='*', default=None,
                        help='Prior pareto_results.json files (v1 or v2) to seed population')
    parser.add_argument('--gpu_fitness', action='store_true',
                        help='Enable experimental GPU fitness prototype (disabled by default)')
    parser.add_argument('--perf_profile', action='store_true',
                        help='Print per-generation performance breakdown')
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = {}
    deprecated_keys = sorted(_DEPRECATED_CONFIG_KEYS.intersection(cfg))
    if deprecated_keys:
        raise ValueError(
            "Deprecated mining config keys are no longer accepted: "
            + ", ".join(deprecated_keys)
        )

    if args.h5_path:
        cfg['h5_path'] = args.h5_path
        os.environ['GP_H5_PATH'] = str(args.h5_path)
    cfg.setdefault('start_date', args.start_date)
    cfg.setdefault('end_date', args.end_date)
    cfg.setdefault('train_end_date', args.train_end_date)
    cfg['minutes_per_period'] = args.period_minutes
    cfg['chunk_periods'] = args.chunk_periods
    cfg['population_size'] = args.population
    cfg['n_generations'] = args.generations
    cfg['backend'] = args.backend
    cfg['gpu_id'] = args.gpu_id
    cfg['top_quantile'] = args.top_quantile
    cfg['trading_cost'] = args.trading_cost
    cfg['cache_dtype'] = args.cache_dtype
    if args.tradable_mask_overlay is not None:
        cfg['tradable_mask_overlay_path'] = args.tradable_mask_overlay
    if args.fundamental_panel is not None:
        cfg['fundamental_panel_path'] = args.fundamental_panel
        if str(args.fundamental_panel).lower() in {'0', 'false', 'none', 'off'}:
            os.environ['GP_ENABLE_FUNDAMENTALS'] = '0'
            os.environ.pop('GP_FUNDAMENTAL_PANEL', None)
        else:
            os.environ['GP_FUNDAMENTAL_PANEL'] = str(args.fundamental_panel)
    cfg['eval_batch_size'] = args.eval_batch_size
    if args.result_target_min is not None:
        cfg['result_target_min'] = args.result_target_min
    if args.result_target_max is not None:
        cfg['result_target_max'] = args.result_target_max
    if args.candidate_pool_size is not None:
        cfg['candidate_pool_size'] = args.candidate_pool_size
    cfg['output_dir'] = args.output_dir
    cfg['seed'] = args.seed
    os.environ.setdefault('GP_RUN_SEED', str(args.seed))
    os.environ.setdefault('GP_FIELD_SLATE_SEED', str(args.seed))
    if args.seed_from:
        cfg['seed_from'] = args.seed_from
    cfg['gpu_fitness'] = bool(cfg.get('gpu_fitness', False) or args.gpu_fitness)
    cfg['perf_profile'] = bool(cfg.get('perf_profile', False) or args.perf_profile)

    if cfg['gpu_fitness']:
        os.environ['GP_V2_GPU_FITNESS'] = '1'
    else:
        os.environ.pop('GP_V2_GPU_FITNESS', None)
    if args.cpu_workers:
        os.environ['GP_CPU_WORKERS'] = str(args.cpu_workers)
    if args.gpu_eval_workers:
        os.environ['GP_V2_GPU_EVAL_WORKERS'] = str(args.gpu_eval_workers)

    from .evolution import run_evolution
    run_evolution(cfg)


if __name__ == '__main__':
    sys.exit(main() or 0)
