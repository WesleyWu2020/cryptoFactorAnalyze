import os
import unittest

import numpy as np

from gp.minute_gp_system.engines.unified_v2.config import OBJECTIVE_NAMES
from gp.minute_gp_system.engines.unified_v2.fitness import (
    _compute_one,
    _postprocess_single,
    _simulate_ls_holdings_single,
    compute_five_objectives_prepared,
    materialize_fitness_context,
    prepare_fitness_context,
)


def _build_test_panel(n_periods=160, n_symbols=30):
    base_factor = np.arange(n_symbols, 0, -1, dtype=np.float32)
    factor = np.tile(base_factor, (n_periods, 1))

    cross_section = np.linspace(0.03, -0.03, n_symbols, dtype=np.float32)
    scales = np.linspace(0.5, 1.5, n_periods, dtype=np.float32)
    returns = scales[:, None] * cross_section[None, :]
    return factor, returns.astype(np.float32)


class ObjectiveIcIrTest(unittest.TestCase):
    def test_compute_one_uses_rankicir_for_obj0_and_preserves_ann_return_in_aux(self):
        factor, returns = _build_test_panel()

        obj, aux = _compute_one(
            factor,
            returns,
            top_frac=0.2,
            periods_per_year=365,
            rolling_window=30,
            return_aux=True,
        )

        factor_pp = _postprocess_single(factor) * aux[0]
        ls_ret, ls_turnover, _, _, _ = _simulate_ls_holdings_single(
            factor_pp,
            returns,
            0.2,
        )
        ls_ret_net = ls_ret - ls_turnover * np.float32(0.0015 / 2.0)
        expected_ann_return = float(np.nanmean(ls_ret_net) * 365)
        expected_sharpe = float(
            np.nanmean(ls_ret_net) / np.nanstd(ls_ret_net) * np.sqrt(365)
        )

        self.assertEqual(len(aux), 11)
        self.assertAlmostEqual(float(obj[0]), float(aux[2]), places=6)
        self.assertAlmostEqual(float(aux[10]), expected_ann_return, places=6)
        self.assertAlmostEqual(float(obj[1]), expected_sharpe, places=5)
        self.assertNotAlmostEqual(float(obj[0]), expected_ann_return, places=6)

    def test_objective_names_and_aux_dict_expose_rankicir_and_real_ann_return(self):
        factor, returns = _build_test_panel()
        daily_factors = factor[None, :, :]
        fitness_context = prepare_fitness_context(returns)
        expected_obj, expected_aux = _compute_one(
            factor,
            returns,
            top_frac=0.2,
            periods_per_year=365,
            rolling_window=30,
            return_aux=True,
        )

        objectives, aux = compute_five_objectives_prepared(
            daily_factors,
            fitness_context,
            periods_per_year=365,
            top_frac=0.2,
            trading_cost=0.0015,
            return_aux=True,
        )

        self.assertEqual(
            OBJECTIVE_NAMES,
            ['rankicir', 'ls_net_sharpe', 'neg_turnover', 'novelty', 'ls_1-maxdd'],
        )
        self.assertIn('ls_netret_ann', aux)
        self.assertAlmostEqual(float(objectives[0, 0]), float(aux['rankicir'][0]), places=6)
        self.assertAlmostEqual(float(objectives[0, 0]), float(expected_obj[0]), places=6)
        self.assertAlmostEqual(float(aux['ls_netret_ann'][0]), float(expected_aux[10]), places=6)
        self.assertNotAlmostEqual(
            float(objectives[0, 0]),
            float(aux['ls_netret_ann'][0]),
            places=6,
        )

    def test_gpu_prepared_path_exposes_same_rankicir_and_ann_return_slots(self):
        try:
            import cupy  # noqa: F401
        except Exception as exc:
            self.skipTest(f"cupy unavailable: {exc}")

        from gp.minute_gp_system.engines.unified_v2.backend import free_gpu, set_backend

        factor, returns = _build_test_panel()
        daily_factors = factor[None, :, :]
        context = prepare_fitness_context(returns)
        expected_obj, expected_aux = _compute_one(
            factor,
            returns,
            top_frac=0.2,
            periods_per_year=365,
            rolling_window=30,
            return_aux=True,
        )
        old_env = os.environ.get('GP_V2_GPU_FITNESS')
        gpu_ready = False
        try:
            os.environ['GP_V2_GPU_FITNESS'] = '1'
            try:
                set_backend('gpu', 0)
                gpu_ready = True
                gpu_context = materialize_fitness_context(context)
            except Exception as exc:
                self.skipTest(f"gpu backend init failed: {exc}")

            objectives, aux = compute_five_objectives_prepared(
                daily_factors,
                gpu_context,
                periods_per_year=365,
                top_frac=0.2,
                trading_cost=0.0015,
                return_aux=True,
            )
        finally:
            if old_env is None:
                os.environ.pop('GP_V2_GPU_FITNESS', None)
            else:
                os.environ['GP_V2_GPU_FITNESS'] = old_env
            if gpu_ready:
                free_gpu()
                set_backend('cpu', 0)

        self.assertAlmostEqual(float(objectives[0, 0]), float(expected_obj[0]), places=5)
        self.assertAlmostEqual(float(aux['rankicir'][0]), float(expected_aux[2]), places=5)
        self.assertAlmostEqual(float(aux['ls_netret_ann'][0]), float(expected_aux[10]), places=5)


if __name__ == '__main__':
    unittest.main()
