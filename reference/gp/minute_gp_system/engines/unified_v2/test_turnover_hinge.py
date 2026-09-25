import os
import unittest

import numpy as np

from minute_gp_system.engines.unified_v2.config import TURNOVER_HINGE_TARGET
from minute_gp_system.engines.unified_v2.fitness import (
    _compute_one,
    compute_five_objectives_prepared,
    materialize_fitness_context,
    prepare_fitness_context,
)


def _build_stable_panel(n_periods=160, n_symbols=30):
    base = np.arange(n_symbols, 0, -1, dtype=np.float32)
    factor = np.tile(base, (n_periods, 1))

    cross_section = np.linspace(0.03, -0.03, n_symbols, dtype=np.float32)
    returns = np.tile(cross_section, (n_periods, 1))
    return factor, returns.astype(np.float32)


def _build_flipping_panel(n_periods=160, n_symbols=30):
    base = np.arange(n_symbols, 0, -1, dtype=np.float32)
    reversed_base = base[::-1].copy()
    factor = np.empty((n_periods, n_symbols), dtype=np.float32)
    returns = np.empty((n_periods, n_symbols), dtype=np.float32)

    cross_section = np.linspace(0.03, -0.03, n_symbols, dtype=np.float32)
    reversed_cross_section = cross_section[::-1].copy()
    for i in range(n_periods):
        if i % 2 == 0:
            factor[i] = base
            returns[i] = cross_section
        else:
            factor[i] = reversed_base
            returns[i] = reversed_cross_section
    return factor, returns.astype(np.float32)


def _turnover_pressure(aux):
    return float(np.nanmax([aux[7], aux[8]]))


class TurnoverHingeTest(unittest.TestCase):
    def test_compute_one_sets_zero_turnover_objective_below_hinge_target(self):
        factor, returns = _build_stable_panel()

        obj, aux = _compute_one(
            factor,
            returns,
            top_frac=0.2,
            periods_per_year=365,
            rolling_window=30,
            return_aux=True,
        )

        pressure = _turnover_pressure(aux)
        self.assertTrue(np.isfinite(pressure))
        self.assertLess(pressure, TURNOVER_HINGE_TARGET)
        self.assertEqual(float(obj[2]), 0.0)

    def test_compute_one_penalizes_only_turnover_excess_above_hinge_target(self):
        factor, returns = _build_flipping_panel()

        obj, aux = _compute_one(
            factor,
            returns,
            top_frac=0.2,
            periods_per_year=365,
            rolling_window=30,
            return_aux=True,
        )

        pressure = _turnover_pressure(aux)
        expected = -max(0.0, pressure - TURNOVER_HINGE_TARGET)

        self.assertGreater(pressure, TURNOVER_HINGE_TARGET)
        self.assertAlmostEqual(float(obj[2]), expected, places=6)

    def test_prepared_paths_apply_same_turnover_hinge_objective(self):
        factor, returns = _build_flipping_panel()
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
        expected_pressure = _turnover_pressure(expected_aux)

        objectives, aux = compute_five_objectives_prepared(
            daily_factors,
            fitness_context,
            periods_per_year=365,
            top_frac=0.2,
            trading_cost=0.0015,
            return_aux=True,
        )

        self.assertAlmostEqual(float(objectives[0, 2]), float(expected_obj[2]), places=6)
        self.assertAlmostEqual(float(aux["ls_turnover"][0]), float(expected_aux[7]), places=6)
        self.assertAlmostEqual(float(aux["long_turnover"][0]), float(expected_aux[8]), places=6)
        self.assertAlmostEqual(
            float(objectives[0, 2]),
            -max(0.0, expected_pressure - TURNOVER_HINGE_TARGET),
            places=6,
        )

    def test_gpu_prepared_path_matches_cpu_turnover_hinge_when_available(self):
        try:
            import cupy  # noqa: F401
        except Exception as exc:
            self.skipTest(f"cupy unavailable: {exc}")

        from minute_gp_system.engines.unified_v2.backend import free_gpu, set_backend

        factor, returns = _build_flipping_panel()
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

        old_env = os.environ.get("GP_V2_GPU_FITNESS")
        gpu_ready = False
        try:
            os.environ["GP_V2_GPU_FITNESS"] = "1"
            try:
                set_backend("gpu", 0)
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
                os.environ.pop("GP_V2_GPU_FITNESS", None)
            else:
                os.environ["GP_V2_GPU_FITNESS"] = old_env
            if gpu_ready:
                free_gpu()
                set_backend("cpu", 0)

        self.assertAlmostEqual(float(objectives[0, 2]), float(expected_obj[2]), places=5)
        self.assertAlmostEqual(float(aux["ls_turnover"][0]), float(expected_aux[7]), places=5)
        self.assertAlmostEqual(float(aux["long_turnover"][0]), float(expected_aux[8]), places=5)


if __name__ == "__main__":
    unittest.main()
