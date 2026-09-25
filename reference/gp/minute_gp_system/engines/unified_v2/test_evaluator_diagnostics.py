import os
import warnings
import unittest

import numpy as np

from gp.minute_gp_system.engines.unified_v2.config import (
    CS_COMP_OPS,
    MASK_RULES,
    WINDOW_CHOICES,
    MODE1_OPS,
)
from gp.minute_gp_system.engines.unified_v2.evaluator import (
    _update_factor_diagnostics,
    evaluate_population_base,
)


class EvaluatorDiagnosticsTest(unittest.TestCase):
    def test_update_factor_diagnostics_records_aggregate_metrics_and_top_templates(self):
        population = np.zeros((2, 15), dtype=np.int32)
        population[:, 6] = 0
        population[:, 7] = MODE1_OPS.index("mean")
        population[:, 5] = MASK_RULES.index("none")
        population[:, 12] = CS_COMP_OPS.index("none")

        results = np.ones((2, 3, 4), dtype=np.float32)
        results[0] = np.nan

        perf_stats = {}
        old = os.environ.get("GP_V2_EVAL_DIAGNOSTICS_TOPK")
        try:
            os.environ["GP_V2_EVAL_DIAGNOSTICS_TOPK"] = "2"
            _update_factor_diagnostics(results, population, perf_stats)
        finally:
            if old is None:
                os.environ.pop("GP_V2_EVAL_DIAGNOSTICS_TOPK", None)
            else:
                os.environ["GP_V2_EVAL_DIAGNOSTICS_TOPK"] = old

        self.assertEqual(perf_stats["eval_all_nan_individual_count"], 1)
        self.assertAlmostEqual(perf_stats["eval_all_nan_individual_share"], 0.5, places=6)
        self.assertAlmostEqual(perf_stats["eval_finite_cell_ratio"], 0.5, places=6)
        self.assertIn("eval_nan_top_templates", perf_stats)
        self.assertEqual(len(perf_stats["eval_nan_top_templates"]), 1)
        top = perf_stats["eval_nan_top_templates"][0]
        self.assertEqual(top["count"], 1)
        self.assertEqual(top["mode"], 1)
        self.assertEqual(top["operator"], "mean")
        self.assertEqual(top["mask_rule"], "none")
        self.assertEqual(top["cs_comp_op"], "none")

    def test_evaluate_population_base_preserves_results_and_populates_diagnostics(self):
        population = np.zeros((2, 15), dtype=np.int32)
        population[:, 6] = 0
        population[:, 7] = MODE1_OPS.index("mean")
        population[:, 5] = MASK_RULES.index("none")
        population[:, 12] = CS_COMP_OPS.index("none")
        population[:, 2] = WINDOW_CHOICES.index(240)
        population[1, 0] = 1

        indicator_data = np.ones((2, 2, 1440, 3), dtype=np.float32)
        indicator_data[:, 1, :, :] = np.nan

        perf_stats = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = evaluate_population_base(population, indicator_data, perf_stats=perf_stats)

        self.assertTrue(np.isfinite(out[0]).all())
        self.assertTrue(np.isnan(out[1]).all())
        self.assertEqual(perf_stats["eval_all_nan_individual_count"], 1)
        self.assertAlmostEqual(perf_stats["eval_all_nan_individual_share"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
