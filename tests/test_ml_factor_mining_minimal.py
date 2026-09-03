import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ML_FACTOR_DIR = REPO_ROOT / "factor_analyse" / "ML_factor_mining"
ML_UTIL_PATH = ML_FACTOR_DIR / "util_factor.py"
_spec = importlib.util.spec_from_file_location("ml_util_factor_module", ML_UTIL_PATH)
ml_util_factor = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(ml_util_factor)


class MLFactorMiningMinimalTest(unittest.TestCase):
    def test_apply_label_purge_to_split(self):
        def _sample(dt: str):
            return (np.array([0.0], dtype=float), 0.0, 0.0, "BTCUSDT", pd.Timestamp(dt))

        train_data = [_sample(f"2024-01-{d:02d}") for d in range(1, 11)]   # 1~10
        val_data = [_sample(f"2024-01-{d:02d}") for d in range(11, 16)]    # 11~15
        test_data = [_sample(f"2024-01-{d:02d}") for d in range(16, 21)]   # 16~20

        train_out, val_out, test_out = ml_util_factor.apply_label_purge_to_split(
            train_data, val_data, test_data, purge_days=3, date_index=4
        )

        self.assertEqual(len(test_out), len(test_data))
        self.assertLessEqual(max(x[4] for x in train_out), pd.Timestamp("2024-01-07"))
        self.assertLessEqual(max(x[4] for x in val_out), pd.Timestamp("2024-01-12"))

    def test_dedupe_predictions_keep_latest_window(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
                "window_idx": [0, 1, 0],
                "xgb_prediction": [0.1, 0.2, 0.3],
            }
        )
        out, removed = ml_util_factor.dedupe_predictions_keep_latest_window(df)
        self.assertEqual(removed, 1)
        self.assertEqual(len(out), 2)
        kept = out[out["symbol"] == "BTCUSDT"].iloc[0]
        self.assertEqual(int(kept["window_idx"]), 1)

    def test_inference_improved_branch_returns_immediately(self):
        n = 140
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        base = 100 + np.linspace(0, 5, n)
        df = pd.DataFrame(
            {
                "date": dates,
                "symbol": ["BTCUSDT"] * n,
                "open": base,
                "high": base * 1.01,
                "low": base * 0.99,
                "close": base * (1 + 0.001 * np.sin(np.arange(n))),
                "volume": 1000 + np.arange(n),
                "quote_volume": (1000 + np.arange(n)) * base,
                "taker_buy_quote": (1000 + np.arange(n)) * base * 0.52,
                "taker_buy_base": (1000 + np.arange(n)) * 0.51,
                "trades_count": 100 + (np.arange(n) % 20),
            }
        )
        out = ml_util_factor.create_features_symbol_inference_latest(
            df, min_history_days=60, use_improved_features=True
        )
        self.assertEqual(len(out), 1)
        self.assertIn("momentum_5", out.columns)
        # improved 分支不应落入完整版特征分支
        self.assertNotIn("open", out.columns)

    def test_train_enhanced_xgboost_model_params_fallback_without_bayes(self):
        if not getattr(ml_util_factor, "XGB_AVAILABLE", False):
            self.skipTest("xgboost unavailable in current environment")

        class _DummyModel:
            def predict(self, dmatrix):
                return np.zeros(len(dmatrix["X"]), dtype=float)

        def _fake_dmatrix(X, label=None):
            return {"X": np.asarray(X), "label": label}

        def _fake_train(*args, **kwargs):
            return _DummyModel()

        X_train = np.random.randn(40, 6).astype(np.float32)
        y_train = np.random.randn(40).astype(np.float32)
        X_val = np.random.randn(20, 6).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32)
        symbols_train = ["BTCUSDT"] * 40
        symbols_val = ["BTCUSDT"] * 20
        dates_train = pd.date_range("2024-01-01", periods=40, freq="D").tolist()
        dates_val = pd.date_range("2024-02-10", periods=20, freq="D").tolist()

        mock_params = {
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 10,
            "early_stopping_rounds": 3,
        }

        with mock.patch.object(ml_util_factor, "get_default_xgb_params", return_value=mock_params), \
             mock.patch.object(ml_util_factor.xgb, "DMatrix", side_effect=_fake_dmatrix), \
             mock.patch.object(ml_util_factor.xgb, "train", side_effect=_fake_train):
            model, val_pred = ml_util_factor.train_enhanced_xgboost_model(
                X_train,
                y_train,
                X_val,
                y_val,
                symbols_train,
                symbols_val,
                dates_train,
                dates_val,
                use_bayesian_opt=False,
                use_robust_training=False,
                regularization_strength="default",
            )

        self.assertIsNotNone(model)
        self.assertEqual(len(val_pred), len(y_val))

    def test_lgbm_rolling_script_wires_purge_and_dedupe(self):
        lgbm_path = (
            REPO_ROOT
            / "factor_analyse"
            / "ML_factor_mining"
            / "LGBM_Prediction_rolling_0302.py"
        )
        text = lgbm_path.read_text(encoding="utf-8")
        self.assertIn("apply_label_purge_to_split(", text)
        self.assertIn("dedupe_predictions_keep_latest_window(", text)


if __name__ == "__main__":
    unittest.main()
