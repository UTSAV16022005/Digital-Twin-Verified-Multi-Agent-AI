"""Small deterministic checks for the hand-written sequence RUL model."""

import unittest

import numpy as np

import pandas as pd

from src.prognostics import (
    NumpyLSTMRULModel,
    PrognosticsAgent,
    PrognosticsConfig,
    TorchSequenceRULModel,
)


class NumpyLSTMRULTests(unittest.TestCase):
    def test_sequence_model_trains_and_predicts_finite_values(self) -> None:
        rng = np.random.default_rng(11)
        train_x = rng.normal(size=(12, 5, 2)).astype(np.float32)
        train_y = (30.0 + 4.0 * train_x[:, -1, 0]).astype(np.float32)
        model = NumpyLSTMRULModel(in_features=2, hidden_size=4, seed=11)
        model.fit(train_x, train_y, max_epochs=2, batch_size=4, lr=1e-3)
        predictions = model.predict(train_x)
        self.assertAlmostEqual(model.target_mean_, float(train_y.mean()), places=5)
        self.assertGreater(model.target_scale_, 0.0)
        self.assertEqual(predictions.shape, train_y.shape)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())

    def test_prognostics_agent_fits_lstm_with_engine_age_channel(self) -> None:
        rows = []
        for unit_id in range(1, 9):
            for cycle in range(1, 15):
                rows.append(
                    {
                        "unit_id": unit_id,
                        "cycle": cycle,
                        "sensor_1": float(np.sin(cycle / 5.0) + unit_id / 100.0),
                        "rul": float(20 + unit_id - cycle),
                    }
                )
        frame = pd.DataFrame(rows)
        config = PrognosticsConfig(
            model_type="lstm",
            window_size=5,
            include_cycle_feature=True,
            lstm_hidden_size=4,
            lstm_epochs=2,
            lstm_batch_size=8,
            lstm_training_stride=2,
            lstm_validation_fraction=0.25,
        )
        agent = PrognosticsAgent.fit(frame, ["sensor_1"], config)

        telemetry = frame[frame["unit_id"] == 1].drop(columns="rul")
        predictions = agent.predict_frame(telemetry)
        self.assertEqual(agent.model.in_features, 2)
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.isfinite(predictions["predicted_rul"]).all())
        self.assertGreaterEqual(agent.model.best_epoch_, 1)

    def test_prognostics_agent_rejects_unknown_model_type(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": np.repeat([1, 2], 5),
                "cycle": list(range(1, 6)) * 2,
                "sensor_1": np.linspace(0.0, 1.0, 10),
                "rul": np.tile(np.arange(4, -1, -1), 2),
            }
        )
        with self.assertRaises(ValueError):
            PrognosticsAgent.fit(
                frame,
                ["sensor_1"],
                PrognosticsConfig(model_type="unsupported", window_size=3),
            )


class TorchSequencePrognosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for unit_id in range(1, 9):
            for cycle in range(1, 15):
                rows.append(
                    {
                        "unit_id": unit_id,
                        "cycle": cycle,
                        "sensor_1": float(np.sin(cycle / 5.0) + unit_id / 100.0),
                        "sensor_2": float(np.cos(cycle / 7.0) + unit_id / 80.0),
                        "rul": float(20 + unit_id - cycle),
                    }
                )
        self.frame = pd.DataFrame(rows)

    def test_explicit_sequence_early_stop_group_must_be_engine_disjoint(self) -> None:
        config = PrognosticsConfig(
            model_type="pytorch_cnn",
            window_size=5,
            torch_hidden_size=8,
            torch_epochs=1,
            torch_training_stride=2,
            torch_validation_fraction=0.0,
        )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            PrognosticsAgent.fit(
                self.frame,
                ["sensor_1", "sensor_2"],
                config,
                validation_frame=self.frame,
            )

    def test_pytorch_cnn_and_transformer_train_and_predict(self) -> None:
        try:
            import torch

            torch.zeros(1)
        except Exception:
            self.skipTest("PyTorch is not available in this environment")
        for model_type in ("pytorch_cnn", "pytorch_transformer"):
            with self.subTest(model_type=model_type):
                config = PrognosticsConfig(
                    model_type=model_type,
                    window_size=5,
                    random_state=13,
                    torch_hidden_size=8,
                    torch_epochs=2,
                    torch_batch_size=8,
                    torch_training_stride=2,
                    torch_validation_fraction=0.25,
                    torch_early_stopping_patience=2,
                    torch_attention_heads=2,
                    torch_transformer_layers=1,
                )
                train = self.frame[self.frame["unit_id"] <= 6].copy()
                early_stop = self.frame[self.frame["unit_id"] > 6].copy()
                agent = PrognosticsAgent.fit(
                    train,
                    ["sensor_1", "sensor_2"],
                    config,
                    validation_frame=early_stop,
                )
                predictions = agent.predict_frame(
                    self.frame[self.frame["unit_id"] == 1].drop(columns="rul")
                )
                self.assertEqual(len(predictions), 10)
                self.assertTrue(np.isfinite(predictions["predicted_rul"]).all())
                self.assertGreaterEqual(agent.model.best_epoch_, 1)

    def test_xgboost_model_fits_the_prognostics_agent_interface(self) -> None:
        try:
            import xgboost

            xgboost.XGBRegressor(n_estimators=1, n_jobs=1)
        except Exception:
            self.skipTest("XGBoost is an optional research dependency")
        config = PrognosticsConfig(
            model_type="xgboost",
            window_size=5,
            random_state=13,
            xgb_n_estimators=10,
            xgb_max_depth=2,
        )
        agent = PrognosticsAgent.fit(
            self.frame, ["sensor_1", "sensor_2"], config
        )
        predictions = agent.predict_frame(
            self.frame[self.frame["unit_id"] == 1].drop(columns="rul")
        )
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.isfinite(predictions["predicted_rul"]).all())


if __name__ == "__main__":
    unittest.main()
