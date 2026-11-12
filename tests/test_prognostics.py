"""Small deterministic checks for the hand-written sequence RUL model."""

import unittest

import numpy as np

from src.prognostics import NumpyLSTMRULModel


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


if __name__ == "__main__":
    unittest.main()
