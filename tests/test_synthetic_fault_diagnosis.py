"""Unit checks for the synthetic sensor-corruption classifier contract."""

import unittest

import numpy as np

from src.synthetic_fault_diagnosis import SyntheticSensorFaultAgent


class SyntheticSensorFaultAgentTests(unittest.TestCase):
    def test_fit_predict_returns_ordered_finite_probabilities(self) -> None:
        rng = np.random.default_rng(5)
        labels = np.repeat(["normal", "bias", "drift"], 20)
        features = rng.normal(0.0, 0.05, size=(len(labels), 3)).astype(np.float32)
        features[labels == "bias", 0] += 2.0
        features[labels == "drift", 1] += 2.0
        agent = SyntheticSensorFaultAgent.fit(
            features,
            labels,
            feature_names=["last_sensor_1", "slope_sensor_1", "std_sensor_1"],
            n_estimators=10,
            random_state=5,
        )

        probabilities = agent.predict_proba(features[:6])
        predictions, confidence = agent.predict(features[:6])
        self.assertEqual(list(probabilities.columns), list(agent.labels))
        self.assertEqual(probabilities.shape, (6, len(agent.labels)))
        self.assertTrue(np.isfinite(probabilities.to_numpy()).all())
        np.testing.assert_allclose(probabilities.sum(axis=1).to_numpy(), 1.0)
        self.assertEqual(predictions.shape, (6,))
        self.assertTrue(np.isfinite(confidence).all())

    def test_fit_rejects_labels_outside_synthetic_vocabulary(self) -> None:
        with self.assertRaises(ValueError):
            SyntheticSensorFaultAgent.fit(
                np.ones((4, 2), dtype=np.float32),
                np.asarray(["normal", "normal", "physical_fault", "physical_fault"]),
                feature_names=["last_sensor_1", "mean_sensor_1"],
                n_estimators=5,
            )


if __name__ == "__main__":
    unittest.main()
