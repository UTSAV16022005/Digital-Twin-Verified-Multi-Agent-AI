"""Unit tests for project-defined health states and diagnosis calibration."""

import unittest

import numpy as np
import pandas as pd

from src.diagnosis import compute_calibration_metrics, health_state_from_rul


class DiagnosisTests(unittest.TestCase):
    def test_health_state_threshold_edges(self) -> None:
        actual = health_state_from_rul(np.array([126, 125, 26, 25, 0]))
        self.assertEqual(actual.tolist(), ["healthy", "degraded", "degraded", "critical", "critical"])

    def test_health_state_rejects_missing_rul(self) -> None:
        with self.assertRaises(ValueError):
            health_state_from_rul(np.array([10.0, np.nan]))

    def test_calibration_aligns_missing_probability_class(self) -> None:
        labels = np.array(["healthy", "critical"])
        probabilities = pd.DataFrame({"healthy": [0.9, 0.1], "critical": [0.1, 0.9]})
        result = compute_calibration_metrics(labels, probabilities)
        self.assertGreaterEqual(result["brier_score"], 0.0)
        self.assertLessEqual(result["expected_calibration_error"], 1.0)

    def test_calibration_rejects_invalid_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            compute_calibration_metrics(np.array(["healthy"]), np.array([[0.0, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
