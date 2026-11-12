"""Reproducibility checks for grouped research-evaluation helpers."""

import unittest

import numpy as np
import pandas as pd

from scripts.run_research_evaluation import OperatingConditionNormalizer, _bootstrap_mean_ci
from src.cmapss_data import OPERATING_COLUMNS, SENSOR_COLUMNS


class ResearchEvaluationTests(unittest.TestCase):
    def test_engine_bootstrap_is_reproducible(self) -> None:
        first = _bootstrap_mean_ci(np.array([1.0, 2.0, 3.0]), seed=17, replicates=500)
        second = _bootstrap_mean_ci(np.array([1.0, 2.0, 3.0]), seed=17, replicates=500)
        self.assertEqual(first, second)
        self.assertEqual(first["n_engines"], 3)

    def test_condition_normalizer_fits_and_transforms_sensor_values(self) -> None:
        rows = []
        for setting in (0.0, 10.0):
            for offset in range(5):
                row = {name: setting for name in OPERATING_COLUMNS}
                row.update({name: setting + offset + index for index, name in enumerate(SENSOR_COLUMNS)})
                rows.append(row)
        frame = pd.DataFrame(rows)
        frame = frame.astype({name: "int64" for name in SENSOR_COLUMNS})
        transformed = OperatingConditionNormalizer(seed=17, max_clusters=2).fit(frame).transform(frame)
        self.assertEqual(transformed.shape, frame.shape)
        self.assertTrue(np.isfinite(transformed[SENSOR_COLUMNS].to_numpy()).all())
        self.assertTrue(np.array_equal(frame[OPERATING_COLUMNS].to_numpy(), transformed[OPERATING_COLUMNS].to_numpy()))


if __name__ == "__main__":
    unittest.main()
