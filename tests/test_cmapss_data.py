"""Window-construction invariants for engine-level evaluation."""

import unittest

import pandas as pd

from src.cmapss_data import make_windows


class CmapssWindowTests(unittest.TestCase):
    def test_strided_windows_include_final_observation_per_engine(self) -> None:
        frame = pd.DataFrame({
            "unit_id": [1] * 8,
            "cycle": list(range(1, 9)),
            "sensor_1": [float(cycle) for cycle in range(1, 9)],
            "rul": [float(8 - cycle) for cycle in range(1, 9)],
        })
        _, targets, metadata = make_windows(frame, ["sensor_1"], window_size=3, stride=4)
        self.assertEqual(metadata["end_cycle"].tolist(), [3, 7, 8])
        self.assertEqual(float(targets[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
