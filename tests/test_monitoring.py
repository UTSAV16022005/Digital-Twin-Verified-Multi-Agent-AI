"""Monitoring baseline regression checks."""

import unittest

import pandas as pd

from src.monitoring import PCAMonitoringAgent


class MonitoringBaselineTests(unittest.TestCase):
    def test_persistent_alerts_accept_filtered_non_contiguous_indices(self) -> None:
        frame = pd.DataFrame([
            {
                "unit_id": unit,
                "cycle": cycle,
                "sensor_1": float(cycle + unit),
                "sensor_2": float(2 * cycle + unit),
            }
            for unit in (1, 2)
            for cycle in range(1, 11)
        ])
        agent = PCAMonitoringAgent.fit(frame[frame["unit_id"] == 1], ["sensor_1", "sensor_2"], healthy_cycles=5)
        filtered = frame[frame["unit_id"] == 2]
        self.assertNotEqual(filtered.index.tolist(), list(range(len(filtered))))
        scored = agent.score(filtered, persistence=3)
        self.assertEqual(len(scored), len(filtered))
        self.assertTrue(scored["confirmed_alert"].isin([True, False]).all())


if __name__ == "__main__":
    unittest.main()
