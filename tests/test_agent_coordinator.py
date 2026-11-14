"""Contract checks for telemetry-only coordinated agent inputs."""

import unittest
from types import SimpleNamespace

import pandas as pd

from src.agent_coordinator import HealthAgentCoordinator
from src.closed_loop import ClosedLoopAbstention


class HealthAgentCoordinatorInputTests(unittest.TestCase):
    def make_coordinator(self) -> HealthAgentCoordinator:
        orchestrator = SimpleNamespace(
            prognostics=SimpleNamespace(feature_columns=[])
        )
        return HealthAgentCoordinator(
            monitor=None,
            diagnosis=None,
            orchestrator=orchestrator,
            diagnosis_scaler=None,
        )

    def test_adapts_lstm_monitor_confirmation_to_common_alert_trace(self) -> None:
        class LSTMStyleMonitor:
            threshold = 0.5

            def monitor(self, frame: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "anomaly_score": [0.75],
                        "threshold": [0.5],
                        "confirmed_alert": [True],
                    }
                )

        class AbstainingOrchestrator:
            prognostics = SimpleNamespace(
                feature_columns=["sensor_1"],
                config=SimpleNamespace(window_size=30),
            )

            def step(self, telemetry, anomaly_score, **kwargs):
                self.last_anomaly_score = anomaly_score
                return ClosedLoopAbstention(
                    unit_id=1,
                    cycle=1,
                    decision_status="insufficient_telemetry",
                    available_cycles=1,
                    required_cycles=30,
                    rationale="Test stub abstention.",
                )

        orchestrator = AbstainingOrchestrator()
        coordinator = HealthAgentCoordinator(
            monitor=LSTMStyleMonitor(),
            diagnosis=None,
            orchestrator=orchestrator,
            diagnosis_scaler=None,
        )
        result = coordinator.step(
            pd.DataFrame({"unit_id": [1], "cycle": [1], "sensor_1": [0.2]})
        )

        self.assertEqual(orchestrator.last_anomaly_score, 0.75)
        self.assertTrue(result.agent_trace[0]["output"]["alert"])
        self.assertEqual(result.agent_trace[0]["output"]["anomaly_score"], 0.75)

    def test_rejects_target_leakage_before_calling_any_agents(self) -> None:
        telemetry = pd.DataFrame(
            {"unit_id": [1], "cycle": [1], "sensor_1": [0.1], "rul": [0.0]}
        )
        with self.assertRaisesRegex(ValueError, "telemetry only"):
            self.make_coordinator().step(telemetry)

    def test_requires_a_single_engine_per_coordinated_step(self) -> None:
        telemetry = pd.DataFrame(
            {"unit_id": [1, 2], "cycle": [1, 1], "sensor_1": [0.1, 0.2]}
        )
        with self.assertRaisesRegex(ValueError, "exactly one engine"):
            self.make_coordinator().step(telemetry)


if __name__ == "__main__":
    unittest.main()
