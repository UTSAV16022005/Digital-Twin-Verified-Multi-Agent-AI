"""Integration checks for state estimation, simulation, planning, and safety."""

import unittest

import numpy as np
import pandas as pd

from src.digital_twin import DataDrivenTransitionModel, DigitalTwin
from src.recovery_planning import RecoveryPlanningAgent
from src.safety_verification import SafetyConfig, SafetyVerifier


def toy_telemetry() -> pd.DataFrame:
    rows = []
    for unit_id in (1, 2):
        for cycle in range(1, 8):
            rows.append({"unit_id": unit_id, "cycle": cycle, "sensor_1": 0.1 * cycle + 0.01 * unit_id})
    return pd.DataFrame(rows)


class DigitalTwinIntegrationTests(unittest.TestCase):
    def make_twin(self, max_sensor_z: float = 5.0) -> DigitalTwin:
        frame = toy_telemetry()
        model = DataDrivenTransitionModel(["sensor_1"], alpha=0.1).fit(frame)
        return DigitalTwin(model, ["sensor_1"], max_sensor_z=max_sensor_z)

    def make_state(self, twin: DigitalTwin):
        window = np.array([[0.2], [0.3], [0.4]], dtype=float)
        return twin.estimate_state(window, unit_id=7, end_cycle=3, predicted_rul=40.0)

    def test_estimates_state_simulates_candidates_and_verifies_safe_action(self) -> None:
        twin = self.make_twin()
        state = self.make_state(twin)
        self.assertEqual(state.cycle, 3)
        self.assertEqual(state.feature_values.shape, (1,))

        outcomes = twin.what_if(state, horizon=3)
        self.assertEqual({item.action for item in outcomes}, set(twin.actions))
        self.assertTrue(all(np.isfinite(item.predicted_state.feature_values).all() for item in outcomes))

        decisions = SafetyVerifier(
            SafetyConfig(max_failure_risk=0.5, max_model_uncertainty=10.0, max_projected_drift=10.0)
        ).verify_all(outcomes)
        self.assertTrue(decisions["reduce_thrust"].approved)

        plan = RecoveryPlanningAgent(risk_budget=0.5).plan(twin, state, horizon=3)
        self.assertEqual(plan.decision_status, "recommendation")
        self.assertIn(plan.selected_action, twin.actions)

    def test_twin_constraint_violation_is_rejected_by_planner(self) -> None:
        twin = self.make_twin(max_sensor_z=0.0)
        state = self.make_state(twin)
        outcomes = twin.what_if(state, actions=["continue_operation"], horizon=3)
        self.assertFalse(outcomes[0].safe_under_constraints)
        self.assertIn("predicted_sensor_state_outside_5_sigma_envelope", outcomes[0].violations)

        plan = RecoveryPlanningAgent(risk_budget=1.0).plan(
            twin, state, horizon=3, actions=["continue_operation"]
        )
        self.assertEqual(plan.decision_status, "no_safe_action")
        self.assertEqual(plan.selected_action, None)
        self.assertEqual(len(plan.rejected_candidates), 1)


if __name__ == "__main__":
    unittest.main()
