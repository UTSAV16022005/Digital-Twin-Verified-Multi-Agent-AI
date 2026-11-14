"""Unit checks for transparent Digital Twin action-effect sensitivity scales."""

import unittest

from scripts.analyze_action_assumption_sensitivity import scaled_action_specs
from src.digital_twin import DEFAULT_ACTIONS


class ActionAssumptionSensitivityTests(unittest.TestCase):
    def test_zero_effect_scale_matches_continue_for_operating_effects(self) -> None:
        actions = scaled_action_specs(DEFAULT_ACTIONS, 0.0, 0.0)
        for name in ("reduce_thrust", "hold_for_inspection"):
            self.assertEqual(actions[name].transition_multiplier, 1.0)
            self.assertEqual(actions[name].degradation_multiplier, 1.0)
        self.assertEqual(
            actions["controlled_shutdown"], DEFAULT_ACTIONS["controlled_shutdown"]
        )

    def test_unit_scale_reproduces_registered_action_assumptions(self) -> None:
        actions = scaled_action_specs(DEFAULT_ACTIONS, 1.0, 1.0)
        for name, spec in DEFAULT_ACTIONS.items():
            self.assertEqual(actions[name], spec)

    def test_extrapolated_effects_are_clipped_to_physical_multiplier_range(self) -> None:
        actions = scaled_action_specs(DEFAULT_ACTIONS, 3.0, 3.0)
        self.assertGreaterEqual(actions["reduce_thrust"].transition_multiplier, 0.0)
        self.assertLessEqual(actions["reduce_thrust"].transition_multiplier, 1.0)
        self.assertGreaterEqual(actions["hold_for_inspection"].degradation_multiplier, 0.0)
        self.assertLessEqual(actions["hold_for_inspection"].degradation_multiplier, 1.0)
        self.assertEqual(actions["continue_operation"], DEFAULT_ACTIONS["continue_operation"])

    def test_rejects_negative_effect_scales(self) -> None:
        with self.assertRaises(ValueError):
            scaled_action_specs(DEFAULT_ACTIONS, -0.1, 1.0)


if __name__ == "__main__":
    unittest.main()
