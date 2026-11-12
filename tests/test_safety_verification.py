"""Adversarial unit tests for the independent Safety Verifier."""

import unittest

import numpy as np

from src.digital_twin import TransitionOutcome, TwinState
from src.safety_verification import SafetyConfig, SafetyVerifier


def make_outcome(
    *,
    action: str = "continue_operation",
    risk: float = 0.10,
    uncertainty: float = 1.0,
    drift: float = 1.0,
    violations: list[str] | None = None,
) -> TransitionOutcome:
    state = TwinState(
        unit_id=1,
        cycle=100,
        feature_columns=["sensor_1"],
        feature_values=np.array([0.0]),
        feature_slopes=np.array([0.0]),
        predicted_rul=50.0,
        health_index=0.6,
    )
    return TransitionOutcome(
        action=action,
        horizon=5,
        predicted_state=state,
        predicted_failure_risk=risk,
        model_uncertainty=uncertainty,
        projected_drift=drift,
        mission_cost=0.1,
        safe_under_constraints=not violations,
        violations=violations or [],
    )


class SafetyVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = SafetyVerifier(
            SafetyConfig(
                max_failure_risk=0.25,
                max_model_uncertainty=6.0,
                max_projected_drift=5.0,
                allow_controlled_shutdown=True,
            )
        )

    def test_accepts_valid_outcome(self) -> None:
        decision = self.verifier.verify(make_outcome())
        self.assertTrue(decision.approved)
        self.assertEqual(decision.violations, [])

    def test_accepts_safe_recovery_action(self) -> None:
        decision = self.verifier.verify(make_outcome(action="reduce_thrust"))
        self.assertTrue(decision.approved)
        self.assertEqual(decision.action, "reduce_thrust")

    def test_rejects_high_failure_risk(self) -> None:
        decision = self.verifier.verify(make_outcome(risk=0.80))
        self.assertFalse(decision.approved)
        self.assertIn("failure_risk_above_safety_limit", decision.violations)

    def test_rejects_non_finite_failure_risk(self) -> None:
        decision = self.verifier.verify(make_outcome(risk=float("nan")))
        self.assertFalse(decision.approved)
        self.assertIn("non_finite_failure_risk", decision.violations)

    def test_rejects_non_finite_predicted_twin_state(self) -> None:
        outcome = make_outcome()
        outcome.predicted_state.feature_values[0] = float("inf")
        decision = self.verifier.verify(outcome)
        self.assertFalse(decision.approved)
        self.assertIn("non_finite_predicted_state", decision.violations)

    def test_rejects_high_uncertainty(self) -> None:
        decision = self.verifier.verify(make_outcome(uncertainty=7.0))
        self.assertFalse(decision.approved)
        self.assertIn("model_uncertainty_above_safety_limit", decision.violations)

    def test_rejects_high_projected_drift(self) -> None:
        decision = self.verifier.verify(make_outcome(drift=6.0))
        self.assertFalse(decision.approved)
        self.assertIn("projected_drift_above_safety_limit", decision.violations)

    def test_propagates_twin_constraint_violation(self) -> None:
        decision = self.verifier.verify(
            make_outcome(violations=["predicted_failure_within_simulation_horizon"])
        )
        self.assertFalse(decision.approved)
        self.assertIn("predicted_failure_within_simulation_horizon", decision.violations)

    def test_shutdown_policy_can_reject_shutdown(self) -> None:
        verifier = SafetyVerifier(SafetyConfig(allow_controlled_shutdown=False))
        decision = verifier.verify(make_outcome(action="controlled_shutdown"))
        self.assertFalse(decision.approved)
        self.assertIn("controlled_shutdown_not_permitted", decision.violations)


if __name__ == "__main__":
    unittest.main()

