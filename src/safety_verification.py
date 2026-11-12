"""Independent safety verification for Digital Twin action outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import math

from src.digital_twin import TransitionOutcome


@dataclass(frozen=True)
class SafetyConfig:
    max_failure_risk: float = 0.25
    max_model_uncertainty: float = 6.0
    max_projected_drift: float = 5.0
    allow_controlled_shutdown: bool = True


@dataclass
class SafetyDecision:
    action: str
    approved: bool
    violations: list[str]
    margins: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "approved": self.approved,
            "violations": self.violations,
            "margins": self.margins,
        }


class SafetyVerifier:
    """Apply hard, deterministic constraints independently of the planner."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def verify(self, outcome: TransitionOutcome) -> SafetyDecision:
        violations = list(outcome.violations)
        risk = float(outcome.predicted_failure_risk)
        uncertainty = float(outcome.model_uncertainty)
        drift = float(outcome.projected_drift)
        if not math.isfinite(risk):
            violations.append("non_finite_failure_risk")
        elif risk < 0.0 or risk > 1.0:
            violations.append("failure_risk_out_of_range")
        elif risk > self.config.max_failure_risk:
            violations.append("failure_risk_above_safety_limit")
        if not math.isfinite(uncertainty) or uncertainty < 0.0:
            violations.append("invalid_model_uncertainty")
        elif uncertainty > self.config.max_model_uncertainty:
            violations.append("model_uncertainty_above_safety_limit")
        if not math.isfinite(drift) or drift < 0.0:
            violations.append("invalid_projected_drift")
        elif drift > self.config.max_projected_drift:
            violations.append("projected_drift_above_safety_limit")
        if not outcome.predicted_state.feature_values.size or not all(
            math.isfinite(float(value)) for value in outcome.predicted_state.feature_values
        ):
            violations.append("non_finite_predicted_state")
        if outcome.action == "controlled_shutdown" and not self.config.allow_controlled_shutdown:
            violations.append("controlled_shutdown_not_permitted")

        return SafetyDecision(
            action=outcome.action,
            approved=not violations,
            violations=violations,
            margins={
                "failure_risk": self.config.max_failure_risk - risk if math.isfinite(risk) else 0.0,
                "model_uncertainty": self.config.max_model_uncertainty - uncertainty if math.isfinite(uncertainty) else 0.0,
                "projected_drift": self.config.max_projected_drift - drift if math.isfinite(drift) else 0.0,
            },
        )

    def verify_all(self, outcomes: list[TransitionOutcome]) -> dict[str, SafetyDecision]:
        return {outcome.action: self.verify(outcome) for outcome in outcomes}

