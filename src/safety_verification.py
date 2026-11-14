"""Independent safety verification for Digital Twin action outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from src.digital_twin import TransitionOutcome


@dataclass(frozen=True)
class SafetyConfig:
    max_failure_risk: float = 0.25
    max_model_uncertainty: float = 6.0
    max_projected_drift: float = 5.0
    allow_controlled_shutdown: bool = True

    def __post_init__(self) -> None:
        limits = (
            ("max_failure_risk", self.max_failure_risk, 0.0, 1.0),
            ("max_model_uncertainty", self.max_model_uncertainty, 0.0, None),
            ("max_projected_drift", self.max_projected_drift, 0.0, None),
        )
        for name, value, minimum, maximum in limits:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite real number")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"{name} must be a finite real number")
            if numeric_value < minimum or (
                maximum is not None and numeric_value > maximum
            ):
                if maximum is None:
                    raise ValueError(f"{name} must be at least {minimum}")
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if not isinstance(self.allow_controlled_shutdown, bool):
            raise ValueError("allow_controlled_shutdown must be a boolean")


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

    @staticmethod
    def _number(value: object) -> float:
        """Convert a numeric field safely; malformed values fail later checks."""

        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return float("nan")

    def verify(self, outcome: TransitionOutcome) -> SafetyDecision:
        raw_twin_violations = outcome.violations
        if isinstance(raw_twin_violations, (list, tuple)) and all(
            isinstance(item, str) and item.strip() for item in raw_twin_violations
        ):
            twin_violations = list(dict.fromkeys(raw_twin_violations))
            violations = list(twin_violations)
        else:
            twin_violations = []
            violations = ["invalid_twin_violation_list"]
        risk = self._number(outcome.predicted_failure_risk)
        uncertainty = self._number(outcome.model_uncertainty)
        drift = self._number(outcome.projected_drift)
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

        state = outcome.predicted_state
        try:
            feature_values = np.asarray(state.feature_values, dtype=np.float64)
            feature_slopes = np.asarray(state.feature_slopes, dtype=np.float64)
            feature_columns = list(state.feature_columns)
        except (AttributeError, TypeError, ValueError):
            feature_values = np.asarray([], dtype=np.float64)
            feature_slopes = np.asarray([], dtype=np.float64)
            feature_columns = []
            violations.append("invalid_predicted_state_shape")
        if (
            feature_values.ndim != 1
            or feature_slopes.ndim != 1
            or feature_values.size == 0
            or feature_values.size != feature_slopes.size
            or feature_values.size != len(feature_columns)
            or any(not isinstance(column, str) or not column for column in feature_columns)
            or len(set(feature_columns)) != len(feature_columns)
        ):
            violations.append("invalid_predicted_state_shape")
        if not np.isfinite(feature_values).all() or not np.isfinite(feature_slopes).all():
            violations.append("non_finite_predicted_state")

        predicted_rul = self._number(getattr(state, "predicted_rul", float("nan")))
        if not math.isfinite(predicted_rul) or predicted_rul < 0.0:
            violations.append("invalid_predicted_rul")
        health_index = self._number(getattr(state, "health_index", float("nan")))
        if not math.isfinite(health_index) or not 0.0 <= health_index <= 1.0:
            violations.append("invalid_predicted_health_index")
        anomaly_score = getattr(state, "anomaly_score", None)
        if anomaly_score is not None and not math.isfinite(self._number(anomaly_score)):
            violations.append("invalid_anomaly_score")

        mission_cost = self._number(outcome.mission_cost)
        if not math.isfinite(mission_cost) or mission_cost < 0.0:
            violations.append("invalid_mission_cost")
        if (
            isinstance(outcome.horizon, bool)
            or not isinstance(outcome.horizon, int)
            or outcome.horizon < 1
        ):
            violations.append("invalid_simulation_horizon")
        if not isinstance(outcome.action, str) or not outcome.action.strip():
            violations.append("invalid_action_name")

        if not isinstance(outcome.safe_under_constraints, bool):
            violations.append("invalid_twin_safety_flag")
        elif not outcome.safe_under_constraints and not twin_violations:
            violations.append("twin_reported_unsafe")
        if outcome.action == "controlled_shutdown" and not self.config.allow_controlled_shutdown:
            violations.append("controlled_shutdown_not_permitted")

        violations = list(dict.fromkeys(violations))

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

