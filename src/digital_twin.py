"""A transparent, data-driven Digital Twin for C-MAPSS what-if simulation.

The public C-MAPSS benchmark contains telemetry and degradation trajectories but
does not contain recovery actions. This module therefore learns the baseline
one-step telemetry transition from training engines and applies explicit,
documented action modifiers for what-if analysis. It is a research prototype,
not a flight-certified physical simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class ActionSpec:
    """Transparent assumptions for an action in the prototype twin."""

    name: str
    description: str
    transition_multiplier: float
    degradation_multiplier: float
    mission_cost: float
    keeps_operating: bool


DEFAULT_ACTIONS = {
    "continue_operation": ActionSpec(
        name="continue_operation",
        description="Continue at the current operating profile.",
        transition_multiplier=1.0,
        degradation_multiplier=1.0,
        mission_cost=0.0,
        keeps_operating=True,
    ),
    "reduce_thrust": ActionSpec(
        name="reduce_thrust",
        description="Reduce demand to slow the learned degradation trend.",
        transition_multiplier=0.70,
        degradation_multiplier=0.70,
        mission_cost=0.20,
        keeps_operating=True,
    ),
    "hold_for_inspection": ActionSpec(
        name="hold_for_inspection",
        description="Hold the current state while inspection or diagnosis runs.",
        transition_multiplier=0.25,
        degradation_multiplier=0.25,
        mission_cost=0.45,
        keeps_operating=True,
    ),
    "controlled_shutdown": ActionSpec(
        name="controlled_shutdown",
        description="Stop operation and freeze the simulated state.",
        transition_multiplier=0.0,
        degradation_multiplier=0.0,
        # Shutdown is safe but has a high mission-interruption cost, so the
        # planner uses it as a fallback when continued operation is unsafe.
        mission_cost=3.0,
        keeps_operating=False,
    ),
}


@dataclass
class TwinState:
    """Estimated state consumed by the Digital Twin."""

    unit_id: int
    cycle: int
    feature_columns: list[str]
    feature_values: np.ndarray
    feature_slopes: np.ndarray
    predicted_rul: float
    health_index: float
    anomaly_score: float | None = None
    operating: bool = True

    def as_dict(self) -> dict:
        result = asdict(self)
        result["feature_values"] = self.feature_values.tolist()
        result["feature_slopes"] = self.feature_slopes.tolist()
        return result


@dataclass
class TransitionOutcome:
    """Result of simulating one candidate action."""

    action: str
    horizon: int
    predicted_state: TwinState
    predicted_failure_risk: float
    model_uncertainty: float
    projected_drift: float
    mission_cost: float
    safe_under_constraints: bool
    violations: list[str]

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "horizon": self.horizon,
            "predicted_state": self.predicted_state.as_dict(),
            "predicted_failure_risk": self.predicted_failure_risk,
            "model_uncertainty": self.model_uncertainty,
            "projected_drift": self.projected_drift,
            "mission_cost": self.mission_cost,
            "safe_under_constraints": self.safe_under_constraints,
            "violations": self.violations,
        }


class DataDrivenTransitionModel:
    """Learn a one-step state transition from engine telemetry."""

    def __init__(self, feature_columns: list[str], alpha: float = 1.0) -> None:
        self.feature_columns = feature_columns
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.residual_std_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "DataDrivenTransitionModel":
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for _, unit_frame in frame.groupby("unit_id", sort=True):
            unit_frame = unit_frame.sort_values("cycle")
            values = unit_frame[self.feature_columns].to_numpy(dtype=np.float64)
            if len(values) >= 2:
                inputs.append(values[:-1])
                targets.append(values[1:])
        if not inputs:
            raise ValueError("At least one engine with two cycles is required")

        x = np.concatenate(inputs, axis=0)
        y = np.concatenate(targets, axis=0)
        self.model.fit(x, y)
        predicted = np.asarray(self.model.predict(x), dtype=np.float64).reshape(y.shape)
        residuals = y - predicted
        self.residual_std_ = np.sqrt(np.mean(residuals**2, axis=0))
        return self

    def predict_next(self, feature_values: np.ndarray) -> np.ndarray:
        if self.residual_std_ is None:
            raise RuntimeError("Fit the transition model before prediction")
        values = np.asarray(feature_values, dtype=np.float64).reshape(1, -1)
        return np.asarray(self.model.predict(values), dtype=np.float64).reshape(1, -1)[0]

    @property
    def residual_std(self) -> np.ndarray:
        if self.residual_std_ is None:
            raise RuntimeError("Fit the transition model before reading uncertainty")
        return self.residual_std_


class DigitalTwin:
    """State estimator and what-if simulator for the prototype system."""

    def __init__(
        self,
        transition_model: DataDrivenTransitionModel,
        feature_columns: list[str],
        actions: dict[str, ActionSpec] | None = None,
        rul_reference: float = 125.0,
        max_sensor_z: float = 5.0,
    ) -> None:
        self.transition_model = transition_model
        self.feature_columns = feature_columns
        self.actions = actions or DEFAULT_ACTIONS
        self.rul_reference = rul_reference
        self.max_sensor_z = max_sensor_z

    def estimate_state(
        self,
        window: np.ndarray,
        unit_id: int,
        end_cycle: int,
        predicted_rul: float,
        anomaly_score: float | None = None,
    ) -> TwinState:
        """Estimate the current twin state from a normalized telemetry window."""

        values = np.asarray(window, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_columns):
            raise ValueError("window must have shape (time, number_of_features)")
        if len(values) < 2:
            raise ValueError("window must contain at least two time steps")

        feature_values = values[-1].copy()
        feature_slopes = (values[-1] - values[0]) / (len(values) - 1)
        predicted_rul = max(0.0, float(predicted_rul))
        health_index = float(
            np.clip(1.0 - predicted_rul / self.rul_reference, 0.0, 1.0)
        )
        return TwinState(
            unit_id=int(unit_id),
            cycle=int(end_cycle),
            feature_columns=list(self.feature_columns),
            feature_values=feature_values,
            feature_slopes=feature_slopes,
            predicted_rul=predicted_rul,
            health_index=health_index,
            anomaly_score=None if anomaly_score is None else float(anomaly_score),
        )

    def simulate(
        self,
        state: TwinState,
        action: str,
        horizon: int = 5,
    ) -> TransitionOutcome:
        """Simulate one candidate action for a fixed number of cycles."""

        if action not in self.actions:
            raise KeyError(f"Unknown action {action!r}; choose from {sorted(self.actions)}")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")

        spec = self.actions[action]
        next_values = state.feature_values.copy()
        uncertainty_squared = np.zeros_like(next_values)
        for _ in range(horizon):
            baseline_next = self.transition_model.predict_next(next_values)
            baseline_delta = baseline_next - next_values
            next_values = next_values + spec.transition_multiplier * baseline_delta
            uncertainty_squared += (
                spec.transition_multiplier * self.transition_model.residual_std
            ) ** 2

        predicted_rul = max(
            0.0,
            state.predicted_rul - horizon * spec.degradation_multiplier,
        )
        health_index = float(
            np.clip(1.0 - predicted_rul / self.rul_reference, 0.0, 1.0)
        )
        predicted_state = TwinState(
            unit_id=state.unit_id,
            cycle=state.cycle + horizon,
            feature_columns=list(state.feature_columns),
            feature_values=next_values,
            feature_slopes=(next_values - state.feature_values) / horizon,
            predicted_rul=predicted_rul,
            health_index=health_index,
            anomaly_score=state.anomaly_score,
            operating=spec.keeps_operating,
        )

        projected_drift = float(np.mean(np.abs(next_values - state.feature_values)))
        model_uncertainty = float(np.sqrt(np.mean(uncertainty_squared)))
        if not spec.keeps_operating:
            failure_risk = 0.0
        elif predicted_rul <= 0:
            failure_risk = 1.0
        else:
            failure_risk = float(
                np.clip(horizon / (predicted_rul + horizon), 0.0, 1.0)
            )

        violations: list[str] = []
        if not np.isfinite(next_values).all():
            violations.append("non_finite_predicted_sensor_state")
        if spec.keeps_operating and float(np.max(np.abs(next_values))) > self.max_sensor_z:
            violations.append("predicted_sensor_state_outside_5_sigma_envelope")
        if spec.keeps_operating and predicted_rul <= 0:
            violations.append("predicted_failure_within_simulation_horizon")

        return TransitionOutcome(
            action=action,
            horizon=horizon,
            predicted_state=predicted_state,
            predicted_failure_risk=failure_risk,
            model_uncertainty=model_uncertainty,
            projected_drift=projected_drift,
            mission_cost=spec.mission_cost,
            safe_under_constraints=not violations,
            violations=violations,
        )

    def what_if(
        self,
        state: TwinState,
        actions: Iterable[str] | None = None,
        horizon: int = 5,
    ) -> list[TransitionOutcome]:
        """Simulate all requested candidate actions."""

        selected = list(actions) if actions is not None else list(self.actions)
        return [self.simulate(state, action, horizon=horizon) for action in selected]

