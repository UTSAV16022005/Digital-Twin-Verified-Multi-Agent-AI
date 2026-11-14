"""Closed-loop orchestration across the health-management components."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.cmapss_data import make_windows, transform_features
from src.digital_twin import DigitalTwin, TransitionOutcome, TwinState
from src.prognostics import PrognosticsAgent
from src.recovery_planning import RecoveryPlanningAgent
from src.safety_verification import SafetyDecision, SafetyVerifier


@dataclass
class ClosedLoopResult:
    """Traceable output from one closed-loop decision step."""

    unit_id: int
    cycle: int
    anomaly_score: float | None
    diagnosis_state: str | None
    diagnosis_confidence: float | None
    # Nominal point estimate is for reporting; the Twin plans using the
    # validation-calibrated lower confidence bound below.
    predicted_rul: float
    planning_rul_lower_95: float
    rul_uncertainty: float
    health_index: float
    twin_state: TwinState
    outcomes: list[TransitionOutcome]
    safety_decisions: dict[str, SafetyDecision]
    selected_action: str | None
    decision_status: str
    rationale: str

    def as_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "cycle": self.cycle,
            "anomaly_score": self.anomaly_score,
            "diagnosis_state": self.diagnosis_state,
            "diagnosis_confidence": self.diagnosis_confidence,
            "predicted_rul": self.predicted_rul,
            "planning_rul_lower_95": self.planning_rul_lower_95,
            "rul_uncertainty": self.rul_uncertainty,
            "health_index": self.health_index,
            "twin_state": self.twin_state.as_dict(),
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
            "safety_decisions": {
                action: decision.as_dict()
                for action, decision in self.safety_decisions.items()
            },
            "selected_action": self.selected_action,
            "decision_status": self.decision_status,
            "rationale": self.rationale,
        }


@dataclass
class ClosedLoopAbstention:
    """Explicit no-action result when telemetry or calibration is insufficient."""

    unit_id: int
    cycle: int
    decision_status: str
    available_cycles: int
    required_cycles: int
    rationale: str
    anomaly_score: float | None = None
    diagnosis_state: str | None = None
    diagnosis_confidence: float | None = None
    predicted_rul: float | None = None
    planning_rul_lower_95: float | None = None
    rul_uncertainty: float | None = None
    health_index: float | None = None
    twin_state: TwinState | None = None
    outcomes: list[TransitionOutcome] = field(default_factory=list)
    safety_decisions: dict[str, SafetyDecision] = field(default_factory=dict)
    selected_action: str | None = None

    def as_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "cycle": self.cycle,
            "anomaly_score": self.anomaly_score,
            "diagnosis_state": self.diagnosis_state,
            "diagnosis_confidence": self.diagnosis_confidence,
            "predicted_rul": self.predicted_rul,
            "planning_rul_lower_95": self.planning_rul_lower_95,
            "rul_uncertainty": self.rul_uncertainty,
            "health_index": self.health_index,
            "twin_state": self.twin_state,
            "outcomes": self.outcomes,
            "safety_decisions": self.safety_decisions,
            "selected_action": self.selected_action,
            "decision_status": self.decision_status,
            "available_cycles": self.available_cycles,
            "required_cycles": self.required_cycles,
            "rationale": self.rationale,
        }


class ClosedLoopOrchestrator:
    """Connect Prognostics, Digital Twin, Planning, and Safety components."""

    def __init__(
        self,
        prognostics: PrognosticsAgent,
        twin: DigitalTwin,
        planner: RecoveryPlanningAgent,
        safety_verifier: SafetyVerifier,
    ) -> None:
        self.prognostics = prognostics
        self.twin = twin
        self.planner = planner
        self.safety_verifier = safety_verifier

    def step(
        self,
        telemetry_frame: pd.DataFrame,
        anomaly_score: float | None = None,
        diagnosis_state: str | None = None,
        diagnosis_confidence: float | None = None,
        horizon: int = 5,
    ) -> ClosedLoopResult | ClosedLoopAbstention:
        """Run one decision step using telemetry from exactly one engine."""

        units = telemetry_frame["unit_id"].unique()
        if len(units) != 1:
            raise ValueError("ClosedLoopOrchestrator.step expects exactly one engine")

        unit_id = int(units[0])
        available_cycles = int(telemetry_frame["cycle"].nunique())
        latest_cycle = int(telemetry_frame["cycle"].max())
        required_cycles = int(self.prognostics.config.window_size)
        if available_cycles < required_cycles:
            return ClosedLoopAbstention(
                unit_id=unit_id,
                cycle=latest_cycle,
                decision_status="insufficient_telemetry",
                available_cycles=available_cycles,
                required_cycles=required_cycles,
                rationale=(
                    f"No action was evaluated: received {available_cycles} distinct cycles, "
                    f"but the prognostics model requires {required_cycles}."
                ),
            )

        calibration_engines = int(
            getattr(self.prognostics, "n_calibration_engines_", 0)
        )
        if calibration_engines < 1:
            return ClosedLoopAbstention(
                unit_id=unit_id,
                cycle=latest_cycle,
                decision_status="uncertainty_not_calibrated",
                available_cycles=available_cycles,
                required_cycles=required_cycles,
                rationale=(
                    "No action was evaluated: calibrate prognostics uncertainty on "
                    "held-out engines before the safety-constrained decision step."
                ),
            )

        predictions = self.prognostics.predict_frame(telemetry_frame)
        if predictions.empty:
            return ClosedLoopAbstention(
                unit_id=unit_id,
                cycle=latest_cycle,
                decision_status="no_valid_telemetry_window",
                available_cycles=available_cycles,
                required_cycles=required_cycles,
                rationale=(
                    "No action was evaluated: the telemetry did not produce a valid "
                    "prognostics window. Check cycle continuity and sensor values."
                ),
            )
        current_prediction = predictions.iloc[-1]
        nominal_rul = float(current_prediction["predicted_rul"])
        planning_rul_lower_95 = float(
            current_prediction["planning_rul_lower_95"]
        )

        scaled_frame = transform_features(
            telemetry_frame,
            self.prognostics.feature_columns,
            self.prognostics.scaler,
        )
        # The shared window builder also emits target metadata, but a live
        # decision may have telemetry only. The placeholder target is ignored;
        # actual RUL labels must never be required by the decision path.
        if "rul" not in scaled_frame.columns:
            scaled_frame = scaled_frame.assign(rul=float("nan"))
        windows, _, metadata = make_windows(
            scaled_frame,
            self.prognostics.feature_columns,
            self.prognostics.config.window_size,
        )
        matching = metadata.index[
            metadata["end_cycle"] == current_prediction["end_cycle"]
        ]
        if len(matching) != 1:
            raise RuntimeError("Could not align the prognostics window with telemetry")

        state = self.twin.estimate_state(
            windows[int(matching[0])],
            unit_id=int(current_prediction["unit_id"]),
            end_cycle=int(current_prediction["end_cycle"]),
            predicted_rul=planning_rul_lower_95,
            anomaly_score=anomaly_score,
        )
        outcomes = self.twin.what_if(state, horizon=horizon)
        safety_decisions = self.safety_verifier.verify_all(outcomes)
        approved = [
            outcome
            for outcome in outcomes
            if safety_decisions[outcome.action].approved
        ]
        approved.sort(key=self.planner.score)

        if approved:
            selected = approved[0]
            selected_action = selected.action
            decision_status = "recommendation"
            rationale = (
                f"{selected_action} passed independent safety verification and had "
                f"the lowest planner score ({self.planner.score(selected):.4f}) among "
                "approved actions. The Twin used the validation-calibrated one-sided "
                f"95% lower RUL bound ({planning_rul_lower_95:.1f} cycles), not the "
                f"nominal estimate ({nominal_rul:.1f})."
            )
        else:
            selected_action = None
            decision_status = "no_safe_action"
            rationale = (
                "No simulated action passed independent safety verification. The "
                f"Twin used the validation-calibrated one-sided 95% lower RUL bound "
                f"({planning_rul_lower_95:.1f} cycles), not the nominal estimate "
                f"({nominal_rul:.1f})."
            )

        return ClosedLoopResult(
            unit_id=state.unit_id,
            cycle=state.cycle,
            anomaly_score=anomaly_score,
            diagnosis_state=diagnosis_state,
            diagnosis_confidence=diagnosis_confidence,
            predicted_rul=nominal_rul,
            planning_rul_lower_95=planning_rul_lower_95,
            rul_uncertainty=float(current_prediction["rul_uncertainty"]),
            health_index=state.health_index,
            twin_state=state,
            outcomes=outcomes,
            safety_decisions=safety_decisions,
            selected_action=selected_action,
            decision_status=decision_status,
            rationale=rationale,
        )

