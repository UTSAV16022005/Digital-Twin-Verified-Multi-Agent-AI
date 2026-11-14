"""Coordinate role-specific health agents and preserve an auditable trace.

This is a deterministic agent coordinator: it invokes each role in a fixed,
traceable sequence, passes telemetry-only context between components, simulates
registered actions, and ranks only candidates accepted by the independent
safety verifier. It does not use an LLM or claim open-ended autonomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.cmapss_data import make_windows, summarize_windows, transform_features
from src.closed_loop import (
    ClosedLoopAbstention,
    ClosedLoopOrchestrator,
    ClosedLoopResult,
)
from src.diagnosis import HealthDiagnosisAgent


@dataclass
class CoordinatedStep:
    """Outcome and role-by-role trace for one telemetry decision."""

    outcome: ClosedLoopResult | ClosedLoopAbstention
    agent_trace: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {**self.outcome.as_dict(), "agent_trace": self.agent_trace}


class HealthAgentCoordinator:
    """Run monitoring, diagnosis, prognostics, Twin, planning, and safety."""

    def __init__(
        self,
        monitor: Any,
        diagnosis: HealthDiagnosisAgent,
        orchestrator: ClosedLoopOrchestrator,
        diagnosis_scaler: Any,
    ) -> None:
        self.monitor = monitor
        self.diagnosis = diagnosis
        self.orchestrator = orchestrator
        self.diagnosis_scaler = diagnosis_scaler
        self.feature_columns = list(orchestrator.prognostics.feature_columns)

    def _latest_monitor_output(
        self, telemetry: pd.DataFrame
    ) -> tuple[float | None, float | None, bool]:
        """Adapt the PCA and LSTM-AE monitor APIs to one coordinator contract."""

        if callable(getattr(self.monitor, "score", None)):
            scores = self.monitor.score(telemetry)
        elif callable(getattr(self.monitor, "monitor", None)):
            scores = self.monitor.monitor(telemetry)
        else:
            raise TypeError("monitor must provide score() or monitor()")
        threshold = getattr(self.monitor, "threshold", None)
        if scores.empty:
            return None, None if threshold is None else float(threshold), False
        row = scores.iloc[-1]
        if "anomaly_score" not in row.index:
            raise ValueError("monitor output must contain anomaly_score")
        score_value = row["anomaly_score"]
        anomaly_score = None if pd.isna(score_value) else float(score_value)
        threshold_value = row.get("threshold", threshold)
        anomaly_threshold = (
            None if threshold_value is None else float(threshold_value)
        )
        if "alert" in row.index:
            alert = bool(row["alert"])
        else:
            alert = bool(row.get("confirmed_alert", False))
        return anomaly_score, anomaly_threshold, alert

    def step(
        self,
        telemetry_frame: pd.DataFrame,
        horizon: int = 5,
    ) -> CoordinatedStep:
        """Run every agent on one engine's telemetry prefix, without RUL labels."""

        if "rul" in telemetry_frame.columns:
            raise ValueError(
                "HealthAgentCoordinator accepts telemetry only; remove the RUL label"
            )
        if not {"unit_id", "cycle"}.issubset(telemetry_frame.columns):
            raise KeyError("telemetry_frame must include unit_id and cycle")
        units = telemetry_frame["unit_id"].unique()
        if len(units) != 1:
            raise ValueError("HealthAgentCoordinator.step expects exactly one engine")
        telemetry = telemetry_frame.sort_values("cycle").reset_index(drop=True)
        unit_id = int(units[0])
        current_cycle = int(telemetry["cycle"].iloc[-1])

        anomaly_score, anomaly_threshold, anomaly_alert = self._latest_monitor_output(
            telemetry
        )
        trace: list[dict[str, Any]] = [
            {
                "agent": "MonitoringAgent",
                "observed_cycle": current_cycle,
                "output": {
                    "anomaly_score": anomaly_score,
                    "threshold": anomaly_threshold,
                    "alert": anomaly_alert,
                },
                "role": "Compare recent sensor behavior with the learned healthy baseline.",
            }
        ]

        diagnosis_state: str | None = None
        diagnosis_confidence: float | None = None
        diagnosis_probabilities: dict[str, float] | None = None
        diagnosis_entropy: float | None = None
        diagnosis_cycle = current_cycle
        window_size = int(self.orchestrator.prognostics.config.window_size)
        if telemetry["cycle"].nunique() >= window_size:
            scaled = transform_features(
                telemetry, self.feature_columns, self.diagnosis_scaler
            )
            if "rul" not in scaled.columns:
                scaled = scaled.assign(rul=np.nan)
            windows, _, metadata = make_windows(
                scaled,
                self.feature_columns,
                window_size=window_size,
                target_column="rul",
            )
            if len(windows):
                diagnosis_features = summarize_windows(windows[-1:])
                states, confidences, entropies = self.diagnosis.predict_with_uncertainty(
                    diagnosis_features
                )
                probabilities = self.diagnosis.predict_proba(
                    diagnosis_features
                ).iloc[0]
                diagnosis_state = str(states[0])
                diagnosis_confidence = float(confidences[0])
                diagnosis_entropy = float(entropies[0])
                diagnosis_probabilities = {
                    str(label): float(value)
                    for label, value in probabilities.items()
                }
                diagnosis_cycle = int(metadata.iloc[-1]["end_cycle"])
        else:
            diagnosis_cycle = current_cycle

        trace.append(
            {
                "agent": "DiagnosisAgent",
                "observed_cycle": diagnosis_cycle,
                "output": (
                    {
                        "health_state": diagnosis_state,
                        "confidence": diagnosis_confidence,
                        "normalized_entropy": diagnosis_entropy,
                        "class_probabilities": diagnosis_probabilities,
                        "label_scope": "RUL-derived project health bands; not physical fault types.",
                    }
                    if diagnosis_state is not None
                    else {
                        "health_state": None,
                        "abstained": True,
                        "reason": f"A {window_size}-cycle window is required.",
                    }
                ),
                "role": "Classify the health state and expose uncertainty.",
            }
        )

        outcome = self.orchestrator.step(
            telemetry,
            anomaly_score=anomaly_score,
            diagnosis_state=diagnosis_state,
            diagnosis_confidence=diagnosis_confidence,
            horizon=horizon,
        )
        if isinstance(outcome, ClosedLoopAbstention):
            trace.extend(
                [
                    {
                        "agent": "PrognosticsAgent",
                        "output": {"abstained": True, "reason": outcome.rationale},
                    },
                    {
                        "agent": "Coordinator",
                        "decision": outcome.decision_status,
                        "reason": outcome.rationale,
                    },
                ]
            )
            return CoordinatedStep(outcome, trace)

        trace.extend(
            [
                {
                    "agent": "PrognosticsAgent",
                    "observed_cycle": outcome.cycle,
                    "output": {
                        "predicted_rul": outcome.predicted_rul,
                        "planning_lower_rul_95": outcome.planning_rul_lower_95,
                        "uncertainty": outcome.rul_uncertainty,
                    },
                    "role": "Estimate RUL and provide the conservative planning bound.",
                },
                {
                    "agent": "DigitalTwin",
                    "output": {
                        "estimated_state": outcome.twin_state.as_dict(),
                        "simulated_outcomes": [item.as_dict() for item in outcome.outcomes],
                    },
                    "role": "Simulate registered what-if actions under documented assumptions.",
                },
                {
                    "agent": "RecoveryPlanningAgent",
                    "output": {
                        "selected_action": outcome.selected_action,
                        "decision_status": outcome.decision_status,
                        "ranked_approved_actions": [
                            {
                                "action": item.action,
                                "objective_score": self.orchestrator.planner.score(item),
                            }
                            for item in sorted(
                                [
                                    item
                                    for item in outcome.outcomes
                                    if outcome.safety_decisions[item.action].approved
                                ],
                                key=self.orchestrator.planner.score,
                            )
                        ],
                        "rationale": outcome.rationale,
                    },
                    "role": "Rank only actions that pass the safety verifier.",
                },
                {
                    "agent": "SafetyVerifier",
                    "output": {
                        action: decision.as_dict()
                        for action, decision in outcome.safety_decisions.items()
                    },
                    "role": "Apply deterministic hard constraints independently of the planner.",
                },
                {
                    "agent": "Coordinator",
                    "unit_id": unit_id,
                    "cycle": current_cycle,
                    "decision": outcome.decision_status,
                    "selected_action": outcome.selected_action,
                },
            ]
        )
        return CoordinatedStep(outcome, trace)
