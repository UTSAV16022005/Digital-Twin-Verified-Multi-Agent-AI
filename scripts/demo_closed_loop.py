"""Run one complete Monitoring -> Safety closed-loop demonstration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.cmapss_data import (
    FEATURE_COLUMNS,
    add_train_rul,
    constant_columns,
    fit_feature_scaler,
    load_telemetry,
    make_windows,
    split_engine_units,
    summarize_windows,
    transform_features,
)
from src.closed_loop import ClosedLoopOrchestrator
from src.digital_twin import DataDrivenTransitionModel, DigitalTwin
from src.diagnosis import HealthDiagnosisAgent
from src.monitoring import PCAMonitoringAgent
from src.prognostics import PrognosticsAgent
from src.recovery_planning import RecoveryPlanningAgent
from src.safety_verification import SafetyConfig, SafetyVerifier


def main() -> None:
    data_dir = Path("data/raw/cmapss")
    full_train = add_train_rul(load_telemetry(data_dir, "FD001", "train"))
    split = split_engine_units(full_train, validation_fraction=0.2, seed=42)
    train_frame = full_train[full_train["unit_id"].isin(split.train_units)].copy()
    validation_frame = full_train[
        full_train["unit_id"].isin(split.validation_units)
    ].copy()

    dropped = constant_columns(train_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]
    scaler = fit_feature_scaler(train_frame, features)
    train_scaled = transform_features(train_frame, features, scaler)
    validation_scaled = transform_features(validation_frame, features, scaler)

    # Monitoring Agent baseline.
    monitor = PCAMonitoringAgent.fit(train_frame, features, healthy_cycles=30)

    # Diagnosis Agent baseline trained on the same leakage-safe train engines.
    train_windows, train_targets, _ = make_windows(train_scaled, features, 30)
    diagnosis = HealthDiagnosisAgent.fit(
        summarize_windows(train_windows),
        train_targets,
        feature_names=[
            f"{stat}_{feature}"
            for stat in ("last", "mean", "std", "min", "max", "slope")
            for feature in features
        ],
    )

    # Formal Prognostics Agent and data-driven Twin.
    prognostics = PrognosticsAgent.fit(train_frame, features)
    transition_model = DataDrivenTransitionModel(features, alpha=1.0).fit(train_scaled)
    twin = DigitalTwin(transition_model, features, rul_reference=125.0, max_sensor_z=5.0)
    planner = RecoveryPlanningAgent(risk_budget=0.25)
    verifier = SafetyVerifier(SafetyConfig(max_failure_risk=0.25))
    orchestrator = ClosedLoopOrchestrator(prognostics, twin, planner, verifier)

    demo_unit = int(sorted(split.validation_units)[0])
    raw_demo = validation_frame[validation_frame["unit_id"] == demo_unit].sort_values(
        "cycle"
    )
    monitor_scores = monitor.score(raw_demo)

    current_predictions = prognostics.predict_frame(raw_demo)
    target_candidates = current_predictions[current_predictions["predicted_rul"] >= 30.0]
    selected_prediction = (
        target_candidates.iloc[-1]
        if not target_candidates.empty
        else current_predictions.iloc[-1]
    )
    selected_cycle = int(selected_prediction["end_cycle"])
    current_monitor_score = float(
        monitor_scores[monitor_scores["cycle"] == selected_cycle]["anomaly_score"].iloc[0]
    )
    raw_at_cycle = raw_demo[raw_demo["cycle"] <= selected_cycle]
    scaled_at_cycle = validation_scaled[
        (validation_scaled["unit_id"] == demo_unit)
        & (validation_scaled["cycle"] <= selected_cycle)
    ]
    diagnosis_windows, _, diagnosis_metadata = make_windows(scaled_at_cycle, features, 30)
    current_diagnosis_window = diagnosis_windows[-1:]
    diagnosis_probabilities = diagnosis.predict_proba(
        summarize_windows(current_diagnosis_window)
    ).iloc[0]
    diagnosis_state = str(diagnosis_probabilities.idxmax())
    diagnosis_confidence = float(diagnosis_probabilities.max())

    result = orchestrator.step(
        raw_at_cycle,
        anomaly_score=current_monitor_score,
        diagnosis_state=diagnosis_state,
        diagnosis_confidence=diagnosis_confidence,
        horizon=5,
    )
    actual_rul_for_evaluation = float(
        raw_demo[raw_demo["cycle"] == selected_cycle]["rul"].iloc[0]
    )
    report = result.as_dict()
    report["actual_rul_visible_only_for_evaluation"] = actual_rul_for_evaluation
    report["ground_truth_rule"] = (
        "Actual RUL is recorded for evaluation only and is not passed into the Twin, planner, or verifier."
    )
    report["safety_config"] = vars(verifier.config)

    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fd001_closed_loop_demo.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("FD001 closed-loop prototype")
    print("===========================")
    print("Monitoring -> Diagnosis -> Prognostics -> Twin -> Planning -> Safety")
    print(f"unit={result.unit_id}, cycle={result.cycle}")
    print(f"anomaly score: {result.anomaly_score:.6f}")
    print(
        f"diagnosis: {result.diagnosis_state} "
        f"(confidence={result.diagnosis_confidence:.3f})"
    )
    print(
        f"predicted RUL: {result.predicted_rul:.2f} "
        f"(uncertainty={result.rul_uncertainty:.2f})"
    )
    print(f"actual RUL used only for evaluation: {actual_rul_for_evaluation:.2f}")
    print(f"selected action: {result.selected_action}")
    print(f"decision status: {result.decision_status}")
    print(result.rationale)
    print("safety decisions:")
    for action, decision in result.safety_decisions.items():
        print(f"  {action}: approved={decision.approved}, violations={decision.violations}")
    print(f"saved report: {output_dir / 'fd001_closed_loop_demo.json'}")


if __name__ == "__main__":
    main()

