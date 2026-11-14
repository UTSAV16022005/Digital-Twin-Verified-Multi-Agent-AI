"""Run one complete Monitoring -> Safety closed-loop demonstration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from src.agent_coordinator import HealthAgentCoordinator
from src.digital_twin import DataDrivenTransitionModel, DigitalTwin
from src.diagnosis import HealthDiagnosisAgent
from src.lstm_monitoring import LSTMMonitoringAgent
from src.monitoring import PCAMonitoringAgent
from src.prognostics import PrognosticsAgent
from src.recovery_planning import RecoveryPlanningAgent
from src.safety_verification import SafetyConfig, SafetyVerifier


def show_telemetry_preview(
    telemetry: pd.DataFrame,
    normalized: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
) -> None:
    """Print a compact before/after view immediately before the agent call."""

    # Split the complete feature set into narrow tables so every value stays
    # readable in a normal terminal window.
    feature_groups = [
        feature_columns[start : start + 6]
        for start in range(0, len(feature_columns), 6)
    ]

    print("\nDATA PREVIEW BEFORE AGENT COORDINATION")
    print("======================================")
    print(
        f"Telemetry rows available: {len(telemetry)} | "
        f"selected model features: {len(feature_columns)} | "
        f"required sequence window: {window_size} cycles"
    )
    print("RUL label passed to agents: no")
    print("All selected features: " + ", ".join(feature_columns))
    print(
        "The normalized table is a preview using the training-fit scaler; "
        "agents may apply their own model-specific transforms."
    )

    for label, frame in (
        ("RAW TELEMETRY (before scaling)", telemetry),
        ("NORMALIZED FEATURE PREVIEW", normalized),
    ):
        for group_index, group in enumerate(feature_groups, start=1):
            display = frame.loc[:, ["unit_id", "cycle", *group]].copy().round(3)
            print(
                f"\n{label} | feature group {group_index}/{len(feature_groups)}: "
                f"{', '.join(group)}"
            )
            print("First 5 rows")
            print(display.head(5).to_string(index=False))
            print("Last 5 rows")
            print(display.tail(5).to_string(index=False))
    print(
        f"\nOne model input window has shape "
        f"({window_size} time steps, {len(feature_columns)} features)."
    )


def main(monitor_backend: str = "pca") -> None:
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

    if monitor_backend == "pca":
        monitor = PCAMonitoringAgent.fit(train_frame, features, healthy_cycles=30)
    elif monitor_backend == "lstm":
        monitor_path = (
            ROOT
            / "models"
            / "fd001_lstm_monitoring_verified"
            / "fd001_lstm_monitoring.pt"
        )
        if not monitor_path.exists():
            raise FileNotFoundError(
                "Train the LSTM monitoring agent first with "
                "python -m scripts.train_lstm_monitoring"
            )
        monitor = LSTMMonitoringAgent.load(monitor_path)
    else:
        raise ValueError("monitor_backend must be 'pca' or 'lstm'")

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
    prognostics.calibrate_uncertainty(validation_frame)
    transition_model = DataDrivenTransitionModel(features, alpha=1.0).fit(train_scaled)
    twin = DigitalTwin(transition_model, features, rul_reference=125.0, max_sensor_z=5.0)
    planner = RecoveryPlanningAgent(risk_budget=0.25)
    verifier = SafetyVerifier(SafetyConfig(max_failure_risk=0.25))
    orchestrator = ClosedLoopOrchestrator(prognostics, twin, planner, verifier)
    coordinator = HealthAgentCoordinator(
        monitor, diagnosis, orchestrator, diagnosis_scaler=scaler
    )

    demo_unit = int(sorted(split.validation_units)[0])
    raw_demo = validation_frame[validation_frame["unit_id"] == demo_unit].sort_values(
        "cycle"
    )
    raw_demo_telemetry = raw_demo.drop(columns=["rul"])
    current_predictions = prognostics.predict_frame(raw_demo_telemetry)
    target_candidates = current_predictions[current_predictions["predicted_rul"] >= 30.0]
    selected_prediction = (
        target_candidates.iloc[-1]
        if not target_candidates.empty
        else current_predictions.iloc[-1]
    )
    selected_cycle = int(selected_prediction["end_cycle"])
    telemetry_prefix = raw_demo_telemetry[
        raw_demo_telemetry["cycle"] <= selected_cycle
    ].copy()
    normalized_preview = transform_features(
        telemetry_prefix, features, scaler
    )
    show_telemetry_preview(
        telemetry_prefix,
        normalized_preview,
        features,
        prognostics.config.window_size,
    )
    coordinated = coordinator.step(telemetry_prefix, horizon=5)
    result = coordinated.outcome
    actual_rul_for_evaluation = float(
        raw_demo[raw_demo["cycle"] == selected_cycle]["rul"].iloc[0]
    )
    report = coordinated.as_dict()
    report["monitoring_backend"] = monitor_backend
    report["actual_rul_visible_only_for_evaluation"] = actual_rul_for_evaluation
    report["ground_truth_rule"] = (
        "Actual RUL is recorded for evaluation only and is not passed into the Twin, planner, or verifier."
    )
    report["safety_config"] = vars(verifier.config)

    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / (
        "fd001_closed_loop_demo_lstm.json"
        if monitor_backend == "lstm"
        else "fd001_closed_loop_demo.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("FD001 closed-loop prototype")
    print("===========================")
    print(
        f"{monitor_backend.upper()} Monitoring -> Diagnosis -> Prognostics -> "
        "Twin -> Planning -> Safety"
    )
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
    print(
        "planning one-sided lower 95% RUL bound: "
        f"{result.planning_rul_lower_95:.2f}"
    )
    print(f"actual RUL used only for evaluation: {actual_rul_for_evaluation:.2f}")
    print(f"selected action: {result.selected_action}")
    print(f"decision status: {result.decision_status}")
    print(result.rationale)
    print("safety decisions:")
    for action, decision in result.safety_decisions.items():
        print(f"  {action}: approved={decision.approved}, violations={decision.violations}")
    print(f"saved report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-backend", choices=("pca", "lstm"), default="pca")
    main(parser.parse_args().monitor_backend)

