"""Evaluate the agent pipeline on the harder FD004 C-MAPSS subset."""

from __future__ import annotations

import json
import importlib.metadata
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.cmapss_data import (
    FEATURE_COLUMNS,
    add_test_rul,
    add_train_rul,
    constant_columns,
    fit_feature_scaler,
    load_test_rul,
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
from src.prognostics import PrognosticsAgent, PrognosticsConfig
from src.recovery_planning import RecoveryPlanningAgent
from src.safety_verification import SafetyConfig, SafetyVerifier


def regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    actual = predictions["rul"].to_numpy(dtype=float)
    predicted = predictions["predicted_rul"].to_numpy(dtype=float)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "interval_coverage_95": float(
            ((actual >= predictions["lower_rul"]) & (actual <= predictions["upper_rul"])).mean()
        ),
    }


def transition_mae(model: DataDrivenTransitionModel, frame: pd.DataFrame) -> float:
    actual: list[float] = []
    predicted: list[float] = []
    for _, unit_frame in frame.groupby("unit_id", sort=True):
        values = unit_frame.sort_values("cycle")[model.feature_columns].to_numpy(dtype=float)
        for index in range(len(values) - 1):
            actual.extend(values[index + 1])
            predicted.extend(model.predict_next(values[index]))
    return float(mean_absolute_error(actual, predicted))


def main() -> None:
    data_dir = Path("data/raw/cmapss")
    full_train = add_train_rul(load_telemetry(data_dir, "FD004", "train"))
    full_test = add_test_rul(
        load_telemetry(data_dir, "FD004", "test"), load_test_rul(data_dir, "FD004")
    )
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

    config = PrognosticsConfig(n_estimators=80)
    prognostics = PrognosticsAgent.fit(train_frame, features, config)
    validation_predictions = prognostics.predict_frame(validation_frame)
    test_predictions = prognostics.predict_frame(full_test)
    validation_terminal = prognostics.predict_current(validation_frame)
    test_terminal = prognostics.predict_current(full_test)

    transition_model = DataDrivenTransitionModel(features, alpha=1.0).fit(train_scaled)
    transition_validation_mae = transition_mae(transition_model, validation_scaled)
    twin = DigitalTwin(transition_model, features, rul_reference=125.0, max_sensor_z=5.0)
    planner = RecoveryPlanningAgent(risk_budget=0.25)
    verifier = SafetyVerifier(SafetyConfig(max_failure_risk=0.25))
    orchestrator = ClosedLoopOrchestrator(prognostics, twin, planner, verifier)

    monitor = PCAMonitoringAgent.fit(train_frame, features, healthy_cycles=30)
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

    # Choose a test window by predicted RUL only. Ground truth is retained only
    # afterward for evaluation reporting.
    candidates = test_predictions[
        (test_predictions["predicted_rul"] >= 30.0)
        & (test_predictions["predicted_rul"] <= 100.0)
    ]
    selected = candidates.iloc[0] if not candidates.empty else test_predictions.iloc[0]
    demo_unit = int(selected["unit_id"])
    demo_cycle = int(selected["end_cycle"])
    raw_demo = full_test[
        (full_test["unit_id"] == demo_unit) & (full_test["cycle"] <= demo_cycle)
    ].sort_values("cycle")
    monitor_score = float(
        monitor.score(raw_demo).iloc[-1]["anomaly_score"]
    )
    diagnosis_scaled = validation_scaled.iloc[0:0].copy()
    test_scaled = transform_features(full_test, features, scaler)
    diagnosis_scaled = test_scaled[
        (test_scaled["unit_id"] == demo_unit) & (test_scaled["cycle"] <= demo_cycle)
    ]
    diagnosis_windows, _, _ = make_windows(diagnosis_scaled, features, 30)
    probabilities = diagnosis.predict_proba(summarize_windows(diagnosis_windows)[-1:]).iloc[0]
    diagnosis_state = str(probabilities.idxmax())
    diagnosis_confidence = float(probabilities.max())
    loop_result = orchestrator.step(
        raw_demo,
        anomaly_score=monitor_score,
        diagnosis_state=diagnosis_state,
        diagnosis_confidence=diagnosis_confidence,
        horizon=5,
    )
    actual_demo_rul = float(
        full_test[
            (full_test["unit_id"] == demo_unit) & (full_test["cycle"] == demo_cycle)
        ]["rul"].iloc[0]
    )

    report = {
        "subset": "FD004",
        "train_engines": len(split.train_units),
        "validation_engines": len(split.validation_units),
        "test_engines": int(full_test["unit_id"].nunique()),
        "features_used": features,
        "features_dropped_as_constant": dropped,
        "prognostics": {
            "config": config.as_dict(),
            "validation_window_metrics": regression_metrics(validation_predictions),
            "validation_terminal_metrics": regression_metrics(validation_terminal),
            "test_window_metrics": regression_metrics(test_predictions),
            "test_terminal_metrics": regression_metrics(test_terminal),
        },
        "digital_twin": {
            "validation_one_step_mae_standardized_units": transition_validation_mae,
        },
        "closed_loop_demo": loop_result.as_dict(),
        "actual_demo_rul_visible_only_for_evaluation": actual_demo_rul,
        "limitation": "FD004 fault-type labels and recovery-action outcomes are not supplied by C-MAPSS.",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "seed": 42,
        },
    }
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fd004_pipeline_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    test_predictions.to_csv(
        output_dir / "fd004_prognostics_test_predictions.csv", index=False
    )

    print("FD004 agent pipeline evaluation")
    print("================================")
    print(f"features used: {len(features)}, dropped constants: {len(dropped)}")
    print(
        "test terminal prognostics: "
        f"MAE={report['prognostics']['test_terminal_metrics']['mae']:.3f}, "
        f"RMSE={report['prognostics']['test_terminal_metrics']['rmse']:.3f}, "
        f"R2={report['prognostics']['test_terminal_metrics']['r2']:.3f}"
    )
    print(f"Twin validation one-step MAE: {transition_validation_mae:.6f}")
    print(
        f"demo unit={demo_unit}, cycle={demo_cycle}, "
        f"predicted RUL={loop_result.predicted_rul:.2f}, "
        f"actual RUL evaluation-only={actual_demo_rul:.2f}"
    )
    print(f"selected action: {loop_result.selected_action}")
    print(f"saved report: {output_dir / 'fd004_pipeline_evaluation.json'}")


if __name__ == "__main__":
    main()

