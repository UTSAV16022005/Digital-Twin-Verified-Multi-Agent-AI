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
        "tree_dispersion_interval_coverage": float(
            ((actual >= predictions["lower_rul"]) & (actual <= predictions["upper_rul"])).mean()
        ),
        "tree_dispersion_mean_interval_width": float(
            (predictions["upper_rul"] - predictions["lower_rul"]).mean()
        ),
        "conformal_interval_coverage_95": float(
            (
                (actual >= predictions["calibrated_lower_95"])
                & (actual <= predictions["calibrated_upper_95"])
            ).mean()
        ),
        "conformal_mean_interval_width": float(
            (predictions["calibrated_upper_95"] - predictions["calibrated_lower_95"]).mean()
        ),
        "one_sided_lower_bound_coverage": float(
            (actual >= predictions["planning_rul_lower_95"]).mean()
        ),
    }


def terminal_error_analysis(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Build an auditable one-row-per-engine error report for held-out test engines."""
    columns = [
        "unit_id",
        "end_cycle",
        "rul",
        "predicted_rul",
        "rul_uncertainty",
        "lower_rul",
        "upper_rul",
        "calibrated_lower_95",
        "calibrated_upper_95",
        "planning_rul_lower_95",
    ]
    per_engine = predictions[columns].copy()
    per_engine["signed_error"] = per_engine["predicted_rul"] - per_engine["rul"]
    per_engine["absolute_error"] = per_engine["signed_error"].abs()
    per_engine["tree_dispersion_covers_rul"] = (
        (per_engine["rul"] >= per_engine["lower_rul"])
        & (per_engine["rul"] <= per_engine["upper_rul"])
    )
    per_engine["conformal_interval_covers_rul"] = (
        (per_engine["rul"] >= per_engine["calibrated_lower_95"])
        & (per_engine["rul"] <= per_engine["calibrated_upper_95"])
    )
    per_engine["planning_lower_bound_covers_rul"] = (
        per_engine["rul"] >= per_engine["planning_rul_lower_95"]
    )
    per_engine = per_engine.sort_values("unit_id").reset_index(drop=True)

    worst = per_engine.loc[per_engine["absolute_error"].idxmax()]
    summary: dict[str, float | int] = {
        "engine_count": int(len(per_engine)),
        "median_absolute_error": float(per_engine["absolute_error"].median()),
        "p90_absolute_error": float(per_engine["absolute_error"].quantile(0.90)),
        "maximum_absolute_error": float(worst["absolute_error"]),
        "worst_error_unit_id": int(worst["unit_id"]),
        "overprediction_count": int((per_engine["signed_error"] > 0).sum()),
        "underprediction_count": int((per_engine["signed_error"] < 0).sum()),
        "tree_dispersion_interval_coverage": float(
            per_engine["tree_dispersion_covers_rul"].mean()
        ),
        "conformal_interval_coverage_95": float(
            per_engine["conformal_interval_covers_rul"].mean()
        ),
        "one_sided_lower_bound_coverage": float(
            per_engine["planning_lower_bound_covers_rul"].mean()
        ),
    }
    return per_engine, summary


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
    q90, q95 = prognostics.calibrate_uncertainty(validation_frame)
    validation_predictions = prognostics.predict_frame(validation_frame)
    test_predictions = prognostics.predict_frame(full_test)
    validation_terminal = prognostics.predict_current(validation_frame)
    test_terminal = prognostics.predict_current(full_test)
    test_terminal_by_engine, terminal_error_summary = terminal_error_analysis(
        test_terminal
    )
    all_test_engine_ids = set(full_test["unit_id"].unique())
    scored_test_engine_ids = set(test_terminal["unit_id"].unique())
    unscored_test_engine_ids = [
        int(unit_id)
        for unit_id in sorted(all_test_engine_ids - scored_test_engine_ids)
    ]
    test_engine_cycle_counts = full_test.groupby("unit_id", sort=True).size()
    unscored_test_engines = pd.DataFrame(
        {
            "unit_id": unscored_test_engine_ids,
            "observed_cycles": [
                int(test_engine_cycle_counts[unit_id])
                for unit_id in unscored_test_engine_ids
            ],
        }
    )
    unscored_test_engines["minimum_required_cycles"] = config.window_size

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
            "uncertainty_calibration": {
                "method": "finite-sample conformal calibration on one RUL-stratified window per held-out development engine",
                "calibration_engines": prognostics.n_calibration_engines_,
                "q90_cycles": q90,
                "q95_cycles": q95,
                "one_sided_lower_bound_margin_95_cycles": (
                    prognostics.lower_bound_margin_95
                ),
            },
            "validation_window_metrics": regression_metrics(validation_predictions),
            "validation_terminal_metrics": regression_metrics(validation_terminal),
            "test_window_metrics": regression_metrics(test_predictions),
            "test_terminal_metrics": regression_metrics(test_terminal),
            "test_terminal_error_analysis": terminal_error_summary,
            "test_terminal_engine_coverage": {
                "total_engines": len(all_test_engine_ids),
                "scored_engines": len(scored_test_engine_ids),
                "unscored_engines": len(unscored_test_engine_ids),
                "minimum_required_cycles": config.window_size,
                "unscored_engine_ids": unscored_test_engine_ids,
            },
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
    test_terminal_by_engine.to_csv(
        output_dir / "fd004_terminal_engine_errors.csv", index=False
    )
    unscored_test_engines.to_csv(
        output_dir / "fd004_unscored_test_engines.csv", index=False
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
    print(
        "per-engine terminal errors: "
        f"median={terminal_error_summary['median_absolute_error']:.3f}, "
        f"p90={terminal_error_summary['p90_absolute_error']:.3f}, "
        f"worst engine={terminal_error_summary['worst_error_unit_id']} "
        f"(abs error={terminal_error_summary['maximum_absolute_error']:.3f})"
    )
    print(
        "conformal interval: "
        f"q95={q95:.2f} cycles; held-out terminal coverage="
        f"{report['prognostics']['test_terminal_metrics']['conformal_interval_coverage_95']:.1%}"
    )
    print(
        "one-sided planning bound: "
        f"margin={prognostics.lower_bound_margin_95:.2f} cycles; held-out terminal "
        f"coverage={report['prognostics']['test_terminal_metrics']['one_sided_lower_bound_coverage']:.1%}"
    )
    print(
        "terminal engine coverage: "
        f"{len(scored_test_engine_ids)}/{len(all_test_engine_ids)} scored; "
        f"{len(unscored_test_engine_ids)} lacked a {config.window_size}-cycle window"
    )
    print(f"Twin validation one-step MAE: {transition_validation_mae:.6f}")
    print(
        f"demo unit={demo_unit}, cycle={demo_cycle}, "
        f"predicted RUL={loop_result.predicted_rul:.2f}, "
        f"planning lower bound={loop_result.planning_rul_lower_95:.2f}, "
        f"actual RUL evaluation-only={actual_demo_rul:.2f}"
    )
    print(f"selected action: {loop_result.selected_action}")
    print(f"saved report: {output_dir / 'fd004_pipeline_evaluation.json'}")
    print(f"saved per-engine errors: {output_dir / 'fd004_terminal_engine_errors.csv'}")
    print(f"saved unscored engines: {output_dir / 'fd004_unscored_test_engines.csv'}")


if __name__ == "__main__":
    main()

