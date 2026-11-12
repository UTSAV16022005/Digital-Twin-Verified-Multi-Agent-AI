"""Train and evaluate the first health-state Diagnosis Agent on FD001."""

from __future__ import annotations

import json
import importlib.metadata
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

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
from src.diagnosis import (
    HEALTH_STATES,
    HealthDiagnosisAgent,
    health_state_from_rul,
    compute_calibration_metrics,
    SyntheticFaultBenchmark,
)


def evaluate(
    agent: HealthDiagnosisAgent,
    features: np.ndarray,
    targets: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    actual = health_state_from_rul(
        targets, warning_rul=agent.warning_rul, critical_rul=agent.critical_rul
    )
    predicted = agent.predict(features)
    probabilities = agent.predict_proba(features)
    confidence = probabilities.max(axis=1).to_numpy()
    filtered_preds, confs, entropies = agent.predict_with_uncertainty(features)
    calibration = compute_calibration_metrics(actual, probabilities)

    class_dist = {str(c): int((actual == c).sum()) for c in HEALTH_STATES}
    uncertain_count = int((filtered_preds == "uncertain").sum())

    per_class = classification_report(
        actual, predicted, labels=HEALTH_STATES, output_dict=True, zero_division=0
    )
    report = {
        "class_distribution": class_dist,
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=HEALTH_STATES, average="macro")),
        "weighted_f1": float(f1_score(actual, predicted, labels=HEALTH_STATES, average="weighted", zero_division=0)),
        "per_class_metrics": {state: per_class[state] for state in HEALTH_STATES},
        "classification_report": per_class,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=HEALTH_STATES).tolist(),
        "calibration": calibration,
        "uncertainty_thresholds": {
            "confidence_threshold": agent.confidence_threshold,
            "entropy_threshold": agent.entropy_threshold,
            "flagged_uncertain_windows": uncertain_count,
            "uncertain_window_fraction": float(uncertain_count / len(features)),
        },
    }
    events = metadata[["unit_id", "end_cycle", "rul"]].copy()
    events["true_state"] = actual
    events["predicted_state"] = predicted
    events["confidence"] = confidence
    events["normalized_entropy"] = entropies
    events["prediction_with_uncertainty"] = filtered_preds
    return report, events


def terminal_events(events: pd.DataFrame) -> pd.DataFrame:
    return events.sort_values(["unit_id", "end_cycle"]).groupby("unit_id", sort=True).tail(1)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    data_dir = Path("data/raw/cmapss")
    full_train = add_train_rul(load_telemetry(data_dir, "FD001", "train"))
    split = split_engine_units(full_train, validation_fraction=0.2, seed=42)
    train_frame = full_train[full_train["unit_id"].isin(split.train_units)].copy()
    validation_frame = full_train[full_train["unit_id"].isin(split.validation_units)].copy()

    dropped = constant_columns(train_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]
    scaler = fit_feature_scaler(train_frame, features)
    train_frame = transform_features(train_frame, features, scaler)
    validation_frame = transform_features(validation_frame, features, scaler)

    train_windows, train_targets, _ = make_windows(train_frame, features, 30)
    validation_windows, validation_targets, validation_metadata = make_windows(
        validation_frame, features, 30
    )
    train_summary = summarize_windows(train_windows)
    validation_summary = summarize_windows(validation_windows)

    agent = HealthDiagnosisAgent.fit(
        train_summary,
        train_targets,
        feature_names=[
            f"{stat}_{feature}"
            for stat in ("last", "mean", "std", "min", "max", "slope")
            for feature in features
        ],
        warning_rul=125,
        critical_rul=25,
    )
    validation_report, validation_events = evaluate(
        agent, validation_summary, validation_targets, validation_metadata
    )
    terminal = terminal_events(validation_events)
    terminal_true = terminal["true_state"].to_numpy()
    terminal_predicted = terminal["predicted_state"].to_numpy()
    terminal_accuracy = float(accuracy_score(terminal_true, terminal_predicted))
    if len(np.unique(terminal_true)) == 1:
        terminal_balanced_accuracy = terminal_accuracy
    else:
        terminal_balanced_accuracy = float(
            balanced_accuracy_score(terminal_true, terminal_predicted)
        )
    terminal_report = {
        "accuracy": terminal_accuracy,
        "balanced_accuracy": terminal_balanced_accuracy,
        "macro_f1": float(
            f1_score(
                terminal_true,
                terminal_predicted,
                labels=HEALTH_STATES,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                terminal_true,
                terminal_predicted,
                labels=HEALTH_STATES,
                average="weighted",
                zero_division=0,
            )
        ),
        "class_distribution": {
            state: int((terminal_true == state).sum()) for state in HEALTH_STATES
        },
        "per_class_metrics": {
            state: classification_report(
                terminal_true,
                terminal_predicted,
                labels=HEALTH_STATES,
                output_dict=True,
                zero_division=0,
            )[state]
            for state in HEALTH_STATES
        },
        "calibration": compute_calibration_metrics(
            terminal_true,
            agent.predict_proba(
                validation_summary[
                    terminal.index.to_numpy()
                ]
            ),
        ),
        "confusion_matrix": confusion_matrix(
            terminal_true, terminal_predicted, labels=HEALTH_STATES
        ).tolist(),
    }

    report = {
        "subset": "FD001",
        "agent": "health-state diagnosis baseline",
        "label_definition": {
            "healthy": "RUL > 125",
            "degraded": "25 < RUL <= 125",
            "critical": "RUL <= 25",
        },
        "train_engines": len(split.train_units),
        "validation_engines": len(split.validation_units),
        "window_size": 30,
        "features_used": features,
        "features_dropped_as_constant": dropped,
        "validation_window_metrics": validation_report,
        "validation_terminal_engine_metrics": terminal_report,
        "training_class_distribution": {
            state: int((health_state_from_rul(train_targets) == state).sum())
            for state in HEALTH_STATES
        },
        "class_balancing": "inverse-frequency sample weights per training window; no cross-engine oversampling",
        "validation_per_engine": terminal[
            ["unit_id", "true_state", "predicted_state", "confidence", "normalized_entropy"]
        ].assign(
            correct=lambda frame: frame["true_state"] == frame["predicted_state"]
        ).to_dict(orient="records"),
        "limitation": "FD001 has one degradation mode; this model classifies health state, not fault type.",
        "environment": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit_learn": package_version("scikit-learn"),
            "seed": 42,
        },
    }

    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fd001_diagnosis_baseline.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    validation_events.to_csv(output_dir / "fd001_diagnosis_validation_events.csv", index=False)
    terminal[
        ["unit_id", "end_cycle", "true_state", "predicted_state", "confidence", "normalized_entropy"]
    ].to_csv(output_dir / "fd001_diagnosis_validation_engine_results.csv", index=False)
    joblib.dump(agent, models_dir / "fd001_diagnosis_baseline.joblib")

    print("FD001 Diagnosis Agent baseline")
    print("==============================")
    print("task: health-state classification, not component-fault diagnosis")
    print("labels: healthy / degraded / critical")
    print(
        "validation windows: "
        f"accuracy={validation_report['accuracy']:.3f}, "
        f"balanced_accuracy={validation_report['balanced_accuracy']:.3f}, "
        f"macro_F1={validation_report['macro_f1']:.3f}"
    )
    print(
        "validation terminal engines: "
        f"accuracy={terminal_report['accuracy']:.3f}, "
        f"balanced_accuracy={terminal_report['balanced_accuracy']:.3f}, "
        f"macro_F1={terminal_report['macro_f1']:.3f}"
    )
    print(f"example: {agent.explain(validation_summary)}")
    print(f"saved report: {output_dir / 'fd001_diagnosis_baseline.json'}")


if __name__ == "__main__":
    main()

