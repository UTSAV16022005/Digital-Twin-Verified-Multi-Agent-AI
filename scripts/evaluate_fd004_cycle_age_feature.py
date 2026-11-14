"""Test an explicit engine-age feature using FD004 fit-only grouped CV."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from src.cmapss_data import (
    FEATURE_COLUMNS,
    add_train_rul,
    constant_columns,
    fit_feature_scaler,
    load_telemetry,
    make_windows,
    summarize_windows,
    transform_features,
    verify_no_leakage,
)


SEED = 42
FOLDS = 5
MILESTONES = (30, 40, 60, 90, 120)
WINDOW_SIZE = 30
TRAIN_STRIDE = 5
RF_TREES = 80
BOOTSTRAP_REPLICATES = 20000
BASELINE_NAME = "rf_summary_30"
AGE_MODEL_NAME = "rf_summary_30_plus_engine_cycle"


def _paired_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    draws = values[indices].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _metric(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    if frame.empty:
        return {"n_states": 0, "engine_count": 0}
    errors = frame["predicted_rul"] - frame["actual_rul"]
    per_engine = frame.assign(abs_error=errors.abs()).groupby("unit_id")[
        "abs_error"
    ].mean()
    return {
        "n_states": int(len(frame)),
        "engine_count": int(frame["unit_id"].nunique()),
        "mae_cycles": float(errors.abs().mean()),
        "engine_macro_mae_cycles": float(per_engine.mean()),
        "engine_cluster_bootstrap_95_ci_mae": _paired_ci(
            per_engine.to_numpy(dtype=float), seed
        ),
        "mean_signed_error_cycles": float(errors.mean()),
        "overprediction_rate": float((errors > 0.0).mean()),
    }


def _fold_predictions(
    fit_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    fold: int,
) -> list[dict[str, Any]]:
    dropped = constant_columns(fit_frame, FEATURE_COLUMNS)
    features = [name for name in FEATURE_COLUMNS if name not in dropped]
    scaler = fit_feature_scaler(fit_frame, features)
    fit_scaled = transform_features(fit_frame, features, scaler)
    validation_scaled = transform_features(validation_frame, features, scaler)

    train_windows, targets, train_meta = make_windows(
        fit_scaled,
        features,
        window_size=WINDOW_SIZE,
        stride=TRAIN_STRIDE,
    )
    train_summary = summarize_windows(train_windows)
    train_age = train_meta["end_cycle"].to_numpy(dtype=float)[:, None] / 100.0
    train_with_age = np.concatenate([train_summary, train_age], axis=1)
    model = RandomForestRegressor(
        n_estimators=RF_TREES,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=SEED + fold,
        n_jobs=-1,
    ).fit(train_with_age, targets)

    windows: list[np.ndarray] = []
    records: list[dict[str, int | float]] = []
    for unit_id, unit_frame in validation_scaled.groupby("unit_id", sort=True):
        unit_frame = unit_frame.sort_values("cycle")
        cycles = unit_frame["cycle"].to_numpy(dtype=int)
        values = unit_frame[features].to_numpy(dtype=np.float32)
        targets = unit_frame["rul"].to_numpy(dtype=float)
        for cycle in MILESTONES:
            positions = np.flatnonzero(cycles == cycle)
            if not len(positions):
                continue
            end = int(positions[0])
            if end + 1 < WINDOW_SIZE:
                continue
            windows.append(values[end - WINDOW_SIZE + 1 : end + 1])
            records.append(
                {
                    "fold": fold,
                    "unit_id": int(unit_id),
                    "cycle": int(cycle),
                    "actual_rul": float(targets[end]),
                }
            )
    if not windows:
        raise ValueError("Fold generated no evaluation windows")
    summaries = summarize_windows(np.stack(windows).astype(np.float32))
    ages = np.asarray([row["cycle"] for row in records], dtype=float)[:, None] / 100.0
    predictions = np.maximum(
        model.predict(np.concatenate([summaries, ages], axis=1)), 0.0
    )
    return [
        {
            **record,
            "predicted_rul": float(prediction),
        }
        for record, prediction in zip(records, predictions)
    ]


def main() -> None:
    started = time.perf_counter()
    output_dir = ROOT / "reports" / "generated"
    development_path = output_dir / "closed_loop_development_evaluation.json"
    baseline_path = output_dir / "fd004_model_selection_oof_predictions.csv"
    if not development_path.exists() or not baseline_path.exists():
        raise FileNotFoundError(
            "Run the closed-loop development evaluation and FD004 model comparison first"
        )

    protocol = json.loads(development_path.read_text(encoding="utf-8"))["datasets"][
        "FD004"
    ]["evaluation_protocol"]
    fit_units = tuple(int(value) for value in protocol["model_fit_engine_ids"])
    calibration_units = tuple(
        int(value) for value in protocol["uncertainty_calibration_engine_ids"]
    )
    final_eval_units = tuple(
        int(value) for value in protocol["decision_evaluation_engine_ids"]
    )
    if (
        set(fit_units) & set(calibration_units)
        or set(fit_units) & set(final_eval_units)
        or set(calibration_units) & set(final_eval_units)
    ):
        raise RuntimeError("Engine groups overlap")

    all_train = add_train_rul(
        load_telemetry(ROOT / "data" / "raw" / "cmapss", "FD004", "train")
    )
    fit_data = all_train[all_train["unit_id"].isin(fit_units)].copy()
    engine_ids = np.asarray(sorted(fit_units), dtype=int)
    splitter = GroupKFold(n_splits=FOLDS)
    rows: list[dict[str, Any]] = []
    fold_info = []
    for fold, (fit_indices, validation_indices) in enumerate(
        splitter.split(engine_ids, groups=engine_ids), start=1
    ):
        fit_fold_units = tuple(int(unit) for unit in engine_ids[fit_indices])
        validation_fold_units = tuple(
            int(unit) for unit in engine_ids[validation_indices]
        )
        verify_no_leakage(fit_fold_units, validation_fold_units)
        fold_fit = fit_data[fit_data["unit_id"].isin(fit_fold_units)].copy()
        fold_validation = fit_data[
            fit_data["unit_id"].isin(validation_fold_units)
        ].copy()
        rows.extend(_fold_predictions(fold_fit, fold_validation, fold))
        fold_info.append(
            {
                "fold": fold,
                "fit_engine_count": len(fit_fold_units),
                "validation_engine_count": len(validation_fold_units),
            }
        )
        print(f"completed fold {fold}/{FOLDS}")

    age_predictions = pd.DataFrame(rows)
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[
        (baseline["model"] == BASELINE_NAME)
        & baseline["cycle"].isin(MILESTONES)
    ][["unit_id", "cycle", "actual_rul", "predicted_rul"]].rename(
        columns={"actual_rul": "baseline_actual_rul", "predicted_rul": "baseline_rul"}
    )
    age_predictions = age_predictions.merge(
        baseline,
        on=["unit_id", "cycle"],
        how="left",
        validate="one_to_one",
    )
    if age_predictions["baseline_rul"].isna().any():
        raise RuntimeError("Could not align age-feature OOF states with RF30 baseline")
    if not np.allclose(
        age_predictions["actual_rul"], age_predictions["baseline_actual_rul"]
    ):
        raise RuntimeError("Baseline and age-feature labels do not match")
    age_predictions["signed_error"] = (
        age_predictions["predicted_rul"] - age_predictions["actual_rul"]
    )
    age_predictions["absolute_error"] = age_predictions["signed_error"].abs()
    age_predictions["baseline_signed_error"] = (
        age_predictions["baseline_rul"] - age_predictions["actual_rul"]
    )
    age_predictions["baseline_absolute_error"] = age_predictions[
        "baseline_signed_error"
    ].abs()

    raw_frame = age_predictions[["unit_id", "cycle", "actual_rul"]].copy()
    raw_frame["predicted_rul"] = age_predictions["baseline_rul"].to_numpy()
    age_frame = age_predictions[["unit_id", "cycle", "actual_rul"]].copy()
    age_frame["predicted_rul"] = age_predictions["predicted_rul"].to_numpy()
    raw_metric = _metric(raw_frame, SEED)
    age_metric = _metric(age_frame, SEED + 1)

    per_engine = age_predictions.groupby("unit_id", sort=True).agg(
        raw_macro_mae=("baseline_absolute_error", "mean"),
        age_macro_mae=("absolute_error", "mean"),
    )
    paired_differences = (
        per_engine["age_macro_mae"] - per_engine["raw_macro_mae"]
    ).to_numpy(dtype=float)
    paired_ci = _paired_ci(paired_differences, SEED + 2)

    per_cycle: dict[str, Any] = {}
    for cycle in MILESTONES:
        group = age_predictions[age_predictions["cycle"] == cycle]
        raw_cycle = group.assign(
            predicted_rul=group["baseline_rul"].to_numpy(),
            absolute_error=group["baseline_absolute_error"].to_numpy(),
        )
        age_cycle = group.assign(
            predicted_rul=group["predicted_rul"].to_numpy(),
            absolute_error=group["absolute_error"].to_numpy(),
        )
        per_cycle[str(cycle)] = {
            "rf30": _metric(raw_cycle, SEED + cycle),
            "rf30_plus_cycle": _metric(age_cycle, SEED + cycle + 1),
        }

    report = {
        "experiment": "FD004 RF30 with explicit cycle-age feature",
        "official_test_files_loaded": False,
        "calibration_or_final_evaluation_engines_used_for_model_selection": False,
        "protocol": {
            "fit_engine_count": len(fit_units),
            "reserved_calibration_engine_count": len(calibration_units),
            "reserved_final_evaluation_engine_count": len(final_eval_units),
            "fold_method": "5-fold GroupKFold by engine inside the fit-engine group",
            "fixed_cycles": list(MILESTONES),
            "window_size": WINDOW_SIZE,
            "training_stride": TRAIN_STRIDE,
            "cycle_feature": "end_cycle divided by 100, appended to window summary statistics",
            "preprocessing_fit_scope": "each fold's training engines only",
        },
        "overall_metrics": {
            "rf30": raw_metric,
            "rf30_plus_cycle": age_metric,
            "paired_engine_macro_mae_difference_age_minus_rf30_cycles": float(
                paired_differences.mean()
            ),
            "paired_engine_cluster_bootstrap_95_ci_difference_cycles": paired_ci,
            "engines_with_lower_mae_after_age_feature": float(
                np.mean(paired_differences < 0.0)
            ),
        },
        "metrics_by_cycle": per_cycle,
        "folds": fold_info,
        "decision": (
            "The cycle feature is an experiment only. Keep the current RF30 model "
            "unless the paired engine-level results show a robust improvement."
        ),
        "limitations": [
            "This compares one added age feature on fit-engine cross-validation only.",
            "Actual RUL is used only after prediction for error metrics.",
            "The reserved calibration and final evaluation engine groups remain untouched.",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }

    json_path = output_dir / "fd004_cycle_age_feature_cv.json"
    csv_path = output_dir / "fd004_cycle_age_feature_oof_predictions.csv"
    age_predictions.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("FD004 explicit cycle-age feature evaluation")
    print("============================================")
    print(
        f"RF30 engine-macro MAE={raw_metric['engine_macro_mae_cycles']:.2f}; "
        f"RF30+cycle MAE={age_metric['engine_macro_mae_cycles']:.2f}"
    )
    print(
        "paired difference (cycle feature minus RF30): "
        f"{paired_differences.mean():.2f} cycles (95% CI {paired_ci})"
    )
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"OOF predictions: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
