"""Compare FD004 prognostics models using grouped CV on fit engines only.

Calibration and final decision-evaluation engines from the closed-loop
development protocol are excluded from every fold. The official C-MAPSS test
files are never loaded. Model comparison uses fixed telemetry cycles and
out-of-fold engine predictions.
"""

from __future__ import annotations

from collections import defaultdict
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

from scripts.run_research_evaluation import OperatingConditionNormalizer
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
    verify_no_leakage,
)
from src.prognostics import NumpyLSTMRULModel, nasa_score


SEED = 42
MILESTONES = (30, 40, 60, 90, 120)
COMMON_MILESTONES = (40, 60, 90, 120)
WINDOW_SIZES = {"rf_summary_30": 30, "rf_summary_40": 40, "rf_condition_normalized_30": 30, "numpy_lstm_30": 30}
RF_TREES = 80
TRAIN_STRIDE = 5
BOOTSTRAP_REPLICATES = 2000
PAIRED_BOOTSTRAP_REPLICATES = 20000


def _cluster_bootstrap_ci(frame: pd.DataFrame, seed: int) -> list[float] | None:
    per_engine = frame.groupby("unit_id", sort=True)["absolute_error"].mean().to_numpy(dtype=float)
    if not len(per_engine):
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(per_engine), size=(BOOTSTRAP_REPLICATES, len(per_engine))
    )
    values = per_engine[indices].mean(axis=1)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _metrics(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    if frame.empty:
        return {"n_states": 0, "engine_count": 0}
    errors = frame["predicted_rul"].to_numpy(dtype=float) - frame["actual_rul"].to_numpy(dtype=float)
    return {
        "n_states": int(len(frame)),
        "engine_count": int(frame["unit_id"].nunique()),
        "mae_cycles": float(np.mean(np.abs(errors))),
        "engine_macro_mae_cycles": float(
            frame.groupby("unit_id")["absolute_error"].mean().mean()
        ),
        "engine_cluster_bootstrap_95_ci_mae": _cluster_bootstrap_ci(frame, seed),
        "rmse_cycles": float(np.sqrt(np.mean(errors**2))),
        "mean_signed_error_cycles": float(np.mean(errors)),
        "overprediction_rate": float(np.mean(errors > 0.0)),
        "mean_nasa_score_per_state": float(
            nasa_score(
                frame["actual_rul"].to_numpy(dtype=float),
                frame["predicted_rul"].to_numpy(dtype=float),
            )
            / len(frame)
        ),
    }


def _fit_rf(
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    seed: int,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=RF_TREES,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(summarize_windows(train_windows), train_targets)
    return model


def _validation_snapshots(
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    windows: list[np.ndarray] = []
    records: list[dict[str, int | float]] = []
    for unit_id, unit_frame in validation_frame.groupby("unit_id", sort=True):
        unit_frame = unit_frame.sort_values("cycle")
        cycles = unit_frame["cycle"].to_numpy(dtype=int)
        values = unit_frame[feature_columns].to_numpy(dtype=np.float32)
        targets = unit_frame["rul"].to_numpy(dtype=float)
        for milestone in MILESTONES:
            matches = np.flatnonzero(cycles == milestone)
            if not len(matches):
                continue
            end = int(matches[0])
            if end + 1 < window_size:
                continue
            windows.append(values[end - window_size + 1 : end + 1])
            records.append(
                {
                    "unit_id": int(unit_id),
                    "cycle": int(milestone),
                    "actual_rul": float(targets[end]),
                }
            )
    if not windows:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            pd.DataFrame(columns=["unit_id", "cycle", "actual_rul"]),
        )
    return np.stack(windows).astype(np.float32), pd.DataFrame(records)


def _fold_predictions(
    fold: int,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    dropped = constant_columns(train_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]

    # Standard RF models: each scaler is fitted on this fold's training engines.
    scaler = fit_feature_scaler(train_frame, features)
    train_scaled = transform_features(train_frame, features, scaler)
    validation_scaled = transform_features(validation_frame, features, scaler)
    train_windows_30, train_targets_30, _ = make_windows(
        train_scaled, features, window_size=30, stride=TRAIN_STRIDE
    )
    train_windows_40, train_targets_40, _ = make_windows(
        train_scaled, features, window_size=40, stride=TRAIN_STRIDE
    )
    validation_windows_30, validation_meta_30 = _validation_snapshots(
        validation_scaled, features, 30
    )
    validation_windows_40, validation_meta_40 = _validation_snapshots(
        validation_scaled, features, 40
    )

    rf30 = _fit_rf(train_windows_30, train_targets_30, SEED + fold)
    rf40 = _fit_rf(train_windows_40, train_targets_40, SEED + fold)
    rf30_predictions = np.maximum(
        rf30.predict(summarize_windows(validation_windows_30)), 0.0
    )
    rf40_predictions = np.maximum(
        rf40.predict(summarize_windows(validation_windows_40)), 0.0
    )

    lstm = NumpyLSTMRULModel(
        in_features=len(features), hidden_size=32, seed=SEED + fold
    )
    lstm.fit(
        train_windows_30,
        train_targets_30,
        max_epochs=8,
        batch_size=128,
        lr=1e-3,
    )
    lstm_predictions = np.maximum(
        lstm.predict(validation_windows_30), 0.0
    )

    # Condition centering is fitted on the training fold only. Its own scaler is
    # then fitted on the transformed training fold, never on validation engines.
    condition_normalizer = OperatingConditionNormalizer(seed=SEED + fold).fit(
        train_frame
    )
    train_condition = condition_normalizer.transform(train_frame)
    validation_condition = condition_normalizer.transform(validation_frame)
    condition_scaler = fit_feature_scaler(train_condition, features)
    train_condition_scaled = transform_features(
        train_condition, features, condition_scaler
    )
    validation_condition_scaled = transform_features(
        validation_condition, features, condition_scaler
    )
    condition_train_windows, condition_train_targets, _ = make_windows(
        train_condition_scaled,
        features,
        window_size=30,
        stride=TRAIN_STRIDE,
    )
    condition_validation_windows, condition_validation_meta = _validation_snapshots(
        validation_condition_scaled, features, 30
    )
    condition_rf = _fit_rf(
        condition_train_windows, condition_train_targets, SEED + fold
    )
    condition_predictions = np.maximum(
        condition_rf.predict(summarize_windows(condition_validation_windows)), 0.0
    )

    outputs: list[dict[str, Any]] = []
    prediction_sets = [
        ("rf_summary_30", validation_meta_30, rf30_predictions),
        ("rf_summary_40", validation_meta_40, rf40_predictions),
        ("numpy_lstm_30", validation_meta_30, lstm_predictions),
        (
            "rf_condition_normalized_30",
            condition_validation_meta,
            condition_predictions,
        ),
    ]
    for model_name, metadata, predictions in prediction_sets:
        for position, prediction in enumerate(predictions):
            record = metadata.iloc[position]
            actual = float(record["actual_rul"])
            predicted = float(prediction)
            outputs.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "unit_id": int(record["unit_id"]),
                    "cycle": int(record["cycle"]),
                    "actual_rul": actual,
                    "predicted_rul": predicted,
                    "signed_error": predicted - actual,
                    "absolute_error": abs(predicted - actual),
                }
            )
    return outputs


def _model_summary(predictions: pd.DataFrame, model_name: str) -> dict[str, Any]:
    model_rows = predictions[predictions["model"] == model_name].copy()
    common_rows = model_rows[model_rows["cycle"].isin(COMMON_MILESTONES)]
    per_cycle = {
        str(cycle): _metrics(
            model_rows[model_rows["cycle"] == cycle], seed=SEED + cycle
        )
        for cycle in MILESTONES
    }
    common_by_rul: dict[str, Any] = {}
    common_rows["actual_rul_band"] = pd.cut(
        common_rows["actual_rul"],
        bins=[-np.inf, 25.0, 50.0, 100.0, 150.0, np.inf],
        labels=["<=25", "25-50", "50-100", "100-150", ">150"],
        right=True,
    )
    for band, group in common_rows.groupby("actual_rul_band", observed=True):
        common_by_rul[str(band)] = _metrics(group, seed=SEED + len(str(band)))
    return {
        "all_supported_milestones": _metrics(model_rows, seed=SEED),
        "common_milestones_40_60_90_120": _metrics(common_rows, seed=SEED + 1),
        "common_milestone_metrics_by_rul_band": common_by_rul,
        "metrics_by_cycle": per_cycle,
        "prediction_state_count": int(len(model_rows)),
        "unique_validation_engine_count": int(model_rows["unit_id"].nunique()),
    }


def _paired_mae_differences(
    predictions: pd.DataFrame,
    baseline: str = "rf_summary_30",
) -> dict[str, dict[str, Any]]:
    common = predictions[
        predictions["cycle"].isin(COMMON_MILESTONES)
    ].copy()
    engine_model_mae = common.pivot_table(
        index="unit_id", columns="model", values="absolute_error", aggfunc="mean"
    )
    if baseline not in engine_model_mae.columns:
        raise ValueError(f"Missing baseline model {baseline}")

    results: dict[str, dict[str, Any]] = {}
    for model_name in engine_model_mae.columns:
        if model_name == baseline:
            continue
        differences = (
            engine_model_mae[model_name] - engine_model_mae[baseline]
        ).to_numpy(dtype=float)
        rng = np.random.default_rng(SEED + sum(ord(char) for char in model_name))
        indices = rng.integers(
            0,
            len(differences),
            size=(PAIRED_BOOTSTRAP_REPLICATES, len(differences)),
        )
        bootstrap_means = differences[indices].mean(axis=1)
        results[model_name] = {
            "baseline": baseline,
            "difference_definition": "candidate engine-macro MAE minus RF30 engine-macro MAE; negative favors candidate",
            "mean_paired_difference_cycles": float(differences.mean()),
            "paired_engine_cluster_bootstrap_95_ci_cycles": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "evaluation_engines_better_than_baseline_rate": float(
                np.mean(differences < 0.0)
            ),
            "evaluation_engine_count": int(len(differences)),
            "bootstrap_replicates": PAIRED_BOOTSTRAP_REPLICATES,
        }
    return results


def main() -> None:
    started = time.perf_counter()
    reports_dir = ROOT / "reports" / "generated"
    development_path = reports_dir / "closed_loop_development_evaluation.json"
    if not development_path.exists():
        raise FileNotFoundError(
            "Run scripts.evaluate_closed_loop_development before model selection"
        )
    development = json.loads(development_path.read_text(encoding="utf-8"))
    fd004_protocol = development["datasets"]["FD004"]["evaluation_protocol"]
    fit_units = tuple(int(value) for value in fd004_protocol["model_fit_engine_ids"])
    calibration_units = tuple(
        int(value) for value in fd004_protocol["uncertainty_calibration_engine_ids"]
    )
    decision_eval_units = tuple(
        int(value) for value in fd004_protocol["decision_evaluation_engine_ids"]
    )

    full_frame = add_train_rul(
        load_telemetry(ROOT / "data" / "raw" / "cmapss", "FD004", "train")
    )
    fit_frame = full_frame[full_frame["unit_id"].isin(fit_units)].copy()
    if (
        set(fit_units) & set(calibration_units)
        or set(fit_units) & set(decision_eval_units)
        or set(calibration_units) & set(decision_eval_units)
    ):
        raise RuntimeError("Calibration/evaluation engines leaked into model-selection data")

    engine_ids = np.asarray(sorted(fit_units), dtype=int)
    splitter = GroupKFold(n_splits=5)
    fold_reports: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(engine_ids, groups=engine_ids), start=1
    ):
        fold_train_units = tuple(int(unit) for unit in engine_ids[train_indices])
        fold_validation_units = tuple(
            int(unit) for unit in engine_ids[validation_indices]
        )
        verify_no_leakage(fold_train_units, fold_validation_units)
        fold_train = fit_frame[fit_frame["unit_id"].isin(fold_train_units)].copy()
        fold_validation = fit_frame[
            fit_frame["unit_id"].isin(fold_validation_units)
        ].copy()
        fold_predictions = _fold_predictions(fold, fold_train, fold_validation)
        all_predictions.extend(fold_predictions)
        fold_reports.append(
            {
                "fold": fold,
                "train_engine_count": len(fold_train_units),
                "validation_engine_count": len(fold_validation_units),
                "train_engine_ids": list(fold_train_units),
                "validation_engine_ids": list(fold_validation_units),
            }
        )
        print(
            f"completed fold {fold}/5: {len(fold_train_units)} fit engines, "
            f"{len(fold_validation_units)} held-out engines"
        )

    predictions = pd.DataFrame(all_predictions)
    model_names = list(WINDOW_SIZES)
    model_summaries = {
        model_name: _model_summary(predictions, model_name)
        for model_name in model_names
    }
    paired_differences = _paired_mae_differences(predictions)
    common_mae = {
        name: summary["common_milestones_40_60_90_120"]["engine_macro_mae_cycles"]
        for name, summary in model_summaries.items()
    }
    ranking = sorted(common_mae, key=common_mae.get)
    report = {
        "experiment": "FD004 prognostics model selection within fit-engine group",
        "official_test_files_loaded": False,
        "calibration_and_final_evaluation_engines_used_for_selection": False,
        "protocol": {
            "fit_engine_count": len(fit_units),
            "reserved_calibration_engine_count": len(calibration_units),
            "reserved_final_evaluation_engine_count": len(decision_eval_units),
            "fold_method": "5-fold GroupKFold by engine, within fit engines only",
            "preprocessing_fit_scope": "each fold's fit engines only",
            "fixed_milestones_cycles": list(MILESTONES),
            "common_comparison_milestones_cycles": list(COMMON_MILESTONES),
            "training_window_stride": TRAIN_STRIDE,
            "random_seed": SEED,
            "training_note": (
                "The same training-window stride and training engine fold are used "
                "for the 30-cycle RF and LSTM; RF40 uses 40-cycle windows."
            ),
        },
        "folds": fold_reports,
        "models": model_summaries,
        "paired_common_cycle_mae_differences_vs_rf_summary_30": paired_differences,
        "common_cycle_engine_macro_mae_ranking": ranking,
        "limitations": [
            "This is a model-selection experiment on fit engines, not the final held-out evaluation.",
            "The RUL bands use labels only after prediction for stratified reporting.",
            "Operating-condition normalization uses six train-fitted clusters; cluster IDs have no external physical label.",
            "A sequence model's worse result would not imply that every LSTM architecture is worse; this compares the configured NumPy baseline only.",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "fd004_model_selection_cv.json"
    predictions_path = reports_dir / "fd004_model_selection_oof_predictions.csv"
    metrics_path = reports_dir / "fd004_model_selection_metrics.csv"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    predictions.to_csv(predictions_path, index=False)
    metric_rows: list[dict[str, Any]] = []
    for model_name, summary in model_summaries.items():
        metric_rows.append(
            {
                "model": model_name,
                "comparison": "common_milestones_40_60_90_120",
                **summary["common_milestones_40_60_90_120"],
            }
        )
        for cycle, metrics in summary["metrics_by_cycle"].items():
            metric_rows.append(
                {"model": model_name, "comparison": f"cycle_{cycle}", **metrics}
            )
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    print("FD004 model selection by engine-grouped cross-validation")
    print("=========================================================")
    for rank, name in enumerate(ranking, start=1):
        summary = model_summaries[name]["common_milestones_40_60_90_120"]
        print(
            f"{rank}. {name}: engine-macro MAE="
            f"{summary['engine_macro_mae_cycles']:.2f} cycles "
            f"(cluster bootstrap CI {summary['engine_cluster_bootstrap_95_ci_mae']})"
        )
    print("paired engine-level MAE differences vs RF30 (negative favors candidate):")
    for model_name, comparison in paired_differences.items():
        print(
            f"  {model_name}: {comparison['mean_paired_difference_cycles']:.2f} cycles "
            f"(95% CI {comparison['paired_engine_cluster_bootstrap_95_ci_cycles']})"
        )
    print(f"OOF states: {len(predictions)}")
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"OOF predictions: {predictions_path.relative_to(ROOT)}")
    print(f"Metrics CSV: {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
