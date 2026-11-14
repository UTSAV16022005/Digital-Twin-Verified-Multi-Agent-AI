"""Evaluate cross-fitted point calibration for FD004 RUL predictions.

Nested engine-grouped out-of-fold predictions from the reserved fit-engine
group train an isotonic calibration map. Outer validation engines estimate
whether the map corrects the life-stage bias. Calibration engines, final
decision-evaluation engines, and official test files are excluded.
"""

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
from sklearn.isotonic import IsotonicRegression
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
OUTER_FOLDS = 5
INNER_FOLDS = 4
MILESTONES = (30, 40, 60, 90, 120)
WINDOW_SIZE = 30
TRAIN_STRIDE = 5
RF_TREES = 80
BOOTSTRAP_REPLICATES = 20000


def _fit_rf(train_frame: pd.DataFrame, validation_frame: pd.DataFrame, seed: int):
    dropped = constant_columns(train_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]
    scaler = fit_feature_scaler(train_frame, features)
    train_scaled = transform_features(train_frame, features, scaler)
    validation_scaled = transform_features(validation_frame, features, scaler)
    train_windows, train_targets, _ = make_windows(
        train_scaled,
        features,
        window_size=WINDOW_SIZE,
        stride=TRAIN_STRIDE,
    )
    model = RandomForestRegressor(
        n_estimators=RF_TREES,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(summarize_windows(train_windows), train_targets)

    rows: list[dict[str, Any]] = []
    for unit_id, unit_frame in validation_scaled.groupby("unit_id", sort=True):
        unit_frame = unit_frame.sort_values("cycle")
        cycles = unit_frame["cycle"].to_numpy(dtype=int)
        values = unit_frame[features].to_numpy(dtype=np.float32)
        actual_rul = unit_frame["rul"].to_numpy(dtype=float)
        for cycle in MILESTONES:
            indices = np.flatnonzero(cycles == cycle)
            if not len(indices):
                continue
            end = int(indices[0])
            if end + 1 < WINDOW_SIZE:
                continue
            rows.append(
                {
                    "unit_id": int(unit_id),
                    "cycle": int(cycle),
                    "actual_rul": float(actual_rul[end]),
                    "window": values[end - WINDOW_SIZE + 1 : end + 1],
                }
            )
    if not rows:
        raise ValueError("Fold produced no fixed-cycle validation states")
    summary = summarize_windows(
        np.stack([row["window"] for row in rows]).astype(np.float32)
    )
    predictions = np.maximum(model.predict(summary), 0.0)
    for row, prediction in zip(rows, predictions):
        row["predicted_rul"] = float(prediction)
        del row["window"]
    return rows


def _cluster_bootstrap(values_by_engine: pd.Series, seed: int) -> list[float]:
    values = values_by_engine.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    estimates = values[indices].mean(axis=1)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _summarize(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    if frame.empty:
        return {"state_count": 0, "engine_count": 0}
    errors = frame["predicted_rul"] - frame["actual_rul"]
    absolute = errors.abs()
    per_engine_mae = frame.assign(absolute_error=absolute).groupby("unit_id")[
        "absolute_error"
    ].mean()
    return {
        "state_count": int(len(frame)),
        "engine_count": int(frame["unit_id"].nunique()),
        "mae_cycles": float(absolute.mean()),
        "engine_macro_mae_cycles": float(per_engine_mae.mean()),
        "engine_cluster_bootstrap_95_ci_mae": _cluster_bootstrap(
            per_engine_mae, seed
        ),
        "rmse_cycles": float(np.sqrt(np.mean(errors.to_numpy(dtype=float) ** 2))),
        "mean_signed_error_cycles": float(errors.mean()),
        "overprediction_rate": float((errors > 0.0).mean()),
    }


def _summarize_variants(
    predictions: pd.DataFrame,
    raw_column: str,
    corrected_column: str,
    seed: int,
) -> dict[str, Any]:
    raw = predictions[
        ["unit_id", "cycle", "actual_rul", raw_column]
    ].rename(columns={raw_column: "predicted_rul"})
    corrected = predictions[
        ["unit_id", "cycle", "actual_rul", corrected_column]
    ].rename(columns={corrected_column: "predicted_rul"})

    per_cycle: dict[str, Any] = {}
    for cycle in MILESTONES:
        per_cycle[str(cycle)] = {
            "raw": _summarize(raw[raw["cycle"] == cycle], seed + cycle),
            "isotonic": _summarize(
                corrected[corrected["cycle"] == cycle], seed + cycle + 1
            ),
        }

    raw_engine = raw.assign(
        absolute_error=(raw["predicted_rul"] - raw["actual_rul"]).abs()
    ).groupby("unit_id")["absolute_error"].mean()
    corrected_engine = corrected.assign(
        absolute_error=(corrected["predicted_rul"] - corrected["actual_rul"]).abs()
    ).groupby("unit_id")["absolute_error"].mean()
    paired_difference = corrected_engine - raw_engine
    paired_ci = _cluster_bootstrap(paired_difference, seed + 100)

    raw_by_band = raw.copy()
    corrected_by_band = corrected.copy()
    bins = [-np.inf, 25.0, 50.0, 100.0, 150.0, np.inf]
    labels = ["<=25", "25-50", "50-100", "100-150", ">150"]
    for frame in (raw_by_band, corrected_by_band):
        frame["actual_rul_band"] = pd.cut(
            frame["actual_rul"], bins=bins, labels=labels, right=True
        )
    by_band: dict[str, Any] = {}
    for band in labels:
        by_band[band] = {
            "raw": _summarize(
                raw_by_band[raw_by_band["actual_rul_band"] == band], seed + len(band)
            ),
            "isotonic": _summarize(
                corrected_by_band[
                    corrected_by_band["actual_rul_band"] == band
                ],
                seed + len(band) + 1,
            ),
        }

    return {
        "all_fixed_milestones": {
            "raw": _summarize(raw, seed),
            "isotonic": _summarize(corrected, seed + 1),
            "paired_engine_macro_mae_difference_isotonic_minus_raw_cycles": float(
                paired_difference.mean()
            ),
            "paired_engine_cluster_bootstrap_95_ci_difference_cycles": paired_ci,
            "engines_with_lower_macro_mae_after_calibration": float(
                (paired_difference < 0.0).mean()
            ),
        },
        "by_cycle": per_cycle,
        "by_actual_rul_band": by_band,
    }


def main() -> None:
    started = time.perf_counter()
    reports_dir = ROOT / "reports" / "generated"
    development_path = reports_dir / "closed_loop_development_evaluation.json"
    model_selection_path = reports_dir / "fd004_model_selection_oof_predictions.csv"
    if not development_path.exists() or not model_selection_path.exists():
        raise FileNotFoundError(
            "Run the closed-loop development evaluation and FD004 model comparison first"
        )

    development = json.loads(development_path.read_text(encoding="utf-8"))
    protocol = development["datasets"]["FD004"]["evaluation_protocol"]
    fit_units = tuple(int(unit) for unit in protocol["model_fit_engine_ids"])
    calibration_units = tuple(
        int(unit) for unit in protocol["uncertainty_calibration_engine_ids"]
    )
    final_eval_units = tuple(
        int(unit) for unit in protocol["decision_evaluation_engine_ids"]
    )

    all_train = add_train_rul(
        load_telemetry(ROOT / "data" / "raw" / "cmapss", "FD004", "train")
    )
    fit_data = all_train[all_train["unit_id"].isin(fit_units)].copy()
    if (
        set(fit_units) & set(calibration_units)
        or set(fit_units) & set(final_eval_units)
        or set(calibration_units) & set(final_eval_units)
    ):
        raise RuntimeError("Fit, calibration, and final evaluation engine groups overlap")

    engine_ids = np.asarray(sorted(fit_units), dtype=int)
    outer_splitter = GroupKFold(n_splits=OUTER_FOLDS)
    all_rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for outer_fold, (outer_train_idx, outer_validation_idx) in enumerate(
        outer_splitter.split(engine_ids, groups=engine_ids), start=1
    ):
        outer_train_units = tuple(int(u) for u in engine_ids[outer_train_idx])
        outer_validation_units = tuple(
            int(u) for u in engine_ids[outer_validation_idx]
        )
        verify_no_leakage(outer_train_units, outer_validation_units)
        outer_train = fit_data[
            fit_data["unit_id"].isin(outer_train_units)
        ].copy()
        outer_validation = fit_data[
            fit_data["unit_id"].isin(outer_validation_units)
        ].copy()

        inner_engine_ids = np.asarray(sorted(outer_train_units), dtype=int)
        inner_splitter = GroupKFold(n_splits=INNER_FOLDS)
        inner_oof_rows: list[dict[str, Any]] = []
        inner_fold_records: list[dict[str, Any]] = []
        for inner_fold, (inner_train_idx, inner_validation_idx) in enumerate(
            inner_splitter.split(inner_engine_ids, groups=inner_engine_ids), start=1
        ):
            inner_train_units = tuple(
                int(u) for u in inner_engine_ids[inner_train_idx]
            )
            inner_validation_units = tuple(
                int(u) for u in inner_engine_ids[inner_validation_idx]
            )
            verify_no_leakage(inner_train_units, inner_validation_units)
            inner_train = outer_train[
                outer_train["unit_id"].isin(inner_train_units)
            ].copy()
            inner_validation = outer_train[
                outer_train["unit_id"].isin(inner_validation_units)
            ].copy()
            fold_rows = _fit_rf(
                inner_train,
                inner_validation,
                seed=SEED + outer_fold * 100 + inner_fold,
            )
            inner_oof_rows.extend(fold_rows)
            inner_fold_records.append(
                {
                    "fold": inner_fold,
                    "train_engine_count": len(inner_train_units),
                    "validation_engine_count": len(inner_validation_units),
                }
            )

        inner_oof = pd.DataFrame(inner_oof_rows)
        per_engine_state_counts = inner_oof.groupby("unit_id")["cycle"].transform(
            "count"
        )
        calibrator = IsotonicRegression(
            y_min=0.0,
            increasing=True,
            out_of_bounds="clip",
        )
        calibrator.fit(
            inner_oof["predicted_rul"].to_numpy(dtype=float),
            inner_oof["actual_rul"].to_numpy(dtype=float),
            sample_weight=(1.0 / per_engine_state_counts.to_numpy(dtype=float)),
        )

        outer_rows = _fit_rf(
            outer_train,
            outer_validation,
            seed=SEED + outer_fold,
        )
        outer_frame = pd.DataFrame(outer_rows)
        outer_frame["isotonic_rul"] = np.maximum(
            calibrator.predict(outer_frame["predicted_rul"].to_numpy(dtype=float)),
            0.0,
        )
        outer_frame["fold"] = outer_fold
        all_rows.extend(outer_frame.to_dict(orient="records"))
        fold_records.append(
            {
                "fold": outer_fold,
                "fit_engine_count": len(outer_train_units),
                "validation_engine_count": len(outer_validation_units),
                "inner_oof_prediction_count": int(len(inner_oof)),
                "inner_calibrator_training_engine_count": int(
                    inner_oof["unit_id"].nunique()
                ),
                "isotonic_knot_count": int(len(calibrator.X_thresholds_)),
                "inner_folds": inner_fold_records,
            }
        )
        print(
            f"completed outer fold {outer_fold}/{OUTER_FOLDS}; "
            f"inner OOF states={len(inner_oof)}, held-out engines={len(outer_validation_units)}"
        )

    predictions = pd.DataFrame(all_rows)
    baseline = pd.read_csv(model_selection_path)
    baseline = baseline[baseline["model"] == "rf_summary_30"]
    baseline = baseline[["unit_id", "cycle", "predicted_rul"]].rename(
        columns={"predicted_rul": "saved_cv_prediction"}
    )
    predictions = predictions.merge(
        baseline,
        on=["unit_id", "cycle"],
        how="left",
        validate="one_to_one",
    )
    if predictions["saved_cv_prediction"].isna().any():
        raise RuntimeError("The nested CV did not match all saved baseline states")
    max_baseline_difference = float(
        np.max(
            np.abs(
                predictions["predicted_rul"].to_numpy(dtype=float)
                - predictions["saved_cv_prediction"].to_numpy(dtype=float)
            )
        )
    )
    if max_baseline_difference > 1e-5:
        raise RuntimeError(
            "Raw predictions differ from the previously saved RF30 cross-validation "
            f"(maximum difference {max_baseline_difference:.6g})"
        )

    summaries = _summarize_variants(
        predictions,
        raw_column="predicted_rul",
        corrected_column="isotonic_rul",
        seed=SEED,
    )
    report = {
        "experiment": "nested cross-fitted isotonic calibration for FD004 RUL",
        "official_test_files_loaded": False,
        "reserved_calibration_and_final_evaluation_engines_used": False,
        "protocol": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "outer_and_inner_split_unit": "whole engine trajectory",
            "fit_engine_count": len(fit_units),
            "reserved_calibration_engine_count": len(calibration_units),
            "reserved_final_evaluation_engine_count": len(final_eval_units),
            "fixed_cycles": list(MILESTONES),
            "calibrator": "monotone isotonic regression from inner OOF predicted RUL to actual RUL",
            "calibrator_weights": "equal total weight per engine across its fixed-cycle states",
            "base_model": {
                "type": "RandomForestRegressor",
                "window_size": WINDOW_SIZE,
                "training_stride": TRAIN_STRIDE,
                "n_estimators": RF_TREES,
                "min_samples_leaf": 2,
                "max_features": 0.7,
            },
            "raw_oof_predictions_match_previous_model_comparison": True,
            "maximum_absolute_baseline_prediction_difference": max_baseline_difference,
        },
        "folds": fold_records,
        "metrics": summaries,
        "decision": (
            "This calibration is an experiment only. Promote it only if the paired "
            "engine-level interval supports a consistent held-out improvement; it "
            "does not alter the production PrognosticsAgent."
        ),
        "limitations": [
            "The isotonic map is trained on fit-engine inner OOF predictions and evaluated on outer fit-engine folds only.",
            "The reserved calibration and final evaluation engines remain unused, but repeated fixed-cycle states are clustered by engine for uncertainty estimates.",
            "RUL-band summaries use actual RUL only after prediction for analysis.",
            "This calibrates point estimates; it does not recalibrate uncertainty or validate recovery actions.",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }

    output_dir = ROOT / "reports" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fd004_bias_calibration_cv.json"
    predictions_path = output_dir / "fd004_bias_calibration_oof_predictions.csv"
    metrics_path = output_dir / "fd004_bias_calibration_metrics.csv"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    predictions.to_csv(predictions_path, index=False)

    metric_rows: list[dict[str, Any]] = []
    for cycle, variants in summaries["by_cycle"].items():
        for variant, metrics in variants.items():
            metric_rows.append(
                {"slice_type": "cycle", "slice": cycle, "variant": variant, **metrics}
            )
    for band, variants in summaries["by_actual_rul_band"].items():
        for variant, metrics in variants.items():
            metric_rows.append(
                {"slice_type": "actual_rul_band", "slice": band, "variant": variant, **metrics}
            )
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    overall = summaries["all_fixed_milestones"]
    print("FD004 cross-fitted point-calibration evaluation")
    print("================================================")
    for variant in ("raw", "isotonic"):
        metric = overall[variant]
        print(
            f"{variant}: engine-macro MAE={metric['engine_macro_mae_cycles']:.2f} "
            f"cycles (95% CI {metric['engine_cluster_bootstrap_95_ci_mae']}); "
            f"bias={metric['mean_signed_error_cycles']:.2f}"
        )
    print(
        "paired isotonic minus raw engine-macro MAE: "
        f"{overall['paired_engine_macro_mae_difference_isotonic_minus_raw_cycles']:.2f} "
        f"cycles (95% CI {overall['paired_engine_cluster_bootstrap_95_ci_difference_cycles']})"
    )
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"OOF predictions: {predictions_path.relative_to(ROOT)}")
    print(f"Metrics CSV: {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
