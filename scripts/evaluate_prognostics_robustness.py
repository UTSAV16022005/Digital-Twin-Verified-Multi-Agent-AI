"""Evaluate variable-length RUL inputs and bias correction on FD004 development engines.

The official FD004 test files are intentionally not loaded by this experiment.
The training split is used to fit feature transforms and models; held-out
development engines are used to assess short telemetry prefixes and a
cross-fitted median-bias correction.
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
from src.prognostics import nasa_score


SEED = 42
STRIDE = 5
VARIABLE_WINDOW_SIZES = tuple(range(10, 31, 2))
FIXED_WINDOW_SIZES = (10, 20, 30)
PREFIX_LENGTHS = tuple(range(10, 31))
BOOTSTRAP_REPLICATES = 2000


def _fit_random_forest(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
) -> RandomForestRegressor:
    windows, targets, _ = make_windows(
        train_frame,
        feature_columns,
        window_size=window_size,
        target_column="rul",
        stride=STRIDE,
    )
    if len(windows) == 0:
        raise ValueError(f"No training windows for length {window_size}")
    model = RandomForestRegressor(
        n_estimators=80,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(summarize_windows(windows), targets)
    return model


def _fit_variable_window_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
) -> RandomForestRegressor:
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    max_window = max(VARIABLE_WINDOW_SIZES)

    for window_size in VARIABLE_WINDOW_SIZES:
        windows, targets, _ = make_windows(
            train_frame,
            feature_columns,
            window_size=window_size,
            target_column="rul",
            stride=STRIDE,
        )
        if len(windows) == 0:
            raise ValueError(f"No training windows for length {window_size}")
        summaries = summarize_windows(windows)
        length_feature = np.full(
            (len(summaries), 1), window_size / max_window, dtype=np.float32
        )
        feature_blocks.append(np.concatenate([summaries, length_feature], axis=1))
        target_blocks.append(targets)
        # Give each configured history length equal total influence. Short
        # windows otherwise contribute more samples because they begin sooner.
        weight_blocks.append(
            np.full(len(targets), 1.0 / (len(VARIABLE_WINDOW_SIZES) * len(targets)))
        )

    model = RandomForestRegressor(
        n_estimators=80,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(
        np.concatenate(feature_blocks),
        np.concatenate(target_blocks),
        sample_weight=np.concatenate(weight_blocks),
    )
    return model


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    signed_error = predicted - actual
    return {
        "n_engines": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mean_signed_error": float(np.mean(signed_error)),
        "overprediction_count": int(np.sum(signed_error > 0)),
        "overprediction_rate": float(np.mean(signed_error > 0)),
        "mean_nasa_score_per_engine": float(nasa_score(actual, predicted) / len(actual)),
    }


def _engine_bootstrap_mae_ci(
    actual: np.ndarray,
    predicted: np.ndarray,
    seed: int,
) -> list[float]:
    errors = np.abs(predicted - actual)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(errors), size=(BOOTSTRAP_REPLICATES, len(errors)))
    estimates = errors[samples].mean(axis=1)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _cross_fitted_bias_adjustment(
    actual: np.ndarray,
    predicted: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    """Estimate bias on other engines, then correct each held-out fold."""
    corrected = np.empty_like(predicted, dtype=float)
    offsets: list[float] = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for calibration_indices, heldout_indices in splitter.split(actual):
        offset = float(
            np.median(predicted[calibration_indices] - actual[calibration_indices])
        )
        corrected[heldout_indices] = np.maximum(
            predicted[heldout_indices] - offset, 0.0
        )
        offsets.append(offset)
    return corrected, offsets


def _summarize_window(
    model: RandomForestRegressor,
    history: np.ndarray,
    history_cycles: int,
    variable_length: bool,
) -> float:
    summary = summarize_windows(history[np.newaxis, :, :])
    if variable_length:
        length_feature = np.array(
            [[history_cycles / max(VARIABLE_WINDOW_SIZES)]], dtype=np.float32
        )
        summary = np.concatenate([summary, length_feature], axis=1)
    return float(max(model.predict(summary)[0], 0.0))


def main() -> None:
    started = time.perf_counter()
    data_dir = Path("data/raw/cmapss")
    full_train = add_train_rul(load_telemetry(data_dir, "FD004", "train"))
    split = split_engine_units(full_train, validation_fraction=0.2, seed=SEED)
    train_frame = full_train[
        full_train["unit_id"].isin(split.train_units)
    ].copy()
    validation_frame = full_train[
        full_train["unit_id"].isin(split.validation_units)
    ].copy()

    dropped = constant_columns(train_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]
    scaler = fit_feature_scaler(train_frame, features)
    train_scaled = transform_features(train_frame, features, scaler)
    validation_scaled = transform_features(validation_frame, features, scaler)

    variable_model = _fit_variable_window_model(train_scaled, features)
    fixed_models = {
        window_size: _fit_random_forest(train_scaled, features, window_size)
        for window_size in FIXED_WINDOW_SIZES
    }

    engine_frames = {
        int(unit_id): unit_frame.sort_values("cycle")
        for unit_id, unit_frame in validation_scaled.groupby("unit_id", sort=True)
    }
    prediction_rows: list[dict[str, Any]] = []
    for unit_id, unit_frame in engine_frames.items():
        values = unit_frame[features].to_numpy(dtype=np.float32)
        rul_values = unit_frame["rul"].to_numpy(dtype=float)
        for history_cycles in PREFIX_LENGTHS:
            if len(values) < history_cycles:
                continue
            history = values[:history_cycles]
            actual_rul = float(rul_values[history_cycles - 1])
            row: dict[str, Any] = {
                "unit_id": unit_id,
                "history_cycles": history_cycles,
                "end_cycle": int(unit_frame.iloc[history_cycles - 1]["cycle"]),
                "actual_rul": actual_rul,
                "variable_window_predicted_rul": _summarize_window(
                    variable_model, history, history_cycles, variable_length=True
                ),
            }
            if history_cycles in fixed_models:
                row["fixed_window_predicted_rul"] = _summarize_window(
                    fixed_models[history_cycles],
                    history,
                    history_cycles,
                    variable_length=False,
                )
            else:
                row["fixed_window_predicted_rul"] = np.nan
            prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise RuntimeError("No validation prefixes were generated")

    summary_rows: list[dict[str, Any]] = []
    for history_cycles, group in predictions.groupby("history_cycles", sort=True):
        group = group.sort_values("unit_id").reset_index(drop=True)
        actual = group["actual_rul"].to_numpy(dtype=float)
        raw = group["variable_window_predicted_rul"].to_numpy(dtype=float)
        corrected, offsets = _cross_fitted_bias_adjustment(
            actual, raw, seed=SEED + int(history_cycles)
        )
        row: dict[str, Any] = {
            "history_cycles": int(history_cycles),
            "variable_window": _metrics(actual, raw),
            "variable_window_mae_engine_bootstrap_ci95": _engine_bootstrap_mae_ci(
                actual, raw, seed=SEED + int(history_cycles)
            ),
            "cross_fitted_bias_adjusted": _metrics(actual, corrected),
            "cross_fitted_bias_adjusted_mae_engine_bootstrap_ci95": (
                _engine_bootstrap_mae_ci(
                    actual, corrected, seed=SEED + 1000 + int(history_cycles)
                )
            ),
            "median_crossfit_bias_offset": float(np.median(offsets)),
        }
        if group["fixed_window_predicted_rul"].notna().all():
            fixed = group["fixed_window_predicted_rul"].to_numpy(dtype=float)
            row["fixed_window_reference"] = _metrics(actual, fixed)
        else:
            row["fixed_window_reference"] = None
        summary_rows.append(row)

    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "fd004_variable_history_predictions.csv"
    summary_path = output_dir / "fd004_variable_history_summary.json"
    predictions.to_csv(prediction_path, index=False)

    report = {
        "experiment": "variable-history FD004 prognostics and cross-fitted bias adjustment",
        "official_test_used": False,
        "subset": "FD004 training trajectories only",
        "seed": SEED,
        "fit_engines": len(split.train_units),
        "validation_engines": len(split.validation_units),
        "features_used": features,
        "features_dropped_as_constant": dropped,
        "fit_scaler_on_fit_engines_only": True,
        "variable_window_sizes_seen_during_training": list(VARIABLE_WINDOW_SIZES),
        "variable_window_length_feature": "history_cycles / 30",
        "sample_weighting": "equal total contribution per configured training window size",
        "bootstrap": {
            "unit": "validation engine",
            "replicates": BOOTSTRAP_REPLICATES,
            "method": "percentile bootstrap",
        },
        "prefix_results": summary_rows,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "interpretation": (
            "Bias-adjusted results are cross-fitted by validation engine. "
            "They are a development experiment, not an official test result or "
            "a deployed policy."
        ),
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("FD004 short-history development experiment")
    print("==========================================")
    print(
        f"fit engines={len(split.train_units)}, validation engines={len(split.validation_units)}, "
        f"official test used={report['official_test_used']}"
    )
    for row in summary_rows:
        length = row["history_cycles"]
        variable = row["variable_window"]
        corrected = row["cross_fitted_bias_adjusted"]
        print(
            f"history={length:2d}: variable MAE={variable['mae']:.2f}, "
            f"bias={variable['mean_signed_error']:+.2f}, "
            f"over={variable['overprediction_rate']:.0%}; "
            f"cross-fit adjusted MAE={corrected['mae']:.2f}"
        )
    print(f"saved predictions: {prediction_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
