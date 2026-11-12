"""Run grouped engine-level validation and reproducible model ablations.

The official test sets are intentionally not read by this experiment script.
Cross-validation and ablations use only run-to-failure training engines.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.cmapss_data import (
    FEATURE_COLUMNS,
    OPERATING_COLUMNS,
    SENSOR_COLUMNS,
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
from src.prognostics import NumpyLSTMRULModel, nasa_score, select_rul_stratified_rows

SEED = 42
CV_STRIDE = 5
CV_FOLDS = 5
RF_TREES = 80
CV_PROTOCOL_VERSION = 5


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _terminal_indices(metadata: pd.DataFrame) -> np.ndarray:
    return metadata.groupby("unit_id", sort=True).tail(1).index.to_numpy(dtype=int)


def _pseudo_current_indices(metadata: pd.DataFrame, seed: int) -> np.ndarray:
    return select_rul_stratified_rows(metadata, rul_reference=125.0, seed=seed).index.to_numpy(dtype=int)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    if len(actual) == 0 or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metrics require non-empty, finite target and prediction arrays")
    r2 = float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 1e-12 else None
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": r2,
        "nasa_score": float(nasa_score(actual, predicted)),
    }


def _bootstrap_mean_ci(values: np.ndarray, seed: int = SEED, replicates: int = 3000) -> dict[str, Any]:
    """Bootstrap a statistic from one independent contribution per engine."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap inputs must be non-empty finite engine-level values")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[sample_indices].mean(axis=1)
    return {
        "n_engines": int(len(values)),
        "estimate": float(values.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "method": "percentile bootstrap resampling independent engine units",
        "replicates": replicates,
    }


class OperatingConditionNormalizer:
    """Normalize sensor readings around train-fitted operating-condition clusters."""

    def __init__(self, seed: int = SEED, max_clusters: int = 6) -> None:
        self.seed = seed
        self.max_clusters = max_clusters
        self.setting_mean: np.ndarray | None = None
        self.setting_scale: np.ndarray | None = None
        self.cluster_centers: np.ndarray | None = None
        self.cluster_sensor_mean: dict[int, np.ndarray] = {}
        self.cluster_sensor_scale: dict[int, np.ndarray] = {}

    def fit(self, frame: pd.DataFrame) -> "OperatingConditionNormalizer":
        settings = frame[OPERATING_COLUMNS].to_numpy(dtype=float)
        sensors = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(settings).all() or not np.isfinite(sensors).all():
            raise ValueError("condition normalizer input must be finite")
        self.setting_mean = settings.mean(axis=0)
        setting_scale = settings.std(axis=0, ddof=0)
        self.setting_scale = np.where(setting_scale > 1e-8, setting_scale, 1.0)
        settings_scaled = (settings - self.setting_mean) / self.setting_scale
        unique_settings = np.unique(np.round(settings_scaled, 3), axis=0)
        n_clusters = max(1, min(self.max_clusters, len(unique_settings)))
        # Deterministic farthest-point initialization over unique operating
        # points, followed by Lloyd updates. This avoids optional native cluster
        # libraries and keeps this ablation runnable in restricted environments.
        first = int(np.lexsort(tuple(unique_settings[:, i] for i in reversed(range(unique_settings.shape[1]))))[0])
        initial = [unique_settings[first]]
        min_distance = np.sum((unique_settings - initial[0]) ** 2, axis=1)
        while len(initial) < n_clusters:
            next_index = int(np.argmax(min_distance))
            initial.append(unique_settings[next_index])
            min_distance = np.minimum(
                min_distance,
                np.sum((unique_settings - unique_settings[next_index]) ** 2, axis=1),
            )
        self.cluster_centers = np.asarray(initial, dtype=float)
        for _ in range(50):
            labels = self._labels(settings_scaled)
            updated = self.cluster_centers.copy()
            for cluster in range(n_clusters):
                if np.any(labels == cluster):
                    updated[cluster] = settings_scaled[labels == cluster].mean(axis=0)
            if np.allclose(updated, self.cluster_centers, atol=1e-5):
                self.cluster_centers = updated
                break
            self.cluster_centers = updated
        labels = self._labels(settings_scaled)
        for cluster in range(n_clusters):
            selected = sensors[labels == cluster]
            if len(selected) == 0:
                raise RuntimeError(f"training cluster {cluster} has no sensor observations")
            self.cluster_sensor_mean[cluster] = selected.mean(axis=0)
            scale = selected.std(axis=0, ddof=0)
            self.cluster_sensor_scale[cluster] = np.where(scale > 1e-8, scale, 1.0)
        return self

    def _labels(self, settings_scaled: np.ndarray) -> np.ndarray:
        if self.cluster_centers is None:
            raise RuntimeError("fit the operating-condition normalizer before transform")
        distances = np.sum(
            (settings_scaled[:, None, :] - self.cluster_centers[None, :, :]) ** 2,
            axis=2,
        )
        return distances.argmin(axis=1)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.cluster_centers is None or self.setting_mean is None or self.setting_scale is None:
            raise RuntimeError("fit the operating-condition normalizer before transform")
        result = frame.copy()
        settings_scaled = (
            frame[OPERATING_COLUMNS].to_numpy(dtype=float) - self.setting_mean
        ) / self.setting_scale
        labels = self._labels(settings_scaled)
        sensor_values = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
        normalized = np.empty_like(sensor_values, dtype=float)
        for cluster in np.unique(labels):
            mask = labels == cluster
            normalized[mask] = (
                sensor_values[mask] - self.cluster_sensor_mean[int(cluster)]
            ) / self.cluster_sensor_scale[int(cluster)]
        result = result.astype({name: "float64" for name in SENSOR_COLUMNS})
        result.loc[:, SENSOR_COLUMNS] = normalized
        return result


def _fold_features(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    window_size: int = 30,
    stride: int = CV_STRIDE,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    dropped = constant_columns(train_frame, feature_columns)
    selected = [name for name in feature_columns if name not in dropped]
    scaler = fit_feature_scaler(train_frame, selected)
    train_scaled = transform_features(train_frame, selected, scaler)
    validation_scaled = transform_features(validation_frame, selected, scaler)
    train_windows, train_y, _ = make_windows(train_scaled, selected, window_size, "rul", stride)
    validation_windows, validation_y, validation_meta = make_windows(
        validation_scaled, selected, window_size, "rul", stride
    )
    if not len(train_windows) or not len(validation_windows):
        raise ValueError("fold did not produce train and validation windows")
    return train_windows, train_y, train_scaled, validation_windows, validation_y, validation_meta


def run_grouped_cv(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Evaluate models on engine-grouped folds at pseudo-current and failure points."""
    units = np.asarray(sorted(frame["unit_id"].unique()), dtype=int)
    if len(units) < CV_FOLDS:
        raise ValueError(f"need at least {CV_FOLDS} engines for grouped CV")
    splitter = GroupKFold(n_splits=CV_FOLDS)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_sets: list[dict[str, Any]] = []
    xgb_class = None
    try:
        from xgboost import XGBRegressor

        xgb_class = XGBRegressor
    except ImportError:
        pass

    for fold, (train_i, validation_i) in enumerate(splitter.split(units, groups=units), start=1):
        train_units = tuple(int(u) for u in units[train_i])
        validation_units = tuple(int(u) for u in units[validation_i])
        verify_no_leakage(train_units, validation_units)
        train = frame[frame["unit_id"].isin(train_units)].copy()
        validation = frame[frame["unit_id"].isin(validation_units)].copy()
        (train_windows, train_y, _, validation_windows, validation_y, validation_meta) = _fold_features(
            train, validation, FEATURE_COLUMNS
        )
        train_summary = summarize_windows(train_windows)
        validation_summary = summarize_windows(validation_windows)
        evaluation_points = {
            "pseudo_current": _pseudo_current_indices(validation_meta, SEED + fold),
            "failure_endpoint": _terminal_indices(validation_meta),
        }
        models: dict[str, Any] = {
            "ridge": Ridge(alpha=10.0),
            "random_forest": RandomForestRegressor(
                n_estimators=RF_TREES,
                min_samples_leaf=2,
                max_features=0.7,
                random_state=SEED + fold,
                n_jobs=-1,
            ),
        }
        if xgb_class is not None:
            models["xgboost"] = xgb_class(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=SEED + fold,
                n_jobs=-1,
            )
        for name, model in models.items():
            model.fit(train_summary, train_y)
            predictions = np.maximum(model.predict(validation_summary), 0.0)
            for evaluation_point, indices in evaluation_points.items():
                point_metrics = _metrics(validation_y[indices], predictions[indices])
                fold_rows.append({"fold": fold, "model": name, "evaluation_point": evaluation_point, **point_metrics})
                for idx in indices:
                    prediction_rows.append({
                        "fold": fold,
                        "unit_id": int(validation_meta.iloc[idx]["unit_id"]),
                        "evaluation_point": evaluation_point,
                        "actual_rul": float(validation_y[idx]),
                        "predicted_rul": float(predictions[idx]),
                        "model": name,
                    })

        lstm = NumpyLSTMRULModel(in_features=train_windows.shape[2], hidden_size=32, seed=SEED + fold)
        lstm.fit(
            train_windows,
            train_y,
            validation_windows,
            validation_y,
            max_epochs=8,
            batch_size=128,
            lr=1e-3,
        )
        lstm_predictions = lstm.predict(validation_windows)
        for evaluation_point, indices in evaluation_points.items():
            point_metrics = _metrics(validation_y[indices], lstm_predictions[indices])
            fold_rows.append({
                "fold": fold,
                "model": "numpy_lstm",
                "evaluation_point": evaluation_point,
                **point_metrics,
            })
            for idx in indices:
                prediction_rows.append({
                    "fold": fold,
                    "unit_id": int(validation_meta.iloc[idx]["unit_id"]),
                    "evaluation_point": evaluation_point,
                    "actual_rul": float(validation_y[idx]),
                    "predicted_rul": float(lstm_predictions[idx]),
                    "model": "numpy_lstm",
                })
        fold_sets.append({
            "fold": fold,
            "train_units": list(train_units),
            "validation_units": list(validation_units),
            "feature_columns_fit_inside_fold": True,
            "scaler_fit_inside_fold": True,
            "window_stride": CV_STRIDE,
            "pseudo_current_reference_rul_range": [0.0, 125.0],
        })

    fold_frame = pd.DataFrame(fold_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    summary: dict[str, Any] = {}
    for (model_name, evaluation_point), group in fold_frame.groupby(
        ["model", "evaluation_point"], sort=True
    ):
        oof = prediction_frame[
            (prediction_frame["model"] == model_name)
            & (prediction_frame["evaluation_point"] == evaluation_point)
        ]
        per_engine_errors = np.abs(oof["actual_rul"].to_numpy() - oof["predicted_rul"].to_numpy())
        summary.setdefault(str(model_name), {})[str(evaluation_point)] = {
            "fold_mae_mean": float(group["mae"].mean()),
            "fold_mae_std": float(group["mae"].std(ddof=1)),
            "fold_rmse_mean": float(group["rmse"].mean()),
            "fold_nasa_score_mean": float(group["nasa_score"].mean()),
            "out_of_fold_metrics": _metrics(
                oof["actual_rul"].to_numpy(), oof["predicted_rul"].to_numpy()
            ),
            "engine_bootstrap_mae_ci95": _bootstrap_mean_ci(per_engine_errors, seed=SEED),
            "n_validation_engines": int(oof["unit_id"].nunique()),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(output_dir / "grouped_cv_fd001_fold_metrics.csv", index=False)
    prediction_frame[prediction_frame["evaluation_point"] == "pseudo_current"].to_csv(
        output_dir / "grouped_cv_fd001_pseudocurrent_predictions.csv", index=False
    )
    prediction_frame[prediction_frame["evaluation_point"] == "failure_endpoint"].to_csv(
        output_dir / "grouped_cv_fd001_terminal_predictions.csv", index=False
    )
    pd.DataFrame([
        {
            "model": model,
            "evaluation_point": evaluation_point,
            **values["out_of_fold_metrics"],
            "engine_bootstrap_mae_ci95_lower": values["engine_bootstrap_mae_ci95"]["ci95_lower"],
            "engine_bootstrap_mae_ci95_upper": values["engine_bootstrap_mae_ci95"]["ci95_upper"],
            "fold_mae_mean": values["fold_mae_mean"],
            "fold_mae_std": values["fold_mae_std"],
        }
        for model, points in summary.items()
        for evaluation_point, values in points.items()
    ]).to_csv(output_dir / "grouped_cv_fd001_summary.csv", index=False)
    report = {
        "protocol_version": CV_PROTOCOL_VERSION,
        "subset": "FD001",
        "method": "5-fold GroupKFold over engine unit IDs",
        "evaluation_points": {
            "pseudo_current": "one window per validation engine selected at reference RUL targets spanning 0-125 cycles",
            "failure_endpoint": "last available window per validation engine; run-to-failure endpoint diagnostic (RUL near zero)",
        },
        "folds": fold_sets,
        "model_summary": summary,
        "xgboost_available": xgb_class is not None,
        "official_test_used": False,
        "window_stride": CV_STRIDE,
        "lstm_config": {"backend": "deterministic NumPy LSTM", "hidden_size": 32, "epochs": 8},
    }
    (output_dir / "grouped_cv_fd001.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def recover_grouped_cv(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Rebuild the CV report from saved out-of-fold predictions and fold scores."""
    fold_frame = pd.read_csv(output_dir / "grouped_cv_fd001_fold_metrics.csv")
    prediction_frame = pd.concat([
        pd.read_csv(output_dir / "grouped_cv_fd001_pseudocurrent_predictions.csv"),
        pd.read_csv(output_dir / "grouped_cv_fd001_terminal_predictions.csv"),
    ], ignore_index=True)
    units = set(int(unit) for unit in frame["unit_id"].unique())
    fold_sets = []
    for fold in sorted(int(value) for value in fold_frame["fold"].unique()):
        validation_units = sorted(
            int(unit) for unit in prediction_frame.loc[
                (prediction_frame["fold"] == fold)
                & (prediction_frame["model"] == "ridge")
                & (prediction_frame["evaluation_point"] == "pseudo_current"),
                "unit_id",
            ].unique()
        )
        fold_sets.append({
            "fold": fold,
            "train_units": sorted(units - set(validation_units)),
            "validation_units": validation_units,
            "feature_columns_fit_inside_fold": True,
            "scaler_fit_inside_fold": True,
            "window_stride": CV_STRIDE,
            "pseudo_current_reference_rul_range": [0.0, 125.0],
        })
    summary: dict[str, Any] = {}
    for (model_name, evaluation_point), group in fold_frame.groupby(
        ["model", "evaluation_point"], sort=True
    ):
        oof = prediction_frame[
            (prediction_frame["model"] == model_name)
            & (prediction_frame["evaluation_point"] == evaluation_point)
        ]
        errors = np.abs(oof["actual_rul"].to_numpy() - oof["predicted_rul"].to_numpy())
        summary.setdefault(str(model_name), {})[str(evaluation_point)] = {
            "fold_mae_mean": float(group["mae"].mean()),
            "fold_mae_std": float(group["mae"].std(ddof=1)),
            "fold_rmse_mean": float(group["rmse"].mean()),
            "fold_nasa_score_mean": float(group["nasa_score"].mean()),
            "out_of_fold_metrics": _metrics(
                oof["actual_rul"].to_numpy(), oof["predicted_rul"].to_numpy()
            ),
            "engine_bootstrap_mae_ci95": _bootstrap_mean_ci(errors, seed=SEED),
            "n_validation_engines": int(oof["unit_id"].nunique()),
        }
    report = {
        "protocol_version": CV_PROTOCOL_VERSION,
        "subset": "FD001",
        "method": "5-fold GroupKFold over engine unit IDs",
        "evaluation_points": {
            "pseudo_current": "one window per validation engine selected at reference RUL targets spanning 0-125 cycles",
            "failure_endpoint": "last available window per validation engine; run-to-failure endpoint diagnostic (RUL near zero)",
        },
        "folds": fold_sets,
        "model_summary": summary,
        "xgboost_available": "xgboost" in summary,
        "official_test_used": False,
        "window_stride": CV_STRIDE,
        "lstm_config": {"backend": "deterministic NumPy LSTM", "hidden_size": 32, "epochs": 8},
    }
    (output_dir / "grouped_cv_fd001.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def _fit_rf_ablation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    name: str,
    feature_columns: list[str],
    window_size: int,
    representation: str = "summary",
    condition_normalize: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if condition_normalize:
        normalizer = OperatingConditionNormalizer(seed=SEED).fit(train)
        train = normalizer.transform(train)
        validation = normalizer.transform(validation)
        feature_columns = [c for c in SENSOR_COLUMNS if c in feature_columns]
    dropped = constant_columns(train, feature_columns)
    selected = [c for c in feature_columns if c not in dropped]
    scaler = fit_feature_scaler(train, selected)
    train_scaled = transform_features(train, selected, scaler)
    validation_scaled = transform_features(validation, selected, scaler)
    train_windows, train_y, _ = make_windows(train_scaled, selected, window_size, "rul", CV_STRIDE)
    validation_windows, validation_y, metadata = make_windows(
        validation_scaled, selected, window_size, "rul", CV_STRIDE
    )
    if representation == "last":
        train_x = train_windows[:, -1, :]
        validation_x = validation_windows[:, -1, :]
    else:
        train_x = summarize_windows(train_windows)
        validation_x = summarize_windows(validation_windows)
    model = RandomForestRegressor(
        n_estimators=RF_TREES,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    predicted = np.maximum(model.predict(validation_x), 0.0)
    pseudo_current = _pseudo_current_indices(metadata, SEED)
    failure_endpoint = _terminal_indices(metadata)
    metrics = _metrics(validation_y[pseudo_current], predicted[pseudo_current])
    endpoint_metrics = _metrics(validation_y[failure_endpoint], predicted[failure_endpoint])
    row = {
        "ablation": name,
        "window_size": window_size,
        "representation": representation,
        "condition_normalized": condition_normalize,
        "features": selected,
        "dropped_constant_features": dropped,
        "evaluation_point": "one RUL-stratified pseudo-current window per validation engine",
        "failure_endpoint_metrics": endpoint_metrics,
        **metrics,
    }
    predictions = [
        {
            "ablation": name,
            "evaluation_point": "pseudo_current",
            "unit_id": int(metadata.loc[idx, "unit_id"]),
            "actual_rul": float(validation_y[idx]),
            "predicted_rul": float(predicted[idx]),
            "absolute_error": float(abs(validation_y[idx] - predicted[idx])),
        }
        for idx in pseudo_current
    ]
    endpoint_predictions = [
        {
            "ablation": name,
            "evaluation_point": "failure_endpoint",
            "unit_id": int(metadata.loc[idx, "unit_id"]),
            "actual_rul": float(validation_y[idx]),
            "predicted_rul": float(predicted[idx]),
            "absolute_error": float(abs(validation_y[idx] - predicted[idx])),
        }
        for idx in failure_endpoint
    ]
    row["pseudo_current_engine_mae_bootstrap_ci95"] = _bootstrap_mean_ci(
        np.array([r["absolute_error"] for r in predictions]), seed=SEED
    )
    row["failure_endpoint_mae_bootstrap_ci95"] = _bootstrap_mean_ci(
        np.array([r["absolute_error"] for r in endpoint_predictions]), seed=SEED
    )
    return row, predictions, endpoint_predictions


def run_ablations(fd004: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Run holdout ablations on FD004; fit every transform on fit engines only."""
    split = split_engine_units(fd004, validation_fraction=0.2, seed=SEED)
    train = fd004[fd004["unit_id"].isin(split.train_units)].copy()
    validation = fd004[fd004["unit_id"].isin(split.validation_units)].copy()
    verify_no_leakage(split.train_units, split.validation_units)
    results: list[dict[str, Any]] = []
    pseudo_current_predictions: list[dict[str, Any]] = []
    endpoint_predictions: list[dict[str, Any]] = []
    experiments = [
        ("raw_sensor_last_window", SENSOR_COLUMNS, 30, "last", False),
        ("sensor_engineered_statistics", SENSOR_COLUMNS, 30, "summary", False),
        ("all_features_engineered_statistics", FEATURE_COLUMNS, 30, "summary", False),
        ("window_size_20", FEATURE_COLUMNS, 20, "summary", False),
        ("window_size_40", FEATURE_COLUMNS, 40, "summary", False),
        ("operating_condition_normalized_sensors", SENSOR_COLUMNS, 30, "summary", True),
    ]
    # Include the 30-cycle reference row explicitly for the window-size study.
    experiments.append(("window_size_30_reference", FEATURE_COLUMNS, 30, "summary", False))
    for name, columns, window_size, representation, condition_normalize in experiments:
        row, pred_rows, endpoint_rows = _fit_rf_ablation(
            train,
            validation,
            name,
            list(columns),
            window_size,
            representation,
            condition_normalize,
        )
        results.append(row)
        pseudo_current_predictions.extend(pred_rows)
        endpoint_predictions.extend(endpoint_rows)

    # Model comparison uses the same fit/validation engines, windows and features.
    base_columns = [c for c in FEATURE_COLUMNS if c not in constant_columns(train, FEATURE_COLUMNS)]
    scaler = fit_feature_scaler(train, base_columns)
    train_scaled = transform_features(train, base_columns, scaler)
    validation_scaled = transform_features(validation, base_columns, scaler)
    train_windows, train_y, _ = make_windows(train_scaled, base_columns, 30, "rul", CV_STRIDE)
    val_windows, val_y, val_meta = make_windows(validation_scaled, base_columns, 30, "rul", CV_STRIDE)
    val_summary = summarize_windows(val_windows)
    pseudo_current = _pseudo_current_indices(val_meta, SEED)
    failure_endpoint = _terminal_indices(val_meta)
    rf = RandomForestRegressor(
        n_estimators=RF_TREES,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=SEED,
        n_jobs=-1,
    ).fit(summarize_windows(train_windows), train_y)
    rf_pred = np.maximum(rf.predict(val_summary), 0.0)
    lstm = NumpyLSTMRULModel(train_windows.shape[2], hidden_size=32, seed=SEED)
    lstm.fit(train_windows, train_y, val_windows, val_y, max_epochs=8, batch_size=128, lr=1e-3)
    lstm_pred = lstm.predict(val_windows)
    for model_name, pred in (("random_forest_summary", rf_pred), ("numpy_lstm_sequence", lstm_pred)):
        pseudo_metrics = _metrics(val_y[pseudo_current], pred[pseudo_current])
        endpoint_metrics = _metrics(val_y[failure_endpoint], pred[failure_endpoint])
        row = {
            "ablation": f"model_comparison_{model_name}",
            "window_size": 30,
            "representation": "summary" if model_name.startswith("random") else "sequence",
            "condition_normalized": False,
            "evaluation_point": "one RUL-stratified pseudo-current window per validation engine",
            "failure_endpoint_metrics": endpoint_metrics,
            **pseudo_metrics,
        }
        per_engine = np.abs(val_y[pseudo_current] - pred[pseudo_current])
        endpoint_errors = np.abs(val_y[failure_endpoint] - pred[failure_endpoint])
        row["pseudo_current_engine_mae_bootstrap_ci95"] = _bootstrap_mean_ci(per_engine, seed=SEED)
        row["failure_endpoint_mae_bootstrap_ci95"] = _bootstrap_mean_ci(endpoint_errors, seed=SEED)
        results.append(row)
        for pos, idx in enumerate(pseudo_current):
            pseudo_current_predictions.append({
                "ablation": row["ablation"],
                "evaluation_point": "pseudo_current",
                "unit_id": int(val_meta.loc[idx, "unit_id"]),
                "actual_rul": float(val_y[idx]),
                "predicted_rul": float(pred[idx]),
                "absolute_error": float(per_engine[pos]),
            })
        for pos, idx in enumerate(failure_endpoint):
            endpoint_predictions.append({
                "ablation": row["ablation"],
                "evaluation_point": "failure_endpoint",
                "unit_id": int(val_meta.loc[idx, "unit_id"]),
                "actual_rul": float(val_y[idx]),
                "predicted_rul": float(pred[idx]),
                "absolute_error": float(endpoint_errors[pos]),
            })

    result_frame = pd.DataFrame(results)
    pseudo_current_frame = pd.DataFrame(pseudo_current_predictions)
    endpoint_frame = pd.DataFrame(endpoint_predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frame.drop(columns=[
        "features", "dropped_constant_features", "failure_endpoint_metrics",
        "pseudo_current_engine_mae_bootstrap_ci95", "failure_endpoint_mae_bootstrap_ci95",
    ], errors="ignore").to_csv(
        output_dir / "fd004_ablation_summary.csv", index=False
    )
    pseudo_current_frame.to_csv(output_dir / "fd004_ablation_pseudocurrent_predictions.csv", index=False)
    endpoint_frame.to_csv(output_dir / "fd004_ablation_terminal_predictions.csv", index=False)
    return {
        "subset": "FD004",
        "protocol": (
            "one held-out engine split; metrics use one RUL-stratified pseudo-current window per validation engine; "
            "failure-endpoint metrics are retained separately; official test data not loaded"
        ),
        "fit_units": list(split.train_units),
        "validation_units": list(split.validation_units),
        "results": results,
        "operating_condition_normalization": (
            "KMeans operating-condition clusters and per-cluster sensor moments fitted on fit-engine rows only."
        ),
    }


def main() -> None:
    started = time.time()
    data_dir = ROOT / "data" / "raw" / "cmapss"
    output_dir = ROOT / "reports" / "generated"
    fd001 = add_train_rul(load_telemetry(data_dir, "FD001", "train"))
    fd004 = add_train_rul(load_telemetry(data_dir, "FD004", "train"))
    cv_fold_path = output_dir / "grouped_cv_fd001_fold_metrics.csv"
    cv_pseudo_path = output_dir / "grouped_cv_fd001_pseudocurrent_predictions.csv"
    cv_endpoint_path = output_dir / "grouped_cv_fd001_terminal_predictions.csv"
    cv_report_path = output_dir / "grouped_cv_fd001.json"
    if all(path.exists() for path in (cv_fold_path, cv_pseudo_path, cv_endpoint_path, cv_report_path)):
        try:
            cached_cv = json.loads(cv_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_cv = {}
        if cached_cv.get("protocol_version") == CV_PROTOCOL_VERSION:
            cv = cached_cv
        else:
            cv = run_grouped_cv(fd001, output_dir)
    else:
        cv = run_grouped_cv(fd001, output_dir)
    ablations = run_ablations(fd004, output_dir)
    report = {
        "experiment": "grouped engine cross-validation and feature/window/model ablations",
        "seed": SEED,
        "environment": {
            "python": platform.python_version(),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit_learn": _version("scikit-learn"),
            "xgboost": _version("xgboost"),
        },
        "cross_validation": cv,
        "ablations": ablations,
        "monitoring_comparison": {
            "report": "fd001_lstm_monitoring.json",
            "note": "LSTM-AE and PCA comparison is generated in one run using the same fit, validation, and official test engine groups.",
        },
        "runtime_seconds": round(time.time() - started, 1),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "research_evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("Grouped CV models:")
    for name, values in cv["model_summary"].items():
        pseudo = values["pseudo_current"]
        ci = pseudo["engine_bootstrap_mae_ci95"]
        print(f"  {name}: pseudo-current MAE={pseudo['out_of_fold_metrics']['mae']:.3f} "
              f"(engine bootstrap 95% CI {ci['ci95_lower']:.3f}-{ci['ci95_upper']:.3f})")
    print(f"FD004 ablation rows: {len(ablations['results'])}")
    print(f"saved report: {output_dir / 'research_evaluation.json'}")


if __name__ == "__main__":
    main()
