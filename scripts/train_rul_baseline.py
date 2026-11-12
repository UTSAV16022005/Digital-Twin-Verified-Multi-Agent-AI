"""Train and evaluate a leakage-safe classical RUL baseline on FD001."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    predictions = np.maximum(predictions, 0.0)
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
        "r2": float(r2_score(targets, predictions)),
    }


def main() -> None:
    data_dir = Path("data/raw/cmapss")
    train = add_train_rul(load_telemetry(data_dir, "FD001", "train"))
    split = split_engine_units(train, validation_fraction=0.2, seed=42)

    train_frame = train[train["unit_id"].isin(split.train_units)].copy()
    validation_frame = train[train["unit_id"].isin(split.validation_units)].copy()
    dropped = constant_columns(train_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]

    # The scaler sees training engines only. Validation values remain unseen here.
    scaler = fit_feature_scaler(train_frame, features)
    train_frame = transform_features(train_frame, features, scaler)
    validation_frame = transform_features(validation_frame, features, scaler)

    train_windows, train_targets, _ = make_windows(train_frame, features, 30)
    validation_windows, validation_targets, validation_metadata = make_windows(
        validation_frame, features, 30
    )
    train_summary = summarize_windows(train_windows)
    validation_summary = summarize_windows(validation_windows)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train_summary, train_targets)
    predictions = model.predict(validation_summary)

    window_metrics = regression_metrics(validation_targets, predictions)
    validation_metadata = validation_metadata.copy()
    validation_metadata["prediction"] = np.maximum(predictions, 0.0)
    terminal = validation_metadata.groupby("unit_id", sort=True).tail(1)
    terminal_metrics = regression_metrics(
        terminal["rul"].to_numpy(), terminal["prediction"].to_numpy()
    )

    report = {
        "subset": "FD001",
        "window_size": 30,
        "seed": 42,
        "train_engines": len(split.train_units),
        "validation_engines": len(split.validation_units),
        "features_used": features,
        "features_dropped_as_constant": dropped,
        "model": "HistGradientBoostingRegressor",
        "window_metrics": window_metrics,
        "terminal_engine_metrics": terminal_metrics,
    }

    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fd001_rul_baseline.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("FD001 RUL baseline")
    print("=================")
    print("model: HistGradientBoostingRegressor")
    print("normalization: StandardScaler fitted on training engines only")
    print(f"training windows: {len(train_targets):,}")
    print(f"validation windows: {len(validation_targets):,}")
    print(
        "window metrics: "
        f"MAE={window_metrics['mae']:.3f}, "
        f"RMSE={window_metrics['rmse']:.3f}, "
        f"R2={window_metrics['r2']:.3f}"
    )
    print(
        "terminal-engine metrics: "
        f"MAE={terminal_metrics['mae']:.3f}, "
        f"RMSE={terminal_metrics['rmse']:.3f}, "
        f"R2={terminal_metrics['r2']:.3f}"
    )
    print(f"saved report: {output_dir / 'fd001_rul_baseline.json'}")


if __name__ == "__main__":
    main()

