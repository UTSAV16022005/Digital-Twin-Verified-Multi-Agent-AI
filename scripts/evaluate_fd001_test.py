"""Evaluate the frozen FD001 RUL baseline on the official test engines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
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
    summarize_windows,
    transform_features,
)


def nasa_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Compute the asymmetric C-MAPSS scoring function."""

    difference = predictions - targets
    penalties = np.where(
        difference < 0,
        np.exp(-difference / 13.0) - 1.0,
        np.exp(difference / 10.0) - 1.0,
    )
    return float(penalties.sum())


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    predictions = np.maximum(predictions, 0.0)
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
        "r2": float(r2_score(targets, predictions)),
        "nasa_score": nasa_score(targets, predictions),
    }


def main() -> None:
    data_dir = Path("data/raw/cmapss")
    train = add_train_rul(load_telemetry(data_dir, "FD001", "train"))
    test = load_telemetry(data_dir, "FD001", "test")
    test = add_test_rul(test, load_test_rul(data_dir, "FD001"))

    dropped = constant_columns(train, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]

    # The validation-selected model configuration is now frozen. We refit on all
    # available labeled training engines before touching the official test set.
    scaler = fit_feature_scaler(train, features)
    train = transform_features(train, features, scaler)
    test = transform_features(test, features, scaler)

    train_windows, train_targets, _ = make_windows(train, features, 30)
    test_windows, test_targets, test_metadata = make_windows(test, features, 30)
    train_summary = summarize_windows(train_windows)
    test_summary = summarize_windows(test_windows)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train_summary, train_targets)
    predictions = np.maximum(model.predict(test_summary), 0.0)

    test_metadata = test_metadata.copy()
    test_metadata["prediction"] = predictions
    terminal = test_metadata.groupby("unit_id", sort=True).tail(1).copy()

    window_metrics = regression_metrics(test_targets, predictions)
    terminal_metrics = regression_metrics(
        terminal["rul"].to_numpy(), terminal["prediction"].to_numpy()
    )

    report = {
        "subset": "FD001",
        "evaluation_split": "official_test",
        "window_size": 30,
        "seed": 42,
        "training_engines": int(train["unit_id"].nunique()),
        "test_engines": int(test["unit_id"].nunique()),
        "features_used": features,
        "features_dropped_as_constant": dropped,
        "model": "HistGradientBoostingRegressor",
        "window_metrics": window_metrics,
        "terminal_engine_metrics": terminal_metrics,
    }

    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fd001_rul_test.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    terminal.to_csv(output_dir / "fd001_rul_test_predictions.csv", index=False)

    print("FD001 official test evaluation")
    print("==============================")
    print("model: HistGradientBoostingRegressor")
    print("training: all 100 FD001 training engines")
    print("test: official 100 FD001 test engines")
    print(
        "window metrics: "
        f"MAE={window_metrics['mae']:.3f}, "
        f"RMSE={window_metrics['rmse']:.3f}, "
        f"R2={window_metrics['r2']:.3f}, "
        f"NASA score={window_metrics['nasa_score']:.3f}"
    )
    print(
        "terminal-engine metrics: "
        f"MAE={terminal_metrics['mae']:.3f}, "
        f"RMSE={terminal_metrics['rmse']:.3f}, "
        f"R2={terminal_metrics['r2']:.3f}, "
        f"NASA score={terminal_metrics['nasa_score']:.3f}"
    )
    print(f"saved report: {output_dir / 'fd001_rul_test.json'}")
    print(f"saved predictions: {output_dir / 'fd001_rul_test_predictions.csv'}")


if __name__ == "__main__":
    main()

