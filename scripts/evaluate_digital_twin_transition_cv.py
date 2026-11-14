"""Five-fold held-out-engine evaluation of the Twin's nominal transition model.

Compares one-step sensor predictions with persistence (next state equals current
state). This validates only the learned no-action transition, not the Twin's
counterfactual action modifiers.
"""

from __future__ import annotations

import argparse
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
from sklearn.model_selection import GroupKFold

from src.cmapss_data import (
    FEATURE_COLUMNS,
    constant_columns,
    fit_feature_scaler,
    load_telemetry,
    transform_features,
    verify_no_leakage,
)
from src.digital_twin import DataDrivenTransitionModel


SEED = 42
FOLDS = 5
BOOTSTRAP_REPLICATES = 20_000


def _bootstrap(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    draws = values[indices].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _evaluate_subset(subset: str) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = load_telemetry(ROOT / "data" / "raw" / "cmapss", subset, "train")
    unit_ids = np.asarray(sorted(frame["unit_id"].unique()), dtype=int)
    splitter = GroupKFold(n_splits=FOLDS)
    output_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    all_sensor_names: set[str] = set()

    for fold, (fit_idx, test_idx) in enumerate(
        splitter.split(unit_ids, groups=unit_ids), start=1
    ):
        fit_units = tuple(int(value) for value in unit_ids[fit_idx])
        test_units = tuple(int(value) for value in unit_ids[test_idx])
        verify_no_leakage(fit_units, test_units)
        fit_frame = frame[frame["unit_id"].isin(fit_units)].copy()
        test_frame = frame[frame["unit_id"].isin(test_units)].copy()
        dropped = constant_columns(fit_frame, FEATURE_COLUMNS)
        features = [name for name in FEATURE_COLUMNS if name not in dropped]
        sensors = [name for name in features if name.startswith("sensor_")]
        if not sensors:
            raise ValueError("No sensor columns remain after fold-specific filtering")
        all_sensor_names.update(sensors)

        scaler = fit_feature_scaler(fit_frame, features)
        fit_scaled = transform_features(fit_frame, features, scaler)
        test_scaled = transform_features(test_frame, features, scaler)
        transition = DataDrivenTransitionModel(features, alpha=1.0).fit(fit_scaled)
        fold_predictions = 0
        for unit_id, engine in test_scaled.groupby("unit_id", sort=True):
            engine = engine.sort_values("cycle")
            values = engine[features].to_numpy(dtype=np.float64)
            if len(values) < 2:
                continue
            predicted = transition.predict_next_batch(values[:-1])
            actual = values[1:]
            persistence = values[:-1]
            for feature_index, feature_name in enumerate(features):
                if not feature_name.startswith("sensor_"):
                    continue
                for time_index in range(len(actual)):
                    output_rows.append(
                        {
                            "subset": subset,
                            "fold": fold,
                            "unit_id": int(unit_id),
                            "cycle": int(engine["cycle"].to_numpy()[time_index + 1]),
                            "feature": feature_name,
                            "actual_next": float(actual[time_index, feature_index]),
                            "predicted_next": float(predicted[time_index, feature_index]),
                            "persistence_next": float(persistence[time_index, feature_index]),
                        }
                    )
            fold_predictions += len(actual)
        fold_reports.append(
            {
                "fold": fold,
                "fit_engine_count": len(fit_units),
                "evaluation_engine_count": len(test_units),
                "fit_engine_ids": list(fit_units),
                "evaluation_engine_ids": list(test_units),
                "transition_train_residual_std_by_feature": {
                    name: float(value)
                    for name, value in zip(features, transition.residual_std)
                },
                "heldout_transition_count": fold_predictions,
            }
        )
        print(
            f"{subset}: transition fold {fold}/{FOLDS} "
            f"({len(fit_units)} fit engines, {len(test_units)} evaluation engines)"
        )

    rows = pd.DataFrame(output_rows)
    if rows.empty:
        raise RuntimeError("No held-out one-step transitions were scored")
    rows["model_abs_error"] = (rows["predicted_next"] - rows["actual_next"]).abs()
    rows["persistence_abs_error"] = (rows["persistence_next"] - rows["actual_next"]).abs()
    rows["model_sq_error"] = (rows["predicted_next"] - rows["actual_next"]) ** 2
    rows["persistence_sq_error"] = (rows["persistence_next"] - rows["actual_next"]) ** 2
    per_engine = rows.groupby("unit_id", sort=True).agg(
        model_rmse=("model_sq_error", lambda x: float(np.sqrt(x.mean()))),
        persistence_rmse=("persistence_sq_error", lambda x: float(np.sqrt(x.mean()))),
        model_mae=("model_abs_error", "mean"),
        persistence_mae=("persistence_abs_error", "mean"),
    )
    paired_rmse_gain = (
        per_engine["persistence_rmse"] - per_engine["model_rmse"]
    ).to_numpy(dtype=float)
    overall = {
        "sensor_transition_rows": int(len(rows)),
        "heldout_engine_count": int(rows["unit_id"].nunique()),
        "data_driven_rmse_normalized_units": float(np.sqrt(rows["model_sq_error"].mean())),
        "persistence_rmse_normalized_units": float(
            np.sqrt(rows["persistence_sq_error"].mean())
        ),
        "data_driven_mae_normalized_units": float(rows["model_abs_error"].mean()),
        "persistence_mae_normalized_units": float(
            rows["persistence_abs_error"].mean()
        ),
        "engine_macro_data_driven_rmse": float(per_engine["model_rmse"].mean()),
        "engine_macro_persistence_rmse": float(per_engine["persistence_rmse"].mean()),
        "paired_engine_persistence_minus_model_rmse_gain": float(
            paired_rmse_gain.mean()
        ),
        "paired_engine_cluster_bootstrap_95_ci_rmse_gain": _bootstrap(
            paired_rmse_gain, SEED + 1
        ),
    }
    per_sensor: dict[str, Any] = {}
    for feature in sorted(all_sensor_names):
        sensor = rows[rows["feature"] == feature]
        per_sensor[feature] = {
            "transition_count": int(len(sensor)),
            "data_driven_rmse": float(np.sqrt(sensor["model_sq_error"].mean())),
            "persistence_rmse": float(
                np.sqrt(sensor["persistence_sq_error"].mean())
            ),
            "data_driven_mae": float(sensor["model_abs_error"].mean()),
            "persistence_mae": float(sensor["persistence_abs_error"].mean()),
        }
    report = {
        "subset": subset,
        "experiment": "held-out-engine one-step nominal Digital Twin transition evaluation",
        "official_test_files_loaded": False,
        "protocol": {
            "folds": "5-fold GroupKFold by engine",
            "preprocessing": "Scaler and transition model fit on outer training engines only.",
            "target": "Next-cycle standardized sensor values from the observed no-action trajectory.",
            "baseline": "Persistence: next sensor value equals current sensor value.",
            "action_counterfactuals_evaluated": False,
        },
        "overall": overall,
        "metrics_by_sensor": per_sensor,
        "folds": fold_reports,
        "limitations": [
            "This evaluates only one-step observational dynamics on held-out engines.",
            "It does not validate action multipliers, recovery effects, or long-horizon rollouts.",
            "These results are model-prediction errors, not operational safety evidence.",
        ],
    }
    return report, rows


def main(subset: str) -> None:
    subset = subset.upper()
    if subset not in {"FD001", "FD004"}:
        raise ValueError("subset must be FD001 or FD004")
    started = time.perf_counter()
    report, rows = _evaluate_subset(subset)
    report["runtime_seconds"] = round(time.perf_counter() - started, 1)
    out_dir = ROOT / "reports" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = subset.lower()
    json_path = out_dir / f"{suffix}_digital_twin_transition_cv.json"
    csv_path = out_dir / f"{suffix}_digital_twin_transition_predictions.csv"
    rows.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"{subset} nominal transition RMSE: {report['overall']['data_driven_rmse_normalized_units']:.4f}")
    print(f"persistence RMSE: {report['overall']['persistence_rmse_normalized_units']:.4f}")
    print(f"paired engine RMSE gain CI: {report['overall']['paired_engine_cluster_bootstrap_95_ci_rmse_gain']}")
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"Transition rows: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD004"), default="FD004")
    main(parser.parse_args().subset)
