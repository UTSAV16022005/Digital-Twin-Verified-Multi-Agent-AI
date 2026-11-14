"""Compare integrated prognostics-agent model types with engine-group CV.

Each outer fold separates complete engine trajectories into fit, calibration,
and evaluation groups. The LSTM uses a further engine-level early-stop split
inside its fit engines. Only C-MAPSS training trajectories are loaded.
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

from scripts.evaluate_fd004_cycle_conditioned_cv import (
    FOLDS,
    MILESTONES,
    SEED,
    _paired_engine_comparison,
    _predict_milestones,
    _summarize,
)
from src.cmapss_data import (
    FEATURE_COLUMNS,
    add_train_rul,
    constant_columns,
    load_telemetry,
    split_engine_units,
    verify_no_leakage,
)
from src.prognostics import PrognosticsAgent, PrognosticsConfig


MODEL_NAMES = (
    "rf30_global",
    "xgboost30_global",
    "rf30_cycle_feature",
    "numpy_lstm_agent",
    "pytorch_cnn_agent",
    "pytorch_transformer_agent",
)
RF_TREES = 80


def main(subset: str = "FD001") -> None:
    subset = subset.upper()
    if subset not in {"FD001", "FD004"}:
        raise ValueError("subset must be FD001 or FD004")
    started = time.perf_counter()
    data_dir = ROOT / "data" / "raw" / "cmapss"
    full_frame = add_train_rul(load_telemetry(data_dir, subset, "train"))
    engine_ids = np.asarray(sorted(full_frame["unit_id"].unique()), dtype=int)
    outer_splitter = GroupKFold(n_splits=FOLDS)
    all_rows: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []

    for fold, (pool_indices, evaluation_indices) in enumerate(
        outer_splitter.split(engine_ids, groups=engine_ids), start=1
    ):
        pool_units = tuple(int(value) for value in engine_ids[pool_indices])
        evaluation_units = tuple(
            int(value) for value in engine_ids[evaluation_indices]
        )
        pool_frame = full_frame[full_frame["unit_id"].isin(pool_units)].copy()
        split = split_engine_units(
            pool_frame, validation_fraction=0.25, seed=SEED + fold
        )
        outer_fit_units, calibration_units = split.train_units, split.validation_units
        outer_fit_frame = full_frame[
            full_frame["unit_id"].isin(outer_fit_units)
        ].copy()
        early_stop_split = split_engine_units(
            outer_fit_frame, validation_fraction=0.15, seed=SEED + 100 + fold
        )
        fit_units = early_stop_split.train_units
        early_stop_units = early_stop_split.validation_units
        verify_no_leakage(fit_units, early_stop_units)
        verify_no_leakage(fit_units, calibration_units)
        verify_no_leakage(early_stop_units, calibration_units)
        verify_no_leakage(fit_units, evaluation_units)
        verify_no_leakage(early_stop_units, evaluation_units)
        verify_no_leakage(calibration_units, evaluation_units)
        fit_frame = full_frame[full_frame["unit_id"].isin(fit_units)].copy()
        early_stop_frame = full_frame[
            full_frame["unit_id"].isin(early_stop_units)
        ].copy()
        calibration_frame = full_frame[
            full_frame["unit_id"].isin(calibration_units)
        ].copy()
        evaluation_frame = full_frame[
            full_frame["unit_id"].isin(evaluation_units)
        ].copy()
        dropped = constant_columns(fit_frame, FEATURE_COLUMNS)
        features = [column for column in FEATURE_COLUMNS if column not in dropped]
        seed = SEED + fold

        configs = {
            "rf30_global": PrognosticsConfig(
                window_size=30,
                n_estimators=RF_TREES,
                min_samples_leaf=2,
                max_features=0.70,
                random_state=seed,
                training_window_stride=5,
            ),
            "rf30_cycle_feature": PrognosticsConfig(
                window_size=30,
                n_estimators=RF_TREES,
                min_samples_leaf=2,
                max_features=0.70,
                random_state=seed,
                include_cycle_feature=True,
                training_window_stride=5,
            ),
            "xgboost30_global": PrognosticsConfig(
                model_type="xgboost",
                window_size=30,
                random_state=seed,
                xgb_n_estimators=250,
                xgb_max_depth=5,
                xgb_learning_rate=0.04,
                xgb_subsample=0.9,
                xgb_colsample_bytree=0.8,
                training_window_stride=5,
            ),
            "numpy_lstm_agent": PrognosticsConfig(
                model_type="lstm",
                window_size=30,
                random_state=seed,
                lstm_hidden_size=24,
                lstm_epochs=8,
                lstm_batch_size=128,
                lstm_training_stride=5,
                lstm_validation_fraction=0.15,
                lstm_early_stopping_patience=2,
            ),
            "pytorch_cnn_agent": PrognosticsConfig(
                model_type="pytorch_cnn",
                window_size=30,
                random_state=seed,
                torch_hidden_size=32,
                torch_epochs=8,
                torch_batch_size=128,
                torch_learning_rate=1e-3,
                torch_training_stride=5,
                torch_validation_fraction=0.0,
                torch_early_stopping_patience=2,
            ),
            "pytorch_transformer_agent": PrognosticsConfig(
                model_type="pytorch_transformer",
                window_size=30,
                random_state=seed,
                torch_hidden_size=32,
                torch_epochs=8,
                torch_batch_size=128,
                torch_learning_rate=1e-3,
                torch_training_stride=5,
                torch_validation_fraction=0.0,
                torch_early_stopping_patience=2,
                torch_attention_heads=4,
                torch_transformer_layers=2,
            ),
        }
        agents: dict[str, PrognosticsAgent] = {}
        for model_name, config in configs.items():
            is_sequence = model_name in {
                "numpy_lstm_agent",
                "pytorch_cnn_agent",
                "pytorch_transformer_agent",
            }
            agent = PrognosticsAgent.fit(
                fit_frame,
                features,
                config,
                validation_frame=early_stop_frame if is_sequence else None,
            )
            agent.calibrate_uncertainty(calibration_frame)
            agents[model_name] = agent

        truth = evaluation_frame[["unit_id", "cycle", "rul"]].copy()
        telemetry = evaluation_frame.drop(columns=["rul"])
        fold_rows = [
            _predict_milestones(
                agents[name],
                telemetry,
                truth,
                fold,
                name,
                use_milestone_margin=False,
            )
            for name in MODEL_NAMES
        ]
        fold_frame = pd.concat(fold_rows, ignore_index=True)
        all_rows.append(fold_frame)
        lstm_model = agents["numpy_lstm_agent"].model
        folds.append(
            {
                "fold": fold,
                "fit_engine_count": len(fit_units),
                "early_stop_engine_count": len(early_stop_units),
                "calibration_engine_count": len(calibration_units),
                "evaluation_engine_count": len(evaluation_units),
                "fit_engine_ids": list(fit_units),
                "early_stop_engine_ids": list(early_stop_units),
                "calibration_engine_ids": list(calibration_units),
                "evaluation_engine_ids": list(evaluation_units),
                "early_stop_fraction_within_outer_fit_pool": 0.15,
                "lstm_epochs_trained": getattr(lstm_model, "epochs_trained_", None),
                "lstm_best_epoch": getattr(lstm_model, "best_epoch_", None),
                "lstm_best_early_stop_loss": getattr(
                    lstm_model, "best_validation_loss_", None
                ),
                "calibration_engines_by_model": {
                    name: int(agent.n_calibration_engines_)
                    for name, agent in agents.items()
                },
                "sequence_models": {
                    name: {
                        "best_epoch": getattr(agents[name].model, "best_epoch_", None),
                        "epochs_trained": getattr(
                            agents[name].model, "epochs_trained_", None
                        ),
                        "best_early_stop_loss": getattr(
                            agents[name].model, "best_validation_loss_", None
                        ),
                    }
                    for name in (
                        "numpy_lstm_agent",
                        "pytorch_cnn_agent",
                        "pytorch_transformer_agent",
                    )
                },
            }
        )
        print(
            f"{subset}: completed fold {fold}/{FOLDS}; "
            f"fit/calibration/evaluation={len(fit_units)}/"
            f"{len(calibration_units)}/{len(evaluation_units)}",
            flush=True,
        )

    predictions = pd.concat(all_rows, ignore_index=True)
    overall = {
        name: _summarize(
            predictions[predictions["model"] == name], SEED + index * 10
        )
        for index, name in enumerate(MODEL_NAMES)
    }
    comparisons = {
        name: _paired_engine_comparison(
            predictions,
            "rf30_global",
            name,
            "absolute_error",
            SEED + 100 + index,
            "negative favors the candidate",
        )
        for index, name in enumerate(MODEL_NAMES[1:], start=1)
    }
    per_cycle: dict[str, Any] = {}
    for cycle in MILESTONES:
        cycle_rows = predictions[predictions["cycle"] == cycle]
        per_cycle[str(cycle)] = {
            name: _summarize(
                cycle_rows[cycle_rows["model"] == name],
                SEED + cycle + index,
            )
            for index, name in enumerate(MODEL_NAMES)
        }

    report = {
        "experiment": f"{subset} integrated PrognosticsAgent model comparison",
        "source": f"official C-MAPSS {subset} training trajectories only",
        "official_test_files_loaded": False,
        "protocol": {
            "outer_evaluation": "5-fold GroupKFold by engine",
            "inner_fit_calibration_split": "75% fit and 25% calibration within outer training engines",
            "approximate_total_proportions": {
                "model_fit": 0.51,
                "early_stop": 0.09,
                "calibration": 0.20,
                "evaluation": 0.20,
            },
            "milestones_cycles": list(MILESTONES),
            "window_size": 30,
            "random_forest_trees": RF_TREES,
            "common_training_window_stride": 5,
            "xgboost": {
                "implementation": "XGBRegressor through PrognosticsAgent.fit(model_type='xgboost')",
                "estimators": 250,
                "max_depth": 5,
                "learning_rate": 0.04,
                "subsample": 0.9,
                "column_subsample": 0.8,
            },
            "lstm": {
                "implementation": "NumpyLSTMRULModel through PrognosticsAgent.fit(model_type='lstm')",
                "hidden_size": 24,
                "maximum_epochs": 8,
                "early_stopping": "engine-disjoint early-stop group, excluded from model fitting",
                "training_window_stride": 5,
            },
            "pytorch_cnn": {
                "architecture": "two Conv1d layers with mean/max temporal pooling",
                "hidden_channels": 32,
                "maximum_epochs": 8,
                "early_stopping": "engine-disjoint early-stop group, excluded from model fitting",
                "training_window_stride": 5,
            },
            "pytorch_transformer": {
                "architecture": "two-layer encoder-only Transformer, d_model=32, 4 heads",
                "maximum_epochs": 8,
                "early_stopping": "engine-disjoint early-stop group, excluded from model fitting",
                "training_window_stride": 5,
            },
            "telemetry_label_access": "Evaluation RUL is joined only after telemetry-only predictions.",
        },
        "overall_metrics": overall,
        "paired_engine_comparisons": comparisons,
        "metrics_by_cycle": per_cycle,
        "folds": folds,
        "limitations": [
            "This compares RUL prediction and calibrated lower bounds, not full Digital Twin action outcomes.",
            "Every candidate is trained on the same engine group and stride-5 windows; neural models use a disjoint early-stop group from the remaining outer-fit pool.",
            "Models use fixed documented configurations rather than an exhaustive per-model hyperparameter search.",
            "C-MAPSS has no observed recovery-action outcomes.",
            "Official test files were not loaded in this model-selection run.",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    out_dir = ROOT / "reports" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = subset.lower()
    json_path = out_dir / f"{suffix}_prognostics_agent_models_cv.json"
    csv_path = out_dir / f"{suffix}_prognostics_agent_models_cv_predictions.csv"
    predictions.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"{subset} integrated PrognosticsAgent comparison")
    for name, metrics in overall.items():
        print(
            f"{name}: MAE={metrics['mae_cycles']:.2f}, "
            f"coverage={metrics['planning_lower_bound_empirical_coverage']:.3f}, "
            f"engine-bootstrap-CI="
            f"{metrics['engine_macro_mae_engine_cluster_bootstrap_95_ci']}"
        )
    print(f"paired engine comparisons: {comparisons}")
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"OOF predictions: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD004"), default="FD001")
    main(parser.parse_args().subset)
