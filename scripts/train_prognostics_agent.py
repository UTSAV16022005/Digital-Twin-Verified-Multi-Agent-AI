"""Train and evaluate the formal FD001 Prognostics Agent and comparative benchmark."""

from __future__ import annotations

import json
import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.cmapss_data import (
    FEATURE_COLUMNS,
    add_test_rul,
    add_train_rul,
    constant_columns,
    load_test_rul,
    load_telemetry,
    split_engine_units,
)
from src.prognostics import (
    PrognosticsAgent,
    PrognosticsConfig,
    benchmark_all_prognostics,
    compute_rul_metrics,
    nasa_score,
    select_rul_stratified_rows,
)


def metrics(predictions: pd.DataFrame) -> dict[str, float]:
    actual = predictions["rul"].to_numpy(dtype=float)
    predicted = predictions["predicted_rul"].to_numpy(dtype=float)
    reg = compute_rul_metrics(actual, predicted)
    reg["uncertainty_mean"] = float(predictions["rul_uncertainty"].mean()) if "rul_uncertainty" in predictions else 0.0
    reg["interval_coverage_95"] = float(
        ((actual >= predictions["lower_rul"]) & (actual <= predictions["upper_rul"])).mean()
    ) if "lower_rul" in predictions else 0.0
    if "calibrated_lower_95" in predictions:
        coverage = float(
            ((actual >= predictions["calibrated_lower_95"]) & (actual <= predictions["calibrated_upper_95"])).mean()
        )
        reg["conformal_coverage_95"] = coverage
        reg["conformal_mean_interval_width"] = float(
            (predictions["calibrated_upper_95"] - predictions["calibrated_lower_95"]).mean()
        )
        reg["conformal_coverage_error"] = abs(coverage - 0.95)
    return reg


def per_engine_error_analysis(predictions: pd.DataFrame) -> list[dict]:
    records = []
    for unit_id, g in predictions.groupby("unit_id", sort=True):
        actual = g["rul"].to_numpy(dtype=float)
        predicted = g["predicted_rul"].to_numpy(dtype=float)
        terminal_actual = float(actual[-1])
        terminal_pred = float(predicted[-1])
        records.append({
            "unit_id": int(unit_id),
            "n_windows": int(len(g)),
            "window_mae": float(mean_absolute_error(actual, predicted)),
            "terminal_actual_rul": terminal_actual,
            "terminal_predicted_rul": terminal_pred,
            "terminal_error": float(abs(terminal_actual - terminal_pred)),
        })
    return records


def engine_bootstrap_mean_ci(values: list[float], seed: int = 42, n_bootstrap: int = 2000) -> dict:
    """Percentile CI by resampling engines, never correlated within-engine rows."""
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("engine-level bootstrap requires finite per-engine values")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    means = array[samples].mean(axis=1)
    return {
        "n_engines": int(len(array)),
        "estimate": float(array.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "method": "percentile bootstrap over engine units",
        "replicates": int(n_bootstrap),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(model_type: str = "random_forest", include_cycle_feature: bool = False) -> None:
    data_dir = Path("data/raw/cmapss")
    full_train = add_train_rul(load_telemetry(data_dir, "FD001", "train"))
    split = split_engine_units(full_train, validation_fraction=0.2, seed=42)
    train_frame = full_train[full_train["unit_id"].isin(split.train_units)].copy()
    validation_frame = full_train[full_train["unit_id"].isin(split.validation_units)].copy()
    test_frame = add_test_rul(
        load_telemetry(data_dir, "FD001", "test"), load_test_rul(data_dir, "FD001")
    )

    dropped = constant_columns(train_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]
    config = PrognosticsConfig(
        model_type=model_type,
        include_cycle_feature=include_cycle_feature,
    )
    agent = PrognosticsAgent.fit(train_frame, features, config)

    # Uncertainty calibration on validation engines (held-out from training)
    q90, q95 = agent.calibrate_uncertainty(validation_frame)

    validation_predictions = agent.predict_frame(validation_frame)
    test_predictions = agent.predict_frame(test_frame)
    validation_terminal = agent.predict_current(validation_frame)
    validation_pseudo_current = select_rul_stratified_rows(
        validation_predictions,
        rul_reference=config.rul_reference,
        seed=config.random_state,
    )
    test_terminal = agent.predict_current(test_frame)

    # Comparative benchmark across models
    if model_type == "random_forest" and not include_cycle_feature:
        benchmark_results = benchmark_all_prognostics(
            train_frame,
            validation_frame,
            test_frame,
            features,
            window_size=config.window_size,
            seed=config.random_state,
        )
    else:
        benchmark_results = {
            "status": "skipped",
            "reason": (
                "Use grouped training-only model comparison before evaluating "
                "a non-baseline agent on the official test set."
            ),
        }

    report = {
        "subset": "FD001",
        "agent": "PrognosticsAgent",
        "backend": model_type,
        "config": config.as_dict(),
        "train_engines": len(split.train_units),
        "validation_engines": len(split.validation_units),
        "test_engines": int(test_frame["unit_id"].nunique()),
        "features_used": features,
        "features_dropped_as_constant": dropped,
        "conformal_calibration_margins": {
            "q90_margin_cycles": q90,
            "q95_margin_cycles": q95,
            "calibration_engines": getattr(agent, "n_calibration_engines_", None),
            "selected_validation_rul_reference_cycles": getattr(agent, "calibration_reference_rul_", []),
        },
        "validation_window_metrics": metrics(validation_predictions),
        "validation_terminal_metrics": metrics(validation_terminal),
        "validation_pseudo_current_metrics": metrics(validation_pseudo_current),
        "validation_point_selection": (
            "One window per held-out validation engine, selected across a fixed 0-125 cycle RUL range; "
            "the true run-to-failure endpoint metrics are retained separately."
        ),
        "test_window_metrics": metrics(test_predictions),
        "test_terminal_metrics": metrics(test_terminal),
        "comparative_benchmark": benchmark_results,
        "per_engine_errors_test": per_engine_error_analysis(test_predictions),
        "per_engine_window_mae_bootstrap_ci95": engine_bootstrap_mean_ci(
            [row["window_mae"] for row in per_engine_error_analysis(test_predictions)],
            seed=config.random_state,
        ),
        "terminal_engine_mae_bootstrap_ci95": engine_bootstrap_mean_ci(
            np.abs(
                test_terminal["rul"].to_numpy(dtype=float)
                - test_terminal["predicted_rul"].to_numpy(dtype=float)
            ).tolist(),
            seed=config.random_state,
        ),
        "uncertainty_note": (
            "The conformal margin is calibrated from one RUL-stratified pseudo-current residual per held-out "
            "validation engine using a finite-sample quantile. Test coverage and interval width are evaluation-only. "
            "Tree dispersion is reported separately as an uncalibrated heuristic."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit_learn": package_version("scikit-learn"),
            "xgboost": package_version("xgboost"),
            "seed": config.random_state,
        },
    }
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    is_legacy_baseline = model_type == "random_forest" and not include_cycle_feature
    model_suffix = (
        "baseline"
        if is_legacy_baseline
        else model_type + ("_cycle" if include_cycle_feature else "")
    )
    if is_legacy_baseline:
        model_path = models_dir / "fd001_prognostics_agent.joblib"
        report_path = output_dir / "fd001_prognostics_agent.json"
        validation_path = output_dir / "fd001_prognostics_validation_predictions.csv"
        test_path = output_dir / "fd001_prognostics_test_predictions.csv"
        per_engine_path = output_dir / "fd001_prognostics_per_engine_test.csv"
    else:
        model_path = models_dir / f"fd001_prognostics_agent_{model_suffix}.joblib"
        report_path = output_dir / f"fd001_prognostics_agent_{model_suffix}.json"
        validation_path = output_dir / f"fd001_prognostics_validation_{model_suffix}.csv"
        test_path = output_dir / f"fd001_prognostics_test_{model_suffix}.csv"
        per_engine_path = output_dir / f"fd001_prognostics_per_engine_test_{model_suffix}.csv"
    agent.save(model_path)
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    validation_predictions.to_csv(
        validation_path, index=False
    )
    test_predictions.to_csv(
        test_path, index=False
    )
    per_engine = pd.DataFrame(report["per_engine_errors_test"])
    per_engine.to_csv(
        per_engine_path,
        index=False,
    )

    print("FD001 Prognostics Agent & Benchmark")
    print("===================================")
    print(f"backend: {model_type}; cycle feature={include_cycle_feature}")
    print(f"validation conformal 95% margin: +/- {q95:.2f} cycles")
    print(
        "validation windows: "
        f"MAE={report['validation_window_metrics']['mae']:.3f}, "
        f"RMSE={report['validation_window_metrics']['rmse']:.3f}, "
        f"R2={report['validation_window_metrics']['r2']:.3f}"
    )
    print(
        "official test terminal engines: "
        f"MAE={report['test_terminal_metrics']['mae']:.3f}, "
        f"RMSE={report['test_terminal_metrics']['rmse']:.3f}, "
        f"R2={report['test_terminal_metrics']['r2']:.3f}, "
        f"NASA score={report['test_terminal_metrics']['nasa_score']:.1f}"
    )
    print(
        "test tree dispersion coverage: "
        f"{report['test_terminal_metrics']['interval_coverage_95']:.3f}"
    )
    print(
        "test conformal calibrated coverage: "
        f"{report['test_terminal_metrics'].get('conformal_coverage_95', 0):.3f}"
    )
    print("comparative benchmark terminal test MAE:")
    for model_name, m in benchmark_results.get("test_terminal_metrics", {}).items():
        print(f"  {model_name}: MAE={m['mae']:.3f}, RMSE={m['rmse']:.3f}, NASA={m['nasa_score']:.1f}")
    print(f"saved model: {model_path}")
    print(f"saved report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-type",
        choices=(
            "random_forest",
            "xgboost",
            "ridge",
            "naive",
            "lstm",
            "pytorch_cnn",
            "pytorch_transformer",
        ),
        default="random_forest",
    )
    parser.add_argument("--cycle-feature", action="store_true")
    args = parser.parse_args()
    main(args.model_type, args.cycle_feature)
