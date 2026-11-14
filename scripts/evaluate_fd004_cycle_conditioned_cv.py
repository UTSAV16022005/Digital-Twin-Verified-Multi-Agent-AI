"""Five-fold grouped evaluation of milestone-specific planning calibration.

Each outer fold holds out whole engines for evaluation. The remaining engines
are split again into model-fit and uncertainty-calibration groups. Only official
C-MAPSS training trajectories are loaded; official test files are never opened.
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
    add_train_rul,
    constant_columns,
    load_telemetry,
    split_engine_units,
    verify_no_leakage,
)
from src.prognostics import PrognosticsAgent, PrognosticsConfig


SEED = 42
FOLDS = 5
MILESTONES = (30, 60, 90, 120)
WINDOW_SIZE = 30
RF_TREES = 80
BOOTSTRAP_REPLICATES = 20_000
MODEL_NAMES = (
    "rf30_global",
    "rf30_cycle_feature_global",
    "rf30_cycle_feature_milestone_margin",
)


def _bootstrap_ci(values: np.ndarray, seed: int) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_REPLICATES, len(values)),
    )
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _engine_macro_values(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("unit_id", sort=True)[column].mean()


def _summarize(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    if frame.empty:
        return {"state_count": 0, "engine_count": 0}
    errors = frame["predicted_rul"].to_numpy(dtype=float) - frame[
        "actual_rul"
    ].to_numpy(dtype=float)
    per_engine_mae = _engine_macro_values(frame, "absolute_error")
    per_engine_coverage = _engine_macro_values(frame, "covered")
    return {
        "state_count": int(len(frame)),
        "engine_count": int(frame["unit_id"].nunique()),
        "mae_cycles": float(np.mean(np.abs(errors))),
        "engine_macro_mae_cycles": float(per_engine_mae.mean()),
        "engine_macro_mae_engine_cluster_bootstrap_95_ci": _bootstrap_ci(
            per_engine_mae.to_numpy(dtype=float), seed
        ),
        "mean_signed_error_cycles": float(np.mean(errors)),
        "overprediction_rate": float(np.mean(errors > 0.0)),
        "planning_lower_bound_empirical_coverage": float(frame["covered"].mean()),
        "engine_average_coverage": float(per_engine_coverage.mean()),
        "engine_cluster_bootstrap_95_ci_coverage": _bootstrap_ci(
            per_engine_coverage.to_numpy(dtype=float), seed + 1
        ),
    }


def _paired_engine_comparison(
    frame: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
    metric_column: str,
    seed: int,
    favorable_direction: str,
) -> dict[str, Any]:
    per_engine = frame.pivot_table(
        index="unit_id",
        columns="model",
        values=metric_column,
        aggfunc="mean",
    )
    paired = per_engine[[baseline_name, candidate_name]].dropna()
    differences = (paired[candidate_name] - paired[baseline_name]).to_numpy(
        dtype=float
    )
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "difference_definition": (
            f"candidate minus baseline engine-mean {metric_column}; "
            f"{favorable_direction}"
        ),
        "mean_paired_engine_difference": float(differences.mean()),
        "paired_engine_cluster_bootstrap_95_ci": _bootstrap_ci(
            differences, seed
        ),
        "engine_count": int(len(differences)),
    }


def _predict_milestones(
    agent: PrognosticsAgent,
    telemetry: pd.DataFrame,
    truth: pd.DataFrame,
    fold: int,
    model_name: str,
    use_milestone_margin: bool,
) -> pd.DataFrame:
    predictions = agent.predict_frame(telemetry)
    predictions = predictions[predictions["end_cycle"].isin(MILESTONES)].copy()
    truth_lookup = truth.set_index(["unit_id", "cycle"])["rul"]
    keys = pd.MultiIndex.from_frame(
        predictions[["unit_id", "end_cycle"]].rename(
            columns={"end_cycle": "cycle"}
        )
    )
    predictions["actual_rul"] = truth_lookup.reindex(keys).to_numpy(dtype=float)
    if predictions["actual_rul"].isna().any():
        raise RuntimeError("A held-out prediction could not be matched to its RUL")

    if use_milestone_margin:
        lower = predictions["planning_rul_lower_95"].to_numpy(dtype=float)
    else:
        margin = float(agent.lower_bound_margin_95)
        lower = np.maximum(
            predictions["predicted_rul"].to_numpy(dtype=float) - margin, 0.0
        )
    result = pd.DataFrame(
        {
            "fold": fold,
            "model": model_name,
            "unit_id": predictions["unit_id"].to_numpy(dtype=int),
            "cycle": predictions["end_cycle"].to_numpy(dtype=int),
            "actual_rul": predictions["actual_rul"].to_numpy(dtype=float),
            "predicted_rul": predictions["predicted_rul"].to_numpy(dtype=float),
            "planning_lower_rul": lower,
        }
    )
    result["absolute_error"] = (
        result["predicted_rul"] - result["actual_rul"]
    ).abs()
    result["covered"] = result["actual_rul"] >= result["planning_lower_rul"]
    return result


def main(subset: str = "FD004") -> None:
    subset = subset.upper()
    if subset not in {"FD001", "FD004"}:
        raise ValueError("subset must be FD001 or FD004")
    started = time.perf_counter()
    data_dir = ROOT / "data" / "raw" / "cmapss"
    full_frame = add_train_rul(load_telemetry(data_dir, subset, "train"))
    engine_ids = np.asarray(sorted(full_frame["unit_id"].unique()), dtype=int)
    splitter = GroupKFold(n_splits=FOLDS)
    fold_reports: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []

    for fold, (pool_indices, evaluation_indices) in enumerate(
        splitter.split(engine_ids, groups=engine_ids), start=1
    ):
        pool_units = tuple(int(value) for value in engine_ids[pool_indices])
        evaluation_units = tuple(
            int(value) for value in engine_ids[evaluation_indices]
        )
        pool_frame = full_frame[full_frame["unit_id"].isin(pool_units)].copy()
        inner_split = split_engine_units(
            pool_frame,
            validation_fraction=0.25,
            seed=SEED + fold,
        )
        fit_units = inner_split.train_units
        calibration_units = inner_split.validation_units
        verify_no_leakage(fit_units, calibration_units)
        verify_no_leakage(fit_units, evaluation_units)
        verify_no_leakage(calibration_units, evaluation_units)

        fit_frame = full_frame[full_frame["unit_id"].isin(fit_units)].copy()
        calibration_frame = full_frame[
            full_frame["unit_id"].isin(calibration_units)
        ].copy()
        heldout = full_frame[
            full_frame["unit_id"].isin(evaluation_units)
        ].copy()
        dropped = constant_columns(fit_frame, FEATURE_COLUMNS)
        features = [name for name in FEATURE_COLUMNS if name not in dropped]

        baseline_config = PrognosticsConfig(
            window_size=WINDOW_SIZE,
            n_estimators=RF_TREES,
            min_samples_leaf=2,
            max_features=0.70,
            random_state=SEED + fold,
        )
        candidate_config = PrognosticsConfig(
            window_size=WINDOW_SIZE,
            n_estimators=RF_TREES,
            min_samples_leaf=2,
            max_features=0.70,
            random_state=SEED + fold,
            include_cycle_feature=True,
            planning_margin_cycles=MILESTONES,
        )
        baseline = PrognosticsAgent.fit(fit_frame, features, baseline_config)
        baseline.calibrate_uncertainty(calibration_frame)
        candidate = PrognosticsAgent.fit(fit_frame, features, candidate_config)
        candidate.calibrate_uncertainty(calibration_frame)

        truth = heldout[["unit_id", "cycle", "rul"]].copy()
        telemetry = heldout.drop(columns=["rul"])
        fold_predictions = [
            _predict_milestones(
                baseline,
                telemetry,
                truth,
                fold,
                "rf30_global",
                use_milestone_margin=False,
            ),
            _predict_milestones(
                candidate,
                telemetry,
                truth,
                fold,
                "rf30_cycle_feature_global",
                use_milestone_margin=False,
            ),
            _predict_milestones(
                candidate,
                telemetry,
                truth,
                fold,
                "rf30_cycle_feature_milestone_margin",
                use_milestone_margin=True,
            ),
        ]
        fold_frame = pd.concat(fold_predictions, ignore_index=True)
        all_predictions.append(fold_frame)
        fold_reports.append(
            {
                "fold": fold,
                "fit_engine_count": len(fit_units),
                "calibration_engine_count": len(calibration_units),
                "evaluation_engine_count": len(evaluation_units),
                "fit_engine_ids": list(fit_units),
                "calibration_engine_ids": list(calibration_units),
                "evaluation_engine_ids": list(evaluation_units),
                "baseline_pooled_margin_cycles": baseline.lower_bound_margin_95,
                "candidate_pooled_margin_cycles": candidate.lower_bound_margin_95,
                "candidate_milestone_margins_cycles": candidate.planning_margins_by_cycle_,
                "candidate_milestone_calibration_engine_counts": (
                    candidate.planning_margin_engine_counts_by_cycle_
                ),
            }
        )
        print(
            f"completed fold {fold}/{FOLDS}: fit/calibration/evaluation="
            f"{len(fit_units)}/{len(calibration_units)}/{len(evaluation_units)}"
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    overall = {
        name: _summarize(
            predictions[predictions["model"] == name], SEED + index * 10
        )
        for index, name in enumerate(MODEL_NAMES)
    }
    comparisons = {
        "cycle_feature_global_vs_rf30_global_mae": _paired_engine_comparison(
            predictions,
            "rf30_global",
            "rf30_cycle_feature_global",
            "absolute_error",
            SEED + 101,
            "negative favors the candidate",
        ),
        "cycle_feature_milestone_margin_vs_rf30_global_mae": _paired_engine_comparison(
            predictions,
            "rf30_global",
            "rf30_cycle_feature_milestone_margin",
            "absolute_error",
            SEED + 102,
            "negative favors the candidate",
        ),
        "milestone_margin_vs_cycle_feature_global_coverage": _paired_engine_comparison(
            predictions,
            "rf30_cycle_feature_global",
            "rf30_cycle_feature_milestone_margin",
            "covered",
            SEED + 103,
            "positive favors milestone calibration",
        ),
    }
    feature_mae = comparisons["cycle_feature_global_vs_rf30_global_mae"]
    margin_coverage = comparisons[
        "milestone_margin_vs_cycle_feature_global_coverage"
    ]
    feature_ci = feature_mae["paired_engine_cluster_bootstrap_95_ci"]
    margin_ci = margin_coverage["paired_engine_cluster_bootstrap_95_ci"]
    decision_parts = [
        (
            f"The cycle feature reduced paired engine-macro MAE by "
            f"{-feature_mae['mean_paired_engine_difference']:.2f} cycles "
            f"(95% CI for candidate-minus-baseline: "
            f"{feature_ci[0]:.2f} to {feature_ci[1]:.2f})."
            if feature_ci[1] < 0.0
            else "The cycle feature did not show a conclusive paired MAE improvement."
        ),
        (
            f"Milestone-specific margins increased engine-average coverage by "
            f"{100 * margin_coverage['mean_paired_engine_difference']:.2f} "
            f"percentage points (95% CI {100 * margin_ci[0]:.2f} to "
            f"{100 * margin_ci[1]:.2f})."
            if margin_ci[0] > 0.0
            else "Milestone-specific margins did not show a conclusive coverage improvement."
        ),
        "Keep this feature opt-in until the final benchmark protocol is locked.",
    ]
    decision = " ".join(decision_parts)
    by_cycle: dict[str, Any] = {}
    for cycle in MILESTONES:
        cycle_rows = predictions[predictions["cycle"] == cycle]
        by_cycle[str(cycle)] = {
            name: _summarize(
                cycle_rows[cycle_rows["model"] == name],
                SEED + cycle + index,
            )
            for index, name in enumerate(MODEL_NAMES)
        }

    report = {
        "experiment": f"{subset} five-fold grouped CV for cycle-conditioned planning margins",
        "source": f"official C-MAPSS {subset} training trajectories only",
        "official_test_files_loaded": False,
        "protocol": {
            "outer_evaluation": "5-fold GroupKFold by engine",
            "inner_split": "within each outer training group, 75% fit and 25% calibration",
            "approximate_total_proportions": {
                "fit": 0.60,
                "calibration": 0.20,
                "evaluation": 0.20,
            },
            "milestones_cycles": list(MILESTONES),
            "window_size": WINDOW_SIZE,
            "trees_per_random_forest": RF_TREES,
            "cycle_feature": "end_cycle / 100 appended to summarized telemetry window",
            "margin_method": (
                "For each milestone, one signed overestimation residual per "
                "calibration engine and the finite-sample 95% order statistic; "
                "global calibration uses the existing RUL-stratified one-point-per-engine method."
            ),
            "evaluation_labels": (
                "RUL labels are joined after telemetry-only predictions and used only for metrics."
            ),
        },
        "overall_metrics": overall,
        "paired_engine_comparisons": comparisons,
        "decision": decision,
        "metrics_by_cycle": by_cycle,
        "folds": fold_reports,
        "limitations": [
            "This cross-validation evaluates RUL predictions and planning lower bounds, not full Digital Twin action outcomes.",
            "Pointwise milestone coverage does not imply simultaneous coverage over a full trajectory.",
            "The cycle-conditioned method is evaluated only at its configured milestones; other cycles use the pooled margin.",
            "Results use official training trajectories and do not replace a final locked evaluation on official test data.",
            "C-MAPSS has no observed recovery-action outcomes, so simulated action effects remain Twin assumptions.",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }

    out_dir = ROOT / "reports" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = subset.lower()
    json_path = out_dir / f"{name}_cycle_conditioned_cv.json"
    csv_path = out_dir / f"{name}_cycle_conditioned_cv_predictions.csv"
    predictions.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(f"{subset} grouped calibration evaluation")
    print("=" * (len(subset) + 31))
    for name, metrics in overall.items():
        print(
            f"{name}: MAE={metrics['mae_cycles']:.2f}, "
            f"engine-macro MAE={metrics['engine_macro_mae_cycles']:.2f}, "
            f"lower-bound coverage={metrics['planning_lower_bound_empirical_coverage']:.3f}"
        )
    print(f"paired comparisons: {comparisons}")
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"OOF predictions: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        choices=("FD001", "FD004"),
        default="FD004",
        help="C-MAPSS training subset to evaluate (default: FD004)",
    )
    main(parser.parse_args().subset)
