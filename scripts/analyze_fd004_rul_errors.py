"""Trace FD004 prognostics errors and the calibrated planning margin.

The analysis rebuilds the current fit/calibration split from FD004 training
trajectories, identifies which calibration residual sets the one-sided margin,
and slices already-held-out decision errors by cycle, RUL band, and
train-fitted operating-condition cluster. It never loads official test files.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.cmapss_data import (
    FEATURE_COLUMNS,
    OPERATING_COLUMNS,
    add_train_rul,
    constant_columns,
    load_telemetry,
    split_engine_units,
)
from src.prognostics import (
    PrognosticsAgent,
    PrognosticsConfig,
    select_rul_stratified_rows,
)


SEED = 42


def _current_split(frame: pd.DataFrame) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    outer = split_engine_units(frame, validation_fraction=0.40, seed=SEED)
    holdout = frame[frame["unit_id"].isin(outer.validation_units)]
    inner = split_engine_units(holdout, validation_fraction=0.50, seed=SEED + 1)
    return outer.train_units, inner.train_units, inner.validation_units


def _regression_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"n": 0}
    signed = frame["signed_error"].to_numpy(dtype=float)
    absolute = np.abs(signed)
    coverage = (
        frame["actual_rul"] >= frame["planning_rul_lower_95"]
        if "planning_rul_lower_95" in frame
        else pd.Series([], dtype=bool)
    )
    result: dict[str, float | int] = {
        "n": int(len(frame)),
        "engine_count": int(frame["unit_id"].nunique()),
        "mae_cycles": float(absolute.mean()),
        "rmse_cycles": float(np.sqrt(np.mean(signed**2))),
        "mean_signed_error_cycles": float(signed.mean()),
        "median_absolute_error_cycles": float(np.median(absolute)),
        "overprediction_rate": float(np.mean(signed > 0.0)),
        "planning_lower_bound_coverage": (
            float(coverage.mean()) if len(coverage) else float("nan")
        ),
    }
    return result


def _slice_metrics(frame: pd.DataFrame, field: str, slice_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(field, sort=True, observed=True):
        rows.append(
            {
                "slice_type": slice_type,
                "slice": str(value),
                **_regression_metrics(group),
            }
        )
    return rows


def _per_engine_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit_id, group in frame.groupby("unit_id", sort=True):
        errors = group["signed_error"].to_numpy(dtype=float)
        rows.append(
            {
                "unit_id": int(unit_id),
                "decision_states": int(len(group)),
                "mae_cycles": float(np.mean(np.abs(errors))),
                "maximum_absolute_error_cycles": float(np.max(np.abs(errors))),
                "mean_signed_error_cycles": float(np.mean(errors)),
                "overprediction_states": int(np.sum(errors > 0.0)),
                "underprediction_states": int(np.sum(errors < 0.0)),
                "selected_actions": ";".join(
                    f"{action}:{count}"
                    for action, count in sorted(
                        group["selected_action"].value_counts().items()
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mae_cycles", "maximum_absolute_error_cycles"], ascending=False
    )


def _calibration_diagnostics(
    train_frame: pd.DataFrame,
    fit_units: tuple[int, ...],
    calibration_units: tuple[int, ...],
    config: PrognosticsConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fit_frame = train_frame[train_frame["unit_id"].isin(fit_units)].copy()
    calibration_frame = train_frame[
        train_frame["unit_id"].isin(calibration_units)
    ].copy()
    dropped = constant_columns(fit_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]
    agent = PrognosticsAgent.fit(fit_frame, features, config=config)
    agent.calibrate_uncertainty(calibration_frame)
    prediction_frame = agent.predict_frame(calibration_frame)
    selected = select_rul_stratified_rows(
        prediction_frame,
        rul_reference=config.rul_reference,
        seed=config.random_state,
    ).copy()
    selected["signed_error"] = selected["predicted_rul"] - selected["rul"]
    selected["absolute_error"] = selected["signed_error"].abs()
    selected["overestimation_residual"] = selected["signed_error"].clip(lower=0.0)
    count = len(selected)
    quantile_order = int(math.ceil((count + 1) * 0.95))
    effective_order = min(quantile_order, count)
    ordered_signed_errors = np.sort(selected["signed_error"].to_numpy(dtype=float))
    recomputed_margin = max(
        0.0,
        float(ordered_signed_errors[effective_order - 1]),
    )
    worst_index = int(selected["signed_error"].idxmax())
    worst = selected.loc[worst_index]
    summary = {
        "calibration_engine_count": int(selected["unit_id"].nunique()),
        "selected_residual_count": int(count),
        "conformal_order_statistic_95": {
            "requested_order_one_based": quantile_order,
            "effective_order_one_based": effective_order,
            "selected_quantile_value_cycles": recomputed_margin,
            "reported_planning_margin_cycles": float(
                agent.lower_bound_margin_95
            ),
            "matches_agent_margin": bool(
                np.isclose(recomputed_margin, agent.lower_bound_margin_95)
            ),
        },
        "symmetric_q90_cycles": float(agent.conformal_q90),
        "symmetric_q95_cycles": float(agent.conformal_q95),
        "calibration_error_metrics": _regression_metrics(
            selected.rename(columns={"rul": "actual_rul"}).assign(
                actual_rul=selected["rul"],
                planning_rul_lower_95=np.maximum(
                    selected["predicted_rul"] - agent.lower_bound_margin_95, 0.0
                ),
            )
        ),
        "largest_overestimation_residual": {
            "unit_id": int(worst["unit_id"]),
            "end_cycle": int(worst["end_cycle"]),
            "actual_rul": float(worst["rul"]),
            "predicted_rul": float(worst["predicted_rul"]),
            "signed_error_cycles": float(worst["signed_error"]),
        },
        "residual_summary": {
            "overprediction_count": int((selected["signed_error"] > 0).sum()),
            "underprediction_count": int((selected["signed_error"] < 0).sum()),
            "maximum_absolute_error_cycles": float(selected["absolute_error"].max()),
            "p90_absolute_error_cycles": float(
                selected["absolute_error"].quantile(0.90, interpolation="higher")
            ),
        },
        "selection_method": (
            "one pseudo-current residual per calibration engine, selected across "
            "the configured RUL range"
        ),
    }
    output = selected[
        [
            "unit_id",
            "end_cycle",
            "rul",
            "predicted_rul",
            "signed_error",
            "absolute_error",
            "overestimation_residual",
        ]
    ].rename(columns={"rul": "actual_rul"})
    return summary, output.sort_values("unit_id")


def main() -> None:
    data_dir = ROOT / "data" / "raw" / "cmapss"
    report_path = ROOT / "reports" / "generated" / "closed_loop_development_evaluation.json"
    states_path = ROOT / "reports" / "generated" / "closed_loop_development_states.csv"
    if not report_path.exists() or not states_path.exists():
        raise FileNotFoundError(
            "Run scripts.evaluate_closed_loop_development before this diagnostic"
        )

    protocol = json.loads(report_path.read_text(encoding="utf-8"))["datasets"]["FD004"][
        "evaluation_protocol"
    ]
    fit_units = tuple(protocol["model_fit_engine_ids"])
    calibration_units = tuple(protocol["uncertainty_calibration_engine_ids"])
    evaluation_units = tuple(protocol["decision_evaluation_engine_ids"])
    config = PrognosticsConfig(**protocol["prognostics_config"])
    full_train = add_train_rul(load_telemetry(data_dir, "FD004", "train"))
    calibration_summary, calibration_residuals = _calibration_diagnostics(
        full_train, fit_units, calibration_units, config
    )

    states = pd.read_csv(states_path)
    states = states[
        (states["subset"] == "FD004")
        & states["predicted_rul"].notna()
    ].copy()
    states["signed_error"] = states["predicted_rul"] - states["actual_rul_eval_only"]
    states["actual_rul"] = states["actual_rul_eval_only"]
    states["actual_rul_band"] = pd.cut(
        states["actual_rul"],
        bins=[-np.inf, 25.0, 50.0, 100.0, 150.0, np.inf],
        labels=["<=25", "25-50", "50-100", "100-150", ">150"],
        right=True,
    )

    fit_frame = full_train[full_train["unit_id"].isin(fit_units)]
    setting_scaler = StandardScaler().fit(
        fit_frame[OPERATING_COLUMNS].to_numpy(dtype=float)
    )
    fit_settings = setting_scaler.transform(
        fit_frame[OPERATING_COLUMNS].to_numpy(dtype=float)
    )
    distinct_settings = np.unique(np.round(fit_settings, 3), axis=0)
    cluster_count = min(6, len(distinct_settings))
    if cluster_count < 2:
        raise ValueError("FD004 fit engines did not contain multiple operating settings")
    condition_model = KMeans(
        n_clusters=cluster_count,
        random_state=SEED,
        n_init=20,
    ).fit(fit_settings)

    telemetry = load_telemetry(data_dir, "FD004", "train")
    observation_keys = states[["unit_id", "cycle"]].drop_duplicates()
    observed_settings = observation_keys.merge(
        telemetry[["unit_id", "cycle", *OPERATING_COLUMNS]],
        on=["unit_id", "cycle"],
        how="left",
        validate="one_to_one",
    )
    if observed_settings[OPERATING_COLUMNS].isna().any().any():
        raise RuntimeError("Could not match all held-out states to operating settings")
    observed_scaled = setting_scaler.transform(
        observed_settings[OPERATING_COLUMNS].to_numpy(dtype=float)
    )
    observed_settings["condition_cluster"] = condition_model.predict(observed_scaled)
    states = states.merge(
        observed_settings[["unit_id", "cycle", "condition_cluster"]],
        on=["unit_id", "cycle"],
        how="left",
        validate="one_to_one",
    )
    states["condition_cluster"] = states["condition_cluster"].map(
        lambda label: f"condition_{int(label)}"
    )

    slice_rows = [
        *_slice_metrics(states, "cycle", "cycle"),
        *_slice_metrics(states, "actual_rul_band", "actual_rul_band"),
        *_slice_metrics(states, "condition_cluster", "train_fitted_condition_cluster"),
    ]
    error_by_engine = _per_engine_metrics(states)

    ablation_path = ROOT / "reports" / "generated" / "fd004_ablation_summary.csv"
    existing_ablation_context: list[dict[str, Any]] = []
    if ablation_path.exists():
        ablations = pd.read_csv(ablation_path)
        relevant = ablations[
            ablations["ablation"].isin(
                [
                    "window_size_30_reference",
                    "window_size_40",
                    "operating_condition_normalized_sensors",
                    "model_comparison_numpy_lstm_sequence",
                ]
            )
        ]
        existing_ablation_context = relevant.to_dict(orient="records")

    cluster_centers = setting_scaler.inverse_transform(
        condition_model.cluster_centers_
    )
    report = {
        "subset": "FD004",
        "official_test_files_loaded": False,
        "split": {
            "model_fit_engine_count": len(fit_units),
            "calibration_engine_count": len(calibration_units),
            "evaluation_engine_count": len(evaluation_units),
            "engine_groups_disjoint": not (
                set(fit_units) & set(calibration_units)
                or set(fit_units) & set(evaluation_units)
                or set(calibration_units) & set(evaluation_units)
            ),
        },
        "evaluation_errors": {
            "overall": _regression_metrics(states),
            "by_cycle_rul_band_and_operating_condition": slice_rows,
            "condition_clustering": {
                "method": "KMeans on standardized operating settings, fit-engine rows only",
                "cluster_count": int(cluster_count),
                "cluster_centers_in_original_setting_units": {
                    f"condition_{index}": [float(value) for value in center]
                    for index, center in enumerate(cluster_centers)
                },
            },
            "worst_ten_engines_by_mean_absolute_error": error_by_engine.head(10).to_dict(
                orient="records"
            ),
        },
        "calibration_residual_diagnostics": calibration_summary,
        "previous_experiment_context": {
            "note": (
                "These pre-existing ablations used a separate 80/20 split and one "
                "RUL-stratified pseudo-current point per validation engine. They are "
                "context only and are not directly comparable to fixed-cycle errors here."
            ),
            "selected_results": existing_ablation_context,
        },
        "interpretation_limits": [
            "Operating-condition cluster IDs are unsupervised labels inferred from settings, not NASA fault labels.",
            "C-MAPSS contains no recovery-action outcomes, so planner actions and Twin action effects remain assumptions.",
            "Evaluation RUL labels are used only after telemetry-only predictions for error scoring.",
            "A finite conformal order statistic does not guarantee coverage at every fixed cycle or operating-condition cluster; calibration uses one RUL-stratified point per engine.",
        ],
    }

    output_dir = ROOT / "reports" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fd004_rul_error_diagnostics.json"
    slices_path = output_dir / "fd004_rul_error_slices.csv"
    engines_path = output_dir / "fd004_rul_error_by_engine.csv"
    calibration_path = output_dir / "fd004_calibration_residuals.csv"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(slice_rows).to_csv(slices_path, index=False)
    error_by_engine.to_csv(engines_path, index=False)
    calibration_residuals.to_csv(calibration_path, index=False)

    print("FD004 RUL error and calibration diagnosis")
    print("==========================================")
    print(
        "calibration one-sided 95% margin: "
        f"{calibration_summary['conformal_order_statistic_95']['reported_planning_margin_cycles']:.2f} cycles"
    )
    print(
        "largest calibration overestimate: "
        f"engine {calibration_summary['largest_overestimation_residual']['unit_id']}, "
        f"cycle {calibration_summary['largest_overestimation_residual']['end_cycle']}, "
        f"+{calibration_summary['largest_overestimation_residual']['signed_error_cycles']:.2f} cycles"
    )
    for group in slice_rows:
        print(
            f"{group['slice_type']}={group['slice']}: n={group['n']}, "
            f"MAE={group['mae_cycles']:.2f}, bias={group['mean_signed_error_cycles']:.2f}, "
            f"lower-bound coverage={group['planning_lower_bound_coverage']:.1%}"
        )
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"Slices CSV: {slices_path.relative_to(ROOT)}")
    print(f"Per-engine CSV: {engines_path.relative_to(ROOT)}")
    print(f"Calibration CSV: {calibration_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
