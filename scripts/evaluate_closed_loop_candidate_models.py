"""Compare prognostics choices inside the coordinated closed-loop prototype.

Uses the existing engine-disjoint development split for each subset. It is a
development evaluation, not a new independent final test. Actual RUL is kept
outside the coordinator and joined only after its decision has been returned.
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

from scripts.evaluate_closed_loop_development import (
    HORIZON,
    MILESTONES,
    SEED,
    _fit_components,
    _split_units,
    _summarize_rows,
)
from scripts.evaluate_fd004_cycle_age_holdout import _decision_row
from src.agent_coordinator import HealthAgentCoordinator
from src.cmapss_data import add_train_rul, load_telemetry, transform_features
from src.closed_loop import ClosedLoopOrchestrator
from src.digital_twin import DataDrivenTransitionModel, DigitalTwin
from src.prognostics import PrognosticsAgent, PrognosticsConfig


def _paired_engine_mae(
    model_rows: dict[str, list[dict[str, Any]]], baseline_name: str
) -> dict[str, Any]:
    baseline = pd.DataFrame(model_rows[baseline_name])
    baseline = baseline[baseline["predicted_rul"].notna()].copy()
    baseline["abs_error"] = (
        baseline["predicted_rul"] - baseline["actual_rul_eval_only"]
    ).abs()
    results: dict[str, Any] = {}
    rng = np.random.default_rng(SEED + 888)
    for model_name, rows in model_rows.items():
        if model_name == baseline_name:
            continue
        candidate = pd.DataFrame(rows)
        candidate = candidate[candidate["predicted_rul"].notna()].copy()
        candidate["abs_error"] = (
            candidate["predicted_rul"] - candidate["actual_rul_eval_only"]
        ).abs()
        paired = baseline[["unit_id", "cycle", "abs_error"]].merge(
            candidate[["unit_id", "cycle", "abs_error"]],
            on=["unit_id", "cycle"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        per_engine = paired.assign(
            difference=paired["abs_error_candidate"]
            - paired["abs_error_baseline"]
        ).groupby("unit_id", sort=True)["difference"].mean()
        values = per_engine.to_numpy(dtype=float)
        indices = rng.integers(0, len(values), size=(20_000, len(values)))
        draws = values[indices].mean(axis=1)
        results[model_name] = {
            "paired_engine_macro_mae_difference_candidate_minus_baseline": float(
                values.mean()
            ),
            "engine_cluster_bootstrap_95_ci": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
            "engine_count": int(len(values)),
        }
    return results


def _evaluate_subset(subset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_dir = ROOT / "data" / "raw" / "cmapss"
    full_frame = add_train_rul(load_telemetry(data_dir, subset, "train"))
    fit_units, calibration_units, evaluation_units = _split_units(full_frame)
    fit_frame = full_frame[full_frame["unit_id"].isin(fit_units)].copy()
    calibration_frame = full_frame[
        full_frame["unit_id"].isin(calibration_units)
    ].copy()
    heldout = full_frame[full_frame["unit_id"].isin(evaluation_units)].copy()

    (
        features,
        scaler,
        monitor,
        diagnosis,
        baseline_agent,
        twin,
        planner,
        verifier,
        _baseline_orchestrator,
    ) = _fit_components(fit_frame, calibration_frame)

    cycle_config = PrognosticsConfig(
        **{
            **baseline_agent.config.as_dict(),
            "include_cycle_feature": True,
            "planning_margin_cycles": (
                tuple(MILESTONES[2:]) if subset == "FD004" else ()
            ),
        }
    )
    cycle_agent = PrognosticsAgent.fit(fit_frame, features, cycle_config)
    cycle_agent.calibrate_uncertainty(calibration_frame)

    lstm_config = PrognosticsConfig(
        model_type="lstm",
        window_size=baseline_agent.config.window_size,
        random_state=SEED,
        lstm_hidden_size=24,
        lstm_epochs=8,
        lstm_batch_size=128,
        lstm_training_stride=5,
        lstm_validation_fraction=0.15,
        lstm_early_stopping_patience=2,
    )
    lstm_agent = PrognosticsAgent.fit(fit_frame, features, lstm_config)
    lstm_agent.calibrate_uncertainty(calibration_frame)

    baseline_orchestrator = ClosedLoopOrchestrator(
        baseline_agent, twin, planner, verifier
    )
    cycle_orchestrator = ClosedLoopOrchestrator(cycle_agent, twin, planner, verifier)
    lstm_fit_scaled = transform_features(
        fit_frame, features, lstm_agent.scaler
    )
    lstm_transition = DataDrivenTransitionModel(features, alpha=1.0).fit(
        lstm_fit_scaled
    )
    lstm_twin = DigitalTwin(
        lstm_transition, features, rul_reference=125.0, max_sensor_z=5.0
    )
    lstm_orchestrator = ClosedLoopOrchestrator(
        lstm_agent, lstm_twin, planner, verifier
    )
    coordinators = {
        "rf30_global": HealthAgentCoordinator(
            monitor, diagnosis, baseline_orchestrator, diagnosis_scaler=scaler
        ),
        "rf30_cycle_feature_global": HealthAgentCoordinator(
            monitor, diagnosis, cycle_orchestrator, diagnosis_scaler=scaler
        ),
        "rf30_cycle_feature_milestone_margin": HealthAgentCoordinator(
            monitor, diagnosis, cycle_orchestrator, diagnosis_scaler=scaler
        ),
        "numpy_lstm_agent": HealthAgentCoordinator(
            monitor, diagnosis, lstm_orchestrator, diagnosis_scaler=scaler
        ),
    }
    cycle_margins = dict(getattr(cycle_agent, "planning_margins_by_cycle_", {}))
    evaluation_truth = heldout[["unit_id", "cycle", "rul"]].copy()
    truth_lookup = {
        (int(row.unit_id), int(row.cycle)): float(row.rul)
        for row in evaluation_truth.itertuples(index=False)
    }
    telemetry = heldout.drop(columns=["rul"])
    model_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in coordinators
    }
    for unit_id in sorted(evaluation_units):
        engine = telemetry[telemetry["unit_id"] == unit_id].sort_values("cycle")
        observed_cycles = set(engine["cycle"].astype(int))
        for milestone in MILESTONES:
            if milestone not in observed_cycles:
                continue
            prefix = engine[engine["cycle"] <= milestone].copy()
            actual_rul = truth_lookup[(unit_id, milestone)]
            for model_name, coordinator in coordinators.items():
                if model_name == "rf30_cycle_feature_global":
                    cycle_agent.planning_margins_by_cycle_ = {}
                elif model_name == "rf30_cycle_feature_milestone_margin":
                    cycle_agent.planning_margins_by_cycle_ = cycle_margins
                coordinated = coordinator.step(prefix, horizon=HORIZON)
                model_rows[model_name].append(
                    _decision_row(
                        model_name,
                        unit_id,
                        milestone,
                        actual_rul,
                        coordinated.outcome,
                        coordinated.outcome.anomaly_score,
                        coordinated.outcome.diagnosis_state,
                        coordinated.outcome.diagnosis_confidence,
                    )
                )

    summaries = {
        model_name: _summarize_rows(rows, len(evaluation_units), SEED + index)
        for index, (model_name, rows) in enumerate(model_rows.items())
    }
    by_cycle: dict[str, Any] = {}
    for cycle in MILESTONES:
        by_cycle[str(cycle)] = {
            model_name: _summarize_rows(
                [row for row in rows if row["cycle"] == cycle],
                len(evaluation_units),
                SEED + cycle + index,
            )
            for index, (model_name, rows) in enumerate(model_rows.items())
        }
    paired = _paired_engine_mae(model_rows, "rf30_global")

    report = {
        "experiment": f"{subset} coordinated closed-loop prognostics comparison",
        "evaluation_status": "development only; reuses the existing fixed engine split",
        "official_test_files_loaded": False,
        "protocol": {
            "source": f"official C-MAPSS {subset} training trajectories",
            "fit_engine_ids": list(fit_units),
            "calibration_engine_ids": list(calibration_units),
            "decision_evaluation_engine_ids": list(evaluation_units),
            "decision_milestones_cycles": list(MILESTONES),
            "agent_models": {
                "rf30_global": baseline_agent.config.as_dict(),
                "rf30_cycle_feature": cycle_config.as_dict(),
                "numpy_lstm_agent": lstm_config.as_dict(),
            },
            "cycle_feature_milestone_margins": cycle_margins,
            "lstm_inner_best_epoch": getattr(lstm_agent.model, "best_epoch_", None),
            "telemetry_label_rule": (
                "The coordinator receives telemetry without RUL; actual labels are joined only after each decision."
            ),
        },
        "overall": summaries,
        "paired_engine_mae_vs_rf30": paired,
        "metrics_by_cycle": by_cycle,
        "limitations": [
            "This is a development evaluation on a split inspected by earlier baseline reports, not an independent final test.",
            "C-MAPSS contains no measured recovery-action outcomes; action effects remain Digital Twin assumptions.",
            "Diagnosis labels are project-defined RUL health bands, not physical fault labels.",
            "This evaluates the code-coordinated agent pipeline, not an LLM or unrestricted autonomous agent system.",
        ],
    }
    return report, [row for rows in model_rows.values() for row in rows]


def main() -> None:
    started = time.perf_counter()
    reports: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for subset in ("FD001", "FD004"):
        report, rows = _evaluate_subset(subset)
        reports[subset] = report
        all_rows.extend(rows)
        print(f"completed coordinated {subset} comparison")
        for name, metrics in report["overall"].items():
            print(
                f"  {name}: MAE={metrics['rul_metrics_on_decision_states']['mae_cycles']}, "
                f"coverage={metrics['rul_metrics_on_decision_states']['planning_lower_bound_empirical_coverage']}, "
                f"actions={metrics['selected_action_counts']}"
            )

    report = {
        "experiment": "Coordinated closed-loop model comparison",
        "datasets": reports,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    out_dir = ROOT / "reports" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "closed_loop_candidate_models.json"
    csv_path = out_dir / "closed_loop_candidate_models_states.csv"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"State-level results: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
