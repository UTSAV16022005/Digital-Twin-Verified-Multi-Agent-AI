"""Evaluate closed-loop decisions at fixed milestones on held-out engines.

Only FD001 and FD004 training trajectories are used. Engines are split into
model-fit, uncertainty-calibration, and decision-evaluation groups. The official
C-MAPSS test files are never loaded. Evaluation RUL labels are joined only
after each telemetry-only decision has been produced.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

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
    verify_no_leakage,
)
from src.closed_loop import ClosedLoopAbstention, ClosedLoopOrchestrator
from src.diagnosis import HealthDiagnosisAgent
from src.digital_twin import DataDrivenTransitionModel, DigitalTwin
from src.monitoring import PCAMonitoringAgent
from src.prognostics import PrognosticsAgent, PrognosticsConfig
from src.recovery_planning import RecoveryPlanningAgent
from src.safety_verification import SafetyConfig, SafetyVerifier


SEED = 42
SUBSETS = ("FD001", "FD004")
MILESTONES = (10, 20, 30, 60, 90, 120)
BOOTSTRAP_REPLICATES = 2000
HORIZON = 5


def _split_units(frame: pd.DataFrame) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return disjoint fit, calibration, and evaluation engine IDs."""

    # A 40% holdout leaves 20% of all engines for calibration and 20% for
    # decision evaluation. At least 19 calibration residuals are needed for a
    # finite split-conformal 95th-percentile order statistic.
    outer = split_engine_units(frame, validation_fraction=0.40, seed=SEED)
    outer_holdout = frame[frame["unit_id"].isin(outer.validation_units)]
    inner = split_engine_units(
        outer_holdout, validation_fraction=0.50, seed=SEED + 1
    )
    fit_units = outer.train_units
    calibration_units = inner.train_units
    evaluation_units = inner.validation_units
    if len(calibration_units) < 19:
        raise ValueError(
            "At least 19 calibration engines are required for a finite "
            "95% split-conformal quantile"
        )
    verify_no_leakage(fit_units, calibration_units)
    verify_no_leakage(fit_units, evaluation_units)
    verify_no_leakage(calibration_units, evaluation_units)
    return fit_units, calibration_units, evaluation_units


def _fit_components(
    fit_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
) -> tuple[
    list[str],
    Any,
    PCAMonitoringAgent,
    HealthDiagnosisAgent,
    PrognosticsAgent,
    DigitalTwin,
    RecoveryPlanningAgent,
    SafetyVerifier,
    ClosedLoopOrchestrator,
]:
    dropped = constant_columns(fit_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]
    prognostics_config = PrognosticsConfig(n_estimators=80)
    prognostics = PrognosticsAgent.fit(
        fit_frame, features, config=prognostics_config
    )
    prognostics.calibrate_uncertainty(calibration_frame)

    scaler = fit_feature_scaler(fit_frame, features)
    fit_scaled = transform_features(fit_frame, features, scaler)
    monitor = PCAMonitoringAgent.fit(fit_frame, features, healthy_cycles=30)
    train_windows, train_targets, _ = make_windows(
        fit_scaled, features, window_size=prognostics_config.window_size
    )
    diagnosis = HealthDiagnosisAgent.fit(
        summarize_windows(train_windows),
        train_targets,
        feature_names=[
            f"{stat}_{feature}"
            for stat in ("last", "mean", "std", "min", "max", "slope")
            for feature in features
        ],
    )
    transition_model = DataDrivenTransitionModel(features, alpha=1.0).fit(
        fit_scaled
    )
    twin = DigitalTwin(
        transition_model, features, rul_reference=125.0, max_sensor_z=5.0
    )
    planner = RecoveryPlanningAgent(risk_budget=0.25)
    verifier = SafetyVerifier(SafetyConfig(max_failure_risk=0.25))
    orchestrator = ClosedLoopOrchestrator(
        prognostics, twin, planner, verifier
    )
    return (
        features,
        scaler,
        monitor,
        diagnosis,
        prognostics,
        twin,
        planner,
        verifier,
        orchestrator,
    )


def _engine_cluster_rate_ci(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    seed: int,
) -> list[float] | None:
    """Percentile CI for a rate, resampling engines as independent clusters."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["unit_id"])].append(row)
    engine_rates = np.asarray(
        [np.mean([predicate(row) for row in group]) for group in grouped.values()],
        dtype=float,
    )
    if not len(engine_rates):
        return None
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(engine_rates),
        size=(BOOTSTRAP_REPLICATES, len(engine_rates)),
    )
    estimates = engine_rates[samples].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _engine_cluster_metric_ci(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    seed: int,
) -> list[float] | None:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["unit_id"])].append(row)
    engine_values = np.asarray(
        [metric(group) for group in grouped.values()], dtype=float
    )
    if not len(engine_values):
        return None
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(engine_values),
        size=(BOOTSTRAP_REPLICATES, len(engine_values)),
    )
    estimates = engine_values[samples].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _wilson_interval(successes: int, trials: int) -> list[float] | None:
    """Wilson score interval when each trial is one independent engine."""

    if trials == 0:
        return None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == trials else min(1.0, center + half_width)
    return [float(lower), float(upper)]


def _summarize_rows(
    rows: list[dict[str, Any]],
    total_evaluation_engines: int,
    seed: int,
) -> dict[str, Any]:
    statuses = Counter(row["decision_status"] for row in rows)
    state_count = len(rows)
    candidate_count = sum(row["candidate_count"] for row in rows)
    approved_count = sum(row["approved_candidate_count"] for row in rows)
    rejected_count = candidate_count - approved_count
    prediction_rows = [row for row in rows if row["predicted_rul"] is not None]
    action_counts = Counter(
        row["selected_action"]
        for row in rows
        if row["selected_action"] is not None
    )
    rejection_reasons = Counter(
        reason for row in rows for reason in row["rejection_reasons"]
    )
    rejected_actions = Counter(
        action for row in rows for action in row["rejected_actions"]
    )

    result: dict[str, Any] = {
        "decision_states": state_count,
        "evaluation_engines_with_state": len({row["unit_id"] for row in rows}),
        "evaluation_engines_total": total_evaluation_engines,
        "state_counts_by_status": dict(sorted(statuses.items())),
        "recommendation_rate": (
            float(statuses["recommendation"] / state_count) if state_count else None
        ),
        "no_safe_action_rate": (
            float(statuses["no_safe_action"] / state_count) if state_count else None
        ),
        "abstention_rate": (
            float(sum(value for key, value in statuses.items() if key not in {
                "recommendation", "no_safe_action"
            }) / state_count)
            if state_count
            else None
        ),
        "candidate_outcomes": candidate_count,
        "approved_candidates": approved_count,
        "rejected_candidates": rejected_count,
        "candidate_approval_rate": (
            float(approved_count / candidate_count) if candidate_count else None
        ),
        "candidate_rejection_rate": (
            float(rejected_count / candidate_count) if candidate_count else None
        ),
        "selected_action_counts": dict(sorted(action_counts.items())),
        "rejected_action_counts": dict(sorted(rejected_actions.items())),
        "safety_rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "engine_cluster_bootstrap_95_ci": {
            "recommendation_rate": _engine_cluster_rate_ci(
                rows,
                lambda row: row["decision_status"] == "recommendation",
                seed,
            ),
            "no_safe_action_rate": _engine_cluster_rate_ci(
                rows,
                lambda row: row["decision_status"] == "no_safe_action",
                seed + 1,
            ),
            "abstention_rate": _engine_cluster_rate_ci(
                rows,
                lambda row: row["decision_status"] not in {
                    "recommendation", "no_safe_action"
                },
                seed + 2,
            ),
        },
    }
    if prediction_rows:
        errors = np.asarray(
            [row["predicted_rul"] - row["actual_rul_eval_only"] for row in prediction_rows],
            dtype=float,
        )
        lower_coverage = np.asarray(
            [
                row["actual_rul_eval_only"] >= row["planning_rul_lower_95"]
                for row in prediction_rows
            ],
            dtype=bool,
        )
        result["rul_metrics_on_decision_states"] = {
            "n_states": len(prediction_rows),
            "mae_cycles": float(np.mean(np.abs(errors))),
            "rmse_cycles": float(np.sqrt(np.mean(errors**2))),
            "mean_signed_error_cycles": float(np.mean(errors)),
            "planning_lower_bound_empirical_coverage": float(lower_coverage.mean()),
            "engine_cluster_bootstrap_95_ci": {
                "mae_cycles": _engine_cluster_metric_ci(
                    prediction_rows,
                    lambda group: float(
                        np.mean(
                            [row["absolute_rul_error"] for row in group]
                        )
                    ),
                    seed + 10,
                ),
                "planning_lower_bound_coverage": _engine_cluster_metric_ci(
                    prediction_rows,
                    lambda group: float(
                        np.mean(
                            [
                                row["actual_rul_eval_only"]
                                >= row["planning_rul_lower_95"]
                                for row in group
                            ]
                        )
                    ),
                    seed + 11,
                ),
            },
        }
    else:
        result["rul_metrics_on_decision_states"] = None
    return result


def _evaluate_subset(subset: str, data_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        prognostics,
        _twin,
        _planner,
        _verifier,
        orchestrator,
    ) = _fit_components(fit_frame, calibration_frame)

    evaluation_truth = heldout[["unit_id", "cycle", "rul"]].copy()
    evaluation_telemetry = heldout.drop(columns=["rul"]).copy()
    truth_lookup = {
        (int(row.unit_id), int(row.cycle)): float(row.rul)
        for row in evaluation_truth.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []
    for unit_id in sorted(int(value) for value in evaluation_units):
        engine = evaluation_telemetry[
            evaluation_telemetry["unit_id"] == unit_id
        ].sort_values("cycle")
        available_cycles = set(int(value) for value in engine["cycle"])
        for milestone in MILESTONES:
            if milestone not in available_cycles:
                continue
            telemetry_prefix = engine[engine["cycle"] <= milestone].copy()
            monitor_scores = monitor.score(telemetry_prefix)
            anomaly_score = float(monitor_scores.iloc[-1]["anomaly_score"])

            diagnosis_state: str | None = None
            diagnosis_confidence: float | None = None
            if milestone >= prognostics.config.window_size:
                scaled_prefix = transform_features(
                    telemetry_prefix, features, scaler
                )
                values = scaled_prefix[features].to_numpy(dtype=np.float32)
                current_window = values[-prognostics.config.window_size :][
                    np.newaxis, :, :
                ]
                probabilities = diagnosis.predict_proba(
                    summarize_windows(current_window)
                ).iloc[0]
                diagnosis_state = str(probabilities.idxmax())
                diagnosis_confidence = float(probabilities.max())

            # This prefix has telemetry only. Ground-truth RUL is looked up after
            # the orchestrator returns and is never passed to any agent.
            result = orchestrator.step(
                telemetry_prefix,
                anomaly_score=anomaly_score,
                diagnosis_state=diagnosis_state,
                diagnosis_confidence=diagnosis_confidence,
                horizon=HORIZON,
            )
            actual_rul = truth_lookup[(unit_id, milestone)]
            is_abstention = isinstance(result, ClosedLoopAbstention)
            safety_decisions = result.safety_decisions
            rejected = [
                (action, decision)
                for action, decision in safety_decisions.items()
                if not decision.approved
            ]
            if result.selected_action is not None:
                selected_decision = safety_decisions[result.selected_action]
                if not selected_decision.approved:
                    raise RuntimeError(
                        "Closed loop selected an action rejected by the verifier"
                    )
            rows.append(
                {
                    "subset": subset,
                    "unit_id": unit_id,
                    "cycle": milestone,
                    "actual_rul_eval_only": actual_rul,
                    "decision_status": result.decision_status,
                    "selected_action": result.selected_action,
                    "predicted_rul": (
                        None if is_abstention else float(result.predicted_rul)
                    ),
                    "planning_rul_lower_95": (
                        None
                        if is_abstention
                        else float(result.planning_rul_lower_95)
                    ),
                    "absolute_rul_error": (
                        None
                        if is_abstention
                        else abs(float(result.predicted_rul) - actual_rul)
                    ),
                    "candidate_count": len(safety_decisions),
                    "approved_candidate_count": sum(
                        decision.approved for decision in safety_decisions.values()
                    ),
                    "rejected_candidate_count": len(rejected),
                    "rejected_actions": [action for action, _ in rejected],
                    "rejection_reasons": [
                        reason
                        for _, decision in rejected
                        for reason in decision.violations
                    ],
                    "anomaly_score": anomaly_score,
                    "diagnosis_state": diagnosis_state,
                    "diagnosis_confidence": diagnosis_confidence,
                }
            )

    by_milestone: dict[str, Any] = {}
    for milestone in MILESTONES:
        milestone_rows = [row for row in rows if row["cycle"] == milestone]
        by_milestone[str(milestone)] = _summarize_rows(
            milestone_rows,
            total_evaluation_engines=len(evaluation_units),
            seed=SEED + milestone,
        )
        milestone_engine_count = len(
            {row["unit_id"] for row in milestone_rows}
        )
        by_milestone[str(milestone)]["independent_engine_wilson_95_ci"] = {
            status: _wilson_interval(
                sum(row["decision_status"] == status for row in milestone_rows),
                milestone_engine_count,
            )
            for status in (
                "recommendation",
                "no_safe_action",
                "insufficient_telemetry",
            )
        }
        by_milestone[str(milestone)]["engines_without_observation_at_milestone"] = (
            len(evaluation_units) - len(milestone_rows)
        )

    report = {
        "subset": subset,
        "evaluation_protocol": {
            "source": "official C-MAPSS training trajectories only",
            "official_test_files_loaded": False,
            "decision_milestones_cycles": list(MILESTONES),
            "prognostics_window_size": prognostics.config.window_size,
            "decision_horizon_cycles": HORIZON,
            "random_seed": SEED,
            "engine_split_proportions": {
                "model_fit": 0.60,
                "uncertainty_calibration": 0.20,
                "decision_evaluation": 0.20,
            },
            "minimum_calibration_engines_for_finite_95_percent_quantile": 19,
            "model_fit_engine_ids": list(fit_units),
            "uncertainty_calibration_engine_ids": list(calibration_units),
            "decision_evaluation_engine_ids": list(evaluation_units),
            "calibration_engine_count": int(prognostics.n_calibration_engines_),
            "prognostics_config": prognostics.config.as_dict(),
            "uncertainty_calibration": {
                "symmetric_q90_cycles": float(prognostics.conformal_q90),
                "symmetric_q95_cycles": float(prognostics.conformal_q95),
                "one_sided_overestimation_margin_95_cycles": float(
                    prognostics.lower_bound_margin_95
                ),
            },
            "calibration_method": (
                "one RUL-stratified pseudo-current window per calibration engine; "
                "held-out decision engines are disjoint"
            ),
            "ground_truth_use": (
                "Evaluation RUL is joined after each telemetry-only decision and is "
                "used only to report predictive error and lower-bound coverage."
            ),
            "confidence_intervals": (
                "Percentile bootstrap resampling is clustered by engine; repeated "
                "milestones from one engine are kept together."
            ),
        },
        "overall": _summarize_rows(
            rows,
            total_evaluation_engines=len(evaluation_units),
            seed=SEED,
        ),
        "by_milestone": by_milestone,
        "limitations": [
            "C-MAPSS supplies no recovery-action outcomes; candidate action effects remain Digital Twin assumptions.",
            "These decision rates describe this prototype and fixed milestone sample, not operational safety probabilities.",
            "Evaluation RUL uses completed training trajectories for post-decision scoring only.",
            "Engine-level confidence intervals may be wide; percentile cluster bootstrap intervals can collapse when every engine has the same observed rate pattern.",
        ],
    }
    return report, rows


def main() -> None:
    data_dir = ROOT / "data" / "raw" / "cmapss"
    reports: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for subset in SUBSETS:
        report, rows = _evaluate_subset(subset, data_dir)
        reports[subset] = report
        all_rows.extend(rows)

    output_dir = ROOT / "reports" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "closed_loop_development_evaluation.json"
    csv_path = output_dir / "closed_loop_development_states.csv"
    aggregate = {
        "experiment": "closed-loop decisions at fixed engine-life milestones",
        "datasets": reports,
        "combined_decision_state_count": len(all_rows),
    }
    json_path.write_text(
        json.dumps(aggregate, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = [
        "subset",
        "unit_id",
        "cycle",
        "actual_rul_eval_only",
        "decision_status",
        "selected_action",
        "predicted_rul",
        "planning_rul_lower_95",
        "absolute_rul_error",
        "candidate_count",
        "approved_candidate_count",
        "rejected_candidate_count",
        "rejected_actions",
        "rejection_reasons",
        "anomaly_score",
        "diagnosis_state",
        "diagnosis_confidence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    **{
                        field: row[field]
                        for field in fields
                        if field not in {"rejected_actions", "rejection_reasons"}
                    },
                    "rejected_actions": ";".join(row["rejected_actions"]),
                    "rejection_reasons": ";".join(row["rejection_reasons"]),
                }
            )

    print("Closed-loop held-out development evaluation")
    print("===========================================")
    for subset, report in reports.items():
        overall = report["overall"]
        print(
            f"{subset}: fit/calibration/evaluation engines="
            f"{len(report['evaluation_protocol']['model_fit_engine_ids'])}/"
            f"{report['evaluation_protocol']['calibration_engine_count']}/"
            f"{len(report['evaluation_protocol']['decision_evaluation_engine_ids'])}; "
            f"states={overall['decision_states']}, "
            f"recommendations={overall['state_counts_by_status'].get('recommendation', 0)}, "
            f"no-safe-action={overall['state_counts_by_status'].get('no_safe_action', 0)}, "
            f"abstentions={sum(v for k, v in overall['state_counts_by_status'].items() if k not in {'recommendation', 'no_safe_action'})}"
        )
        print(
            f"  candidate outcomes={overall['candidate_outcomes']}, "
            f"rejected={overall['rejected_candidates']}, "
            f"selected actions={overall['selected_action_counts']}"
        )
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"State-level CSV: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
