"""Compare RF30 with and without engine-cycle age on reserved FD004 engines.

Model fitting uses only the predeclared fit engines. Conformal uncertainty is
calibrated only on the separate calibration engines. Held-out decision engines
are used once for the final development comparison; official NASA test files are
never opened.
"""

from __future__ import annotations

from dataclasses import replace
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
from src.cmapss_data import (
    add_train_rul,
    load_telemetry,
    summarize_windows,
    transform_features,
)
from src.closed_loop import ClosedLoopAbstention, ClosedLoopOrchestrator
from src.prognostics import PrognosticsAgent


BOOTSTRAP_REPLICATES = 20_000


def _paired_engine_bootstrap(differences: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(BOOTSTRAP_REPLICATES, len(differences))
    )
    means = differences[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _decision_row(
    model_name: str,
    unit_id: int,
    cycle: int,
    actual_rul: float,
    result: Any,
    anomaly_score: float,
    diagnosis_state: str | None,
    diagnosis_confidence: float | None,
) -> dict[str, Any]:
    is_abstention = isinstance(result, ClosedLoopAbstention)
    safety_decisions = result.safety_decisions
    rejected = [
        (action, decision)
        for action, decision in safety_decisions.items()
        if not decision.approved
    ]
    if result.selected_action is not None:
        if not safety_decisions[result.selected_action].approved:
            raise RuntimeError("Orchestrator selected an action rejected by safety")
    return {
        "model": model_name,
        "subset": "FD004",
        "unit_id": unit_id,
        "cycle": cycle,
        "actual_rul_eval_only": actual_rul,
        "decision_status": result.decision_status,
        "selected_action": result.selected_action,
        "predicted_rul": None if is_abstention else float(result.predicted_rul),
        "planning_rul_lower_95": (
            None if is_abstention else float(result.planning_rul_lower_95)
        ),
        "absolute_rul_error": (
            None if is_abstention else abs(float(result.predicted_rul) - actual_rul)
        ),
        "candidate_count": len(safety_decisions),
        "approved_candidate_count": sum(
            decision.approved for decision in safety_decisions.values()
        ),
        "rejected_candidate_count": len(rejected),
        "rejected_actions": [action for action, _ in rejected],
        "rejection_reasons": [
            reason for _, decision in rejected for reason in decision.violations
        ],
        "anomaly_score": anomaly_score,
        "diagnosis_state": diagnosis_state,
        "diagnosis_confidence": diagnosis_confidence,
    }


def _paired_prediction_metrics(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    base = pd.DataFrame(baseline_rows)
    candidate = pd.DataFrame(candidate_rows)
    paired = base[
        ["unit_id", "cycle", "actual_rul_eval_only", "predicted_rul"]
    ].merge(
        candidate[["unit_id", "cycle", "actual_rul_eval_only", "predicted_rul"]],
        on=["unit_id", "cycle"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    paired = paired[paired["predicted_rul_baseline"].notna()].copy()
    if paired.empty or paired["predicted_rul_candidate"].isna().any():
        raise RuntimeError("Baseline and candidate prediction states did not align")
    if not np.allclose(
        paired["actual_rul_eval_only_baseline"],
        paired["actual_rul_eval_only_candidate"],
    ):
        raise RuntimeError("The paired model rows have inconsistent evaluation labels")

    paired["baseline_abs_error"] = (
        paired["predicted_rul_baseline"]
        - paired["actual_rul_eval_only_baseline"]
    ).abs()
    paired["candidate_abs_error"] = (
        paired["predicted_rul_candidate"]
        - paired["actual_rul_eval_only_candidate"]
    ).abs()
    per_engine = paired.groupby("unit_id", sort=True).agg(
        baseline_macro_mae=("baseline_abs_error", "mean"),
        candidate_macro_mae=("candidate_abs_error", "mean"),
    )
    differences = (
        per_engine["candidate_macro_mae"] - per_engine["baseline_macro_mae"]
    ).to_numpy(dtype=float)
    return {
        "paired_prediction_state_count": int(len(paired)),
        "paired_engine_count": int(len(per_engine)),
        "baseline_engine_macro_mae_cycles": float(
            per_engine["baseline_macro_mae"].mean()
        ),
        "cycle_feature_engine_macro_mae_cycles": float(
            per_engine["candidate_macro_mae"].mean()
        ),
        "paired_engine_macro_mae_difference_candidate_minus_baseline_cycles": float(
            differences.mean()
        ),
        "paired_engine_cluster_bootstrap_95_ci_difference_cycles": (
            _paired_engine_bootstrap(differences, SEED + 404)
        ),
        "engines_with_lower_macro_mae_with_cycle_feature": float(
            np.mean(differences < 0.0)
        ),
    }


def main() -> None:
    started = time.perf_counter()
    out_dir = ROOT / "reports" / "generated"
    protocol_file = out_dir / "closed_loop_development_evaluation.json"
    if not protocol_file.exists():
        raise FileNotFoundError(
            "Run scripts/evaluate_closed_loop_development.py first to establish the split"
        )
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))["datasets"][
        "FD004"
    ]["evaluation_protocol"]

    full_frame = add_train_rul(
        load_telemetry(ROOT / "data" / "raw" / "cmapss", "FD004", "train")
    )
    fit_units, calibration_units, evaluation_units = _split_units(full_frame)
    expected = (
        tuple(int(value) for value in protocol["model_fit_engine_ids"]),
        tuple(int(value) for value in protocol["uncertainty_calibration_engine_ids"]),
        tuple(int(value) for value in protocol["decision_evaluation_engine_ids"]),
    )
    if (fit_units, calibration_units, evaluation_units) != expected:
        raise RuntimeError("The held-out comparison split differs from the saved protocol")

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
        baseline_orchestrator,
    ) = _fit_components(fit_frame, calibration_frame)
    candidate_config = replace(
        baseline_agent.config, include_cycle_feature=True
    )
    candidate_agent = PrognosticsAgent.fit(
        fit_frame, features, config=candidate_config
    )
    candidate_agent.calibrate_uncertainty(calibration_frame)
    candidate_orchestrator = ClosedLoopOrchestrator(
        candidate_agent, twin, planner, verifier
    )

    # Evaluation labels are kept in a separate lookup and are only accessed
    # after both agents return their telemetry-only decisions.
    truth_lookup = {
        (int(row.unit_id), int(row.cycle)): float(row.rul)
        for row in heldout[["unit_id", "cycle", "rul"]].itertuples(index=False)
    }
    telemetry = heldout.drop(columns=["rul"])
    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for unit_id in sorted(evaluation_units):
        engine = telemetry[telemetry["unit_id"] == unit_id].sort_values("cycle")
        available_cycles = set(engine["cycle"].astype(int))
        for milestone in MILESTONES:
            if milestone not in available_cycles:
                continue
            prefix = engine[engine["cycle"] <= milestone].copy()
            anomaly_score = float(monitor.score(prefix).iloc[-1]["anomaly_score"])
            diagnosis_state = None
            diagnosis_confidence = None
            if milestone >= baseline_agent.config.window_size:
                normalized = transform_features(prefix, features, scaler)
                window = normalized[features].to_numpy(dtype=np.float32)[
                    -baseline_agent.config.window_size :
                ][None, :, :]
                probabilities = diagnosis.predict_proba(summarize_windows(window)).iloc[0]
                diagnosis_state = str(probabilities.idxmax())
                diagnosis_confidence = float(probabilities.max())

            base_result = baseline_orchestrator.step(
                prefix,
                anomaly_score=anomaly_score,
                diagnosis_state=diagnosis_state,
                diagnosis_confidence=diagnosis_confidence,
                horizon=HORIZON,
            )
            candidate_result = candidate_orchestrator.step(
                prefix,
                anomaly_score=anomaly_score,
                diagnosis_state=diagnosis_state,
                diagnosis_confidence=diagnosis_confidence,
                horizon=HORIZON,
            )
            actual_rul = truth_lookup[(unit_id, milestone)]
            baseline_rows.append(
                _decision_row(
                    "rf_summary_30", unit_id, milestone, actual_rul, base_result,
                    anomaly_score, diagnosis_state, diagnosis_confidence
                )
            )
            candidate_rows.append(
                _decision_row(
                    "rf_summary_30_plus_engine_cycle", unit_id, milestone,
                    actual_rul, candidate_result, anomaly_score, diagnosis_state,
                    diagnosis_confidence
                )
            )

    baseline_summary = _summarize_rows(
        baseline_rows, len(evaluation_units), SEED + 500
    )
    candidate_summary = _summarize_rows(
        candidate_rows, len(evaluation_units), SEED + 600
    )
    comparison = _paired_prediction_metrics(baseline_rows, candidate_rows)
    by_cycle: dict[str, Any] = {}
    for cycle in MILESTONES:
        by_cycle[str(cycle)] = {
            "baseline": _summarize_rows(
                [row for row in baseline_rows if row["cycle"] == cycle],
                len(evaluation_units), SEED + cycle,
            ),
            "cycle_feature": _summarize_rows(
                [row for row in candidate_rows if row["cycle"] == cycle],
                len(evaluation_units), SEED + cycle + 1,
            ),
        }

    report = {
        "experiment": "FD004 closed-loop holdout comparison: RF30 versus RF30 plus engine cycle",
        "source": "official C-MAPSS training trajectories only",
        "official_test_files_loaded": False,
        "model_selection_data": "fit-engine grouped cross-validation only",
        "evaluation_protocol": {
            "model_fit_engine_ids": list(fit_units),
            "uncertainty_calibration_engine_ids": list(calibration_units),
            "decision_evaluation_engine_ids": list(evaluation_units),
            "milestones_cycles": list(MILESTONES),
            "window_size": baseline_agent.config.window_size,
            "cycle_feature": "end_cycle / 100 appended to summarized telemetry window",
            "baseline_config": baseline_agent.config.as_dict(),
            "candidate_config": candidate_agent.config.as_dict(),
            "baseline_lower_bound_margin_95_cycles": baseline_agent.lower_bound_margin_95,
            "candidate_lower_bound_margin_95_cycles": candidate_agent.lower_bound_margin_95,
            "baseline_calibration_engines": baseline_agent.n_calibration_engines_,
            "candidate_calibration_engines": candidate_agent.n_calibration_engines_,
            "evaluation_label_access": "RUL is joined only after both telemetry-only decisions are returned.",
        },
        "overall": {
            "baseline": baseline_summary,
            "cycle_feature": candidate_summary,
            "paired_prediction_comparison": comparison,
        },
        "model_decision": (
            "Keep the cycle feature opt-in: the paired engine-level MAE confidence "
            "interval includes zero, and planning lower-bound empirical coverage "
            "on the held-out engines is below the nominal 95%."
        ),
        "by_cycle": by_cycle,
        "limitations": [
            "This is a single reserved-engine development holdout from the official training trajectories, not the official NASA test set.",
            "C-MAPSS contains no observed recovery-action outcomes; simulated actions and their effects remain Digital Twin assumptions.",
            "The cycle-age feature is useful only when elapsed engine cycles are available and meaningful at deployment.",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }

    json_path = out_dir / "fd004_cycle_age_holdout.json"
    csv_path = out_dir / "fd004_cycle_age_holdout_states.csv"
    rows = baseline_rows + candidate_rows
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("FD004 reserved-engine closed-loop comparison")
    print("============================================")
    print(f"baseline RF30: {baseline_summary['rul_metrics_on_decision_states']}")
    print(f"RF30 + cycle: {candidate_summary['rul_metrics_on_decision_states']}")
    print(f"paired comparison: {comparison}")
    print(
        "planning lower-bound coverage: "
        f"{baseline_summary['rul_metrics_on_decision_states']['planning_lower_bound_empirical_coverage']:.3f} baseline, "
        f"{candidate_summary['rul_metrics_on_decision_states']['planning_lower_bound_empirical_coverage']:.3f} cycle feature"
    )
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"State-level CSV: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
