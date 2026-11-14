"""Measure how assumed Twin action effects change closed-loop recommendations.

This is a policy-sensitivity experiment on the existing development split. It
does not validate the counterfactual action effects against physical outcomes;
C-MAPSS has no such action/outcome labels.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.evaluate_closed_loop_development import (
    HORIZON,
    MILESTONES,
    _fit_components,
    _split_units,
)
from src.agent_coordinator import HealthAgentCoordinator
from src.closed_loop import ClosedLoopOrchestrator
from src.cmapss_data import add_train_rul, load_telemetry
from src.digital_twin import ActionSpec


SCENARIOS = {
    "no_action_benefit": {"transition_scale": 0.0, "degradation_scale": 0.0},
    "telemetry_effect_only": {"transition_scale": 1.0, "degradation_scale": 0.0},
    "degradation_effect_only": {"transition_scale": 0.0, "degradation_scale": 1.0},
    "configured_assumptions": {"transition_scale": 1.0, "degradation_scale": 1.0},
    "amplified_assumptions": {"transition_scale": 1.5, "degradation_scale": 1.5},
}


def scaled_action_specs(
    action_specs: dict[str, ActionSpec],
    transition_scale: float,
    degradation_scale: float,
) -> dict[str, ActionSpec]:
    """Scale claimed operating-action benefits around the continue baseline."""

    if transition_scale < 0 or degradation_scale < 0:
        raise ValueError("action-effect scales must be non-negative")
    scaled: dict[str, ActionSpec] = {}
    for name, spec in action_specs.items():
        if not spec.keeps_operating or name == "continue_operation":
            scaled[name] = spec
            continue
        transition = 1.0 + transition_scale * (spec.transition_multiplier - 1.0)
        degradation = 1.0 + degradation_scale * (spec.degradation_multiplier - 1.0)
        scaled[name] = replace(
            spec,
            transition_multiplier=float(min(1.0, max(0.0, transition))),
            degradation_multiplier=float(min(1.0, max(0.0, degradation))),
        )
    return scaled


def _evaluate_subset(subset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_dir = ROOT / "data" / "raw" / "cmapss"
    full = add_train_rul(load_telemetry(data_dir, subset, "train"))
    fit_units, calibration_units, evaluation_units = _split_units(full)
    fit_frame = full[full["unit_id"].isin(fit_units)].copy()
    calibration_frame = full[full["unit_id"].isin(calibration_units)].copy()
    evaluation = full[full["unit_id"].isin(evaluation_units)].drop(columns="rul")

    (
        features,
        diagnosis_scaler,
        monitor,
        diagnosis,
        prognostics,
        twin,
        planner,
        verifier,
        _,
    ) = _fit_components(fit_frame, calibration_frame)
    orchestrator = ClosedLoopOrchestrator(prognostics, twin, planner, verifier)
    coordinator = HealthAgentCoordinator(
        monitor, diagnosis, orchestrator, diagnosis_scaler=diagnosis_scaler
    )
    original_specs = dict(twin.actions)
    rows: list[dict[str, Any]] = []

    configured = SCENARIOS["configured_assumptions"]
    configured_specs = scaled_action_specs(
        original_specs,
        configured["transition_scale"],
        configured["degradation_scale"],
    )
    for unit_id in sorted(evaluation_units):
        engine = evaluation[evaluation["unit_id"] == unit_id].sort_values("cycle")
        observed_cycles = set(engine["cycle"].astype(int))
        for milestone in MILESTONES:
            if milestone not in observed_cycles:
                continue
            # Monitoring, diagnosis, and prognostics depend only on observed
            # telemetry and are identical across action-effect scenarios.
            twin.actions = configured_specs
            step = coordinator.step(
                engine[engine["cycle"] <= milestone].copy(), horizon=HORIZON
            )
            configured_outcome = step.outcome
            for scenario, scales in SCENARIOS.items():
                twin.actions = scaled_action_specs(
                    original_specs,
                    scales["transition_scale"],
                    scales["degradation_scale"],
                )
                if scenario == "configured_assumptions":
                    outcome = configured_outcome
                    selected_action = outcome.selected_action
                    decision_status = outcome.decision_status
                    decisions = outcome.safety_decisions
                elif configured_outcome.twin_state is None:
                    # The pipeline has abstained before the Twin; changing
                    # action assumptions cannot change that upstream result.
                    selected_action = None
                    decision_status = configured_outcome.decision_status
                    decisions = {}
                else:
                    outcomes = twin.what_if(
                        configured_outcome.twin_state, horizon=HORIZON
                    )
                    decisions = verifier.verify_all(outcomes)
                    approved_outcomes = [
                        candidate
                        for candidate in outcomes
                        if decisions[candidate.action].approved
                    ]
                    approved_outcomes.sort(key=planner.score)
                    selected_action = (
                        approved_outcomes[0].action if approved_outcomes else None
                    )
                    decision_status = (
                        "recommendation" if selected_action is not None else "no_safe_action"
                    )
                approved = sum(
                    decision.approved
                    for decision in decisions.values()
                )
                rows.append(
                    {
                        "subset": subset,
                        "scenario": scenario,
                        "transition_effect_scale": scales["transition_scale"],
                        "degradation_effect_scale": scales["degradation_scale"],
                        "unit_id": int(unit_id),
                        "cycle": int(milestone),
                        "decision_status": decision_status,
                        "selected_action": selected_action,
                        "candidate_count": len(decisions),
                        "approved_candidate_count": approved,
                        "approved_candidate_fraction": (
                            approved / len(decisions) if decisions else None
                        ),
                    }
                )

    state_rows = pd.DataFrame(rows)
    configured = state_rows[
        state_rows["scenario"] == "configured_assumptions"
    ][["unit_id", "cycle", "selected_action"]].rename(
        columns={"selected_action": "configured_selected_action"}
    )
    state_rows = state_rows.merge(
        configured, on=["unit_id", "cycle"], how="left", validate="many_to_one"
    )
    report_scenarios: dict[str, Any] = {}
    for scenario, group in state_rows.groupby("scenario", sort=False):
        action_counts = group["selected_action"].fillna("no_action").value_counts()
        report_scenarios[scenario] = {
            **SCENARIOS[scenario],
            "decision_count": int(len(group)),
            "selected_action_counts": {
                str(action): int(count) for action, count in action_counts.items()
            },
            "selected_action_fractions": {
                str(action): float(count / len(group))
                for action, count in action_counts.items()
            },
            "no_safe_action_fraction": float(
                group["decision_status"].eq("no_safe_action").mean()
            ),
            "mean_approved_candidate_fraction": (
                float(group["approved_candidate_fraction"].mean())
                if group["approved_candidate_fraction"].notna().any()
                else None
            ),
            "recommendation_changed_vs_configured_fraction": float(
                group["selected_action"].fillna("no_action").ne(
                    group["configured_selected_action"].fillna("no_action")
                ).mean()
            ),
        }

    report = {
        "experiment": f"{subset} Digital Twin action-assumption sensitivity",
        "evaluation_status": (
            "development sensitivity only; reuses the previously inspected fixed engine split"
        ),
        "official_test_files_loaded": False,
        "protocol": {
            "fit_engine_ids": list(fit_units),
            "calibration_engine_ids": list(calibration_units),
            "decision_evaluation_engine_ids": list(evaluation_units),
            "decision_milestones_cycles": list(MILESTONES),
            "prognostics_features": list(features),
            "telemetry_label_rule": (
                "Coordinator receives telemetry only; C-MAPSS RUL labels are not sent to it."
            ),
            "action_rule": (
                "Scales interpolate each operating recovery multiplier between the continue-operation value (1.0) and the registered action assumption; mission costs and controlled shutdown are fixed."
            ),
            "scenarios": SCENARIOS,
        },
        "scenarios": report_scenarios,
        "limitations": [
            "C-MAPSS has no measured action/outcome pairs; no scenario establishes real action efficacy.",
            "Sensitivity results show how the configured planner responds to assumptions, not causal recovery benefits.",
            "The evaluation split has been inspected in earlier development work and is not an independent final test.",
            "Controlled shutdown remains a frozen zero-risk fallback by design.",
        ],
    }
    return report, rows


def main() -> None:
    started = time.perf_counter()
    reports: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for subset in ("FD001", "FD004"):
        print(f"evaluating {subset} action sensitivity", flush=True)
        report, rows = _evaluate_subset(subset)
        reports[subset] = report
        all_rows.extend(rows)
        print(f"completed {subset} action sensitivity")
        for scenario, metrics in report["scenarios"].items():
            print(
                f"  {scenario}: actions={metrics['selected_action_counts']}, "
                f"changed-vs-configured={metrics['recommendation_changed_vs_configured_fraction']:.3f}"
            )

    report = {
        "experiment": "Digital Twin action-assumption sensitivity",
        "datasets": reports,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    out_dir = ROOT / "reports" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "action_assumption_sensitivity.json"
    csv_path = out_dir / "action_assumption_sensitivity_states.csv"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"State-level results: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
