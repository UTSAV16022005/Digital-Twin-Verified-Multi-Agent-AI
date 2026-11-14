"""Evaluate deterministic Safety Verifier rules on fixed synthetic scenarios.

This checks that the gate enforces its configured rules. It does not estimate
real-world safety, validate the Digital Twin against an aircraft, or prove that
any recovery action is physically safe.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.digital_twin import TransitionOutcome, TwinState
from src.safety_verification import SafetyConfig, SafetyVerifier


DEFAULT_POLICY = {
    "max_failure_risk": 0.25,
    "max_model_uncertainty": 6.0,
    "max_projected_drift": 5.0,
    "allow_controlled_shutdown": True,
}


SCENARIOS: list[dict[str, Any]] = [
    {"id": "nominal_continue", "description": "Nominal operating outcome", "expected": True},
    {
        "id": "at_all_numeric_limits",
        "description": "Risk, uncertainty, and drift exactly at their allowed limits",
        "expected": True,
        "outcome": {"predicted_failure_risk": 0.25, "model_uncertainty": 6.0, "projected_drift": 5.0},
    },
    {
        "id": "risk_over_limit",
        "description": "Failure risk just above the configured limit",
        "expected": False,
        "outcome": {"predicted_failure_risk": 0.2501},
    },
    {
        "id": "risk_not_finite",
        "description": "Failure risk is NaN",
        "expected": False,
        "outcome": {"predicted_failure_risk": float("nan")},
    },
    {
        "id": "uncertainty_over_limit",
        "description": "Model uncertainty above its configured limit",
        "expected": False,
        "outcome": {"model_uncertainty": 6.01},
    },
    {
        "id": "negative_uncertainty",
        "description": "Negative model uncertainty",
        "expected": False,
        "outcome": {"model_uncertainty": -0.01},
    },
    {
        "id": "drift_over_limit",
        "description": "Projected drift above its configured limit",
        "expected": False,
        "outcome": {"projected_drift": 5.01},
    },
    {
        "id": "twin_constraint_violation",
        "description": "Twin reports a named constraint violation",
        "expected": False,
        "outcome": {
            "safe_under_constraints": False,
            "violations": ["predicted_failure_within_simulation_horizon"],
        },
    },
    {
        "id": "unsafe_flag_without_reason",
        "description": "Twin marks the outcome unsafe but supplies no violation string",
        "expected": False,
        "outcome": {"safe_under_constraints": False},
    },
    {
        "id": "malformed_twin_safety_flag",
        "description": "Twin safety flag is a string instead of a boolean",
        "expected": False,
        "outcome": {"safe_under_constraints": "false"},
    },
    {
        "id": "malformed_twin_violations",
        "description": "Twin violations field is not a list of non-empty strings",
        "expected": False,
        "outcome": {"violations": [""]},
    },
    {
        "id": "non_finite_sensor_value",
        "description": "Predicted sensor state contains infinity",
        "expected": False,
        "state": {"feature_values": [0.1, float("inf")]},
    },
    {
        "id": "non_finite_sensor_slope",
        "description": "Predicted sensor slope contains NaN",
        "expected": False,
        "state": {"feature_slopes": [0.0, float("nan")]},
    },
    {
        "id": "mismatched_state_shape",
        "description": "Sensor value count does not match feature names and slopes",
        "expected": False,
        "state": {"feature_values": [0.1]},
    },
    {
        "id": "negative_predicted_rul",
        "description": "Predicted remaining useful life is negative",
        "expected": False,
        "state": {"predicted_rul": -1.0},
    },
    {
        "id": "health_index_out_of_range",
        "description": "Predicted health index is above one",
        "expected": False,
        "state": {"health_index": 1.01},
    },
    {
        "id": "invalid_mission_cost",
        "description": "Mission cost is negative",
        "expected": False,
        "outcome": {"mission_cost": -0.1},
    },
    {
        "id": "invalid_horizon",
        "description": "Simulation horizon is zero",
        "expected": False,
        "outcome": {"horizon": 0},
    },
    {
        "id": "invalid_action_name",
        "description": "Action name is empty",
        "expected": False,
        "outcome": {"action": ""},
    },
    {
        "id": "shutdown_allowed",
        "description": "Controlled shutdown under a policy that permits it",
        "expected": True,
        "outcome": {"action": "controlled_shutdown"},
    },
    {
        "id": "shutdown_disallowed",
        "description": "Controlled shutdown under a policy that forbids it",
        "expected": False,
        "outcome": {"action": "controlled_shutdown"},
        "config": {"allow_controlled_shutdown": False},
    },
]


INVALID_CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("risk_limit_above_one", {"max_failure_risk": 1.01}),
    ("risk_limit_nan", {"max_failure_risk": float("nan")}),
    ("negative_uncertainty_limit", {"max_model_uncertainty": -0.1}),
    ("infinite_drift_limit", {"max_projected_drift": float("inf")}),
    ("non_boolean_shutdown_policy", {"allow_controlled_shutdown": "false"}),
]


def _build_outcome(scenario: dict[str, Any]) -> TransitionOutcome:
    state = TwinState(
        unit_id=1,
        cycle=100,
        feature_columns=["sensor_a", "sensor_b"],
        feature_values=np.array([0.1, -0.2], dtype=np.float64),
        feature_slopes=np.array([0.01, -0.02], dtype=np.float64),
        predicted_rul=50.0,
        health_index=0.6,
        anomaly_score=0.1,
    )
    for name, value in scenario.get("state", {}).items():
        if name in {"feature_values", "feature_slopes"}:
            value = np.asarray(value, dtype=np.float64)
        setattr(state, name, value)

    values: dict[str, Any] = {
        "action": "continue_operation",
        "horizon": 5,
        "predicted_state": state,
        "predicted_failure_risk": 0.10,
        "model_uncertainty": 1.0,
        "projected_drift": 1.0,
        "mission_cost": 0.1,
        "safe_under_constraints": True,
        "violations": [],
    }
    values.update(scenario.get("outcome", {}))
    return TransitionOutcome(**values)


def _check_invalid_configurations() -> list[dict[str, Any]]:
    results = []
    for name, override in INVALID_CONFIGS:
        config_values = {**DEFAULT_POLICY, **override}
        try:
            SafetyConfig(**config_values)
        except (TypeError, ValueError) as error:
            results.append(
                {"id": name, "rejected": True, "validation_message": str(error)}
            )
        else:
            results.append(
                {"id": name, "rejected": False, "validation_message": ""}
            )
    return results


def main() -> None:
    scenario_results: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        config_values = {**DEFAULT_POLICY, **scenario.get("config", {})}
        decision = SafetyVerifier(SafetyConfig(**config_values)).verify(
            _build_outcome(scenario)
        )
        expected = bool(scenario["expected"])
        scenario_results.append(
            {
                "id": scenario["id"],
                "description": scenario["description"],
                "expected_approved": expected,
                "actual_approved": decision.approved,
                "matched_expectation": decision.approved == expected,
                "violations": decision.violations,
                "margins": decision.margins,
            }
        )

    invalid_config_results = _check_invalid_configurations()
    all_scenarios_match = all(
        result["matched_expectation"] for result in scenario_results
    )
    all_invalid_configs_rejected = all(
        result["rejected"] for result in invalid_config_results
    )

    report = {
        "evaluation_type": "deterministic synthetic rule-gate scenarios",
        "limitations": [
            "Scenario counts are not empirical safety rates or probabilities.",
            "The evaluation does not validate the Digital Twin against a physical engine or aircraft.",
            "It does not prove that any recovery action is physically safe.",
        ],
        "policy": DEFAULT_POLICY,
        "scenario_count": len(scenario_results),
        "approved_scenario_count": sum(
            result["actual_approved"] for result in scenario_results
        ),
        "rejected_scenario_count": sum(
            not result["actual_approved"] for result in scenario_results
        ),
        "all_scenarios_matched_expectations": all_scenarios_match,
        "invalid_config_check_count": len(invalid_config_results),
        "all_invalid_configs_rejected": all_invalid_configs_rejected,
        "scenarios": scenario_results,
        "invalid_config_checks": invalid_config_results,
    }

    output_dir = ROOT / "reports" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "safety_verifier_scenarios.json"
    csv_path = output_dir / "safety_verifier_scenarios.csv"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "id",
                "description",
                "expected_approved",
                "actual_approved",
                "matched_expectation",
                "violations",
                "risk_margin",
                "uncertainty_margin",
                "drift_margin",
            ],
        )
        writer.writeheader()
        for result in scenario_results:
            writer.writerow(
                {
                    **{
                        key: result[key]
                        for key in (
                            "id",
                            "description",
                            "expected_approved",
                            "actual_approved",
                            "matched_expectation",
                        )
                    },
                    "violations": ";".join(result["violations"]),
                    "risk_margin": result["margins"]["failure_risk"],
                    "uncertainty_margin": result["margins"]["model_uncertainty"],
                    "drift_margin": result["margins"]["projected_drift"],
                }
            )

    print("Safety Verifier synthetic scenario evaluation")
    print("==============================================")
    print(
        f"scenarios matched: {sum(r['matched_expectation'] for r in scenario_results)}"
        f"/{len(scenario_results)}"
    )
    print(
        "invalid policies rejected: "
        f"{sum(r['rejected'] for r in invalid_config_results)}"
        f"/{len(invalid_config_results)}"
    )
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"CSV report: {csv_path.relative_to(ROOT)}")
    if not all_scenarios_match or not all_invalid_configs_rejected:
        raise SystemExit("Safety Verifier scenario evaluation found an unexpected result")


if __name__ == "__main__":
    main()
