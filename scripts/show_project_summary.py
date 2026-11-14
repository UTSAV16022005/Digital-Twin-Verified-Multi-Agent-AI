"""Print a concise, presentation-friendly summary of saved project results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "generated"


def read_report(name: str) -> dict[str, Any]:
    path = REPORTS / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing report: {path}\n"
            "Run the project evaluation commands first; see README.md."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    monitor = read_report("fd001_lstm_monitoring.json")
    rul_001 = read_report("fd001_prognostics_agent.json")
    fd004 = read_report("fd004_pipeline_evaluation.json")
    twin = read_report("fd004_digital_twin_transition_cv.json")
    safety = read_report("safety_verifier_scenarios.json")
    closed = read_report("fd001_closed_loop_demo_lstm.json")
    sensitivity = read_report("action_assumption_sensitivity.json")

    monitor_test = monitor["test"]["fleet"]
    rul001_metrics = rul_001["test_terminal_metrics"]
    rul004_metrics = fd004["prognostics"]["test_terminal_metrics"]
    twin_metrics = twin["overall"]
    fd004_coverage = fd004["prognostics"]["test_terminal_engine_coverage"]
    fd001_action_shift = sensitivity["datasets"]["FD001"]["scenarios"][
        "no_action_benefit"
    ]["recommendation_changed_vs_configured_fraction"]
    fd004_action_shift = sensitivity["datasets"]["FD004"]["scenarios"][
        "no_action_benefit"
    ]["recommendation_changed_vs_configured_fraction"]

    print("=" * 68)
    print("DIGITAL TWIN + MULTI-AGENT AI | PROJECT RESULTS")
    print("C-MAPSS research prototype")
    print("=" * 68)

    print("\n1. LSTM ANOMALY MONITORING | FD001 official test split")
    print(
        f"   Detected degradation in {monitor_test['engine_detection_rate']:.1%} "
        f"of {monitor_test['n_engines_with_degraded_windows']} engines "
        "that had degraded windows"
    )
    print(
        f"   Healthy-window false alarm rate: "
        f"{pct(monitor_test['pooled_healthy_false_alarm_rate'])}"
    )
    print(f"   Median warning lead time: {monitor_test['median_lead_time']:.0f} cycles")

    print("\n2. REMAINING USEFUL LIFE (RUL) | terminal-engine test results")
    print(
        f"   FD001: MAE {rul001_metrics['mae']:.1f} cycles | "
        f"R-squared {rul001_metrics['r2']:.3f} | "
        f"95% interval coverage {pct(rul001_metrics['conformal_coverage_95'])}"
    )
    print(
        f"   FD004: MAE {rul004_metrics['mae']:.1f} cycles | "
        f"R-squared {rul004_metrics['r2']:.3f} | "
        f"lower-bound coverage {pct(rul004_metrics['one_sided_lower_bound_coverage'])}"
    )
    print(
        f"   FD004 scored {fd004_coverage['scored_engines']}/"
        f"{fd004_coverage['total_engines']} engines; shorter histories were not scored"
    )

    print("\n3. DIGITAL TWIN | FD004 five-fold engine-grouped evaluation")
    print(
        f"   One-step sensor RMSE: {twin_metrics['data_driven_rmse_normalized_units']:.3f} "
        f"(persistence baseline {twin_metrics['persistence_rmse_normalized_units']:.3f})"
    )
    print(
        f"   Evaluated on {twin_metrics['heldout_engine_count']} held-out engines; "
        "observed no-action transitions"
    )

    print("\n4. SAFETY RULE CHECKS")
    print(
        f"   {safety['scenario_count']}/{safety['scenario_count']} designed scenarios "
        f"matched expected decisions; {safety['invalid_config_check_count']}/"
        f"{safety['invalid_config_check_count']} invalid policies rejected"
    )

    print("\n5. ONE END-TO-END EXAMPLE | FD001")
    print(
        f"   Unit {closed['unit_id']}, cycle {closed['cycle']}: "
        f"{closed['diagnosis_state']} health band, "
        f"predicted RUL {closed['predicted_rul']:.1f} cycles"
    )
    print(
        f"   Recovery recommendation: {closed['selected_action'].replace('_', ' ')}"
    )
    print(
        f"   Planning lower bound: {closed['planning_rul_lower_95']:.1f} cycles; "
        f"actual RUL {closed['actual_rul_visible_only_for_evaluation']:.1f} cycles "
        "was used only for evaluation"
    )

    print("\nIMPORTANT LIMITS")
    print(
        "   - Health bands are not physical component-fault diagnosis.\n"
        "   - The Twin result evaluates next-step prediction on observed trajectories;\n"
        "     it does not validate recovery-action effects.\n"
        "   - Safety scenarios check configured rules; they are not real-world safety proof.\n"
        "   - Recovery planning is a constrained heuristic, not PPO."
    )
    print(
        f"   - Removing assumed action benefits changed recommendations in "
        f"{pct(fd001_action_shift)} of FD001 and {pct(fd004_action_shift)} of FD004 "
        "development decisions."
    )
    print("\nDetailed evidence: reports\\generated\\*.json")
    print("=" * 68)


if __name__ == "__main__":
    main()
