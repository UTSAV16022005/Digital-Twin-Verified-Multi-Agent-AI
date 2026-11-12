"""Demonstrate Recovery Planning Agent ranking over Digital Twin outcomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cmapss_data import (
    FEATURE_COLUMNS,
    add_train_rul,
    constant_columns,
    fit_feature_scaler,
    load_telemetry,
    make_windows,
    split_engine_units,
    transform_features,
)
from src.digital_twin import DataDrivenTransitionModel, DigitalTwin
from src.recovery_planning import RecoveryPlanningAgent


def main() -> None:
    data_dir = Path("data/raw/cmapss")
    full_train = add_train_rul(load_telemetry(data_dir, "FD001", "train"))
    split = split_engine_units(full_train, validation_fraction=0.2, seed=42)
    train_frame = full_train[full_train["unit_id"].isin(split.train_units)].copy()
    validation_frame = full_train[
        full_train["unit_id"].isin(split.validation_units)
    ].copy()

    dropped = constant_columns(train_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]
    scaler = fit_feature_scaler(train_frame, features)
    train_scaled = transform_features(train_frame, features, scaler)
    validation_scaled = transform_features(validation_frame, features, scaler)
    transition_model = DataDrivenTransitionModel(features, alpha=1.0).fit(train_scaled)
    twin = DigitalTwin(transition_model, features, rul_reference=125.0, max_sensor_z=5.0)

    demo_unit = int(sorted(split.validation_units)[0])
    demo_frame = validation_scaled[validation_scaled["unit_id"] == demo_unit].sort_values(
        "cycle"
    )
    demo_windows, _, metadata = make_windows(demo_frame, features, window_size=30)
    rul_by_cycle = (
        validation_frame[validation_frame["unit_id"] == demo_unit]
        .set_index("cycle")["rul"]
    )
    selected_index = len(metadata) - 1
    for candidate_index in range(len(metadata) - 1, -1, -1):
        if float(rul_by_cycle.loc[int(metadata.iloc[candidate_index]["end_cycle"])]) >= 30:
            selected_index = candidate_index
            break
    selected_meta = metadata.iloc[selected_index]
    end_cycle = int(selected_meta["end_cycle"])
    current_rul = float(rul_by_cycle.loc[end_cycle])
    state = twin.estimate_state(
        demo_windows[selected_index],
        unit_id=demo_unit,
        end_cycle=end_cycle,
        predicted_rul=current_rul,
    )

    planner = RecoveryPlanningAgent(risk_budget=0.25)
    plan = planner.plan(twin, state, horizon=5)
    report = {
        "subset": "FD001",
        "component": "Recovery Planning Agent prototype",
        "demo_state": state.as_dict(),
        "planning_weights": vars(planner.weights),
        "risk_budget": planner.risk_budget,
        "plan": plan.as_dict(),
        "limitation": "Action effects are prototype assumptions because C-MAPSS has no recovery-action labels.",
    }
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fd001_recovery_planner_demo.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("FD001 Recovery Planning Agent prototype")
    print("========================================")
    print(f"state: unit={demo_unit}, cycle={end_cycle}, RUL={current_rul:.0f}")
    print(f"decision status: {plan.decision_status}")
    print(f"selected action: {plan.selected_action}")
    print(plan.rationale)
    print("ranked candidates:")
    for candidate in plan.ranked_candidates:
        print(
            f"  {candidate['action']}: score={candidate['objective_score']:.4f}, "
            f"risk={candidate['predicted_failure_risk']:.3f}, "
            f"predicted_rul={candidate['predicted_rul']:.1f}"
        )
    print("rejected candidates:")
    for candidate in plan.rejected_candidates:
        print(f"  {candidate['action']}: {candidate['violations']}")
    print(f"saved report: {output_dir / 'fd001_recovery_planner_demo.json'}")


if __name__ == "__main__":
    main()

