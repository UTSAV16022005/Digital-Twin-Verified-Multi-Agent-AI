"""Run Prognostics -> Digital Twin -> Recovery Planner without true RUL input."""

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
    load_telemetry,
    make_windows,
    split_engine_units,
    transform_features,
)
from src.digital_twin import DataDrivenTransitionModel, DigitalTwin
from src.prognostics import PrognosticsAgent
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
    prognostics = PrognosticsAgent.fit(train_frame, features)
    train_scaled = transform_features(train_frame, features, prognostics.scaler)
    validation_scaled = transform_features(validation_frame, features, prognostics.scaler)
    transition_model = DataDrivenTransitionModel(features, alpha=1.0).fit(train_scaled)
    twin = DigitalTwin(transition_model, features, rul_reference=125.0, max_sensor_z=5.0)

    demo_unit = int(sorted(split.validation_units)[0])
    raw_demo = validation_frame[validation_frame["unit_id"] == demo_unit].sort_values(
        "cycle"
    )
    prediction_frame = prognostics.predict_frame(raw_demo)
    if prediction_frame.empty:
        raise RuntimeError("No prognostics windows were generated")
    pre_failure_candidates = prediction_frame[
        prediction_frame["predicted_rul"] >= 30.0
    ]
    selected = (
        pre_failure_candidates.iloc[-1]
        if not pre_failure_candidates.empty
        else prediction_frame.iloc[-1]
    )
    window_frame = validation_scaled[validation_scaled["unit_id"] == demo_unit].sort_values(
        "cycle"
    )
    windows, _, metadata = make_windows(window_frame, features, 30)
    matching = metadata.index[metadata["end_cycle"] == selected["end_cycle"]]
    if len(matching) != 1:
        raise RuntimeError("Could not align prognostics output with a telemetry window")
    state = twin.estimate_state(
        windows[int(matching[0])],
        unit_id=demo_unit,
        end_cycle=int(selected["end_cycle"]),
        predicted_rul=float(selected["predicted_rul"]),
    )
    planner = RecoveryPlanningAgent(risk_budget=0.25)
    plan = planner.plan(twin, state, horizon=5)

    actual_rul_for_evaluation = float(selected["rul"])
    report = {
        "subset": "FD001",
        "pipeline": [
            "PrognosticsAgent",
            "DigitalTwin",
            "RecoveryPlanningAgent",
        ],
        "demo_unit": demo_unit,
        "end_cycle": state.cycle,
        "predicted_rul_used_by_twin": state.predicted_rul,
        "actual_rul_visible_only_for_evaluation": actual_rul_for_evaluation,
        "rul_uncertainty": float(selected["rul_uncertainty"]),
        "predicted_health_index": state.health_index,
        "recovery_plan": plan.as_dict(),
        "ground_truth_rule": "Actual RUL is recorded for evaluation only and is not passed into the Twin or planner.",
    }
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fd001_agent_pipeline_demo.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("FD001 agent pipeline")
    print("====================")
    print("PrognosticsAgent -> DigitalTwin -> RecoveryPlanningAgent")
    print(f"unit={demo_unit}, cycle={state.cycle}")
    print(f"predicted RUL used by Twin: {state.predicted_rul:.2f}")
    print(f"actual RUL used only for evaluation: {actual_rul_for_evaluation:.2f}")
    print(f"RUL uncertainty: {selected['rul_uncertainty']:.2f}")
    print(f"selected recovery action: {plan.selected_action}")
    print(f"saved report: {output_dir / 'fd001_agent_pipeline_demo.json'}")


if __name__ == "__main__":
    main()

