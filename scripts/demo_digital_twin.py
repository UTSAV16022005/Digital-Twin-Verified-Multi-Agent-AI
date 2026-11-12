"""Train and demonstrate the first C-MAPSS Digital Twin prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

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


def one_step_validation_error(
    model: DataDrivenTransitionModel,
    frame: pd.DataFrame,
) -> float:
    actual: list[float] = []
    predicted: list[float] = []
    for _, unit_frame in frame.groupby("unit_id", sort=True):
        unit_frame = unit_frame.sort_values("cycle")
        values = unit_frame[model.feature_columns].to_numpy(dtype=float)
        if len(values) >= 2:
            for index in range(len(values) - 1):
                actual.extend(values[index + 1])
                predicted.extend(model.predict_next(values[index]))
    return float(mean_absolute_error(actual, predicted))


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
    validation_mae = one_step_validation_error(transition_model, validation_scaled)
    twin = DigitalTwin(
        transition_model,
        features,
        rul_reference=125.0,
        max_sensor_z=5.0,
    )

    demo_unit = int(sorted(split.validation_units)[0])
    demo_frame = validation_scaled[validation_scaled["unit_id"] == demo_unit].sort_values(
        "cycle"
    )
    demo_window, _, demo_metadata = make_windows(demo_frame, features, window_size=30)
    if len(demo_window) == 0:
        raise RuntimeError("Demo unit did not have enough cycles for a 30-cycle window")
    demo_rul_by_cycle = (
        validation_frame[validation_frame["unit_id"] == demo_unit]
        .set_index("cycle")["rul"]
    )
    selected_index = len(demo_metadata) - 1
    current_rul = 0.0
    for candidate_index in range(len(demo_metadata) - 1, -1, -1):
        candidate_cycle = int(demo_metadata.iloc[candidate_index]["end_cycle"])
        candidate_rul = float(demo_rul_by_cycle.loc[candidate_cycle])
        if candidate_rul >= 30.0:
            selected_index = candidate_index
            current_rul = candidate_rul
            break
    current_window = demo_window[selected_index]
    current_meta = demo_metadata.iloc[selected_index]
    state = twin.estimate_state(
        current_window,
        unit_id=demo_unit,
        end_cycle=int(current_meta["end_cycle"]),
        predicted_rul=current_rul,
    )
    outcomes = twin.what_if(state, horizon=5)

    report = {
        "subset": "FD001",
        "component": "data-driven Digital Twin prototype",
        "train_engines": len(split.train_units),
        "validation_engines": len(split.validation_units),
        "features_used": features,
        "features_dropped_as_constant": dropped,
        "transition_model": "Ridge multi-output one-step telemetry model",
        "transition_model_alpha": 1.0,
        "validation_one_step_mae_standardized_units": validation_mae,
        "state_estimation": {
            "demo_unit": demo_unit,
            "end_cycle": state.cycle,
            "predicted_rul": state.predicted_rul,
            "health_index": state.health_index,
        },
        "action_assumptions": {name: vars(spec) for name, spec in twin.actions.items()},
        "what_if_outcomes": [outcome.as_dict() for outcome in outcomes],
        "limitations": [
            "C-MAPSS contains no recovery-action labels.",
            "Action effects are explicit prototype assumptions, not measured engine-control effects.",
            "Sensor states are normalized features, so the 5-sigma envelope is a generic constraint.",
        ],
    }
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(twin, models_dir / "fd001_digital_twin.joblib")
    (output_dir / "fd001_digital_twin_demo.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("FD001 Digital Twin prototype")
    print("============================")
    print("transition model: Ridge multi-output one-step telemetry model")
    print(f"validation one-step MAE: {validation_mae:.6f} standardized units")
    print(f"demo unit: {demo_unit}, state cycle: {state.cycle}, RUL: {state.predicted_rul:.0f}")
    print(f"current health index: {state.health_index:.3f}")
    print("what-if outcomes:")
    for outcome in outcomes:
        print(
            f"  {outcome.action}: RUL={outcome.predicted_state.predicted_rul:.1f}, "
            f"risk={outcome.predicted_failure_risk:.3f}, "
            f"uncertainty={outcome.model_uncertainty:.4f}, "
            f"safe={outcome.safe_under_constraints}"
        )
    print(f"saved model: {models_dir / 'fd001_digital_twin.joblib'}")
    print(f"saved report: {output_dir / 'fd001_digital_twin_demo.json'}")


if __name__ == "__main__":
    main()

