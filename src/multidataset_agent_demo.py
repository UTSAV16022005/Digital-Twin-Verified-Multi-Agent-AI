"""Dataset-aware, artifact-backed terminal demo for the existing agent models.

The router keeps each dataset's targets and metadata outside prediction inputs.
The C-MAPSS Digital Twin and Recovery Planner are the existing configured
prototype; NASA and PHM have separate, narrower workflows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.cmapss_data import add_train_rul, load_telemetry, transform_features
from src.data_adapters import DatasetBundle, load_phm_bundle
from src.digital_twin import DataDrivenTransitionModel, DigitalTwin
from src.nasa_turbofan_ds2 import (
    _read_audit,
    load_rich_cycle_summaries,
)
from src.prognostics import PrognosticsAgent, PrognosticsConfig
from src.recovery_planning import RecoveryPlanningAgent


ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOT = ROOT / "models" / "generated" / "evaluation_quality_20261115"
EVALUATION_REPORTS = ROOT / "reports" / "generated" / "evaluation_quality_20261115"
DEMO_MODEL_ROOT = ROOT / "models" / "generated" / "multidataset_agent_demo"
DEFAULT_TRACE_ROOT = ROOT / "reports" / "generated" / "multidataset_agent_demo"

SUPPORTED_ROLES: dict[str, tuple[str, ...]] = {
    "cmapss": ("prognostics", "digital_twin", "recovery_planner"),
    "nasa_turbofan_ds2": ("prognostics",),
    "phm2024": ("faulty_classifier", "torque_margin_regressor"),
}

ROLE_NOT_SUPPORTED_REASONS = {
    "cmapss": {
        "faulty_classifier": "This demo uses the evaluated C-MAPSS RUL artifact; no matching condition-classifier artifact is registered here.",
        "torque_margin_regressor": "Torque margin is not a C-MAPSS target.",
        "sequence_agent": "The selected C-MAPSS artifact is a window-summary RUL model, not the separate sequence-agent role.",
        "monitoring": "No matching saved monitor artifact is part of this artifact-backed demo route.",
        "diagnosis": "No matching saved diagnosis artifact is part of this artifact-backed demo route.",
        "safety_verifier": "The compact three-stage C-MAPSS demo does not run the separate safety-verification role.",
    },
    "nasa_turbofan_ds2": {
        "digital_twin": "The C-MAPSS Twin has not been validated on NASA Simulation-2 data.",
        "recovery_planner": "The C-MAPSS Recovery Planner has not been validated on NASA Simulation-2 data.",
        "faulty_classifier": "No evaluated NASA faulty-classification model is available.",
        "torque_margin_regressor": "No evaluated NASA torque-margin model is available.",
        "sequence_agent": "The evaluated NASA artifact is the random-forest window-summary RUL model, not a sequence-agent model.",
        "monitoring": "No evaluated NASA monitoring artifact is available.",
        "diagnosis": "No evaluated NASA diagnosis artifact is available.",
    },
    "phm2024": {
        "prognostics": "PHM 2024 has no evaluated RUL model in this project route.",
        "sequence_agent": "PHM 2024 is row-oriented here and has no evaluated sequence model.",
        "digital_twin": "PHM rows are not engine trajectories; the C-MAPSS Twin is unsupported for this dataset.",
        "recovery_planner": "PHM rows are not engine trajectories; no PHM action policy is validated.",
        "monitoring": "No evaluated PHM monitoring agent is available.",
        "diagnosis": "The PHM workflow uses its faulty classifier, not the C-MAPSS diagnosis agent.",
        "safety_verifier": "No PHM action policy or safety-verification route is defined.",
    },
}

PROTECTED_COLUMNS = {"rul", "faulty", "trq_margin", "unit_id", "cycle", "id"}


def role_route(dataset: str, role: str) -> dict[str, Any]:
    """Describe a supported handoff or return an explicit unsupported result."""

    if dataset not in SUPPORTED_ROLES:
        raise ValueError(f"Unknown dataset {dataset!r}")
    if role in SUPPORTED_ROLES[dataset]:
        return {"status": "supported", "dataset": dataset, "role": role}
    reason = ROLE_NOT_SUPPORTED_REASONS[dataset].get(
        role, f"The {role} role is not registered for {dataset}."
    )
    return {
        "status": "not_supported",
        "dataset": dataset,
        "role": role,
        "reason": reason,
    }


def phm_prediction_only_bundle(bundle: DatasetBundle, row_index: int) -> DatasetBundle:
    """Build a one-row PHM inference object with labels inaccessible to a model."""

    if bundle.dataset_name != "phm2024":
        raise ValueError("PHM prediction bundles require PHM source data")
    if not 0 <= int(row_index) < len(bundle.X):
        raise IndexError("PHM sample row is outside the loaded split")
    X = bundle.X.iloc[[int(row_index)]].reset_index(drop=True).copy()
    metadata = bundle.metadata.iloc[[int(row_index)]].reset_index(drop=True).copy()
    if "id" in X or PROTECTED_COLUMNS.intersection(X.columns):
        raise ValueError("PHM prediction features contain metadata or a target")
    return DatasetBundle(
        dataset_name=bundle.dataset_name,
        split=bundle.split,
        task_type=bundle.task_type,
        X=X,
        y=None,
        metadata=metadata,
        feature_names=tuple(X.columns),
        source_to_model_mapping=bundle.source_to_model_mapping,
    )


def _artifact_version(path: Path, source: str) -> dict[str, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    resolved_path = path.resolve()
    try:
        artifact_path = str(resolved_path.relative_to(ROOT))
    except ValueError:
        artifact_path = str(resolved_path)
    return {
        "source": source,
        "artifact": artifact_path,
        "sha256": digest,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _make_result(
    *,
    dataset: str,
    task: str,
    metadata: dict[str, Any],
    prediction: dict[str, Any],
    model_version: dict[str, str],
    downstream_roles_allowed: list[str],
    explanation: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "task": task,
        "metadata": metadata,
        "prediction": prediction,
        "model_version": model_version,
        "downstream_roles_allowed": downstream_roles_allowed,
        "explanation": explanation,
    }


def _not_supported_trace(dataset: str, task: str, role: str) -> dict[str, Any]:
    route = role_route(dataset, role)
    return {
        "schema_version": 1,
        "dataset": dataset,
        "task": task,
        "metadata": {"status": "unavailable", "reason": "Role rejected before loading a dataset example."},
        "roles_run": [],
        "roles_skipped": [route],
        "agent_results": [],
        "predictions": [],
        "action_advisory": {
            "status": "not_supported",
            "source": "none",
            "human_review_required": True,
            "reason": "No action role ran.",
        },
        "ground_truth": {"status": "not_read", "reason": "No prediction was made."},
    }


def _validate_features(features: list[str], *, context: str) -> None:
    forbidden = sorted(set(features).intersection(PROTECTED_COLUMNS))
    if forbidden:
        raise ValueError(f"{context} model feature list contains protected fields: {forbidden}")


def cmapss_prediction_only_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove labels before an inference frame is handed to a C-MAPSS model."""

    return frame.drop(columns=["rul"], errors="ignore").copy()


def nasa_prediction_only_frame(
    frame: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Keep official window metadata plus approved rich sensor features only."""

    _validate_features(feature_columns, context="NASA Simulation-2")
    missing = sorted(set(["unit_id", "cycle", *feature_columns]) - set(frame.columns))
    if missing:
        raise KeyError(f"NASA inference frame is missing fields: {missing}")
    result = frame.loc[:, ["unit_id", "cycle", *feature_columns]].copy()
    if PROTECTED_COLUMNS.intersection(result.columns) - {"unit_id", "cycle"}:
        raise ValueError("NASA target or sample metadata leaked into inference features")
    return result


def _cmapss_paths(
    subset: str,
    evaluation_model_root: Path,
    evaluation_report_root: Path,
) -> tuple[Path, Path]:
    normalized = subset.upper()
    if normalized not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError("C-MAPSS subset must be FD001, FD002, FD003, or FD004")
    return (
        evaluation_model_root / "cmapss" / normalized / f"{normalized.lower()}_prognostics_agent.joblib",
        evaluation_report_root / "cmapss" / normalized / "results.json",
    )


def _load_or_fit_cmapss_twin(
    *,
    subset: str,
    data_root: Path,
    agent: PrognosticsAgent,
    train_units: list[int],
    demo_model_root: Path,
) -> tuple[DataDrivenTransitionModel, Path, str]:
    """Load a cached prototype transition model or fit it on saved fit units.

    The repository's documented C-MAPSS demo fits this transition model on the
    fit-unit trajectories. No validation or official test trajectory is used.
    """

    model_path = demo_model_root / "cmapss" / subset / "digital_twin_transition.joblib"
    if model_path.exists():
        payload = joblib.load(model_path)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("model"), DataDrivenTransitionModel
        ):
            raise TypeError(f"Invalid C-MAPSS Twin artifact: {model_path}")
        expected = {
            "subset": subset,
            "fit_unit_ids": sorted(int(value) for value in train_units),
            "feature_columns": list(agent.feature_columns),
            "training_source": "train split only",
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"Cached Twin artifact metadata mismatch for {key}")
        return payload["model"], model_path, "cached configured C-MAPSS prototype artifact"

    raw_training = load_telemetry(data_root / "cmapss", subset, "train")
    fit_frame = raw_training[raw_training["unit_id"].isin(train_units)].copy()
    if not set(fit_frame["unit_id"].astype(int).unique()) == set(train_units):
        raise ValueError("C-MAPSS Twin fit split does not match the saved evaluation report")
    scaled_training = transform_features(
        fit_frame, agent.feature_columns, agent.scaler
    )
    transition = DataDrivenTransitionModel(agent.feature_columns, alpha=1.0).fit(
        scaled_training
    )
    payload = {
        "model": transition,
        "subset": subset,
        "fit_unit_ids": sorted(int(value) for value in train_units),
        "feature_columns": list(agent.feature_columns),
        "training_source": "train split only",
        "purpose": "assumption-based Digital Twin demo; not an action-effect validation artifact",
        "training_path": "existing scripts/demo_agent_pipeline.py transition-model fitting step",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path, compress=3)
    return transition, model_path, "created from fit units using the existing C-MAPSS demo training step"


def _run_cmapss(
    *,
    subset: str,
    data_root: Path,
    unit_id: int | None,
    cycle: int,
    requested_role: str | None,
    evaluation_model_root: Path,
    evaluation_report_root: Path,
) -> dict[str, Any]:
    model_path, report_path = _cmapss_paths(
        subset, evaluation_model_root, evaluation_report_root
    )
    if not model_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            f"Evaluated C-MAPSS artifact or report is missing for {subset}: {model_path}"
        )
    evaluation_report = json.loads(report_path.read_text(encoding="utf-8"))
    agent = PrognosticsAgent.load(model_path)
    _validate_features(agent.feature_columns, context=f"C-MAPSS {subset}")
    if agent.config.model_type != "random_forest":
        raise ValueError(f"Expected saved evaluated RF model for {subset}")

    train_units = [int(value) for value in evaluation_report["splits"]["train_unit_ids"]]
    validation_units = [
        int(value) for value in evaluation_report["splits"]["validation_unit_ids"]
    ]
    if set(train_units).intersection(validation_units):
        raise ValueError("Saved C-MAPSS fit and validation unit IDs overlap")
    if requested_role is not None and role_route("cmapss", requested_role)["status"] != "supported":
        return _not_supported_trace("cmapss", "remaining_useful_life_regression", requested_role)

    labeled_training = add_train_rul(
        load_telemetry(data_root / "cmapss", subset, "train")
    )
    candidates = labeled_training[labeled_training["unit_id"].isin(validation_units)]
    available_validation_units = sorted(int(value) for value in candidates["unit_id"].unique())
    selected_unit = int(unit_id) if unit_id is not None else available_validation_units[0]
    if selected_unit not in set(available_validation_units):
        raise ValueError(
            f"unit_id must be from the saved validation split {available_validation_units}"
        )
    full_unit = candidates[candidates["unit_id"] == selected_unit].sort_values("cycle")
    if cycle < 1 or cycle > int(full_unit["cycle"].max()):
        raise ValueError(
            f"cycle must be within this validation unit's observed history 1..{int(full_unit['cycle'].max())}"
        )
    prefix_labeled = full_unit[full_unit["cycle"] <= cycle].copy()
    # This is the object handed to prognostics: the RUL column is removed.
    telemetry = cmapss_prediction_only_frame(prefix_labeled)
    predictions = agent.predict_current_available_history(telemetry)
    if predictions.empty:
        raise RuntimeError("C-MAPSS prognostics could not make a prediction")
    prediction_row = predictions.iloc[-1]
    actual_cycle = int(prediction_row["end_cycle"])
    predicted_rul = float(prediction_row["predicted_rul"])

    roles_run = ["PrognosticsAgent"]
    roles_skipped: list[dict[str, Any]] = []
    if requested_role == "prognostics":
        run_twin = run_planner = False
    else:
        run_twin = requested_role in {None, "digital_twin", "recovery_planner"}
        run_planner = requested_role in {None, "recovery_planner"}

    twin_state: dict[str, Any] | None = None
    plan_output: dict[str, Any] | None = None
    twin_model_version: dict[str, str] | None = None
    advisory: dict[str, Any]
    if run_twin:
        transition, twin_path, twin_artifact_source = _load_or_fit_cmapss_twin(
            subset=subset,
            data_root=data_root,
            agent=agent,
            train_units=train_units,
            demo_model_root=DEMO_MODEL_ROOT,
        )
        twin = DigitalTwin(
            transition,
            agent.feature_columns,
            rul_reference=agent.config.rul_reference,
            max_sensor_z=5.0,
        )
        scaled_prefix = transform_features(telemetry, agent.feature_columns, agent.scaler)
        twin_window = scaled_prefix.tail(agent.config.window_size)[
            agent.feature_columns
        ].to_numpy(dtype=np.float32)
        if len(twin_window) < 2:
            raise ValueError("Digital Twin needs at least two observed cycles")
        state = twin.estimate_state(
            twin_window,
            unit_id=selected_unit,
            end_cycle=actual_cycle,
            predicted_rul=predicted_rul,
        )
        twin_state = {
            "cycle": actual_cycle,
            "predicted_health_index": float(state.health_index),
            "state_summary": "normalized current sensor values and slopes from the observed history",
            "model_version": _artifact_version(twin_path, twin_artifact_source),
        }
        twin_model_version = twin_state["model_version"]
        roles_run.append("DigitalTwin")
        if run_planner:
            planner = RecoveryPlanningAgent(risk_budget=0.25)
            plan = planner.plan(twin, state, horizon=5)
            plan_output = {
                "selected_action": plan.selected_action,
                "decision_status": plan.decision_status,
                "simulated_candidate_count": len(plan.ranked_candidates) + len(plan.rejected_candidates),
                "planning_basis": "configured ranking of Twin-simulated candidates; software constraints are prototype checks, not physical safety validation",
            }
            roles_run.append("RecoveryPlanningAgent")
            advisory = {
                "status": "advisory",
                "action": str(plan.selected_action),
                "source": "existing configured RecoveryPlanningAgent heuristic using the C-MAPSS DigitalTwin prototype",
                "policy": {
                    "type": "explicitly configured rule-based candidate ranking",
                    "risk_budget": 0.25,
                    "simulation_horizon_cycles": 5,
                },
                "human_review_required": True,
                "limitations": [
                    "The Digital Twin action-effect multipliers are assumptions because C-MAPSS has no observed recovery-action outcomes.",
                    "This prototype does not establish real-world action effectiveness or physical safety.",
                    "The selected candidate is not a learned or proven best action; review by a human is required.",
                ],
            }
        else:
            advisory = {
                "status": "not_supported",
                "source": "none",
                "human_review_required": True,
                "reason": "Recovery planning was not requested and no independently validated action policy exists.",
            }
    else:
        advisory = {
            "status": "not_supported",
            "source": "none",
            "human_review_required": True,
            "reason": "This run requested prognostics only; no action policy was invoked.",
        }

    if run_planner is False and requested_role == "digital_twin":
        roles_skipped.append(
            {"role": "RecoveryPlanningAgent", "status": "not_requested", "reason": "Only the requested Twin role and its prognosis dependency ran."}
        )
    if requested_role == "prognostics":
        roles_skipped.extend(
            [
                {"role": "DigitalTwin", "status": "not_requested", "reason": "Only the requested prognostics role ran."},
                {"role": "RecoveryPlanningAgent", "status": "not_requested", "reason": "Only the requested prognostics role ran."},
            ]
        )
    for role in ("monitoring", "diagnosis", "safety_verifier"):
        roles_skipped.append({"role": role, "status": "not_supported", "reason": ROLE_NOT_SUPPORTED_REASONS["cmapss"][role]})

    metadata = {
        "source_split": "official C-MAPSS training trajectories, saved validation units",
        "subset": subset,
        "unit_id": selected_unit,
        "cycle": actual_cycle,
        "history_cycles_observed": int(len(prefix_labeled)),
        "unit_id_source": "official source-file engine ID",
    }
    model_result = _make_result(
        dataset=f"cmapss_{subset.lower()}",
        task="remaining_useful_life_regression",
        metadata=metadata,
        prediction={
            "predicted_rul_cycles": predicted_rul,
            "tree_spread_cycles": float(prediction_row["rul_uncertainty"]),
            "uncertainty_calibrated": False,
        },
        model_version=_artifact_version(model_path, "evaluated C-MAPSS RUL artifact"),
        downstream_roles_allowed=["digital_twin"] if requested_role != "prognostics" else [],
        explanation="The evaluated random-forest RUL model summarized only this unit's observed cycles; its tree spread is not a calibrated interval.",
    )
    agent_results = [model_result]
    if twin_state is not None and twin_model_version is not None:
        agent_results.append(
            _make_result(
                dataset=f"cmapss_{subset.lower()}",
                task="digital_twin_state_estimation",
                metadata=metadata,
                prediction={
                    "predicted_health_index": float(twin_state["predicted_health_index"]),
                    "state_summary": twin_state["state_summary"],
                },
                model_version=twin_model_version,
                downstream_roles_allowed=["recovery_planner"] if run_planner else [],
                explanation="The existing C-MAPSS prototype estimates a current normalized state from observed telemetry; its action-effect simulation is assumption-based.",
            )
        )
    if plan_output is not None:
        agent_results.append(
            _make_result(
                dataset=f"cmapss_{subset.lower()}",
                task="rule_based_recovery_advisory",
                metadata=metadata,
                prediction=plan_output,
                model_version={
                    "source": "existing deterministic RecoveryPlanningAgent policy",
                    "configuration": "risk_budget=0.25; simulation_horizon_cycles=5",
                },
                downstream_roles_allowed=[],
                explanation="This is a configured prototype ranking of simulated candidates; it is not a learned or validated best action.",
            )
        )
    # Truth is looked up only after the prediction, Twin and optional advisory.
    truth_row = prefix_labeled[prefix_labeled["cycle"] == actual_cycle].iloc[-1]
    ground_truth = {
        "status": "available_after_prediction",
        "source": "RUL derived from this labeled C-MAPSS training trajectory",
        "unit_id": selected_unit,
        "cycle": actual_cycle,
        "rul_cycles": float(truth_row["rul"]),
        "comparison_scope": "illustrative saved validation unit; not a new evaluation score",
    }
    return {
        "schema_version": 1,
        "dataset": "cmapss",
        "task": "remaining_useful_life_regression plus existing C-MAPSS prototype roles when requested",
        "metadata": metadata,
        "roles_run": roles_run,
        "roles_skipped": roles_skipped,
        "agent_results": agent_results,
        "predictions": [
            {"role": "PrognosticsAgent", **model_result["prediction"]},
            *([{"role": "DigitalTwin", **twin_state}] if twin_state else []),
            *([{"role": "RecoveryPlanningAgent", **plan_output}] if plan_output else []),
        ],
        "action_advisory": advisory,
        "ground_truth": ground_truth,
    }


def _nasa_paths(
    evaluation_report_root: Path,
) -> tuple[Path, Path]:
    return (
        DEMO_MODEL_ROOT / "nasa_turbofan_ds2" / "rich_summary_prognostics_agent.joblib",
        evaluation_report_root / "nasa_ablation" / "results.json",
    )


def _ensure_nasa_rich_artifact(
    data_root: Path,
    evaluation_report_root: Path,
    audit_path: Path | None,
) -> tuple[PrognosticsAgent, Path, str, dict[str, Any]]:
    """Load the selected rich model or reproduce its missing training artifact.

    The saved ablation report selected the rich representation on six official
    development units. Its evaluation run did not persist the fitted object, so
    the same final-development fit is reproduced here; official test units are
    never opened by this function.
    """

    model_path, report_path = _nasa_paths(evaluation_report_root)
    if not report_path.is_file():
        raise FileNotFoundError(f"NASA ablation report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report["feature_selection"]["selected_representation"]
    if selected != "rich_summary":
        raise ValueError(f"Expected audited rich-summary model selection, found {selected!r}")
    feature_columns = [str(value) for value in report["selected_features"]]
    _validate_features(feature_columns, context="NASA Simulation-2")
    official_dev_ids = [int(value) for value in report["official_development_unit_ids"]]
    expected_config = PrognosticsConfig(**report["config"])

    if model_path.is_file():
        agent = PrognosticsAgent.load(model_path)
        if agent.feature_columns != feature_columns or asdict(agent.config) != asdict(expected_config):
            raise ValueError("Saved NASA demo artifact does not match the audited selected model")
        return agent, model_path, "recreated selected rich-summary evaluation model", report

    data_dir = data_root / "nasa_turbofan_ds2"
    _read_audit(data_dir, audit_path)
    development, _ = load_rich_cycle_summaries(
        data_dir, "dev", unit_ids=official_dev_ids
    )
    if set(development["unit_id"].astype(int).unique()) != set(official_dev_ids):
        raise ValueError("NASA rich-summary loader changed official development unit IDs")
    development = development.drop(columns=["source_samples_in_cycle"]).copy()
    agent = PrognosticsAgent.fit(development, feature_columns, expected_config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)
    _write_json(
        model_path.with_suffix(".metadata.json"),
        {
            "artifact_source": "rebuilt because the selected model object was absent",
            "training_path": "same final-development fit as scripts/evaluate_nasa_cycle_summary_ablation.py",
            "training_source_split": "official dev only",
            "official_development_unit_ids": official_dev_ids,
            "official_test_units_opened": False,
            "selected_representation": selected,
            "selected_features": feature_columns,
            "config": expected_config.as_dict(),
        },
    )
    return agent, model_path, "rebuilt from audited official development units", report


def _run_nasa(
    *,
    data_root: Path,
    unit_id: int | None,
    cycle: int,
    requested_role: str | None,
    evaluation_report_root: Path,
    nasa_audit_path: Path | None,
) -> dict[str, Any]:
    if requested_role is not None and role_route("nasa_turbofan_ds2", requested_role)["status"] != "supported":
        return _not_supported_trace(
            "nasa_turbofan_ds2", "remaining_useful_life_regression", requested_role
        )
    agent, model_path, artifact_source, report = _ensure_nasa_rich_artifact(
        data_root, evaluation_report_root, nasa_audit_path
    )
    official_dev_ids = [int(value) for value in report["official_development_unit_ids"]]
    selected_unit = int(unit_id) if unit_id is not None else min(official_dev_ids)
    if selected_unit not in set(official_dev_ids):
        raise ValueError(
            "NASA demo unit must be an official development unit; this demo does not rescore the three test units. "
            f"Allowed: {official_dev_ids}"
        )
    data_dir = data_root / "nasa_turbofan_ds2"
    rich, info = load_rich_cycle_summaries(
        data_dir, "dev", unit_ids=[selected_unit]
    )
    if cycle < 1 or cycle > int(rich["cycle"].max()):
        raise ValueError(f"cycle must be within this unit's observed flight cycles 1..{int(rich['cycle'].max())}")
    prefix = rich[rich["cycle"] <= cycle].sort_values("cycle")
    truth_row = prefix[prefix["cycle"] == cycle].iloc[-1]
    # Only identifiers and the selected sensor summaries reach the model.
    model_frame = nasa_prediction_only_frame(prefix, agent.feature_columns)
    if "rul" in model_frame or "source_samples_in_cycle" in model_frame:
        raise AssertionError("NASA target/source count leaked into inference features")
    # The shared window builder expects a target-named column, so provide only
    # an all-NaN sentinel. This matches PrognosticsAgent's internal fallback;
    # no NASA target values are passed to prediction.
    model_frame["rul"] = np.nan
    prediction = agent.predict_current_available_history(model_frame)
    if prediction.empty:
        raise RuntimeError("NASA prognostics could not make a prediction")
    output = prediction.iloc[-1]
    uncertainty = float(output["rul_uncertainty"])
    metadata = {
        "unit_id": selected_unit,
        "cycle": int(output["end_cycle"]),
        "source_split": "official development units",
        "unit_id_source": "official NASA Simulation-2 unit ID",
        "cycle_source": "official flight-cycle index",
    }
    model_result = _make_result(
        dataset="nasa_turbofan_ds2",
        task="remaining_useful_life_regression",
        metadata=metadata,
        prediction={
            "predicted_rul_cycles": float(output["predicted_rul"]),
            "tree_spread_cycles": uncertainty,
            "uncertainty_calibrated": False,
        },
        model_version=_artifact_version(model_path, artifact_source),
        downstream_roles_allowed=[],
        explanation="The selected random-forest model uses rich summaries from completed cycles for this official unit; the displayed tree spread is uncalibrated.",
    )
    roles_skipped = [
        {
            "role": role,
            "status": "not_supported",
            "reason": ROLE_NOT_SUPPORTED_REASONS["nasa_turbofan_ds2"].get(
                role, "No evaluated NASA model is registered for this role."
            ),
        }
        for role in ("digital_twin", "recovery_planner", "faulty_classifier", "torque_margin_regressor", "sequence_agent")
    ]
    return {
        "schema_version": 1,
        "dataset": "nasa_turbofan_ds2",
        "task": "remaining_useful_life_regression",
        "metadata": metadata,
        "roles_run": ["PrognosticsAgent"],
        "roles_skipped": roles_skipped,
        "agent_results": [model_result],
        "predictions": [{"role": "PrognosticsAgent", **model_result["prediction"]}],
        "action_advisory": {
            "status": "not_supported",
            "source": "none",
            "human_review_required": True,
            "reason": "No NASA action policy is validated; C-MAPSS Twin/Planner roles are not routed here.",
        },
        "ground_truth": {
            "status": "available_after_prediction",
            "source": "NASA Y_dev value for this official development unit and cycle",
            "unit_id": selected_unit,
            "cycle": int(truth_row["cycle"]),
            "rul_cycles": float(truth_row["rul"]),
            "comparison_scope": "illustrative development sample; the final test-unit metrics remain those in the saved evaluation report",
        },
        "adapter_metadata": {
            "source_samples_in_cycle": int(truth_row["source_samples_in_cycle"]),
            "future_cycle_samples_used": bool(info["future_cycle_samples_used"]),
        },
    }


def _run_phm(
    *,
    data_root: Path,
    sample_id: str | None,
    requested_role: str | None,
    evaluation_model_root: Path,
) -> dict[str, Any]:
    if requested_role is not None and role_route("phm2024", requested_role)["status"] != "supported":
        return _not_supported_trace(
            "phm2024", "faulty classification and torque-margin regression", requested_role
        )
    source_root = data_root / "phm2024" / "extracted"
    classifier_path = evaluation_model_root / "phm2024" / "faulty_classifier.joblib"
    regressor_path = evaluation_model_root / "phm2024" / "torque_margin_regressor.joblib"
    if not classifier_path.is_file() or not regressor_path.is_file():
        raise FileNotFoundError("Evaluated PHM classifier or regressor artifact is missing")

    classification = load_phm_bundle("train", "classification", source_root)
    regression = load_phm_bundle("train", "regression", source_root)
    ids = classification.metadata["id"].astype(str)
    if sample_id is None:
        selected_index = int(np.argmin(classification.metadata["id"].to_numpy()))
    else:
        matches = np.flatnonzero(ids.to_numpy() == str(sample_id))
        if len(matches) != 1:
            raise ValueError(f"PHM sample id {sample_id!r} was not found in the labeled training file")
        selected_index = int(matches[0])
    selected_id = classification.metadata.iloc[selected_index]["id"]
    if regression.metadata.iloc[selected_index]["id"] != selected_id:
        raise ValueError("PHM task adapters do not align on the same sample ID")

    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)
    classification_input = phm_prediction_only_bundle(classification, selected_index)
    regression_input = phm_prediction_only_bundle(regression, selected_index)
    if classification_input.y is not None or regression_input.y is not None:
        raise AssertionError("PHM labels were exposed to a prediction agent")
    if "id" in classification_input.X or PROTECTED_COLUMNS.intersection(classification_input.X.columns):
        raise AssertionError("PHM classifier received metadata or target columns")
    if "id" in regression_input.X or PROTECTED_COLUMNS.intersection(regression_input.X.columns):
        raise AssertionError("PHM regressor received metadata or target columns")

    run_classifier = requested_role in {None, "faulty_classifier"}
    run_regressor = requested_role in {None, "torque_margin_regressor"}
    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    roles_run: list[str] = []
    if run_classifier:
        classifier_prediction = classifier.predict(classification_input).predictions.iloc[0]
        cls_prediction = {
            "faulty_prediction": int(classifier_prediction["condition_alert"]),
            "faulty_probability": float(classifier_prediction["condition_alert_probability"]),
            "decision_threshold": float(classifier_prediction["decision_threshold"]),
        }
        results.append(
            _make_result(
                dataset="phm2024",
                task="faulty_binary_classification",
                metadata={"id": str(selected_id), "id_semantics": "sample metadata; not an asset/engine ID"},
                prediction=cls_prediction,
                model_version=_artifact_version(classifier_path, "evaluated PHM faulty classifier"),
                downstream_roles_allowed=[],
                explanation="The faulty classifier sees the seven measurements only; the sample ID and both targets are outside its feature matrix.",
            )
        )
        predictions.append({"role": "FaultyClassificationAgent", **cls_prediction})
        roles_run.append("FaultyClassificationAgent")

    if run_regressor:
        if not isinstance(regressor, dict) or regressor.get("target") != "trq_margin":
            raise ValueError("PHM torque-margin artifact has an unexpected target contract")
        feature_names = [str(value) for value in regressor["feature_names"]]
        _validate_features(feature_names, context="PHM torque-margin")
        if tuple(regression_input.X.columns) != tuple(feature_names):
            raise ValueError("PHM torque-margin artifact input columns do not match the adapter")
        transformed = regressor["preprocessor"].transform(regression_input.X)
        margin_prediction = float(regressor["model"].predict(transformed.to_numpy(dtype=np.float64))[0])
        reg_prediction = {"predicted_trq_margin": margin_prediction}
        results.append(
            _make_result(
                dataset="phm2024",
                task="torque_margin_regression",
                metadata={"id": str(selected_id), "id_semantics": "sample metadata; not an asset/engine ID"},
                prediction=reg_prediction,
                model_version=_artifact_version(regressor_path, "evaluated PHM torque-margin regressor"),
                downstream_roles_allowed=[],
                explanation="The torque-margin regressor sees the seven measurements only; the sample ID and both targets are outside its feature matrix.",
            )
        )
        predictions.append({"role": "TorqueMarginRegressionAgent", **reg_prediction})
        roles_run.append("TorqueMarginRegressionAgent")

    roles_skipped: list[dict[str, Any]] = []
    for role, will_run in (("faulty_classifier", run_classifier), ("torque_margin_regressor", run_regressor)):
        if not will_run:
            roles_skipped.append({"role": role, "status": "not_requested", "reason": "A different supported PHM role was explicitly requested."})
    for role in ("prognostics", "sequence_agent", "digital_twin", "recovery_planner"):
        roles_skipped.append({"role": role, "status": "not_supported", "reason": ROLE_NOT_SUPPORTED_REASONS["phm2024"][role]})

    # Labels are read from the separate bundle only after both model calls.
    truth: dict[str, Any] = {"status": "available_after_prediction", "source": "PHM labeled training target file", "id": str(selected_id), "id_semantics": "sample metadata; not an asset/engine ID"}
    if classification.y is not None:
        truth["faulty"] = int(classification.y.iloc[selected_index])
    if regression.y is not None:
        truth["trq_margin"] = float(regression.y.iloc[selected_index])
    truth["comparison_scope"] = "illustrative labeled training sample; not an independent evaluation result"

    return {
        "schema_version": 1,
        "dataset": "phm2024",
        "task": "faulty classification and torque-margin regression",
        "metadata": {"id": str(selected_id), "id_semantics": "sample metadata; not an asset/engine ID", "source_split": "labeled training file"},
        "roles_run": roles_run,
        "roles_skipped": roles_skipped,
        "agent_results": results,
        "predictions": predictions,
        "action_advisory": {
            "status": "not_supported",
            "source": "none",
            "human_review_required": True,
            "reason": "No PHM action policy is configured or validated.",
        },
        "ground_truth": truth,
    }


def run_demo(
    *,
    dataset: str,
    data_root: Path = ROOT / "data" / "raw",
    subset: str = "FD001",
    unit_id: int | None = None,
    cycle: int = 30,
    sample_id: str | None = None,
    requested_role: str | None = None,
    evaluation_model_root: Path = QUALITY_ROOT,
    evaluation_report_root: Path = EVALUATION_REPORTS,
    nasa_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Run the native evaluated route for one selected sample."""

    if dataset not in SUPPORTED_ROLES:
        raise ValueError(f"dataset must be one of {sorted(SUPPORTED_ROLES)}")
    data_root = Path(data_root)
    evaluation_model_root = Path(evaluation_model_root)
    evaluation_report_root = Path(evaluation_report_root)
    nasa_audit_path = Path(nasa_audit_path) if nasa_audit_path is not None else None
    if requested_role is not None and requested_role not in {
        "prognostics", "digital_twin", "recovery_planner", "faulty_classifier",
        "torque_margin_regressor", "sequence_agent", "monitoring", "diagnosis", "safety_verifier",
    }:
        raise ValueError(f"Unknown role {requested_role!r}")
    if dataset == "cmapss":
        return _run_cmapss(
            subset=subset.upper(), data_root=data_root, unit_id=unit_id,
            cycle=cycle, requested_role=requested_role,
            evaluation_model_root=evaluation_model_root,
            evaluation_report_root=evaluation_report_root,
        )
    if dataset == "nasa_turbofan_ds2":
        return _run_nasa(
            data_root=data_root, unit_id=unit_id, cycle=cycle,
            requested_role=requested_role,
            evaluation_report_root=evaluation_report_root,
            nasa_audit_path=nasa_audit_path,
        )
    return _run_phm(
        data_root=data_root, sample_id=sample_id,
        requested_role=requested_role,
        evaluation_model_root=evaluation_model_root,
    )


def print_demo(trace: dict[str, Any], trace_path: Path | None = None) -> None:
    """Print the handoffs in a stable order suitable for a live walkthrough."""

    task = trace["task"]
    if trace["roles_skipped"] and trace["roles_skipped"][0]["status"] == "not_supported" and not trace["agent_results"]:
        task = f"{task} (requested role is unsupported)"
    print(f"Dataset/task: {trace['dataset']} / {task}")
    print(f"Metadata: {json.dumps(trace['metadata'], sort_keys=True)}")
    ran = ", ".join(trace["roles_run"]) if trace["roles_run"] else "none"
    print(f"Agent handoff: ran {ran}")
    if trace["roles_skipped"]:
        skipped = "; ".join(
            f"{item['role']}={item['status']} ({item['reason']})"
            for item in trace["roles_skipped"]
        )
        print(f"Agent handoff: skipped {skipped}")
    print("Predictions:")
    if not trace["predictions"]:
        print("  none")
    for prediction in trace["predictions"]:
        print(f"  {prediction['role']}: {json.dumps(prediction, sort_keys=True)}")
    for result in trace["agent_results"]:
        version = result["model_version"]
        version_text = version["source"]
        if "sha256" in version:
            version_text += f" (sha256 {version['sha256'][:12]})"
        if "configuration" in version:
            version_text += f" [{version['configuration']}]"
        print(f"  model: {result['task']} from {version_text}")
        print(f"  note: {result['explanation']}")
    print("Advisory:")
    print(f"  {json.dumps(trace['action_advisory'], sort_keys=True)}")
    print("Ground truth (read after prediction):")
    print(f"  {json.dumps(trace['ground_truth'], sort_keys=True)}")
    if trace_path is not None:
        print(f"Trace saved: {trace_path}")
