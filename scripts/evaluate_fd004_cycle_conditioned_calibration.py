"""Explore milestone-conditioned RUL safety margins on the prior FD004 holdout.

This is explicitly a development analysis: the same engine holdout was already
examined in the cycle-age comparison. It must not be presented as an independent
final test. The official C-MAPSS test files are not loaded.
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
    _summarize_rows,
)
from scripts.evaluate_fd004_cycle_age_holdout import (
    _decision_row,
    _paired_prediction_metrics,
)
from src.cmapss_data import add_train_rul, load_telemetry, summarize_windows, transform_features
from src.closed_loop import ClosedLoopOrchestrator
from src.prognostics import PrognosticsAgent


CALIBRATION_CYCLES = MILESTONES[2:]


def main() -> None:
    started = time.perf_counter()
    out_dir = ROOT / "reports" / "generated"
    previous_report = out_dir / "fd004_cycle_age_holdout.json"
    if not previous_report.exists():
        raise FileNotFoundError(
            "Run scripts/evaluate_fd004_cycle_age_holdout.py first"
        )
    prior_protocol = json.loads(
        previous_report.read_text(encoding="utf-8")
    )["evaluation_protocol"]
    fit_units = tuple(int(value) for value in prior_protocol["model_fit_engine_ids"])
    calibration_units = tuple(
        int(value)
        for value in prior_protocol["uncertainty_calibration_engine_ids"]
    )
    evaluation_units = tuple(
        int(value)
        for value in prior_protocol["decision_evaluation_engine_ids"]
    )
    if (
        set(fit_units) & set(calibration_units)
        or set(fit_units) & set(evaluation_units)
        or set(calibration_units) & set(evaluation_units)
    ):
        raise RuntimeError("Saved fit, calibration, and evaluation engines overlap")

    full_frame = add_train_rul(
        load_telemetry(ROOT / "data" / "raw" / "cmapss", "FD004", "train")
    )
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
        baseline_agent.config,
        include_cycle_feature=True,
        planning_margin_cycles=CALIBRATION_CYCLES,
    )
    candidate_agent = PrognosticsAgent.fit(
        fit_frame, features, config=candidate_config
    )
    candidate_agent.calibrate_uncertainty(calibration_frame)
    cycle_margins = dict(candidate_agent.planning_margins_by_cycle_)
    cycle_margin_counts = dict(
        candidate_agent.planning_margin_engine_counts_by_cycle_
    )
    candidate_orchestrator = ClosedLoopOrchestrator(
        candidate_agent, twin, planner, verifier
    )

    truth_lookup = {
        (int(row.unit_id), int(row.cycle)): float(row.rul)
        for row in heldout[["unit_id", "cycle", "rul"]].itertuples(index=False)
    }
    telemetry = heldout.drop(columns=["rul"])
    model_rows: dict[str, list[dict[str, Any]]] = {
        "rf30_global": [],
        "rf30_cycle_feature_global": [],
        "rf30_cycle_feature_milestone_margin": [],
    }
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
                probabilities = diagnosis.predict_proba(
                    summarize_windows(window)
                ).iloc[0]
                diagnosis_state = str(probabilities.idxmax())
                diagnosis_confidence = float(probabilities.max())

            # The same trained cycle-feature model is evaluated with either its
            # pooled margin or its milestone-specific margin map.
            candidate_agent.planning_margins_by_cycle_ = {}
            global_result = candidate_orchestrator.step(
                prefix,
                anomaly_score=anomaly_score,
                diagnosis_state=diagnosis_state,
                diagnosis_confidence=diagnosis_confidence,
                horizon=HORIZON,
            )
            candidate_agent.planning_margins_by_cycle_ = cycle_margins
            conditioned_result = candidate_orchestrator.step(
                prefix,
                anomaly_score=anomaly_score,
                diagnosis_state=diagnosis_state,
                diagnosis_confidence=diagnosis_confidence,
                horizon=HORIZON,
            )
            baseline_result = baseline_orchestrator.step(
                prefix,
                anomaly_score=anomaly_score,
                diagnosis_state=diagnosis_state,
                diagnosis_confidence=diagnosis_confidence,
                horizon=HORIZON,
            )
            actual_rul = truth_lookup[(unit_id, milestone)]
            common = (
                unit_id,
                milestone,
                actual_rul,
            )
            model_rows["rf30_global"].append(
                _decision_row(
                    "rf30_global", *common, baseline_result, anomaly_score,
                    diagnosis_state, diagnosis_confidence
                )
            )
            model_rows["rf30_cycle_feature_global"].append(
                _decision_row(
                    "rf30_cycle_feature_global", *common, global_result,
                    anomaly_score, diagnosis_state, diagnosis_confidence
                )
            )
            model_rows["rf30_cycle_feature_milestone_margin"].append(
                _decision_row(
                    "rf30_cycle_feature_milestone_margin", *common,
                    conditioned_result, anomaly_score, diagnosis_state,
                    diagnosis_confidence
                )
            )

    summaries = {
        name: _summarize_rows(rows, len(evaluation_units), SEED + index)
        for index, (name, rows) in enumerate(model_rows.items())
    }
    paired = {
        "cycle_feature_global_vs_rf30_global": _paired_prediction_metrics(
            model_rows["rf30_global"],
            model_rows["rf30_cycle_feature_global"],
        ),
        "cycle_feature_milestone_margin_vs_rf30_global": _paired_prediction_metrics(
            model_rows["rf30_global"],
            model_rows["rf30_cycle_feature_milestone_margin"],
        ),
        "milestone_margin_vs_cycle_feature_global": _paired_prediction_metrics(
            model_rows["rf30_cycle_feature_global"],
            model_rows["rf30_cycle_feature_milestone_margin"],
        ),
    }
    by_cycle: dict[str, Any] = {}
    for cycle in MILESTONES:
        by_cycle[str(cycle)] = {
            name: _summarize_rows(
                [row for row in rows if row["cycle"] == cycle],
                len(evaluation_units),
                SEED + cycle + index,
            )
            for index, (name, rows) in enumerate(model_rows.items())
        }

    report = {
        "experiment": "FD004 cycle-feature and milestone-conditioned planning-margin analysis",
        "evaluation_status": "exploratory; reuses the previously inspected development holdout",
        "official_test_files_loaded": False,
        "protocol": {
            "source": "official C-MAPSS training trajectories only",
            "fit_engine_ids": list(fit_units),
            "calibration_engine_ids": list(calibration_units),
            "decision_evaluation_engine_ids": list(evaluation_units),
            "calibration_cycles": list(CALIBRATION_CYCLES),
            "window_size": candidate_agent.config.window_size,
            "cycle_feature": "end_cycle / 100 appended to summarized telemetry window",
            "milestone_margin_method": (
                "At each configured cycle, use one signed overestimation residual "
                "per calibration engine and the finite-sample 95% order statistic. "
                "Other cycles fall back to the pooled margin."
            ),
            "milestone_calibration_engine_counts": cycle_margin_counts,
            "pooled_cycle_feature_margin_cycles": candidate_agent.lower_bound_margin_95,
            "milestone_planning_margins_cycles": cycle_margins,
            "candidate_config": candidate_agent.config.as_dict(),
            "evaluation_label_access": (
                "Actual RUL is joined only after the telemetry-only decisions are returned."
            ),
        },
        "overall": summaries,
        "paired_comparisons": paired,
        "decision": (
            "Milestone-specific margins improved observed coverage at the tested "
            "cycles without changing RUL predictions, but cycle-60 coverage remained "
            "below 95%. Keep this calibration mode opt-in pending a stronger, "
            "independent evaluation."
        ),
        "by_cycle": by_cycle,
        "limitations": [
            "This reuses the development holdout already inspected in the preceding cycle-age experiment; results are exploratory, not an independent confirmation.",
            "Milestone-specific margins are evaluated only at the listed cycles; intermediate cycles use the pooled margin.",
            "Empirical coverage on a finite holdout is an estimate and does not prove deployment safety.",
            "C-MAPSS contains no observed recovery-action outcomes, so simulated action effects remain Digital Twin assumptions.",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }

    json_path = out_dir / "fd004_cycle_conditioned_calibration.json"
    csv_path = out_dir / "fd004_cycle_conditioned_calibration_states.csv"
    pd.DataFrame(
        [row for rows in model_rows.values() for row in rows]
    ).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("FD004 cycle-conditioned calibration exploration")
    print("===============================================")
    for name, summary in summaries.items():
        metrics = summary["rul_metrics_on_decision_states"]
        print(
            f"{name}: MAE={metrics['mae_cycles']:.2f}, "
            f"planning coverage={metrics['planning_lower_bound_empirical_coverage']:.3f}, "
            f"actions={summary['selected_action_counts']}"
        )
    for cycle, margin in cycle_margins.items():
        print(
            f"cycle {cycle}: calibration engines={cycle_margin_counts[cycle]}, "
            f"one-sided margin={margin:.2f} cycles"
        )
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"State-level CSV: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
