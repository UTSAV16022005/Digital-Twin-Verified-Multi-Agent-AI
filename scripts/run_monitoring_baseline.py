"""Run the reconstruction-based Monitoring Agent on FD001."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.cmapss_data import (
    FEATURE_COLUMNS,
    add_test_rul,
    add_train_rul,
    constant_columns,
    load_test_rul,
    load_telemetry,
    split_engine_units,
)
from src.monitoring import PCAMonitoringAgent, evaluate_alert_events


def failure_cycles(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("unit_id")["cycle"].max()


def main() -> None:
    data_dir = Path("data/raw/cmapss")
    train = add_train_rul(load_telemetry(data_dir, "FD001", "train"))
    test = add_test_rul(
        load_telemetry(data_dir, "FD001", "test"),
        load_test_rul(data_dir, "FD001"),
    )
    split = split_engine_units(train, validation_fraction=0.2, seed=42)
    fit_frame = train[train["unit_id"].isin(split.train_units)].copy()
    validation_frame = train[train["unit_id"].isin(split.validation_units)].copy()

    dropped = constant_columns(fit_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]
    agent = PCAMonitoringAgent.fit(
        fit_frame,
        features,
        healthy_cycles=30,
        threshold_quantile=0.99,
    )

    validation_scored = agent.score(validation_frame)
    test_scored = agent.score(test)
    validation_metrics, validation_events = evaluate_alert_events(
        validation_scored,
        failure_cycles(validation_frame),
        agent.healthy_cycles,
    )
    test_failure_cycles = test.groupby("unit_id")["cycle"].max() + test.groupby("unit_id")["rul"].last()
    test_metrics, test_events = evaluate_alert_events(
        test_scored,
        test_failure_cycles,
        agent.healthy_cycles,
    )

    report = {
        "subset": "FD001",
        "agent": "PCA reconstruction monitor",
        "fit_engines": len(split.train_units),
        "validation_engines": len(split.validation_units),
        "test_engines": int(test["unit_id"].nunique()),
        "healthy_cycles": agent.healthy_cycles,
        "threshold_quantile": 0.99,
        "threshold": agent.threshold,
        "pca_components": int(agent.pca.n_components_),
        "explained_variance_ratio": float(agent.pca.explained_variance_ratio_.sum()),
        "features_used": features,
        "features_dropped_as_constant": dropped,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fd001_monitoring_baseline.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    validation_events.to_csv(output_dir / "fd001_monitoring_validation_events.csv", index=False)
    test_events.to_csv(output_dir / "fd001_monitoring_test_events.csv", index=False)

    print("FD001 Monitoring Agent baseline")
    print("===============================")
    print("agent: PCA reconstruction monitor")
    print(f"healthy training cycles: {agent.healthy_cycles}")
    print(f"PCA components: {agent.pca.n_components_}")
    print(f"variance retained: {agent.pca.explained_variance_ratio_.sum():.3f}")
    print(f"alert threshold: {agent.threshold:.6f}")
    print(
        "validation: "
        f"detection_rate={validation_metrics['detection_rate']:.3f}, "
        f"healthy_false_alarm_rate={validation_metrics['healthy_false_alarm_rate']:.3f}, "
        f"median_lead_time={validation_metrics['median_lead_time_cycles']:.1f} cycles"
    )
    print(
        "official test: "
        f"detection_rate={test_metrics['detection_rate']:.3f}, "
        f"healthy_false_alarm_rate={test_metrics['healthy_false_alarm_rate']:.3f}, "
        f"median_lead_time={test_metrics['median_lead_time_cycles']:.1f} cycles"
    )
    print(f"saved report: {output_dir / 'fd001_monitoring_baseline.json'}")


if __name__ == "__main__":
    main()

