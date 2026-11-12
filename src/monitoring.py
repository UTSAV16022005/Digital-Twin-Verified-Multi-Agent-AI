"""Monitoring-agent models and event-level evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAMonitoringAgent:
    """A transparent reconstruction-based monitoring agent.

    This is the baseline implementation of the monitoring-agent contract. It
    can later be replaced by an LSTM autoencoder without changing the scoring
    and evaluation interfaces.
    """

    feature_columns: list[str]
    scaler: StandardScaler
    pca: PCA
    threshold: float
    healthy_cycles: int

    @classmethod
    def fit(
        cls,
        training_frame: pd.DataFrame,
        feature_columns: list[str],
        healthy_cycles: int = 30,
        threshold_quantile: float = 0.99,
    ) -> "PCAMonitoringAgent":
        healthy = training_frame[training_frame["cycle"] <= healthy_cycles]
        if healthy.empty:
            raise ValueError("No healthy rows found for the selected cycle limit")

        scaler = StandardScaler()
        healthy_scaled = scaler.fit_transform(healthy[feature_columns])
        pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
        pca.fit(healthy_scaled)
        healthy_scores = cls._reconstruction_error(pca, healthy_scaled)
        threshold = float(np.quantile(healthy_scores, threshold_quantile))
        return cls(
            feature_columns=feature_columns,
            scaler=scaler,
            pca=pca,
            threshold=threshold,
            healthy_cycles=healthy_cycles,
        )

    @staticmethod
    def _reconstruction_error(pca: PCA, scaled_values: np.ndarray) -> np.ndarray:
        reconstructed = pca.inverse_transform(pca.transform(scaled_values))
        return np.mean((scaled_values - reconstructed) ** 2, axis=1)

    def per_feature_reconstruction_error(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute squared error per sensor feature."""
        scaled = self.scaler.transform(frame[self.feature_columns])
        reconstructed = self.pca.inverse_transform(self.pca.transform(scaled))
        sq_err = (scaled - reconstructed) ** 2
        return pd.DataFrame(sq_err, columns=[f"err_{c}" for c in self.feature_columns], index=frame.index)

    def score(self, frame: pd.DataFrame, persistence: int = 1) -> pd.DataFrame:
        """Return telemetry plus anomaly score, alert fields, and feature contributions."""

        # Persistence buffers are position-indexed; reset filtered input labels
        # so an engine subset with original DataFrame indices cannot index past
        # the allocated arrays.
        result = frame.copy().reset_index(drop=True)
        scaled = self.scaler.transform(result[self.feature_columns])
        reconstructed = self.pca.inverse_transform(self.pca.transform(scaled))
        per_feature_sq_err = (scaled - reconstructed) ** 2
        result["anomaly_score"] = np.mean(per_feature_sq_err, axis=1)
        raw_alert = result["anomaly_score"] > self.threshold
        result["alert"] = raw_alert

        if persistence > 1:
            confirmed = np.zeros(len(result), dtype=bool)
            consecutive = np.zeros(len(result), dtype=int)
            for _, idx in result.groupby("unit_id", sort=True).groups.items():
                run = 0
                for i in idx:
                    run = run + 1 if raw_alert.loc[i] else 0
                    consecutive[i] = run
                    confirmed[i] = run >= persistence
            result["consecutive_alerts"] = consecutive
            result["confirmed_alert"] = confirmed
            result["alert"] = confirmed
        else:
            result["confirmed_alert"] = raw_alert

        # Per-sensor contributions
        total_err = np.clip(per_feature_sq_err.sum(axis=1, keepdims=True), 1e-12, None)
        shares = per_feature_sq_err / total_err
        top_indices = np.argsort(-per_feature_sq_err, axis=1)[:, :3]
        result["top_sensor_contributors"] = [
            ";".join(f"{self.feature_columns[j]}:{shares[r, j]:.3f}" for j in row)
            for r, row in enumerate(top_indices)
        ]
        return result


def evaluate_alert_events(
    scored_frame: pd.DataFrame,
    failure_cycle_by_unit: pd.Series,
    healthy_cycles: int,
    degraded_rul: int = 50,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate first-alert detection, lead time, precision, recall, and per-sensor attribution."""

    event_rows: list[dict[str, float | int | bool | str]] = []
    healthy_rows = scored_frame[scored_frame["cycle"] <= healthy_cycles]
    healthy_false_alarm_rate = float(healthy_rows["alert"].mean()) if not healthy_rows.empty else 0.0

    # Window-level precision/recall/F1 on well-defined ground-truth regions:
    # Positive class: degraded windows (RUL <= degraded_rul)
    # Negative class: healthy windows (cycle <= healthy_cycles)
    scored = scored_frame.copy()
    if "rul" not in scored.columns and "unit_id" in scored.columns:
        failure_map = failure_cycle_by_unit.to_dict()
        scored["failure_cycle"] = scored["unit_id"].map(failure_map)
        scored["rul"] = scored["failure_cycle"] - scored["cycle"]

    degraded_mask = (scored["rul"] <= degraded_rul) if "rul" in scored.columns else pd.Series(False, index=scored.index)
    healthy_mask = scored["cycle"] <= healthy_cycles

    tp = int((scored.loc[degraded_mask, "alert"]).sum())
    fn = int((~scored.loc[degraded_mask, "alert"]).sum())
    fp = int((scored.loc[healthy_mask, "alert"]).sum())
    tn = int((~scored.loc[healthy_mask, "alert"]).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    for unit_id, unit_frame in scored_frame.groupby("unit_id", sort=True):
        unit_frame = unit_frame.sort_values("cycle")
        failure_cycle = float(failure_cycle_by_unit.loc[unit_id])
        alerts = unit_frame[unit_frame["alert"]]
        first_alert_cycle = float(alerts["cycle"].iloc[0]) if not alerts.empty else np.nan
        detected = bool(np.isfinite(first_alert_cycle) and first_alert_cycle < failure_cycle)
        lead_time = failure_cycle - first_alert_cycle if detected else np.nan
        top_contrib = str(alerts["top_sensor_contributors"].iloc[0]) if not alerts.empty and "top_sensor_contributors" in alerts else ""
        event_rows.append(
            {
                "unit_id": int(unit_id),
                "failure_cycle": failure_cycle,
                "first_alert_cycle": first_alert_cycle,
                "detected_before_failure": detected,
                "lead_time_cycles": lead_time,
                "top_sensor_at_first_alert": top_contrib,
            }
        )

    events = pd.DataFrame(event_rows)
    detected_events = events[events["detected_before_failure"]]
    metrics = {
        "units": float(len(events)),
        "detection_rate": float(events["detected_before_failure"].mean()),
        "healthy_false_alarm_rate": healthy_false_alarm_rate,
        "median_lead_time_cycles": float(detected_events["lead_time_cycles"].median())
        if not detected_events.empty
        else float("nan"),
        "mean_lead_time_cycles": float(detected_events["lead_time_cycles"].mean())
        if not detected_events.empty
        else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "healthy_cycle_assumption": f"cycle <= {healthy_cycles} assumed healthy",
        "degraded_rul_assumption": f"rul <= {degraded_rul} assumed degraded",
    }
    return metrics, events

