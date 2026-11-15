"""Health-state diagnosis agent for the C-MAPSS development benchmark.

SCOPE:
This agent classifies the overall engine health-state (healthy, degraded, critical).
Direct physical component-fault diagnosis is NOT claimed on FD001 because physical fault
labels are not provided by NASA C-MAPSS. Any fault-injection experiment is strictly
synthetic and clearly labeled as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


HEALTH_STATES = ("healthy", "degraded", "critical")
UNCERTAIN_STATE = "uncertain"
DEFAULT_WARNING_RUL = 125


def health_state_from_rul(
    rul: np.ndarray | pd.Series,
    warning_rul: int = 125,
    critical_rul: int = 25,
) -> np.ndarray:
    """Create project labels from RUL thresholds.

    These are project-defined health states, not labels supplied by C-MAPSS.
    """
    if critical_rul >= warning_rul:
        raise ValueError("critical_rul must be lower than warning_rul")
    values = np.asarray(rul, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("RUL labels must be finite")
    return np.select(
        [values <= critical_rul, values <= warning_rul],
        ["critical", "degraded"],
        default="healthy",
    )


def compute_calibration_metrics(
    actual: np.ndarray,
    probabilities: pd.DataFrame | np.ndarray,
    classes: Sequence[str] = HEALTH_STATES,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute multi-class Brier score and Expected Calibration Error (ECE)."""
    class_list = list(classes)
    if isinstance(probabilities, pd.DataFrame):
        probs = probabilities.reindex(columns=class_list, fill_value=0.0).to_numpy(dtype=float)
    else:
        probs = np.asarray(probabilities, dtype=float)
    actual = np.asarray(actual)
    if probs.ndim != 2 or probs.shape != (len(actual), len(class_list)):
        raise ValueError("probabilities must have one column per configured class and one row per label")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probs.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("each probability row must have positive mass")
    probs = probs / row_sums[:, None]

    # 1-hot encode ground truth
    y_true_1hot = np.zeros_like(probs)
    for c_idx, c_name in enumerate(class_list):
        y_true_1hot[actual == c_name, c_idx] = 1.0

    # Multi-class Brier score: mean squared difference between probabilities and 1-hot ground truth
    brier_score = float(np.mean(np.sum((probs - y_true_1hot) ** 2, axis=1)))

    # Expected Calibration Error (ECE) for top-1 prediction
    confidences = np.max(probs, axis=1)
    predictions = np.asarray([class_list[i] for i in np.argmax(probs, axis=1)])
    accuracies = (predictions == actual).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return {
        "brier_score": brier_score,
        "expected_calibration_error": float(ece),
        "mean_confidence": float(np.mean(confidences)),
    }


@dataclass
class HealthDiagnosisAgent:
    """Classify the health state represented by a telemetry window."""

    model: HistGradientBoostingClassifier
    feature_names: list[str]
    warning_rul: int
    critical_rul: int
    confidence_threshold: float = 0.50
    entropy_threshold: float = 0.85

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        rul_targets: np.ndarray,
        feature_names: list[str],
        warning_rul: int = 125,
        critical_rul: int = 25,
        confidence_threshold: float = 0.50,
        entropy_threshold: float = 0.85,
    ) -> "HealthDiagnosisAgent":
        labels = health_state_from_rul(rul_targets, warning_rul, critical_rul)
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
        class_counts = pd.Series(labels).value_counts()
        observed_classes = list(model.classes_) if hasattr(model, "classes_") else sorted(class_counts.index)
        # Give each observed class equal aggregate weight. This avoids depending
        # on classes absent from a small/grouped training fold.
        weights = np.asarray(
            [len(labels) / (len(observed_classes) * class_counts[label]) for label in labels],
            dtype=float,
        )
        model.fit(features, labels, sample_weight=weights)
        return cls(
            model,
            feature_names,
            warning_rul,
            critical_rul,
            confidence_threshold=confidence_threshold,
            entropy_threshold=entropy_threshold,
        )

    def predict(self, features: np.ndarray, allow_uncertain: bool = False) -> np.ndarray:
        if not allow_uncertain:
            return self.model.predict(features)
        preds, _, _ = self.predict_with_uncertainty(features)
        return preds

    def predict_proba(self, features: np.ndarray) -> pd.DataFrame:
        probabilities = self.model.predict_proba(features)
        return pd.DataFrame(probabilities, columns=self.model.classes_)

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict health state with calibrated confidence and entropy thresholding."""
        probs = self.predict_proba(features)
        values = probs.to_numpy()
        confidences = values.max(axis=1)
        k = values.shape[1]
        eps = 1e-12
        entropy = -np.sum(values * np.log2(values + eps), axis=1) / np.log2(k)
        raw_preds = probs.idxmax(axis=1).to_numpy()

        uncertain_mask = (confidences < self.confidence_threshold) | (entropy > self.entropy_threshold)
        filtered_preds = raw_preds.copy()
        filtered_preds[uncertain_mask] = UNCERTAIN_STATE
        return filtered_preds, confidences, entropy

    def explain(self, features: np.ndarray, index: int = 0) -> str:
        probabilities = self.predict_proba(features).iloc[index]
        state = probabilities.idxmax()
        confidence = float(probabilities.max())
        eps = 1e-12
        probs_vec = probabilities.to_numpy()
        k = len(probs_vec)
        norm_entropy = float(-np.sum(probs_vec * np.log2(probs_vec + eps)) / np.log2(k))
        uncertain = (confidence < self.confidence_threshold) or (norm_entropy > self.entropy_threshold)

        ranking = probabilities.sort_values(ascending=False)
        runner_up = ranking.index[1] if len(ranking) > 1 else "none"
        status_tag = f"UNCERTAIN ({state})" if uncertain else state
        return (
            f"Predicted state: {status_tag}; confidence={confidence:.3f}; "
            f"normalized entropy={norm_entropy:.3f}; next most likely state={runner_up}. "
            "This agent classifies operational health state and does not claim physical component-fault diagnosis."
        )


class SyntheticFaultBenchmark:
    """Controlled synthetic fault injection for algorithm testing only.

    Explicit disclaimer: NASA C-MAPSS does not contain true component fault labels.
    All labels generated here are synthetic and purely for testing multi-fault classifiers.
    """
    is_synthetic: bool = True

    @staticmethod
    def inject_synthetic_faults(features: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        fault_types = ["synthetic_normal", "synthetic_hpc_degradation", "synthetic_fan_erosion"]
        labels = []
        modified = features.copy()
        for i in range(len(features)):
            choice = rng.choice(fault_types)
            labels.append(choice)
            if choice == "synthetic_hpc_degradation":
                modified[i, :2] += rng.normal(0.5, 0.1, 2)
            elif choice == "synthetic_fan_erosion":
                modified[i, 2:4] -= rng.normal(0.5, 0.1, 2)
        return modified, np.asarray(labels)
