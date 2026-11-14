"""Classifier for controlled synthetic sensor-fault scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.fault_injection import SYNTHETIC_FAULT_LABELS


@dataclass
class SyntheticSensorFaultAgent:
    """Recognize injected sensor bias, drift, stuck-value, and spike patterns.

    The classes are synthetic signal corruptions, not physical engine faults.
    """

    model: RandomForestClassifier
    feature_names: list[str]
    labels: tuple[str, ...] = SYNTHETIC_FAULT_LABELS

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str],
        *,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> "SyntheticSensorFaultAgent":
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=str)
        if values.ndim != 2 or values.shape[1] != len(feature_names):
            raise ValueError("features must have shape (samples, len(feature_names))")
        if len(values) != len(targets) or not len(targets):
            raise ValueError("features and labels must have the same non-zero length")
        if not np.isfinite(values).all():
            raise ValueError("synthetic fault features must be finite")
        unknown = sorted(set(targets) - set(SYNTHETIC_FAULT_LABELS))
        if unknown:
            raise ValueError(f"unknown synthetic fault labels: {unknown}")
        if len(set(targets)) < 2:
            raise ValueError("at least two synthetic classes are required")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features=0.7,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(values, targets)
        return cls(model=model, feature_names=list(feature_names))

    def predict_proba(self, features: np.ndarray) -> pd.DataFrame:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("features have the wrong shape for this fault agent")
        probabilities = self.model.predict_proba(values)
        result = pd.DataFrame(
            0.0, index=np.arange(len(values)), columns=self.labels, dtype=float
        )
        for index, label in enumerate(self.model.classes_):
            result[str(label)] = probabilities[:, index]
        return result

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_proba(features)
        values = probabilities.to_numpy(dtype=float)
        positions = np.argmax(values, axis=1)
        labels = np.asarray(self.labels, dtype=object)[positions]
        confidence = values[np.arange(len(values)), positions]
        return labels, confidence
