"""Shared binary condition classifier and dataset-aware task router.

This module reuses the HistGradientBoostingClassifier approach from the
project's Diagnosis Agent.  A distinct model is fitted for each dataset; the
router selects the matching fitted model and never sends unrelated data into
the C-MAPSS Digital Twin or action-planning path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.cmapss_data import FEATURE_COLUMNS
from src.data_adapters import (
    DatasetBundle,
    load_cmapss_bundle,
    load_phm_bundle,
    load_secom_bundle,
)
from src.diagnosis import DEFAULT_WARNING_RUL


CONDITION_TARGET = "condition_alert"
CONDITION_POSITIVE_MEANING = "dataset-specific alert/fail/faulty condition"
MODEL_CONFIG: dict[str, Any] = {
    "learning_rate": 0.08,
    "max_iter": 250,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "random_state": 42,
    # Avoid the estimator creating an implicit random validation split. The
    # explicit validation partition selects the same score threshold for all.
    "early_stopping": False,
}


def make_condition_bundle(source: DatasetBundle) -> DatasetBundle:
    """Map each source target to binary ``condition_alert`` without leakage."""

    if source.dataset_name.startswith("cmapss_"):
        if source.y is None:
            target = None
        else:
            # The existing Diagnosis Agent defines healthy as RUL > 125.
            # Therefore degraded and critical rows are project-derived alerts.
            target = (source.y.to_numpy(dtype=float) <= DEFAULT_WARNING_RUL).astype(
                "int8"
            )
    elif source.dataset_name == "secom":
        target = None if source.y is None else source.y.to_numpy(dtype="int8")
    elif source.dataset_name == "phm2024":
        if source.task_type not in {"binary_classification", "classification"}:
            raise ValueError("PHM condition classification requires the faulty target")
        target = None if source.y is None else source.y.to_numpy(dtype="int8")
    else:
        raise ValueError(f"Unsupported condition dataset: {source.dataset_name}")

    y = None if target is None else pd.Series(target, name=CONDITION_TARGET)
    if y is not None and not set(y.unique()) <= {0, 1}:
        raise ValueError("condition_alert target must contain only 0 and 1")
    X = source.X.reset_index(drop=True).copy()
    metadata = source.metadata.reset_index(drop=True).copy()
    if y is not None:
        y = y.reset_index(drop=True)
    return DatasetBundle(
        dataset_name=source.dataset_name,
        split=source.split,
        task_type="binary_condition_classification",
        X=X,
        y=y,
        metadata=metadata,
        feature_names=tuple(X.columns),
        source_to_model_mapping=source.source_to_model_mapping,
    )


def load_condition_dataset(dataset_name: str) -> DatasetBundle:
    """Load labeled development data; never opens official C-MAPSS test RUL."""

    if dataset_name == "cmapss_fd001":
        source = load_cmapss_bundle("FD001", "train")
    elif dataset_name == "secom":
        source = load_secom_bundle()
    elif dataset_name == "phm2024":
        source = load_phm_bundle("train", "classification")
    else:
        raise ValueError(
            "dataset_name must be cmapss_fd001, secom, or phm2024"
        )
    return make_condition_bundle(source)


def split_condition_indices(
    bundle: DatasetBundle, seed: int = 42
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create deterministic fit/threshold-validation/final-test partitions.

    C-MAPSS is grouped by engine. SECOM uses chronological 60/20/20 blocks.
    PHM uses stratified folds of exact feature groups so duplicate rows remain
    together. PHM IDs are never treated as asset IDs.
    """

    if bundle.y is None:
        raise ValueError("Labeled development data is required to create splits")
    y = bundle.y.to_numpy(dtype=np.int8)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Every development dataset split must have both classes")

    if bundle.dataset_name.startswith("cmapss_"):
        units = np.asarray(sorted(bundle.metadata["unit_id"].unique()), dtype=int)
        shuffled = np.random.default_rng(seed).permutation(units)
        train_end = int(round(0.60 * len(shuffled)))
        validation_end = int(round(0.80 * len(shuffled)))
        unit_parts = {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:validation_end],
            "test": shuffled[validation_end:],
        }
        unit_values = bundle.metadata["unit_id"].to_numpy()
        splits = {
            name: np.flatnonzero(np.isin(unit_values, part))
            for name, part in unit_parts.items()
        }
        if set(unit_parts["train"]) & set(unit_parts["validation"]):
            raise AssertionError("C-MAPSS train/validation engine overlap")
        if set(unit_parts["train"]) & set(unit_parts["test"]):
            raise AssertionError("C-MAPSS train/test engine overlap")
        if set(unit_parts["validation"]) & set(unit_parts["test"]):
            raise AssertionError("C-MAPSS validation/test engine overlap")
        details = {
            "strategy": "deterministic 60/20/20 engine-unit split from FD001 training trajectories",
            "seed": int(seed),
            "official_final_test_used": False,
            "engine_ids": {
                name: sorted(int(value) for value in part)
                for name, part in unit_parts.items()
            },
        }
    elif bundle.dataset_name == "secom":
        timestamps = pd.to_datetime(
            bundle.metadata["timestamp"],
            format="%d/%m/%Y %H:%M:%S",
            errors="raise",
        )
        order = np.argsort(timestamps.to_numpy(), kind="stable")
        train_end = int(0.60 * len(order))
        validation_end = int(0.80 * len(order))
        splits = {
            "train": order[:train_end],
            "validation": order[train_end:validation_end],
            "test": order[validation_end:],
        }
        details = {
            "strategy": "chronological 60/20/20 split sorted by the supplied timestamp",
            "seed": None,
            "timestamp_min": str(timestamps.min()),
            "timestamp_max": str(timestamps.max()),
            "threshold_validation_is_chronologically_before_test": True,
        }
    elif bundle.dataset_name == "phm2024":
        hashes = pd.util.hash_pandas_object(bundle.X, index=False).to_numpy(
            dtype=np.uint64
        )
        unique_hashes, inverse, _ = np.unique(
            hashes, return_inverse=True, return_counts=True
        )
        unique_feature_rows = int((~bundle.X.duplicated()).sum())
        if len(unique_hashes) != unique_feature_rows:
            raise ValueError(
                "Feature hash collision detected; cannot safely group duplicate rows"
            )
        positive_by_group = np.bincount(inverse, weights=(y == 1))
        negative_by_group = np.bincount(inverse, weights=(y == 0))
        mixed_group_count = int(
            ((positive_by_group > 0) & (negative_by_group > 0)).sum()
        )
        group_labels = (positive_by_group > negative_by_group).astype(np.int8)
        if len(np.unique(group_labels)) != 2:
            raise ValueError("PHM feature groups do not contain both target classes")
        fold_for_group = np.full(len(unique_hashes), -1, dtype=np.int8)
        group_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (_, held_out_groups) in enumerate(
            group_cv.split(np.zeros(len(unique_hashes)), group_labels)
        ):
            fold_for_group[held_out_groups] = fold
        row_fold = fold_for_group[inverse]
        splits = {
            "test": np.flatnonzero(row_fold == 0),
            "validation": np.flatnonzero(row_fold == 1),
            "train": np.flatnonzero(row_fold >= 2),
        }
        group_parts = {name: set(inverse[index]) for name, index in splits.items()}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if group_parts[left] & group_parts[right]:
                raise AssertionError(f"PHM duplicate feature groups overlap: {left}/{right}")
        details = {
            "strategy": "five stratified folds of exact seven-feature groups; fold 0 test, fold 1 threshold validation, folds 2-4 train",
            "seed": int(seed),
            "unique_feature_groups": int(len(unique_hashes)),
            "duplicate_rows_beyond_first": int(len(bundle.X) - unique_feature_rows),
            "mixed_target_duplicate_groups": mixed_group_count,
            "asset_independent": False,
        }
    else:
        raise ValueError(f"No split strategy defined for {bundle.dataset_name}")

    if sorted(np.concatenate(list(splits.values())).tolist()) != list(range(len(y))):
        raise AssertionError("Splits must cover every row exactly once")
    for name, indices in splits.items():
        if set(np.unique(y[indices])) != {0, 1}:
            raise ValueError(f"{name} split does not contain both classes")
    return splits, details


def subset_condition_bundle(
    bundle: DatasetBundle, indices: np.ndarray, split: str
) -> DatasetBundle:
    """Return an aligned row subset without moving IDs or labels into X."""

    X = bundle.X.iloc[indices].reset_index(drop=True).copy()
    metadata = bundle.metadata.iloc[indices].reset_index(drop=True).copy()
    y = None if bundle.y is None else bundle.y.iloc[indices].reset_index(drop=True).copy()
    return DatasetBundle(
        dataset_name=bundle.dataset_name,
        split=split,
        task_type=bundle.task_type,
        X=X,
        y=y,
        metadata=metadata,
        feature_names=tuple(X.columns),
        source_to_model_mapping=bundle.source_to_model_mapping,
    )


@dataclass
class CmapssConditionPreprocessor:
    """Training-only constant removal and scaling for C-MAPSS telemetry."""

    input_features: tuple[str, ...]
    retained_features: tuple[str, ...]
    dropped_constant: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X_train: pd.DataFrame) -> "CmapssConditionPreprocessor":
        if tuple(X_train.columns) != tuple(FEATURE_COLUMNS):
            raise ValueError("Unexpected C-MAPSS feature names or order")
        values = X_train.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("C-MAPSS training inputs must be finite")
        constant_mask = X_train.nunique(dropna=False).to_numpy() <= 1
        keep = np.flatnonzero(~constant_mask)
        if not len(keep):
            raise ValueError("All C-MAPSS training features are constant")
        retained = tuple(X_train.columns[position] for position in keep)
        kept_values = values[:, keep]
        mean = kept_values.mean(axis=0)
        scale = kept_values.std(axis=0, ddof=0)
        scale = np.where(scale == 0, 1.0, scale)
        return cls(
            input_features=tuple(X_train.columns),
            retained_features=retained,
            dropped_constant=tuple(
                name for name, is_constant in zip(X_train.columns, constant_mask) if is_constant
            ),
            mean=mean,
            scale=scale,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if tuple(X.columns) != self.input_features:
            raise ValueError("C-MAPSS feature names/order changed after fitting")
        values = X.loc[:, self.retained_features].to_numpy(dtype=np.float64)
        transformed = (values - self.mean) / self.scale
        return pd.DataFrame(transformed, columns=self.retained_features, index=X.index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dropped_constant_training_columns": list(self.dropped_constant),
            "retained_features": list(self.retained_features),
            "mean_by_feature": {
                name: float(value) for name, value in zip(self.retained_features, self.mean)
            },
            "scale_by_feature": {
                name: float(value) for name, value in zip(self.retained_features, self.scale)
            },
            "fit_scope": "fit engine units only",
        }


def select_f1_threshold(y_validation: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Choose the probability threshold maximizing positive-class F1 on validation."""

    from sklearn.metrics import precision_recall_curve

    actual = np.asarray(y_validation, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if len(actual) != len(scores) or set(np.unique(actual)) != {0, 1}:
        raise ValueError("Threshold selection requires aligned validation rows with both classes")
    if not np.isfinite(scores).all():
        raise ValueError("Validation probabilities must be finite")
    precision, recall, thresholds = precision_recall_curve(actual, scores)
    if len(thresholds) == 0:
        raise ValueError("No threshold candidates are available")
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    best_f1 = float(np.max(f1_values))
    best_positions = np.flatnonzero(np.isclose(f1_values, best_f1, rtol=0, atol=1e-12))
    # A higher threshold wins exact ties, so tied choices produce fewer alerts.
    best_position = best_positions[np.argmax(thresholds[best_positions])]
    return float(thresholds[best_position]), best_f1


@dataclass
class ConditionClassificationModel:
    """One dataset's fitted preprocessing, binary model, and validation threshold."""

    dataset_name: str
    input_feature_names: tuple[str, ...]
    preprocessor: Any
    classifier: HistGradientBoostingClassifier
    model_feature_names: tuple[str, ...]
    decision_threshold: float = 0.5
    threshold_validation_f1: float | None = None
    source_to_model_mapping: dict[str, str] | None = None

    @classmethod
    def fit(
        cls,
        dataset_name: str,
        preprocessor: Any,
        X_train_transformed: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        source_to_model_mapping: dict[str, str] | None = None,
    ) -> "ConditionClassificationModel":
        labels = np.asarray(y_train, dtype=np.int8)
        if len(X_train_transformed) != len(labels):
            raise ValueError("Training features and condition labels are misaligned")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Training data must contain both condition classes")
        counts = np.bincount(labels, minlength=2)
        if (counts == 0).any():
            raise ValueError("Both condition classes are required for training")
        sample_weights = np.where(
            labels == 1,
            len(labels) / (2.0 * counts[1]),
            len(labels) / (2.0 * counts[0]),
        )
        classifier = HistGradientBoostingClassifier(**MODEL_CONFIG)
        classifier.fit(
            X_train_transformed.to_numpy(dtype=np.float64),
            labels,
            sample_weight=sample_weights,
        )
        return cls(
            dataset_name=dataset_name,
            input_feature_names=tuple(preprocessor.input_features)
            if hasattr(preprocessor, "input_features")
            else tuple(X_train_transformed.columns),
            preprocessor=preprocessor,
            classifier=classifier,
            model_feature_names=tuple(X_train_transformed.columns),
            source_to_model_mapping=source_to_model_mapping or {},
        )

    def positive_probability(self, transformed: pd.DataFrame) -> np.ndarray:
        if tuple(transformed.columns) != self.model_feature_names:
            raise ValueError("Transformed feature names do not match the fitted classifier")
        class_positions = np.flatnonzero(self.classifier.classes_ == 1)
        if len(class_positions) != 1:
            raise ValueError("Fitted classifier does not expose positive class 1")
        return self.classifier.predict_proba(
            transformed.to_numpy(dtype=np.float64)
        )[:, int(class_positions[0])]

    def transform_bundle(self, bundle: DatasetBundle) -> pd.DataFrame:
        if bundle.dataset_name != self.dataset_name:
            raise ValueError(
                f"Model for {self.dataset_name} cannot predict {bundle.dataset_name} data"
            )
        if tuple(bundle.X.columns) != self.input_feature_names:
            raise ValueError("Prediction input columns do not match the fitted adapter")
        transformed = self.preprocessor.transform(bundle.X)
        if tuple(transformed.columns) != self.model_feature_names:
            raise ValueError("Preprocessor output columns changed after fitting")
        return transformed

    def predict(self, bundle: DatasetBundle) -> "StructuredConditionPrediction":
        transformed = self.transform_bundle(bundle)
        positive_probability = self.positive_probability(transformed)
        predicted = (positive_probability >= self.decision_threshold).astype("int8")
        output = bundle.metadata.reset_index(drop=True).copy()
        output["condition_alert_probability"] = positive_probability
        output["decision_threshold"] = float(self.decision_threshold)
        output[CONDITION_TARGET] = predicted
        return StructuredConditionPrediction(
            dataset_name=self.dataset_name,
            split=bundle.split,
            predictions=output,
        )

    def set_validation_threshold(
        self, y_validation: np.ndarray, validation_probabilities: np.ndarray
    ) -> None:
        self.decision_threshold, self.threshold_validation_f1 = select_f1_threshold(
            y_validation, validation_probabilities
        )

    def artifact_summary(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "classifier": "sklearn.ensemble.HistGradientBoostingClassifier",
            "model_config": dict(MODEL_CONFIG),
            "input_feature_names": list(self.input_feature_names),
            "model_feature_names": list(self.model_feature_names),
            "source_to_model_mapping": self.source_to_model_mapping or {},
            "decision_threshold": float(self.decision_threshold),
            "threshold_selection": "maximize positive-class F1 on validation probabilities; highest threshold breaks ties",
            "threshold_validation_f1": self.threshold_validation_f1,
            "preprocessing": self.preprocessor.to_dict(),
        }


@dataclass(frozen=True)
class StructuredConditionPrediction:
    """Per-row metadata plus alert probability, selected threshold, and label."""

    dataset_name: str
    split: str
    predictions: pd.DataFrame


class ConditionTaskRouter:
    """Route a feature bundle to the separately fitted model for its dataset."""

    def __init__(self) -> None:
        self._models: dict[str, ConditionClassificationModel] = {}

    def register(self, model: ConditionClassificationModel) -> None:
        if model.dataset_name in self._models:
            raise ValueError(f"A model is already registered for {model.dataset_name}")
        self._models[model.dataset_name] = model

    def predict(self, bundle: DatasetBundle) -> StructuredConditionPrediction:
        try:
            model = self._models[bundle.dataset_name]
        except KeyError as error:
            raise ValueError(
                f"No condition classifier is registered for {bundle.dataset_name}"
            ) from error
        return model.predict(bundle)


def binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the shared binary metrics and positive-class confusion matrix."""

    actual = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    predicted = (scores >= float(threshold)).astype(np.int8)
    roc_auc: float | None
    try:
        roc_auc = float(roc_auc_score(actual, scores, sample_weight=sample_weight))
    except ValueError:
        roc_auc = None
    return {
        "pr_auc_average_precision": float(
            average_precision_score(actual, scores, sample_weight=sample_weight)
        ),
        "positive_precision": float(
            precision_score(
                actual, predicted, zero_division=0, sample_weight=sample_weight
            )
        ),
        "positive_recall": float(
            recall_score(actual, predicted, zero_division=0, sample_weight=sample_weight)
        ),
        "positive_f1": float(
            f1_score(actual, predicted, zero_division=0, sample_weight=sample_weight)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted, sample_weight=sample_weight)
        ),
        "roc_auc": roc_auc,
        "confusion_matrix_labels_0_1": confusion_matrix(
            actual, predicted, labels=[0, 1], sample_weight=sample_weight
        ).tolist(),
        "threshold": float(threshold),
    }
