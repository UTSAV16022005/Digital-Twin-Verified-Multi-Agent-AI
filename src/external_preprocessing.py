"""Train-only preprocessing helpers for SECOM and PHM 2024."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SecomPreprocessor:
    """Median-impute, select, then standardize SECOM features.

    Every learned quantity in this object is fit from one training partition:
    missing-value medians, the all-missing/constant checks, ANOVA feature
    ranking, and scaling statistics.
    """

    input_features: tuple[str, ...]
    retained_features: tuple[str, ...]
    dropped_all_missing: tuple[str, ...]
    dropped_constant: tuple[str, ...]
    medians: np.ndarray
    selected_features: tuple[str, ...]
    selection_scores: tuple[float, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    top_k: int

    @classmethod
    def fit(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        top_k: int = 40,
    ) -> "SecomPreprocessor":
        if top_k < 1:
            raise ValueError("top_k must be positive")
        values = X_train.to_numpy(dtype=np.float64)
        labels = np.asarray(y_train)
        if values.ndim != 2 or len(values) != len(labels):
            raise ValueError("SECOM training features and labels are misaligned")
        if not np.isfinite(values[~np.isnan(values)]).all():
            raise ValueError("SECOM features contain infinite values")

        columns = tuple(str(name) for name in X_train.columns)
        all_missing_mask = np.isnan(values).all(axis=0)
        dropped_all_missing = tuple(
            name for name, drop in zip(columns, all_missing_mask) if drop
        )
        retained_positions = np.flatnonzero(~all_missing_mask)
        if len(retained_positions) == 0:
            raise ValueError("All SECOM features are missing in this training split")
        retained_values = values[:, retained_positions]
        medians = np.nanmedian(retained_values, axis=0)
        missing_rows, missing_columns = np.where(np.isnan(retained_values))
        imputed_values = retained_values.copy()
        imputed_values[missing_rows, missing_columns] = medians[missing_columns]
        retained_names = tuple(columns[position] for position in retained_positions)

        constant_mask = np.ptp(imputed_values, axis=0) == 0
        dropped_constant = tuple(
            name for name, drop in zip(retained_names, constant_mask) if drop
        )
        keep_positions = np.flatnonzero(~constant_mask)
        if len(keep_positions) == 0:
            raise ValueError("All non-missing SECOM training features are constant")
        candidate_values = imputed_values[:, keep_positions]
        candidate_names = tuple(retained_names[position] for position in keep_positions)

        classes = np.unique(labels)
        if len(classes) != 2:
            raise ValueError("SECOM training data must contain both classes")
        # Univariate ANOVA F scores rank features without using held-out labels.
        # A zero within-class sum of squares with separated class means has an
        # infinite score and is ranked first; stable ordering breaks ties.
        overall_mean = candidate_values.mean(axis=0)
        ss_between = np.zeros(candidate_values.shape[1], dtype=np.float64)
        ss_within = np.zeros(candidate_values.shape[1], dtype=np.float64)
        for class_value in classes:
            class_values = candidate_values[labels == class_value]
            class_mean = class_values.mean(axis=0)
            ss_between += len(class_values) * (class_mean - overall_mean) ** 2
            ss_within += ((class_values - class_mean) ** 2).sum(axis=0)
        numerator = ss_between / (len(classes) - 1)
        denominator_df = len(labels) - len(classes)
        if denominator_df <= 0:
            raise ValueError("Not enough SECOM training rows for feature selection")
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = numerator / (ss_within / denominator_df)
        scores = np.where(np.isnan(scores), -np.inf, scores)
        ranking = np.argsort(-scores, kind="stable")
        selected_positions = ranking[: min(top_k, len(ranking))]
        selected_features = tuple(candidate_names[position] for position in selected_positions)
        selected_scores = tuple(float(scores[position]) for position in selected_positions)
        selected_values = candidate_values[:, selected_positions]
        scaler_mean = selected_values.mean(axis=0)
        scaler_scale = selected_values.std(axis=0, ddof=0)
        scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)

        return cls(
            input_features=columns,
            retained_features=retained_names,
            dropped_all_missing=dropped_all_missing,
            dropped_constant=dropped_constant,
            medians=medians,
            selected_features=selected_features,
            selection_scores=selected_scores,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            top_k=top_k,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [name for name in self.input_features if name not in X]
        if missing_columns:
            raise ValueError(f"Missing SECOM columns: {missing_columns[:10]}")
        values = X.loc[:, self.input_features].to_numpy(dtype=np.float64)
        retained_positions = [self.input_features.index(name) for name in self.retained_features]
        retained_values = values[:, retained_positions].copy()
        missing_rows, missing_columns = np.where(np.isnan(retained_values))
        retained_values[missing_rows, missing_columns] = self.medians[missing_columns]
        retained_positions_by_name = {
            name: position for position, name in enumerate(self.retained_features)
        }
        selection_positions = [
            retained_positions_by_name[name] for name in self.selected_features
        ]
        selected_values = retained_values[:, selection_positions]
        transformed = (selected_values - self.scaler_mean) / self.scaler_scale
        if not np.isfinite(transformed).all():
            raise ValueError("SECOM preprocessing produced non-finite values")
        return pd.DataFrame(transformed, columns=self.selected_features, index=X.index)

    def to_dict(self) -> dict[str, Any]:
        def score_value(value: float) -> float | str:
            return float(value) if np.isfinite(value) else "infinity" if value > 0 else "undefined"

        return {
            "imputer": "median, fitted on this training partition only",
            "all_missing_columns_removed": list(self.dropped_all_missing),
            "constant_columns_removed_after_training_median_imputation": list(
                self.dropped_constant
            ),
            "retained_after_missing_and_constant_checks": list(self.retained_features),
            "feature_selection": {
                "method": "univariate ANOVA F score (binary faulty target)",
                "top_k": self.top_k,
                "selected_features": list(self.selected_features),
                "selected_scores_in_feature_order": [
                    score_value(score) for score in self.selection_scores
                ],
            },
            "scaler": "standard score using population standard deviation; fit on this training partition only",
            "imputation_medians_by_retained_feature": {
                name: float(value)
                for name, value in zip(self.retained_features, self.medians)
            },
            "scaler_mean_by_selected_feature": {
                name: float(value)
                for name, value in zip(self.selected_features, self.scaler_mean)
            },
            "scaler_scale_by_selected_feature": {
                name: float(value)
                for name, value in zip(self.selected_features, self.scaler_scale)
            },
        }


@dataclass(frozen=True)
class Standardizer:
    """Small numerical standardizer whose statistics come only from training."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    fit_rows: int

    @classmethod
    def fit(cls, X_train: pd.DataFrame) -> "Standardizer":
        values = X_train.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("PHM training features must be finite before scaling")
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        scale = np.where(scale == 0, 1.0, scale)
        return cls(tuple(X_train.columns), mean, scale, int(len(X_train)))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if tuple(X.columns) != self.feature_names:
            raise ValueError("PHM feature columns/order changed after fitting")
        values = X.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("PHM features contain missing or infinite values")
        transformed = (values - self.mean) / self.scale
        return pd.DataFrame(transformed, columns=self.feature_names, index=X.index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "standard score using population standard deviation",
            "fit_split": "official labeled training split only",
            "fit_rows": self.fit_rows,
            "feature_names": list(self.feature_names),
            "mean_by_feature": {
                name: float(value) for name, value in zip(self.feature_names, self.mean)
            },
            "scale_by_feature": {
                name: float(value) for name, value in zip(self.feature_names, self.scale)
            },
            "missing_value_handling": "none required; training audit found no missing values",
            "feature_selection": "none; retain all seven source inputs",
        }
