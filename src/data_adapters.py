"""Read-only adapters for C-MAPSS, SECOM, and PHM 2024 data.

The external datasets keep their own tasks and field meanings.  These adapters
do not route SECOM or PHM data through the C-MAPSS sequence pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.cmapss_data import (
    FEATURE_COLUMNS,
    add_test_rul,
    add_train_rul,
    load_telemetry,
    load_test_rul,
)


SECOM_MEASUREMENT_COUNT = 590
SECOM_FEATURES = tuple(
    f"measurement_{index:03d}" for index in range(1, SECOM_MEASUREMENT_COUNT + 1)
)
PHM_INPUTS = (
    "trq_measured",
    "oat",
    "mgt",
    "pa",
    "ias",
    "np",
    "ng",
)
PHM_TARGETS = {"classification": "faulty", "regression": "trq_margin"}


@dataclass(frozen=True)
class DatasetBundle:
    """Common row-oriented data view with targets and metadata kept separate.

    ``X`` contains only predictive inputs.  Identifiers, timestamps, source
    labels, and other descriptive fields live in ``metadata``.  ``y`` is
    absent for unlabeled prediction splits.
    """

    dataset_name: str
    split: str
    task_type: str
    X: pd.DataFrame
    y: pd.Series | None
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]
    source_to_model_mapping: Mapping[str, str]

    def __post_init__(self) -> None:
        if tuple(self.X.columns) != self.feature_names:
            raise ValueError("feature_names must match X columns in order")
        if len(self.X) != len(self.metadata):
            raise ValueError("X and metadata must have the same row count")
        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError("X and y must have the same row count")
        if not self.X.index.equals(self.metadata.index):
            raise ValueError("X and metadata indexes must be aligned")
        if self.y is not None and not self.X.index.equals(self.y.index):
            raise ValueError("X and y indexes must be aligned")


def _raw_root(data_dir: str | Path) -> Path:
    return Path(data_dir)


def load_secom_bundle(data_dir: str | Path = "data/raw/secom") -> DatasetBundle:
    """Parse and row-align the two SECOM source files without imputing values."""

    root = _raw_root(data_dir)
    measurements = pd.read_csv(
        root / "secom.data",
        sep=r"\s+",
        header=None,
        names=list(SECOM_FEATURES),
        na_values=["NaN"],
    )
    labels = pd.read_csv(
        root / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["source_label", "timestamp"],
        dtype={"timestamp": "string"},
    )
    if measurements.shape[1] != SECOM_MEASUREMENT_COUNT:
        raise ValueError(
            f"SECOM has {measurements.shape[1]} measurements; expected "
            f"{SECOM_MEASUREMENT_COUNT}"
        )
    if len(measurements) != len(labels):
        raise ValueError(
            "SECOM row mismatch: "
            f"{len(measurements)} measurement rows vs {len(labels)} label rows"
        )
    if labels["source_label"].isna().any():
        raise ValueError("SECOM source labels contain missing values")
    labels["source_label"] = labels["source_label"].astype("int64")
    unexpected_labels = sorted(set(labels["source_label"].unique()) - {-1, 1})
    if unexpected_labels:
        raise ValueError(f"Unexpected SECOM source labels: {unexpected_labels}")
    parsed_timestamps = pd.to_datetime(
        labels["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="raise"
    )
    if parsed_timestamps.isna().any():
        raise ValueError("SECOM contains timestamps that could not be parsed")

    # The SECOM measurement file has no sample key.  The documented pairing is
    # row-for-row, so the zero-based row number is kept as metadata.
    sample_id = pd.Series(np.arange(len(measurements), dtype=np.int64), name="sample_id")
    metadata = pd.DataFrame(
        {
            "sample_id": sample_id,
            "source_label": labels["source_label"].to_numpy(),
            # Keep the exact source timestamp text; preparation parses a copy
            # solely to create a chronological split.
            "timestamp": labels["timestamp"].astype(str).to_numpy(),
        }
    )
    target = labels["source_label"].map({-1: 0, 1: 1}).astype("int8")
    target.name = "faulty"
    mapping = {
        name: (
            f"secom.data whitespace field {position} (one-based); the raw file "
            "has no feature header"
        )
        for position, name in enumerate(SECOM_FEATURES, start=1)
    }
    return DatasetBundle(
        dataset_name="secom",
        split="all_labeled_rows",
        task_type="binary_classification",
        X=measurements,
        y=target,
        metadata=metadata,
        feature_names=SECOM_FEATURES,
        source_to_model_mapping=mapping,
    )


def load_phm_bundle(
    split: str,
    task: str,
    data_dir: str | Path = "data/raw/phm2024/extracted",
) -> DatasetBundle:
    """Load one PHM task while keeping IDs and target columns outside X.

    The official test and validation files have no local labels, so this
    adapter always returns ``y=None`` for those splits.
    """

    if split not in {"train", "test", "validation"}:
        raise ValueError("PHM split must be train, test, or validation")
    if task not in PHM_TARGETS:
        raise ValueError("PHM task must be classification or regression")

    root = _raw_root(data_dir)
    input_paths = {
        "train": root / "training" / "X_train.csv",
        "test": root / "test" / "X_test.csv",
        "validation": root / "validation" / "X_validation.csv",
    }
    X_source = pd.read_csv(input_paths[split])
    expected_input_columns = ["id", *PHM_INPUTS]
    if X_source.columns.tolist() != expected_input_columns:
        raise ValueError(
            f"Unexpected PHM {split} input columns: {X_source.columns.tolist()}"
        )
    if X_source["id"].isna().any() or not X_source["id"].is_unique:
        raise ValueError(f"PHM {split} sample IDs must be present and unique")
    for column in PHM_INPUTS:
        X_source[column] = pd.to_numeric(X_source[column], errors="raise")
        if not np.isfinite(X_source[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"PHM {split} input {column} has a missing or infinite value")

    y: pd.Series | None = None
    if split == "train":
        y_source = pd.read_csv(root / "training" / "y_train.csv")
        expected_target_columns = ["id", "faulty", "trq_margin"]
        if y_source.columns.tolist() != expected_target_columns:
            raise ValueError(
                "Unexpected PHM training target columns: "
                f"{y_source.columns.tolist()}"
            )
        if y_source["id"].isna().any() or not y_source["id"].is_unique:
            raise ValueError("PHM training target IDs must be present and unique")
        if y_source[["faulty", "trq_margin"]].isna().any().any():
            raise ValueError("PHM training targets contain missing values")
        y_source["faulty"] = pd.to_numeric(y_source["faulty"], errors="raise")
        y_source["trq_margin"] = pd.to_numeric(y_source["trq_margin"], errors="raise")
        faulty_values = set(y_source["faulty"].unique())
        if not faulty_values <= {0, 1}:
            raise ValueError(f"PHM faulty target must be binary 0/1; found {faulty_values}")
        if not np.isfinite(y_source["trq_margin"].to_numpy(dtype=np.float64)).all():
            raise ValueError("PHM trq_margin contains an infinite value")
        if set(X_source["id"]) != set(y_source["id"]):
            missing_targets = sorted(set(X_source["id"]) - set(y_source["id"]))[:10]
            extra_targets = sorted(set(y_source["id"]) - set(X_source["id"]))[:10]
            raise ValueError(
                "PHM input/target ID coverage mismatch; "
                f"missing targets (first 10)={missing_targets}, "
                f"extra targets (first 10)={extra_targets}"
            )
        # Reindex by the supplied sample ID rather than relying on source order.
        aligned_targets = y_source.set_index("id").loc[X_source["id"].to_numpy()]
        if not np.array_equal(aligned_targets.index.to_numpy(), X_source["id"].to_numpy()):
            raise ValueError("PHM targets could not be aligned to training input IDs")
        target_column = PHM_TARGETS[task]
        y = aligned_targets[target_column].reset_index(drop=True)
        y.name = target_column

    metadata = pd.DataFrame({"id": X_source["id"].to_numpy()})
    X = X_source.loc[:, PHM_INPUTS].copy().reset_index(drop=True)
    return DatasetBundle(
        dataset_name="phm2024",
        split=split,
        task_type=("binary_classification" if task == "classification" else "regression"),
        X=X,
        y=y,
        metadata=metadata,
        feature_names=PHM_INPUTS,
        source_to_model_mapping={name: name for name in PHM_INPUTS},
    )


def load_cmapss_bundle(
    subset: str,
    split: str,
    data_dir: str | Path = "data/raw/cmapss",
) -> DatasetBundle:
    """Expose C-MAPSS row features and RUL through the shared bundle shape.

    Existing sequence/window consumers should continue using ``cmapss_data``;
    this adapter is only a common inspection interface.
    """

    frame = load_telemetry(data_dir, subset, split)
    if split == "train":
        frame = add_train_rul(frame)
    elif split == "test":
        frame = add_test_rul(frame, load_test_rul(data_dir, subset))
    else:
        raise ValueError("C-MAPSS split must be train or test")
    X = frame.loc[:, FEATURE_COLUMNS].copy()
    target = frame["rul"].copy().rename("rul")
    metadata = frame.loc[:, ["unit_id", "cycle"]].copy()
    return DatasetBundle(
        dataset_name=f"cmapss_{subset.lower()}",
        split=split,
        task_type="remaining_useful_life_regression",
        X=X,
        y=target,
        metadata=metadata,
        feature_names=tuple(FEATURE_COLUMNS),
        source_to_model_mapping={name: name for name in FEATURE_COLUMNS},
    )
