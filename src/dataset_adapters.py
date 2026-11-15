"""Dataset adapters for external tabular PHM/quality benchmarks.

These adapters provide a common container while preserving each dataset's
native task and feature meanings. They do not force unrelated data into the
C-MAPSS RUL/time-window schema and do not fit preprocessing transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    """Aligned dataset interface retaining native values and task semantics."""

    name: str
    split: str
    task: str
    features: pd.DataFrame
    targets: pd.DataFrame | None
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        row_count = len(self.features)
        if len(self.metadata) != row_count:
            raise ValueError("metadata rows must align with feature rows")
        if self.targets is not None and len(self.targets) != row_count:
            raise ValueError("target rows must align with feature rows")
        if self.features.columns.duplicated().any():
            raise ValueError("feature names must be unique")
        if self.features.empty or not len(self.features.columns):
            raise ValueError("feature matrix must contain rows and columns")


def load_secom(data_dir: str | Path) -> DatasetBundle:
    """Load UCI SECOM into a binary pass/fail tabular task.

    Original label -1 (pass) maps to `faulty=0`; original label 1 (fail) maps
    to `faulty=1`. Missing sensor values are intentionally preserved for
    train-fold-only preprocessing later.
    """

    base = Path(data_dir)
    feature_path = base / "secom.data"
    label_path = base / "secom_labels.data"
    missing = [path.name for path in (feature_path, label_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing SECOM raw file(s) in {base}: {', '.join(missing)}"
        )

    features = pd.read_csv(
        feature_path,
        sep=r"\s+",
        header=None,
        na_values=["NaN"],
        engine="python",
    )
    features.columns = [f"measurement_{index:03d}" for index in range(1, len(features.columns) + 1)]
    label_rows: list[tuple[int, str]] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed SECOM label row {line_number}")
        try:
            label_value = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"Invalid SECOM label on row {line_number}") from exc
        label_rows.append((label_value, parts[1].strip().strip('"')))
    raw_labels = pd.DataFrame(label_rows, columns=["source_label", "timestamp"])
    if len(features) != len(raw_labels):
        raise ValueError(
            f"SECOM row mismatch: {len(features)} examples but "
            f"{len(raw_labels)} labels"
        )
    if not raw_labels["source_label"].isin([-1, 1]).all():
        raise ValueError("SECOM labels must be -1 (pass) or 1 (fail)")
    timestamps = pd.to_datetime(
        raw_labels["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    if timestamps.isna().any():
        raise ValueError("SECOM contains an invalid timestamp")

    targets = pd.DataFrame(
        {"faulty": raw_labels["source_label"].eq(1).astype("int8")}
    )
    metadata = pd.DataFrame(
        {
            "sample_id": np.arange(len(features), dtype=np.int64),
            "timestamp": timestamps,
            "source_label": raw_labels["source_label"].astype("int8"),
        }
    )
    feature_values = features.to_numpy(dtype=np.float64)
    if np.isinf(feature_values).any():
        raise ValueError("SECOM contains infinite sensor values")
    return DatasetBundle(
        name="secom",
        split="labeled_all",
        task="binary pass/fail process classification",
        features=features,
        targets=targets,
        metadata=metadata,
    )


_PHM_FILES = {
    "train": ("training", "X_train.csv", "y_train.csv"),
    "test": ("test", "X_test.csv", None),
    "validation": ("validation", "X_validation.csv", None),
}
_PHM_FEATURES = ("trq_measured", "oat", "mgt", "pa", "ias", "np", "ng")


def load_phm2024(data_dir: str | Path, split: str) -> DatasetBundle:
    """Load one PHM 2024 split for faulty classification/torque-margin regression.

    `id` remains metadata and is not treated as a predictive feature. Hidden
    test/validation targets are represented as None and are never fabricated.
    """

    normalized_split = split.lower()
    if normalized_split not in _PHM_FILES:
        raise ValueError("split must be train, test, or validation")
    folder, x_name, y_name = _PHM_FILES[normalized_split]
    split_dir = Path(data_dir) / "extracted" / folder
    x_path = split_dir / x_name
    if not x_path.is_file():
        raise FileNotFoundError(f"Missing PHM {normalized_split} features: {x_path}")
    raw_x = pd.read_csv(x_path)
    required_x = {"id", *_PHM_FEATURES}
    missing_x = sorted(required_x - set(raw_x.columns))
    if missing_x:
        raise ValueError(f"PHM {normalized_split} features missing columns: {missing_x}")
    if raw_x["id"].isna().any() or raw_x["id"].duplicated().any():
        raise ValueError(f"PHM {normalized_split} sample IDs must be unique/non-null")

    features = raw_x.loc[:, list(_PHM_FEATURES)].copy()
    metadata = raw_x.loc[:, ["id"]].rename(columns={"id": "sample_id"}).copy()
    targets: pd.DataFrame | None = None
    if y_name is not None:
        y_path = split_dir / y_name
        if not y_path.is_file():
            raise FileNotFoundError(f"Missing PHM training targets: {y_path}")
        raw_y = pd.read_csv(y_path)
        required_y = {"id", "faulty", "trq_margin"}
        missing_y = sorted(required_y - set(raw_y.columns))
        if missing_y:
            raise ValueError(f"PHM training targets missing columns: {missing_y}")
        if raw_y["id"].isna().any() or raw_y["id"].duplicated().any():
            raise ValueError("PHM training target IDs must be unique/non-null")
        if set(raw_x["id"]) != set(raw_y["id"]):
            raise ValueError("PHM feature and target sample IDs do not match")
        indexed_y = raw_y.set_index("id")
        targets = indexed_y.loc[raw_x["id"], ["faulty", "trq_margin"]].reset_index(drop=True)
        targets["faulty"] = targets["faulty"].astype("int8")
        if not targets["faulty"].isin([0, 1]).all():
            raise ValueError("PHM faulty target must be binary 0/1")
        if not np.isfinite(targets["trq_margin"].to_numpy(dtype=float)).all():
            raise ValueError("PHM torque-margin targets must be finite")

    numeric = features.to_numpy(dtype=float)
    finite_or_missing = np.isfinite(numeric) | np.isnan(numeric)
    if not finite_or_missing.all():
        raise ValueError(f"PHM {normalized_split} features contain infinite values")
    return DatasetBundle(
        name="phm2024",
        split=normalized_split,
        task="faulty classification and torque-margin regression",
        features=features,
        targets=targets,
        metadata=metadata,
    )
