"""Data loading and label/window construction for NASA C-MAPSS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{index}" for index in range(1, 22)],
]

OPERATING_COLUMNS = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
FEATURE_COLUMNS = [*OPERATING_COLUMNS, *SENSOR_COLUMNS]


@dataclass(frozen=True)
class EngineUnitSplit:
    """Deterministic train/validation split at the engine level."""

    train_units: tuple[int, ...]
    validation_units: tuple[int, ...]


def _telemetry_path(data_dir: Path, subset: str, split: str) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    return data_dir / f"{split}_{subset}.txt"


def load_telemetry(data_dir: Path | str, subset: str, split: str) -> pd.DataFrame:
    """Load a C-MAPSS telemetry file with explicit column names."""

    path = _telemetry_path(Path(data_dir), subset, split)
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=BASE_COLUMNS,
        engine="python",
    )
    if frame.shape[1] != len(BASE_COLUMNS):
        raise ValueError(f"Expected {len(BASE_COLUMNS)} columns in {path}")
    return frame


def add_train_rul(frame: pd.DataFrame) -> pd.DataFrame:
    """Add exact row-level RUL labels to run-to-failure training trajectories."""

    result = frame.copy()
    final_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul"] = final_cycle - result["cycle"]
    return result


def load_test_rul(data_dir: Path | str, subset: str) -> pd.Series:
    """Load the official RUL-at-last-observed-cycle values for test units."""

    path = Path(data_dir) / f"RUL_{subset}.txt"
    values = pd.read_csv(path, header=None, names=["rul"])["rul"]
    values.index = np.arange(1, len(values) + 1)
    values.index.name = "unit_id"
    return values


def add_test_rul(
    frame: pd.DataFrame,
    rul_at_end: pd.Series,
) -> pd.DataFrame:
    """Add row-level test RUL using each unit's official terminal RUL."""

    result = frame.copy()
    final_cycle = result.groupby("unit_id")["cycle"].transform("max")
    terminal_rul = result["unit_id"].map(rul_at_end)
    if terminal_rul.isna().any():
        missing_units = sorted(result.loc[terminal_rul.isna(), "unit_id"].unique())
        raise ValueError(f"Missing test RUL values for units: {missing_units}")
    result["rul"] = terminal_rul + final_cycle - result["cycle"]
    return result


def verify_no_leakage(train_units: tuple[int, ...] | list[int], validation_units: tuple[int, ...] | list[int]) -> bool:
    """Validate that train and validation engine units are completely disjoint."""
    overlap = set(train_units) & set(validation_units)
    if overlap:
        raise AssertionError(f"Data leakage detected! Overlapping engine units: {sorted(overlap)}")
    return True


def split_engine_units(
    frame: pd.DataFrame,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> EngineUnitSplit:
    """Split whole engine trajectories reproducibly, never individual rows."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    units = np.array(sorted(frame["unit_id"].unique()), dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    validation_count = max(1, int(round(len(units) * validation_fraction)))
    validation_units = tuple(sorted(shuffled[:validation_count].tolist()))
    train_units = tuple(sorted(shuffled[validation_count:].tolist()))
    verify_no_leakage(train_units, validation_units)
    return EngineUnitSplit(train_units=train_units, validation_units=validation_units)


def constant_columns(frame: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    """Return columns that are constant in the supplied training frame."""

    columns = columns or FEATURE_COLUMNS
    return [column for column in columns if frame[column].nunique(dropna=False) <= 1]


def fit_feature_scaler(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> StandardScaler:
    """Fit a standardizer on training telemetry only."""

    scaler = StandardScaler()
    scaler.fit(frame[feature_columns].to_numpy(dtype=np.float64))
    return scaler


def transform_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Transform selected features while preserving identifiers and labels."""

    result = frame.copy()
    result[feature_columns] = result[feature_columns].astype("float64")
    result.loc[:, feature_columns] = scaler.transform(
        result[feature_columns].to_numpy(dtype=np.float64)
    )
    return result


def make_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    target_column: str = "rul",
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create chronological sliding windows independently for each engine."""

    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    missing = [column for column in [*feature_columns, target_column] if column not in frame]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, int | float]] = []

    for unit_id, unit_frame in frame.groupby("unit_id", sort=True):
        unit_frame = unit_frame.sort_values("cycle")
        values = unit_frame[feature_columns].to_numpy(dtype=np.float32)
        target_values = unit_frame[target_column].to_numpy(dtype=np.float32)
        cycles = unit_frame["cycle"].to_numpy(dtype=int)

        end_indices = list(range(window_size - 1, len(unit_frame), stride))
        # Keep the last observed cycle for every engine even when it is not
        # aligned with the requested stride. This is required for terminal
        # engine evaluation and official C-MAPSS terminal RUL scoring.
        final_end = len(unit_frame) - 1
        if end_indices and end_indices[-1] != final_end:
            end_indices.append(final_end)
        for end in end_indices:
            start = end - window_size + 1
            windows.append(values[start : end + 1])
            targets.append(float(target_values[end]))
            metadata.append(
                {
                    "unit_id": int(unit_id),
                    "end_cycle": int(cycles[end]),
                    "rul": float(target_values[end]),
                }
            )

    if not windows:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DataFrame(columns=["unit_id", "end_cycle", "rul"]),
        )

    return (
        np.stack(windows).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        pd.DataFrame(metadata),
    )


def summarize_windows(windows: np.ndarray) -> np.ndarray:
    """Create interpretable summary features for a classical baseline."""

    if windows.ndim != 3:
        raise ValueError("windows must have shape (samples, time, features)")
    if windows.shape[1] < 2:
        raise ValueError("windows must contain at least two time steps")

    means = windows.mean(axis=1)
    standard_deviations = windows.std(axis=1)
    minima = windows.min(axis=1)
    maxima = windows.max(axis=1)
    slopes = (windows[:, -1, :] - windows[:, 0, :]) / (windows.shape[1] - 1)
    return np.concatenate(
        [windows[:, -1, :], means, standard_deviations, minima, maxima, slopes],
        axis=1,
    ).astype(np.float32)

