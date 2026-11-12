"""Audit the extracted NASA C-MAPSS files before modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{index}" for index in range(1, 22)],
]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def read_telemetry(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=BASE_COLUMNS,
        engine="python",
    )
    if frame.shape[1] != len(BASE_COLUMNS):
        raise ValueError(
            f"{path.name} has {frame.shape[1]} columns; "
            f"expected {len(BASE_COLUMNS)}"
        )
    return frame


def trajectory_checks(frame: pd.DataFrame) -> dict[str, int]:
    duplicate_pairs = int(frame.duplicated(["unit_id", "cycle"]).sum())
    non_monotonic_units = 0
    non_contiguous_units = 0

    for _, unit_frame in frame.groupby("unit_id", sort=True):
        cycles = unit_frame["cycle"].to_numpy()
        if not np.all(np.diff(cycles) > 0):
            non_monotonic_units += 1
        expected = np.arange(cycles.min(), cycles.max() + 1)
        if not np.array_equal(cycles, expected):
            non_contiguous_units += 1

    return {
        "units": int(frame["unit_id"].nunique()),
        "rows": int(len(frame)),
        "duplicate_unit_cycle_pairs": duplicate_pairs,
        "non_monotonic_units": non_monotonic_units,
        "non_contiguous_units": non_contiguous_units,
        "min_cycle": int(frame["cycle"].min()),
        "max_cycle": int(frame["cycle"].max()),
    }


def finite_value_checks(frame: pd.DataFrame) -> dict[str, int]:
    values = frame.to_numpy(dtype=float)
    return {
        "missing_values": int(frame.isna().sum().sum()),
        "non_finite_values": int((~np.isfinite(values)).sum()),
    }


def audit_subset(data_dir: Path, subset: str) -> None:
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"

    missing = [path.name for path in (train_path, test_path, rul_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for {subset}: {', '.join(missing)}")

    train = read_telemetry(train_path)
    test = read_telemetry(test_path)
    rul = pd.read_csv(rul_path, header=None, names=["rul"])

    train_checks = {**trajectory_checks(train), **finite_value_checks(train)}
    test_checks = {**trajectory_checks(test), **finite_value_checks(test)}

    sensor_columns = [column for column in BASE_COLUMNS if column.startswith("sensor_")]
    constant_sensors = [
        column for column in sensor_columns if train[column].nunique(dropna=False) <= 1
    ]

    print(f"\n{subset}")
    print("=" * len(subset))
    print(
        f"train: {train_checks['rows']:,} rows, {train_checks['units']} units, "
        f"cycles {train_checks['min_cycle']}..{train_checks['max_cycle']}"
    )
    print(
        f"test : {test_checks['rows']:,} rows, {test_checks['units']} units, "
        f"cycles {test_checks['min_cycle']}..{test_checks['max_cycle']}"
    )
    print(f"test RUL labels: {len(rul):,} (test units: {test_checks['units']})")
    print(
        "quality: "
        f"missing={train_checks['missing_values'] + test_checks['missing_values']}, "
        f"non_finite={train_checks['non_finite_values'] + test_checks['non_finite_values']}, "
        f"duplicate_pairs={train_checks['duplicate_unit_cycle_pairs'] + test_checks['duplicate_unit_cycle_pairs']}"
    )
    print(
        "trajectory checks: "
        f"non_monotonic_units={train_checks['non_monotonic_units'] + test_checks['non_monotonic_units']}, "
        f"non_contiguous_units={train_checks['non_contiguous_units'] + test_checks['non_contiguous_units']}"
    )
    print(
        "constant training sensors: "
        + (", ".join(constant_sensors) if constant_sensors else "none")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/cmapss"),
        help="Directory containing train_FDxxx.txt, test_FDxxx.txt, and RUL_FDxxx.txt",
    )
    parser.add_argument(
        "--subset",
        choices=[*SUBSETS, "ALL"],
        default="ALL",
        help="Subset to audit (default: ALL)",
    )
    args = parser.parse_args()

    subsets = SUBSETS if args.subset == "ALL" else (args.subset,)
    for subset in subsets:
        audit_subset(args.data_dir, subset)


if __name__ == "__main__":
    main()

