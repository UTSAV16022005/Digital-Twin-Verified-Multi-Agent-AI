"""Audit loaded UCI SECOM and PHM 2024 files without fitting models."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.dataset_adapters import DatasetBundle, load_phm2024, load_secom


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"


def _class_counts(bundle: DatasetBundle) -> dict[str, int] | None:
    if bundle.targets is None or "faulty" not in bundle.targets:
        return None
    counts = bundle.targets["faulty"].value_counts().sort_index()
    return {str(int(key)): int(value) for key, value in counts.items()}


def _preview(bundle: DatasetBundle) -> dict[str, Any]:
    joined = bundle.metadata.copy()
    if bundle.targets is not None:
        joined = joined.join(bundle.targets.reset_index(drop=True))
    display_columns = list(bundle.features.columns[:8])
    joined = pd.concat([joined, bundle.features.loc[:, display_columns]], axis=1)
    return {
        "first_5": joined.head(5).to_dict(orient="records"),
        "last_5": joined.tail(5).to_dict(orient="records"),
        "displayed_feature_columns": display_columns,
        "total_feature_columns": len(bundle.features.columns),
    }


def audit_bundle(bundle: DatasetBundle) -> None:
    feature_values = bundle.features.to_numpy(dtype=float)
    finite_mask = np.isfinite(feature_values)
    missing_mask = np.isnan(feature_values)
    constant_features = int(
        bundle.features.nunique(dropna=True).le(1).sum()
    )
    print(f"\n{bundle.name.upper()} | split={bundle.split}")
    print("=" * (len(bundle.name) + len(bundle.split) + 9))
    print(f"task: {bundle.task}")
    print(
        f"shape: {len(bundle.features):,} rows x "
        f"{len(bundle.features.columns):,} features"
    )
    print(f"missing feature cells: {int(missing_mask.sum()):,}")
    print(f"infinite feature cells: {int((~finite_mask & ~missing_mask).sum()):,}")
    print(f"constant features (at most one observed value): {constant_features:,}")
    print(f"duplicate feature rows: {int(bundle.features.duplicated().sum()):,}")
    print(f"metadata columns: {', '.join(bundle.metadata.columns)}")
    print(f"feature name sample: {', '.join(bundle.features.columns[:8])}")
    if bundle.targets is None:
        print("targets: not supplied for this split (correctly left unavailable)")
    else:
        print(f"target columns: {', '.join(bundle.targets.columns)}")
        if "faulty" in bundle.targets:
            print(f"faulty label counts (0=normal, 1=faulty): {_class_counts(bundle)}")
        if "trq_margin" in bundle.targets:
            margin = bundle.targets["trq_margin"]
            print(
                "torque margin: "
                f"min={margin.min():.3f}, median={margin.median():.3f}, "
                f"max={margin.max():.3f}"
            )
    preview = _preview(bundle)
    print(
        "\nFirst 5 rows (metadata/targets + first "
        f"{len(preview['displayed_feature_columns'])} features)"
    )
    print(pd.DataFrame(preview["first_5"]).to_string(index=False))
    print(f"\nLast 5 rows (same columns; {preview['total_feature_columns']} total features)")
    print(pd.DataFrame(preview["last_5"]).to_string(index=False))


def audit_phm_archives(phm_root: Path) -> None:
    print("\nPHM 2024 downloaded archive check")
    expected = {
        "Data_Challenge_PHM2024_training_data.zip": {
            "X_train.csv": phm_root / "extracted" / "training" / "X_train.csv",
            "y_train.csv": phm_root / "extracted" / "training" / "y_train.csv",
        },
        "Data_Challenge_PHM2024_test_data.zip": {
            "X_test.csv": phm_root / "extracted" / "test" / "X_test.csv",
        },
        "Data_Challenge_PHM2024_validation_data.zip": {
            "X_validation.csv": (
                phm_root / "extracted" / "validation" / "X_validation.csv"
            ),
        },
    }
    for archive_name, extracted_files in expected.items():
        path = phm_root / archive_name
        if not path.is_file():
            print(f"ERROR: missing archive {path}")
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                members = {
                    Path(name).name: name
                    for name in archive.namelist()
                    if not name.endswith("/")
                }
                names = set(members)
                if not names:
                    print(f"WARNING: {path.name} is an empty ZIP archive (22-byte placeholder).")
                    continue
                missing = sorted(set(extracted_files) - names)
                if missing:
                    print(f"ERROR: {path.name} is missing expected members: {missing}")
                    continue
                bad = archive.testzip()
                if bad is not None:
                    print(f"ERROR: {path.name} has a corrupt member: {bad}")
                    continue
                mismatches = []
                for member_name, extracted_path in extracted_files.items():
                    if not extracted_path.is_file():
                        mismatches.append(f"missing extracted {extracted_path}")
                        continue
                    archived_digest = hashlib.sha256(
                        archive.read(members[member_name])
                    ).hexdigest()
                    extracted_digest = hashlib.sha256(
                        extracted_path.read_bytes()
                    ).hexdigest()
                    if archived_digest != extracted_digest:
                        mismatches.append(
                            f"{member_name} differs from extracted CSV"
                        )
                if mismatches:
                    print(f"ERROR: {path.name}: {'; '.join(mismatches)}")
                else:
                    print(
                        f"{path.name}: {len(names)} member(s), archive OK, "
                        "extracted CSV hashes match"
                    )
        except zipfile.BadZipFile:
            print(f"ERROR: {path.name} is not a valid ZIP archive.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--dataset", choices=("all", "secom", "phm2024"), default="all")
    args = parser.parse_args()
    raw_root = args.raw_root

    if args.dataset in {"all", "secom"}:
        audit_bundle(load_secom(raw_root / "secom"))
    if args.dataset in {"all", "phm2024"}:
        phm_root = raw_root / "phm2024"
        audit_phm_archives(phm_root)
        for split in ("train", "test", "validation"):
            audit_bundle(load_phm2024(phm_root, split))


if __name__ == "__main__":
    main()
