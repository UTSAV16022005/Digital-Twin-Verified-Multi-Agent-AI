"""Profile FD001 and demonstrate leakage-safe window construction."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cmapss_data import (
    FEATURE_COLUMNS,
    add_train_rul,
    constant_columns,
    load_telemetry,
    make_windows,
    split_engine_units,
)


def main() -> None:
    data_dir = Path("data/raw/cmapss")
    train = add_train_rul(load_telemetry(data_dir, "FD001", "train"))
    split = split_engine_units(train, validation_fraction=0.2, seed=42)

    train_frame = train[train["unit_id"].isin(split.train_units)]
    validation_frame = train[train["unit_id"].isin(split.validation_units)]
    dropped = constant_columns(train_frame, FEATURE_COLUMNS)
    features = [column for column in FEATURE_COLUMNS if column not in dropped]

    train_windows, train_targets, train_meta = make_windows(
        train_frame, features, window_size=30
    )
    validation_windows, validation_targets, validation_meta = make_windows(
        validation_frame, features, window_size=30
    )

    print("FD001 leakage-safe preparation")
    print("==============================")
    print(f"training engines:   {len(split.train_units)}")
    print(f"validation engines: {len(split.validation_units)}")
    print(f"raw features:       {len(FEATURE_COLUMNS)}")
    print(f"constant features:  {', '.join(dropped) if dropped else 'none'}")
    print(f"used features:      {len(features)}")
    print(f"window size:        30 cycles")
    print(f"train windows:      {train_windows.shape}")
    print(f"validation windows: {validation_windows.shape}")
    print(
        "target RUL range:   "
        f"train {train_targets.min():.0f}..{train_targets.max():.0f}, "
        f"validation {validation_targets.min():.0f}..{validation_targets.max():.0f}"
    )
    print(
        "metadata example:   "
        f"unit={int(train_meta.iloc[0]['unit_id'])}, "
        f"end_cycle={int(train_meta.iloc[0]['end_cycle'])}, "
        f"rul={train_meta.iloc[0]['rul']:.0f}"
    )


if __name__ == "__main__":
    main()

