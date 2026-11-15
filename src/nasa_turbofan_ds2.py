"""Read and evaluate the official N-CMAPSS DS02-006 run-to-failure split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import joblib
import numpy as np
import pandas as pd

from src.cmapss_data import constant_columns
from src.multidataset_config import shared_rul_config
from src.prognostics import PrognosticsAgent, compute_rul_metrics


ROOT = Path(__file__).resolve().parents[1]
SELECTED_FILE = "N-CMAPSS_DS02-006.h5"
FEATURE_GROUPS = ("W", "X_s", "X_v")
RICH_CYCLE_STATISTICS = ("mean", "std", "min", "max", "slope_per_second")


def _read_audit(data_dir: Path, report_path: Path | None = None) -> dict[str, Any]:
    path = report_path or ROOT / "data" / "processed" / "nasa_turbofan_ds2" / "audit.json"
    if not path.is_file():
        raise RuntimeError(
            "NASA dataset has not passed the local audit. Run "
            ".venv/Scripts/python -m scripts.audit_nasa_turbofan_ds2 first."
        )
    audit = json.loads(path.read_text(encoding="utf-8"))
    if Path(audit.get("data_dir", "")).resolve() != data_dir.resolve():
        raise RuntimeError(
            "The saved NASA audit belongs to a different data directory; "
            "rerun scripts.audit_nasa_turbofan_ds2 for this --data-root."
        )
    if not audit.get("ready_for_modeling", False):
        raise RuntimeError(
            "NASA DS02 audit did not approve this dataset for training: "
            + "; ".join(audit.get("validation_errors", []))
        )
    expected_size = audit["selected_experiment"]["file_size_bytes"]
    file_path = data_dir / "extracted" / "data_set" / SELECTED_FILE
    if not file_path.is_file() or file_path.stat().st_size != expected_size:
        raise RuntimeError(
            "Selected NASA HDF5 file is missing or changed since audit; rerun the audit."
        )
    return audit


def _variable_names(file: h5py.File, name: str) -> list[str]:
    values = np.asarray(file[name][()]).reshape(-1)
    return [
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def load_cycle_summaries(
    data_dir: str | Path,
    source_split: str,
    *,
    unit_ids: Iterable[int] | None = None,
    row_chunk: int = 200_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reduce 1 Hz data to mean feature values per documented flight cycle.

    Each unit is processed independently and its raw data is read in row
    chunks. The function never creates a combined table containing all 6.5M
    seconds of DS02 observations.
    """

    if source_split not in {"dev", "test"}:
        raise ValueError("source_split must be 'dev' or 'test'")
    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    path = Path(data_dir) / "extracted" / "data_set" / SELECTED_FILE
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[pd.DataFrame] = []
    raw_row_counts: dict[str, int] = {}
    cycle_counts: dict[str, int] = {}
    feature_names: list[str] = []
    source_to_model: dict[str, str] = {}

    with h5py.File(path, "r") as hdf:
        auxiliary = hdf[f"A_{source_split}"]
        unit_column = auxiliary[:, 0].astype(np.int64)
        unit_boundaries = np.r_[
            0, np.flatnonzero(unit_column[1:] != unit_column[:-1]) + 1, len(unit_column)
        ]
        ranges = {
            int(unit_column[start]): (int(start), int(end))
            for start, end in zip(unit_boundaries[:-1], unit_boundaries[1:])
        }
        available_units = sorted(ranges)
        selected_units = available_units if unit_ids is None else [int(x) for x in unit_ids]
        unknown = sorted(set(selected_units) - set(available_units))
        if unknown:
            raise ValueError(
                f"Unknown {source_split} unit IDs requested: {unknown}; "
                f"available official IDs are {available_units}"
            )

        variable_groups = {
            group: _variable_names(hdf, f"{group}_var")
            for group in FEATURE_GROUPS
        }
        feature_names = [name for group in FEATURE_GROUPS for name in variable_groups[group]]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("NASA source feature names are not unique")
        source_to_model = {name: name for name in feature_names}

        for unit_id in selected_units:
            start, end = ranges[unit_id]
            row_count = end - start
            raw_row_counts[str(unit_id)] = row_count
            unit_aux = auxiliary[start:end, :2]
            cycles = unit_aux[:, 1].astype(np.int64)
            if np.any(np.diff(cycles) < 0):
                raise ValueError(f"NASA unit {unit_id} flight cycles are not chronological")
            cycle_starts = np.r_[0, np.flatnonzero(cycles[1:] != cycles[:-1]) + 1]
            cycle_ends = np.r_[cycle_starts[1:], len(cycles)]
            cycle_ids = cycles[cycle_starts]
            if cycle_ids.tolist() != list(range(1, len(cycle_ids) + 1)):
                raise ValueError(f"NASA unit {unit_id} does not expose a contiguous cycle sequence")
            cycle_counts[str(unit_id)] = int(len(cycle_ids))

            # Store one unit at a time in float32, filling it in chunks from
            # the four source HDF5 feature datasets. This bounds memory use.
            features = np.empty((row_count, len(feature_names)), dtype=np.float32)
            output_start = 0
            for group in FEATURE_GROUPS:
                dataset = hdf[f"{group}_{source_split}"]
                names = variable_groups[group]
                width = len(names)
                for chunk_start in range(start, end, row_chunk):
                    chunk_end = min(end, chunk_start + row_chunk)
                    block = dataset[chunk_start:chunk_end, :]
                    local_start = chunk_start - start
                    local_end = chunk_end - start
                    features[local_start:local_end, output_start : output_start + width] = block
                output_start += width

            raw_rul = hdf[f"Y_{source_split}"][start:end, 0].astype(np.float32)
            counts = cycle_ends - cycle_starts
            means = np.add.reduceat(features, cycle_starts, axis=0) / counts[:, None]
            cycle_rul = raw_rul[cycle_ends - 1]
            if len(raw_rul) and raw_rul[-1] != 0:
                raise ValueError(f"NASA unit {unit_id} does not end at RUL zero")

            rows = pd.DataFrame(means, columns=feature_names)
            rows.insert(0, "cycle", cycle_ids)
            rows.insert(0, "unit_id", unit_id)
            rows["rul"] = cycle_rul
            rows["source_samples_in_cycle"] = counts
            records.append(rows)
            del features, means, raw_rul, unit_aux, cycles

    frame = pd.concat(records, ignore_index=True)
    info = {
        "dataset_file": SELECTED_FILE,
        "source_split": source_split,
        "official_unit_ids": [int(x) for x in selected_units],
        "raw_1hz_rows_by_unit": raw_row_counts,
        "flight_cycles_by_unit": cycle_counts,
        "raw_1hz_rows": int(sum(raw_row_counts.values())),
        "cycle_rows": int(len(frame)),
        "feature_names": feature_names,
        "source_to_model_mapping": source_to_model,
        "cycle_aggregation": "arithmetic mean of each W/X_s/X_v input over all 1 Hz samples in a unit/flight-cycle group",
        "cycle_rul_target": "Y at the last 1 Hz sample in the flight cycle; audit verifies Y is constant within cycle",
        "prediction_time_basis": "end of the current flight cycle, after its samples have been observed",
        "future_cycle_samples_used": False,
        "aggregation_group_key": ["unit_id", "cycle"],
    }
    return frame, info


def load_rich_cycle_summaries(
    data_dir: str | Path,
    source_split: str,
    *,
    unit_ids: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize one flight cycle at a time without loading whole units.

    For each measurement, calculate population mean/std, min, max, and the
    least-squares trend per source second. A single cycle's raw rows are the
    largest feature buffer held at once; no future flight cycle is read to
    construct an earlier cycle's features.
    """

    if source_split not in {"dev", "test"}:
        raise ValueError("source_split must be 'dev' or 'test'")
    path = Path(data_dir) / "extracted" / "data_set" / SELECTED_FILE
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[dict[str, Any]] = []
    raw_row_counts: dict[str, int] = {}
    cycle_counts: dict[str, int] = {}
    feature_names: list[str] = []
    with h5py.File(path, "r") as hdf:
        auxiliary = hdf[f"A_{source_split}"]
        unit_column = auxiliary[:, 0].astype(np.int64)
        unit_boundaries = np.r_[
            0, np.flatnonzero(unit_column[1:] != unit_column[:-1]) + 1, len(unit_column)
        ]
        ranges = {
            int(unit_column[start]): (int(start), int(end))
            for start, end in zip(unit_boundaries[:-1], unit_boundaries[1:])
        }
        available_units = sorted(ranges)
        selected_units = available_units if unit_ids is None else [int(x) for x in unit_ids]
        unknown = sorted(set(selected_units) - set(available_units))
        if unknown:
            raise ValueError(
                f"Unknown {source_split} unit IDs requested: {unknown}; "
                f"available official IDs are {available_units}"
            )

        variable_groups = {
            group: _variable_names(hdf, f"{group}_var") for group in FEATURE_GROUPS
        }
        source_features = [
            name for group in FEATURE_GROUPS for name in variable_groups[group]
        ]
        if len(source_features) != len(set(source_features)):
            raise ValueError("NASA source feature names are not unique")
        feature_names = [
            f"{name}__{statistic}"
            for name in source_features
            for statistic in RICH_CYCLE_STATISTICS
        ]

        for unit_id in selected_units:
            unit_start, unit_end = ranges[unit_id]
            unit_aux = auxiliary[unit_start:unit_end, :2]
            cycles = unit_aux[:, 1].astype(np.int64)
            if np.any(np.diff(cycles) < 0):
                raise ValueError(f"NASA unit {unit_id} flight cycles are not chronological")
            cycle_starts = np.r_[0, np.flatnonzero(cycles[1:] != cycles[:-1]) + 1]
            cycle_ends = np.r_[cycle_starts[1:], len(cycles)]
            cycle_ids = cycles[cycle_starts]
            if cycle_ids.tolist() != list(range(1, len(cycle_ids) + 1)):
                raise ValueError(f"NASA unit {unit_id} does not expose a contiguous cycle sequence")
            raw_row_counts[str(unit_id)] = int(unit_end - unit_start)
            cycle_counts[str(unit_id)] = int(len(cycle_ids))

            for cycle_id, local_start, local_end in zip(
                cycle_ids, cycle_starts, cycle_ends
            ):
                source_start = unit_start + int(local_start)
                source_end = unit_start + int(local_end)
                groups = [
                    hdf[f"{group}_{source_split}"][source_start:source_end, :].astype(
                        np.float32
                    )
                    for group in FEATURE_GROUPS
                ]
                values = np.concatenate(groups, axis=1)
                sample_count = len(values)
                mean = (
                    np.add.reduce(values, axis=0, dtype=np.float32)
                    / np.float32(sample_count)
                )
                std = values.std(axis=0, dtype=np.float64).astype(np.float32)
                minimum = values.min(axis=0)
                maximum = values.max(axis=0)
                time = np.arange(sample_count, dtype=np.float64)
                centered_time = time - time.mean()
                time_ss = float(centered_time @ centered_time)
                centered_values = values.astype(np.float64) - values.mean(
                    axis=0, dtype=np.float64
                )
                if sample_count < 2 or time_ss <= 0:
                    slope = np.zeros(values.shape[1], dtype=np.float32)
                else:
                    slope = (
                        (centered_time @ centered_values) / time_ss
                    ).astype(np.float32)
                summary_by_stat = {
                    "mean": mean,
                    "std": std,
                    "min": minimum,
                    "max": maximum,
                    "slope_per_second": slope,
                }
                row_features = np.stack(
                    [
                        summary_by_stat[statistic][feature_index]
                        for feature_index in range(len(source_features))
                        for statistic in RICH_CYCLE_STATISTICS
                    ]
                )
                rul_values = hdf[f"Y_{source_split}"][source_start:source_end, 0]
                if len(rul_values) and np.any(rul_values != rul_values[-1]):
                    raise ValueError(
                        f"NASA unit {unit_id}, cycle {cycle_id} has non-constant RUL"
                    )
                record = {
                    "unit_id": int(unit_id),
                    "cycle": int(cycle_id),
                    "rul": float(rul_values[-1]),
                    "source_samples_in_cycle": int(sample_count),
                }
                record.update(zip(feature_names, row_features.tolist()))
                records.append(record)
                del values, groups, centered_values, summary_by_stat, row_features
                del rul_values, centered_time, slope

    frame = pd.DataFrame.from_records(records)
    info = {
        "dataset_file": SELECTED_FILE,
        "source_split": source_split,
        "official_unit_ids": [int(x) for x in selected_units],
        "raw_1hz_rows_by_unit": raw_row_counts,
        "flight_cycles_by_unit": cycle_counts,
        "raw_1hz_rows": int(sum(raw_row_counts.values())),
        "cycle_rows": int(len(frame)),
        "source_feature_names": source_features,
        "feature_names": feature_names,
        "summary_statistics": list(RICH_CYCLE_STATISTICS),
        "standard_deviation": "population standard deviation within the flight cycle",
        "trend": "least-squares slope per 1 Hz sample-second within the flight cycle",
        "cycle_rul_target": "Y at the last 1 Hz sample in the flight cycle; verified constant within cycle",
        "prediction_time_basis": "end of the current flight cycle, after its samples have been observed",
        "future_cycle_samples_used": False,
        "aggregation_group_key": ["unit_id", "cycle"],
        "max_raw_feature_buffer": "one unit/flight-cycle slice across W, X_s, and X_v at a time",
    }
    return frame, info


def _rul_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    result = compute_rul_metrics(
        frame["rul"].to_numpy(dtype=np.float64),
        frame["predicted_rul"].to_numpy(dtype=np.float64),
    )
    return {
        name: float(value) if np.isfinite(value) else None
        for name, value in result.items()
    }


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_nasa_turbofan_ds2(
    *,
    data_dir: Path,
    report_root: Path,
    model_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Fit the shared C-MAPSS RUL model separately on NASA DS02-006."""

    audit = _read_audit(Path(data_dir))
    dev_info = audit["selected_experiment"]["dev"]
    test_info = audit["selected_experiment"]["test"]
    official_dev_ids = [int(x) for x in dev_info["unit_ids"]]
    official_test_ids = [int(x) for x in test_info["unit_ids"]]

    development, development_info = load_cycle_summaries(
        data_dir, "dev", unit_ids=official_dev_ids
    )
    # Same deterministic engine/unit split as C-MAPSS; there are only six
    # official development units, so this yields five fit units and one local
    # validation unit. The official three test units remain untouched until
    # their one final scoring pass.
    from src.cmapss_data import split_engine_units

    unit_split = split_engine_units(development, validation_fraction=0.2, seed=42)
    training_frame = development[
        development["unit_id"].isin(unit_split.train_units)
    ].drop(columns=["source_samples_in_cycle"]).copy()
    validation_frame = development[
        development["unit_id"].isin(unit_split.validation_units)
    ].drop(columns=["source_samples_in_cycle"]).copy()

    test_info_from_loader: dict[str, Any] | None = None
    test_model_frame: pd.DataFrame | None = None
    if not smoke:
        test_frame, test_info_from_loader = load_cycle_summaries(
            data_dir, "test", unit_ids=official_test_ids
        )
        test_model_frame = test_frame.drop(columns=["source_samples_in_cycle"]).copy()
    config = shared_rul_config(smoke=smoke)
    dropped = constant_columns(training_frame, development_info["feature_names"])
    features = [name for name in development_info["feature_names"] if name not in dropped]
    agent = PrognosticsAgent.fit(training_frame, features, config)
    validation_predictions = agent.predict_frame(validation_frame)
    validation_terminal = agent.predict_current(validation_frame)
    report_dir = report_root / "nasa_turbofan_ds2"
    model_dir = model_root / "nasa_turbofan_ds2"
    if smoke:
        report_dir = report_dir / "smoke"
        model_dir = model_dir / "smoke"

    report: dict[str, Any] = {
        "dataset": "NASA Turbofan Engine Degradation Simulation-2 / N-CMAPSS DS02-006",
        "task": "RUL sequence regression in flight cycles",
        "model": "existing PrognosticsAgent random_forest window-summary architecture; independently retrained for DS02-006",
        "smoke_run": bool(smoke),
        "config": config.as_dict(),
        "selected_source_file": SELECTED_FILE,
        "feature_groups": {
            "flight_conditions": [name for name in development_info["feature_names"] if name in audit["selected_experiment"]["features"]["flight_conditions_W"]],
            "physical_sensor_readings": audit["selected_experiment"]["features"]["physical_sensor_readings_X_s"],
            "virtual_sensor_readings": audit["selected_experiment"]["features"]["virtual_sensor_readings_X_v"],
            "units": audit["selected_experiment"]["features"]["feature_units"],
        },
        "features_used": features,
        "features_dropped_as_constant_on_fit_units": dropped,
        "feature_preprocessing": "constant-column removal is determined on fit units; PrognosticsAgent StandardScaler is fitted on fit-unit cycle rows only",
        "cycle_adapter": {
            "development": development_info,
            "official_test": test_info_from_loader,
            "time_resolution": "source observations are 1 Hz; the model sequence contains one row per flight cycle",
            "metadata_excluded_from_features": ["unit_id", "cycle", "Fc", "hs", "T_*", "Y"],
            "aggregation_limit": "per-flight means preserve operating levels but do not model within-flight temporal variation",
        },
        "splits": {
            "official_development_unit_ids": official_dev_ids,
            "fit_unit_ids": list(unit_split.train_units),
            "validation_unit_ids": list(unit_split.validation_units),
            "official_test_unit_ids": official_test_ids,
            "fit_units": int(len(unit_split.train_units)),
            "validation_units": int(len(unit_split.validation_units)),
            "official_test_units": int(len(official_test_ids)),
            "fit_validation_unit_disjoint": not bool(
                set(unit_split.train_units) & set(unit_split.validation_units)
            ),
            "official_test_is_unit_disjoint": not bool(
                set(official_dev_ids) & set(official_test_ids)
            ),
            "fit_cycle_rows": int(len(training_frame)),
            "validation_cycle_rows": int(len(validation_frame)),
            "official_test_cycle_rows": (
                None if test_model_frame is None else int(len(test_model_frame))
            ),
            "scaler_fit_rows": int(len(training_frame)),
        },
        "validation_cycle_metrics": _rul_metrics(validation_predictions),
        "validation_terminal_unit_metrics": _rul_metrics(validation_terminal),
        "limitations": [
            "DS02-006 was synthetically generated with the C-MAPSS engine model using real-flight profiles; it tests added flight-condition complexity, not independent real-fleet failures.",
            "Only five of the six development units fit the final model and one unit is local validation; validation uncertainty is high with one unit.",
            "The three official test units are scored once and are too few for a stable fleet-level uncertainty interval.",
            "Cycle-level averaging reduces millions of 1 Hz samples to flight-cycle rows and loses within-flight temporal detail.",
            "The archive contains ten sibling HDF5 variants; this run uses only the documented DS02-006 file and does not pool variants.",
            "The sibling N-CMAPSS_DS08d-010.h5 file failed HDF5 open validation in the archive audit; it is outside this DS02-006 experiment.",
        ],
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    if not smoke:
        assert test_model_frame is not None
        test_predictions = agent.predict_frame(test_model_frame)
        test_terminal = agent.predict_current(test_model_frame)
        report["official_test_cycle_metrics"] = _rul_metrics(test_predictions)
        report["official_test_terminal_unit_metrics"] = _rul_metrics(test_terminal)
        report["official_test_scored"] = True
        report["test_protocol"] = (
            "Use the official DS02 test units 11, 14, and 15; report all eligible flight-cycle windows and the final flight cycle per unit."
        )
        test_predictions.to_csv(
            report_dir / "official_test_cycle_predictions.csv", index=False
        )
        test_terminal.to_csv(
            report_dir / "official_test_terminal_predictions.csv", index=False
        )
    else:
        report["official_test_scored"] = False
        report["test_protocol"] = "Not scored during smoke run; only official development units are loaded."

    _write_json(report_dir / "results.json", report)
    validation_predictions.to_csv(
        report_dir / "validation_cycle_predictions.csv", index=False
    )
    validation_terminal.to_csv(
        report_dir / "validation_terminal_predictions.csv", index=False
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(agent, model_dir / "prognostics_agent.joblib", compress=3)
    return report
