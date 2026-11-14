"""Deterministic sensor-level fault perturbations for synthetic experiments.

These modes are artificial test labels. They do not represent NASA-provided or
physically validated engine fault classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np


SYNTHETIC_SENSOR_FAULTS = ("bias", "drift", "stuck", "spike")
SYNTHETIC_FAULT_LABELS = ("normal", *SYNTHETIC_SENSOR_FAULTS)


def inject_sensor_fault(
    window: np.ndarray,
    feature_names: list[str],
    fault_type: str,
    *,
    rng: np.random.Generator | None = None,
    severity: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a copied window with one synthetic sensor corruption applied."""

    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("window must have shape (time, len(feature_names))")
    if values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("window must contain at least two finite time steps")
    if not np.isfinite(severity) or severity <= 0.0:
        raise ValueError("severity must be a finite positive number")
    if fault_type not in SYNTHETIC_FAULT_LABELS:
        raise ValueError(f"unsupported synthetic fault type: {fault_type!r}")

    corrupted = values.copy()
    if fault_type == "normal":
        return corrupted, {
            "fault_type": "normal",
            "fault_sensor": None,
            "severity": 0.0,
            "start_index": None,
            "synthetic_label": True,
        }

    sensor_indices = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("sensor_")
    ]
    if not sensor_indices:
        raise ValueError("feature_names must include at least one sensor_* feature")
    random = rng or np.random.default_rng()
    sensor_index = int(random.choice(sensor_indices))
    sign = float(random.choice(np.array([-1.0, 1.0])))
    start_index: int | None = None

    if fault_type == "bias":
        corrupted[:, sensor_index] += np.float32(sign * severity)
        start_index = 0
    elif fault_type == "drift":
        ramp = np.linspace(0.0, sign * severity, values.shape[0], dtype=np.float32)
        corrupted[:, sensor_index] += ramp
        start_index = 0
    elif fault_type == "stuck":
        lower = max(1, values.shape[0] // 3)
        start_index = int(random.integers(lower, values.shape[0]))
        corrupted[start_index:, sensor_index] = corrupted[start_index, sensor_index]
    elif fault_type == "spike":
        start_index = int(random.integers(0, values.shape[0]))
        corrupted[start_index, sensor_index] += np.float32(sign * severity)

    return corrupted, {
        "fault_type": fault_type,
        "fault_sensor": feature_names[sensor_index],
        "severity": float(severity),
        "start_index": start_index,
        "synthetic_label": True,
    }
