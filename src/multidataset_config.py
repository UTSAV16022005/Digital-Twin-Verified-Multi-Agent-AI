"""Shared settings for the cross-dataset RUL comparison."""

from __future__ import annotations

from dataclasses import replace

from src.prognostics import PrognosticsConfig


# This is the repository's existing random-forest prognostics configuration.
# Simulation-2's 1 Hz observations are first reduced to flight-cycle rows so
# its millions of raw samples do not create a huge window tensor.
SHARED_RUL_CONFIG = PrognosticsConfig()


def shared_rul_config(*, smoke: bool = False) -> PrognosticsConfig:
    """Return a fresh common config; smoke runs use fewer trees only."""

    return replace(SHARED_RUL_CONFIG, n_estimators=8) if smoke else replace(SHARED_RUL_CONFIG)
