"""Tests for clearly labeled synthetic sensor-fault scenarios."""

import unittest

import numpy as np

from src.fault_injection import (
    SYNTHETIC_FAULT_LABELS,
    inject_sensor_fault,
)


class SyntheticFaultInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = np.arange(60, dtype=np.float32).reshape(30, 2) / 30.0
        self.features = ["sensor_1", "sensor_2"]

    def test_every_fault_returns_copy_with_synthetic_provenance(self) -> None:
        original = self.window.copy()
        for label in SYNTHETIC_FAULT_LABELS:
            corrupted, provenance = inject_sensor_fault(
                self.window,
                self.features,
                label,
                rng=np.random.default_rng(7),
            )
            self.assertEqual(corrupted.shape, self.window.shape)
            self.assertTrue(np.isfinite(corrupted).all())
            self.assertTrue(provenance["synthetic_label"])
            self.assertEqual(provenance["fault_type"], label)
        np.testing.assert_array_equal(self.window, original)

    def test_fault_generation_is_reproducible_for_fixed_seed(self) -> None:
        first, first_info = inject_sensor_fault(
            self.window,
            self.features,
            "drift",
            rng=np.random.default_rng(11),
            severity=2.0,
        )
        second, second_info = inject_sensor_fault(
            self.window,
            self.features,
            "drift",
            rng=np.random.default_rng(11),
            severity=2.0,
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_info, second_info)

    def test_rejects_non_sensor_feature_set_and_invalid_type(self) -> None:
        with self.assertRaises(ValueError):
            inject_sensor_fault(
                self.window,
                ["setting_1", "setting_2"],
                "bias",
                rng=np.random.default_rng(1),
            )
        with self.assertRaises(ValueError):
            inject_sensor_fault(
                self.window,
                self.features,
                "physical_fault",
                rng=np.random.default_rng(1),
            )
