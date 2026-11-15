"""Focused routing and prediction-input contract checks for the terminal demo."""

import unittest

import pandas as pd

from src.data_adapters import DatasetBundle
from src.multidataset_agent_demo import (
    cmapss_prediction_only_frame,
    nasa_prediction_only_frame,
    phm_prediction_only_bundle,
    role_route,
)


class MultiDatasetAgentRoutingTests(unittest.TestCase):
    def test_routes_only_native_roles(self) -> None:
        self.assertEqual(
            role_route("cmapss", "recovery_planner")["status"], "supported"
        )
        self.assertEqual(
            role_route("nasa_turbofan_ds2", "recovery_planner")["status"],
            "not_supported",
        )
        self.assertEqual(
            role_route("phm2024", "faulty_classifier")["status"], "supported"
        )

    def test_cmapss_prediction_frame_does_not_contain_rul(self) -> None:
        frame = pd.DataFrame(
            {"unit_id": [2], "cycle": [30], "sensor_2": [0.5], "rul": [15.0]}
        )
        inference = cmapss_prediction_only_frame(frame)
        self.assertNotIn("rul", inference.columns)
        self.assertEqual(inference.loc[0, "unit_id"], 2)

    def test_nasa_inference_keeps_cycle_metadata_but_not_target_fields(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [2],
                "cycle": [30],
                "rul": [7.0],
                "source_samples_in_cycle": [3600],
                "W1__mean": [1.2],
            }
        )
        inference = nasa_prediction_only_frame(frame, ["W1__mean"])
        self.assertEqual(inference.columns.tolist(), ["unit_id", "cycle", "W1__mean"])
        with self.assertRaisesRegex(ValueError, "protected fields"):
            nasa_prediction_only_frame(frame, ["rul"])

    def test_phm_id_is_metadata_and_targets_are_absent_from_model_bundle(self) -> None:
        source = DatasetBundle(
            dataset_name="phm2024",
            split="train",
            task_type="binary_classification",
            X=pd.DataFrame({"trq_measured": [4.2], "oat": [10.0]}),
            y=pd.Series([1], name="faulty"),
            metadata=pd.DataFrame({"id": [123]}),
            feature_names=("trq_measured", "oat"),
            source_to_model_mapping={"trq_measured": "trq_measured", "oat": "oat"},
        )
        inference = phm_prediction_only_bundle(source, 0)
        self.assertIsNone(inference.y)
        self.assertEqual(inference.metadata["id"].tolist(), [123])
        self.assertNotIn("id", inference.X.columns)
        self.assertNotIn("faulty", inference.X.columns)
        self.assertNotIn("trq_margin", inference.X.columns)

    def test_unsupported_role_returns_reason_not_another_dataset_model(self) -> None:
        result = role_route("phm2024", "digital_twin")
        self.assertEqual(result["status"], "not_supported")
        self.assertIn("engine trajectories", result["reason"])


if __name__ == "__main__":
    unittest.main()
