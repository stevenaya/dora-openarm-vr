"""Tests for Quest controller value mappings."""

from __future__ import annotations

import unittest

import numpy as np

from dora_openarm_vr.quest_receiver import _map_trigger_to_gripper


class TriggerMappingTest(unittest.TestCase):
    def test_calibrated_gripper_endpoints_are_preserved(self) -> None:
        self.assertAlmostEqual(_map_trigger_to_gripper(0.0, "right"), np.deg2rad(-45.0))
        self.assertAlmostEqual(_map_trigger_to_gripper(1.0, "right"), np.deg2rad(8.0))
        self.assertAlmostEqual(_map_trigger_to_gripper(0.0, "left"), np.deg2rad(45.0))
        self.assertAlmostEqual(_map_trigger_to_gripper(1.0, "left"), np.deg2rad(-8.0))

    def test_trigger_value_is_clipped_before_mapping(self) -> None:
        self.assertAlmostEqual(
            _map_trigger_to_gripper(-1.0, "right"),
            _map_trigger_to_gripper(0.0, "right"),
        )
        self.assertAlmostEqual(
            _map_trigger_to_gripper(2.0, "left"),
            _map_trigger_to_gripper(1.0, "left"),
        )


if __name__ == "__main__":
    unittest.main()
