"""Regression tests for quaternion-aware VR pose smoothing."""

from __future__ import annotations

import unittest

import numpy as np

from dora_openarm_vr.smoothing import OneEuroPoseSmoother


def _roll_pose(angle: float) -> np.ndarray:
    return np.array(
        [
            0.0,
            0.0,
            0.0,
            np.cos(0.5 * angle),
            np.sin(0.5 * angle),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def _position_pose(x: float) -> np.ndarray:
    return np.array(
        [x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )


def _roll_delta(first: np.ndarray, second: np.ndarray) -> float:
    w1, x1 = first[3], first[4]
    w2, x2 = second[3], second[4]
    relative_w = w1 * w2 + x1 * x2
    relative_x = w1 * x2 - x1 * w2
    return float(2.0 * np.arctan2(relative_x, relative_w))


class OneEuroPoseSmootherTest(unittest.TestCase):
    def test_position_speed_is_measured_raw_to_raw(self) -> None:
        smoother = OneEuroPoseSmoother(
            min_cutoff=0.1,
            beta=0.0,
            d_cutoff=1e6,
            min_cutoff_rot=0.1,
            beta_rot=0.0,
            d_cutoff_rot=1.0,
        )

        smoother.smooth(0.0, _position_pose(0.0))
        smoother.smooth(0.1, _position_pose(0.1))
        smoother.smooth(0.2, _position_pose(0.2))

        self.assertAlmostEqual(smoother.dp_prev[0], 1.0, places=4)
        self.assertAlmostEqual(smoother.dp_prev[1], 0.0, places=4)
        self.assertAlmostEqual(smoother.dp_prev[2], 0.0, places=4)

    def test_rotation_speed_is_measured_raw_to_raw(self) -> None:
        smoother = OneEuroPoseSmoother(
            min_cutoff=0.1,
            beta=0.0,
            d_cutoff=1.0,
            min_cutoff_rot=0.1,
            beta_rot=0.0,
            d_cutoff_rot=1e6,
        )

        smoother.smooth(0.0, _roll_pose(0.0))
        smoother.smooth(0.1, _roll_pose(0.1))
        smoother.smooth(0.2, _roll_pose(0.2))

        self.assertAlmostEqual(smoother.dq_prev, 1.0, places=4)

    def test_fast_forward_and_reverse_roll_do_not_switch_direction(self) -> None:
        smoother = OneEuroPoseSmoother(
            min_cutoff=2.0,
            beta=0.04,
            d_cutoff=1.5,
            min_cutoff_rot=3.0,
            beta_rot=0.25,
            d_cutoff_rot=3.0,
        )
        dt = 0.01
        step = 0.05
        angles = [step * index for index in range(20)]
        angles.extend(angles[-1] - step * index for index in range(1, 41))
        angles.extend(angles[-1] + step * index for index in range(1, 21))

        outputs = [
            smoother.smooth(index * dt, _roll_pose(angle))
            for index, angle in enumerate(angles)
        ]
        filtered = [pose for pose in outputs if pose is not None]
        deltas = np.array(
            [
                _roll_delta(previous, current)
                for previous, current in zip(filtered, filtered[1:])
            ]
        )

        self.assertTrue(np.all(deltas[:19] >= -1e-7))
        self.assertTrue(np.all(deltas[30:55] <= 1e-7))
        self.assertTrue(np.all(deltas[-10:] >= -1e-7))


if __name__ == "__main__":
    unittest.main()
