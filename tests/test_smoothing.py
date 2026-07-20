"""Regression tests for the optional independent rotation filter."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from dora_openarm_vr.smoothing import OneEuroPoseSmoother, _slerp_quat


def _roll_pose(angle: float, x: float = 0.0) -> np.ndarray:
    return np.array(
        [
            x,
            0.0,
            0.0,
            np.cos(0.5 * angle),
            np.sin(0.5 * angle),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def _roll_angle(pose: np.ndarray) -> float:
    return float(2.0 * np.arctan2(pose[4], pose[3]))


def _roll_delta(first: np.ndarray, second: np.ndarray) -> float:
    w1, x1 = first[3], first[4]
    w2, x2 = second[3], second[4]
    relative_w = w1 * w2 + x1 * x2
    relative_x = w1 * x2 - x1 * w2
    return float(2.0 * np.arctan2(relative_x, relative_w))


class OneEuroPoseSmootherTest(unittest.TestCase):
    def test_default_position_derivative_keeps_mainline_catchup_behavior(self) -> None:
        smoother = OneEuroPoseSmoother(
            min_cutoff=0.1,
            beta=0.0,
            d_cutoff=1e6,
        )

        smoother.smooth(0.0, _roll_pose(0.0, x=0.0))
        previous = smoother.smooth(0.1, _roll_pose(0.0, x=0.1))
        smoother.smooth(0.2, _roll_pose(0.0, x=0.2))

        assert previous is not None
        expected = (0.2 - float(previous[0])) / 0.1
        self.assertAlmostEqual(smoother.dp_prev[0], expected, places=4)
        self.assertGreater(smoother.dp_prev[0], 1.0)

    def test_raw_to_raw_speed_uses_consecutive_raw_positions(self) -> None:
        smoother = OneEuroPoseSmoother(
            min_cutoff=0.1,
            beta=0.0,
            d_cutoff=1e6,
            raw_to_raw_speed=True,
        )

        smoother.smooth(0.0, _roll_pose(0.0, x=0.0))
        smoother.smooth(0.1, _roll_pose(0.0, x=0.1))
        smoother.smooth(0.2, _roll_pose(0.0, x=0.2))

        self.assertAlmostEqual(smoother.dp_prev[0], 1.0, places=4)

    def test_rotation_parameters_do_not_change_default_mode(self) -> None:
        baseline = OneEuroPoseSmoother(
            min_cutoff=2.0,
            beta=0.04,
            d_cutoff=1.5,
        )
        configured = OneEuroPoseSmoother(
            min_cutoff=2.0,
            beta=0.04,
            d_cutoff=1.5,
            min_cutoff_rot=30.0,
            beta_rot=10.0,
            d_cutoff_rot=30.0,
        )

        poses = [_roll_pose(0.0), _roll_pose(0.2), _roll_pose(0.4)]
        baseline_outputs = [
            baseline.smooth(index * 0.01, pose) for index, pose in enumerate(poses)
        ]
        configured_outputs = [
            configured.smooth(index * 0.01, pose) for index, pose in enumerate(poses)
        ]

        for expected, actual in zip(baseline_outputs, configured_outputs):
            np.testing.assert_allclose(actual, expected)

    def test_independent_mode_uses_filtered_to_raw_angular_speed_by_default(
        self,
    ) -> None:
        smoother = OneEuroPoseSmoother(
            min_cutoff=2.0,
            beta=0.04,
            d_cutoff=1.5,
            independent_rotation_filter=True,
            min_cutoff_rot=3.0,
            beta_rot=0.25,
            d_cutoff_rot=1e6,
        )

        smoother.smooth(0.0, _roll_pose(0.0))
        previous = smoother.smooth(0.1, _roll_pose(0.1))
        smoother.smooth(0.2, _roll_pose(0.2))

        assert previous is not None
        expected = abs(_roll_delta(previous, _roll_pose(0.2))) / 0.1
        self.assertAlmostEqual(smoother.dq_prev, expected, places=4)
        self.assertGreater(smoother.dq_prev, 1.0)

    def test_raw_to_raw_speed_uses_consecutive_raw_quaternions(self) -> None:
        smoother = OneEuroPoseSmoother(
            independent_rotation_filter=True,
            raw_to_raw_speed=True,
            min_cutoff_rot=3.0,
            beta_rot=0.25,
            d_cutoff_rot=1e6,
        )

        smoother.smooth(0.0, _roll_pose(0.0))
        smoother.smooth(0.1, _roll_pose(0.1))
        smoother.smooth(0.2, _roll_pose(0.2))

        self.assertAlmostEqual(smoother.dq_prev, 1.0, places=4)

    def test_independent_mode_keeps_shortest_path_slerp(self) -> None:
        smoother = OneEuroPoseSmoother(independent_rotation_filter=True)
        smoother.smooth(0.0, _roll_pose(0.0))

        with patch("dora_openarm_vr.smoothing._slerp_quat", wraps=_slerp_quat) as slerp:
            smoother.smooth(0.1, _roll_pose(0.2))

        self.assertTrue(slerp.call_args.kwargs["shortest"])

    def test_continuous_path_alone_controls_unwrap_and_slerp_path(self) -> None:
        independent = OneEuroPoseSmoother(independent_rotation_filter=True)
        continuous = OneEuroPoseSmoother(continuous_rotation_path=True)
        initial = _roll_pose(0.0)
        equivalent_with_opposite_sign = initial.copy()
        equivalent_with_opposite_sign[3:7] *= -1.0

        independent.smooth(0.0, initial)
        continuous.smooth(0.0, initial)
        independent.smooth(0.1, equivalent_with_opposite_sign)

        with patch("dora_openarm_vr.smoothing._slerp_quat", wraps=_slerp_quat) as slerp:
            continuous.smooth(0.1, equivalent_with_opposite_sign)

        assert independent.q_raw_prev is not None
        assert continuous.q_raw_prev is not None
        self.assertLess(np.dot(initial[3:7], independent.q_raw_prev), 0.0)
        self.assertGreater(np.dot(initial[3:7], continuous.q_raw_prev), 0.0)
        self.assertFalse(slerp.call_args.kwargs["shortest"])

    def test_toggle_changes_only_rotation_response(self) -> None:
        kwargs = dict(min_cutoff=2.0, beta=0.04, d_cutoff=1.5)
        baseline = OneEuroPoseSmoother(**kwargs)
        independent = OneEuroPoseSmoother(
            **kwargs,
            independent_rotation_filter=True,
            min_cutoff_rot=3.0,
            beta_rot=0.25,
            d_cutoff_rot=3.0,
        )

        first = _roll_pose(0.0, x=0.0)
        second = _roll_pose(0.5, x=0.1)
        baseline.smooth(0.0, first)
        independent.smooth(0.0, first)
        baseline_output = baseline.smooth(0.01, second)
        independent_output = independent.smooth(0.01, second)

        assert baseline_output is not None
        assert independent_output is not None
        np.testing.assert_allclose(independent_output[:3], baseline_output[:3])
        self.assertNotAlmostEqual(
            _roll_angle(independent_output),
            _roll_angle(baseline_output),
            places=5,
        )

    def test_independent_mode_preserves_fast_roll_direction(self) -> None:
        smoother = OneEuroPoseSmoother(
            min_cutoff=2.0,
            beta=0.04,
            d_cutoff=1.5,
            independent_rotation_filter=True,
            min_cutoff_rot=3.0,
            beta_rot=0.25,
            d_cutoff_rot=3.0,
        )
        step = 0.05
        angles = [step * index for index in range(20)]
        angles.extend(angles[-1] - step * index for index in range(1, 41))
        angles.extend(angles[-1] + step * index for index in range(1, 21))

        outputs = [
            smoother.smooth(index * 0.01, _roll_pose(angle))
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
