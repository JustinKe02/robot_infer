from __future__ import annotations

import numpy as np
import pytest

from tk_infer.pi05_optimized.runtime.paired_trajectory import PairedTrajectory
from tk_infer.pi05_optimized.runtime.trajectory_processor import PassThroughTrajectoryProcessor

from .helpers import make_request, make_response


def _trajectory() -> PairedTrajectory:
    request = make_request(mode="rtc")
    return PairedTrajectory.from_response(
        make_response(request),
        source_observation_seq=request.obs_sequence_id,
        predicted_delay_steps=request.predicted_delay_steps,
    )


def test_paired_trajectory_copies_input_and_makes_it_read_only() -> None:
    request = make_request()
    response = make_response(request)
    original_model = response.raw_actions.copy()
    original_robot = response.processed_actions.copy()

    trajectory = PairedTrajectory.from_response(
        response,
        source_observation_seq=request.obs_sequence_id,
        predicted_delay_steps=request.predicted_delay_steps,
    )
    response.raw_actions[:] = -1
    response.processed_actions[:] = -1

    np.testing.assert_array_equal(trajectory.model_actions, original_model)
    np.testing.assert_array_equal(trajectory.robot_actions, original_robot)
    assert trajectory.model_actions.flags.writeable is False
    assert trajectory.robot_actions.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        trajectory.model_actions[0, 0] = 1


def test_pass_through_is_numerically_exact_and_breaks_aliases() -> None:
    original = _trajectory()

    processed = PassThroughTrajectoryProcessor().process(original)

    assert processed is not original
    assert not np.shares_memory(processed.model_actions, original.model_actions)
    assert not np.shares_memory(processed.robot_actions, original.robot_actions)
    np.testing.assert_array_equal(processed.model_actions, original.model_actions)
    np.testing.assert_array_equal(processed.robot_actions, original.robot_actions)
    assert processed.request_id == original.request_id
    assert processed.source_observation_seq == original.source_observation_seq


@pytest.mark.parametrize(
    ("model_shape", "robot_shape", "message"),
    [
        ((3, 15), (3, 18), "model_actions must have shape"),
        ((3, 16), (3, 17), "robot_actions must have shape"),
        ((2, 16), (3, 18), "same temporal length"),
        ((0, 16), (0, 18), "trajectory length must be"),
    ],
)
def test_paired_trajectory_rejects_shape_errors(
    model_shape: tuple[int, int],
    robot_shape: tuple[int, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PairedTrajectory(
            model_actions=np.zeros(model_shape, dtype=np.float32),
            robot_actions=np.zeros(robot_shape, dtype=np.float32),
            request_id=1,
            mode="single_step",
            source_observation_seq=1,
            predicted_delay_steps=0,
        )


def test_paired_trajectory_rejects_non_finite_and_wrong_force() -> None:
    request = make_request()
    response = make_response(request)
    response.raw_actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        PairedTrajectory(
            model_actions=response.raw_actions,
            robot_actions=response.processed_actions,
            request_id=1,
            mode="single_step",
            source_observation_seq=1,
            predicted_delay_steps=0,
        )

    response = make_response(request)
    response.processed_actions[0, 17] = 79.0
    with pytest.raises(ValueError, match=r"force slots \[15, 17\] must be exactly 80"):
        PairedTrajectory(
            model_actions=response.raw_actions,
            robot_actions=response.processed_actions,
            request_id=1,
            mode="single_step",
            source_observation_seq=1,
            predicted_delay_steps=0,
        )
