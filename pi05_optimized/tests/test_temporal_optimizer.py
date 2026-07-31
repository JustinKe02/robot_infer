from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tk_infer.pi05_optimized.backends.torch_backend import TorchPolicyBackend
from tk_infer.pi05_optimized.runtime.paired_trajectory import PairedTrajectory
from tk_infer.pi05_optimized.runtime.policy_service import OptimizedPolicyService
from tk_infer.pi05_optimized.runtime.temporal_optimizer import (
    PairedTemporalTrajectoryProcessor,
    TemporalInterpolationMap,
    TemporalOptimizationConfig,
    TemporalOptimizationError,
    TemporalOptimizationInfeasibleError,
    TemporalOptimizationTimeoutError,
    optional_qp_dependency_status,
    require_qp_dependencies,
)

from .helpers import FakeReferenceService, make_request


def _trajectory(*, steps: int = 5, joint_step: float = 0.01) -> PairedTrajectory:
    source_position = np.arange(steps, dtype=np.float32)
    model = np.zeros((steps, 16), dtype=np.float32)
    raw = np.zeros((steps, 18), dtype=np.float32)
    for joint_index in range(14):
        model[:, joint_index] = source_position * (10.0 + joint_index)
        raw[:, joint_index] = source_position * joint_step * (1.0 - joint_index * 0.01)
    model[:, 14] = source_position * 3.0
    model[:, 15] = source_position * -4.0
    raw[:, 14] = source_position * 7.0
    raw[:, 15] = 80.0
    raw[:, 16] = source_position * -8.0
    raw[:, 17] = 80.0
    return PairedTrajectory(
        model_actions=model,
        robot_actions=raw,
        request_id=4,
        mode="rtc",
        source_observation_seq=9,
        predicted_delay_steps=2,
    )


def test_speed_one_is_exact_when_source_already_meets_joint_bound() -> None:
    source = _trajectory(joint_step=0.01)
    processor = PairedTemporalTrajectoryProcessor()

    output = processor.process(source)

    np.testing.assert_array_equal(output.model_actions, source.model_actions)
    np.testing.assert_array_equal(output.robot_actions, source.robot_actions)
    assert output is not source
    report = processor.last_report
    assert report is not None
    assert report.speed_factor == 1.0
    assert report.limited_output_steps == 0
    assert report.final_source_position == source.steps - 1
    assert report.acceleration_objective_enabled is False
    assert report.jerk_objective_enabled is False


def test_one_map_limits_joints_and_resamples_model_raw_and_grippers_independently() -> None:
    source = _trajectory(joint_step=0.04)
    processor = PairedTemporalTrajectoryProcessor(
        TemporalOptimizationConfig(max_joint_step_rad=0.02)
    )

    output = processor.process(source)

    source_positions_from_model = output.model_actions[:, 0] / 10.0
    source_positions_from_raw = output.robot_actions[:, 0] / 0.04
    np.testing.assert_allclose(source_positions_from_model, source_positions_from_raw, atol=1e-6)
    np.testing.assert_allclose(output.model_actions[:, 14] / 3.0, source_positions_from_model)
    np.testing.assert_allclose(output.robot_actions[:, 14] / 7.0, source_positions_from_model)
    assert not np.array_equal(output.model_actions[:, 14], output.robot_actions[:, 14])
    assert np.max(np.abs(np.diff(output.robot_actions[:, :14], axis=0))) <= 0.0200001
    np.testing.assert_array_equal(output.robot_actions[:, 15], 80.0)
    np.testing.assert_array_equal(output.robot_actions[:, 17], 80.0)
    assert processor.last_report is not None
    assert processor.last_report.limited_output_steps > 0
    assert processor.last_report.source_completion_ratio < 1.0


def test_gripper_jumps_do_not_change_joint_time_map_or_force_slots() -> None:
    source = _trajectory(joint_step=0.01)
    raw = source.robot_actions.copy()
    raw[:, 14] = np.arange(source.steps, dtype=np.float32) * 1000.0
    raw[:, 16] = np.arange(source.steps, dtype=np.float32) * -2000.0
    gripper_jump = replace(source, robot_actions=raw)

    output = PairedTemporalTrajectoryProcessor().process(gripper_jump)

    np.testing.assert_array_equal(output.model_actions, source.model_actions)
    np.testing.assert_array_equal(output.robot_actions[:, 14], raw[:, 14])
    np.testing.assert_array_equal(output.robot_actions[:, 16], raw[:, 16])
    np.testing.assert_array_equal(output.robot_actions[:, 15], 80.0)
    np.testing.assert_array_equal(output.robot_actions[:, 17], 80.0)


def test_solver_exception_is_wrapped_and_never_falls_back() -> None:
    class ExplodingSolver:
        @property
        def name(self) -> str:
            return "exploding"

        def solve(self, raw_joint_actions: np.ndarray, config: TemporalOptimizationConfig) -> object:
            raise OSError("solver unavailable")

    processor = PairedTemporalTrajectoryProcessor(solver=ExplodingSolver())  # type: ignore[arg-type]

    with pytest.raises(TemporalOptimizationError, match="solver unavailable"):
        processor.process(_trajectory())
    assert processor.last_report is None


def test_solver_infeasible_and_timeout_fail_closed() -> None:
    class InfeasibleSolver:
        @property
        def name(self) -> str:
            return "infeasible"

        def solve(self, raw_joint_actions: np.ndarray, config: TemporalOptimizationConfig) -> object:
            raise TemporalOptimizationInfeasibleError("infeasible constraints")

    with pytest.raises(TemporalOptimizationInfeasibleError, match="infeasible constraints"):
        PairedTemporalTrajectoryProcessor(solver=InfeasibleSolver()).process(_trajectory())  # type: ignore[arg-type]

    class IdentitySolver:
        @property
        def name(self) -> str:
            return "identity"

        def solve(
            self, raw_joint_actions: np.ndarray, config: TemporalOptimizationConfig
        ) -> TemporalInterpolationMap:
            return TemporalInterpolationMap(
                np.arange(len(raw_joint_actions), dtype=np.float64),
                len(raw_joint_actions),
                self.name,
            )

    clock = iter((0.0, 0.1))
    processor = PairedTemporalTrajectoryProcessor(
        TemporalOptimizationConfig(solver_timeout_s=0.05),
        solver=IdentitySolver(),
        clock=lambda: next(clock),
    )
    with pytest.raises(TemporalOptimizationTimeoutError, match="exceeded"):
        processor.process(_trajectory())


def test_invalid_solver_result_and_joint_bound_violation_fail_closed() -> None:
    class InvalidResultSolver:
        @property
        def name(self) -> str:
            return "invalid"

        def solve(self, raw_joint_actions: np.ndarray, config: TemporalOptimizationConfig) -> object:
            return object()

    with pytest.raises(TemporalOptimizationError, match="must return TemporalInterpolationMap"):
        PairedTemporalTrajectoryProcessor(solver=InvalidResultSolver()).process(_trajectory())  # type: ignore[arg-type]

    class UnsafeIdentitySolver:
        @property
        def name(self) -> str:
            return "unsafe_identity"

        def solve(
            self, raw_joint_actions: np.ndarray, config: TemporalOptimizationConfig
        ) -> TemporalInterpolationMap:
            return TemporalInterpolationMap(
                np.arange(len(raw_joint_actions), dtype=np.float64),
                len(raw_joint_actions),
                self.name,
            )

    processor = PairedTemporalTrajectoryProcessor(solver=UnsafeIdentitySolver())
    with pytest.raises(TemporalOptimizationError, match="exceeds"):
        processor.process(_trajectory(joint_step=0.04))


def test_optional_qp_dependencies_are_explicitly_version_gated() -> None:
    status = optional_qp_dependency_status()

    assert set(status) == {"scipy", "osqp"}
    assert status["scipy"]["required_version"] == "1.15.3"
    assert status["osqp"]["required_version"] == "1.0.4"
    if not all(values["available"] for values in status.values()):
        with pytest.raises(TemporalOptimizationError, match="dependencies unavailable"):
            require_qp_dependencies()


def test_policy_service_allows_only_declared_temporal_changes_and_reports_phase() -> None:
    backend = TorchPolicyBackend(FakeReferenceService())  # type: ignore[arg-type]
    processor = PairedTemporalTrajectoryProcessor()
    service = OptimizedPolicyService(backend=backend, trajectory_processor=processor)

    response = service.infer(make_request())

    assert response.processed_actions.shape == (3, 18)
    assert np.max(np.abs(np.diff(response.processed_actions[:, :14], axis=0))) <= 0.0200001
    health = service.health()
    assert health["optimized_runtime_phase"] == 6
    assert health["trajectory_processor"] == "paired_temporal"
    assert health["trajectory_processor_health"]["last_report"] is not None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"speed_factor": 0.0}, "finite and positive"),
        ({"speed_factor": 2.1}, "must be <= 2.0"),
        ({"max_joint_step_rad": float("nan")}, "finite and positive"),
        ({"solver_timeout_s": True}, "must be a real number"),
        ({"bisection_iterations": 4}, "integer in 8..80"),
    ],
)
def test_temporal_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TemporalOptimizationConfig(**kwargs)  # type: ignore[arg-type]
