from __future__ import annotations

from typing import Protocol, runtime_checkable

from tk_infer.pi05_optimized.config import OptimizedRuntimeConfig

from .paired_trajectory import PairedTrajectory
from .temporal_optimizer import PairedTemporalTrajectoryProcessor, TemporalOptimizationConfig


@runtime_checkable
class TrajectoryProcessor(Protocol):
    phase: int
    allows_action_changes: bool

    @property
    def name(self) -> str: ...

    def process(self, trajectory: PairedTrajectory) -> PairedTrajectory: ...


class PassThroughTrajectoryProcessor:
    """Phase 0 processor that preserves values while breaking mutable aliases."""

    phase = 0
    allows_action_changes = False

    @property
    def name(self) -> str:
        return "pass_through"

    def process(self, trajectory: PairedTrajectory) -> PairedTrajectory:
        if not isinstance(trajectory, PairedTrajectory):
            raise TypeError(f"trajectory must be PairedTrajectory, got {type(trajectory)}")
        return trajectory.copy()


def build_trajectory_processor(config: OptimizedRuntimeConfig) -> TrajectoryProcessor:
    if not isinstance(config, OptimizedRuntimeConfig):
        raise TypeError("config must be OptimizedRuntimeConfig")
    if config.trajectory_processor == "pass_through":
        return PassThroughTrajectoryProcessor()
    if config.trajectory_processor == "paired_temporal":
        return PairedTemporalTrajectoryProcessor(
            TemporalOptimizationConfig(
                speed_factor=config.temporal_speed_factor,
                max_joint_step_rad=config.temporal_max_joint_step_rad,
                solver_timeout_s=config.temporal_solver_timeout_s,
            )
        )
    raise ValueError(f"unsupported trajectory processor: {config.trajectory_processor!r}")


__all__ = [
    "PassThroughTrajectoryProcessor",
    "TrajectoryProcessor",
    "build_trajectory_processor",
]
