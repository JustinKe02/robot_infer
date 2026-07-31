from __future__ import annotations

import hashlib
import importlib.metadata
import math
import threading
import time
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .paired_trajectory import (
    COMMAND_FORCE,
    LEFT_FORCE_INDEX,
    RIGHT_FORCE_INDEX,
    PairedTrajectory,
)

JOINT_DIM = 14
MODEL_GRIPPER_INDICES = (14, 15)
RAW_GRIPPER_INDICES = (14, 16)
TEMPORAL_QP_DEPENDENCIES = {"scipy": "1.15.3", "osqp": "1.0.4"}
_POSITION_TOLERANCE = 1e-12
_SAFETY_TOLERANCE = 1e-7


class TemporalOptimizationError(RuntimeError):
    """Base fail-closed error for temporal processing."""


class TemporalOptimizationInfeasibleError(TemporalOptimizationError):
    """Raised when no valid temporal solution can be produced."""


class TemporalOptimizationTimeoutError(TemporalOptimizationError):
    """Raised when a temporal solver exceeds its configured deadline."""


@dataclass(frozen=True, slots=True)
class TemporalOptimizationConfig:
    speed_factor: float = 1.0
    max_joint_step_rad: float = 0.02
    solver_timeout_s: float = 0.05
    bisection_iterations: int = 40

    def __post_init__(self) -> None:
        speed_factor = _finite_positive("speed_factor", self.speed_factor)
        if speed_factor > 2.0:
            raise ValueError("speed_factor must be <= 2.0")
        max_joint_step_rad = _finite_positive("max_joint_step_rad", self.max_joint_step_rad)
        solver_timeout_s = _finite_positive("solver_timeout_s", self.solver_timeout_s)
        if (
            isinstance(self.bisection_iterations, bool)
            or not isinstance(self.bisection_iterations, int)
            or not 8 <= self.bisection_iterations <= 80
        ):
            raise ValueError("bisection_iterations must be an integer in 8..80")
        object.__setattr__(self, "speed_factor", speed_factor)
        object.__setattr__(self, "max_joint_step_rad", max_joint_step_rad)
        object.__setattr__(self, "solver_timeout_s", solver_timeout_s)


@dataclass(frozen=True, slots=True)
class TemporalInterpolationMap:
    source_positions: NDArray[np.float64]
    source_steps: int
    solver_name: str

    def __post_init__(self) -> None:
        positions = np.array(self.source_positions, dtype=np.float64, copy=True)
        if isinstance(self.source_steps, bool) or not isinstance(self.source_steps, int) or self.source_steps < 1:
            raise ValueError("source_steps must be a positive integer")
        if positions.shape != (self.source_steps,):
            raise ValueError(
                f"source_positions must have shape ({self.source_steps},), got {positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("source_positions contains NaN/Inf")
        if positions[0] != 0.0:
            raise ValueError("source_positions must start at 0")
        if np.any(np.diff(positions) < 0):
            raise ValueError("source_positions must be monotonic non-decreasing")
        if np.any(positions < 0) or np.any(positions > self.source_steps - 1):
            raise ValueError("source_positions exceeds the source trajectory bounds")
        if not isinstance(self.solver_name, str) or not self.solver_name.strip():
            raise ValueError("solver_name must be a non-empty string")
        positions.setflags(write=False)
        object.__setattr__(self, "source_positions", positions)
        object.__setattr__(self, "solver_name", self.solver_name.strip())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.source_positions.tobytes()).hexdigest()


@runtime_checkable
class TemporalMapSolver(Protocol):
    @property
    def name(self) -> str: ...

    def solve(
        self,
        raw_joint_actions: NDArray[np.float32],
        config: TemporalOptimizationConfig,
    ) -> TemporalInterpolationMap: ...


class DeterministicVelocityMapSolver:
    """Greedy fixed-horizon time scaling with a hard per-step joint bound."""

    def __init__(self, *, clock: Any | None = None) -> None:
        self._clock = clock or time.perf_counter

    @property
    def name(self) -> str:
        return "deterministic_velocity_map_v1"

    def solve(
        self,
        raw_joint_actions: NDArray[np.float32],
        config: TemporalOptimizationConfig,
    ) -> TemporalInterpolationMap:
        joints = np.asarray(raw_joint_actions, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != JOINT_DIM or joints.shape[0] < 1:
            raise TemporalOptimizationInfeasibleError(
                f"raw_joint_actions must have shape (T,{JOINT_DIM}) with T >= 1, got {joints.shape}"
            )
        if not np.isfinite(joints).all():
            raise TemporalOptimizationInfeasibleError("raw_joint_actions contains NaN/Inf")
        started_s = self._clock()
        positions = np.zeros(joints.shape[0], dtype=np.float64)
        previous_output = joints[0].copy()
        for output_index in range(1, joints.shape[0]):
            _check_deadline(started_s, self._clock(), config.solver_timeout_s)
            desired_position = min(
                float(joints.shape[0] - 1),
                positions[output_index - 1] + config.speed_factor,
            )
            positions[output_index] = _furthest_safe_position(
                joints,
                start_position=positions[output_index - 1],
                desired_position=desired_position,
                previous_output=previous_output,
                max_joint_step_rad=config.max_joint_step_rad,
                bisection_iterations=config.bisection_iterations,
            )
            previous_output = _interpolate_one(joints, positions[output_index])
        _check_deadline(started_s, self._clock(), config.solver_timeout_s)
        return TemporalInterpolationMap(
            source_positions=positions,
            source_steps=joints.shape[0],
            solver_name=self.name,
        )


@dataclass(frozen=True, slots=True)
class TemporalOptimizationReport:
    solver_name: str
    source_steps: int
    output_steps: int
    speed_factor: float
    max_joint_step_rad: float
    input_max_joint_step_rad: float
    output_max_joint_step_rad: float
    limited_output_steps: int
    final_source_position: float
    source_completion_ratio: float
    interpolation_map_sha256: str
    solver_elapsed_s: float
    acceleration_objective_enabled: bool = False
    jerk_objective_enabled: bool = False
    force_slots_exact_80: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PairedTemporalTrajectoryProcessor:
    phase = 6
    allows_action_changes = True

    def __init__(
        self,
        config: TemporalOptimizationConfig | None = None,
        *,
        solver: TemporalMapSolver | None = None,
        clock: Any | None = None,
    ) -> None:
        self.config = config or TemporalOptimizationConfig()
        self.solver = solver or DeterministicVelocityMapSolver()
        if not isinstance(self.solver, TemporalMapSolver):
            raise TypeError("solver does not implement the TemporalMapSolver contract")
        self._clock = clock or time.perf_counter
        self._lock = threading.Lock()
        self._last_report: TemporalOptimizationReport | None = None
        self._last_interpolation_map: TemporalInterpolationMap | None = None

    @property
    def name(self) -> str:
        return "paired_temporal"

    @property
    def last_report(self) -> TemporalOptimizationReport | None:
        with self._lock:
            return self._last_report

    @property
    def last_interpolation_map(self) -> TemporalInterpolationMap | None:
        with self._lock:
            return self._last_interpolation_map

    def health(self) -> dict[str, object]:
        report = self.last_report
        return {
            "name": self.name,
            "phase": self.phase,
            "solver": self.solver.name,
            "speed_factor": self.config.speed_factor,
            "max_joint_step_rad": self.config.max_joint_step_rad,
            "solver_timeout_s": self.config.solver_timeout_s,
            "acceleration_objective_enabled": False,
            "jerk_objective_enabled": False,
            "optional_qp_dependencies": optional_qp_dependency_status(),
            "last_report": None if report is None else report.to_dict(),
        }

    def process(self, trajectory: PairedTrajectory) -> PairedTrajectory:
        if not isinstance(trajectory, PairedTrajectory):
            raise TypeError(f"trajectory must be PairedTrajectory, got {type(trajectory)}")
        started_s = self._clock()
        try:
            interpolation_map = self.solver.solve(trajectory.robot_actions[:, :JOINT_DIM], self.config)
        except TemporalOptimizationError:
            raise
        except Exception as exc:
            raise TemporalOptimizationError(
                f"temporal solver {self.solver.name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        elapsed_s = self._clock() - started_s
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise TemporalOptimizationError("temporal solver clock produced an invalid duration")
        if elapsed_s > self.config.solver_timeout_s:
            raise TemporalOptimizationTimeoutError(
                f"temporal solver exceeded {self.config.solver_timeout_s:.6f}s deadline: {elapsed_s:.6f}s"
            )
        if not isinstance(interpolation_map, TemporalInterpolationMap):
            raise TemporalOptimizationError(
                f"temporal solver must return TemporalInterpolationMap, got {type(interpolation_map)}"
            )
        if interpolation_map.source_steps != trajectory.steps:
            raise TemporalOptimizationError(
                "temporal solver returned a map for a different source trajectory length"
            )

        model_actions = _interpolate_actions(trajectory.model_actions, interpolation_map)
        robot_actions = _interpolate_actions(trajectory.robot_actions, interpolation_map)
        robot_actions[:, LEFT_FORCE_INDEX] = COMMAND_FORCE
        robot_actions[:, RIGHT_FORCE_INDEX] = COMMAND_FORCE
        _validate_joint_step_bound(robot_actions[:, :JOINT_DIM], self.config.max_joint_step_rad)
        processed = PairedTrajectory(
            model_actions=model_actions,
            robot_actions=robot_actions,
            request_id=trajectory.request_id,
            mode=trajectory.mode,
            source_observation_seq=trajectory.source_observation_seq,
            predicted_delay_steps=trajectory.predicted_delay_steps,
        )
        positions = interpolation_map.source_positions
        ideal_progress = np.minimum(
            np.arange(trajectory.steps, dtype=np.float64) * self.config.speed_factor,
            trajectory.steps - 1,
        )
        limited_steps = int(np.count_nonzero(positions + _POSITION_TOLERANCE < ideal_progress))
        denominator = max(trajectory.steps - 1, 1)
        report = TemporalOptimizationReport(
            solver_name=interpolation_map.solver_name,
            source_steps=trajectory.steps,
            output_steps=processed.steps,
            speed_factor=self.config.speed_factor,
            max_joint_step_rad=self.config.max_joint_step_rad,
            input_max_joint_step_rad=_max_joint_step(trajectory.robot_actions[:, :JOINT_DIM]),
            output_max_joint_step_rad=_max_joint_step(processed.robot_actions[:, :JOINT_DIM]),
            limited_output_steps=limited_steps,
            final_source_position=float(positions[-1]),
            source_completion_ratio=float(positions[-1] / denominator),
            interpolation_map_sha256=interpolation_map.fingerprint,
            solver_elapsed_s=elapsed_s,
        )
        with self._lock:
            self._last_report = report
            self._last_interpolation_map = interpolation_map
        return processed


def optional_qp_dependency_status() -> dict[str, dict[str, object]]:
    status: dict[str, dict[str, object]] = {}
    for distribution, required_version in TEMPORAL_QP_DEPENDENCIES.items():
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed_version = None
        status[distribution] = {
            "required_version": required_version,
            "installed_version": installed_version,
            "available": installed_version == required_version,
        }
    return status


def require_qp_dependencies() -> None:
    status = optional_qp_dependency_status()
    unavailable = [name for name, values in status.items() if not values["available"]]
    if unavailable:
        details = ", ".join(
            f"{name}=={status[name]['required_version']} (installed={status[name]['installed_version']})"
            for name in unavailable
        )
        raise TemporalOptimizationError(f"optional temporal QP dependencies unavailable: {details}")


def _furthest_safe_position(
    joints: NDArray[np.float64],
    *,
    start_position: float,
    desired_position: float,
    previous_output: NDArray[np.float64],
    max_joint_step_rad: float,
    bisection_iterations: int,
) -> float:
    position = start_position
    while position + _POSITION_TOLERANCE < desired_position:
        next_boundary = min(desired_position, float(math.floor(position + _POSITION_TOLERANCE) + 1))
        candidate = _interpolate_one(joints, next_boundary)
        if _within_joint_step(candidate, previous_output, max_joint_step_rad):
            position = next_boundary
            continue
        low = position
        high = next_boundary
        for _ in range(bisection_iterations):
            midpoint = (low + high) * 0.5
            midpoint_joints = _interpolate_one(joints, midpoint)
            if _within_joint_step(midpoint_joints, previous_output, max_joint_step_rad):
                low = midpoint
            else:
                high = midpoint
        return low
    return desired_position


def _interpolate_actions(
    actions: NDArray[np.float32],
    interpolation_map: TemporalInterpolationMap,
) -> NDArray[np.float32]:
    source = np.asarray(actions, dtype=np.float64)
    positions = interpolation_map.source_positions
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, interpolation_map.source_steps - 1)
    ratio = (positions - lower).reshape(-1, 1)
    output = source[lower] + (source[upper] - source[lower]) * ratio
    if not np.isfinite(output).all():
        raise TemporalOptimizationError("temporal interpolation produced NaN/Inf")
    return np.ascontiguousarray(output.astype(np.float32))


def _interpolate_one(actions: NDArray[np.float64], position: float) -> NDArray[np.float64]:
    lower = int(math.floor(position))
    upper = min(lower + 1, actions.shape[0] - 1)
    ratio = position - lower
    return actions[lower] + (actions[upper] - actions[lower]) * ratio


def _within_joint_step(
    candidate: NDArray[np.float64],
    previous: NDArray[np.float64],
    max_joint_step_rad: float,
) -> bool:
    return bool(np.max(np.abs(candidate - previous), initial=0.0) <= max_joint_step_rad)


def _validate_joint_step_bound(actions: NDArray[np.float32], max_joint_step_rad: float) -> None:
    measured = _max_joint_step(actions)
    if not math.isfinite(measured) or measured > max_joint_step_rad + _SAFETY_TOLERANCE:
        raise TemporalOptimizationError(
            f"temporal output joint step {measured:.9f} exceeds {max_joint_step_rad:.9f} rad"
        )


def _max_joint_step(actions: NDArray[np.float32]) -> float:
    if len(actions) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(np.asarray(actions, dtype=np.float64), axis=0)), initial=0.0))


def _check_deadline(started_s: object, now_s: object, timeout_s: float) -> None:
    if not isinstance(started_s, Real) or not isinstance(now_s, Real):
        raise TemporalOptimizationError("temporal solver clock must return real values")
    elapsed_s = float(now_s) - float(started_s)
    if not math.isfinite(elapsed_s) or elapsed_s < 0:
        raise TemporalOptimizationError("temporal solver clock moved backwards or returned NaN/Inf")
    if elapsed_s > timeout_s:
        raise TemporalOptimizationTimeoutError(
            f"temporal solver exceeded {timeout_s:.6f}s deadline: {elapsed_s:.6f}s"
        )


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


__all__ = [
    "DeterministicVelocityMapSolver",
    "JOINT_DIM",
    "MODEL_GRIPPER_INDICES",
    "PairedTemporalTrajectoryProcessor",
    "RAW_GRIPPER_INDICES",
    "TEMPORAL_QP_DEPENDENCIES",
    "TemporalInterpolationMap",
    "TemporalMapSolver",
    "TemporalOptimizationConfig",
    "TemporalOptimizationError",
    "TemporalOptimizationInfeasibleError",
    "TemporalOptimizationReport",
    "TemporalOptimizationTimeoutError",
    "optional_qp_dependency_status",
    "require_qp_dependencies",
]
