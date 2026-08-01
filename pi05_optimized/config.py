from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal, TypeAlias

from lerobot.configs.types import RTCAttentionSchedule
from tk_infer.pi05.runtime.protocol import MAX_ACTION_CHUNK_STEPS
from tk_infer.pi05_optimized.constants import DEFAULT_TRACE_BACKUP_COUNT, DEFAULT_TRACE_MAX_BYTES

BackendName: TypeAlias = Literal[
    "torch",
    "torch_optimized",
    "torch_rtc_conditioned",
    "triton",
    "realtime_vla_v2",
]
TrajectoryProcessorName: TypeAlias = Literal["pass_through", "paired_temporal"]

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 18088
DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024
DEFAULT_METRICS_WINDOW_SIZE = 512
OPTIMIZED_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class OptimizedRuntimeConfig:
    """Validated optimized runtime configuration with immutable semantics."""

    backend: BackendName = "torch"
    trajectory_processor: TrajectoryProcessorName = "pass_through"
    server_host: str = DEFAULT_SERVER_HOST
    server_port: int = DEFAULT_SERVER_PORT
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    device: str = "cuda"
    policy_path: Path | None = None
    tokenizer_path: Path | None = None
    triton_artifact_path: Path | None = None
    realtime_vla_v2_artifact_path: Path | None = None
    require_complete_step: bool = True
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = RTCAttentionSchedule.LINEAR.value
    rtc_debug: bool = False
    rtc_conditioned_task: str | None = None
    metrics_window_size: int = DEFAULT_METRICS_WINDOW_SIZE
    trace_path: Path | None = None
    trace_strict: bool = False
    trace_max_bytes: int = DEFAULT_TRACE_MAX_BYTES
    trace_backup_count: int = DEFAULT_TRACE_BACKUP_COUNT
    torch_inference_mode: bool = False
    torch_bf16_autocast: bool = False
    torch_pinned_memory: bool = False
    torch_non_blocking_copies: bool = False
    torch_static_buffers: bool = False
    torch_cuda_graph: bool = False
    torch_warmup_iterations: int = 0
    torch_warmup_seed: int = 12345
    temporal_speed_factor: float = 1.0
    temporal_max_joint_step_rad: float = 0.02
    temporal_solver_timeout_s: float = 0.05

    def __post_init__(self) -> None:
        if self.backend not in {
            "torch",
            "torch_optimized",
            "torch_rtc_conditioned",
            "triton",
            "realtime_vla_v2",
        }:
            raise ValueError(
                "backend must be 'torch', 'torch_optimized', 'torch_rtc_conditioned', 'triton', "
                "or 'realtime_vla_v2', "
                f"got {self.backend!r}"
            )
        if self.trajectory_processor not in {"pass_through", "paired_temporal"}:
            raise ValueError(
                "trajectory_processor must be 'pass_through' or 'paired_temporal', "
                f"got {self.trajectory_processor!r}"
            )
        if not isinstance(self.server_host, str) or not self.server_host.strip():
            raise ValueError("server_host must be a non-empty string")
        if isinstance(self.server_port, bool) or not isinstance(self.server_port, int):
            raise ValueError("server_port must be an integer")
        if not 0 <= self.server_port <= 65535:
            raise ValueError("server_port must be in 0..65535")
        if isinstance(self.max_request_bytes, bool) or not isinstance(self.max_request_bytes, int):
            raise ValueError("max_request_bytes must be an integer")
        if self.max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be positive")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if not isinstance(self.require_complete_step, bool):
            raise ValueError("require_complete_step must be boolean")
        if isinstance(self.rtc_execution_horizon, bool) or not isinstance(self.rtc_execution_horizon, int):
            raise ValueError("rtc_execution_horizon must be an integer")
        if not 1 <= self.rtc_execution_horizon <= MAX_ACTION_CHUNK_STEPS:
            raise ValueError(
                f"rtc_execution_horizon must be in 1..{MAX_ACTION_CHUNK_STEPS}, "
                f"got {self.rtc_execution_horizon}"
            )
        if isinstance(self.rtc_max_guidance_weight, bool) or not isinstance(
            self.rtc_max_guidance_weight, Real
        ):
            raise ValueError("rtc_max_guidance_weight must be a real number")
        guidance_weight = float(self.rtc_max_guidance_weight)
        if not math.isfinite(guidance_weight) or guidance_weight < 0:
            raise ValueError("rtc_max_guidance_weight must be finite and non-negative")
        try:
            RTCAttentionSchedule(self.rtc_prefix_attention_schedule)
        except ValueError as exc:
            choices = ", ".join(schedule.value for schedule in RTCAttentionSchedule)
            raise ValueError(
                f"rtc_prefix_attention_schedule must be one of [{choices}], "
                f"got {self.rtc_prefix_attention_schedule!r}"
            ) from exc
        if not isinstance(self.rtc_debug, bool):
            raise ValueError("rtc_debug must be boolean")
        rtc_conditioned_task = (
            None if self.rtc_conditioned_task is None else self.rtc_conditioned_task.strip()
        )
        rtc_conditioned_backends = {"torch_rtc_conditioned", "realtime_vla_v2"}
        if self.backend in rtc_conditioned_backends and not rtc_conditioned_task:
            raise ValueError(f"backend={self.backend!r} requires rtc_conditioned_task")
        if self.backend not in rtc_conditioned_backends and rtc_conditioned_task is not None:
            raise ValueError(
                "rtc_conditioned_task requires backend='torch_rtc_conditioned' or backend='realtime_vla_v2'"
            )
        if (
            isinstance(self.metrics_window_size, bool)
            or not isinstance(self.metrics_window_size, int)
            or self.metrics_window_size <= 0
        ):
            raise ValueError("metrics_window_size must be a positive integer")
        if not isinstance(self.trace_strict, bool):
            raise ValueError("trace_strict must be boolean")
        if isinstance(self.trace_max_bytes, bool) or not isinstance(self.trace_max_bytes, int):
            raise ValueError("trace_max_bytes must be an integer")
        if self.trace_max_bytes < 256:
            raise ValueError("trace_max_bytes must be >= 256")
        if (
            isinstance(self.trace_backup_count, bool)
            or not isinstance(self.trace_backup_count, int)
            or self.trace_backup_count < 0
        ):
            raise ValueError("trace_backup_count must be a non-negative integer")
        torch_flags = {
            "torch_inference_mode": self.torch_inference_mode,
            "torch_bf16_autocast": self.torch_bf16_autocast,
            "torch_pinned_memory": self.torch_pinned_memory,
            "torch_non_blocking_copies": self.torch_non_blocking_copies,
            "torch_static_buffers": self.torch_static_buffers,
            "torch_cuda_graph": self.torch_cuda_graph,
        }
        for name, value in torch_flags.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        if (
            isinstance(self.torch_warmup_iterations, bool)
            or not isinstance(self.torch_warmup_iterations, int)
            or self.torch_warmup_iterations < 0
        ):
            raise ValueError("torch_warmup_iterations must be a non-negative integer")
        if isinstance(self.torch_warmup_seed, bool) or not isinstance(self.torch_warmup_seed, int):
            raise ValueError("torch_warmup_seed must be an integer")
        temporal_speed_factor = _finite_positive("temporal_speed_factor", self.temporal_speed_factor)
        if temporal_speed_factor > 2.0:
            raise ValueError("temporal_speed_factor must be <= 2.0")
        temporal_max_joint_step_rad = _finite_positive(
            "temporal_max_joint_step_rad", self.temporal_max_joint_step_rad
        )
        temporal_solver_timeout_s = _finite_positive(
            "temporal_solver_timeout_s", self.temporal_solver_timeout_s
        )
        temporal_defaults_changed = (
            temporal_speed_factor != 1.0
            or temporal_max_joint_step_rad != 0.02
            or temporal_solver_timeout_s != 0.05
        )
        if self.trajectory_processor == "pass_through" and temporal_defaults_changed:
            raise ValueError("temporal optimization settings require trajectory_processor='paired_temporal'")
        optimized_requested = any(torch_flags.values()) or self.torch_warmup_iterations > 0
        if self.backend != "torch_optimized" and optimized_requested:
            raise ValueError("torch optimization flags require backend='torch_optimized'")
        if self.backend in {"triton", "realtime_vla_v2"} and not self.device.strip().lower().startswith(
            "cuda"
        ):
            raise ValueError(f"backend={self.backend!r} requires a CUDA device")
        if self.torch_non_blocking_copies and not self.torch_pinned_memory:
            raise ValueError("torch_non_blocking_copies requires torch_pinned_memory=true")
        if self.torch_cuda_graph and not self.torch_static_buffers:
            raise ValueError("torch_cuda_graph requires torch_static_buffers=true")
        cuda_only_requested = any(
            (
                self.torch_bf16_autocast,
                self.torch_pinned_memory,
                self.torch_non_blocking_copies,
                self.torch_static_buffers,
                self.torch_cuda_graph,
            )
        )
        if cuda_only_requested and not self.device.strip().lower().startswith("cuda"):
            raise ValueError("BF16 and memory optimization flags require a CUDA device")
        trace_path = _normalize_optional_path(self.trace_path)
        if trace_path is not None:
            trace_path = trace_path.resolve()
            if not trace_path.is_relative_to(OPTIMIZED_ROOT):
                raise ValueError(f"trace_path must stay inside {OPTIMIZED_ROOT}, got {trace_path}")
        object.__setattr__(self, "server_host", self.server_host.strip())
        object.__setattr__(self, "device", self.device.strip())
        object.__setattr__(self, "rtc_max_guidance_weight", guidance_weight)
        object.__setattr__(self, "rtc_conditioned_task", rtc_conditioned_task)
        object.__setattr__(self, "temporal_speed_factor", temporal_speed_factor)
        object.__setattr__(self, "temporal_max_joint_step_rad", temporal_max_joint_step_rad)
        object.__setattr__(self, "temporal_solver_timeout_s", temporal_solver_timeout_s)
        object.__setattr__(self, "policy_path", _normalize_optional_path(self.policy_path))
        object.__setattr__(self, "tokenizer_path", _normalize_optional_path(self.tokenizer_path))
        triton_artifact_path = _normalize_optional_path(self.triton_artifact_path)
        if triton_artifact_path is not None:
            triton_artifact_path = triton_artifact_path.resolve()
            if not triton_artifact_path.is_relative_to(OPTIMIZED_ROOT):
                raise ValueError(
                    f"triton_artifact_path must stay inside {OPTIMIZED_ROOT}, got {triton_artifact_path}"
                )
        object.__setattr__(self, "triton_artifact_path", triton_artifact_path)
        realtime_vla_v2_artifact_path = _normalize_optional_path(self.realtime_vla_v2_artifact_path)
        if realtime_vla_v2_artifact_path is not None:
            realtime_vla_v2_artifact_path = realtime_vla_v2_artifact_path.resolve()
            if not realtime_vla_v2_artifact_path.is_relative_to(OPTIMIZED_ROOT):
                raise ValueError(
                    "realtime_vla_v2_artifact_path must stay inside "
                    f"{OPTIMIZED_ROOT}, got {realtime_vla_v2_artifact_path}"
                )
        object.__setattr__(
            self,
            "realtime_vla_v2_artifact_path",
            realtime_vla_v2_artifact_path,
        )
        object.__setattr__(self, "trace_path", trace_path)

    def require_model_paths(self) -> tuple[Path, Path]:
        if self.policy_path is None:
            raise ValueError("policy_path is required to load the torch backend")
        if self.tokenizer_path is None:
            raise ValueError("tokenizer_path is required to load the torch backend")
        return self.policy_path, self.tokenizer_path

    def require_triton_artifact_path(self) -> Path:
        if self.triton_artifact_path is None:
            raise ValueError("triton_artifact_path is required to load the Triton backend")
        return self.triton_artifact_path

    def require_realtime_vla_v2_artifact_path(self) -> Path:
        if self.realtime_vla_v2_artifact_path is None:
            raise ValueError("realtime_vla_v2_artifact_path is required to load the Realtime-VLA v2 backend")
        return self.realtime_vla_v2_artifact_path

    def public_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "trajectory_processor": self.trajectory_processor,
            "server_host": self.server_host,
            "server_port": self.server_port,
            "max_request_bytes": self.max_request_bytes,
            "device": self.device,
            "policy_path": None if self.policy_path is None else str(self.policy_path),
            "tokenizer_path": None if self.tokenizer_path is None else str(self.tokenizer_path),
            "triton_artifact_path": (
                None if self.triton_artifact_path is None else str(self.triton_artifact_path)
            ),
            "realtime_vla_v2_artifact_path": (
                None
                if self.realtime_vla_v2_artifact_path is None
                else str(self.realtime_vla_v2_artifact_path)
            ),
            "require_complete_step": self.require_complete_step,
            "rtc_execution_horizon": self.rtc_execution_horizon,
            "rtc_max_guidance_weight": self.rtc_max_guidance_weight,
            "rtc_prefix_attention_schedule": self.rtc_prefix_attention_schedule,
            "rtc_debug": self.rtc_debug,
            "rtc_conditioned_task": self.rtc_conditioned_task,
            "metrics_window_size": self.metrics_window_size,
            "trace_path": None if self.trace_path is None else str(self.trace_path),
            "trace_strict": self.trace_strict,
            "trace_max_bytes": self.trace_max_bytes,
            "trace_backup_count": self.trace_backup_count,
            "torch_inference_mode": self.torch_inference_mode,
            "torch_bf16_autocast": self.torch_bf16_autocast,
            "torch_pinned_memory": self.torch_pinned_memory,
            "torch_non_blocking_copies": self.torch_non_blocking_copies,
            "torch_static_buffers": self.torch_static_buffers,
            "torch_cuda_graph": self.torch_cuda_graph,
            "torch_warmup_iterations": self.torch_warmup_iterations,
            "torch_warmup_seed": self.torch_warmup_seed,
            "temporal_speed_factor": self.temporal_speed_factor,
            "temporal_max_joint_step_rad": self.temporal_max_joint_step_rad,
            "temporal_solver_timeout_s": self.temporal_solver_timeout_s,
        }


def _normalize_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


__all__ = [
    "BackendName",
    "DEFAULT_MAX_REQUEST_BYTES",
    "DEFAULT_METRICS_WINDOW_SIZE",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "OptimizedRuntimeConfig",
    "TrajectoryProcessorName",
]
