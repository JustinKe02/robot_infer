from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from numbers import Real
from typing import Any

import numpy as np

from tk_infer.pi05.runtime.protocol import (
    MODEL_ACTION_DIM,
    PROTOCOL_VERSION,
    SUPPORTED_MODES,
    WIRE_ACTION_DIM,
    InferenceRequest,
    InferenceResponse,
)
from tk_infer.pi05_optimized.backends.base import PolicyBackend

from .metrics import InferenceMetrics, InferenceTimings
from .paired_trajectory import PairedTrajectory
from .trace import InferenceTraceRecorder
from .trajectory_processor import PassThroughTrajectoryProcessor, TrajectoryProcessor


class OptimizedPolicyService:
    """Fail-closed service facade for backend and paired-trajectory processing."""

    def __init__(
        self,
        *,
        backend: PolicyBackend,
        trajectory_processor: TrajectoryProcessor | None = None,
        metrics: InferenceMetrics | None = None,
        trace_recorder: InferenceTraceRecorder | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(backend, PolicyBackend):
            raise TypeError("backend does not implement the PolicyBackend contract")
        processor = trajectory_processor or PassThroughTrajectoryProcessor()
        if not isinstance(processor, TrajectoryProcessor):
            raise TypeError("trajectory_processor does not implement the TrajectoryProcessor contract")
        if trace_recorder is not None and not isinstance(trace_recorder, InferenceTraceRecorder):
            raise TypeError("trace_recorder does not implement the InferenceTraceRecorder contract")
        self.backend = backend
        self.trajectory_processor = processor
        self.metrics = metrics or InferenceMetrics()
        self.trace_recorder = trace_recorder
        self._clock = clock or time.perf_counter
        self._execution_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.inference_count = 0
        self.failure_count = 0
        self.last_latency_s = 0.0

    def health(self) -> dict[str, Any]:
        health = dict(self.backend.health())
        backend_phase = health.get("backend_phase", 1)
        if isinstance(backend_phase, bool) or not isinstance(backend_phase, int) or backend_phase < 1:
            raise ValueError(f"backend_phase must be a positive integer, got {backend_phase!r}")
        processor_phase = getattr(self.trajectory_processor, "phase", None)
        if isinstance(processor_phase, bool) or not isinstance(processor_phase, int) or processor_phase < 0:
            raise ValueError(f"trajectory processor reported invalid phase: {processor_phase!r}")
        backend_stage_metrics = health.get("backend_stage_metrics")
        backend_supported_modes = health.get("supported_modes", list(SUPPORTED_MODES))
        if (
            not isinstance(backend_supported_modes, list | tuple)
            or not backend_supported_modes
            or any(mode not in SUPPORTED_MODES for mode in backend_supported_modes)
        ):
            raise ValueError(f"backend reported invalid supported_modes: {backend_supported_modes!r}")
        with self._stats_lock:
            inference_count = self.inference_count
            failure_count = self.failure_count
            last_latency_s = self.last_latency_s
        metrics = self.metrics.snapshot().to_dict()
        trace = self.trace_recorder.stats().to_dict() if self.trace_recorder is not None else None
        health.update(
            {
                "ok": bool(health.get("ok", True)),
                "protocol_version": PROTOCOL_VERSION,
                "supported_modes": list(backend_supported_modes),
                "model_action_dim": MODEL_ACTION_DIM,
                "wire_action_dim": WIRE_ACTION_DIM,
                "optimized_runtime": True,
                "optimized_runtime_phase": max(backend_phase, processor_phase),
                "backend": self.backend.name,
                "trajectory_processor": self.trajectory_processor.name,
                "trajectory_processor_health": (
                    self.trajectory_processor.health()
                    if callable(getattr(self.trajectory_processor, "health", None))
                    else None
                ),
                "optimized_inference_count": inference_count,
                "optimized_failure_count": failure_count,
                "optimized_last_latency_s": last_latency_s,
                "optimized_metrics": metrics,
                "trace": trace,
                "timing_stage_availability": {
                    "optimized_total": True,
                    "lock_wait": True,
                    "backend_total": True,
                    "trajectory": True,
                    "response": True,
                    "preprocess": backend_stage_metrics is not None,
                    "model_only": backend_stage_metrics is not None,
                    "postprocess": backend_stage_metrics is not None,
                    "backend_reported_model_note": (
                        "backend model_latency_s remains end-to-end for safe delay accounting; "
                        "torch_optimized exposes separate diagnostic stage metrics"
                        if backend_stage_metrics is not None
                        else "trusted torch backend reports prepare+preprocess+predict+postprocess as one value"
                    ),
                },
            }
        )
        return health

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        started_s = self._clock()
        durations: dict[str, float] = {}
        try:
            request.validate()
            lock_wait_started_s = self._clock()
            with self._execution_lock:
                backend_started_s = self._clock()
                durations["lock_wait_s"] = _duration("lock_wait_s", lock_wait_started_s, backend_started_s)
                backend_response = self.backend.infer(request)
                backend_finished_s = self._clock()
                durations["backend_s"] = _duration("backend_s", backend_started_s, backend_finished_s)
                _validate_response_identity(request, backend_response)
                trajectory_started_s = self._clock()
                trajectory = PairedTrajectory.from_response(
                    backend_response,
                    source_observation_seq=request.obs_sequence_id,
                    predicted_delay_steps=request.predicted_delay_steps,
                )
                processed = self.trajectory_processor.process(trajectory)
                _validate_processed_identity(
                    trajectory,
                    processed,
                    allow_action_changes=self.trajectory_processor.allows_action_changes,
                )
                trajectory_finished_s = self._clock()
                durations["trajectory_s"] = _duration(
                    "trajectory_s", trajectory_started_s, trajectory_finished_s
                )

            response_started_s = self._clock()
            response = InferenceResponse(
                request_id=request.request_id,
                mode=request.mode,
                raw_actions=processed.model_actions.copy(),
                processed_actions=processed.robot_actions.copy(),
                server_latency_s=0.0,
                model_latency_s=backend_response.model_latency_s,
                raw_action_shape=tuple(processed.model_actions.shape),
                processed_action_shape=tuple(processed.robot_actions.shape),
                error=backend_response.error,
            )
            response.validate()
            response_finished_s = self._clock()
            durations["response_s"] = _duration("response_s", response_started_s, response_finished_s)
            total_s = _duration("total_s", started_s, response_finished_s)
            response.server_latency_s = total_s
            response.validate()
            timings = InferenceTimings(
                total_s=total_s,
                lock_wait_s=durations["lock_wait_s"],
                backend_s=durations["backend_s"],
                trajectory_s=durations["trajectory_s"],
                response_s=durations["response_s"],
                backend_reported_server_s=backend_response.server_latency_s,
                backend_reported_model_s=backend_response.model_latency_s,
            )
            if self.trace_recorder is not None:
                self.trace_recorder.record_inference(request, timings)
            self.metrics.record_success(timings)
            with self._stats_lock:
                self.inference_count += 1
                self.last_latency_s = timings.total_s
            return response
        except Exception as error:
            self.metrics.record_failure()
            with self._stats_lock:
                self.failure_count += 1
            try:
                failure_finished_s = self._clock()
                durations["total_s"] = _duration("total_s", started_s, failure_finished_s)
            except Exception as timing_error:
                _attach_diagnostic(error, f"failure duration could not be measured: {timing_error}")
            if self.trace_recorder is not None:
                try:
                    self.trace_recorder.record_failure(request, durations_s=durations, error=error)
                except Exception as trace_error:
                    _attach_diagnostic(error, f"failure trace could not be written: {trace_error}")
            raise


def _validate_response_identity(request: InferenceRequest, response: InferenceResponse) -> None:
    if not isinstance(response, InferenceResponse):
        raise TypeError(f"backend must return InferenceResponse, got {type(response)}")
    response.validate()
    _validate_latency("server_latency_s", response.server_latency_s)
    _validate_latency("model_latency_s", response.model_latency_s)
    if response.request_id != request.request_id:
        raise ValueError(
            f"backend response request_id mismatch: expected {request.request_id}, got {response.request_id}"
        )
    if response.mode != request.mode:
        raise ValueError(f"backend response mode mismatch: expected {request.mode!r}, got {response.mode!r}")
    if response.error is not None:
        raise RuntimeError(f"backend returned an inference error: {response.error}")


def _validate_processed_identity(
    before: PairedTrajectory,
    after: PairedTrajectory,
    *,
    allow_action_changes: bool,
) -> None:
    if not isinstance(after, PairedTrajectory):
        raise TypeError(f"trajectory processor must return PairedTrajectory, got {type(after)}")
    identity_fields = ("request_id", "mode", "source_observation_seq", "predicted_delay_steps")
    changed = [field for field in identity_fields if getattr(before, field) != getattr(after, field)]
    if changed:
        raise ValueError(f"trajectory processor changed immutable identity fields: {changed}")
    if allow_action_changes:
        return
    if not np.array_equal(before.model_actions, after.model_actions):
        raise ValueError("pass-through trajectory processor changed model16 action values")
    if not np.array_equal(before.robot_actions, after.robot_actions):
        raise ValueError("pass-through trajectory processor changed raw18 action values")


def _validate_latency(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"backend {name} must be a real number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"backend {name} must be finite and non-negative")


def _duration(name: str, started_s: object, finished_s: object) -> float:
    _validate_latency(f"{name} start", started_s)
    _validate_latency(f"{name} finish", finished_s)
    duration = float(finished_s) - float(started_s)
    if duration < 0:
        raise ValueError(f"monotonic clock moved backwards while measuring {name}")
    return duration


def _attach_diagnostic(error: BaseException, message: str) -> None:
    try:
        existing = tuple(getattr(error, "_pi05_optimized_diagnostics", ()))
        error._pi05_optimized_diagnostics = (*existing, message)  # type: ignore[attr-defined]
    except Exception:
        return


__all__ = ["OptimizedPolicyService"]
