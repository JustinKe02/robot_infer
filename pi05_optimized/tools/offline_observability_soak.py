#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
OPTIMIZED_ROOT = REPO_ROOT / "tk_infer/pi05_optimized"
DEFAULT_TRACE_PATH = OPTIMIZED_ROOT / "logs/soak/observability_trace.jsonl"
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/phase1_observability_soak.json"
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05.runtime.protocol import InferenceRequest, InferenceResponse  # noqa: E402
from tk_infer.pi05_optimized.runtime.client_telemetry import ClientTelemetry  # noqa: E402
from tk_infer.pi05_optimized.runtime.metrics import InferenceMetrics  # noqa: E402
from tk_infer.pi05_optimized.runtime.policy_service import OptimizedPolicyService  # noqa: E402
from tk_infer.pi05_optimized.runtime.trace import JsonlTraceWriter  # noqa: E402


class _SoakBackend:
    @property
    def name(self) -> str:
        return "offline_soak_fake"

    def health(self) -> dict[str, Any]:
        return {"ok": True, "checkpoint_fingerprint": "offline-soak-fake"}

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        model_actions = np.full((50, 16), request.request_id % 17, dtype=np.float32)
        robot_actions = np.zeros((50, 18), dtype=np.float32)
        robot_actions[:, :14] = model_actions[:, :14]
        robot_actions[:, 14] = model_actions[:, 14]
        robot_actions[:, 15] = 80.0
        robot_actions[:, 16] = model_actions[:, 15]
        robot_actions[:, 17] = 80.0
        return InferenceResponse(
            request_id=request.request_id,
            mode=request.mode,
            raw_actions=model_actions,
            processed_actions=robot_actions,
            server_latency_s=0.001,
            model_latency_s=0.0005,
            raw_action_shape=model_actions.shape,
            processed_action_shape=robot_actions.shape,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-hardware observability soak for the optimized service.")
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument(
        "--iterations",
        type=int,
        help="Run a fixed number of iterations without sleeping; intended for accelerated smoke checks.",
    )
    parser.add_argument("--metrics-window-size", type=int, default=512)
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--trace-max-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--trace-backup-count", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_soak(
    *,
    duration_s: float,
    rate_hz: float,
    iterations: int | None,
    metrics_window_size: int,
    trace_path: Path,
    trace_max_bytes: int,
    trace_backup_count: int,
) -> dict[str, Any]:
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be finite and positive")
    if iterations is not None and (isinstance(iterations, bool) or iterations <= 0):
        raise ValueError("iterations must be a positive integer when provided")

    trace_writer = JsonlTraceWriter(
        trace_path,
        max_bytes=trace_max_bytes,
        backup_count=trace_backup_count,
    )
    service = OptimizedPolicyService(
        backend=_SoakBackend(),
        metrics=InferenceMetrics(window_size=metrics_window_size),
        trace_recorder=trace_writer,
    )
    client = ClientTelemetry(window_size=metrics_window_size)
    initial_threads = {thread.ident for thread in threading.enumerate()}
    period_s = 1.0 / rate_hz
    completed = 0
    started_s = time.perf_counter()
    next_tick_s = started_s
    tracemalloc.start()
    try:
        while True:
            now_s = time.perf_counter()
            if iterations is not None:
                if completed >= iterations:
                    break
            elif now_s - started_s >= duration_s:
                break

            mode = "rtc" if completed % 2 else "single_step"
            request = _request(request_id=completed, mode=mode)
            request_started_s = time.perf_counter()
            response = service.infer(request)
            request_total_s = time.perf_counter() - request_started_s
            client.record_request(request, response, total_s=request_total_s)
            client.record_queue(depth=49, dropped_steps=0, stale_chunk=False)
            client.record_frame(observation_sequence_id=completed, source_frame_id=completed)
            logical_tick_s = completed * period_s
            client.record_sensor_tick(timestamp_s=logical_tick_s, target_period_s=period_s)
            client.record_actor_tick(timestamp_s=logical_tick_s, target_period_s=period_s)
            completed += 1

            if iterations is None:
                next_tick_s += period_s
                sleep_s = next_tick_s - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick_s = time.perf_counter()
    finally:
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    elapsed_s = time.perf_counter() - started_s
    final_threads = {thread.ident for thread in threading.enumerate()}
    leaked_threads = sorted(
        thread_id for thread_id in final_threads - initial_threads if thread_id is not None
    )
    service_health = service.health()
    client_snapshot = client.snapshot().to_dict()
    if service_health["optimized_failure_count"] != 0:
        raise RuntimeError(f"soak observed service failures: {service_health['optimized_failure_count']}")
    if trace_writer.stats().dropped_events != 0:
        raise RuntimeError(f"soak dropped trace events: {trace_writer.stats().to_dict()}")
    if leaked_threads:
        raise RuntimeError(f"soak leaked threads: {leaked_threads}")
    return {
        "status": "PASS",
        "hardware_access": False,
        "network_access": False,
        "mode": "accelerated" if iterations is not None else "wall_clock",
        "configured_duration_s": duration_s,
        "configured_rate_hz": rate_hz,
        "configured_iterations": iterations,
        "completed_iterations": completed,
        "elapsed_s": elapsed_s,
        "service_metrics": service_health["optimized_metrics"],
        "client_metrics": client_snapshot,
        "trace": trace_writer.stats().to_dict(),
        "tracemalloc_current_bytes": current_bytes,
        "tracemalloc_peak_bytes": peak_bytes,
        "leaked_threads": leaked_threads,
    }


def _request(*, request_id: int, mode: str) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        mode=mode,  # type: ignore[arg-type]
        observation_frame={"observation.state": np.zeros(18, dtype=np.float32)},
        task="offline observability soak",
        robot_type="jz_robot_pin_timed",
        obs_sequence_id=request_id,
        predicted_delay_steps=1 if mode == "rtc" else 0,
        prev_chunk_left_over=np.zeros((10, 16), dtype=np.float32) if mode == "rtc" else None,
        execution_horizon=10,
    )


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-json must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_soak(
        duration_s=args.duration_s,
        rate_hz=args.rate_hz,
        iterations=args.iterations,
        metrics_window_size=args.metrics_window_size,
        trace_path=args.trace_path,
        trace_max_bytes=args.trace_max_bytes,
        trace_backup_count=args.trace_backup_count,
    )
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
