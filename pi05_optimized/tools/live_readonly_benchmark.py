#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np


def resolve_repo_root(script_path: Path) -> Path:
    for candidate in script_path.resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
OPTIMIZED_ROOT = REPO_ROOT / "tk_infer/pi05_optimized"
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/live_readonly_benchmark.json"
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05.runtime.remote_client import RemotePolicyClient  # noqa: E402
from tk_infer.pi05_optimized.runtime.live_readonly import (  # noqa: E402
    CAMERA_KEYS,
    LiveReadOnlyConfig,
    LiveReadOnlyObservationSource,
    RecordingActionSink,
)
from tk_infer.pi05_optimized.runtime.optimized_client import (  # noqa: E402
    OptimizedClient,
    OptimizedClientConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live read-only PI0.5 benchmark; records policy outputs locally and has no command transport."
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:18088")
    parser.add_argument("--orin-ip", default="192.168.1.81")
    parser.add_argument("--state-bind-ip", default="0.0.0.0")
    parser.add_argument("--state-port", type=int, default=39010)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--measure-requests", type=int, default=30)
    parser.add_argument("--control-hz", type=float, default=5.0)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--state-timeout-s", type=float, default=1.0)
    parser.add_argument("--camera-timeout-ms", type=int, default=5000)
    parser.add_argument("--max-camera-state-receive-skew-ms", type=float, default=250.0)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--task", default="jz robot pin timed vr teleoperation")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_benchmark(
    args: argparse.Namespace,
    *,
    source: LiveReadOnlyObservationSource | None = None,
    policy_client: Any = None,
    clock: Any = time.perf_counter,
    sleeper: Any = time.sleep,
) -> dict[str, Any]:
    _validate_args(args)
    selected_policy = policy_client or RemotePolicyClient(
        args.server_url,
        auth_token=os.getenv("JZ_PI05_OPT_SERVER_AUTH_TOKEN") or None,
        timeout_s=args.request_timeout_s,
    )
    health_before = selected_policy.health()
    _validate_health(health_before)
    selected_source = source or LiveReadOnlyObservationSource(
        LiveReadOnlyConfig(
            orin_ip=args.orin_ip,
            state_bind_ip=args.state_bind_ip,
            state_port=args.state_port,
            connect_timeout_s=args.connect_timeout_s,
            state_timeout_s=args.state_timeout_s,
            camera_timeout_ms=args.camera_timeout_ms,
            max_camera_state_receive_skew_ms=args.max_camera_state_receive_skew_ms,
        )
    )
    warmup_sink = RecordingActionSink()
    measured_sink = RecordingActionSink()
    warmup_results = ()
    measured_results = ()
    warmup_cycle_s: list[float] = []
    measured_cycle_s: list[float] = []
    cleanup_error = None
    selected_source.connect()
    try:
        if args.warmup_requests:
            warmup_client = _client(args, selected_source, selected_policy, warmup_sink, clock)
            warmup_results, warmup_cycle_s = _run_paced(
                warmup_client,
                count=args.warmup_requests,
                control_hz=args.control_hz,
                clock=clock,
                sleeper=sleeper,
            )
        measured_client = _client(args, selected_source, selected_policy, measured_sink, clock)
        measured_results, measured_cycle_s = _run_paced(
            measured_client,
            count=args.measure_requests,
            control_hz=args.control_hz,
            clock=clock,
            sleeper=sleeper,
        )
        measured_telemetry = measured_client.telemetry.snapshot().to_dict()
    finally:
        try:
            selected_source.disconnect()
        except BaseException as error:
            cleanup_error = f"{type(error).__name__}: {error}"
    if cleanup_error is not None:
        raise RuntimeError(f"live source did not shut down cleanly: {cleanup_error}")

    health_after = selected_policy.health()
    _validate_health(health_after)
    expected_delta = args.warmup_requests + args.measure_requests
    before_count = _server_inference_count(health_before)
    after_count = _server_inference_count(health_after)
    if after_count - before_count != expected_delta:
        raise RuntimeError(
            f"server inference count changed by {after_count - before_count}, expected {expected_delta}"
        )
    if measured_sink.count != args.measure_requests:
        raise RuntimeError(
            f"local recording sink captured {measured_sink.count} outputs, expected {args.measure_requests}"
        )
    source_diagnostics = selected_source.diagnostics
    return {
        "status": "PASS",
        "phase": "P4.5_live_read_only",
        "server_url": args.server_url,
        "mode": "single_step",
        "camera_profile": "head_right",
        "warmup_requests": args.warmup_requests,
        "measure_requests": args.measure_requests,
        "control_hz": args.control_hz,
        "checkpoint": _health_subset(health_after),
        "server_count": {
            "before": before_count,
            "after": after_count,
            "delta": after_count - before_count,
        },
        "warmup": {
            "cycle_active_s": _distribution(warmup_cycle_s),
            "results": [asdict(result) for result in warmup_results],
            "local_policy_output_records": warmup_sink.summary(),
        },
        "measured": {
            "cycle_active_s": _distribution(measured_cycle_s),
            "client_telemetry": measured_telemetry,
            "results": [asdict(result) for result in measured_results],
            "local_policy_output_records": measured_sink.summary(),
        },
        "source_diagnostics": source_diagnostics,
        "safety": {
            "live_sensor_access": True,
            "robot_created": False,
            "command_transport_created": False,
            "policy_outputs_recorded_locally": True,
            "action_sent": False,
            "armed_capability": False,
            "state_receiver_stopped": bool(source_diagnostics["receiver_stopped"]),
        },
        "errors": [],
    }


def _client(
    args: argparse.Namespace,
    source: Any,
    policy_client: Any,
    sink: RecordingActionSink,
    clock: Any,
) -> OptimizedClient:
    return OptimizedClient(
        config=OptimizedClientConfig(
            task=args.task,
            mode="single_step",
            control_hz=args.control_hz,
            execution_horizon=10,
            strict_source_timestamps=True,
            required_camera_keys=CAMERA_KEYS,
        ),
        observation_source=source,
        policy_client=policy_client,
        action_sink=sink,
        clock=clock,
    )


def _run_paced(
    client: OptimizedClient,
    *,
    count: int,
    control_hz: float,
    clock: Any,
    sleeper: Any,
) -> tuple[tuple[Any, ...], list[float]]:
    period_s = 1.0 / control_hz
    deadline_s = clock()
    results = []
    cycle_active_s = []
    for _ in range(count):
        remaining_s = deadline_s - clock()
        if remaining_s > 0:
            sleeper(remaining_s)
        started_s = clock()
        results.append(client.run_cycle())
        cycle_active_s.append(clock() - started_s)
        deadline_s += period_s
    return tuple(results), cycle_active_s


def _validate_args(args: argparse.Namespace) -> None:
    parsed = urlsplit(args.server_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("live read-only benchmark requires a loopback HTTP server URL")
    for name, allow_zero in (("warmup_requests", True), ("measure_requests", False)):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < int(not allow_zero)
            or value > 1000
        ):
            minimum = 0 if allow_zero else 1
            raise ValueError(f"{name} must be an integer in {minimum}..1000")
    if not 1.0 <= float(args.control_hz) <= 30.0:
        raise ValueError("control_hz must be in 1..30")
    for name in (
        "connect_timeout_s",
        "state_timeout_s",
        "max_camera_state_receive_skew_ms",
        "request_timeout_s",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if isinstance(args.camera_timeout_ms, bool) or args.camera_timeout_ms <= 0:
        raise ValueError("camera_timeout_ms must be positive")
    if not isinstance(args.task, str) or not args.task.strip():
        raise ValueError("task must be non-empty")


def _validate_health(health: dict[str, Any]) -> None:
    required = {
        "ok": True,
        "optimized_runtime": True,
        "camera_profile": "head_right",
        "checkpoint_step": 15900,
        "configured_steps": 15900,
        "complete_step": True,
        "model_state_dim": 16,
        "model_action_dim": 16,
        "wire_action_dim": 18,
    }
    mismatches = {
        name: {"expected": expected, "actual": health.get(name)}
        for name, expected in required.items()
        if health.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"optimized server health contract mismatch: {mismatches}")
    if "single_step" not in health.get("supported_modes", []):
        raise ValueError("optimized server does not support single_step")


def _health_subset(health: dict[str, Any]) -> dict[str, Any]:
    names = (
        "backend",
        "trajectory_processor",
        "checkpoint_path",
        "checkpoint_fingerprint",
        "checkpoint_step",
        "configured_steps",
        "complete_step",
        "optimized_runtime",
        "optimized_runtime_phase",
        "inference_count",
        "optimized_inference_count",
        "backend_inference_count",
        "optimized_failure_count",
        "trace",
    )
    return {name: health.get(name) for name in names}


def _server_inference_count(health: dict[str, Any]) -> int:
    value = health.get("optimized_inference_count", health.get("inference_count"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"optimized server exposed an invalid inference count: {value!r}")
    return value


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-json must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {resolved}")
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(args)
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
