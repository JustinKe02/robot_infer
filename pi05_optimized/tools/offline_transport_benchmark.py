#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/phase4_protocol_v3_transport_benchmark.json"
DEFAULT_PHASE2_REPORT = OPTIMIZED_ROOT / "outputs/phase2_torch_benchmark.json"
AUTH_TOKEN = "phase4-offline-loopback-benchmark"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05.runtime.http_server import make_server  # noqa: E402
from tk_infer.pi05.runtime.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    InferenceRequest,
    InferenceResponse,
    dumps_payload,
    loads_payload,
)
from tk_infer.pi05.runtime.remote_client import RemotePolicyClient  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline protocol v3 serialization and loopback benchmark.")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--phase2-report", type=Path, default=DEFAULT_PHASE2_REPORT)
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _validate_count("warmup", args.warmup, allow_zero=True)
    _validate_count("repetitions", args.repetitions, allow_zero=False)
    request = _make_request()
    response = _make_response(request)
    request_body = dumps_payload(request)
    response_body = dumps_payload(response)
    request_roundtrip = loads_payload(request_body)
    response_roundtrip = loads_payload(response_body)
    if not isinstance(request_roundtrip, InferenceRequest):
        raise TypeError("request payload did not round-trip as InferenceRequest")
    if not isinstance(response_roundtrip, InferenceResponse):
        raise TypeError("response payload did not round-trip as InferenceResponse")

    measurements = {
        "request_serialize_s": _measure(
            lambda: dumps_payload(request),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        "request_restricted_deserialize_s": _measure(
            lambda: loads_payload(request_body),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        "response_serialize_s": _measure(
            lambda: dumps_payload(response),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        "response_restricted_deserialize_s": _measure(
            lambda: loads_payload(response_body),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
    }
    fake_service = CapturedResponseService(response)
    before_threads = {thread.ident for thread in threading.enumerate() if thread.ident is not None}
    with _running_server(fake_service) as server_url:
        client = RemotePolicyClient(server_url, auth_token=AUTH_TOKEN, timeout_s=5.0)
        health = client.health()
        loopback = _measure(
            lambda: client.infer(request),
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
    after_threads = {
        thread.ident
        for thread in threading.enumerate()
        if thread.ident is not None and thread.ident not in before_threads
    }
    if after_threads:
        raise RuntimeError(f"loopback benchmark leaked threads: {sorted(after_threads)}")
    measurements["authenticated_http_loopback_s"] = loopback
    measurements["fake_service_s"] = _measure(
        lambda: fake_service.infer(request),
        warmup=args.warmup,
        repetitions=args.repetitions,
    )
    phase2_context = _phase2_context(args.phase2_report)
    model_reference_p95 = phase2_context.get("minimum_reference_model_p95_s")
    transport_fraction = None
    if isinstance(model_reference_p95, int | float) and model_reference_p95 > 0:
        transport_fraction = loopback["p95"] / model_reference_p95
    material_bottleneck = transport_fraction is not None and transport_fraction >= 0.20
    return {
        "status": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "hardware_access": False,
        "external_network_access": False,
        "loopback_socket_access": True,
        "authenticated": health.get("optimized_runtime") is True,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "request_bytes": len(request_body),
        "response_bytes": len(response_body),
        "request_image_encoding": "none; protocol v3 serializes fixed NumPy uint8 arrays",
        "measurements": measurements,
        "phase2_context": phase2_context,
        "loopback_p95_fraction_of_min_reference_model_p95": transport_fraction,
        "material_transport_bottleneck_threshold": 0.20,
        "material_transport_bottleneck_observed": material_bottleneck,
        "protocol_v4_recommendation": (
            "evaluate_v4"
            if material_bottleneck
            else "retain_v3; model execution dominates and v4 is not justified by this host"
        ),
        "leaked_threads": [],
    }


class CapturedResponseService:
    def __init__(self, response: InferenceResponse) -> None:
        self.response = response

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "optimized_runtime": True,
            "backend": "captured_transport_benchmark",
        }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if request.request_id != self.response.request_id or request.mode != self.response.mode:
            raise ValueError("benchmark request identity differs from captured response")
        return self.response


@contextmanager
def _running_server(service: CapturedResponseService) -> Iterator[str]:
    server = make_server(
        host="127.0.0.1",
        port=0,
        service=service,  # type: ignore[arg-type]
        auth_token=AUTH_TOKEN,
    )
    thread = threading.Thread(target=server.serve_forever, name="phase4-loopback", daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("loopback server thread did not stop")


def _make_request() -> InferenceRequest:
    return InferenceRequest(
        request_id=1,
        mode="rtc",
        observation_frame={
            "observation.state": np.zeros(18, dtype=np.float32),
            "observation.images.camera_head": np.zeros((720, 1280, 3), dtype=np.uint8),
            "observation.images.camera_right": np.zeros((480, 640, 3), dtype=np.uint8),
        },
        task="jz robot pin timed vr teleoperation",
        robot_type="jz_robot_pin_timed",
        obs_sequence_id=1,
        predicted_delay_steps=1,
        prev_chunk_left_over=np.zeros((40, 16), dtype=np.float32),
        execution_horizon=10,
    )


def _make_response(request: InferenceRequest) -> InferenceResponse:
    model = np.zeros((50, 16), dtype=np.float32)
    robot = np.zeros((50, 18), dtype=np.float32)
    robot[:, 15] = 80
    robot[:, 17] = 80
    return InferenceResponse(
        request_id=request.request_id,
        mode=request.mode,
        raw_actions=model,
        processed_actions=robot,
        server_latency_s=0.0,
        model_latency_s=0.0,
        raw_action_shape=model.shape,
        processed_action_shape=robot.shape,
    )


def _measure(
    operation: Callable[[], Any],
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        operation()
    samples: list[float] = []
    for _ in range(repetitions):
        started_s = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started_s)
    array = np.asarray(samples, dtype=np.float64)
    return {
        "count": repetitions,
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _phase2_context(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"report": str(resolved), "available": False}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    reference_p95 = []
    for mode in ("single_step", "rtc"):
        try:
            reference_p95.append(float(payload["latency_summary"][mode]["reference"]["end_to_end_s"]["p95"]))
        except (KeyError, TypeError, ValueError):
            continue
    return {
        "report": str(resolved),
        "available": True,
        "checkpoint_fingerprint": payload.get("checkpoint_fingerprint"),
        "reference_model_p95_s": reference_p95,
        "minimum_reference_model_p95_s": min(reference_p95) if reference_p95 else None,
    }


def _validate_count(name: str, value: object, *, allow_zero: bool) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-json must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
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
