#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
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
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/phase5_alignment_shadow_replay.json"
CAMERA_KEY = "observation.images.camera_head"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05_optimized.runtime.timed_observation import (  # noqa: E402
    SourceTimestamp,
    TimedObservation,
)
from tk_infer.pi05_optimized.runtime.timestamp_alignment import (  # noqa: E402
    TimestampAlignmentConfig,
    TimestampAlignmentShadow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic no-hardware Phase 5 timestamp alignment replay.")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--camera-delay-s", type=float, default=0.03)
    parser.add_argument("--readout-delay-s", type=float, default=0.005)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("duration_s", "rate_hz"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("camera_delay_s", "readout_delay_s"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    iterations = int(round(args.duration_s * args.rate_hz))
    if iterations < 2:
        raise ValueError("alignment replay requires at least two samples")
    period_s = 1.0 / args.rate_hz
    total_image_delay_s = args.camera_delay_s + args.readout_delay_s
    if total_image_delay_s >= period_s:
        raise ValueError("this deterministic replay requires total image delay below one state period")
    shadow = TimestampAlignmentShadow(
        TimestampAlignmentConfig(
            camera_keys=(CAMERA_KEY,),
            source_clock_domain="synthetic_sensor_clock",
            state_delay_s=0.0,
            camera_delay_s={CAMERA_KEY: args.camera_delay_s},
            readout_delay_s={CAMERA_KEY: args.readout_delay_s},
            max_skew_s=period_s,
            history_window_s=5.0,
            history_max_samples=512,
        )
    )
    base_time_s = 10.0
    interpolation_errors = []
    shadow_deltas = []
    max_history_samples = 0
    for sequence in range(iterations):
        source_time_s = base_time_s + sequence * period_s
        state = _linear_raw18(source_time_s)
        observation = TimedObservation(
            observation_frame={
                "observation.state": state,
                CAMERA_KEY: np.zeros((2, 2, 3), dtype=np.uint8),
            },
            sequence_id=sequence,
            receive_monotonic_s=source_time_s,
            build_started_monotonic_s=source_time_s,
            build_ready_monotonic_s=source_time_s,
            local_clock_domain="synthetic_process_clock",
            state_source_timestamp=SourceTimestamp(
                source_time_s,
                "synthetic_sensor_clock",
                "synthetic_raw18",
            ),
            camera_source_timestamps={
                CAMERA_KEY: SourceTimestamp(
                    source_time_s,
                    "synthetic_sensor_clock",
                    "synthetic_camera_publish",
                )
            },
        )
        results = shadow.observe(observation)
        max_history_samples = max(max_history_samples, len(shadow.history.snapshot()))
        for result in results:
            expected = _linear_raw18(source_time_s - total_image_delay_s)
            interpolation_errors.extend(
                np.abs(result.aligned_raw18.astype(np.float64) - expected.astype(np.float64)).tolist()
            )
            shadow_deltas.append(result.max_abs_delta)
            if result.changed_policy_input:
                raise AssertionError("alignment shadow changed policy input")
    snapshot = shadow.snapshot()
    errors = np.asarray(interpolation_errors, dtype=np.float64)
    deltas = np.asarray(shadow_deltas, dtype=np.float64)
    report = {
        "status": "PASS",
        "hardware_access": False,
        "network_access": False,
        "mode": "synthetic_shadow_replay",
        "duration_s": args.duration_s,
        "rate_hz": args.rate_hz,
        "iterations": iterations,
        "camera_delay_s": args.camera_delay_s,
        "readout_delay_s": args.readout_delay_s,
        "total_image_delay_s": total_image_delay_s,
        "source_clock_domain": "synthetic_sensor_clock",
        "changed_policy_input": False,
        "shadow_snapshot": snapshot,
        "max_history_samples": max_history_samples,
        "history_capacity": 512,
        "interpolation_error": _distribution(errors),
        "aligned_vs_current_max_abs_delta": _distribution(deltas),
        "live_motion_delay_identified": False,
        "live_motion_delay_status": "blocked_pending_explicit_on_site_authorization",
    }
    if snapshot["failure_count"] != 0:
        raise AssertionError(f"alignment replay recorded failures: {snapshot}")
    if max_history_samples > 512:
        raise AssertionError("alignment history exceeded its configured capacity")
    if report["interpolation_error"]["max"] > 1e-5:
        raise AssertionError(f"linear interpolation error exceeded tolerance: {report}")
    return report


def _linear_raw18(timestamp_s: float) -> np.ndarray:
    slopes = np.linspace(0.01, 0.18, 18, dtype=np.float64)
    offsets = np.linspace(-1.0, 1.0, 18, dtype=np.float64)
    return np.ascontiguousarray((offsets + slopes * timestamp_s).astype(np.float32))


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0, "max": 0.0, "mean": 0.0, "p99": 0.0}
    return {
        "count": int(values.size),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "p99": float(np.percentile(values, 99)),
    }


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-json must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_replay(args)
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
