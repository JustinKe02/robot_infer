#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/phase7_tracker_replay.json"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05_optimized.runtime.local_tracker import (  # noqa: E402
    LocalActionTracker,
    LocalTrackerConfig,
)
from tk_infer.pi05_optimized.runtime.temporal_optimizer import (  # noqa: E402
    optional_qp_dependency_status,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic no-hardware Phase 7 tracker replay.")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.02)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    if not np.isfinite(args.duration_s) or args.duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(args.rate_hz) or args.rate_hz <= 0:
        raise ValueError("rate_hz must be finite and positive")
    iterations = int(round(args.duration_s * args.rate_hz))
    if iterations < 2:
        raise ValueError("tracker replay requires at least two iterations")
    config = LocalTrackerConfig(
        control_period_s=1.0 / args.rate_hz,
        max_joint_step_rad=args.max_joint_step_rad,
    )
    first = _run_once(config, iterations)
    second = _run_once(config, iterations)
    deterministic = first["action_sha256"] == second["action_sha256"]
    dependencies = optional_qp_dependency_status()
    qp_available = all(values["available"] for values in dependencies.values())
    gate_failures = []
    if not deterministic:
        gate_failures.append("tracker_replay_not_deterministic")
    if first["max_output_joint_step_rad"] > config.max_joint_step_rad + 1e-7:
        gate_failures.append("tracker_joint_step_bound_exceeded")
    if not first["force_slots_exact_80"]:
        gate_failures.append("force_slots_not_exact_80")
    if not first["finite"]:
        gate_failures.append("non_finite_tracker_output")
    if first["max_history_samples"] > config.history_max_samples:
        gate_failures.append("tracker_history_unbounded")
    report = {
        "status": "PASS" if not gate_failures else "FAIL",
        "hardware_access": False,
        "network_access": False,
        "action_transport_created": False,
        "duration_s": args.duration_s,
        "rate_hz": args.rate_hz,
        "iterations": iterations,
        "tracker_replay_passed": not gate_failures,
        "deterministic": deterministic,
        "first_run": first,
        "second_action_sha256": second["action_sha256"],
        "contact_innovation_role": "slowdown_only_not_safety",
        "mpc": {
            "evaluated": False,
            "status": "BLOCKED",
            "reason": (
                "no audited MPC solver is implemented"
                if qp_available
                else "exact pinned SciPy/OSQP dependencies are unavailable"
            ),
            "tracker_replay_prerequisite_passed": not gate_failures,
            "optional_qp_dependencies": dependencies,
            "silent_fallback_allowed": False,
        },
        "gate_failures": gate_failures,
    }
    if gate_failures:
        raise AssertionError(f"Phase 7 tracker replay failed: {gate_failures}")
    return report


def _run_once(config: LocalTrackerConfig, iterations: int) -> dict[str, object]:
    tracker = LocalActionTracker(config)
    observed = np.zeros(18, dtype=np.float32)
    previous_output = None
    actions = []
    max_step = 0.0
    max_history_samples = 0
    min_contact_slowdown = 1.0
    max_lag_innovation = 0.0
    for index in range(iterations):
        timestamp_s = 10.0 + index * config.control_period_s
        phase = index * config.control_period_s
        requested = np.zeros(18, dtype=np.float32)
        for joint_index in range(14):
            requested[joint_index] = np.sin(phase * (0.8 + joint_index * 0.02)) * 0.5
        requested[14] = 50.0 + 40.0 * np.sin(phase)
        requested[15] = 80.0
        requested[16] = 50.0 + 40.0 * np.cos(phase)
        requested[17] = 80.0
        contact_innovation = 3.0 if iterations // 3 <= index < iterations // 3 + 20 else 0.0
        output = tracker.track(
            requested_action=requested,
            observed_state=observed,
            timestamp_s=timestamp_s,
            contact_innovation=contact_innovation,
        )
        output_array = output.numpy()
        actions.append(output_array.copy())
        if previous_output is not None:
            max_step = max(
                max_step,
                float(np.max(np.abs(output_array[:14] - previous_output[:14]), initial=0.0)),
            )
        previous_output = output_array
        observed[:14] += 0.25 * (output_array[:14] - observed[:14])
        observed[14] = output_array[14]
        observed[16] = output_array[16]
        report = tracker.last_report
        assert report is not None
        min_contact_slowdown = min(min_contact_slowdown, report.contact_slowdown_factor)
        max_lag_innovation = max(max_lag_innovation, report.lag_max_abs_innovation)
        max_history_samples = max(max_history_samples, len(tracker.history.snapshot()))
    action_array = np.ascontiguousarray(np.stack(actions).astype(np.float32))
    return {
        "action_sha256": hashlib.sha256(action_array.tobytes()).hexdigest(),
        "max_output_joint_step_rad": max_step,
        "force_slots_exact_80": bool(
            np.equal(action_array[:, 15], 80.0).all()
            and np.equal(action_array[:, 17], 80.0).all()
        ),
        "finite": bool(np.isfinite(action_array).all()),
        "max_history_samples": max_history_samples,
        "history_capacity": config.history_max_samples,
        "min_contact_slowdown_factor": min_contact_slowdown,
        "max_lag_innovation": max_lag_innovation,
        "contact_used_as_safety": False,
        "tracker_health": tracker.health(),
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
