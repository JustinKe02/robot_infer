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
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/phase9_fixed_speed_profiles.json"
PHASE3_REPORT = OPTIMIZED_ROOT / "outputs/phase3_triton_benchmark.json"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05_optimized.runtime.speed_profile_study import (  # noqa: E402
    DEFAULT_MIN_TRIALS_PER_PROFILE,
    FIXED_SPEED_PROFILES,
    evaluate_labeled_speed_trials,
    trial_from_dict,
)
from tk_infer.pi05_optimized.runtime.temporal_optimizer import (  # noqa: E402
    PairedTemporalTrajectoryProcessor,
    TemporalOptimizationConfig,
)
from tk_infer.pi05_optimized.tools.offline_phase6_temporal_replay import (  # noqa: E402
    DEFAULT_DATASET_REPO_ID,
    DEFAULT_DATASET_ROOT,
    DEFAULT_POLICY_PATH,
    DEFAULT_TASK,
    DEFAULT_TOKENIZER_PATH,
    _recorded_trajectory,
    _synthetic_trajectory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-hardware Phase 9 fixed speed-profile study.")
    parser.add_argument("--source", choices=["synthetic", "recorded"], default="synthetic")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--require-complete-step", action="store_true")
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.02)
    parser.add_argument("--solver-timeout-s", type=float, default=0.05)
    parser.add_argument("--outcomes-json", type=Path)
    parser.add_argument("--min-trials-per-profile", type=int, default=DEFAULT_MIN_TRIALS_PER_PROFILE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    if not np.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("control_hz must be finite and positive")
    if args.source == "recorded":
        trajectory, source_metadata = _recorded_trajectory(args)
    elif args.source == "synthetic":
        trajectory = _synthetic_trajectory()
        source_metadata = {
            "source": "deterministic_synthetic_chunk",
            "recorded_observation": False,
            "checkpoint_inference": False,
        }
    else:
        raise ValueError(f"unsupported source: {args.source!r}")
    scheduler_profiles = []
    scheduler_gate_failures = []
    control_period_s = 1.0 / args.control_hz
    for speed_factor in FIXED_SPEED_PROFILES:
        processor = PairedTemporalTrajectoryProcessor(
            TemporalOptimizationConfig(
                speed_factor=speed_factor,
                max_joint_step_rad=args.max_joint_step_rad,
                solver_timeout_s=args.solver_timeout_s,
            )
        )
        output = processor.process(trajectory)
        temporal_report = processor.last_report
        interpolation_map = processor.last_interpolation_map
        if temporal_report is None or interpolation_map is None:
            raise AssertionError("temporal processor did not retain report/map")
        completion_indices = np.flatnonzero(
            interpolation_map.source_positions >= trajectory.steps - 1 - 1e-12
        )
        first_completion_index = int(completion_indices[0]) if len(completion_indices) else None
        force_exact = bool(
            np.equal(output.robot_actions[:, 15], 80.0).all()
            and np.equal(output.robot_actions[:, 17], 80.0).all()
        )
        finite = bool(np.isfinite(output.model_actions).all() and np.isfinite(output.robot_actions).all())
        profile_failures = []
        if temporal_report.output_max_joint_step_rad > args.max_joint_step_rad + 1e-7:
            profile_failures.append("joint_step_bound_exceeded")
        if not force_exact:
            profile_failures.append("force_slots_not_exact_80")
        if not finite:
            profile_failures.append("non_finite_output")
        scheduler_gate_failures.extend(f"{speed_factor:g}x:{reason}" for reason in profile_failures)
        scheduler_profiles.append(
            {
                "speed_factor": speed_factor,
                "model16_shape": list(output.model_actions.shape),
                "raw18_shape": list(output.robot_actions.shape),
                "scheduler_reached_source_end": first_completion_index is not None,
                "first_source_end_step": first_completion_index,
                "scheduler_time_to_source_end_s": (
                    None if first_completion_index is None else first_completion_index * control_period_s
                ),
                "nominal_unconstrained_time_to_source_end_s": (
                    (trajectory.steps - 1) * control_period_s / speed_factor
                ),
                "full_output_horizon_s": trajectory.steps * control_period_s,
                "task_success_rate": None,
                "physical_cycle_time_s": None,
                "force_slots_exact_80": force_exact,
                "finite": finite,
                "temporal_report": temporal_report.to_dict(),
                "gate_failures": profile_failures,
            }
        )

    labeled_study = _evaluate_outcomes(args)
    phase3 = _read_json_object(PHASE3_REPORT)
    triton_p95_s = phase3.get("latency_s", {}).get("triton", {}).get("p95")
    realtime_gate = isinstance(triton_p95_s, int | float) and triton_p95_s <= 0.05
    return {
        "phase": 9,
        "status": "PASS" if not scheduler_gate_failures else "FAIL",
        "research_track": "fixed_speed_profiles_before_learned_adaptation",
        "learned_speed_adaptation_enabled": False,
        "hardware_access": False,
        "network_access": False,
        "action_transport_created": False,
        "fixed_profiles": list(FIXED_SPEED_PROFILES),
        "source_metadata": source_metadata,
        "scheduler_profiles": scheduler_profiles,
        "scheduler_gate_failures": scheduler_gate_failures,
        "labeled_task_curves": labeled_study,
        "task_curve_status": labeled_study["status"],
        "task_curve_note": (
            "scheduler completion is not task success; task curves require labeled fixed-profile trials"
        ),
        "performance_context": {
            "phase3_triton_model_p95_s": triton_p95_s,
            "full_pipeline_20hz_p95_gate_s": 0.05,
            "model_only_gate_met": realtime_gate,
            "may_be_described_as_20hz_realtime": False,
        },
        "robot_throttle_rollout": {
            "status": "BLOCKED_PENDING_SEPARATE_ON_SITE_AUTHORIZATION",
            "collected": False,
            "labels_created": False,
        },
    }


def _evaluate_outcomes(args: argparse.Namespace) -> dict[str, object]:
    if args.outcomes_json is None:
        result = evaluate_labeled_speed_trials(
            [], min_trials_per_profile=args.min_trials_per_profile
        )
        return result.to_dict()
    path = args.outcomes_json.expanduser().resolve(strict=True)
    value = _read_json_object(path)
    if set(value) != {"schema_version", "trials"} or value["schema_version"] != 1:
        raise ValueError("outcomes JSON must contain schema_version=1 and trials")
    if not isinstance(value["trials"], list):
        raise ValueError("outcomes trials must be a list")
    trials = [trial_from_dict(trial) for trial in value["trials"]]
    return evaluate_labeled_speed_trials(
        trials,
        min_trials_per_profile=args.min_trials_per_profile,
    ).to_dict()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-json must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_study(args)
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
