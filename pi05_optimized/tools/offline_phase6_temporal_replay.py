#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
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
DEFAULT_POLICY_PATH = REPO_ROOT / "tk_infer/pi05/checkpoints/010600/pretrained_model"
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "assets/modelscope/google/paligemma-3b-pt-224"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/jz_robot_pin_timed_merged_100eps_20260728"
DEFAULT_DATASET_REPO_ID = "local/jz_robot_pin_timed_merged_100eps_20260728_pi05_head_right"
DEFAULT_TASK = "jz robot pin timed vr teleoperation"
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/phase6_temporal_replay.json"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from tk_infer.pi05.runtime.checkpoint import inspect_checkpoint  # noqa: E402
from tk_infer.pi05.runtime.policy_service import PolicyService, PolicyServiceConfig  # noqa: E402
from tk_infer.pi05_optimized.runtime.paired_trajectory import PairedTrajectory  # noqa: E402
from tk_infer.pi05_optimized.runtime.temporal_optimizer import (  # noqa: E402
    PairedTemporalTrajectoryProcessor,
    TemporalOptimizationConfig,
    optional_qp_dependency_status,
)
from tk_infer.pi05_optimized.tools.offline_reference_parity import (  # noqa: E402
    _configure_offline_environment,
    _load_observation,
    _make_request,
    _reset_reference_state,
    _synchronize,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-hardware Phase 6 paired temporal replay.")
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
    parser.add_argument("--speed-factor", type=float, default=1.0)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.02)
    parser.add_argument("--solver-timeout-s", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    config = TemporalOptimizationConfig(
        speed_factor=args.speed_factor,
        max_joint_step_rad=args.max_joint_step_rad,
        solver_timeout_s=args.solver_timeout_s,
    )
    if args.source == "synthetic":
        trajectory = _synthetic_trajectory()
        source_metadata: dict[str, object] = {
            "source": "deterministic_synthetic_chunk",
            "recorded_observation": False,
            "checkpoint_inference": False,
        }
    elif args.source == "recorded":
        trajectory, source_metadata = _recorded_trajectory(args)
    else:
        raise ValueError(f"unsupported replay source: {args.source!r}")
    return evaluate_trajectory(trajectory, config=config, source_metadata=source_metadata)


def evaluate_trajectory(
    trajectory: PairedTrajectory,
    *,
    config: TemporalOptimizationConfig,
    source_metadata: dict[str, object],
) -> dict[str, Any]:
    processor = PairedTemporalTrajectoryProcessor(config)
    started_s = time.perf_counter()
    output = processor.process(trajectory)
    replay_elapsed_s = time.perf_counter() - started_s
    report = processor.last_report
    if report is None:
        raise AssertionError("temporal processor did not retain its aggregate report")
    identity_preserved = (
        output.request_id == trajectory.request_id
        and output.mode == trajectory.mode
        and output.source_observation_seq == trajectory.source_observation_seq
        and output.predicted_delay_steps == trajectory.predicted_delay_steps
    )
    force_exact = bool(
        np.equal(output.robot_actions[:, 15], 80.0).all()
        and np.equal(output.robot_actions[:, 17], 80.0).all()
    )
    finite = bool(np.isfinite(output.model_actions).all() and np.isfinite(output.robot_actions).all())
    first_action_preserved = bool(
        np.array_equal(output.model_actions[0], trajectory.model_actions[0])
        and np.array_equal(output.robot_actions[0], trajectory.robot_actions[0])
    )
    gate_failures = []
    if output.model_actions.shape[0] != output.robot_actions.shape[0]:
        gate_failures.append("model16_raw18_temporal_length_mismatch")
    if report.output_max_joint_step_rad > config.max_joint_step_rad + 1e-7:
        gate_failures.append("joint_step_bound_exceeded")
    if not identity_preserved:
        gate_failures.append("request_identity_changed")
    if not force_exact:
        gate_failures.append("force_slots_not_exact_80")
    if not finite:
        gate_failures.append("non_finite_output")
    if not first_action_preserved:
        gate_failures.append("first_action_changed")
    result = {
        "status": "PASS" if not gate_failures else "FAIL",
        "hardware_access": False,
        "network_access": False,
        "action_transport_created": False,
        "payload_retained_in_report": False,
        "source_metadata": source_metadata,
        "processor": "paired_temporal",
        "processor_config": {
            "speed_factor": config.speed_factor,
            "max_joint_step_rad": config.max_joint_step_rad,
            "solver_timeout_s": config.solver_timeout_s,
            "acceleration_objective_enabled": False,
            "jerk_objective_enabled": False,
        },
        "input_shapes": {
            "model16": list(trajectory.model_actions.shape),
            "raw18": list(trajectory.robot_actions.shape),
        },
        "output_shapes": {
            "model16": list(output.model_actions.shape),
            "raw18": list(output.robot_actions.shape),
        },
        "single_interpolation_map_applied_to_pair": True,
        "model16_raw18_identity_preserved": identity_preserved,
        "first_action_preserved": first_action_preserved,
        "force_slots_exact_80": force_exact,
        "finite": finite,
        "initial_state_guard_evaluated": False,
        "initial_state_guard_owner": "existing JZ robot execution boundary",
        "temporal_report": report.to_dict(),
        "optional_qp_dependencies": optional_qp_dependency_status(),
        "replay_elapsed_s": replay_elapsed_s,
        "gate_failures": gate_failures,
    }
    if gate_failures:
        raise AssertionError(f"Phase 6 replay failed: {gate_failures}")
    return result


def _recorded_trajectory(args: argparse.Namespace) -> tuple[PairedTrajectory, dict[str, object]]:
    _configure_offline_environment()
    policy_path = args.policy_path.expanduser().resolve(strict=True)
    tokenizer_path = args.tokenizer_path.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    metadata, _ = inspect_checkpoint(
        policy_path,
        tokenizer_path=tokenizer_path,
        require_complete_step=args.require_complete_step,
    )
    observation = _load_observation(
        dataset_root=dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        sample_index=args.sample_index,
        camera_shapes=metadata.camera_shapes,
    )
    request = _make_request(
        request_id=6,
        mode="single_step",
        observation_frame=observation,
        task=args.task,
    )
    reference = PolicyService.from_config(
        PolicyServiceConfig(
            policy_path=policy_path,
            tokenizer_path=tokenizer_path,
            device=args.device,
            require_complete_step=args.require_complete_step,
        ),
        rtc_config=RTCConfig(enabled=False),
    )
    _reset_reference_state(reference, seed=args.seed)
    _synchronize(args.device)
    started_s = time.perf_counter()
    response = reference.infer(request)
    _synchronize(args.device)
    inference_elapsed_s = time.perf_counter() - started_s
    trajectory = PairedTrajectory.from_response(
        response,
        source_observation_seq=request.obs_sequence_id,
        predicted_delay_steps=request.predicted_delay_steps,
    )
    return trajectory, {
        "source": "recorded_dataset_observation_real_checkpoint_chunk",
        "recorded_observation": True,
        "checkpoint_inference": True,
        "policy_path": str(policy_path),
        "checkpoint_fingerprint": metadata.checkpoint_fingerprint,
        "checkpoint_step": metadata.checkpoint_step,
        "configured_steps": metadata.configured_steps,
        "complete_step": metadata.complete_step,
        "require_complete_step": args.require_complete_step,
        "dataset_root": str(dataset_root),
        "dataset_repo_id": args.dataset_repo_id,
        "sample_index": args.sample_index,
        "seed": args.seed,
        "device": args.device,
        "inference_elapsed_s": inference_elapsed_s,
    }


def _synthetic_trajectory() -> PairedTrajectory:
    steps = 50
    source = np.arange(steps, dtype=np.float32)
    model = np.zeros((steps, 16), dtype=np.float32)
    raw = np.zeros((steps, 18), dtype=np.float32)
    for joint_index in range(14):
        phase = source * (0.2 + joint_index * 0.01)
        raw[:, joint_index] = np.sin(phase) * (0.15 + joint_index * 0.005)
        model[:, joint_index] = np.cos(phase) * (1.0 + joint_index * 0.1)
    model[:, 14] = source / (steps - 1)
    model[:, 15] = 1.0 - model[:, 14]
    raw[:, 14] = model[:, 14] * 100.0
    raw[:, 15] = 80.0
    raw[:, 16] = model[:, 15] * 100.0
    raw[:, 17] = 80.0
    return PairedTrajectory(
        model_actions=model,
        robot_actions=raw,
        request_id=6,
        mode="single_step",
        source_observation_seq=0,
        predicted_delay_steps=0,
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
    report = run_replay(args)
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
