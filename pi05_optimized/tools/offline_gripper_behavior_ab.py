#!/usr/bin/env python

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
OPTIMIZED_ROOT = REPO_ROOT / "tk_infer/pi05_optimized"
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "assets/modelscope/google/paligemma-3b-pt-224"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/jz_robot_pin_timed_merged_100eps_20260728"
DEFAULT_DATASET_REPO_ID = "local/jz_robot_pin_timed_merged_100eps_20260728_pi05_head_right"
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/gripper_behavior_ab.json"
DEFAULT_CHECKPOINTS = (
    ("step_010600", REPO_ROOT / "tk_infer/pi05/checkpoints/010600/pretrained_model"),
    ("step_015900", REPO_ROOT / "tk_infer/pi05/checkpoints/015900/pretrained_model"),
)
DEFAULT_PROMPTS = (
    ("generic_runtime", "jz robot pin timed vr teleoperation"),
    ("bottle_basket_training", "Put the bottle on the right into the basket on the left."),
    ("plate_evaluation", "Pick up the plate."),
)
DEFAULT_PROMPT_SOURCES = {
    "generic_runtime": "audited_runtime_default",
    "bottle_basket_training": "training_default",
    "plate_evaluation": "synthetic_evaluation_no_historical_prompt_found",
}
DEFAULT_SEEDS = (12345, 12346, 12347)
BACKEND_NAMES = ("reference", "torch_optimized")
RIGHT_GRIPPER_INDEX = 16
LEFT_FORCE_INDEX = 15
RIGHT_FORCE_INDEX = 17
COMMAND_FORCE = 80.0

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from lerobot.configs.types import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from tk_infer.pi05.runtime.checkpoint import CheckpointMetadata, inspect_checkpoint  # noqa: E402
from tk_infer.pi05.runtime.policy_service import PolicyService, PolicyServiceConfig  # noqa: E402
from tk_infer.pi05.runtime.protocol import InferenceResponse  # noqa: E402
from tk_infer.pi05_optimized.backends.torch_backend import TorchPolicyBackend  # noqa: E402
from tk_infer.pi05_optimized.backends.torch_optimized_backend import (  # noqa: E402
    TorchBackendOptions,
    TorchOptimizedBackend,
)
from tk_infer.pi05_optimized.tools.offline_phase2_benchmark import (  # noqa: E402
    _compare_actions,
    _configure_offline_environment,
    _load_observation,
    _make_request,
    _reset_components,
    _synchronize,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline deterministic PI0.5 checkpoint/backend/prompt matrix with right-gripper behavior "
            "diagnostics. This tool does not create sockets or robot adapters."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Repeat to override the default step_010600 and step_015900 checkpoint matrix.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        metavar="LABEL=TEXT",
        help="Repeat to override the default generic, training, and synthetic plate prompt matrix.",
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--open-threshold", type=float, default=45.0)
    parser.add_argument("--closed-threshold", type=float, default=55.0)
    parser.add_argument("--require-complete-step", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_behavior_ab(args: argparse.Namespace) -> dict[str, Any]:
    _configure_offline_environment()
    checkpoints = _resolve_checkpoints(args.checkpoint)
    prompts = _resolve_prompts(args.prompt)
    seeds = _parse_seeds(args.seeds)
    _validate_thresholds(args.open_threshold, args.closed_threshold)
    tokenizer_path = args.tokenizer_path.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)

    metadata_by_label: dict[str, CheckpointMetadata] = {}
    for label, policy_path in checkpoints:
        metadata, _ = inspect_checkpoint(
            policy_path,
            tokenizer_path=tokenizer_path,
            require_complete_step=args.require_complete_step,
        )
        metadata_by_label[label] = metadata
    camera_shapes = _require_common_camera_shapes(metadata_by_label)
    observation = _load_observation(
        dataset_root=dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        sample_index=args.sample_index,
        camera_shapes=camera_shapes,
    )
    observation_sha256 = _observation_sha256(observation)

    cases: list[dict[str, Any]] = []
    request_id = 1
    for checkpoint_label, policy_path in checkpoints:
        metadata = metadata_by_label[checkpoint_label]
        service = PolicyService.from_config(
            PolicyServiceConfig(
                policy_path=policy_path,
                tokenizer_path=tokenizer_path,
                device=args.device,
                require_complete_step=args.require_complete_step,
            ),
            rtc_config=RTCConfig(
                enabled=True,
                prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
                max_guidance_weight=10.0,
                execution_horizon=10,
            ),
        )
        backends = {
            "reference": TorchPolicyBackend(service),
            "torch_optimized": TorchOptimizedBackend(service, options=TorchBackendOptions()),
        }
        for prompt_label, task in prompts:
            for seed in seeds:
                request = _make_request(
                    request_id=request_id,
                    mode="single_step",
                    observation_frame=observation,
                    task=task,
                )
                responses: dict[str, InferenceResponse] = {}
                for backend_name in BACKEND_NAMES:
                    _reset_components(service, seed=seed)
                    _synchronize(args.device)
                    responses[backend_name] = backends[backend_name].infer(request)
                    _synchronize(args.device)
                correctness = _compare_actions(
                    responses["reference"],
                    responses["torch_optimized"],
                    reduced_precision=False,
                )
                cases.append(
                    {
                        "checkpoint": checkpoint_label,
                        "checkpoint_step": metadata.checkpoint_step,
                        "prompt": prompt_label,
                        "task": task,
                        "seed": seed,
                        "request_id": request_id,
                        "backend_correctness": correctness,
                        "backends": {
                            backend_name: analyze_right_gripper(
                                responses[backend_name].processed_actions,
                                open_threshold=args.open_threshold,
                                closed_threshold=args.closed_threshold,
                            )
                            for backend_name in BACKEND_NAMES
                        },
                    }
                )
                request_id += 1
        del backends
        del service
        gc.collect()
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    failures = [
        f"{case['checkpoint']}/{case['prompt']}/{case['seed']}"
        for case in cases
        if not case["backend_correctness"]["gate_passed"]
    ]
    return {
        "status": "PASS" if not failures else "FAIL",
        "hardware_access": False,
        "network_access": False,
        "mode": "single_step_full_chunk",
        "comparison_contract": (
            "same in-process PolicyService, identical recorded observation, task, seed, and request; "
            "component state reset before each true backend forward"
        ),
        "gripper_semantics": {
            "processed_raw18_index": RIGHT_GRIPPER_INDEX,
            "raw_open": 0.0,
            "raw_closed": 100.0,
            "open_threshold_inclusive": args.open_threshold,
            "closed_threshold_inclusive": args.closed_threshold,
            "between_thresholds": "retain the last resolved state; unresolved before the first boundary",
            "threshold_source": "analysis-only hysteresis convention; not an execution control",
        },
        "observation": {
            "dataset_root": str(dataset_root),
            "dataset_repo_id": args.dataset_repo_id,
            "sample_index": args.sample_index,
            "sha256": observation_sha256,
        },
        "device": args.device,
        "seeds": list(seeds),
        "prompts": [
            {
                "label": label,
                "task": task,
                "source": DEFAULT_PROMPT_SOURCES.get(label, "cli_override"),
            }
            for label, task in prompts
        ],
        "checkpoints": [
            {
                "label": label,
                "path": str(path),
                "checkpoint_step": metadata_by_label[label].checkpoint_step,
                "configured_steps": metadata_by_label[label].configured_steps,
                "complete_step": metadata_by_label[label].complete_step,
                "fingerprint": metadata_by_label[label].checkpoint_fingerprint,
            }
            for label, path in checkpoints
        ],
        "backend_gate_failures": failures,
        "cases": cases,
        "independent_chunk_diagnostics": _independent_chunk_diagnostics(cases),
        "cross_checkpoint_diagnostics": _cross_checkpoint_diagnostics(cases),
    }


def analyze_right_gripper(
    processed_actions: object,
    *,
    open_threshold: float,
    closed_threshold: float,
) -> dict[str, Any]:
    _validate_thresholds(open_threshold, closed_threshold)
    actions = np.asarray(processed_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 18 or not len(actions):
        raise ValueError(f"processed_actions must have non-empty shape (T,18), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("processed_actions contain non-finite values")
    values = actions[:, RIGHT_GRIPPER_INDEX]
    below_range = values < 0.0
    above_range = values > 100.0

    current: str | None = None
    resolved_states: list[str] = []
    transitions: list[dict[str, Any]] = []
    for step, value in enumerate(values):
        candidate = _threshold_state(
            float(value),
            open_threshold=open_threshold,
            closed_threshold=closed_threshold,
        )
        if candidate is None or candidate == current:
            continue
        previous = current
        current = candidate
        resolved_states.append(candidate)
        transitions.append(
            {
                "step": step,
                "from": previous if previous is not None else "unresolved",
                "to": candidate,
                "value": float(value),
            }
        )

    closure_entries = sum(transition["to"] == "closed" for transition in transitions)
    open_to_closed = sum(
        transition["from"] == "open" and transition["to"] == "closed"
        for transition in transitions
    )
    reopen_after_close = sum(
        transition["from"] == "closed" and transition["to"] == "open"
        for transition in transitions
    )
    return {
        "steps": len(values),
        "values": [float(value) for value in values],
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "execution_range_valid": bool(not np.any(below_range | above_range)),
        "below_range_steps": np.flatnonzero(below_range).tolist(),
        "above_range_steps": np.flatnonzero(above_range).tolist(),
        "maximum_range_violation": float(
            np.maximum(np.maximum(-values, 0.0), np.maximum(values - 100.0, 0.0)).max(initial=0.0)
        ),
        "initial_resolved_state": resolved_states[0] if resolved_states else "unresolved",
        "final_resolved_state": resolved_states[-1] if resolved_states else "unresolved",
        "state_sequence": resolved_states,
        "transitions": transitions,
        "closure_entries": closure_entries,
        "open_to_closed_transitions": open_to_closed,
        "reopen_after_close_transitions": reopen_after_close,
        "reclose_entries": max(closure_entries - 1, 0),
        "force_slots_exact_80": bool(
            np.equal(actions[:, LEFT_FORCE_INDEX], COMMAND_FORCE).all()
            and np.equal(actions[:, RIGHT_FORCE_INDEX], COMMAND_FORCE).all()
        ),
    }


def _independent_chunk_diagnostics(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for case in cases:
        for backend_name, stats in case["backends"].items():
            grouped.setdefault((case["checkpoint"], case["prompt"], backend_name), []).append(
                {"seed": case["seed"], **stats}
            )
    reports: list[dict[str, Any]] = []
    for (checkpoint, prompt, backend), chunks in grouped.items():
        boundary_flips: list[dict[str, Any]] = []
        for previous, current in zip(chunks, chunks[1:], strict=False):
            previous_state = previous["final_resolved_state"]
            current_state = current["initial_resolved_state"]
            if "unresolved" not in {previous_state, current_state} and previous_state != current_state:
                boundary_flips.append(
                    {
                        "previous_seed": previous["seed"],
                        "current_seed": current["seed"],
                        "from": previous_state,
                        "to": current_state,
                    }
                )
        reports.append(
            {
                "checkpoint": checkpoint,
                "prompt": prompt,
                "backend": backend,
                "ordered_seeds": [chunk["seed"] for chunk in chunks],
                "boundary_flip_count": len(boundary_flips),
                "boundary_flips": boundary_flips,
                "chunks_with_reclose": sum(chunk["reclose_entries"] > 0 for chunk in chunks),
                "total_closure_entries": sum(chunk["closure_entries"] for chunk in chunks),
                "interpretation": (
                    "Seeds are independent resamples of one fixed observation, not a temporal rollout."
                ),
            }
        )
    return reports


def _cross_checkpoint_diagnostics(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault((case["prompt"], case["seed"], "reference"), []).append(case)
    reports: list[dict[str, Any]] = []
    for (prompt, seed, backend), matched_cases in grouped.items():
        if len(matched_cases) < 2:
            continue
        ordered = sorted(matched_cases, key=lambda case: case["checkpoint"])
        for left, right in zip(ordered, ordered[1:], strict=False):
            left_stats = left["backends"][backend]
            right_stats = right["backends"][backend]
            difference = np.abs(
                np.asarray(left_stats["values"], dtype=np.float64)
                - np.asarray(right_stats["values"], dtype=np.float64)
            )
            reports.append(
                {
                    "prompt": prompt,
                    "seed": seed,
                    "backend": backend,
                    "left_checkpoint": left["checkpoint"],
                    "right_checkpoint": right["checkpoint"],
                    "state_sequence_equal": left_stats["state_sequence"] == right_stats["state_sequence"],
                    "right_gripper_max_abs": float(difference.max(initial=0.0)),
                    "right_gripper_mean_abs": float(difference.mean()) if difference.size else 0.0,
                }
            )
    return reports


def _threshold_state(
    value: float,
    *,
    open_threshold: float,
    closed_threshold: float,
) -> str | None:
    if value <= open_threshold:
        return "open"
    if value >= closed_threshold:
        return "closed"
    return None


def _resolve_checkpoints(values: list[str] | None) -> tuple[tuple[str, Path], ...]:
    if values is None:
        entries = DEFAULT_CHECKPOINTS
    else:
        entries = tuple((label, Path(value)) for label, value in _parse_named_values(values, "checkpoint"))
    resolved: list[tuple[str, Path]] = []
    for label, path in entries:
        resolved.append((label, path.expanduser().resolve(strict=True)))
    return tuple(resolved)


def _resolve_prompts(values: list[str] | None) -> tuple[tuple[str, str], ...]:
    if values is None:
        return DEFAULT_PROMPTS
    return _parse_named_values(values, "prompt")


def _parse_named_values(values: list[str], kind: str) -> tuple[tuple[str, str], ...]:
    parsed: list[tuple[str, str]] = []
    labels: set[str] = set()
    for raw in values:
        label, separator, value = raw.partition("=")
        label = label.strip()
        value = value.strip()
        if not separator or not label or not value:
            raise ValueError(f"{kind} entries must use non-empty LABEL=VALUE syntax, got {raw!r}")
        if label in labels:
            raise ValueError(f"duplicate {kind} label: {label!r}")
        labels.add(label)
        parsed.append((label, value))
    if not parsed:
        raise ValueError(f"at least one {kind} is required")
    return tuple(parsed)


def _parse_seeds(value: str) -> tuple[int, ...]:
    raw_values = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_values:
        raise ValueError("at least one seed is required")
    try:
        seeds = tuple(dict.fromkeys(int(item) for item in raw_values))
    except ValueError as error:
        raise ValueError("seeds must be comma-separated integers") from error
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    return seeds


def _validate_thresholds(open_threshold: float, closed_threshold: float) -> None:
    if not np.isfinite(open_threshold) or not np.isfinite(closed_threshold):
        raise ValueError("gripper thresholds must be finite")
    if not 0.0 <= open_threshold < closed_threshold <= 100.0:
        raise ValueError("gripper thresholds must satisfy 0 <= open < closed <= 100")


def _require_common_camera_shapes(
    metadata_by_label: dict[str, CheckpointMetadata],
) -> dict[str, tuple[int, int, int]]:
    iterator = iter(metadata_by_label.items())
    first_label, first_metadata = next(iterator)
    expected = first_metadata.camera_shapes
    for label, metadata in iterator:
        if metadata.camera_shapes != expected:
            raise ValueError(
                f"checkpoint camera shapes differ: {first_label}={expected} {label}={metadata.camera_shapes}"
            )
    return expected


def _observation_sha256(observation: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        value = np.ascontiguousarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-json must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_behavior_ab(args)
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
