#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
DEFAULT_POLICY_PATH = REPO_ROOT / "tk_infer/pi05/checkpoints/010600/pretrained_model"
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "assets/modelscope/google/paligemma-3b-pt-224"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/jz_robot_pin_timed_merged_100eps_20260728"
DEFAULT_DATASET_REPO_ID = "local/jz_robot_pin_timed_merged_100eps_20260728_pi05_head_right"
DEFAULT_ARTIFACT = OPTIMIZED_ROOT / "artifacts/triton/realtime_vla_b86a942"
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/phase3_triton_benchmark.json"
DEFAULT_TASK = "jz robot pin timed vr teleoperation"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from lerobot.configs.types import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from tk_infer.pi05.runtime.checkpoint import inspect_checkpoint  # noqa: E402
from tk_infer.pi05.runtime.policy_service import PolicyService, PolicyServiceConfig  # noqa: E402
from tk_infer.pi05.runtime.protocol import InferenceRequest, InferenceResponse  # noqa: E402
from tk_infer.pi05_optimized.backends.triton_backend import (  # noqa: E402
    TritonArtifact,
    TritonPolicyBackend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline same-seed Torch vs Realtime-VLA Triton PI0.5 benchmark."
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--require-complete-step", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _configure_offline_environment()
    _validate_args(args)
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
    request = InferenceRequest(
        request_id=1,
        mode="single_step",
        observation_frame=observation,
        task=args.task,
        robot_type="jz_robot_pin_timed",
        obs_sequence_id=1,
        predicted_delay_steps=0,
        execution_horizon=10,
    )
    reference = PolicyService.from_config(
        PolicyServiceConfig(
            policy_path=policy_path,
            tokenizer_path=tokenizer_path,
            device="cuda",
            require_complete_step=args.require_complete_step,
        ),
        rtc_config=RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
            max_guidance_weight=10.0,
            execution_horizon=10,
        ),
    )
    artifact = TritonArtifact.load(args.artifact)
    triton_load_started_s = time.perf_counter()
    triton_backend = TritonPolicyBackend(
        processor_service=reference,
        artifact=artifact,
        tokenizer_path=tokenizer_path,
    )
    triton_load_s = time.perf_counter() - triton_load_started_s

    _reset_reference(reference, seed=args.seed)
    torch.cuda.synchronize()
    expected = reference.infer(request)
    torch.cuda.synchronize()
    _reset_reference(reference, seed=args.seed)
    torch.cuda.synchronize()
    actual = triton_backend.infer(request)
    torch.cuda.synchronize()
    correctness = _compare(expected, actual)

    latency = {
        "reference": _measure(
            lambda: reference.infer(request),
            reset=lambda seed: _reset_reference(reference, seed=seed),
            warmup=args.warmup,
            repetitions=args.repetitions,
            seed=args.seed + 10_000,
        ),
        "triton": _measure(
            lambda: triton_backend.infer(request),
            reset=lambda seed: _reset_reference(reference, seed=seed),
            warmup=args.warmup,
            repetitions=args.repetitions,
            seed=args.seed + 20_000,
        ),
    }
    reference_p95 = latency["reference"]["p95"]
    triton_p95 = latency["triton"]["p95"]
    p95_improvement = (reference_p95 - triton_p95) / reference_p95
    gate_failures = list(correctness["gate_failures"])
    if p95_improvement < 0.20:
        gate_failures.append("Triton p95 improvement is below the required 20%")
    return {
        "status": "PASS" if not gate_failures else "FAIL",
        "hardware_access": False,
        "network_access": False,
        "mode": "single_step",
        "policy_path": str(policy_path),
        "checkpoint_fingerprint": metadata.checkpoint_fingerprint,
        "checkpoint_step": metadata.checkpoint_step,
        "complete_step": metadata.complete_step,
        "artifact_path": str(artifact.directory),
        "artifact_sha256": artifact.manifest["output_sha256"],
        "upstream_commit": artifact.manifest["upstream_commit"],
        "converter_version": artifact.manifest["converter_version"],
        "seed": args.seed,
        "sample_index": args.sample_index,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "triton_load_and_graph_capture_s": triton_load_s,
        "correctness": correctness,
        "latency_s": latency,
        "p95_improvement_fraction": p95_improvement,
        "gate_failures": gate_failures,
        "backend_health": triton_backend.health(),
    }


def _measure(
    operation: Any,
    *,
    reset: Any,
    warmup: int,
    repetitions: int,
    seed: int,
) -> dict[str, float | int | None]:
    for index in range(warmup):
        reset(seed + index)
        torch.cuda.synchronize()
        operation()
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for index in range(repetitions):
        reset(seed + warmup + index)
        torch.cuda.synchronize()
        started_s = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started_s)
    array = np.asarray(samples, dtype=np.float64)
    return {
        "count": repetitions,
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _compare(expected: InferenceResponse, actual: InferenceResponse) -> dict[str, Any]:
    expected_model = np.asarray(expected.raw_actions, dtype=np.float32)
    actual_model = np.asarray(actual.raw_actions, dtype=np.float32)
    expected_robot = np.asarray(expected.processed_actions, dtype=np.float32)
    actual_robot = np.asarray(actual.processed_actions, dtype=np.float32)
    if expected_model.shape != actual_model.shape or expected_robot.shape != actual_robot.shape:
        raise ValueError("Torch and Triton action shapes differ")
    model_error = _error(expected_model, actual_model)
    joint_error = _error(expected_robot[:, :14], actual_robot[:, :14])
    gripper_error = _error(expected_robot[:, [14, 16]], actual_robot[:, [14, 16]])
    force_exact = bool(
        np.equal(actual_robot[:, 15], 80.0).all()
        and np.equal(actual_robot[:, 17], 80.0).all()
    )
    finite = bool(np.isfinite(actual_model).all() and np.isfinite(actual_robot).all())
    failures = []
    if joint_error["p99_abs"] >= 0.005:
        failures.append("joint p99 must be < 0.005 rad")
    if joint_error["max_abs"] >= 0.01:
        failures.append("joint max must be < 0.01 rad")
    if gripper_error["p99_abs"] >= 0.5:
        failures.append("gripper p99 must be < 0.5")
    if not force_exact:
        failures.append("force slots are not exactly 80")
    if not finite:
        failures.append("Triton output contains NaN/Inf")
    return {
        "model16_error": model_error,
        "joint_error": joint_error,
        "gripper_error": gripper_error,
        "force_slots_exact_80": force_exact,
        "finite": finite,
        "gate_passed": not failures,
        "gate_failures": failures,
    }


def _error(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64)).reshape(-1)
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "p99_abs": float(np.percentile(difference, 99)) if difference.size else 0.0,
    }


def _load_observation(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    sample_index: int,
    camera_shapes: dict[str, tuple[int, int, int]],
) -> dict[str, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, video_backend="pyav")
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample_index {sample_index} is outside [0, {len(dataset)})")
    sample = dataset[sample_index]
    state = torch.as_tensor(sample["observation.state"], dtype=torch.float32).detach().cpu()
    if state.shape != (18,) or not torch.isfinite(state).all():
        raise ValueError(f"dataset state must be one finite raw18 vector, got {tuple(state.shape)}")
    frame = {"observation.state": np.ascontiguousarray(state.numpy())}
    for key, expected_shape in camera_shapes.items():
        image = torch.as_tensor(sample[key]).detach().cpu()
        if tuple(image.shape) != expected_shape:
            raise ValueError(f"dataset image {key} must be CHW {expected_shape}, got {tuple(image.shape)}")
        if not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError(f"dataset image {key} must contain finite floating-point values")
        frame[key] = np.ascontiguousarray(image.permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy())
    return frame


def _reset_reference(reference: PolicyService, *, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for component in (reference.policy, reference.preprocessor, reference.postprocessor):
        reset = getattr(component, "reset", None)
        if callable(reset):
            reset()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("warmup", "repetitions"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not args.task.strip():
        raise ValueError("task must be a non-empty string")


def _configure_offline_environment() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")


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
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
