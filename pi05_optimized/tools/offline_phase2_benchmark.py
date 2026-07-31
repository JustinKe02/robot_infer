#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
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
DEFAULT_TASK = "jz robot pin timed vr teleoperation"
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/phase2_torch_benchmark.json"
DEFAULT_RAW_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/phase2_torch_benchmark_raw.json"
DEFAULT_VARIANTS = (
    "reference",
    "optimized_plain",
    "inference_mode",
    "bf16",
    "inference_mode_bf16",
    "pinned_memory",
    "pinned_non_blocking",
)

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from lerobot.configs.types import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from tk_infer.pi05.runtime.checkpoint import inspect_checkpoint  # noqa: E402
from tk_infer.pi05.runtime.policy_service import PolicyService, PolicyServiceConfig  # noqa: E402
from tk_infer.pi05.runtime.protocol import InferenceRequest, InferenceResponse  # noqa: E402
from tk_infer.pi05_optimized.backends.torch_backend import TorchPolicyBackend  # noqa: E402
from tk_infer.pi05_optimized.backends.torch_optimized_backend import (  # noqa: E402
    TorchBackendOptions,
    TorchOptimizedBackend,
)
from tk_infer.pi05_optimized.runtime.backend_manifest import TorchFeatureFlags  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline, same-checkpoint Phase 2 PI0.5 backend parity and latency benchmark."
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--stage-repetitions", type=int, default=10)
    parser.add_argument("--mode", choices=["single_step", "rtc", "both"], default="both")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--require-complete-step", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--raw-output-json", type=Path, default=DEFAULT_RAW_OUTPUT_PATH)
    return parser


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    _configure_offline_environment()
    _validate_args(args)
    variants = _parse_variants(args.variants)
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
    backends = _build_backends(service, variants)
    reference = backends["reference"]
    modes = ("single_step", "rtc") if args.mode == "both" else (args.mode,)
    single_request = _make_request(
        request_id=1,
        mode="single_step",
        observation_frame=observation,
        task=args.task,
    )
    _reset_components(service, seed=args.seed)
    reference_single = reference.infer(single_request)
    leftover_start = min(10, len(reference_single.raw_actions) - 1)
    leftover = np.ascontiguousarray(reference_single.raw_actions[leftover_start:])
    requests = {
        "single_step": single_request,
        "rtc": _make_request(
            request_id=2,
            mode="rtc",
            observation_frame=observation,
            task=args.task,
            prev_chunk_left_over=leftover,
        ),
    }

    correctness: dict[str, dict[str, Any]] = {}
    raw_latency: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for mode_index, mode in enumerate(modes):
        request = requests[mode]
        mode_seed = args.seed + mode_index * 100_000
        correctness[mode] = {}
        raw_latency[mode] = {}
        summaries[mode] = {}
        _reset_components(service, seed=mode_seed)
        expected = reference.infer(request)
        for variant_index, variant in enumerate(variants):
            backend = backends[variant]
            if mode not in backend.health().get("supported_modes", ("single_step", "rtc")):
                reason = "torch.inference_mode is incompatible with autograd-based RTC guidance"
                correctness[mode][variant] = {
                    "status": "UNSUPPORTED",
                    "gate_passed": None,
                    "reason": reason,
                }
                raw_latency[mode][variant] = {"status": "UNSUPPORTED", "reason": reason}
                summaries[mode][variant] = {"status": "UNSUPPORTED", "reason": reason}
                continue
            _reset_components(service, seed=mode_seed)
            actual = backend.infer(request)
            comparison = _compare_actions(expected, actual, reduced_precision="bf16" in variant)
            correctness[mode][variant] = comparison
            if not comparison["gate_passed"]:
                failures.append(f"{mode}/{variant}: {comparison['gate_failures']}")
            measurement = _measure(
                backend=backend,
                service=service,
                request=request,
                device=args.device,
                warmup=args.warmup,
                repetitions=args.repetitions,
                seed=mode_seed + (variant_index + 1) * 10_000,
            )
            raw_latency[mode][variant] = measurement["raw"]
            summaries[mode][variant] = measurement["summary"]

            if isinstance(backend, TorchOptimizedBackend) and args.stage_repetitions:
                diagnostic = TorchOptimizedBackend(
                    service,
                    options=replace(backend.options, synchronize_stages=True),
                )
                stage_measurement = _measure_stages(
                    backend=diagnostic,
                    service=service,
                    request=request,
                    repetitions=args.stage_repetitions,
                    seed=mode_seed + (variant_index + 1) * 1_000_000,
                )
                raw_latency[mode][variant]["synchronized_stage_samples_s"] = stage_measurement[
                    "raw"
                ]
                summaries[mode][variant]["synchronized_stage_diagnostic"] = stage_measurement[
                    "summary"
                ]

    common = {
        "hardware_access": False,
        "network_access": False,
        "policy_path": str(policy_path),
        "checkpoint_fingerprint": metadata.checkpoint_fingerprint,
        "checkpoint_step": metadata.checkpoint_step,
        "dataset_root": str(dataset_root),
        "dataset_repo_id": args.dataset_repo_id,
        "sample_index": args.sample_index,
        "device": args.device,
        "seed": args.seed,
        "warmup_per_variant_mode": args.warmup,
        "repetitions_per_variant_mode": args.repetitions,
        "stage_repetitions_per_variant_mode": args.stage_repetitions,
        "modes": list(modes),
        "variants": list(variants),
        "timing_contract": {
            "end_to_end": "torch.cuda.synchronize before and after each measured infer call",
            "stages": "synchronized boundary diagnostic; excluded from throughput comparison",
        },
        "deferred_features": {
            "static_buffers": "startup-rejected until fixed observation/token ownership is implemented",
            "cuda_graph": "startup-rejected for dynamic KV cache, RTC branch, and denoising loop",
        },
    }
    summary_report = {
        "status": "PASS" if not failures else "FAIL",
        **common,
        "correctness": correctness,
        "latency_summary": summaries,
        "gate_failures": failures,
        "backend_manifests": {
            name: backend.health().get("backend_manifest") for name, backend in backends.items()
        },
    }
    raw_report = {
        "status": summary_report["status"],
        **common,
        "raw_latency_samples": raw_latency,
    }
    return summary_report, raw_report


def _build_backends(service: PolicyService, variants: tuple[str, ...]) -> dict[str, Any]:
    feature_map = {
        "optimized_plain": TorchFeatureFlags(),
        "inference_mode": TorchFeatureFlags(inference_mode=True),
        "bf16": TorchFeatureFlags(bf16_autocast=True),
        "inference_mode_bf16": TorchFeatureFlags(inference_mode=True, bf16_autocast=True),
        "pinned_memory": TorchFeatureFlags(pinned_memory=True),
        "pinned_non_blocking": TorchFeatureFlags(pinned_memory=True, non_blocking_copies=True),
    }
    backends: dict[str, Any] = {}
    for variant in variants:
        if variant == "reference":
            backends[variant] = TorchPolicyBackend(service)
        else:
            backends[variant] = TorchOptimizedBackend(
                service,
                options=TorchBackendOptions(features=feature_map[variant]),
            )
    if "reference" not in backends:
        backends["reference"] = TorchPolicyBackend(service)
    return backends


def _measure(
    *,
    backend: Any,
    service: PolicyService,
    request: InferenceRequest,
    device: str,
    warmup: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    for index in range(warmup):
        _reset_components(service, seed=seed + index)
        _synchronize(device)
        backend.infer(request)
        _synchronize(device)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    allocated_before = _cuda_memory("allocated", device)
    reserved_before = _cuda_memory("reserved", device)
    elapsed_samples: list[float] = []
    server_samples: list[float] = []
    model_samples: list[float] = []
    for index in range(repetitions):
        _reset_components(service, seed=seed + warmup + index)
        _synchronize(device)
        started_s = time.perf_counter()
        response = backend.infer(request)
        _synchronize(device)
        elapsed_samples.append(time.perf_counter() - started_s)
        server_samples.append(response.server_latency_s)
        model_samples.append(response.model_latency_s)
    memory = {
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": _cuda_memory("allocated", device),
        "reserved_before_bytes": reserved_before,
        "reserved_after_bytes": _cuda_memory("reserved", device),
        "peak_allocated_bytes": _cuda_memory("peak_allocated", device),
        "peak_reserved_bytes": _cuda_memory("peak_reserved", device),
    }
    return {
        "raw": {
            "end_to_end_s": elapsed_samples,
            "server_reported_s": server_samples,
            "model_reported_s": model_samples,
        },
        "summary": {
            "end_to_end_s": _distribution(elapsed_samples),
            "server_reported_s": _distribution(server_samples),
            "model_reported_s": _distribution(model_samples),
            "cuda_memory": memory,
        },
    }


def _measure_stages(
    *,
    backend: TorchOptimizedBackend,
    service: PolicyService,
    request: InferenceRequest,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    samples: dict[str, list[float]] = {}
    for index in range(repetitions):
        _reset_components(service, seed=seed + index)
        backend.infer(request)
        for name, value in backend.last_stage_timings.items():
            samples.setdefault(name, []).append(value)
    return {
        "raw": samples,
        "summary": {name: _distribution(values) for name, values in samples.items()},
    }


def _compare_actions(
    expected: InferenceResponse,
    actual: InferenceResponse,
    *,
    reduced_precision: bool,
) -> dict[str, Any]:
    expected_model = np.asarray(expected.raw_actions, dtype=np.float32)
    actual_model = np.asarray(actual.raw_actions, dtype=np.float32)
    expected_robot = np.asarray(expected.processed_actions, dtype=np.float32)
    actual_robot = np.asarray(actual.processed_actions, dtype=np.float32)
    if expected_model.shape != actual_model.shape or expected_robot.shape != actual_robot.shape:
        raise ValueError("reference and optimized action shapes differ")
    model_error = _error_metrics(expected_model, actual_model)
    robot_error = _error_metrics(expected_robot, actual_robot)
    joint_error = _error_metrics(expected_robot[:, :14], actual_robot[:, :14])
    gripper_indices = np.asarray((14, 16))
    gripper_error = _error_metrics(
        expected_robot[:, gripper_indices],
        actual_robot[:, gripper_indices],
    )
    force_exact = bool(
        np.equal(actual_robot[:, 15], 80.0).all()
        and np.equal(actual_robot[:, 17], 80.0).all()
    )
    finite = bool(np.isfinite(actual_model).all() and np.isfinite(actual_robot).all())
    failures: list[str] = []
    if reduced_precision:
        if joint_error["p99_abs"] >= 0.005:
            failures.append("joint p99 must be < 0.005 rad")
        if joint_error["max_abs"] >= 0.01:
            failures.append("joint max must be < 0.01 rad")
        if gripper_error["p99_abs"] >= 0.5:
            failures.append("gripper p99 must be < 0.5")
    elif robot_error["max_abs"] > 1e-5 or robot_error["mean_abs"] > 1e-6:
        failures.append("FP32 raw18 error exceeds max=1e-5 or mean=1e-6")
    if not force_exact:
        failures.append("force slots are not exactly 80")
    if not finite:
        failures.append("optimized output contains NaN/Inf")
    return {
        "precision_gate": "reduced_bf16" if reduced_precision else "fp32",
        "model16_error": model_error,
        "raw18_error": robot_error,
        "joint_error": joint_error,
        "gripper_error": gripper_error,
        "force_slots_exact_80": force_exact,
        "finite": finite,
        "gate_passed": not failures,
        "gate_failures": failures,
    }


def _error_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64)).reshape(-1)
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "p99_abs": float(np.percentile(difference, 99)) if difference.size else 0.0,
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
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
        if torch.any(image < 0) or torch.any(image > 1):
            raise ValueError(f"dataset image {key} must be normalized to [0,1]")
        frame[key] = np.ascontiguousarray(image.permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy())
    return frame


def _make_request(
    *,
    request_id: int,
    mode: str,
    observation_frame: dict[str, np.ndarray],
    task: str,
    prev_chunk_left_over: np.ndarray | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        mode=mode,  # type: ignore[arg-type]
        observation_frame=observation_frame,
        task=task,
        robot_type="jz_robot_pin_timed",
        obs_sequence_id=1,
        predicted_delay_steps=1 if mode == "rtc" else 0,
        prev_chunk_left_over=prev_chunk_left_over,
        execution_horizon=10,
    )


def _reset_components(service: PolicyService, *, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for component in (service.policy, service.preprocessor, service.postprocessor):
        reset = getattr(component, "reset", None)
        if callable(reset):
            reset()


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_memory(kind: str, device: str) -> int | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    functions = {
        "allocated": torch.cuda.memory_allocated,
        "reserved": torch.cuda.memory_reserved,
        "peak_allocated": torch.cuda.max_memory_allocated,
        "peak_reserved": torch.cuda.max_memory_reserved,
    }
    return int(functions[kind]())


def _parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; expected subset of {list(DEFAULT_VARIANTS)}")
    if not variants:
        raise ValueError("at least one benchmark variant is required")
    return variants


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("warmup", "repetitions", "stage_repetitions"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if args.repetitions == 0:
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
        raise ValueError(f"output path must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, raw = run_benchmark(args)
    summary_path = _write_report(args.output_json, summary)
    raw_path = _write_report(args.raw_output_json, raw)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary_report={summary_path}")
    print(f"raw_report={raw_path}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
