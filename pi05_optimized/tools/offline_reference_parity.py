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
DEFAULT_TASK = "jz robot pin timed vr teleoperation"
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/phase0_reference_parity.json"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from lerobot.configs.types import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from tk_infer.pi05.runtime.checkpoint import inspect_checkpoint  # noqa: E402
from tk_infer.pi05.runtime.policy_service import PolicyService, PolicyServiceConfig  # noqa: E402
from tk_infer.pi05.runtime.protocol import InferenceRequest, InferenceResponse  # noqa: E402
from tk_infer.pi05_optimized.runtime.policy_service import OptimizedPolicyService  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the trusted PI0.5 service with the Phase 0 optimized pass-through path offline."
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mode", choices=["single_step", "rtc", "both"], default="both")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--require-complete-step", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_parity(args: argparse.Namespace) -> dict[str, Any]:
    _configure_offline_environment()
    policy_path = args.policy_path.expanduser().resolve(strict=True)
    tokenizer_path = args.tokenizer_path.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    if not args.task.strip():
        raise ValueError("task must be a non-empty string")
    if args.atol < 0 or not np.isfinite(args.atol):
        raise ValueError("atol must be finite and non-negative")
    metadata, _ = inspect_checkpoint(
        policy_path,
        tokenizer_path=tokenizer_path,
        require_complete_step=args.require_complete_step,
    )
    observation_frame = _load_observation(
        dataset_root=dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        sample_index=args.sample_index,
        camera_shapes=metadata.camera_shapes,
    )
    reference = PolicyService.from_config(
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
    reports: list[dict[str, Any]] = []
    baseline_single: InferenceResponse | None = None
    if args.mode in {"single_step", "both"}:
        single_request = _make_request(
            request_id=1,
            mode="single_step",
            observation_frame=observation_frame,
            task=args.task,
        )
        baseline_single, report = _compare_one(
            reference=reference,
            request=single_request,
            seed=args.seed,
            device=args.device,
            atol=args.atol,
        )
        reports.append(report)

    if args.mode in {"rtc", "both"}:
        if baseline_single is None:
            seed_request = _make_request(
                request_id=0,
                mode="single_step",
                observation_frame=observation_frame,
                task=args.task,
            )
            _reset_reference_state(reference, seed=args.seed)
            baseline_single = reference.infer(seed_request)
        leftover_start = min(10, len(baseline_single.raw_actions) - 1)
        leftover = np.ascontiguousarray(baseline_single.raw_actions[leftover_start:])
        rtc_request = _make_request(
            request_id=2,
            mode="rtc",
            observation_frame=observation_frame,
            task=args.task,
            prev_chunk_left_over=leftover,
        )
        _, report = _compare_one(
            reference=reference,
            request=rtc_request,
            seed=args.seed + 1,
            device=args.device,
            atol=args.atol,
        )
        reports.append(report)

    return {
        "status": "PASS",
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
        "atol": args.atol,
        "comparison_strategy": "captured_reference_response_through_optimized_facade",
        "comparisons": reports,
    }


def _compare_one(
    *,
    reference: PolicyService,
    request: InferenceRequest,
    seed: int,
    device: str,
    atol: float,
) -> tuple[InferenceResponse, dict[str, Any]]:
    _reset_reference_state(reference, seed=seed)
    _synchronize(device)
    baseline_started_s = time.perf_counter()
    baseline = reference.infer(request)
    _synchronize(device)
    baseline_elapsed_s = time.perf_counter() - baseline_started_s

    optimized = OptimizedPolicyService(backend=_CapturedReferenceBackend(baseline, reference.health()))
    optimized_started_s = time.perf_counter()
    actual = optimized.infer(request)
    optimized_elapsed_s = time.perf_counter() - optimized_started_s

    model_metrics = action_error_metrics(baseline.raw_actions, actual.raw_actions)
    robot_metrics = action_error_metrics(baseline.processed_actions, actual.processed_actions)
    if model_metrics["max_abs"] > atol or robot_metrics["max_abs"] > atol:
        raise AssertionError(
            f"{request.mode} parity failed at atol={atol}: model={model_metrics} robot={robot_metrics}"
        )
    return baseline, {
        "mode": request.mode,
        "model_action_shape": list(actual.raw_actions.shape),
        "robot_action_shape": list(actual.processed_actions.shape),
        "model_error": model_metrics,
        "robot_error": robot_metrics,
        "baseline_elapsed_s": baseline_elapsed_s,
        "optimized_elapsed_s": optimized_elapsed_s,
        "force_slots_exact_80": bool(
            np.equal(actual.processed_actions[:, 15], 80.0).all()
            and np.equal(actual.processed_actions[:, 17], 80.0).all()
        ),
    }


class _CapturedReferenceBackend:
    """Replay one real response so facade parity needs no second stochastic model forward."""

    def __init__(self, response: InferenceResponse, health: dict[str, Any]) -> None:
        self._response = response
        self._health = dict(health)

    @property
    def name(self) -> str:
        return "captured_reference"

    def health(self) -> dict[str, Any]:
        return dict(self._health)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if request.request_id != self._response.request_id or request.mode != self._response.mode:
            raise ValueError("captured reference response does not match the replay request")
        return self._response


def action_error_metrics(expected: object, actual: object) -> dict[str, float]:
    expected_array = np.asarray(expected, dtype=np.float32)
    actual_array = np.asarray(actual, dtype=np.float32)
    if expected_array.shape != actual_array.shape:
        raise ValueError(
            f"action shape mismatch: expected={expected_array.shape} actual={actual_array.shape}"
        )
    if not np.isfinite(expected_array).all() or not np.isfinite(actual_array).all():
        raise ValueError("action comparison contains non-finite values")
    difference = np.abs(expected_array.astype(np.float64) - actual_array.astype(np.float64))
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
    }


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
        image_hwc = image.permute(1, 2, 0).mul(255).round().to(torch.uint8)
        frame[key] = np.ascontiguousarray(image_hwc.numpy())
    return frame


def _reset_reference_state(reference: PolicyService, *, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for component in (reference.policy, reference.preprocessor, reference.postprocessor):
        reset = getattr(component, "reset", None)
        if callable(reset):
            reset()


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


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
    report = run_parity(args)
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
