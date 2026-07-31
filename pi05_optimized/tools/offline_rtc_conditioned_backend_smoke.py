#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
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
DEFAULT_POLICY_PATH = (
    REPO_ROOT / "tk_infer/pi05/checkpoints/"
    "pi05_jz100_model16_head_left_right_expert_b_rtc_e10_seed1000_010600/pretrained_model"
)
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "assets/modelscope/google/paligemma-3b-pt-224"
DEFAULT_OUTPUT_PATH = OPTIMIZED_ROOT / "outputs/rtc_conditioned_backend_smoke.json"
DEFAULT_TASK = "jz robot pin timed vr teleoperation"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05.runtime.protocol import InferenceRequest  # noqa: E402
from tk_infer.pi05_optimized.backends.torch_rtc_conditioned_backend import (  # noqa: E402
    TorchRTCConditionedBackend,
)
from tk_infer.pi05_optimized.config import OptimizedRuntimeConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline real-checkpoint smoke for the independent PI0.5 RTC-conditioned backend."
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix-len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    _configure_offline_environment()
    backend = TorchRTCConditionedBackend.from_runtime_config(
        OptimizedRuntimeConfig(
            backend="torch_rtc_conditioned",
            policy_path=args.policy_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            require_complete_step=False,
            rtc_conditioned_task=args.task,
        )
    )
    if not 0 <= args.prefix_len <= backend.contract.maximum_prefix_length:
        raise ValueError(
            f"prefix-len must be in [0, {backend.contract.maximum_prefix_length}], got {args.prefix_len}"
        )
    observation = _synthetic_observation(backend.bundle.metadata.camera_shapes)
    common = {
        "observation_frame": observation,
        "task": args.task,
        "robot_type": "jz_robot_pin_timed",
        "obs_sequence_id": 1,
        "execution_horizon": 10,
    }

    _reset_backend(backend, seed=args.seed)
    ordinary = backend.infer(InferenceRequest(request_id=1, mode="single_step", **common))
    _reset_backend(backend, seed=args.seed)
    zero_prefix = backend.infer(
        InferenceRequest(
            request_id=2,
            mode="rtc",
            predicted_delay_steps=0,
            prev_chunk_left_over=None,
            **common,
        )
    )
    _reset_backend(backend, seed=args.seed)
    clamped = backend.infer(
        InferenceRequest(
            request_id=3,
            mode="rtc",
            predicted_delay_steps=args.prefix_len,
            prev_chunk_left_over=np.ascontiguousarray(ordinary.raw_actions),
            **common,
        )
    )

    prefix_model_error = _max_abs(
        ordinary.raw_actions[: args.prefix_len],
        clamped.raw_actions[: args.prefix_len],
    )
    prefix_robot_error = _max_abs(
        ordinary.processed_actions[: args.prefix_len],
        clamped.processed_actions[: args.prefix_len],
    )
    zero_model_error = _max_abs(ordinary.raw_actions, zero_prefix.raw_actions)
    zero_robot_error = _max_abs(ordinary.processed_actions, zero_prefix.processed_actions)
    if zero_model_error != 0.0 or zero_robot_error != 0.0:
        raise AssertionError("zero-prefix RTC call differs from ordinary full-chunk inference")
    if prefix_model_error != 0.0 or prefix_robot_error != 0.0:
        raise AssertionError("conditioned backend did not preserve the exact clean prefix")

    force_exact = bool(
        np.equal(clamped.processed_actions[:, 15], 80.0).all()
        and np.equal(clamped.processed_actions[:, 17], 80.0).all()
    )
    if not force_exact:
        raise AssertionError("conditioned backend did not preserve exact raw18 force slots")
    health = backend.health()
    return {
        "status": "PASS",
        "hardware_access": False,
        "network_access": False,
        "backend": backend.name,
        "policy_path": str(args.policy_path.expanduser().resolve()),
        "checkpoint_fingerprint": health["checkpoint_fingerprint"],
        "camera_profile": health["camera_profile"],
        "task": args.task,
        "seed": args.seed,
        "prefix_len": args.prefix_len,
        "rtc_conditioning": health["rtc_conditioning"],
        "inference_time_vjp_rtc_enabled": health["inference_time_vjp_rtc_enabled"],
        "model_action_shape": list(clamped.raw_actions.shape),
        "wire_action_shape": list(clamped.processed_actions.shape),
        "zero_prefix_model_max_abs": zero_model_error,
        "zero_prefix_wire_max_abs": zero_robot_error,
        "clamped_prefix_model_max_abs": prefix_model_error,
        "clamped_prefix_wire_max_abs": prefix_robot_error,
        "postfix_model_max_abs_vs_ordinary": _max_abs(
            ordinary.raw_actions[args.prefix_len :],
            clamped.raw_actions[args.prefix_len :],
        ),
        "force_slots_exact_80": force_exact,
        "backend_health": health,
    }


def _synthetic_observation(
    camera_shapes: dict[str, tuple[int, int, int]],
) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {"observation.state": np.zeros(18, dtype=np.float32)}
    for key, (channels, height, width) in camera_shapes.items():
        if channels != 3:
            raise ValueError(f"camera {key} must have three channels, got {channels}")
        observation[key] = np.zeros((height, width, channels), dtype=np.uint8)
    return observation


def _reset_backend(backend: TorchRTCConditionedBackend, *, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for component in (backend.policy, backend.preprocessor, backend.postprocessor):
        reset = getattr(component, "reset", None)
        if callable(reset):
            reset()


def _max_abs(expected: object, actual: object) -> float:
    expected_array = np.asarray(expected, dtype=np.float32)
    actual_array = np.asarray(actual, dtype=np.float32)
    if expected_array.shape != actual_array.shape:
        raise ValueError(f"shape mismatch: {expected_array.shape} != {actual_array.shape}")
    difference = np.abs(expected_array.astype(np.float64) - actual_array.astype(np.float64))
    return float(difference.max(initial=0.0))


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
    report = run_smoke(args)
    output_path = _write_report(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
