#!/usr/bin/env python
"""Offline PI0.5 training-time RTC inference check using synthetic observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "lerobot").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.checkpoint import (  # noqa: E402
    load_policy_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix-len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--task", default="pick up the object")
    return parser.parse_args()


def synthetic_observation() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros(18, dtype=np.float32),
        "observation.images.camera_head": np.zeros((720, 1280, 3), dtype=np.uint8),
        "observation.images.camera_left": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.images.camera_right": np.zeros((480, 640, 3), dtype=np.uint8),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    bundle = load_policy_bundle(
        args.policy_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        require_complete_step=True,
    )
    policy = bundle.policy
    rtc_training = policy.config.rtc_training
    if not rtc_training.enabled:
        raise RuntimeError("Checkpoint does not have rtc_training.enabled=true")
    if not 0 <= args.prefix_len < policy.config.chunk_size:
        raise ValueError(f"prefix-len must be in [0, {policy.config.chunk_size - 1}]")

    batch = prepare_observation_for_inference(
        synthetic_observation(),
        device=device,
        task=args.task,
        robot_type="jz_robot_pin_timed",
    )
    batch = bundle.preprocessor(batch)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        generator=generator,
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        ordinary = policy.predict_action_chunk(batch, noise=noise.clone())
        empty_prefix = ordinary[:, :0]
        prefix_zero = policy.predict_action_chunk(
            batch,
            noise=noise.clone(),
            action_prefix=empty_prefix,
            prefix_length=0,
        )
        clean_prefix = ordinary[:, : args.prefix_len].clone()
        clamped = policy.predict_action_chunk(
            batch,
            noise=noise.clone(),
            action_prefix=clean_prefix,
            prefix_length=args.prefix_len,
        )

    if not torch.equal(ordinary, prefix_zero):
        raise AssertionError("prefix_len=0 differs from ordinary single-step inference")
    if not torch.equal(clamped[:, : args.prefix_len], clean_prefix):
        max_error = (clamped[:, : args.prefix_len] - clean_prefix).abs().max().item()
        raise AssertionError(f"Euler prefix clamp failed; max_abs_error={max_error}")

    raw18 = bundle.postprocessor(ordinary.detach().to(device="cpu", dtype=torch.float32))
    result = {
        "status": "PASS",
        "checkpoint": str(args.policy_path.resolve()),
        "rtc_training": {
            "enabled": rtc_training.enabled,
            "max_delay": rtc_training.max_delay,
            "min_postfix_steps": rtc_training.min_postfix_steps,
        },
        "model_action_shape": list(ordinary.shape),
        "wire_action_shape": list(raw18.shape),
        "prefix_len_zero_matches_ordinary": True,
        "clamped_prefix_len": args.prefix_len,
        "clamped_prefix_exact": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
