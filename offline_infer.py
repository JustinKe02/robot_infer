#!/usr/bin/env python
"""Run an isolated, read-only policy inference task on dataset samples.

This task loads a local LeRobot checkpoint, runs the checkpoint's serialized
preprocessor and postprocessor, and writes a small JSON report. It never
constructs or connects a robot and never sends an action to hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

CAMERA_PREFIX = "observation.images."
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "train_config.json",
)


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object with a useful path in any error message."""

    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except OSError as error:
        raise OSError(f"Could not read JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def parse_indices(raw: str, dataset_length: int) -> list[int]:
    """Parse comma-separated integer indices and first/middle/last aliases."""

    if dataset_length <= 0:
        raise ValueError("The dataset must contain at least one sample")

    aliases = {
        "first": 0,
        "middle": dataset_length // 2,
        "last": dataset_length - 1,
    }
    indices: list[int] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in aliases:
            index = aliases[token]
        elif token.lstrip("-").isdigit():
            index = int(token)
        else:
            raise ValueError(f"Invalid sample index token: {token!r}")

        if index < 0:
            index += dataset_length
        if not 0 <= index < dataset_length:
            raise IndexError(f"Sample index {index} is outside [0, {dataset_length})")
        if index not in indices:
            indices.append(index)

    if not indices:
        raise ValueError("At least one sample index is required")
    return indices


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` and reject unavailable accelerator requests early."""

    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return requested
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA inference was requested but torch.cuda.is_available() is false")
        return requested
    mps_backend = getattr(torch.backends, "mps", None)
    if requested == "mps":
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS inference was requested but torch.backends.mps.is_available() is false")
        return requested
    raise ValueError(f"Unsupported device {requested!r}; expected auto, cpu, cuda[:index], or mps")


def validate_checkpoint(policy_path: Path) -> dict[str, Any]:
    """Validate files needed by a local checkpoint before importing policy code."""

    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy directory does not exist: {policy_path}")

    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (policy_path / name).is_file()]
    has_full_weights = (policy_path / "model.safetensors").is_file()
    has_adapter_weights = all(
        (policy_path / name).is_file() for name in ("adapter_config.json", "adapter_model.safetensors")
    )
    if not has_full_weights and not has_adapter_weights:
        missing.append("model weights (model.safetensors or adapter_config.json + adapter_model.safetensors)")
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required files: {missing}")

    config = load_json(policy_path / "config.json")
    if "input_features" not in config or "output_features" not in config:
        raise ValueError(
            f"Checkpoint config has no input_features/output_features: {policy_path / 'config.json'}"
        )
    return config


def validate_tokenizer_path(tokenizer_path: Path) -> Path:
    """Validate a local Hugging Face tokenizer directory."""

    tokenizer_path = tokenizer_path.expanduser().resolve()
    missing = [
        name for name in ("tokenizer.json", "tokenizer_config.json") if not (tokenizer_path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Tokenizer directory is missing required files: {missing}")
    return tokenizer_path


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_inference(
    *,
    policy_path: Path,
    dataset_root: Path,
    dataset_repo_id: str,
    sample_indices: str,
    device: str,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    """Load the policy and return the report for the requested dataset samples."""

    # Keep LeRobot imports here so argument validation and unit tests stay
    # independent of cameras, robot drivers, and optional video backends.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.robots.jz_robot_pin_timed.training_schema import load_training_schema_from_local_checkpoint

    policy_path = policy_path.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    device = resolve_device(device)
    config = validate_checkpoint(policy_path)
    tokenizer_path = validate_tokenizer_path(tokenizer_path) if tokenizer_path is not None else None

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata is missing: {info_path}")

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path), local_files_only=True)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(
        str(policy_path),
        config=policy_cfg,
        strict=False,
        local_files_only=True,
    ).to(device)
    policy.eval()

    preprocessor_overrides: dict[str, dict[str, Any]] = {"device_processor": {"device": device}}
    if tokenizer_path is not None:
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer_path)}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides=preprocessor_overrides,
    )
    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, video_backend="pyav")
    schema = load_training_schema_from_local_checkpoint(policy_path)
    if schema is not None:
        schema.ensure_trainable()
        schema.validate_raw_features(dataset.meta.features)
    indices = parse_indices(sample_indices, len(dataset))

    info = load_json(info_path)
    action_feature = info.get("features", {}).get("action", {})
    action_names = action_feature.get("names", [])
    results: list[dict[str, Any]] = []
    warnings: list[str] = []

    for index in indices:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        raw_sample = dataset[index]

        _synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            processed = preprocessor(raw_sample)
            model_action = policy.select_action(processed)
            raw_action = postprocessor(model_action)
        _synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000

        model_vector = model_action.detach().cpu().reshape(-1, model_action.shape[-1])[0]
        raw_vector = raw_action.detach().cpu().reshape(-1, raw_action.shape[-1])[0]
        if not torch.isfinite(model_action).all() or not torch.isfinite(raw_action).all():
            raise ValueError(f"Inference sample {index} produced non-finite action values")

        camera_shapes = {
            key: list(value.shape)
            for key, value in processed.items()
            if key.startswith(CAMERA_PREFIX) and hasattr(value, "shape")
        }
        raw_action_values: Any = raw_vector.tolist()
        if len(action_names) == len(raw_action_values):
            raw_action_values = dict(zip(action_names, raw_action_values, strict=True))
        gripper_out_of_range: dict[str, float] = {}
        if raw_vector.numel() == 18:
            for name, position in (("left_gripper.width", 14), ("right_gripper.width", 16)):
                value = float(raw_vector[position].item())
                if not 0.0 <= value <= 100.0:
                    gripper_out_of_range[name] = value
            if gripper_out_of_range:
                warning = (
                    f"sample {index}: gripper width is outside [0, 100] before a live Robot boundary clamp: "
                    f"{gripper_out_of_range}"
                )
                warnings.append(warning)
                print(f"WARNING {warning}")

        results.append(
            {
                "sample_index": index,
                "episode_index": int(raw_sample["episode_index"].item())
                if "episode_index" in raw_sample
                else None,
                "frame_index": int(raw_sample["frame_index"].item()) if "frame_index" in raw_sample else None,
                "latency_ms": latency_ms,
                "raw_state_shape": list(raw_sample["observation.state"].shape)
                if hasattr(raw_sample.get("observation.state"), "shape")
                else None,
                "model_action_shape": list(model_action.shape),
                "raw_action_shape": list(raw_action.shape),
                "processed_camera_shapes": camera_shapes,
                "model_action": model_vector.tolist(),
                "raw_action": raw_action_values,
                "gripper_out_of_range_before_robot_clamp": gripper_out_of_range,
            }
        )
        print(
            f"sample={index} latency_ms={latency_ms:.3f} "
            f"model_action={tuple(model_action.shape)} raw_action={tuple(raw_action.shape)}"
        )

    return {
        "status": "PASS",
        "policy_path": str(policy_path),
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "device": device,
        "policy_type": policy_cfg.type,
        "policy_config": {
            "chunk_size": getattr(policy_cfg, "chunk_size", None),
            "n_action_steps": getattr(policy_cfg, "n_action_steps", None),
        },
        "checkpoint_input_features": config.get("input_features"),
        "checkpoint_output_features": config.get("output_features"),
        "warnings": warnings,
        "samples": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True, help="checkpoint/.../pretrained_model")
    parser.add_argument("--dataset-root", type=Path, required=True, help="local LeRobot dataset root")
    parser.add_argument("--dataset-repo-id", required=True, help="dataset repo id stored in metadata")
    parser.add_argument("--sample-indices", default="first,middle,last")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--tokenizer-path", type=Path, help="optional local tokenizer override for VLA policies"
    )
    parser.add_argument("--output-json", type=Path, help="optional report path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_inference(
        policy_path=args.policy_path,
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        sample_indices=args.sample_indices,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
    )
    if args.output_json:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report={output_path}")
    print(f"status=PASS samples={len(report['samples'])} device={report['device']}")


if __name__ == "__main__":
    main()
