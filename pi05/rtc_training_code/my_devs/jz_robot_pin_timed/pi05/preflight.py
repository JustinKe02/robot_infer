#!/usr/bin/env python

"""Offline-only readiness check for JZ Pin PI0.5 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.jz_robot_pin_timed.training_schema import (
    JZPinTrainingDatasetView,
    JZPinTrainingSchema,
    TRAINING_DIM,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-fps", type=int, required=True)
    parser.add_argument("--expected-task", required=True)
    parser.add_argument("--camera-key", action="append", dest="camera_keys", required=True)
    return parser.parse_args()


def require_file(path: Path, minimum_bytes: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        raise FileNotFoundError(f"Missing or incomplete file: {path}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    schema_path = args.schema.resolve()
    stats_path = args.stats.resolve()
    pi05_base = args.pi05_base.resolve()
    tokenizer = args.tokenizer.resolve()

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("robot_type") != "jz_robot_pin_timed":
        raise ValueError(f"Unexpected robot_type={info.get('robot_type')!r}")
    if (
        info.get("total_episodes") != args.expected_episodes
        or info.get("total_frames") != args.expected_frames
        or info.get("fps") != args.expected_fps
    ):
        raise ValueError(
            f"Expected {args.expected_episodes} episodes / {args.expected_frames} frames / "
            f"{args.expected_fps} fps, got {info.get('total_episodes')} / "
            f"{info.get('total_frames')} / {info.get('fps')}"
        )

    require_file(pi05_base / "config.json")
    require_file(pi05_base / "model.safetensors", minimum_bytes=10_000_000_000)
    require_file(pi05_base / "policy_preprocessor.json")
    require_file(pi05_base / "policy_postprocessor.json")
    require_file(tokenizer / "tokenizer.json", minimum_bytes=1_000_000)
    require_file(tokenizer / "tokenizer_config.json")

    schema = JZPinTrainingSchema.from_file(schema_path)
    schema.ensure_trainable()
    stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
    stats = stats_payload["stats"]
    for feature_key in ("observation.state", "action"):
        for stat_name in ("mean", "std", "q01", "q99"):
            values = np.asarray(stats[feature_key][stat_name])
            if values.shape != (TRAINING_DIM,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid stats: {feature_key}.{stat_name} {values.shape}")

    dataset = LeRobotDataset(args.dataset_repo_id, root=dataset_root, video_backend="pyav")
    projected = JZPinTrainingDatasetView(dataset, schema)
    projected.meta.stats = stats
    sample = projected[0]
    camera_keys = sorted(key for key in sample if key.startswith("observation.images."))
    expected_camera_keys = sorted(args.camera_keys)
    if camera_keys != expected_camera_keys:
        raise ValueError(f"Expected cameras {expected_camera_keys}, got {camera_keys}")
    if tuple(sample["observation.state"].shape) != (TRAINING_DIM,):
        raise ValueError(f"Unexpected state shape: {sample['observation.state'].shape}")
    if tuple(sample["action"].shape) != (TRAINING_DIM,):
        raise ValueError(f"Unexpected action shape: {sample['action'].shape}")
    if sample.get("task") != args.expected_task:
        raise ValueError(
            f"Dataset task differs from the requested prompt: {sample.get('task')!r} != "
            f"{args.expected_task!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the mandated Conda environment")

    print(
        "[jz/pi05/preflight] PASS "
        f"episodes={dataset.num_episodes} frames={dataset.num_frames} cameras={camera_keys} "
        f"state={tuple(sample['observation.state'].shape)} action={tuple(sample['action'].shape)} "
        f"task={sample['task']!r} gpu={torch.cuda.get_device_name(0)}"
    )


if __name__ == "__main__":
    main()
