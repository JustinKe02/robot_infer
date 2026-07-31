#!/usr/bin/env python

"""Compute exact PI0.5 normalization statistics for the JZ Pin model16 view.

The curated recording stays immutable raw18.  This utility reads its parquet
files, applies the audited raw18-to-model16 projection in memory, and writes a
small statistics manifest under this PI0.5 module only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.robots.jz_robot_pin_timed.training_schema import JZPinTrainingSchema, TRAINING_DIM

STAT_NAMES = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_stats(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[1] != TRAINING_DIM:
        raise ValueError(f"Expected (*, {TRAINING_DIM}) values, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Projected training values contain NaN or infinity")

    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    stats = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "q01": quantiles[0],
        "q10": quantiles[1],
        "q50": quantiles[2],
        "q90": quantiles[3],
        "q99": quantiles[4],
        "count": np.asarray([values.shape[0]], dtype=np.int64),
    }
    return {name: value.tolist() for name, value in stats.items()}


def load_raw_vectors(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No data parquet files found under {dataset_root / 'data'}")

    action_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    for path in parquet_files:
        frame = pd.read_parquet(path, columns=["action", "observation.state"])
        action_parts.append(np.stack(frame["action"].to_numpy()).astype(np.float64, copy=False))
        state_parts.append(
            np.stack(frame["observation.state"].to_numpy()).astype(np.float64, copy=False)
        )
    return np.concatenate(action_parts), np.concatenate(state_parts), parquet_files


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    schema_path = args.schema.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    info_path = dataset_root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    if not schema_path.is_file():
        raise FileNotFoundError(schema_path)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    schema = JZPinTrainingSchema.from_file(schema_path)
    schema.ensure_trainable()
    raw_action, raw_state, parquet_files = load_raw_vectors(dataset_root)
    if raw_action.shape[0] != info["total_frames"] or raw_state.shape[0] != info["total_frames"]:
        raise ValueError(
            f"Metadata declares {info['total_frames']} frames, parquet has "
            f"action={raw_action.shape[0]} state={raw_state.shape[0]}"
        )

    model_action = np.asarray(schema.project_action(raw_action), dtype=np.float64)
    model_state = np.asarray(schema.project_observation(raw_state), dtype=np.float64)
    payload = {
        "format": "jz_pi05_model16_stats",
        "version": 1,
        "dataset_root": str(dataset_root),
        "dataset_info_sha256": file_sha256(info_path),
        "training_schema_sha256": file_sha256(schema_path),
        "total_frames": int(info["total_frames"]),
        "source_parquet_files": [str(path.relative_to(dataset_root)) for path in parquet_files],
        "stats": {
            "action": vector_stats(model_action),
            "observation.state": vector_stats(model_state),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(
        f"[jz/pi05/stats] wrote {output_path} "
        f"frames={info['total_frames']} action_dim={model_action.shape[1]} state_dim={model_state.shape[1]}"
    )


if __name__ == "__main__":
    main()

