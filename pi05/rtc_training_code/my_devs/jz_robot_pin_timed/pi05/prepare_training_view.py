#!/usr/bin/env python

"""Build a metadata-only PI0.5 view without modifying the recorded dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CAMERA_PREFIX = "observation.images."
VIEW_MANIFEST = "meta/pi05_training_view.json"
VIEW_FORMAT = "jz_pi05_training_view"
VIEW_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--camera-key", action="append", dest="camera_keys", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)


def make_relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, start=destination.parent), target_is_directory=True)


def filtered_mapping(values: dict[str, Any], camera_keys: tuple[str, ...]) -> dict[str, Any]:
    selected = set(camera_keys)
    return {
        key: value for key, value in values.items() if not key.startswith(CAMERA_PREFIX) or key in selected
    }


def write_episode_metadata(
    source_dir: Path,
    output_dir: Path,
    task: str,
    camera_keys: tuple[str, ...],
) -> None:
    selected = set(camera_keys)
    parquet_files = sorted(source_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode metadata under {source_dir}")

    for source_path in parquet_files:
        table = pq.read_table(source_path)
        tasks_index = table.schema.get_field_index("tasks")
        if tasks_index < 0:
            raise ValueError(f"Episode metadata lacks tasks: {source_path}")
        tasks_type = table.schema.field(tasks_index).type
        table = table.set_column(
            tasks_index,
            "tasks",
            pa.array([[task] for _ in range(table.num_rows)], type=tasks_type),
        )

        excluded_cameras = set(_camera_keys_from_episode_columns(table.column_names)) - selected
        dropped_columns = []
        for column_name in table.column_names:
            for camera_key in excluded_cameras:
                if column_name.startswith(f"videos/{camera_key}/") or column_name.startswith(
                    f"stats/{camera_key}/"
                ):
                    dropped_columns.append(column_name)
                    break
        if dropped_columns:
            table = table.drop(dropped_columns)

        destination = output_dir / source_path.relative_to(source_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)


def _camera_keys_from_episode_columns(column_names: list[str]) -> tuple[str, ...]:
    camera_keys = set()
    for column_name in column_names:
        for prefix in ("videos/", "stats/"):
            if not column_name.startswith(prefix):
                continue
            remainder = column_name[len(prefix) :]
            feature_key = remainder.split("/", 1)[0]
            if feature_key.startswith(CAMERA_PREFIX):
                camera_keys.add(feature_key)
    return tuple(sorted(camera_keys))


def expected_manifest(
    source_root: Path,
    source_info: Path,
    source_schema: Path,
    task: str,
    camera_keys: tuple[str, ...],
    excluded_camera_keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "format": VIEW_FORMAT,
        "version": VIEW_VERSION,
        "source_root": str(source_root),
        "source_info_sha256": file_sha256(source_info),
        "source_training_schema_sha256": file_sha256(source_schema),
        "task": task,
        "camera_keys": list(camera_keys),
        "excluded_camera_keys": list(excluded_camera_keys),
    }


def validate_existing_view(output_root: Path, manifest: dict[str, Any]) -> None:
    manifest_path = output_root / VIEW_MANIFEST
    require_file(manifest_path)
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual != manifest:
        raise ValueError(
            f"Existing training view does not match the requested configuration: {output_root}"
        )

    info = json.loads((output_root / "meta/info.json").read_text(encoding="utf-8"))
    actual_cameras = sorted(key for key in info["features"] if key.startswith(CAMERA_PREFIX))
    if actual_cameras != sorted(manifest["camera_keys"]):
        raise ValueError(f"Existing training view has unexpected cameras: {actual_cameras}")

    tasks = pd.read_parquet(output_root / "meta/tasks.parquet")
    if tasks.index.tolist() != [manifest["task"]] or tasks["task_index"].tolist() != [0]:
        raise ValueError("Existing training view has unexpected task metadata")

    require_directory(output_root / "data")
    for camera_key in manifest["camera_keys"]:
        require_directory(output_root / "videos" / camera_key)


def build_view(
    source_root: Path,
    output_root: Path,
    task: str,
    camera_keys: tuple[str, ...],
) -> None:
    if source_root == output_root:
        raise ValueError("The derived training view must not overwrite the source dataset")
    if not task.strip():
        raise ValueError("Task prompt must not be empty")
    if len(camera_keys) != len(set(camera_keys)):
        raise ValueError(f"Duplicate camera keys: {camera_keys}")

    source_info_path = source_root / "meta/info.json"
    source_stats_path = source_root / "meta/stats.json"
    source_schema_path = source_root / "meta/jz_pin_training_schema.json"
    source_episodes_dir = source_root / "meta/episodes"
    for path in (source_info_path, source_stats_path, source_schema_path, source_root / "meta/tasks.parquet"):
        require_file(path)
    require_directory(source_episodes_dir)
    require_directory(source_root / "data")

    source_info = json.loads(source_info_path.read_text(encoding="utf-8"))
    source_features = source_info.get("features", {})
    available_cameras = tuple(sorted(key for key in source_features if key.startswith(CAMERA_PREFIX)))
    missing_cameras = sorted(set(camera_keys) - set(available_cameras))
    if missing_cameras:
        raise ValueError(f"Requested cameras are missing from the source dataset: {missing_cameras}")
    for camera_key in camera_keys:
        if source_features[camera_key].get("dtype") != "video":
            raise ValueError(f"Expected a video feature for {camera_key}")
        require_directory(source_root / "videos" / camera_key)

    excluded_camera_keys = tuple(sorted(set(available_cameras) - set(camera_keys)))
    manifest = expected_manifest(
        source_root,
        source_info_path,
        source_schema_path,
        task,
        camera_keys,
        excluded_camera_keys,
    )
    if output_root.exists():
        validate_existing_view(output_root, manifest)
        print(
            f"[jz/pi05/view] REUSE output={output_root} episodes={source_info['total_episodes']} "
            f"cameras={list(camera_keys)} task={task!r}"
        )
        return

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = output_root.parent / f".{output_root.name}.tmp-{os.getpid()}"
    if temporary_root.exists():
        raise FileExistsError(temporary_root)

    try:
        (temporary_root / "meta").mkdir(parents=True)
        derived_info = dict(source_info)
        derived_info["features"] = filtered_mapping(source_features, camera_keys)
        derived_info["total_tasks"] = 1
        (temporary_root / "meta/info.json").write_text(
            json.dumps(derived_info, indent=4) + "\n", encoding="utf-8"
        )

        source_stats = json.loads(source_stats_path.read_text(encoding="utf-8"))
        derived_stats = filtered_mapping(source_stats, camera_keys)
        (temporary_root / "meta/stats.json").write_text(
            json.dumps(derived_stats, indent=4) + "\n", encoding="utf-8"
        )
        shutil.copy2(source_schema_path, temporary_root / "meta/jz_pin_training_schema.json")

        tasks = pd.DataFrame({"task_index": [0]}, index=[task])
        tasks.to_parquet(temporary_root / "meta/tasks.parquet")
        write_episode_metadata(
            source_episodes_dir,
            temporary_root / "meta/episodes",
            task,
            camera_keys,
        )

        make_relative_symlink(source_root / "data", temporary_root / "data")
        for camera_key in camera_keys:
            make_relative_symlink(
                source_root / "videos" / camera_key,
                temporary_root / "videos" / camera_key,
            )

        (temporary_root / VIEW_MANIFEST).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary_root.rename(output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(
        f"[jz/pi05/view] CREATED output={output_root} episodes={source_info['total_episodes']} "
        f"cameras={list(camera_keys)} excluded={list(excluded_camera_keys)} task={task!r}"
    )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    camera_keys = tuple(args.camera_keys)
    build_view(source_root, output_root, args.task, camera_keys)


if __name__ == "__main__":
    main()
