"""Opt-in JZ Pin dataset boundary for the official PI0.5 trainer.

The LeRobot training loop remains untouched.  This module only patches the two
factories imported by ``lerobot-train`` so the immutable raw18 recording is
presented as the audited model16 view and the same boundary is serialized into
the resulting checkpoint processors.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

import lerobot.scripts.lerobot_train as train_module
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.robots.jz_robot_pin_timed.training_schema import (
    JZPinRaw18ToTraining16ProcessorStep,
    JZPinTraining16ToRaw18ActionProcessorStep,
    JZPinTrainingDatasetView,
    JZPinTrainingSchema,
    TRAINING_DIM,
)

_INSTALLED = False


def _required_path(environment_name: str) -> Path:
    raw_path = os.environ.get(environment_name, "").strip()
    if not raw_path:
        raise ValueError(f"{environment_name} must point to an explicit file")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{environment_name} does not exist: {path}")
    return path


def _load_schema() -> tuple[Path, JZPinTrainingSchema]:
    path = _required_path("JZ_PI05_TRAINING_SCHEMA")
    schema = JZPinTrainingSchema.from_file(path)
    schema.ensure_trainable()
    return path, schema


def _load_model16_stats() -> tuple[Path, dict[str, dict[str, Any]]]:
    path = _required_path("JZ_PI05_MODEL16_STATS")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "jz_pi05_model16_stats":
        raise ValueError(f"Unexpected PI0.5 stats format in {path}")
    stats = payload.get("stats")
    if not isinstance(stats, Mapping):
        raise ValueError(f"PI0.5 stats file lacks a stats mapping: {path}")
    for feature_key in ("observation.state", "action"):
        feature_stats = stats.get(feature_key)
        if not isinstance(feature_stats, Mapping):
            raise ValueError(f"PI0.5 stats lacks {feature_key}")
        for stat_name in ("mean", "std", "q01", "q99"):
            values = np.asarray(feature_stats.get(stat_name), dtype=np.float64)
            if values.shape != (TRAINING_DIM,) or not np.isfinite(values).all():
                raise ValueError(
                    f"Invalid {feature_key}.{stat_name}: expected finite ({TRAINING_DIM},), got {values.shape}"
                )
    return path, dict(stats)


def _insert_schema_steps(preprocessor, postprocessor, schema: JZPinTrainingSchema) -> None:
    normalizer_index = next(
        index for index, step in enumerate(preprocessor.steps) if isinstance(step, NormalizerProcessorStep)
    )
    projection_steps = [
        step for step in preprocessor.steps if isinstance(step, JZPinRaw18ToTraining16ProcessorStep)
    ]
    if len(projection_steps) > 1:
        raise ValueError("Preprocessor contains multiple JZ Pin raw18-to-model16 steps")
    if projection_steps:
        if projection_steps[0].schema.to_dict() != schema.to_dict():
            raise ValueError("Checkpoint preprocessor schema differs from the requested dataset schema")
        if preprocessor.steps.index(projection_steps[0]) >= normalizer_index:
            raise ValueError("JZ Pin projection must run before normalization")
    else:
        preprocessor.steps.insert(
            normalizer_index,
            JZPinRaw18ToTraining16ProcessorStep(schema=schema.to_dict()),
        )

    unnormalizer_index = next(
        index for index, step in enumerate(postprocessor.steps) if isinstance(step, UnnormalizerProcessorStep)
    )
    expansion_steps = [
        step for step in postprocessor.steps if isinstance(step, JZPinTraining16ToRaw18ActionProcessorStep)
    ]
    if len(expansion_steps) > 1:
        raise ValueError("Postprocessor contains multiple JZ Pin model16-to-raw18 steps")
    if expansion_steps:
        if expansion_steps[0].schema.to_dict() != schema.to_dict():
            raise ValueError("Checkpoint postprocessor schema differs from the requested dataset schema")
        if postprocessor.steps.index(expansion_steps[0]) <= unnormalizer_index:
            raise ValueError("JZ Pin action expansion must run after unnormalization")
    else:
        postprocessor.steps.insert(
            unnormalizer_index + 1,
            JZPinTraining16ToRaw18ActionProcessorStep(schema=schema.to_dict()),
        )


def install() -> None:
    """Install the idempotent PI0.5 factory patch before training starts."""

    global _INSTALLED
    if _INSTALLED:
        return

    schema_path, schema = _load_schema()
    stats_path, model16_stats = _load_model16_stats()
    original_dataset_factory: Callable = train_module.make_dataset
    original_processor_factory: Callable = train_module.make_pre_post_processors

    def make_projected_dataset(*args, **kwargs):
        raw_dataset = original_dataset_factory(*args, **kwargs)
        dataset = JZPinTrainingDatasetView(raw_dataset, schema)
        dataset.meta.stats = model16_stats
        return dataset

    def make_processors_with_jz_boundary(*args, **kwargs):
        policy_cfg = kwargs.get("policy_cfg", args[0] if args else None)
        if getattr(policy_cfg, "type", None) != "pi05":
            raise ValueError("The JZ PI0.5 hook may only be used with policy.type=pi05")
        preprocessor, postprocessor = original_processor_factory(*args, **kwargs)
        _insert_schema_steps(preprocessor, postprocessor, schema)
        return preprocessor, postprocessor

    train_module.make_dataset = make_projected_dataset
    train_module.make_pre_post_processors = make_processors_with_jz_boundary
    _INSTALLED = True
    print(
        "[jz/pi05/hook] installed "
        f"schema={schema_path} stats={stats_path} raw18->model16 "
        f"sources={schema.observation_sources}",
        flush=True,
    )


__all__ = ["install"]
