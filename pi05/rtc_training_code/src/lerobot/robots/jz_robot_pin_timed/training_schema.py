#!/usr/bin/env python

"""Explicit raw-18D to model-16D schema for JZ Pin timed datasets.

The UDP and dataset contracts remain 18D. This module provides a named,
versioned projection at the training boundary and never infers a field by its
numeric position alone.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import EnvTransition, ProcessorStep, TransitionKey

RAW_SCHEMA_ID = "jz_pin_raw18_v1"
TRAINING_SCHEMA_ID = "jz_pin_opening16_v1"
TRAINING_SCHEMA_FILENAME = "jz_pin_training_schema.json"
TRAINING_SCHEMA_VERSION = 1
RAW_DIM = 18
TRAINING_DIM = 16
CANONICAL_CLOSED = 0.0
CANONICAL_OPEN = 100.0
OPENING_SOURCES = frozenset({"measured_opening", "commanded_opening", "unavailable"})

RAW_JOINT_NAMES = tuple(
    [f"left_left_joint{index}.pos" for index in range(1, 8)]
    + [f"right_right_joint{index}.pos" for index in range(1, 8)]
)
RAW_FEATURE_NAMES = (
    *RAW_JOINT_NAMES,
    "left_gripper.width",
    "left_gripper.force",
    "right_gripper.width",
    "right_gripper.force",
)
TRAINING_OBSERVATION_NAMES = (
    *RAW_JOINT_NAMES,
    "left_gripper.opening",
    "right_gripper.opening",
)
TRAINING_ACTION_NAMES = (
    *RAW_JOINT_NAMES,
    "left_gripper.target_opening",
    "right_gripper.target_opening",
)


class JZPinTrainingSchemaError(ValueError):
    """Raised when schema semantics or feature metadata are not explicit and consistent."""


def build_training_schema_manifest(
    *,
    left_observation_source: str = "unavailable",
    right_observation_source: str = "unavailable",
    left_observation_raw_closed: float = 0.0,
    left_observation_raw_open: float = 100.0,
    right_observation_raw_closed: float = 100.0,
    right_observation_raw_open: float = 0.0,
    left_action_raw_closed: float = 100.0,
    left_action_raw_open: float = 0.0,
    right_action_raw_closed: float = 100.0,
    right_action_raw_open: float = 0.0,
    left_command_force: float = 80.0,
    right_command_force: float = 80.0,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete manifest; all direction inversions are visible in its endpoints."""

    def gripper(
        side: str,
        observation_source: str,
        observation_raw_closed: float,
        observation_raw_open: float,
        action_raw_closed: float,
        action_raw_open: float,
        command_force: float,
    ) -> dict[str, Any]:
        return {
            "observation": {
                "raw_field": f"{side}_gripper.width",
                "training_field": f"{side}_gripper.opening",
                "source": observation_source,
                "raw_closed": observation_raw_closed,
                "raw_open": observation_raw_open,
            },
            "action": {
                "raw_field": f"{side}_gripper.width",
                "training_field": f"{side}_gripper.target_opening",
                "source": "commanded_opening",
                "raw_closed": action_raw_closed,
                "raw_open": action_raw_open,
            },
            "wire_force": {
                "raw_field": f"{side}_gripper.force",
                "value": command_force,
                "source": "explicit_x86_boundary_config",
            },
        }

    return {
        "format": "jz_pin_training_projection",
        "schema_version": TRAINING_SCHEMA_VERSION,
        "raw_schema": {
            "id": RAW_SCHEMA_ID,
            "dimension": RAW_DIM,
            "features": {
                "observation.state": {"names": list(RAW_FEATURE_NAMES)},
                "action": {"names": list(RAW_FEATURE_NAMES)},
            },
        },
        "training_schema": {
            "id": TRAINING_SCHEMA_ID,
            "dimension": TRAINING_DIM,
            "canonical_opening": {
                "closed": CANONICAL_CLOSED,
                "open": CANONICAL_OPEN,
                "definition": "0=closed,100=open",
            },
            "features": {
                "observation.state": {"names": list(TRAINING_OBSERVATION_NAMES)},
                "action": {"names": list(TRAINING_ACTION_NAMES)},
            },
        },
        "grippers": {
            "left": gripper(
                "left",
                left_observation_source,
                left_observation_raw_closed,
                left_observation_raw_open,
                left_action_raw_closed,
                left_action_raw_open,
                left_command_force,
            ),
            "right": gripper(
                "right",
                right_observation_source,
                right_observation_raw_closed,
                right_observation_raw_open,
                right_action_raw_closed,
                right_action_raw_open,
                right_command_force,
            ),
        },
        "provenance": dict(provenance or {}),
    }


class JZPinTrainingSchema:
    """Validated schema and projection operations for one dataset/checkpoint."""

    def __init__(self, manifest: Mapping[str, Any]):
        self._manifest = copy.deepcopy(dict(manifest))
        self._validate_manifest()

    @classmethod
    def from_file(cls, path: str | Path) -> JZPinTrainingSchema:
        with Path(path).open(encoding="utf-8") as stream:
            return cls(json.load(stream))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    def semantic_dict(self) -> dict[str, Any]:
        """Return checkpoint/dataset semantics without descriptive provenance."""

        semantics = self.to_dict()
        semantics.pop("provenance", None)
        return semantics

    @property
    def raw_feature_names(self) -> dict[str, tuple[str, ...]]:
        return {key: tuple(value["names"]) for key, value in self._manifest["raw_schema"]["features"].items()}

    @property
    def training_feature_names(self) -> dict[str, tuple[str, ...]]:
        return {
            key: tuple(value["names"]) for key, value in self._manifest["training_schema"]["features"].items()
        }

    @property
    def observation_sources(self) -> dict[str, str]:
        return {side: self._manifest["grippers"][side]["observation"]["source"] for side in ("left", "right")}

    @property
    def keep_indices(self) -> dict[str, tuple[int, ...]]:
        return {key: self._projection_indices(key) for key in ("observation.state", "action")}

    @property
    def dropped_force_indices(self) -> dict[str, tuple[int, int]]:
        result = {}
        for feature_key in ("observation.state", "action"):
            names = self.raw_feature_names[feature_key]
            result[feature_key] = tuple(
                names.index(self._manifest["grippers"][side]["wire_force"]["raw_field"])
                for side in ("left", "right")
            )
        return result

    def ensure_trainable(self) -> None:
        unavailable = [side for side, source in self.observation_sources.items() if source == "unavailable"]
        if unavailable:
            raise JZPinTrainingSchemaError(
                "Gripper observation source is unavailable for strict training: " + ", ".join(unavailable)
            )

    def validate_raw_features(self, features: Mapping[str, Mapping[str, Any]]) -> None:
        for feature_key in ("observation.state", "action"):
            if feature_key not in features:
                raise JZPinTrainingSchemaError(f"Missing raw dataset feature {feature_key!r}")
            feature = features[feature_key]
            shape = tuple(feature.get("shape", ()))
            if shape != (RAW_DIM,):
                raise JZPinTrainingSchemaError(
                    f"Raw feature {feature_key!r} must have shape [{RAW_DIM}], got {list(shape)}"
                )
            actual_names = tuple(feature.get("names", ()))
            expected_names = self.raw_feature_names[feature_key]
            if actual_names != expected_names:
                raise JZPinTrainingSchemaError(
                    f"Raw feature order mismatch for {feature_key!r}; "
                    f"expected {list(expected_names)}, got {list(actual_names)}"
                )

    def project_features(self, features: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        self.validate_raw_features(features)
        projected = copy.deepcopy(dict(features))
        for feature_key in ("observation.state", "action"):
            projected[feature_key]["shape"] = [TRAINING_DIM]
            projected[feature_key]["names"] = list(self.training_feature_names[feature_key])
        return projected

    def project_observation(
        self,
        value: Any,
        *,
        allow_projected: bool = False,
        require_available_source: bool = True,
    ) -> Any:
        if require_available_source:
            self.ensure_trainable()
        return self._project_value(value, "observation.state", allow_projected=allow_projected)

    def project_action(self, value: Any, *, allow_projected: bool = False) -> Any:
        return self._project_value(value, "action", allow_projected=allow_projected)

    def expand_action(self, value: Any) -> Any:
        if _last_dimension(value) != TRAINING_DIM:
            raise JZPinTrainingSchemaError(
                f"Model action must have last dimension {TRAINING_DIM}, got {_shape(value)}"
            )
        _require_finite_openings(value, feature_key="action", indices=(14, 15))
        output = _zeros_with_last_dimension(value, RAW_DIM)
        raw_names = self.raw_feature_names["action"]
        training_names = self.training_feature_names["action"]
        for training_index, training_name in enumerate(training_names[:14]):
            raw_index = raw_names.index(training_name)
            _assign_last(output, raw_index, _select_last(value, training_index))
        for training_index, side in ((14, "left"), (15, "right")):
            spec = self._manifest["grippers"][side]["action"]
            raw_index = raw_names.index(spec["raw_field"])
            raw_value = self._canonical_to_raw(_select_last(value, training_index), spec)
            _assign_last(output, raw_index, raw_value)
            force_spec = self._manifest["grippers"][side]["wire_force"]
            force_index = raw_names.index(force_spec["raw_field"])
            _assign_last(output, force_index, force_spec["value"])
        return output

    def project_stats(self, stats: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        projected = copy.deepcopy(dict(stats))
        for feature_key in ("observation.state", "action"):
            if feature_key not in stats:
                raise JZPinTrainingSchemaError(f"Missing statistics for {feature_key!r}")
            raw_feature_stats = stats[feature_key]
            projected_feature_stats: dict[str, Any] = {}
            for statistic_name, value in raw_feature_stats.items():
                if _last_dimension(value) != RAW_DIM:
                    projected_feature_stats[statistic_name] = _copy_value(value)
                    continue
                selected = _select_indices(value, self._projection_indices(feature_key))
                for output_index, side in ((14, "left"), (15, "right")):
                    spec = self._manifest["grippers"][side][self._modality(feature_key)]
                    a, b = self._affine(spec)
                    source_statistic = _reversed_statistic_name(statistic_name) if a < 0 else statistic_name
                    if source_statistic not in raw_feature_stats:
                        raise JZPinTrainingSchemaError(
                            f"Statistics for {feature_key!r} lack {source_statistic!r} needed by "
                            f"direction-reversed {statistic_name!r}"
                        )
                    source_value = raw_feature_stats[source_statistic]
                    raw_index = self.raw_feature_names[feature_key].index(spec["raw_field"])
                    raw_statistic_value = _select_last(source_value, raw_index)
                    if statistic_name == "std":
                        transformed = abs(a) * raw_statistic_value
                    else:
                        transformed = a * raw_statistic_value + b
                    _assign_last(selected, output_index, transformed)
                projected_feature_stats[statistic_name] = selected
            projected[feature_key] = projected_feature_stats
        return projected

    def _validate_manifest(self) -> None:
        manifest = self._manifest
        if manifest.get("format") != "jz_pin_training_projection":
            raise JZPinTrainingSchemaError("Unsupported JZ Pin training manifest format")
        if manifest.get("schema_version") != TRAINING_SCHEMA_VERSION:
            raise JZPinTrainingSchemaError(
                f"Unsupported schema_version={manifest.get('schema_version')!r}; "
                f"expected {TRAINING_SCHEMA_VERSION}"
            )
        raw_schema = manifest.get("raw_schema", {})
        training_schema = manifest.get("training_schema", {})
        if raw_schema.get("id") != RAW_SCHEMA_ID or raw_schema.get("dimension") != RAW_DIM:
            raise JZPinTrainingSchemaError("Manifest does not describe the supported JZ Pin raw18 schema")
        if (
            training_schema.get("id") != TRAINING_SCHEMA_ID
            or training_schema.get("dimension") != TRAINING_DIM
        ):
            raise JZPinTrainingSchemaError("Manifest does not describe the supported JZ Pin model16 schema")
        canonical = training_schema.get("canonical_opening", {})
        if canonical.get("closed") != CANONICAL_CLOSED or canonical.get("open") != CANONICAL_OPEN:
            raise JZPinTrainingSchemaError(
                "Canonical opening must be explicitly defined as 0=closed,100=open"
            )

        raw_features = self.raw_feature_names
        training_features = self.training_feature_names
        for feature_key in ("observation.state", "action"):
            if len(raw_features.get(feature_key, ())) != RAW_DIM:
                raise JZPinTrainingSchemaError(f"Manifest raw feature {feature_key!r} is not {RAW_DIM}D")
            if len(training_features.get(feature_key, ())) != TRAINING_DIM:
                raise JZPinTrainingSchemaError(
                    f"Manifest training feature {feature_key!r} is not {TRAINING_DIM}D"
                )
            if len(set(raw_features[feature_key])) != RAW_DIM:
                raise JZPinTrainingSchemaError(f"Manifest raw feature {feature_key!r} has duplicate names")
            if len(set(training_features[feature_key])) != TRAINING_DIM:
                raise JZPinTrainingSchemaError(
                    f"Manifest training feature {feature_key!r} has duplicate names"
                )

        if raw_features["observation.state"] != RAW_FEATURE_NAMES:
            raise JZPinTrainingSchemaError("Raw observation names do not match audited jz_pin_raw18_v1")
        if raw_features["action"] != RAW_FEATURE_NAMES:
            raise JZPinTrainingSchemaError("Raw action names do not match audited jz_pin_raw18_v1")
        if training_features["observation.state"] != TRAINING_OBSERVATION_NAMES:
            raise JZPinTrainingSchemaError("Training observation names do not match model16 schema")
        if training_features["action"] != TRAINING_ACTION_NAMES:
            raise JZPinTrainingSchemaError("Training action names do not match model16 schema")

        for side in ("left", "right"):
            gripper = manifest.get("grippers", {}).get(side)
            if not isinstance(gripper, Mapping):
                raise JZPinTrainingSchemaError(f"Missing gripper mapping for {side}")
            for modality in ("observation", "action"):
                spec = gripper.get(modality, {})
                source = spec.get("source")
                if source not in OPENING_SOURCES:
                    raise JZPinTrainingSchemaError(
                        f"Invalid {side} {modality} source {source!r}; expected one of {sorted(OPENING_SOURCES)}"
                    )
                if modality == "action" and source != "commanded_opening":
                    raise JZPinTrainingSchemaError(f"{side} action source must be commanded_opening")
                raw_field = spec.get("raw_field")
                training_field = spec.get("training_field")
                expected_raw_field = f"{side}_gripper.width"
                expected_training_field = (
                    f"{side}_gripper.opening"
                    if modality == "observation"
                    else f"{side}_gripper.target_opening"
                )
                if raw_field != expected_raw_field:
                    raise JZPinTrainingSchemaError(
                        f"{side} {modality} raw_field must be {expected_raw_field!r}, got {raw_field!r}"
                    )
                if training_field != expected_training_field:
                    raise JZPinTrainingSchemaError(
                        f"{side} {modality} training_field must be {expected_training_field!r}, "
                        f"got {training_field!r}"
                    )
                self._affine(spec)
            force_spec = gripper.get("wire_force", {})
            expected_force_field = f"{side}_gripper.force"
            if force_spec.get("raw_field") != expected_force_field:
                raise JZPinTrainingSchemaError(
                    f"Wire force raw_field for {side} must be {expected_force_field!r}"
                )
            if force_spec.get("source") != "explicit_x86_boundary_config":
                raise JZPinTrainingSchemaError(
                    f"Wire force for {side} must be explicit_x86_boundary_config, not model or feedback data"
                )
            force = force_spec.get("value")
            if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force):
                raise JZPinTrainingSchemaError(f"Wire force for {side} must be a finite number")

        expected_keep = (*range(15), 16)
        for feature_key in ("observation.state", "action"):
            if self._projection_indices(feature_key) != expected_keep:
                raise JZPinTrainingSchemaError(
                    f"Audited {feature_key} projection must keep [0..14,16] and drop force [15,17]"
                )

    def _projection_indices(self, feature_key: str) -> tuple[int, ...]:
        raw_names = self.raw_feature_names[feature_key]
        training_names = self.training_feature_names[feature_key]
        indices = [raw_names.index(name) for name in training_names[:14]]
        modality = self._modality(feature_key)
        indices.extend(
            raw_names.index(self._manifest["grippers"][side][modality]["raw_field"])
            for side in ("left", "right")
        )
        return tuple(indices)

    def _project_value(self, value: Any, feature_key: str, *, allow_projected: bool) -> Any:
        last_dimension = _last_dimension(value)
        if last_dimension == TRAINING_DIM and allow_projected:
            result = _copy_value(value)
            _require_finite_openings(result, feature_key=feature_key, indices=(14, 15))
            return result
        if last_dimension != RAW_DIM:
            raise JZPinTrainingSchemaError(
                f"Raw {feature_key} must have last dimension {RAW_DIM}, got {_shape(value)}"
            )
        projected = _select_indices(value, self._projection_indices(feature_key))
        modality = self._modality(feature_key)
        for output_index, side in ((14, "left"), (15, "right")):
            spec = self._manifest["grippers"][side][modality]
            raw_index = self.raw_feature_names[feature_key].index(spec["raw_field"])
            raw_value = _select_last(value, raw_index)
            _assign_last(projected, output_index, self._raw_to_canonical(raw_value, spec))
        _require_finite_openings(projected, feature_key=feature_key, indices=(14, 15))
        return projected

    @staticmethod
    def _modality(feature_key: str) -> str:
        return "observation" if feature_key == "observation.state" else "action"

    @staticmethod
    def _affine(spec: Mapping[str, Any]) -> tuple[float, float]:
        raw_closed = spec.get("raw_closed")
        raw_open = spec.get("raw_open")
        for name, value in (("raw_closed", raw_closed), ("raw_open", raw_open)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise JZPinTrainingSchemaError(f"{name} must be a finite number, got {value!r}")
        if raw_closed == raw_open:
            raise JZPinTrainingSchemaError("raw_closed and raw_open must differ")
        a = (CANONICAL_OPEN - CANONICAL_CLOSED) / (raw_open - raw_closed)
        b = CANONICAL_CLOSED - a * raw_closed
        return a, b

    def _raw_to_canonical(self, value: Any, spec: Mapping[str, Any]) -> Any:
        a, b = self._affine(spec)
        return a * value + b

    def _canonical_to_raw(self, value: Any, spec: Mapping[str, Any]) -> Any:
        a, b = self._affine(spec)
        return (value - b) / a


class JZPinProjectedMetadata:
    """Read-only metadata facade exposing model16 features/stats over a raw18 dataset."""

    def __init__(
        self,
        raw_metadata: Any,
        schema: JZPinTrainingSchema,
        *,
        require_stats: bool = True,
    ):
        self.raw_metadata = raw_metadata
        self.features = schema.project_features(raw_metadata.features)
        raw_stats = raw_metadata.stats
        has_vector_stats = isinstance(raw_stats, Mapping) and all(
            feature_key in raw_stats for feature_key in ("observation.state", "action")
        )
        if has_vector_stats:
            self.stats = schema.project_stats(raw_stats)
        elif require_stats:
            raise JZPinTrainingSchemaError("Raw dataset metadata lacks observation/action statistics")
        else:
            self.stats = None

    @property
    def names(self) -> dict[str, Any]:
        return {key: value.get("names") for key, value in self.features.items()}

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {key: tuple(value["shape"]) for key, value in self.features.items()}

    @property
    def image_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature["dtype"] in ("video", "image")]

    def __getattr__(self, name: str) -> Any:
        if name == "raw_metadata":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "raw_metadata"), name)


class JZPinTrainingDatasetView(torch.utils.data.Dataset):
    """Non-mutating model16 view over a raw18 LeRobot dataset."""

    def __init__(self, raw_dataset: Any, schema: JZPinTrainingSchema):
        schema.ensure_trainable()
        schema.validate_raw_features(raw_dataset.meta.features)
        self.raw_dataset = raw_dataset
        self.schema = schema
        self.meta = JZPinProjectedMetadata(raw_dataset.meta, schema)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw_item = self.raw_dataset[index]
        item = dict(raw_item)
        if "observation.state" not in item:
            raise JZPinTrainingSchemaError("Dataset item lacks observation.state; feedback is unavailable")
        if "action" not in item:
            raise JZPinTrainingSchemaError("Dataset item lacks action")
        item["observation.state"] = self.schema.project_observation(item["observation.state"])
        item["action"] = self.schema.project_action(item["action"])
        return item

    def __getattr__(self, name: str) -> Any:
        if name in {"raw_dataset", "schema", "meta"}:
            raise AttributeError(name)
        return getattr(self.raw_dataset, name)


class JZPinRaw18ToTraining16ProcessorStep(ProcessorStep):
    """Serialized preprocessor boundary accepting raw18 or a validated training-view model16 batch."""

    def __init__(self, schema: Mapping[str, Any], allow_already_projected: bool = True):
        self.schema = JZPinTrainingSchema(schema)
        self.allow_already_projected = allow_already_projected
        self.schema.ensure_trainable()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        updated = transition.copy()
        observation = updated.get(TransitionKey.OBSERVATION)
        if observation is None or "observation.state" not in observation:
            raise JZPinTrainingSchemaError("Preprocessor requires observation.state feedback")
        projected_observation = dict(observation)
        projected_observation["observation.state"] = self.schema.project_observation(
            observation["observation.state"], allow_projected=self.allow_already_projected
        )
        updated[TransitionKey.OBSERVATION] = projected_observation
        action = updated.get(TransitionKey.ACTION)
        if action is not None:
            updated[TransitionKey.ACTION] = self.schema.project_action(
                action, allow_projected=self.allow_already_projected
            )
        return updated

    def get_config(self) -> dict[str, Any]:
        return {
            "schema": self.schema.to_dict(),
            "allow_already_projected": self.allow_already_projected,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = copy.deepcopy(features)
        observation_features = transformed.get(PipelineFeatureType.OBSERVATION, {})
        if "observation.state" in observation_features:
            feature = observation_features["observation.state"]
            observation_features["observation.state"] = PolicyFeature(feature.type, (TRAINING_DIM,))
        action_features = transformed.get(PipelineFeatureType.ACTION, {})
        if "action" in action_features:
            feature = action_features["action"]
            action_features["action"] = PolicyFeature(feature.type, (TRAINING_DIM,))
        return transformed


class JZPinTraining16ToRaw18ActionProcessorStep(ProcessorStep):
    """Serialized postprocessor boundary expanding model16 action to the unchanged raw18 contract."""

    def __init__(self, schema: Mapping[str, Any]):
        self.schema = JZPinTrainingSchema(schema)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        updated = transition.copy()
        action = updated.get(TransitionKey.ACTION)
        if action is None:
            raise JZPinTrainingSchemaError("Postprocessor requires a model action")
        updated[TransitionKey.ACTION] = self.schema.expand_action(action)
        return updated

    def get_config(self) -> dict[str, Any]:
        return {"schema": self.schema.to_dict()}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = copy.deepcopy(features)
        action_features = transformed.get(PipelineFeatureType.ACTION, {})
        if "action" in action_features:
            feature = action_features["action"]
            action_features["action"] = PolicyFeature(feature.type, (RAW_DIM,))
        return transformed


def load_training_schema_from_local_checkpoint(
    pretrained_path: str | Path | None,
) -> JZPinTrainingSchema | None:
    """Return the serialized model16 schema, or None for a legacy raw18 checkpoint.

    Remote checkpoint discovery is intentionally left to the normal policy loader;
    this helper only prevents local model16 checkpoints from being reconfigured by
    raw18 recording metadata before their weights are loaded.
    """

    if pretrained_path is None:
        return None
    checkpoint_path = Path(pretrained_path).expanduser()
    if not checkpoint_path.is_dir():
        return None
    config_path = checkpoint_path / "policy_preprocessor.json"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    matching_steps = [
        step
        for step in config.get("steps", [])
        if step.get("class", "").endswith(".JZPinRaw18ToTraining16ProcessorStep")
    ]
    if not matching_steps:
        return None
    if len(matching_steps) != 1:
        raise JZPinTrainingSchemaError(
            f"Checkpoint {checkpoint_path} contains multiple raw18-to-model16 projection steps"
        )
    preprocessor_schema_config = matching_steps[0].get("config", {}).get("schema")
    if not isinstance(preprocessor_schema_config, Mapping):
        raise JZPinTrainingSchemaError(
            f"Checkpoint {checkpoint_path} projection step lacks a serialized schema"
        )
    schema = JZPinTrainingSchema(preprocessor_schema_config)
    schema.ensure_trainable()

    postprocessor_path = checkpoint_path / "policy_postprocessor.json"
    if not postprocessor_path.is_file():
        raise JZPinTrainingSchemaError(
            f"Model16 checkpoint {checkpoint_path} lacks policy_postprocessor.json"
        )
    with postprocessor_path.open(encoding="utf-8") as stream:
        postprocessor_config = json.load(stream)
    expansion_steps = [
        step
        for step in postprocessor_config.get("steps", [])
        if step.get("class", "").endswith(".JZPinTraining16ToRaw18ActionProcessorStep")
    ]
    if len(expansion_steps) != 1:
        raise JZPinTrainingSchemaError(
            f"Model16 checkpoint {checkpoint_path} must contain exactly one model16-to-raw18 step"
        )
    postprocessor_schema_config = expansion_steps[0].get("config", {}).get("schema")
    if not isinstance(postprocessor_schema_config, Mapping):
        raise JZPinTrainingSchemaError(
            f"Checkpoint {checkpoint_path} expansion step lacks a serialized schema"
        )
    if JZPinTrainingSchema(postprocessor_schema_config).to_dict() != schema.to_dict():
        raise JZPinTrainingSchemaError(
            f"Checkpoint {checkpoint_path} preprocessor/postprocessor schemas differ"
        )
    return schema


def write_training_schema_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Create a manifest once, or verify that an existing one has identical semantics."""

    schema = JZPinTrainingSchema(manifest)
    path = Path(path)
    if path.exists():
        existing = JZPinTrainingSchema.from_file(path)
        if existing.semantic_dict() != schema.semantic_dict():
            raise JZPinTrainingSchemaError(
                f"Refusing to change existing training semantics manifest at {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(schema.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(path)


def _shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(value.shape)
    return tuple(np.asarray(value).shape)


def _last_dimension(value: Any) -> int | None:
    shape = _shape(value)
    return shape[-1] if shape else None


def _copy_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _select_indices(value: Any, indices: Sequence[int]) -> Any:
    if isinstance(value, torch.Tensor):
        index = torch.tensor(indices, dtype=torch.long, device=value.device)
        return torch.index_select(value, -1, index)
    array = np.asarray(value)
    selected = np.take(array, indices, axis=-1)
    return selected.tolist() if isinstance(value, list) else selected


def _select_last(value: Any, index: int) -> Any:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value[..., index]
    return np.asarray(value)[..., index]


def _assign_last(value: Any, index: int, assigned: Any) -> None:
    if isinstance(value, list):
        array = np.asarray(value)
        array[..., index] = assigned
        value[:] = array.tolist()
        return
    value[..., index] = assigned


def _zeros_with_last_dimension(value: Any, dimension: int) -> Any:
    shape = (*_shape(value)[:-1], dimension)
    if isinstance(value, torch.Tensor):
        return torch.zeros(shape, dtype=value.dtype, device=value.device)
    if isinstance(value, np.ndarray):
        return np.zeros(shape, dtype=value.dtype)
    return np.zeros(shape, dtype=np.asarray(value).dtype).tolist()


def _require_finite_openings(value: Any, *, feature_key: str, indices: tuple[int, int]) -> None:
    for side, index in zip(("left", "right"), indices, strict=True):
        opening = _select_last(value, index)
        if isinstance(opening, torch.Tensor):
            is_finite = bool(torch.isfinite(opening).all().item())
        else:
            is_finite = bool(np.isfinite(opening).all())
        if not is_finite:
            raise JZPinTrainingSchemaError(
                f"{feature_key} {side} gripper opening is missing or non-finite; cached values are not substituted"
            )


def _reversed_statistic_name(statistic_name: str) -> str:
    return {
        "min": "max",
        "max": "min",
        "q01": "q99",
        "q10": "q90",
        "q90": "q10",
        "q99": "q01",
    }.get(statistic_name, statistic_name)


__all__ = [
    "CANONICAL_CLOSED",
    "CANONICAL_OPEN",
    "JZPinRaw18ToTraining16ProcessorStep",
    "JZPinProjectedMetadata",
    "JZPinTraining16ToRaw18ActionProcessorStep",
    "JZPinTrainingDatasetView",
    "JZPinTrainingSchema",
    "JZPinTrainingSchemaError",
    "RAW_DIM",
    "RAW_FEATURE_NAMES",
    "RAW_SCHEMA_ID",
    "TRAINING_ACTION_NAMES",
    "TRAINING_DIM",
    "TRAINING_OBSERVATION_NAMES",
    "TRAINING_SCHEMA_FILENAME",
    "TRAINING_SCHEMA_ID",
    "TRAINING_SCHEMA_VERSION",
    "build_training_schema_manifest",
    "load_training_schema_from_local_checkpoint",
    "write_training_schema_manifest",
]
