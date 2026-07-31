from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.robots.jz_robot_pin_timed.training_schema import (
    RAW_DIM,
    TRAINING_DIM,
    TRAINING_SCHEMA_ID,
    JZPinTrainingSchema,
    build_training_schema_manifest,
    load_training_schema_from_local_checkpoint,
)

from .camera_profiles import (
    DEFAULT_CAMERA_PROFILE,
    camera_feature_keys,
    camera_feature_shapes,
    infer_camera_profile,
)

EXPECTED_CAMERA_KEYS = camera_feature_keys(DEFAULT_CAMERA_PROFILE)
EXPECTED_CAMERA_SHAPES = camera_feature_shapes(DEFAULT_CAMERA_PROFILE)
EXPECTED_STATE_KEY = "observation.state"
EXPECTED_ACTION_KEY = "action"
PROCESSOR_CONFIG_FILES = ("policy_preprocessor.json", "policy_postprocessor.json")


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    policy_path: Path
    policy_type: str
    camera_profile: str
    camera_keys: tuple[str, ...]
    camera_shapes: dict[str, tuple[int, int, int]]
    model_state_dim: int
    model_action_dim: int
    wire_action_dim: int
    schema_id: str
    schema_version: int
    schema_fingerprint: str
    checkpoint_fingerprint: str
    checkpoint_step: int | None
    configured_steps: int
    complete_step: bool | None
    weight_file: str

    @property
    def processed_action_dim(self) -> int:
        """Compatibility alias; health metadata uses the unambiguous wire name."""

        return self.wire_action_dim

    def health_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.policy_path),
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "checkpoint_step": self.checkpoint_step,
            "configured_steps": self.configured_steps,
            "complete_step": self.complete_step,
            "policy_type": self.policy_type,
            "camera_profile": self.camera_profile,
            "camera_keys": list(self.camera_keys),
            "camera_shapes": {key: list(value) for key, value in self.camera_shapes.items()},
            "model_state_dim": self.model_state_dim,
            "model_action_dim": self.model_action_dim,
            "wire_action_dim": self.wire_action_dim,
            "schema_id": self.schema_id,
            "training_schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "schema_fingerprint": self.schema_fingerprint,
        }


@dataclass(slots=True)
class PolicyBundle:
    metadata: CheckpointMetadata
    policy_config: Any
    policy: Any
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]]
    postprocessor: Callable[[Any], Any]
    schema: JZPinTrainingSchema


def inspect_checkpoint(
    policy_path: str | Path,
    *,
    tokenizer_path: str | Path,
    require_complete_step: bool = True,
) -> tuple[CheckpointMetadata, JZPinTrainingSchema]:
    policy_path = Path(policy_path).expanduser().resolve(strict=True)
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy path is not a directory: {policy_path}")
    _validate_tokenizer_path(tokenizer_path)

    adapter_config = policy_path / "adapter_config.json"
    weight_file = "adapter_model.safetensors" if adapter_config.is_file() else "model.safetensors"
    required_files = (
        "config.json",
        weight_file,
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    )
    missing = [name for name in required_files if not (policy_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required files: {missing}")

    config = _load_json_object(policy_path / "config.json")
    policy_type = config.get("type")
    if policy_type != "pi05":
        raise ValueError(f"Checkpoint policy type must be 'pi05', got {policy_type!r}")
    camera_profile, camera_keys, camera_shapes = _validate_policy_features(config)

    preprocessor_config = _validate_processor_config(policy_path, "policy_preprocessor.json")
    postprocessor_config = _validate_processor_config(policy_path, "policy_postprocessor.json")
    _validate_schema_step_order(preprocessor_config, postprocessor_config)

    schema = load_training_schema_from_local_checkpoint(policy_path)
    if schema is None:
        raise ValueError("Checkpoint lacks the serialized JZ raw18-to-model16 processor boundary")
    schema_manifest = schema.to_dict()
    training_schema = schema_manifest["training_schema"]
    raw_schema = schema_manifest["raw_schema"]
    if training_schema.get("id") != TRAINING_SCHEMA_ID:
        raise ValueError(
            f"Checkpoint training schema must be {TRAINING_SCHEMA_ID!r}, got {training_schema.get('id')!r}"
        )
    if training_schema.get("dimension") != TRAINING_DIM or raw_schema.get("dimension") != RAW_DIM:
        raise ValueError("Checkpoint schema dimensions must be model16/raw18")
    schema.ensure_trainable()
    expected_schema = JZPinTrainingSchema(
        build_training_schema_manifest(
            left_observation_source="measured_opening",
            right_observation_source="commanded_opening",
            left_observation_raw_closed=0.0,
            left_observation_raw_open=100.0,
            right_observation_raw_closed=100.0,
            right_observation_raw_open=0.0,
            left_action_raw_closed=100.0,
            left_action_raw_open=0.0,
            right_action_raw_closed=100.0,
            right_action_raw_open=0.0,
            left_command_force=80.0,
            right_command_force=80.0,
        )
    )
    if schema.semantic_dict() != expected_schema.semantic_dict():
        raise ValueError("Checkpoint JZ schema semantics differ from the audited live raw18/model16 boundary")

    train_config = _load_json_object(policy_path / "train_config.json")
    configured_steps = _positive_int(train_config.get("steps"), label="train_config.steps")
    checkpoint_step = _checkpoint_step(policy_path)
    complete_step = None if checkpoint_step is None else checkpoint_step == configured_steps
    if require_complete_step and checkpoint_step is None:
        raise ValueError(
            "Cannot prove checkpoint completeness from its resolved path; expected "
            ".../checkpoints/<numeric-step>/pretrained_model"
        )
    if require_complete_step and not complete_step:
        raise ValueError(f"Checkpoint is not the configured final step: {checkpoint_step}/{configured_steps}")

    schema_fingerprint = _json_fingerprint(schema.semantic_dict())
    checkpoint_fingerprint = _checkpoint_fingerprint(
        policy_path,
        weight_file=weight_file,
        schema_fingerprint=schema_fingerprint,
    )
    metadata = CheckpointMetadata(
        policy_path=policy_path,
        policy_type=policy_type,
        camera_profile=camera_profile,
        camera_keys=camera_keys,
        camera_shapes=camera_shapes,
        model_state_dim=TRAINING_DIM,
        model_action_dim=TRAINING_DIM,
        wire_action_dim=RAW_DIM,
        schema_id=TRAINING_SCHEMA_ID,
        schema_version=int(schema_manifest["schema_version"]),
        schema_fingerprint=schema_fingerprint,
        checkpoint_fingerprint=checkpoint_fingerprint,
        checkpoint_step=checkpoint_step,
        configured_steps=configured_steps,
        complete_step=complete_step,
        weight_file=weight_file,
    )
    return metadata, schema


def _validate_tokenizer_path(tokenizer_path: str | Path) -> Path:
    raw_path = Path(tokenizer_path).expanduser()
    if not raw_path.is_absolute():
        raise ValueError(f"tokenizer_path must be absolute, got {raw_path}")
    path = raw_path.resolve(strict=True)
    if not path.is_dir():
        raise FileNotFoundError(f"Tokenizer path is not a directory: {path}")
    required = ("tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Tokenizer path is missing files: {missing}")
    return path


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _shape(feature: Any, *, key: str) -> tuple[int, ...]:
    if not isinstance(feature, Mapping):
        raise ValueError(f"Feature {key!r} must be a mapping")
    shape = feature.get("shape")
    if not isinstance(shape, list):
        raise ValueError(f"Feature {key!r} has an invalid shape: {shape!r}")
    invalid_dimension = any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
    if invalid_dimension:
        raise ValueError(f"Feature {key!r} has an invalid shape: {shape!r}")
    return tuple(shape)


def _require_feature_type(feature: Any, *, key: str, expected: str) -> None:
    if not isinstance(feature, Mapping) or feature.get("type") != expected:
        actual = feature.get("type") if isinstance(feature, Mapping) else None
        raise ValueError(f"Feature {key!r} must have type {expected}, got {actual!r}")


def _validate_policy_features(
    config: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], dict[str, tuple[int, int, int]]]:
    inputs = config.get("input_features")
    outputs = config.get("output_features")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("PI0.5 config must contain input_features and output_features mappings")
    visual_keys = tuple(sorted(key for key in inputs if key.startswith("observation.images.")))
    camera_profile = infer_camera_profile(visual_keys)
    camera_keys = camera_feature_keys(camera_profile)
    camera_shapes = camera_feature_shapes(camera_profile)
    expected_input_keys = {EXPECTED_STATE_KEY, *camera_keys}
    if set(inputs) != expected_input_keys:
        raise ValueError(
            "Checkpoint cameras/input feature set differs from the audited JZ contract; "
            f"expected={sorted(expected_input_keys)} got={sorted(inputs)}"
        )
    if set(outputs) != {EXPECTED_ACTION_KEY}:
        raise ValueError(f"Checkpoint output feature set must contain only action; got={sorted(outputs)}")
    state_shape = _shape(inputs.get(EXPECTED_STATE_KEY), key=EXPECTED_STATE_KEY)
    _require_feature_type(inputs.get(EXPECTED_STATE_KEY), key=EXPECTED_STATE_KEY, expected="STATE")
    if state_shape != (TRAINING_DIM,):
        raise ValueError(f"observation.state must be model16, got {state_shape}")

    for key in camera_keys:
        camera_shape = _shape(inputs[key], key=key)
        _require_feature_type(inputs[key], key=key, expected="VISUAL")
        expected_shape = camera_shapes[key]
        if camera_shape != expected_shape:
            raise ValueError(f"Camera feature {key!r} must be CHW {expected_shape}, got {camera_shape}")

    action_shape = _shape(outputs.get(EXPECTED_ACTION_KEY), key=EXPECTED_ACTION_KEY)
    _require_feature_type(outputs.get(EXPECTED_ACTION_KEY), key=EXPECTED_ACTION_KEY, expected="ACTION")
    if action_shape != (TRAINING_DIM,):
        raise ValueError(f"action must be model16, got {action_shape}")
    return camera_profile, camera_keys, camera_shapes


def _validate_processor_config(policy_path: Path, filename: str) -> dict[str, Any]:
    config = _load_json_object(policy_path / filename)
    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{filename} must contain a non-empty steps list")
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"{filename} step {index} must be a mapping")
        if "registry_name" not in step and "class" not in step:
            raise ValueError(f"{filename} step {index} lacks registry_name/class")
        state_file = step.get("state_file")
        if state_file is None:
            continue
        if not isinstance(state_file, str) or not state_file:
            raise ValueError(f"{filename} step {index} has an invalid state_file")
        state_path = (policy_path / state_file).resolve()
        if not state_path.is_relative_to(policy_path):
            raise ValueError(f"{filename} step {index} state_file escapes the checkpoint directory")
        if not state_path.is_file():
            raise FileNotFoundError(f"{filename} references missing state_file: {state_file}")
    return config


def _step_name(step: Mapping[str, Any]) -> str:
    value = step.get("registry_name") or step.get("class")
    return value if isinstance(value, str) else ""


def _find_step_indices(config: Mapping[str, Any], suffixes: tuple[str, ...]) -> list[int]:
    return [
        index
        for index, step in enumerate(config["steps"])
        if any(_step_name(step).endswith(suffix) for suffix in suffixes)
    ]


def _validate_schema_step_order(
    preprocessor_config: Mapping[str, Any],
    postprocessor_config: Mapping[str, Any],
) -> None:
    projection = _find_step_indices(
        preprocessor_config,
        ("JZPinRaw18ToTraining16ProcessorStep",),
    )
    normalizer = _find_step_indices(
        preprocessor_config,
        ("normalizer_processor", "NormalizerProcessorStep"),
    )
    if len(projection) != 1 or len(normalizer) != 1 or projection[0] >= normalizer[0]:
        raise ValueError("JZ raw18-to-model16 step must appear exactly once before normalization")

    expansion = _find_step_indices(
        postprocessor_config,
        ("JZPinTraining16ToRaw18ActionProcessorStep",),
    )
    unnormalizer = _find_step_indices(
        postprocessor_config,
        ("unnormalizer_processor", "UnnormalizerProcessorStep"),
    )
    if len(expansion) != 1 or len(unnormalizer) != 1 or expansion[0] <= unnormalizer[0]:
        raise ValueError("JZ model16-to-raw18 step must appear exactly once after unnormalization")


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _checkpoint_step(policy_path: Path) -> int | None:
    checkpoint_dir = policy_path.parent.name
    if not checkpoint_dir.isdecimal():
        return None
    return int(checkpoint_dir, 10)


def _json_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_fingerprint(policy_path: Path, *, weight_file: str, schema_fingerprint: str) -> str:
    digest = hashlib.sha256()
    digest.update(schema_fingerprint.encode("ascii"))
    for filename in ("config.json", *PROCESSOR_CONFIG_FILES, "train_config.json"):
        digest.update(filename.encode("utf-8"))
        digest.update((policy_path / filename).read_bytes())
    weight_path = policy_path / weight_file
    digest.update(weight_file.encode("utf-8"))
    digest.update(str(weight_path.stat().st_size).encode("ascii"))
    with weight_path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
    return digest.hexdigest()


def load_policy_bundle(
    policy_path: str | Path,
    *,
    tokenizer_path: str | Path,
    device: str,
    require_complete_step: bool = True,
) -> PolicyBundle:
    """Load a validated PI0.5 checkpoint and its serialized JZ processors."""

    metadata, schema = inspect_checkpoint(
        policy_path,
        tokenizer_path=tokenizer_path,
        require_complete_step=require_complete_step,
    )
    tokenizer_path = _validate_tokenizer_path(tokenizer_path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from lerobot import policies  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    policy_config = PreTrainedConfig.from_pretrained(
        str(metadata.policy_path),
        local_files_only=True,
    )
    if policy_config.type != "pi05":
        raise ValueError(f"Loaded policy config is not PI0.5: {policy_config.type!r}")
    policy_config.pretrained_path = metadata.policy_path
    policy_config.device = device
    policy_class = get_policy_class("pi05")
    policy = _load_policy_weights(metadata.policy_path, policy_class, policy_config)
    policy.to(device)
    policy.eval()

    preprocessor_config = _load_json_object(metadata.policy_path / "policy_preprocessor.json")
    postprocessor_config = _load_json_object(metadata.policy_path / "policy_postprocessor.json")
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(metadata.policy_path),
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        overrides=_processor_overrides(
            preprocessor_config,
            tokenizer_path=tokenizer_path,
            device=device,
        ),
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        str(metadata.policy_path),
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        overrides=_processor_overrides(
            postprocessor_config,
            tokenizer_path=None,
            # Robot-ready raw18 actions cross the HTTP boundary as CPU arrays.
            # Keeping the postprocessor on CPU also avoids retaining an extra
            # CUDA action tensor after model inference.
            device="cpu",
        ),
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return PolicyBundle(
        metadata=metadata,
        policy_config=policy_config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        schema=schema,
    )


def _load_policy_weights(policy_path: Path, policy_class: Any, policy_config: Any) -> Any:
    adapter_config_path = policy_path / "adapter_config.json"
    if not adapter_config_path.is_file():
        return policy_class.from_pretrained(
            str(policy_path),
            config=policy_config,
            local_files_only=True,
            strict=False,
        )

    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(str(policy_path), local_files_only=True)
    base_model_path = peft_config.base_model_name_or_path
    if not base_model_path:
        raise ValueError(f"Missing base_model_name_or_path in {adapter_config_path}")
    base_policy = policy_class.from_pretrained(
        base_model_path,
        config=policy_config,
        local_files_only=True,
        strict=False,
    )
    policy = PeftModel.from_pretrained(
        base_policy,
        str(policy_path),
        config=peft_config,
        local_files_only=True,
    )
    policy.config = base_policy.config
    return policy


def _processor_overrides(
    config: Mapping[str, Any],
    *,
    tokenizer_path: Path | None,
    device: str,
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    tokenizer_keys: list[str] = []
    for step in config["steps"]:
        name = _step_name(step)
        key = _step_override_key(step)
        if name == "device_processor" or name.endswith("DeviceProcessorStep"):
            overrides[key] = {"device": device}
        if name == "tokenizer_processor" or name.endswith("TokenizerProcessorStep"):
            tokenizer_keys.append(key)
    if tokenizer_path is not None:
        if len(tokenizer_keys) != 1:
            raise ValueError(
                f"PI0.5 preprocessor must contain exactly one tokenizer step, got {tokenizer_keys}"
            )
        overrides[tokenizer_keys[0]] = {"tokenizer_name": str(tokenizer_path)}
    elif tokenizer_keys:
        raise ValueError("Postprocessor unexpectedly contains a tokenizer step")
    return overrides


def _step_override_key(step: Mapping[str, Any]) -> str:
    registry_name = step.get("registry_name")
    if isinstance(registry_name, str) and registry_name:
        return registry_name
    class_name = step.get("class")
    if isinstance(class_name, str) and class_name:
        return class_name.rsplit(".", 1)[-1]
    raise ValueError("Processor step lacks an override key")


def install_policy_rtc(policy: Any, rtc_config: Any) -> None:
    """Attach one RTC processor; request mode later toggles only its enabled flag."""

    initialized = False
    for candidate in _policy_candidates(policy):
        config = getattr(candidate, "config", None)
        if config is not None and hasattr(config, "rtc_config"):
            config.rtc_config = rtc_config
        if not initialized and hasattr(candidate, "init_rtc_processor"):
            candidate.config.rtc_config = rtc_config
            candidate.init_rtc_processor()
            initialized = True
    if not initialized:
        raise TypeError("Cannot enable RTC: underlying PI05Policy was not found")


def set_policy_rtc_enabled(policy: Any, enabled: bool) -> list[tuple[Any, bool]]:
    """Set the shared RTC enabled bit and return state needed for restoration."""

    changed: list[tuple[Any, bool]] = []
    seen_configs: set[int] = set()
    for candidate in _policy_candidates(policy):
        config = getattr(candidate, "config", None)
        rtc_config = None if config is None else getattr(config, "rtc_config", None)
        if rtc_config is None or id(rtc_config) in seen_configs:
            continue
        seen_configs.add(id(rtc_config))
        previous = bool(rtc_config.enabled)
        rtc_config.enabled = bool(enabled)
        changed.append((rtc_config, previous))
    if not changed:
        raise RuntimeError("PI0.5 policy has no installed RTC configuration")
    return changed


def restore_policy_rtc_enabled(changed: list[tuple[Any, bool]]) -> None:
    for rtc_config, previous in changed:
        rtc_config.enabled = previous


def _policy_candidates(policy: Any) -> list[Any]:
    base_model = getattr(policy, "base_model", None)
    values = (
        policy,
        base_model,
        getattr(base_model, "model", None),
        getattr(policy, "model", None),
    )
    result: list[Any] = []
    seen: set[int] = set()
    for value in values:
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        result.append(value)
    return result
