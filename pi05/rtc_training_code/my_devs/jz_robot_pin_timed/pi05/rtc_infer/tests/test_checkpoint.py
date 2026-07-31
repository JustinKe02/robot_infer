from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from lerobot.robots.jz_robot_pin_timed.training_schema import build_training_schema_manifest
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.checkpoint import (
    EXPECTED_CAMERA_KEYS,
    inspect_checkpoint,
)


@dataclass(frozen=True)
class SyntheticCheckpoint:
    policy_path: Path
    tokenizer_path: Path

    def read_json(self, filename: str) -> dict[str, Any]:
        return json.loads((self.policy_path / filename).read_text(encoding="utf-8"))

    def write_json(self, filename: str, value: dict[str, Any]) -> None:
        (self.policy_path / filename).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _audited_manifest() -> dict[str, Any]:
    return build_training_schema_manifest(
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
        provenance={"test": "synthetic checkpoint"},
    )


def _make_checkpoint(tmp_path: Path) -> SyntheticCheckpoint:
    policy_path = tmp_path / "outputs" / "checkpoints" / "15" / "pretrained_model"
    policy_path.mkdir(parents=True)
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (tokenizer_path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")

    manifest = _audited_manifest()
    config = {
        "type": "pi05",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [16]},
            "observation.images.camera_head": {"type": "VISUAL", "shape": [3, 720, 1280]},
            "observation.images.camera_left": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.images.camera_right": {"type": "VISUAL", "shape": [3, 480, 640]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [16]}},
    }
    preprocessor = {
        "steps": [
            {
                "class": (
                    "lerobot.robots.jz_robot_pin_timed.training_schema."
                    "JZPinRaw18ToTraining16ProcessorStep"
                ),
                "config": {"schema": manifest},
            },
            {
                "registry_name": "normalizer_processor",
                "state_file": "preprocessor_normalizer.safetensors",
            },
            {"registry_name": "tokenizer_processor"},
            {"registry_name": "device_processor"},
        ]
    }
    postprocessor = {
        "steps": [
            {"registry_name": "device_processor"},
            {
                "registry_name": "unnormalizer_processor",
                "state_file": "postprocessor_unnormalizer.safetensors",
            },
            {
                "class": (
                    "lerobot.robots.jz_robot_pin_timed.training_schema."
                    "JZPinTraining16ToRaw18ActionProcessorStep"
                ),
                "config": {"schema": manifest},
            },
        ]
    }
    for filename, value in (
        ("config.json", config),
        ("policy_preprocessor.json", preprocessor),
        ("policy_postprocessor.json", postprocessor),
        ("train_config.json", {"steps": 15}),
    ):
        (policy_path / filename).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (policy_path / "model.safetensors").write_bytes(b"synthetic weights, never loaded")
    (policy_path / "preprocessor_normalizer.safetensors").write_bytes(b"preprocessor state")
    (policy_path / "postprocessor_unnormalizer.safetensors").write_bytes(b"postprocessor state")
    return SyntheticCheckpoint(policy_path=policy_path, tokenizer_path=tokenizer_path)


@pytest.fixture
def synthetic_checkpoint(tmp_path: Path) -> SyntheticCheckpoint:
    return _make_checkpoint(tmp_path)


def _inspect(checkpoint: SyntheticCheckpoint):
    return inspect_checkpoint(
        checkpoint.policy_path,
        tokenizer_path=checkpoint.tokenizer_path,
        require_complete_step=True,
    )


def _mutate_embedded_schemas(
    checkpoint: SyntheticCheckpoint,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        config = checkpoint.read_json(filename)
        schema_steps = [
            step
            for step in config["steps"]
            if "schema" in step.get("config", {})
        ]
        assert len(schema_steps) == 1
        manifest = copy.deepcopy(schema_steps[0]["config"]["schema"])
        mutate(manifest)
        schema_steps[0]["config"]["schema"] = manifest
        checkpoint.write_json(filename, config)


def test_inspect_checkpoint_accepts_complete_audited_synthetic_checkpoint(
    synthetic_checkpoint: SyntheticCheckpoint,
) -> None:
    metadata, schema = _inspect(synthetic_checkpoint)

    assert metadata.policy_type == "pi05"
    assert metadata.camera_keys == EXPECTED_CAMERA_KEYS
    assert metadata.model_state_dim == 16
    assert metadata.model_action_dim == 16
    assert metadata.processed_action_dim == 18
    assert metadata.checkpoint_step == 15
    assert metadata.configured_steps == 15
    assert metadata.complete_step is True
    assert metadata.weight_file == "model.safetensors"
    assert schema.semantic_dict() == {
        key: value for key, value in _audited_manifest().items() if key != "provenance"
    }


@pytest.mark.parametrize(
    ("feature_group", "feature_key"),
    [
        ("input_features", "observation.state"),
        ("output_features", "action"),
    ],
)
def test_inspect_checkpoint_rejects_non_model16_dimensions(
    synthetic_checkpoint: SyntheticCheckpoint,
    feature_group: str,
    feature_key: str,
) -> None:
    config = synthetic_checkpoint.read_json("config.json")
    config[feature_group][feature_key]["shape"] = [18]
    synthetic_checkpoint.write_json("config.json", config)

    with pytest.raises(ValueError, match="model16"):
        _inspect(synthetic_checkpoint)


def test_inspect_checkpoint_rejects_wrong_camera_resolution(
    synthetic_checkpoint: SyntheticCheckpoint,
) -> None:
    config = synthetic_checkpoint.read_json("config.json")
    config["input_features"]["observation.images.camera_head"]["shape"] = [3, 480, 640]
    synthetic_checkpoint.write_json("config.json", config)

    with pytest.raises(ValueError, match="camera_head"):
        _inspect(synthetic_checkpoint)


def test_inspect_checkpoint_rejects_wrong_camera_set(
    synthetic_checkpoint: SyntheticCheckpoint,
) -> None:
    config = synthetic_checkpoint.read_json("config.json")
    del config["input_features"]["observation.images.camera_right"]
    synthetic_checkpoint.write_json("config.json", config)

    with pytest.raises(ValueError, match="Checkpoint cameras"):
        _inspect(synthetic_checkpoint)


def test_inspect_checkpoint_rejects_extra_policy_features(
    synthetic_checkpoint: SyntheticCheckpoint,
) -> None:
    config = synthetic_checkpoint.read_json("config.json")
    config["input_features"]["observation.environment_state"] = {
        "type": "ENV",
        "shape": [1],
    }
    synthetic_checkpoint.write_json("config.json", config)

    with pytest.raises(ValueError, match="input feature set"):
        _inspect(synthetic_checkpoint)


def test_inspect_checkpoint_rejects_semantically_different_jz_schema(
    synthetic_checkpoint: SyntheticCheckpoint,
) -> None:
    def alter_wire_force(manifest: dict[str, Any]) -> None:
        manifest["grippers"]["left"]["wire_force"]["value"] = 79.0

    _mutate_embedded_schemas(synthetic_checkpoint, alter_wire_force)

    with pytest.raises(ValueError, match="schema|semantic|force"):
        _inspect(synthetic_checkpoint)


def test_inspect_checkpoint_rejects_schema_feature_order_change(
    synthetic_checkpoint: SyntheticCheckpoint,
) -> None:
    def swap_raw_joint_names(manifest: dict[str, Any]) -> None:
        for feature_key in ("observation.state", "action"):
            names = manifest["raw_schema"]["features"][feature_key]["names"]
            names[0], names[1] = names[1], names[0]

    _mutate_embedded_schemas(synthetic_checkpoint, swap_raw_joint_names)

    with pytest.raises(ValueError, match="names|order|schema"):
        _inspect(synthetic_checkpoint)


def test_inspect_checkpoint_rejects_projection_after_normalization(
    synthetic_checkpoint: SyntheticCheckpoint,
) -> None:
    config = synthetic_checkpoint.read_json("policy_preprocessor.json")
    config["steps"][0], config["steps"][1] = config["steps"][1], config["steps"][0]
    synthetic_checkpoint.write_json("policy_preprocessor.json", config)

    with pytest.raises(ValueError, match="before normalization"):
        _inspect(synthetic_checkpoint)


def test_inspect_checkpoint_rejects_missing_referenced_processor_state(
    synthetic_checkpoint: SyntheticCheckpoint,
) -> None:
    (synthetic_checkpoint.policy_path / "preprocessor_normalizer.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="missing state_file"):
        _inspect(synthetic_checkpoint)


def test_inspect_checkpoint_rejects_state_file_escaping_checkpoint(
    synthetic_checkpoint: SyntheticCheckpoint,
) -> None:
    config = synthetic_checkpoint.read_json("policy_preprocessor.json")
    config["steps"][1]["state_file"] = "../outside.safetensors"
    synthetic_checkpoint.write_json("policy_preprocessor.json", config)

    with pytest.raises(ValueError, match="escapes the checkpoint"):
        _inspect(synthetic_checkpoint)


def test_inspect_checkpoint_requires_configured_final_step(
    synthetic_checkpoint: SyntheticCheckpoint,
) -> None:
    synthetic_checkpoint.write_json("train_config.json", {"steps": 16})

    with pytest.raises(ValueError, match=r"15/16"):
        _inspect(synthetic_checkpoint)
