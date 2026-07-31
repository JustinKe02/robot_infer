from __future__ import annotations

import json

import pytest
import torch

from tk_infer.offline_infer import (
    parse_indices,
    resolve_device,
    validate_checkpoint,
    validate_tokenizer_path,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("first,middle,last", [0, 2, 4]),
        ("0,0,-1,2", [0, 4, 2]),
        (" last, ", [4]),
    ],
)
def test_parse_indices_supports_aliases_and_deduplicates(raw: str, expected: list[int]) -> None:
    assert parse_indices(raw, dataset_length=5) == expected


def test_parse_indices_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="Invalid sample index"):
        parse_indices("first,banana", dataset_length=5)
    with pytest.raises(ValueError, match="At least one"):
        parse_indices(",,", dataset_length=5)
    with pytest.raises(IndexError, match="outside"):
        parse_indices("5", dataset_length=5)


def test_resolve_device_auto_uses_available_backend() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == expected


def test_resolve_device_rejects_unavailable_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_resolve_device_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("not-a-device")


def test_validate_checkpoint_accepts_full_weights(tmp_path) -> None:
    config = {"input_features": {"observation.state": {"shape": [16]}}, "output_features": {}}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in (
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
        "model.safetensors",
    ):
        (tmp_path / name).write_bytes(b"{}")

    assert validate_checkpoint(tmp_path)["input_features"]["observation.state"]["shape"] == [16]


def test_validate_checkpoint_requires_weights(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"input_features": {}, "output_features": {}}), encoding="utf-8"
    )
    for name in ("policy_preprocessor.json", "policy_postprocessor.json", "train_config.json"):
        (tmp_path / name).write_bytes(b"{}")

    with pytest.raises(FileNotFoundError, match="weights"):
        validate_checkpoint(tmp_path)


def test_validate_tokenizer_path_requires_both_configs(tmp_path) -> None:
    (tmp_path / "tokenizer.json").write_bytes(b"{}")
    with pytest.raises(FileNotFoundError, match="tokenizer_config.json"):
        validate_tokenizer_path(tmp_path)

    (tmp_path / "tokenizer_config.json").write_bytes(b"{}")
    assert validate_tokenizer_path(tmp_path) == tmp_path.resolve()
