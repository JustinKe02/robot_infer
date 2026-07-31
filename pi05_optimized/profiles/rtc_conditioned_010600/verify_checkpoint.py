#!/usr/bin/env python

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def resolve_repo_root(path: Path) -> Path:
    for candidate in path.resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05.runtime.checkpoint import inspect_checkpoint  # noqa: E402
from tk_infer.pi05_optimized.backends.torch_rtc_conditioned_backend import (  # noqa: E402
    inspect_rtc_conditioned_checkpoint,
)

POLICY_PATH = (
    REPO_ROOT / "tk_infer/pi05/checkpoints/"
    "pi05_jz100_model16_head_left_right_expert_b_rtc_e10_seed1000_010600/pretrained_model"
)
TOKENIZER_PATH = REPO_ROOT / "assets/modelscope/google/paligemma-3b-pt-224"
EXPECTED_FILE_HASHES = {
    "config.json": "5da4e4e2aa4e45c52840a582c8b3d69bc260a01f3633e8102eb3891fc7ab37cc",
    "model.safetensors": "a532c9cfbb56a6feb1b9da8ec5d40bcb17d5aec0a9f14e29923ff0bf2aa7021f",
    "policy_postprocessor.json": "21e036d535326e9cac81ca5ae14504d5bf1e64b0766180f5d35be976507d2ad4",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": (
        "5f676bd3c2ab9fa2758fdec576f57ea6f964b183a6901c9ff49904dd89fe00ca"
    ),
    "policy_preprocessor.json": "227f9396f7ce5af054671ecc6d2a8de73eec8b4fd1d04120eeaf1d4ca0c6162f",
    "policy_preprocessor_step_3_normalizer_processor.safetensors": (
        "5f676bd3c2ab9fa2758fdec576f57ea6f964b183a6901c9ff49904dd89fe00ca"
    ),
    "train_config.json": "bd01c144cb125104633c3bea5ad475ef65ab34778bca162a9b0e45f88e3d6b33",
}
EXPECTED_TOKENIZER_HASHES = {
    "special_tokens_map.json": "5ef37093ae4236587b6e8266acb815b46e2db8ce656c66552bfa574d32880405",
    "tokenizer.json": "ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563",
    "tokenizer.model": "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6",
    "tokenizer_config.json": "3259402b1d1802e02417d7bff75a889ec61d359d15be6050a957b307c48edbbe",
}
EXPECTED_METADATA = {
    "checkpoint_step": None,
    "configured_steps": 10600,
    "complete_step": None,
    "checkpoint_fingerprint": "039ef411871f75e8504b7b72ccb299c29c4cdf3a99e7bfbc241a3daae7bfaa57",
    "camera_profile": "three_camera",
    "model_state_dim": 16,
    "model_action_dim": 16,
    "wire_action_dim": 18,
    "schema_id": "jz_pin_opening16_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint() -> dict[str, Any]:
    policy_path = POLICY_PATH.resolve(strict=True)
    tokenizer_path = TOKENIZER_PATH.resolve(strict=True)
    actual_hashes = verify_file_set("checkpoint", policy_path, EXPECTED_FILE_HASHES)
    actual_tokenizer_hashes = verify_file_set(
        "tokenizer",
        tokenizer_path,
        EXPECTED_TOKENIZER_HASHES,
    )

    metadata, _schema = inspect_checkpoint(
        policy_path,
        tokenizer_path=tokenizer_path,
        require_complete_step=False,
    )
    health = metadata.health_dict()
    mismatched_metadata = {
        key: {"expected": expected, "actual": health.get(key)}
        for key, expected in EXPECTED_METADATA.items()
        if health.get(key) != expected
    }
    if mismatched_metadata:
        raise RuntimeError(f"RTC-conditioned checkpoint metadata mismatch: {mismatched_metadata}")
    contract = inspect_rtc_conditioned_checkpoint(policy_path)
    return {
        "ok": True,
        "policy_path": policy_path.as_posix(),
        "file_count": len(actual_hashes),
        "tokenizer_path": tokenizer_path.as_posix(),
        "tokenizer_file_count": len(actual_tokenizer_hashes),
        "rtc_conditioning": contract.to_dict(),
        **EXPECTED_METADATA,
    }


def verify_file_set(label: str, directory: Path, expected_hashes: dict[str, str]) -> dict[str, str]:
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    expected_files = set(expected_hashes)
    if actual_files != expected_files:
        raise RuntimeError(
            f"RTC-conditioned {label} file set differs from the audited source; "
            f"missing={sorted(expected_files - actual_files)} extra={sorted(actual_files - expected_files)}"
        )
    actual_hashes = {name: sha256_file(directory / name) for name in sorted(expected_files)}
    mismatched_hashes = {
        name: {"expected": expected_hashes[name], "actual": digest}
        for name, digest in actual_hashes.items()
        if digest != expected_hashes[name]
    }
    if mismatched_hashes:
        raise RuntimeError(f"RTC-conditioned {label} SHA-256 mismatch: {mismatched_hashes}")
    return actual_hashes


def main() -> int:
    print(json.dumps(verify_checkpoint(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
