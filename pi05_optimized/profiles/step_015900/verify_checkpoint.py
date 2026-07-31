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

POLICY_PATH = REPO_ROOT / "tk_infer/pi05/checkpoints/015900/pretrained_model"
TOKENIZER_PATH = REPO_ROOT / "assets/modelscope/google/paligemma-3b-pt-224"
EXPECTED_FILE_HASHES = {
    "config.json": "d78c68f4c43d27a436f30c8adf0e8f9ff4f71696ff81edfc96578186670170ff",
    "model.safetensors": "00d75a7857fdc3eb45a4dbe6f40de90a1d789d9b1d54e876786dc0e9f908b0b9",
    "policy_postprocessor.json": "21e036d535326e9cac81ca5ae14504d5bf1e64b0766180f5d35be976507d2ad4",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": (
        "5f676bd3c2ab9fa2758fdec576f57ea6f964b183a6901c9ff49904dd89fe00ca"
    ),
    "policy_preprocessor.json": "a83195deef840e0dcc7d61423966d218b77a57d4644c93975c64d581f0097dcb",
    "policy_preprocessor_step_3_normalizer_processor.safetensors": (
        "5f676bd3c2ab9fa2758fdec576f57ea6f964b183a6901c9ff49904dd89fe00ca"
    ),
    "train_config.json": "b8656a54e7beff2c4c3e0a4569dea012f8c9bd88ecf2b871fc698c9f92fd065a",
}
EXPECTED_METADATA = {
    "checkpoint_step": 15900,
    "configured_steps": 15900,
    "complete_step": True,
    "checkpoint_fingerprint": "9d6d37f6111a034209c9bdc2899423a3258cc35070cb8294194c9c594197b58a",
    "camera_profile": "head_right",
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
    actual_files = {path.name for path in policy_path.iterdir() if path.is_file()}
    expected_files = set(EXPECTED_FILE_HASHES)
    if actual_files != expected_files:
        raise RuntimeError(
            "step_015900 file set differs from the audited source; "
            f"missing={sorted(expected_files - actual_files)} extra={sorted(actual_files - expected_files)}"
        )
    actual_hashes = {name: sha256_file(policy_path / name) for name in sorted(expected_files)}
    mismatched_hashes = {
        name: {"expected": EXPECTED_FILE_HASHES[name], "actual": digest}
        for name, digest in actual_hashes.items()
        if digest != EXPECTED_FILE_HASHES[name]
    }
    if mismatched_hashes:
        raise RuntimeError(f"step_015900 SHA-256 mismatch: {mismatched_hashes}")

    metadata, _schema = inspect_checkpoint(
        policy_path,
        tokenizer_path=TOKENIZER_PATH,
        require_complete_step=True,
    )
    health = metadata.health_dict()
    mismatched_metadata = {
        key: {"expected": expected, "actual": health.get(key)}
        for key, expected in EXPECTED_METADATA.items()
        if health.get(key) != expected
    }
    if mismatched_metadata:
        raise RuntimeError(f"step_015900 metadata mismatch: {mismatched_metadata}")
    return {
        "ok": True,
        "policy_path": policy_path.as_posix(),
        "file_count": len(actual_hashes),
        **EXPECTED_METADATA,
    }


def main() -> int:
    print(json.dumps(verify_checkpoint(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
