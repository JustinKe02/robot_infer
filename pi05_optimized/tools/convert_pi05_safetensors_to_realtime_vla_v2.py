#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def resolve_repo_root(script_path: Path) -> Path:
    resolved = script_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


REPO_ROOT = resolve_repo_root(Path(__file__))
OPTIMIZED_ROOT = REPO_ROOT / "tk_infer/pi05_optimized"
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "assets/modelscope/google/paligemma-3b-pt-224"
DEFAULT_OUTPUT_DIR = OPTIMIZED_ROOT / "artifacts/realtime_vla_v2/pending_checkpoint"
DEFAULT_REPORT_PATH = OPTIMIZED_ROOT / "outputs/realtime_vla_v2_conversion.json"

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05.runtime.protocol import WIRE_ACTION_DIM  # noqa: E402
from tk_infer.pi05_optimized.backends.torch_rtc_conditioned_backend import (  # noqa: E402
    inspect_rtc_conditioned_checkpoint,
)
from tk_infer.pi05_optimized.runtime.realtime_vla_v2_contract import (  # noqa: E402
    ARTIFACT_FORMAT,
    CONVERTER_VERSION,
    EXPECTED_OUTPUT_SHAPES,
    FORCE_SLOT_INDICES,
    FORCE_SLOT_VALUES,
    KERNEL_CONTRACT,
    MANIFEST_SCHEMA_VERSION,
    RTC_INFERENCE_CONTRACT,
    RTC_SOURCE_TENSOR_SHAPES,
    expected_manifest_values,
)
from tk_infer.pi05_optimized.third_party.realtime_vla_v2 import (  # noqa: E402
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
)
from tk_infer.pi05_optimized.tools.convert_pi05_safetensors_to_realtime_vla import (  # noqa: E402
    _bf16,
    _convert_tensors,
    validate_source as validate_v1_source,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a local training-time RTC PI0.5 checkpoint to the pinned Realtime-VLA v2 "
            "safetensors contract."
        )
    )
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--rtc-conditioned-task", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--require-complete-step",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def validate_source(args: argparse.Namespace) -> dict[str, Any]:
    task = args.rtc_conditioned_task.strip()
    if not task:
        raise ValueError("rtc-conditioned-task must be non-empty")
    validation = validate_v1_source(args)
    policy_path = Path(validation["policy_path"])
    contract = inspect_rtc_conditioned_checkpoint(policy_path)
    source_path = Path(validation["source_path"])
    with safe_open(source_path, framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        missing = sorted(set(RTC_SOURCE_TENSOR_SHAPES) - source_keys)
        if missing:
            raise ValueError(f"RTC checkpoint lacks required learned tensors: {missing}")
        shape_errors = {}
        dtype_errors = {}
        for name, expected_shape in RTC_SOURCE_TENSOR_SHAPES.items():
            tensor_slice = source.get_slice(name)
            actual_shape = tuple(tensor_slice.get_shape())
            if actual_shape != expected_shape:
                shape_errors[name] = {"expected": expected_shape, "actual": actual_shape}
            dtype = tensor_slice.get_dtype()
            if dtype not in {"BF16", "F32"}:
                dtype_errors[name] = dtype
        if shape_errors:
            raise ValueError(f"RTC learned tensor shape mismatches: {shape_errors}")
        if dtype_errors:
            raise ValueError(f"RTC learned tensors must be BF16 or F32: {dtype_errors}")
    return {
        **validation,
        "rtc_conditioned_task": task,
        "rtc_training": contract.to_dict(),
        "rtc_training_enabled": True,
        "rtc_inference_contract": RTC_INFERENCE_CONTRACT,
        "required_rtc_source_tensors": sorted(RTC_SOURCE_TENSOR_SHAPES),
        "wire_action_dim": WIRE_ACTION_DIM,
        "force_slot_indices": list(FORCE_SLOT_INDICES),
        "force_slot_values": list(FORCE_SLOT_VALUES),
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    validation = validate_source(args)
    if args.validate_only:
        return {
            **validation,
            "artifact_format": ARTIFACT_FORMAT,
            "kernel_contract": KERNEL_CONTRACT,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "validate_only": True,
            "output_written": False,
        }

    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-dir must stay inside {OPTIMIZED_ROOT}, got {output_dir}")
    output_path = output_dir / "model.safetensors"
    manifest_path = output_dir / "manifest.json"
    if not args.force and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing v2 artifact in {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(validation["source_path"])
    source_sha256 = _sha256_file(source_path)
    with safe_open(source_path, framework="pt", device="cpu") as source:
        tensors = _convert_tensors(source)
        tensors.update(_convert_rtc_tensors(source))
    _validate_output_tensors(tensors)

    temporary_output = output_dir / ".model.safetensors.tmp"
    temporary_manifest = output_dir / ".manifest.json.tmp"
    for temporary in (temporary_output, temporary_manifest):
        temporary.unlink(missing_ok=True)
    try:
        header = {
            "format": ARTIFACT_FORMAT,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "kernel_contract": KERNEL_CONTRACT,
            "converter_version": str(CONVERTER_VERSION),
            "source_sha256": source_sha256,
            "checkpoint_fingerprint": str(validation["checkpoint_fingerprint"]),
        }
        save_file(tensors, temporary_output, metadata=header)
        output_sha256 = _sha256_file(temporary_output)
        manifest = {
            **validation,
            **expected_manifest_values(),
            "validate_only": False,
            "output_written": True,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "output_sha256": output_sha256,
            "output_path": str(output_path),
            "output_bytes": temporary_output.stat().st_size,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_output, output_path)
        os.replace(temporary_manifest, manifest_path)
        return manifest
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


def _convert_rtc_tensors(source: Any) -> dict[str, torch.Tensor]:
    get = source.get_tensor
    return {
        "rtc_prefix_embedding": _bf16(get("model.rtc_prefix_embedding.weight")),
        "rtc_token_time_mlp_in_w": _bf16(get("model.rtc_token_time_mlp_in.weight").T),
        "rtc_token_time_mlp_in_b": _bf16(get("model.rtc_token_time_mlp_in.bias")),
        "rtc_token_time_mlp_out_w": _bf16(get("model.rtc_token_time_mlp_out.weight").T),
        "rtc_token_time_mlp_out_b": _bf16(get("model.rtc_token_time_mlp_out.bias")),
    }


def _validate_output_tensors(tensors: dict[str, torch.Tensor]) -> None:
    if set(tensors) != set(EXPECTED_OUTPUT_SHAPES):
        missing = sorted(set(EXPECTED_OUTPUT_SHAPES) - set(tensors))
        extra = sorted(set(tensors) - set(EXPECTED_OUTPUT_SHAPES))
        raise ValueError(f"converted v2 tensor key mismatch: missing={missing}, extra={extra}")
    errors = {}
    for name, expected_shape in EXPECTED_OUTPUT_SHAPES.items():
        tensor = tensors[name]
        if (
            tuple(tensor.shape) != expected_shape
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
        ):
            errors[name] = {
                "expected_shape": expected_shape,
                "actual_shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "contiguous": tensor.is_contiguous(),
            }
    if errors:
        raise ValueError(f"converted v2 tensor contract failed: {errors}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"report-json must stay inside {OPTIMIZED_ROOT}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = convert(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "policy_path": str(args.policy_path.expanduser()),
            "rtc_conditioned_task": args.rtc_conditioned_task.strip(),
            "require_complete_step": args.require_complete_step,
            "artifact_format": ARTIFACT_FORMAT,
            "kernel_contract": KERNEL_CONTRACT,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "validate_only": args.validate_only,
            "output_written": False,
        }
        report_path = _write_report(args.report_json, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"report={report_path}")
        return 2
    report_path = _write_report(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
