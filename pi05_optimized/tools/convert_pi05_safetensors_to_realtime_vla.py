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
DEFAULT_POLICY_PATH = REPO_ROOT / "tk_infer/pi05/checkpoints/010600/pretrained_model"
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "assets/modelscope/google/paligemma-3b-pt-224"
DEFAULT_OUTPUT_DIR = OPTIMIZED_ROOT / "artifacts/triton/realtime_vla_b86a942"
DEFAULT_REPORT_PATH = OPTIMIZED_ROOT / "outputs/phase3_conversion.json"
UPSTREAM_REPOSITORY = "https://github.com/Dexmal/realtime-vla"
UPSTREAM_COMMIT = "b86a942a073ea241f9bd6916a705f81906f4638b"
CONVERTER_VERSION = 1
INTERNAL_ACTION_DIM = 32
EXPOSED_MODEL_ACTION_DIM = 16
NUM_VIEWS = 2
CHUNK_SIZE = 50
NUM_DENOISE_STEPS = 10

for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    if import_path.as_posix() not in sys.path:
        sys.path.insert(0, import_path.as_posix())

from tk_infer.pi05.runtime.checkpoint import inspect_checkpoint  # noqa: E402

EXPECTED_OUTPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "embedding_weight": (257152, 2048),
    "vision_patch_embedding_w": (14, 14, 3, 1152),
    "vision_patch_embedding_b": (1152,),
    "vision_position_embedding": (256, 1152),
    "vision_attn_qkv_w": (27, 1152, 3456),
    "vision_attn_qkv_b": (27, 3456),
    "vision_attn_o_w": (27, 1152, 1152),
    "vision_attn_o_b": (27, 1152),
    "vision_ffn_up_w": (27, 1152, 4304),
    "vision_ffn_up_b": (27, 4304),
    "vision_ffn_down_w": (27, 4304, 1152),
    "vision_ffn_down_b": (27, 1152),
    "vision_pre_attn_norm_w": (27, 1152),
    "vision_pre_attn_norm_b": (27, 1152),
    "vision_pre_ffn_norm_w": (27, 1152),
    "vision_pre_ffn_norm_b": (27, 1152),
    "vision_final_norm_w": (1152,),
    "vision_final_norm_b": (1152,),
    "encoder_multi_modal_projector_w": (1152, 2048),
    "encoder_multi_modal_projector_b": (2048,),
    "encoder_attn_qkv_w": (18, 2048, 2560),
    "encoder_attn_o_w": (18, 2048, 2048),
    "encoder_ffn_gate_w": (18, 2048, 16384),
    "encoder_ffn_up_w": (18, 2048, 16384),
    "encoder_ffn_down_w": (18, 16384, 2048),
    "decoder_time_embeds": (10, 1024),
    "decoder_time_mlp_in_w": (1024, 1024),
    "decoder_time_mlp_in_b": (1024,),
    "decoder_time_mlp_out_w": (1024, 1024),
    "decoder_time_mlp_out_b": (1024,),
    "decoder_action_in_proj_w": (32, 1024),
    "decoder_action_in_proj_b": (1024,),
    "decoder_pre_attn_norm_mod_w": (18, 1024, 3072),
    "decoder_pre_attn_norm_mod_b": (18, 3072),
    "decoder_pre_ffn_norm_mod_w": (18, 1024, 3072),
    "decoder_pre_ffn_norm_mod_b": (18, 3072),
    "decoder_attn_qkv_w": (18, 1024, 2560),
    "decoder_attn_o_w": (18, 2048, 1024),
    "decoder_ffn_gate_w": (18, 1024, 4096),
    "decoder_ffn_up_w": (18, 1024, 4096),
    "decoder_ffn_down_w": (18, 4096, 1024),
    "decoder_action_out_proj_w": (1024, 32),
    "decoder_action_out_proj_b": (32,),
    "decoder_final_norm_mod_w": (1024, 3072),
    "decoder_final_norm_mod_b": (3072,),
    "language_embeds": (1, 2048),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the audited local PI0.5 safetensors to Realtime-VLA v1 fused safetensors."
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--require-complete-step", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def validate_source(args: argparse.Namespace) -> dict[str, Any]:
    policy_path = args.policy_path.expanduser().resolve(strict=True)
    tokenizer_path = args.tokenizer_path.expanduser().resolve(strict=True)
    metadata, schema = inspect_checkpoint(
        policy_path,
        tokenizer_path=tokenizer_path,
        require_complete_step=args.require_complete_step,
    )
    config = json.loads((policy_path / "config.json").read_text(encoding="utf-8"))
    expected_config = {
        "type": "pi05",
        "dtype": "bfloat16",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "chunk_size": CHUNK_SIZE,
        "max_action_dim": INTERNAL_ACTION_DIM,
        "num_inference_steps": NUM_DENOISE_STEPS,
        "tokenizer_max_length": 200,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"checkpoint is incompatible with the pinned Triton specialization: {mismatches}")
    if metadata.camera_profile != "head_right" or len(metadata.camera_keys) != NUM_VIEWS:
        raise ValueError(
            f"Triton specialization requires two-view head_right, got {metadata.camera_profile}"
        )
    output_features = config.get("output_features", {})
    action_shape = output_features.get("action", {}).get("shape")
    if action_shape != [EXPOSED_MODEL_ACTION_DIM]:
        raise ValueError(f"checkpoint exposed action must be model16, got {action_shape}")

    source_path = policy_path / metadata.weight_file
    required_shapes = _required_source_shapes()
    with safe_open(source_path, framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        missing = sorted(set(required_shapes) - source_keys)
        if missing:
            raise ValueError(f"source safetensors lacks required keys: {missing[:10]}")
        mismatched_shapes = {}
        mismatched_dtypes = {}
        source_dtype_counts: dict[str, int] = {}
        for key, expected_shape in required_shapes.items():
            tensor_slice = source.get_slice(key)
            actual_shape = tuple(tensor_slice.get_shape())
            if actual_shape != expected_shape:
                mismatched_shapes[key] = {"expected": expected_shape, "actual": actual_shape}
            source_dtype = tensor_slice.get_dtype()
            source_dtype_counts[source_dtype] = source_dtype_counts.get(source_dtype, 0) + 1
            if source_dtype not in {"BF16", "F32"}:
                mismatched_dtypes[key] = source_dtype
        if mismatched_shapes:
            raise ValueError(f"source tensor shape mismatches: {mismatched_shapes}")
        if mismatched_dtypes:
            raise ValueError(f"source tensors must be BF16 or F32: {mismatched_dtypes}")
        source_tensor_count = len(source_keys)
    action_names = schema.to_dict()["training_schema"]["features"]["action"]["names"]
    if len(action_names) != EXPOSED_MODEL_ACTION_DIM:
        raise ValueError("training schema action names do not prove the first 16 output dimensions")
    return {
        "status": "PASS",
        "policy_path": str(policy_path),
        "source_path": str(source_path),
        "tokenizer_path": str(tokenizer_path),
        "checkpoint_fingerprint": metadata.checkpoint_fingerprint,
        "checkpoint_step": metadata.checkpoint_step,
        "configured_steps": metadata.configured_steps,
        "complete_step": metadata.complete_step,
        "camera_profile": metadata.camera_profile,
        "camera_keys": list(metadata.camera_keys),
        "source_tensor_count": source_tensor_count,
        "required_source_tensor_count": len(required_shapes),
        "required_source_dtype_counts": source_dtype_counts,
        "internal_action_dim": INTERNAL_ACTION_DIM,
        "exposed_model_action_dim": EXPOSED_MODEL_ACTION_DIM,
        "exposed_action_indices": list(range(EXPOSED_MODEL_ACTION_DIM)),
        "exposed_action_names": action_names,
        "action_mapping_proof": (
            "PI05Policy.predict_action_chunk slices the internal 32-dimensional model output to the "
            "checkpoint output_features action dimension 16; the serialized training schema supplies the "
            "ordered names for indices 0..15"
        ),
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    validation = validate_source(args)
    if args.validate_only:
        return {**validation, "validate_only": True, "output_written": False}
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_relative_to(OPTIMIZED_ROOT.resolve()):
        raise ValueError(f"output-dir must stay inside {OPTIMIZED_ROOT}, got {output_dir}")
    output_path = output_dir / "model.safetensors"
    manifest_path = output_dir / "manifest.json"
    if not args.force and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing Triton artifact in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(validation["source_path"])
    source_sha256 = _sha256_file(source_path)
    with safe_open(source_path, framework="pt", device="cpu") as source:
        tensors = _convert_tensors(source)
    _validate_output_tensors(tensors)
    temporary_output = output_dir / ".model.safetensors.tmp"
    temporary_manifest = output_dir / ".manifest.json.tmp"
    for temporary in (temporary_output, temporary_manifest):
        if temporary.exists():
            temporary.unlink()
    try:
        save_file(
            tensors,
            temporary_output,
            metadata={
                "format": "pi05_realtime_vla_v1_fused",
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "converter_version": str(CONVERTER_VERSION),
                "source_sha256": source_sha256,
                "checkpoint_fingerprint": str(validation["checkpoint_fingerprint"]),
            },
        )
        output_sha256 = _sha256_file(temporary_output)
        manifest = {
            **validation,
            "validate_only": False,
            "output_written": True,
            "manifest_schema_version": 1,
            "converter_version": CONVERTER_VERSION,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "source_sha256": source_sha256,
            "output_sha256": output_sha256,
            "output_path": str(output_path),
            "output_bytes": temporary_output.stat().st_size,
            "output_tensor_count": len(tensors),
            "dtype": "bfloat16",
            "num_views": NUM_VIEWS,
            "chunk_size": CHUNK_SIZE,
            "num_denoise_steps": NUM_DENOISE_STEPS,
            "supported_modes": ["single_step"],
            "rtc_supported": False,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "tensor_shapes": {key: list(value.shape) for key, value in sorted(tensors.items())},
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


def _convert_tensors(source: Any) -> dict[str, torch.Tensor]:
    get = source.get_tensor
    vision = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    language = "model.paligemma_with_expert.paligemma.model.language_model"
    expert = "model.paligemma_with_expert.gemma_expert.model"
    tensors: dict[str, torch.Tensor] = {
        "embedding_weight": _bf16(get("model.paligemma_with_expert.paligemma.lm_head.weight")),
        "vision_patch_embedding_w": _bf16(
            get(f"{vision}.embeddings.patch_embedding.weight").permute(2, 3, 1, 0)
        ),
        "vision_patch_embedding_b": _bf16(get(f"{vision}.embeddings.patch_embedding.bias")),
        "vision_position_embedding": _bf16(get(f"{vision}.embeddings.position_embedding.weight")),
        "vision_attn_qkv_w": _stack_vision_qkv(get, vision, weight=True),
        "vision_attn_qkv_b": _stack_vision_qkv(get, vision, weight=False),
        "vision_attn_o_w": _stack_transposed(
            get, f"{vision}.encoder.layers.{{}}.self_attn.out_proj.weight", 27
        ),
        "vision_attn_o_b": _stack(get, f"{vision}.encoder.layers.{{}}.self_attn.out_proj.bias", 27),
        "vision_ffn_up_w": _stack_transposed(get, f"{vision}.encoder.layers.{{}}.mlp.fc1.weight", 27),
        "vision_ffn_up_b": _stack(get, f"{vision}.encoder.layers.{{}}.mlp.fc1.bias", 27),
        "vision_ffn_down_w": _stack_transposed(
            get, f"{vision}.encoder.layers.{{}}.mlp.fc2.weight", 27
        ),
        "vision_ffn_down_b": _stack(get, f"{vision}.encoder.layers.{{}}.mlp.fc2.bias", 27),
        "vision_pre_attn_norm_w": _stack(
            get, f"{vision}.encoder.layers.{{}}.layer_norm1.weight", 27
        ),
        "vision_pre_attn_norm_b": _stack(
            get, f"{vision}.encoder.layers.{{}}.layer_norm1.bias", 27
        ),
        "vision_pre_ffn_norm_w": _stack(
            get, f"{vision}.encoder.layers.{{}}.layer_norm2.weight", 27
        ),
        "vision_pre_ffn_norm_b": _stack(
            get, f"{vision}.encoder.layers.{{}}.layer_norm2.bias", 27
        ),
        "vision_final_norm_w": _bf16(get(f"{vision}.post_layernorm.weight")),
        "vision_final_norm_b": _bf16(get(f"{vision}.post_layernorm.bias")),
        "encoder_multi_modal_projector_w": _bf16(
            get(
                "model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
            ).T
        ),
        "encoder_multi_modal_projector_b": _bf16(
            get("model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias")
        ),
        "encoder_attn_qkv_w": _stack_encoder_qkv(get, language),
        "encoder_attn_o_w": _stack_transposed(
            get, f"{language}.layers.{{}}.self_attn.o_proj.weight", 18
        ),
        "encoder_ffn_gate_w": _stack_encoder_mlp(get, language, "gate_proj"),
        "encoder_ffn_up_w": _stack_encoder_mlp(get, language, "up_proj"),
        "encoder_ffn_down_w": _stack_transposed(
            get, f"{language}.layers.{{}}.mlp.down_proj.weight", 18
        ),
        "decoder_time_embeds": _time_embeddings(),
        "decoder_time_mlp_in_w": _bf16(get("model.time_mlp_in.weight").T),
        "decoder_time_mlp_in_b": _bf16(get("model.time_mlp_in.bias")),
        "decoder_time_mlp_out_w": _bf16(get("model.time_mlp_out.weight").T),
        "decoder_time_mlp_out_b": _bf16(get("model.time_mlp_out.bias")),
        "decoder_action_in_proj_w": _bf16(get("model.action_in_proj.weight").T),
        "decoder_action_in_proj_b": _bf16(get("model.action_in_proj.bias")),
        "decoder_pre_attn_norm_mod_w": _stack_transposed(
            get, f"{expert}.layers.{{}}.input_layernorm.dense.weight", 18
        ),
        "decoder_pre_attn_norm_mod_b": _stack(
            get, f"{expert}.layers.{{}}.input_layernorm.dense.bias", 18
        ),
        "decoder_pre_ffn_norm_mod_w": _stack_transposed(
            get, f"{expert}.layers.{{}}.post_attention_layernorm.dense.weight", 18
        ),
        "decoder_pre_ffn_norm_mod_b": _stack(
            get, f"{expert}.layers.{{}}.post_attention_layernorm.dense.bias", 18
        ),
        "decoder_attn_qkv_w": _stack_decoder_qkv(get, expert),
        "decoder_attn_o_w": _stack_transposed(
            get, f"{expert}.layers.{{}}.self_attn.o_proj.weight", 18
        ),
        "decoder_ffn_gate_w": _stack_transposed(
            get, f"{expert}.layers.{{}}.mlp.gate_proj.weight", 18
        ),
        "decoder_ffn_up_w": _stack_transposed(
            get, f"{expert}.layers.{{}}.mlp.up_proj.weight", 18
        ),
        "decoder_ffn_down_w": _stack_transposed(
            get, f"{expert}.layers.{{}}.mlp.down_proj.weight", 18
        ),
        "decoder_action_out_proj_w": _bf16(get("model.action_out_proj.weight").T),
        "decoder_action_out_proj_b": _bf16(get("model.action_out_proj.bias")),
        "decoder_final_norm_mod_w": _bf16(get(f"{expert}.norm.dense.weight").T),
        "decoder_final_norm_mod_b": _bf16(get(f"{expert}.norm.dense.bias")),
        "language_embeds": torch.zeros(1, 2048, dtype=torch.bfloat16),
    }
    return tensors


def _stack_vision_qkv(get: Any, prefix: str, *, weight: bool) -> torch.Tensor:
    suffix = "weight" if weight else "bias"
    values = []
    for layer in range(27):
        projections = [
            get(f"{prefix}.encoder.layers.{layer}.self_attn.{name}_proj.{suffix}")
            for name in ("q", "k", "v")
        ]
        if weight:
            projections = [value.T for value in projections]
        values.append(torch.cat(projections, dim=-1))
    return _bf16(torch.stack(values))


def _stack_encoder_qkv(get: Any, prefix: str) -> torch.Tensor:
    values = []
    for layer in range(18):
        layer_prefix = f"{prefix}.layers.{layer}"
        scale = 1.0 + get(f"{layer_prefix}.input_layernorm.weight").to(torch.float32)
        q = get(f"{layer_prefix}.self_attn.q_proj.weight").to(torch.float32).T * scale[:, None]
        k = get(f"{layer_prefix}.self_attn.k_proj.weight").to(torch.float32).T * scale[:, None]
        v = get(f"{layer_prefix}.self_attn.v_proj.weight").to(torch.float32).T * scale[:, None]
        values.append(torch.cat((_interleave_rope(q, 8), _interleave_rope(k, 1), v), dim=1))
    return _bf16(torch.stack(values))


def _stack_encoder_mlp(get: Any, prefix: str, projection: str) -> torch.Tensor:
    values = []
    for layer in range(18):
        layer_prefix = f"{prefix}.layers.{layer}"
        scale = 1.0 + get(f"{layer_prefix}.post_attention_layernorm.weight").to(torch.float32)
        weight = get(f"{layer_prefix}.mlp.{projection}.weight").to(torch.float32).T
        values.append(weight * scale[:, None])
    return _bf16(torch.stack(values))


def _stack_decoder_qkv(get: Any, prefix: str) -> torch.Tensor:
    values = []
    for layer in range(18):
        layer_prefix = f"{prefix}.layers.{layer}.self_attn"
        q = get(f"{layer_prefix}.q_proj.weight").T
        k = get(f"{layer_prefix}.k_proj.weight").T
        v = get(f"{layer_prefix}.v_proj.weight").T
        values.append(torch.cat((_interleave_rope(q, 8), _interleave_rope(k, 1), v), dim=1))
    return _bf16(torch.stack(values))


def _interleave_rope(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    input_dim, output_dim = weight.shape
    expected_output = num_heads * 256
    if output_dim != expected_output:
        raise ValueError(f"RoPE weight expected output {expected_output}, got {tuple(weight.shape)}")
    return weight.reshape(input_dim, num_heads, 2, 128).permute(0, 1, 3, 2).reshape(input_dim, output_dim)


def _stack(get: Any, template: str, layers: int) -> torch.Tensor:
    return _bf16(torch.stack([get(template.format(layer)) for layer in range(layers)]))


def _stack_transposed(get: Any, template: str, layers: int) -> torch.Tensor:
    return _bf16(torch.stack([get(template.format(layer)).T for layer in range(layers)]))


def _time_embeddings() -> torch.Tensor:
    fraction = torch.linspace(0.0, 1.0, 512, dtype=torch.float32)
    period = 0.004 * (4.0 / 0.004) ** fraction
    values = []
    for step in range(NUM_DENOISE_STEPS):
        time_value = 1.0 - step / NUM_DENOISE_STEPS
        sinusoid = torch.tensor(time_value, dtype=torch.float32).unsqueeze(-1) * (1.0 / period) * 2 * torch.pi
        values.append(torch.cat((torch.sin(sinusoid), torch.cos(sinusoid)), dim=-1))
    return _bf16(torch.stack(values))


def _bf16(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()


def _validate_output_tensors(tensors: dict[str, torch.Tensor]) -> None:
    if set(tensors) != set(EXPECTED_OUTPUT_SHAPES):
        missing = sorted(set(EXPECTED_OUTPUT_SHAPES) - set(tensors))
        extra = sorted(set(tensors) - set(EXPECTED_OUTPUT_SHAPES))
        raise ValueError(f"converted tensor key mismatch: missing={missing}, extra={extra}")
    errors = {}
    for key, expected_shape in EXPECTED_OUTPUT_SHAPES.items():
        tensor = tensors[key]
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            errors[key] = {
                "expected_shape": expected_shape,
                "actual_shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "contiguous": tensor.is_contiguous(),
            }
    if errors:
        raise ValueError(f"converted tensor contract failed: {errors}")


def _required_source_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.paligemma_with_expert.paligemma.lm_head.weight": (257152, 2048),
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight": (
            1152,
            3,
            14,
            14,
        ),
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias": (
            1152,
        ),
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight": (
            256,
            1152,
        ),
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight": (
            1152,
        ),
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias": (
            1152,
        ),
        "model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight": (
            2048,
            1152,
        ),
        "model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias": (2048,),
        "model.action_in_proj.weight": (1024, 32),
        "model.action_in_proj.bias": (1024,),
        "model.action_out_proj.weight": (32, 1024),
        "model.action_out_proj.bias": (32,),
        "model.time_mlp_in.weight": (1024, 1024),
        "model.time_mlp_in.bias": (1024,),
        "model.time_mlp_out.weight": (1024, 1024),
        "model.time_mlp_out.bias": (1024,),
    }
    vision = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers"
    for layer in range(27):
        prefix = f"{vision}.{layer}"
        for projection in ("q", "k", "v"):
            shapes[f"{prefix}.self_attn.{projection}_proj.weight"] = (1152, 1152)
            shapes[f"{prefix}.self_attn.{projection}_proj.bias"] = (1152,)
        shapes[f"{prefix}.self_attn.out_proj.weight"] = (1152, 1152)
        shapes[f"{prefix}.self_attn.out_proj.bias"] = (1152,)
        shapes[f"{prefix}.mlp.fc1.weight"] = (4304, 1152)
        shapes[f"{prefix}.mlp.fc1.bias"] = (4304,)
        shapes[f"{prefix}.mlp.fc2.weight"] = (1152, 4304)
        shapes[f"{prefix}.mlp.fc2.bias"] = (1152,)
        shapes[f"{prefix}.layer_norm1.weight"] = (1152,)
        shapes[f"{prefix}.layer_norm1.bias"] = (1152,)
        shapes[f"{prefix}.layer_norm2.weight"] = (1152,)
        shapes[f"{prefix}.layer_norm2.bias"] = (1152,)
    language = "model.paligemma_with_expert.paligemma.model.language_model.layers"
    expert = "model.paligemma_with_expert.gemma_expert.model.layers"
    for layer in range(18):
        language_prefix = f"{language}.{layer}"
        shapes[f"{language_prefix}.input_layernorm.weight"] = (2048,)
        shapes[f"{language_prefix}.post_attention_layernorm.weight"] = (2048,)
        shapes[f"{language_prefix}.self_attn.q_proj.weight"] = (2048, 2048)
        shapes[f"{language_prefix}.self_attn.k_proj.weight"] = (256, 2048)
        shapes[f"{language_prefix}.self_attn.v_proj.weight"] = (256, 2048)
        shapes[f"{language_prefix}.self_attn.o_proj.weight"] = (2048, 2048)
        shapes[f"{language_prefix}.mlp.gate_proj.weight"] = (16384, 2048)
        shapes[f"{language_prefix}.mlp.up_proj.weight"] = (16384, 2048)
        shapes[f"{language_prefix}.mlp.down_proj.weight"] = (2048, 16384)
        expert_prefix = f"{expert}.{layer}"
        shapes[f"{expert_prefix}.input_layernorm.dense.weight"] = (3072, 1024)
        shapes[f"{expert_prefix}.input_layernorm.dense.bias"] = (3072,)
        shapes[f"{expert_prefix}.post_attention_layernorm.dense.weight"] = (3072, 1024)
        shapes[f"{expert_prefix}.post_attention_layernorm.dense.bias"] = (3072,)
        shapes[f"{expert_prefix}.self_attn.q_proj.weight"] = (2048, 1024)
        shapes[f"{expert_prefix}.self_attn.k_proj.weight"] = (256, 1024)
        shapes[f"{expert_prefix}.self_attn.v_proj.weight"] = (256, 1024)
        shapes[f"{expert_prefix}.self_attn.o_proj.weight"] = (1024, 2048)
        shapes[f"{expert_prefix}.mlp.gate_proj.weight"] = (4096, 1024)
        shapes[f"{expert_prefix}.mlp.up_proj.weight"] = (4096, 1024)
        shapes[f"{expert_prefix}.mlp.down_proj.weight"] = (1024, 4096)
    shapes["model.paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = (3072, 1024)
    shapes["model.paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = (3072,)
    return shapes


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
    report = convert(args)
    report_path = _write_report(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
