from __future__ import annotations

from typing import Final

from tk_infer.pi05.runtime.protocol import MODEL_ACTION_DIM, WIRE_ACTION_DIM
from tk_infer.pi05_optimized.third_party.realtime_vla_v2 import (
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
)

ARTIFACT_FORMAT: Final = "pi05_realtime_vla_v2_rtc_conditioned"
CONVERTER_VERSION: Final = 1
MANIFEST_SCHEMA_VERSION: Final = 1
KERNEL_CONTRACT: Final = "lerobot_pi05_rtc_conditioned_v1"
RTC_INFERENCE_CONTRACT: Final = "training_time_action_conditioning_v1"
NUM_VIEWS: Final = 2
CHUNK_SIZE: Final = 50
INTERNAL_ACTION_DIM: Final = 32
NUM_DENOISE_STEPS: Final = 10
CAMERA_PROFILE: Final = "head_right"
CAMERA_KEYS: Final = (
    "observation.images.camera_head",
    "observation.images.camera_right",
)
SUPPORTED_MODES: Final = ("single_step", "rtc")
FORCE_SLOT_INDICES: Final = (15, 17)
FORCE_SLOT_VALUES: Final = (80.0, 80.0)

RTC_SOURCE_TENSOR_SHAPES: Final = {
    "model.rtc_prefix_embedding.weight": (2, 1024),
    "model.rtc_token_time_mlp_in.weight": (1024, 1024),
    "model.rtc_token_time_mlp_in.bias": (1024,),
    "model.rtc_token_time_mlp_out.weight": (1024, 1024),
    "model.rtc_token_time_mlp_out.bias": (1024,),
}

RTC_OUTPUT_TENSOR_SHAPES: Final = {
    "rtc_prefix_embedding": (2, 1024),
    "rtc_token_time_mlp_in_w": (1024, 1024),
    "rtc_token_time_mlp_in_b": (1024,),
    "rtc_token_time_mlp_out_w": (1024, 1024),
    "rtc_token_time_mlp_out_b": (1024,),
}

EXPECTED_OUTPUT_SHAPES: Final = {
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
    **RTC_OUTPUT_TENSOR_SHAPES,
}


def expected_manifest_values() -> dict[str, object]:
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_format": ARTIFACT_FORMAT,
        "converter_version": CONVERTER_VERSION,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "kernel_contract": KERNEL_CONTRACT,
        "rtc_inference_contract": RTC_INFERENCE_CONTRACT,
        "dtype": "bfloat16",
        "num_views": NUM_VIEWS,
        "camera_profile": CAMERA_PROFILE,
        "camera_keys": list(CAMERA_KEYS),
        "chunk_size": CHUNK_SIZE,
        "num_denoise_steps": NUM_DENOISE_STEPS,
        "internal_action_dim": INTERNAL_ACTION_DIM,
        "exposed_model_action_dim": MODEL_ACTION_DIM,
        "wire_action_dim": WIRE_ACTION_DIM,
        "exposed_action_indices": list(range(MODEL_ACTION_DIM)),
        "force_slot_indices": list(FORCE_SLOT_INDICES),
        "force_slot_values": list(FORCE_SLOT_VALUES),
        "supported_modes": list(SUPPORTED_MODES),
        "rtc_supported": True,
        "rtc_training_enabled": True,
        "output_tensor_count": len(EXPECTED_OUTPUT_SHAPES),
        "tensor_shapes": {name: list(shape) for name, shape in sorted(EXPECTED_OUTPUT_SHAPES.items())},
    }


__all__ = [
    "ARTIFACT_FORMAT",
    "CAMERA_KEYS",
    "CAMERA_PROFILE",
    "CHUNK_SIZE",
    "CONVERTER_VERSION",
    "EXPECTED_OUTPUT_SHAPES",
    "FORCE_SLOT_INDICES",
    "FORCE_SLOT_VALUES",
    "INTERNAL_ACTION_DIM",
    "KERNEL_CONTRACT",
    "MANIFEST_SCHEMA_VERSION",
    "NUM_DENOISE_STEPS",
    "NUM_VIEWS",
    "RTC_INFERENCE_CONTRACT",
    "RTC_OUTPUT_TENSOR_SHAPES",
    "RTC_SOURCE_TENSOR_SHAPES",
    "SUPPORTED_MODES",
    "expected_manifest_values",
]
