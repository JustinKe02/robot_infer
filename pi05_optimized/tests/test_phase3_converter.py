from __future__ import annotations

import torch

from tk_infer.pi05_optimized.tools.convert_pi05_safetensors_to_realtime_vla import (
    EXPECTED_OUTPUT_SHAPES,
    _interleave_rope,
    _required_source_shapes,
    _time_embeddings,
)


def test_rope_conversion_interleaves_half_dimensions_per_head() -> None:
    weight = torch.arange(2 * 256, dtype=torch.float32).reshape(2, 256)

    actual = _interleave_rope(weight, num_heads=1)

    assert actual.shape == weight.shape
    assert actual[0, :6].tolist() == [0, 128, 1, 129, 2, 130]
    assert actual[1, :6].tolist() == [256, 384, 257, 385, 258, 386]


def test_time_embeddings_match_pinned_ten_step_bfloat16_contract() -> None:
    embeddings = _time_embeddings()

    assert embeddings.shape == (10, 1024)
    assert embeddings.dtype == torch.bfloat16
    assert torch.isfinite(embeddings).all()


def test_converter_contract_has_only_expected_specialized_shapes() -> None:
    required = _required_source_shapes()

    assert len(required) == 810
    assert EXPECTED_OUTPUT_SHAPES["decoder_action_in_proj_w"] == (32, 1024)
    assert EXPECTED_OUTPUT_SHAPES["decoder_action_out_proj_w"] == (1024, 32)
    assert EXPECTED_OUTPUT_SHAPES["embedding_weight"] == (257152, 2048)
    assert EXPECTED_OUTPUT_SHAPES["language_embeds"] == (1, 2048)
