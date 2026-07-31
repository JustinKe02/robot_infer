#!/usr/bin/env python

import re

from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def test_default_peft_targets_match_pi05_modules() -> None:
    target_modules = PI05Policy._get_default_peft_targets(None)["target_modules"]
    expected_modules = (
        "model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj",
        "model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.v_proj",
        "model.action_in_proj",
        "model.action_out_proj",
        "model.time_mlp_in",
        "model.time_mlp_out",
    )

    assert all(re.fullmatch(target_modules, module_name) for module_name in expected_modules)
    assert re.fullmatch(target_modules, "model.action_time_mlp_in") is None
