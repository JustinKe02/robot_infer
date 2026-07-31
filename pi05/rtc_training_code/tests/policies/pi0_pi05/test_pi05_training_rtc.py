#!/usr/bin/env python

from types import SimpleNamespace

import draccus
import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config, RTCTrainingConfig
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Pytorch,
    build_rtc_training_inputs,
    prepare_rtc_action_prefix,
    rtc_masked_loss,
    sample_rtc_training_delay,
)


def test_rtc_training_is_disabled_by_default() -> None:
    config = PI05Config(device="cpu")

    assert config.rtc_training.enabled is False
    assert config.rtc_training.max_delay == 10
    assert config.rtc_training.min_postfix_steps == 1


def test_rtc_training_can_be_enabled_from_cli_overrides() -> None:
    with draccus.config_type("json"):
        config = draccus.parse(
            PI05Config,
            args=[
                "--device=cpu",
                "--rtc_training.enabled=true",
                "--rtc_training.max_delay=6",
                "--rtc_training.min_postfix_steps=2",
            ],
        )

    assert config.rtc_training.enabled is True
    assert config.rtc_training.max_delay == 6
    assert config.rtc_training.min_postfix_steps == 2


def test_rtc_training_config_round_trip(tmp_path) -> None:
    config = PI05Config(
        device="cpu",
        rtc_training=RTCTrainingConfig(enabled=True, max_delay=6, min_postfix_steps=2),
    )
    config._save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert loaded.rtc_training.enabled is True
    assert loaded.rtc_training.max_delay == 6
    assert loaded.rtc_training.min_postfix_steps == 2


def test_rtc_training_config_rejects_invalid_histogram() -> None:
    with pytest.raises(ValueError, match=r"max_delay \+ 1"):
        RTCTrainingConfig(max_delay=2, observed_delay_histogram=(1.0, 0.0))


def test_delay_is_sampled_per_item_and_respects_postfix_limit() -> None:
    torch.manual_seed(7)
    config = RTCTrainingConfig(enabled=True, max_delay=8, min_postfix_steps=3)

    delay = sample_rtc_training_delay(batch_size=64, horizon=10, config=config, device=torch.device("cpu"))

    assert delay.shape == (64,)
    assert delay.dtype == torch.long
    assert delay.min().item() >= 0
    assert delay.max().item() <= 7
    assert torch.unique(delay).numel() > 1


def test_observed_delay_histogram_can_drive_sampling() -> None:
    config = RTCTrainingConfig(
        enabled=True,
        max_delay=2,
        observed_delay_histogram=(0.0, 1.0, 0.0),
        observed_histogram_weight=1.0,
    )

    delay = sample_rtc_training_delay(batch_size=16, horizon=5, config=config, device=torch.device("cpu"))

    assert torch.equal(delay, torch.ones(16, dtype=torch.long))


def test_build_rtc_inputs_keeps_prefix_clean_and_uses_local_time() -> None:
    actions = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    noise = actions + 10.0
    global_time = torch.tensor([0.25, 0.75])
    delay = torch.tensor([2, 0])

    x_t, token_time, prefix_mask = build_rtc_training_inputs(actions, noise, global_time, delay)

    assert torch.equal(prefix_mask, torch.tensor([[True, True, False, False], [False] * 4]))
    assert torch.equal(token_time[0], torch.tensor([0.0, 0.0, 0.25, 0.25]))
    assert torch.equal(token_time[1], torch.full((4,), 0.75))
    assert torch.equal(x_t[0, :2], actions[0, :2])
    assert torch.allclose(x_t[0, 2:], 0.25 * noise[0, 2:] + 0.75 * actions[0, 2:])
    assert torch.allclose(x_t[1], 0.75 * noise[1] + 0.25 * actions[1])


def test_delay_zero_matches_standard_flow_input() -> None:
    actions = torch.randn(3, 5, 2)
    noise = torch.randn_like(actions)
    global_time = torch.tensor([0.1, 0.5, 0.9])

    x_t, token_time, prefix_mask = build_rtc_training_inputs(
        actions,
        noise,
        global_time,
        delay=torch.zeros(3, dtype=torch.long),
    )
    expected = global_time[:, None, None] * noise + (1.0 - global_time[:, None, None]) * actions

    assert torch.allclose(x_t, expected)
    assert torch.equal(token_time, global_time[:, None].expand(3, 5))
    assert not prefix_mask.any()


def test_masked_loss_gives_each_sample_equal_weight() -> None:
    losses = torch.stack(
        [
            torch.ones(4, 2),
            torch.full((4, 2), 9.0),
        ]
    )
    postfix_mask = torch.tensor(
        [
            [True, True, True, True],
            [False, False, False, True],
        ]
    )

    loss, per_sample = rtc_masked_loss(losses, postfix_mask)

    assert torch.equal(per_sample, torch.tensor([1.0, 9.0]))
    assert loss.item() == pytest.approx(5.0)


def test_masked_loss_rejects_empty_postfix() -> None:
    with pytest.raises(ValueError, match="at least one postfix"):
        rtc_masked_loss(torch.ones(1, 3, 2), torch.zeros(1, 3, dtype=torch.bool))


def test_prepare_action_prefix_pads_dimension_and_builds_mask() -> None:
    reference = torch.randn(1, 5, 4)
    action_prefix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    prefix_values, prefix_mask = prepare_rtc_action_prefix(
        action_prefix,
        prefix_length=2,
        reference=reference,
    )

    assert torch.equal(prefix_mask, torch.tensor([[True, True, False, False, False]]))
    assert torch.equal(prefix_values[0, :2, :2], action_prefix[:2])
    assert torch.count_nonzero(prefix_values[0, :2, 2:]) == 0
    assert torch.count_nonzero(prefix_values[0, 2:]) == 0


def test_sample_actions_clamps_prefix_after_every_euler_step() -> None:
    model = object.__new__(PI05Pytorch)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=5,
        max_action_dim=4,
        num_inference_steps=2,
        rtc_training=RTCTrainingConfig(enabled=True, max_delay=2),
        rtc_config=None,
    )
    model.rtc_processor = None
    model.embed_prefix = lambda *_args: (
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
    )
    model._prepare_attention_masks_4d = lambda mask: mask[:, None]
    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(
            language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
        ),
        forward=lambda **_kwargs: ([None, None], object()),
    )
    model.denoise_step = lambda **kwargs: torch.ones_like(kwargs["x_t"])

    action_prefix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = model.sample_actions(
        images=[],
        img_masks=[],
        tokens=torch.zeros(1, 1, dtype=torch.long),
        masks=torch.ones(1, 1, dtype=torch.bool),
        noise=torch.zeros(1, 5, 4),
        action_prefix=action_prefix,
        prefix_length=2,
    )

    assert torch.equal(result[0, :2, :2], action_prefix)
    assert torch.count_nonzero(result[0, :2, 2:]) == 0
    assert torch.equal(result[0, 2:], torch.full((3, 4), -1.0))
    assert result.requires_grad is False
