from __future__ import annotations

import pytest
import torch

from lerobot.robots.jz_robot_pin_timed.training_schema import (
    JZPinTrainingSchema,
    build_training_schema_manifest,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.action_queue import (
    ActionChunk,
    ActionChunkQueue,
    select_single_step,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.policy_service import (
    postprocess_action_chunk,
)
from my_devs.jz_robot_pin_timed.pi05.rtc_infer.jz_pi05_runtime.client_runtime import (
    ClientRuntimeResult,
    MetricsSnapshot,
    raise_for_runtime_error,
)


def _chunk(*, steps: int = 4, drop_steps: int = 0) -> ActionChunk:
    raw = torch.arange(steps * 16, dtype=torch.float32).reshape(steps, 16)
    processed = torch.arange(steps * 18, dtype=torch.float32).reshape(steps, 18)
    return ActionChunk(
        raw_actions=raw,
        processed_actions=processed,
        observation_timestamp_s=10.0,
        ready_timestamp_s=10.1,
        drop_steps=drop_steps,
        predicted_delay_steps=1,
        source_observation_seq=5,
    )


def _audited_schema() -> JZPinTrainingSchema:
    return JZPinTrainingSchema(
        build_training_schema_manifest(
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
        )
    )


def test_action_queue_tracks_model16_leftover_and_pops_raw18_actions() -> None:
    chunk = _chunk(steps=5, drop_steps=1)
    queue = ActionChunkQueue(max_queue_size=3, empty_queue_strategy="skip_send")

    result = queue.merge_rtc(chunk)

    assert result.input_steps == 5
    assert result.dropped_steps == 1
    assert result.enqueued_steps == 3
    assert queue.depth() == 3
    leftover = queue.get_raw_leftover()
    assert leftover is not None
    assert leftover.shape == (3, 16)
    torch.testing.assert_close(leftover, chunk.raw_actions[1:4])

    first = queue.pop_processed_action()
    assert first is not None
    assert first.shape == (18,)
    torch.testing.assert_close(first, chunk.processed_actions[1])
    assert queue.depth() == 2
    leftover = queue.get_raw_leftover()
    assert leftover is not None
    torch.testing.assert_close(leftover, chunk.raw_actions[2:4])


def test_single_step_selects_first_non_stale_action_only() -> None:
    chunk = _chunk(steps=4, drop_steps=2)

    selection = select_single_step(chunk)

    assert selection.input_steps == 4
    assert selection.dropped_steps == 2
    assert selection.dropped_all is False
    assert selection.raw_action is not None
    assert selection.processed_action is not None
    torch.testing.assert_close(selection.raw_action, chunk.raw_actions[2])
    torch.testing.assert_close(selection.processed_action, chunk.processed_actions[2])


def test_single_step_reports_fully_stale_chunk() -> None:
    selection = select_single_step(_chunk(steps=3, drop_steps=99))

    assert selection.dropped_steps == 3
    assert selection.dropped_all is True
    assert selection.raw_action is None
    assert selection.processed_action is None


def test_postprocess_fallback_flattens_chunk_and_restores_actual_raw18_dimension() -> None:
    schema = _audited_schema()

    class FlatOnlyPostprocessor:
        def __call__(self, actions: torch.Tensor) -> torch.Tensor:
            if actions.ndim != 2:
                raise RuntimeError("this fake processor accepts flat batches only")
            return schema.expand_action(actions)

    raw = torch.zeros((1, 2, 16), dtype=torch.float32)
    raw[0, 0, :14] = torch.arange(14, dtype=torch.float32)
    raw[0, 0, 14:] = torch.tensor([20.0, 70.0])
    raw[0, 1, 14:] = torch.tensor([100.0, 0.0])

    processed = postprocess_action_chunk(FlatOnlyPostprocessor(), raw)

    assert processed.shape == (1, 2, 18)
    torch.testing.assert_close(processed[0, 0, :14], raw[0, 0, :14])
    torch.testing.assert_close(processed[0, :, 14], torch.tensor([80.0, 0.0]))
    torch.testing.assert_close(processed[0, :, 15], torch.tensor([80.0, 80.0]))
    torch.testing.assert_close(processed[0, :, 16], torch.tensor([30.0, 100.0]))
    torch.testing.assert_close(processed[0, :, 17], torch.tensor([80.0, 80.0]))


def test_fail_closed_stop_reasons_are_not_reported_as_success() -> None:
    metrics = MetricsSnapshot(0.0, 0.0, 0.0, 50, 0, 3)
    failed = ClientRuntimeResult(
        mode="single_step",
        stop_reason="3 consecutive inference chunks were fully stale",
        sensor_ticks=3,
        actor_ticks=0,
        sent_actions=0,
        inference_requests=3,
        metrics=metrics,
        queue=None,
    )
    with pytest.raises(RuntimeError, match="fully stale"):
        raise_for_runtime_error(failed)

    completed = ClientRuntimeResult(
        mode="single_step",
        stop_reason="run_time_s elapsed: 10.000s",
        sensor_ticks=1,
        actor_ticks=1,
        sent_actions=1,
        inference_requests=1,
        metrics=metrics,
        queue=None,
    )
    raise_for_runtime_error(completed)
