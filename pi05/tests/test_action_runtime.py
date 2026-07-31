from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.robots.jz_robot_pin_timed.training_schema import (
    JZPinTrainingSchema,
    build_training_schema_manifest,
)
from tk_infer.pi05.runtime.action_queue import (
    ActionChunk,
    ActionChunkQueue,
    select_single_step,
)
from tk_infer.pi05.runtime.client_runtime import (
    ClientMetrics,
    ClientRuntimeConfig,
    ClientRuntimeResult,
    ClientRuntimeState,
    FrameBuffer,
    MetricsSnapshot,
    raise_for_runtime_error,
    run_producer_loop,
)
from tk_infer.pi05.runtime.policy_service import (
    postprocess_action_chunk,
)
from tk_infer.pi05.runtime.protocol import InferenceResponse
from tk_infer.pi05.runtime.safety import ActionSafety


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


@pytest.mark.parametrize(
    ("runtime_mode", "wire_mode", "expected_delay", "expects_leftover"),
    [
        ("async_single_step", "single_step", 0, False),
        ("rtc", "rtc", 2, True),
    ],
)
def test_async_producer_preserves_single_step_or_rtc_wire_contract(
    runtime_mode: str,
    wire_mode: str,
    expected_delay: int,
    expects_leftover: bool,
) -> None:
    config = ClientRuntimeConfig(
        task="jz robot pin timed vr teleoperation",
        mode=runtime_mode,  # type: ignore[arg-type]
        sensor_fps=20,
        control_fps=20,
        empty_queue_strategy="skip_send",
    )
    state = ClientRuntimeState()
    metrics = ClientMetrics()
    metrics.record_request(total_s=0.1, server_s=0.08, drop_steps=0, predicted_delay_steps=0)
    frame_buffer = FrameBuffer()
    frame_buffer.update(
        raw_observation={"raw": 1},
        observation_frame={"observation.state": np.zeros(18, dtype=np.float32)},
        timestamp_s=0.0,
    )
    action_queue = ActionChunkQueue(max_queue_size=50, empty_queue_strategy="skip_send")
    action_queue.merge_rtc(_chunk(steps=3))

    class OneRequestPolicy:
        def __init__(self) -> None:
            self.request = None

        def infer(self, request: object) -> InferenceResponse:
            self.request = request
            state.request_stop("test complete")
            processed = np.zeros((4, 18), dtype=np.float32)
            processed[:, 15] = 80.0
            processed[:, 17] = 80.0
            return InferenceResponse(
                request_id=request.request_id,  # type: ignore[attr-defined]
                mode=request.mode,  # type: ignore[attr-defined]
                raw_actions=np.zeros((4, 16), dtype=np.float32),
                processed_actions=processed,
                server_latency_s=0.08,
                model_latency_s=0.07,
                raw_action_shape=(4, 16),
                processed_action_shape=(4, 18),
            )

    policy = OneRequestPolicy()
    clock = iter((0.0, 0.1))
    requests = run_producer_loop(
        config=config,
        state=state,
        metrics=metrics,
        remote_policy=policy,  # type: ignore[arg-type]
        frame_buffer=frame_buffer,
        action_queue=action_queue,
        robot_type="jz_robot_pin_timed",
        safety=ActionSafety(),
        perf_counter=lambda: next(clock),
        sleep_fn=lambda _seconds: None,
    )

    assert requests == 1
    assert policy.request is not None
    assert policy.request.mode == wire_mode
    assert policy.request.predicted_delay_steps == expected_delay
    assert (policy.request.prev_chunk_left_over is not None) is expects_leftover


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

    bounded = ClientRuntimeResult(
        mode="single_step",
        stop_reason="max_sent_actions reached: 1",
        sensor_ticks=1,
        actor_ticks=1,
        sent_actions=1,
        inference_requests=1,
        metrics=metrics,
        queue=None,
    )
    raise_for_runtime_error(bounded)
