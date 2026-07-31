from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ActionChunk:
    """One server response in both RTC and robot action spaces.

    ``raw_actions`` are normalized model16 actions and must stay untouched for
    ``prev_chunk_left_over``. ``processed_actions`` are postprocessed raw18
    actions ready for the JZ robot boundary. Their action dimensions are
    intentionally different; only their time dimensions must match.
    """

    raw_actions: Tensor
    processed_actions: Tensor
    observation_timestamp_s: float
    ready_timestamp_s: float
    drop_steps: int
    predicted_delay_steps: int
    source_observation_seq: int


@dataclass(frozen=True, slots=True)
class MergeResult:
    input_steps: int
    dropped_steps: int
    enqueued_steps: int
    queue_depth_after: int
    dropped_all: bool


@dataclass(frozen=True, slots=True)
class SingleStepSelection:
    raw_action: Tensor | None
    processed_action: Tensor | None
    input_steps: int
    dropped_steps: int
    dropped_all: bool


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    depth: int
    absolute_cursor: int
    raw_leftover_steps: int
    has_last_action: bool
    merge_calls: int
    popped_actions: int
    empty_events: int
    hold_last_events: int
    skip_send_events: int
    stop_events: int
    dropped_all_chunks: int


class ActionChunkQueue:
    """Thread-safe RTC queue with model16 leftovers and raw18 execution data."""

    def __init__(self, *, max_queue_size: int = 50, empty_queue_strategy: str = "stop") -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if empty_queue_strategy not in {"stop", "skip_send", "hold_last_action"}:
            raise ValueError("empty_queue_strategy must be stop, skip_send, or hold_last_action")
        self.max_queue_size = int(max_queue_size)
        self.empty_queue_strategy = empty_queue_strategy
        self._raw_queue: Tensor | None = None
        self._processed_queue: Tensor | None = None
        self._cursor = 0
        self._absolute_cursor = 0
        self._last_action: Tensor | None = None
        self._lock = threading.RLock()
        self._merge_calls = 0
        self._popped_actions = 0
        self._empty_events = 0
        self._hold_last_events = 0
        self._skip_send_events = 0
        self._stop_events = 0
        self._dropped_all_chunks = 0

    def depth(self) -> int:
        with self._lock:
            return self._depth_locked()

    def absolute_cursor(self) -> int:
        with self._lock:
            return self._absolute_cursor

    def get_raw_leftover(self) -> Tensor | None:
        """Return unconsumed normalized model16 actions for the next RTC request."""

        with self._lock:
            if self._raw_queue is None:
                return None
            return self._raw_queue[self._cursor :].detach().clone()

    def merge_rtc(self, chunk: ActionChunk) -> MergeResult:
        """Replace the RTC queue after dropping the response's stale prefix."""

        raw_actions, processed_actions = _validate_and_copy_chunk(chunk)
        input_steps = int(raw_actions.shape[0])
        dropped_steps = min(max(int(chunk.drop_steps), 0), input_steps)
        raw_actions = raw_actions[dropped_steps : dropped_steps + self.max_queue_size]
        processed_actions = processed_actions[dropped_steps : dropped_steps + self.max_queue_size]

        with self._lock:
            self._merge_calls += 1
            self._raw_queue = raw_actions if len(raw_actions) else None
            self._processed_queue = processed_actions if len(processed_actions) else None
            self._cursor = 0
            dropped_all = len(processed_actions) == 0
            if dropped_all:
                self._dropped_all_chunks += 1
            return MergeResult(
                input_steps=input_steps,
                dropped_steps=dropped_steps,
                enqueued_steps=len(processed_actions),
                queue_depth_after=self._depth_locked(),
                dropped_all=dropped_all,
            )

    def pop_processed_action(self) -> Tensor | None:
        """Pop one raw18 action, applying the configured fail-safe empty policy."""

        with self._lock:
            if self._processed_queue is not None and self._cursor < len(self._processed_queue):
                action = self._processed_queue[self._cursor].detach().clone()
                self._cursor += 1
                self._absolute_cursor += 1
                self._popped_actions += 1
                self._last_action = action.detach().clone()
                if self._cursor >= len(self._processed_queue):
                    self._raw_queue = None
                    self._processed_queue = None
                    self._cursor = 0
                return action

            self._empty_events += 1
            if self.empty_queue_strategy == "hold_last_action" and self._last_action is not None:
                self._hold_last_events += 1
                return self._last_action.detach().clone()
            if self.empty_queue_strategy in {"skip_send", "hold_last_action"}:
                self._skip_send_events += 1
                return None
            self._stop_events += 1
            raise RuntimeError("RTC action queue is empty; fail-closed stop requested")

    def snapshot(self) -> QueueSnapshot:
        with self._lock:
            raw_leftover_steps = 0
            if self._raw_queue is not None:
                raw_leftover_steps = max(0, len(self._raw_queue) - self._cursor)
            return QueueSnapshot(
                depth=self._depth_locked(),
                absolute_cursor=self._absolute_cursor,
                raw_leftover_steps=raw_leftover_steps,
                has_last_action=self._last_action is not None,
                merge_calls=self._merge_calls,
                popped_actions=self._popped_actions,
                empty_events=self._empty_events,
                hold_last_events=self._hold_last_events,
                skip_send_events=self._skip_send_events,
                stop_events=self._stop_events,
                dropped_all_chunks=self._dropped_all_chunks,
            )

    def _depth_locked(self) -> int:
        if self._processed_queue is None:
            return 0
        return max(0, len(self._processed_queue) - self._cursor)


def select_single_step(chunk: ActionChunk) -> SingleStepSelection:
    """Select exactly one fresh action from a normal chunk response.

    The action at ``drop_steps`` is the first action whose intended execution
    time has not already elapsed on the client. No remaining action is queued.
    """

    raw_actions, processed_actions = _validate_and_copy_chunk(chunk)
    input_steps = int(raw_actions.shape[0])
    dropped_steps = min(max(int(chunk.drop_steps), 0), input_steps)
    if dropped_steps >= input_steps:
        return SingleStepSelection(
            raw_action=None,
            processed_action=None,
            input_steps=input_steps,
            dropped_steps=dropped_steps,
            dropped_all=True,
        )
    return SingleStepSelection(
        raw_action=raw_actions[dropped_steps].detach().clone(),
        processed_action=processed_actions[dropped_steps].detach().clone(),
        input_steps=input_steps,
        dropped_steps=dropped_steps,
        dropped_all=False,
    )


def _validate_and_copy_chunk(chunk: ActionChunk) -> tuple[Tensor, Tensor]:
    raw_actions = torch.as_tensor(chunk.raw_actions).detach().to(device="cpu", dtype=torch.float32).clone()
    processed_actions = (
        torch.as_tensor(chunk.processed_actions).detach().to(device="cpu", dtype=torch.float32).clone()
    )
    if raw_actions.ndim != 2 or processed_actions.ndim != 2:
        raise ValueError(
            "Action chunks must be 2D tensors shaped (time_steps, action_dim); "
            f"got raw={tuple(raw_actions.shape)} processed={tuple(processed_actions.shape)}"
        )
    if raw_actions.shape[0] != processed_actions.shape[0]:
        raise ValueError(
            "model16 and raw18 chunks must have the same time dimension; "
            f"got {raw_actions.shape[0]} and {processed_actions.shape[0]}"
        )
    if raw_actions.shape[1] != 16:
        raise ValueError(f"RTC raw action chunk must be model16, got {tuple(raw_actions.shape)}")
    if processed_actions.shape[1] != 18:
        raise ValueError(f"RTC processed action chunk must be raw18, got {tuple(processed_actions.shape)}")
    if not torch.isfinite(raw_actions).all():
        raise ValueError("RTC model16 action chunk contains non-finite values")
    if not torch.isfinite(processed_actions).all():
        raise ValueError("RTC raw18 action chunk contains non-finite values")
    return raw_actions, processed_actions


__all__ = [
    "ActionChunk",
    "ActionChunkQueue",
    "MergeResult",
    "QueueSnapshot",
    "SingleStepSelection",
    "select_single_step",
]
