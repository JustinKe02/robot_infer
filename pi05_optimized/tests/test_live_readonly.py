from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.robots.jz_robot_udp.state_cache import CachedState
from tk_infer.pi05_optimized.runtime.live_readonly import (
    CAMERA_NAMES,
    LiveReadOnlyConfig,
    LiveReadOnlyObservationSource,
    RecordingActionSink,
    raw18_from_state,
)


def _packet(sequence: int) -> dict[str, object]:
    return {
        "version": 1,
        "type": "state",
        "robot": "robot1",
        "seq": sequence,
        "stamp_ns": sequence * 1_000_000,
        "joints": {
            "left": {f"left_joint{index}": float(index) for index in range(1, 8)},
            "right": {f"right_joint{index}": float(index + 10) for index in range(1, 8)},
        },
        "grippers": {
            "left": {"width": 20.0, "force": 80.0},
            "right": {"width": 30.0, "force": 80.0},
        },
        "source_timing": {
            "schema_version": 1,
            "sources": {
                name: {
                    "generation": sequence,
                    "recv_wall_ns": sequence * 1_000_000,
                    "recv_monotonic_ns": sequence * 1_000_000,
                    "header_stamp_ns": sequence * 1_000_000 if name.endswith("_joints") else None,
                    "age_ms": 1.0,
                }
                for name in ("left_joints", "right_joints", "left_gripper", "right_gripper")
            },
            "source_skew_ms": 0.0,
        },
    }


class FakeCache:
    def __init__(self, states: list[CachedState]) -> None:
        self.states = states

    def wait_after_revision(self, timeout_s: float, after_revision: int) -> CachedState | None:
        del timeout_s
        return next((state for state in self.states if state.revision > after_revision), None)


class FakeReceiver:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeCamera:
    def __init__(self, name: str, shape: tuple[int, int, int], sequences: list[int]) -> None:
        self.name = name
        self.shape = shape
        self.sequences = iter(sequences)
        self.connected = False
        self.reads = 0

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def diagnostics(self) -> dict[str, object]:
        return {"accepted_frames": self.reads, "invalid_messages": 0}

    def connect(self, warmup: bool = True) -> None:
        assert warmup is False
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_timed_nearest(
        self,
        target_monotonic_ns: int,
        *,
        max_receive_skew_ms: float | None = None,
        wait_timeout_ms: float = 0,
    ) -> object:
        assert max_receive_skew_ms == 250.0
        assert wait_timeout_ms == 250.0
        sequence = next(self.sequences)
        self.reads += 1
        return SimpleNamespace(
            image=np.full(self.shape, sequence, dtype=np.uint8),
            sequence=sequence,
            receive_monotonic_ns=target_monotonic_ns + 1_000_000,
            camera_timing={"capture_monotonic_ns": sequence * 1_000_000},
        )


def _state(sequence: int, revision: int, sender: str = "192.168.1.81") -> CachedState:
    return CachedState(
        packet=_packet(sequence),
        sender=(sender, 39010),
        received_monotonic_s=time.monotonic(),
        received_wall_ns=time.time_ns(),
        revision=revision,
    )


def test_raw18_mapping_preserves_training_order() -> None:
    state = raw18_from_state(_packet(1))

    assert state.shape == (18,)
    np.testing.assert_array_equal(
        state,
        np.asarray([1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 20, 80, 30, 80], dtype=np.float32),
    )


def test_live_source_reads_new_state_and_timestamped_rgb_frames_then_stops() -> None:
    cache = FakeCache([_state(10, 1), _state(11, 2)])
    receiver = FakeReceiver()
    cameras = {
        "camera_head": FakeCamera("camera_head", (720, 1280, 3), [101]),
        "camera_right": FakeCamera("camera_right", (480, 640, 3), [201]),
    }
    source = LiveReadOnlyObservationSource(
        LiveReadOnlyConfig(),
        state_cache=cache,
        state_receiver=receiver,
        cameras=cameras,  # type: ignore[arg-type]
    )

    source.connect()
    observation = source.read()
    source.disconnect()

    assert receiver.started is True
    assert receiver.stopped is True
    assert observation.sequence_id == 0
    assert observation.observation_frame["observation.state"].shape == (18,)
    assert observation.observation_frame["observation.images.camera_head"].shape == (720, 1280, 3)
    assert observation.observation_frame["observation.images.camera_right"].shape == (480, 640, 3)
    assert observation.state_source_timestamp is not None
    assert set(observation.camera_source_timestamps) == {
        "observation.images.camera_head",
        "observation.images.camera_right",
    }
    assert source.diagnostics["receiver_stopped"] is True
    assert source.diagnostics["command_transport_created"] is False
    assert source.diagnostics["robot_created"] is False


def test_live_source_rejects_unexpected_state_sender_and_cleans_up() -> None:
    receiver = FakeReceiver()
    source = LiveReadOnlyObservationSource(
        LiveReadOnlyConfig(),
        state_cache=FakeCache([_state(1, 1, sender="192.168.1.99")]),
        state_receiver=receiver,
        cameras={
            "camera_head": FakeCamera("camera_head", (720, 1280, 3), [1]),
            "camera_right": FakeCamera("camera_right", (480, 640, 3), [1]),
        },  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="does not match"):
        source.connect()

    assert receiver.stopped is True
    assert source.is_connected is False


def test_recording_sink_retains_policy_outputs_only_in_memory() -> None:
    sink = RecordingActionSink()
    action = torch.zeros(18)
    action[15] = 80
    action[17] = 80

    sink.write(action)
    summary = sink.summary()

    assert summary["count"] == 1
    assert summary["force_slots_exact"] is True
    assert summary["finite"] is True


def test_recording_sink_rejects_invalid_force_without_recording() -> None:
    sink = RecordingActionSink()

    with pytest.raises(ValueError, match="force slots"):
        sink.write(torch.zeros(18))

    assert sink.count == 0


def test_live_source_camera_set_is_fixed_to_head_right() -> None:
    with pytest.raises(ValueError, match="exactly"):
        LiveReadOnlyObservationSource(
            LiveReadOnlyConfig(),
            state_cache=FakeCache([_state(1, 1)]),
            state_receiver=FakeReceiver(),
            cameras={"camera_head": FakeCamera("camera_head", (720, 1280, 3), [1])},  # type: ignore[arg-type]
        )

    assert CAMERA_NAMES == ("camera_head", "camera_right")
