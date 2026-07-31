from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol

import numpy as np
import torch

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
from lerobot.robots.jz_robot_pin_timed.timestamped_zmq_camera import (
    TimestampedZMQCamera,
    TimestampedZMQFrame,
)
from lerobot.robots.jz_robot_pin_timed.training_schema import RAW_FEATURE_NAMES
from lerobot.robots.jz_robot_udp.config_jz_robot_udp import (
    DEFAULT_LEFT_JOINT_NAMES,
    DEFAULT_RIGHT_JOINT_NAMES,
)
from lerobot.robots.jz_robot_udp.protocol import validate_source_timing
from lerobot.robots.jz_robot_udp.state_cache import CachedState, StateCache
from lerobot.robots.jz_robot_udp.udp_client import UDPStateReceiver
from tk_infer.pi05.runtime.camera_profiles import CAMERA_SPECS

from .timed_observation import SourceTimestamp, TimedObservation

CAMERA_NAMES = ("camera_head", "camera_right")
CAMERA_KEYS = tuple(f"observation.images.{name}" for name in CAMERA_NAMES)
LOCAL_CLOCK_DOMAIN = "process_perf_counter"
STATE_CLOCK_DOMAIN = "orin_state_source_monotonic"
CAMERA_CLOCK_DOMAIN = "orin_camera_capture_monotonic"


class _StateCache(Protocol):
    def wait_after_revision(self, timeout_s: float, after_revision: int) -> CachedState | None: ...


class _StateReceiver(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class _TimedCamera(Protocol):
    @property
    def is_connected(self) -> bool: ...

    @property
    def diagnostics(self) -> dict[str, Any]: ...

    def connect(self, warmup: bool = True) -> None: ...

    def disconnect(self) -> None: ...

    def read_timed_nearest(
        self,
        target_monotonic_ns: int,
        *,
        max_receive_skew_ms: float | None = None,
        wait_timeout_ms: float = 0,
    ) -> TimestampedZMQFrame: ...


@dataclass(frozen=True, slots=True)
class LiveReadOnlyConfig:
    orin_ip: str = "192.168.1.81"
    state_bind_ip: str = "0.0.0.0"
    state_port: int = 39010
    connect_timeout_s: float = 5.0
    state_timeout_s: float = 1.0
    camera_timeout_ms: int = 5000
    camera_buffer_size: int = 8
    camera_stale_frame_timeout_ms: int = 1000
    max_camera_state_receive_skew_ms: float = 250.0

    def __post_init__(self) -> None:
        for name in ("orin_ip", "state_bind_ip"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if isinstance(self.state_port, bool) or not isinstance(self.state_port, int):
            raise ValueError("state_port must be an integer")
        if not 1 <= self.state_port <= 65535:
            raise ValueError("state_port must be in 1..65535")
        for name in ("connect_timeout_s", "state_timeout_s", "max_camera_state_receive_skew_ms"):
            object.__setattr__(self, name, _finite_positive(name, getattr(self, name)))
        for name in ("camera_timeout_ms", "camera_buffer_size", "camera_stale_frame_timeout_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class LiveReadOnlyObservationSource:
    """Real camera/state source that owns no robot or command transport."""

    def __init__(
        self,
        config: LiveReadOnlyConfig,
        *,
        state_cache: _StateCache | None = None,
        state_receiver: _StateReceiver | None = None,
        cameras: dict[str, _TimedCamera] | None = None,
        perf_counter: Any = time.perf_counter,
    ) -> None:
        self.config = config
        cache = state_cache or StateCache()
        self._state_cache = cache
        self._state_receiver = state_receiver or UDPStateReceiver(
            config.state_bind_ip,
            config.state_port,
            cache,  # type: ignore[arg-type]
            label="pi05_optimized_live_readonly",
        )
        self._cameras = cameras or _build_cameras(config)
        if set(self._cameras) != set(CAMERA_NAMES):
            raise ValueError(f"live source cameras must be exactly {CAMERA_NAMES}")
        self._perf_counter = perf_counter
        self._connected = False
        self._last_state_revision = 0
        self._last_state_sequence: int | None = None
        self._last_state_stamp_ns: int | None = None
        self._sequence_id = 0
        self._read_count = 0
        self._first_state_sequence: int | None = None
        self._camera_state_skew_ms: dict[str, list[float]] = {name: [] for name in CAMERA_NAMES}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "read_count": self._read_count,
            "first_state_sequence": self._first_state_sequence,
            "last_state_sequence": self._last_state_sequence,
            "last_state_stamp_ns": self._last_state_stamp_ns,
            "camera_state_receive_skew_ms": {
                name: _distribution(values) for name, values in self._camera_state_skew_ms.items()
            },
            "cameras": {name: copy.deepcopy(camera.diagnostics) for name, camera in self._cameras.items()},
            "receiver_stopped": not self._connected,
            "command_transport_created": False,
            "robot_created": False,
        }

    def connect(self) -> None:
        if self._connected:
            raise RuntimeError("live read-only source is already connected")
        connected_cameras: list[_TimedCamera] = []
        try:
            self._state_receiver.start()
            first = self._state_cache.wait_after_revision(
                timeout_s=self.config.connect_timeout_s,
                after_revision=0,
            )
            if first is None:
                raise TimeoutError(
                    f"no state received on udp://{self.config.state_bind_ip}:{self.config.state_port}"
                )
            self._validate_state(first, enforce_progress=False)
            self._last_state_revision = first.revision
            for name in CAMERA_NAMES:
                camera = self._cameras[name]
                camera.connect(warmup=False)
                connected_cameras.append(camera)
            self._connected = True
        except BaseException:
            for camera in reversed(connected_cameras):
                if camera.is_connected:
                    camera.disconnect()
            self._state_receiver.stop()
            raise

    def disconnect(self) -> None:
        errors: list[str] = []
        for name in reversed(CAMERA_NAMES):
            camera = self._cameras[name]
            if not camera.is_connected:
                continue
            try:
                camera.disconnect()
            except BaseException as error:
                errors.append(f"{name}: {type(error).__name__}: {error}")
        try:
            self._state_receiver.stop()
        except BaseException as error:
            errors.append(f"state: {type(error).__name__}: {error}")
        self._connected = False
        if errors:
            raise RuntimeError(f"live read-only source cleanup failed: {errors}")

    def read(self) -> TimedObservation:
        if not self._connected:
            raise RuntimeError("live read-only source is not connected")
        build_started_s = self._perf_counter()
        state = self._state_cache.wait_after_revision(
            timeout_s=self.config.state_timeout_s,
            after_revision=self._last_state_revision,
        )
        if state is None:
            raise TimeoutError(
                f"state did not advance within {self.config.state_timeout_s:.3f}s after revision "
                f"{self._last_state_revision}"
            )
        self._validate_state(state, enforce_progress=True)
        self._last_state_revision = state.revision
        state_receive_ns = int(state.received_monotonic_s * 1_000_000_000)
        observation_frame: dict[str, Any] = {"observation.state": raw18_from_state(state.packet)}
        camera_timestamps: dict[str, SourceTimestamp] = {}
        for name in CAMERA_NAMES:
            frame = self._cameras[name].read_timed_nearest(
                state_receive_ns,
                max_receive_skew_ms=self.config.max_camera_state_receive_skew_ms,
                wait_timeout_ms=self.config.max_camera_state_receive_skew_ms,
            )
            _validate_image(name, frame.image)
            key = f"observation.images.{name}"
            observation_frame[key] = frame.image
            camera_timestamps[key] = SourceTimestamp(
                int(frame.camera_timing["capture_monotonic_ns"]) / 1_000_000_000,
                CAMERA_CLOCK_DOMAIN,
                name,
            )
            skew_ms = abs(frame.receive_monotonic_ns - state_receive_ns) / 1_000_000
            self._camera_state_skew_ms[name].append(skew_ms)

        build_ready_s = self._perf_counter()
        sequence_id = self._sequence_id
        self._sequence_id += 1
        self._read_count += 1
        return TimedObservation(
            observation_frame=observation_frame,
            sequence_id=sequence_id,
            receive_monotonic_s=build_started_s,
            build_started_monotonic_s=build_started_s,
            build_ready_monotonic_s=build_ready_s,
            local_clock_domain=LOCAL_CLOCK_DOMAIN,
            state_source_timestamp=SourceTimestamp(
                int(state.packet["stamp_ns"]) / 1_000_000_000,
                STATE_CLOCK_DOMAIN,
                "raw18_state",
            ),
            camera_source_timestamps=camera_timestamps,
        )

    def _validate_state(self, state: CachedState, *, enforce_progress: bool) -> None:
        if state.sender[0] != self.config.orin_ip:
            raise RuntimeError(
                f"state sender {state.sender[0]}:{state.sender[1]} does not match {self.config.orin_ip}"
            )
        age_s = time.monotonic() - state.received_monotonic_s
        if age_s < 0 or age_s > self.config.state_timeout_s:
            raise TimeoutError(
                f"state is stale: age_s={age_s:.6f} timeout_s={self.config.state_timeout_s:.6f}"
            )
        packet = state.packet
        validate_source_timing(packet.get("source_timing"))
        sequence = int(packet["seq"])
        stamp_ns = int(packet["stamp_ns"])
        if enforce_progress and self._last_state_sequence is not None:
            if sequence <= self._last_state_sequence:
                raise RuntimeError(
                    f"state sequence did not advance: current={sequence} previous={self._last_state_sequence}"
                )
            if self._last_state_stamp_ns is not None and stamp_ns < self._last_state_stamp_ns:
                raise RuntimeError(
                    f"state timestamp regressed: current={stamp_ns} previous={self._last_state_stamp_ns}"
                )
        if self._first_state_sequence is None:
            self._first_state_sequence = sequence
        self._last_state_sequence = sequence
        self._last_state_stamp_ns = stamp_ns


class RecordingActionSink:
    """In-memory policy-output recorder; it has no external transport."""

    def __init__(self) -> None:
        self._actions: list[torch.Tensor] = []

    @property
    def count(self) -> int:
        return len(self._actions)

    def write(self, action: torch.Tensor) -> None:
        copied = action.detach().to(device="cpu", dtype=torch.float32).clone()
        if copied.shape != (18,):
            raise ValueError(f"recorded raw18 policy output must have shape (18,), got {tuple(copied.shape)}")
        if not torch.isfinite(copied).all():
            raise ValueError("recorded raw18 policy output contains non-finite values")
        if copied[15].item() != 80.0 or copied[17].item() != 80.0:
            raise ValueError("recorded raw18 policy output force slots must equal 80")
        self._actions.append(copied)

    def summary(self) -> dict[str, Any]:
        if not self._actions:
            return {
                "count": 0,
                "shapes": [],
                "finite": True,
                "force_slots_exact": True,
                "joint_min": None,
                "joint_max": None,
            }
        stacked = torch.stack(self._actions)
        return {
            "count": len(self._actions),
            "shapes": [18],
            "finite": bool(torch.isfinite(stacked).all().item()),
            "force_slots_exact": bool(
                torch.all(stacked[:, 15] == 80.0).item() and torch.all(stacked[:, 17] == 80.0).item()
            ),
            "joint_min": float(stacked[:, :14].min().item()),
            "joint_max": float(stacked[:, :14].max().item()),
        }


def raw18_from_state(packet: dict[str, Any]) -> np.ndarray:
    values: dict[str, float] = {}
    for side, joint_names in (
        ("left", DEFAULT_LEFT_JOINT_NAMES),
        ("right", DEFAULT_RIGHT_JOINT_NAMES),
    ):
        joints = packet["joints"][side]
        for joint in joint_names:
            if joint not in joints:
                raise RuntimeError(f"state packet is missing {side} joint {joint!r}")
            values[f"{side}_{joint}.pos"] = float(joints[joint])
        gripper = packet["grippers"][side]
        for field in ("width", "force"):
            if field not in gripper:
                raise RuntimeError(f"state packet is missing {side} gripper {field!r}")
            values[f"{side}_gripper.{field}"] = float(gripper[field])
    state = np.asarray([values[name] for name in RAW_FEATURE_NAMES], dtype=np.float32)
    if state.shape != (18,) or not np.isfinite(state).all():
        raise ValueError("state packet did not produce one finite raw18 vector")
    return state


def _build_cameras(config: LiveReadOnlyConfig) -> dict[str, TimestampedZMQCamera]:
    cameras: dict[str, TimestampedZMQCamera] = {}
    for name in CAMERA_NAMES:
        spec = CAMERA_SPECS[name]
        camera_config = ZMQCameraConfig(
            server_address=config.orin_ip,
            port=spec.port,
            camera_name=name,
            fps=30,
            width=spec.width,
            height=spec.height,
            color_mode=ColorMode.RGB,
            timeout_ms=config.camera_timeout_ms,
        )
        cameras[name] = TimestampedZMQCamera(
            camera_config,
            buffer_size=config.camera_buffer_size,
            stale_frame_timeout_ms=config.camera_stale_frame_timeout_ms,
        )
    return cameras


def _validate_image(name: str, image: np.ndarray) -> None:
    spec = CAMERA_SPECS[name]
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} image must be a NumPy array")
    if image.dtype != np.uint8 or image.shape != spec.hwc_shape:
        raise ValueError(
            f"{name} image must be uint8 HWC {spec.hwc_shape}, got dtype={image.dtype} shape={image.shape}"
        )


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


__all__ = [
    "CAMERA_KEYS",
    "CAMERA_NAMES",
    "LiveReadOnlyConfig",
    "LiveReadOnlyObservationSource",
    "RecordingActionSink",
    "raw18_from_state",
]
