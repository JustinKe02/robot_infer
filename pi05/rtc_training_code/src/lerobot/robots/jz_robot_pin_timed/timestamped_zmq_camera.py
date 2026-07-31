#!/usr/bin/env python

from __future__ import annotations

import base64
import copy
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.zmq.camera_zmq import ZMQCamera
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

PROTOCOL = "jz_realsense_zmq"
PROTOCOL_VERSION = 1
TIMESTAMP_STAGE = "x86_after_zmq_receive_before_json_decode"


class ZMQCameraProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class TimestampedZMQFrame:
    image: np.ndarray
    sequence: int
    sequence_gap: int
    receive_wall_ns: int
    receive_monotonic_ns: int
    decode_completed_monotonic_ns: int
    camera_timing: dict[str, Any]
    decode_timing: dict[str, float]

    def copy(self) -> TimestampedZMQFrame:
        return replace(
            self,
            image=self.image.copy(),
            camera_timing=copy.deepcopy(self.camera_timing),
            decode_timing=copy.deepcopy(self.decode_timing),
        )

    def timing(self, now_monotonic_ns: int) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "timestamp_stage": TIMESTAMP_STAGE,
            "sequence": self.sequence,
            "sequence_gap": self.sequence_gap,
            "receive_wall_ns": self.receive_wall_ns,
            "receive_monotonic_ns": self.receive_monotonic_ns,
            "decode_completed_monotonic_ns": self.decode_completed_monotonic_ns,
            "age_ms": max(0.0, (now_monotonic_ns - self.receive_monotonic_ns) / 1_000_000),
            "camera_timing": copy.deepcopy(self.camera_timing),
            "decode_timing": copy.deepcopy(self.decode_timing),
        }


@dataclass(frozen=True)
class _ReceivedZMQMessage:
    payload: bytes
    receive_wall_ns: int
    receive_monotonic_ns: int


class TimestampedZMQCamera(ZMQCamera):
    """Strict latest-frame ZMQ receiver for the direct Orin RealSense protocol."""

    def __init__(
        self,
        config: ZMQCameraConfig,
        *,
        buffer_size: int = 8,
        stale_frame_timeout_ms: int = 1000,
        wall_time_ns: Any = time.time_ns,
        monotonic_ns: Any = time.monotonic_ns,
    ) -> None:
        super().__init__(config)
        if config.color_mode != ColorMode.RGB:
            raise ValueError("TimestampedZMQCamera output must be configured as RGB")
        if config.width is None or config.height is None or config.width <= 0 or config.height <= 0:
            raise ValueError("TimestampedZMQCamera width and height must be positive")
        if isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ValueError("buffer_size must be a positive integer")
        if (
            isinstance(stale_frame_timeout_ms, bool)
            or not isinstance(stale_frame_timeout_ms, int)
            or stale_frame_timeout_ms <= 0
        ):
            raise ValueError("stale_frame_timeout_ms must be a positive integer")

        self.buffer_size = buffer_size
        self.stale_frame_timeout_ms = stale_frame_timeout_ms
        self._wall_time_ns = wall_time_ns
        self._monotonic_ns = monotonic_ns
        self._condition = threading.Condition()
        self._frames: deque[TimestampedZMQFrame] = deque(maxlen=buffer_size)
        self._last_read_frame: TimestampedZMQFrame | None = None
        self._reader_thread: threading.Thread | None = None
        self._decoder_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._frame_ready = threading.Event()
        self._pending_messages: deque[_ReceivedZMQMessage] = deque(maxlen=buffer_size)
        self._last_received_sequence: int | None = None
        self._received_messages = 0
        self._accepted_frames = 0
        self._invalid_messages = 0
        self._duplicate_or_out_of_order_messages = 0
        self._sequence_gaps = 0
        self._raw_queue_drops = 0
        self._last_accepted_receive_monotonic_ns: int | None = None
        self._last_interarrival_ms: float | None = None
        self._max_interarrival_ms = 0.0
        self._last_capture_monotonic_ns: int | None = None
        self._last_capture_interarrival_ms: float | None = None
        self._max_capture_interarrival_ms = 0.0
        self._max_queue_delay_ms = 0.0
        self._max_decode_ms = 0.0
        self._last_error: BaseException | None = None

    @property
    def is_connected(self) -> bool:
        return (
            super().is_connected
            and not self._reader_stop.is_set()
            and self._reader_thread is not None
            and self._reader_thread.is_alive()
            and self._decoder_thread is not None
            and self._decoder_thread.is_alive()
            and self._frame_ready.is_set()
        )

    @property
    def diagnostics(self) -> dict[str, Any]:
        with self._condition:
            latest = self._frames[-1] if self._frames else None
            return {
                "received_messages": self._received_messages,
                "accepted_frames": self._accepted_frames,
                "invalid_messages": self._invalid_messages,
                "duplicate_or_out_of_order_messages": self._duplicate_or_out_of_order_messages,
                "sequence_gaps": self._sequence_gaps,
                "raw_queue_drops": self._raw_queue_drops,
                "last_interarrival_ms": self._last_interarrival_ms,
                "max_interarrival_ms": self._max_interarrival_ms,
                "last_capture_interarrival_ms": self._last_capture_interarrival_ms,
                "max_capture_interarrival_ms": self._max_capture_interarrival_ms,
                "max_queue_delay_ms": self._max_queue_delay_ms,
                "max_decode_ms": self._max_decode_ms,
                "latest_sequence": None if latest is None else latest.sequence,
                "latest_camera_capture_monotonic_ns": (
                    None if latest is None else latest.camera_timing["capture_monotonic_ns"]
                ),
                "latest_camera_capture_wall_ns": (
                    None if latest is None else latest.camera_timing["capture_wall_ns"]
                ),
                "latest_decode_timing": None if latest is None else copy.deepcopy(latest.decode_timing),
                "buffered_frames": len(self._frames),
                "pending_raw_messages": len(self._pending_messages),
                "raw_queue_capacity": self.buffer_size,
                "last_error": None if self._last_error is None else repr(self._last_error),
            }

    @property
    def last_read_timing(self) -> dict[str, Any] | None:
        with self._condition:
            frame = self._last_read_frame
            if frame is None:
                return None
            return frame.timing(self._monotonic_ns())

    def connect(self, warmup: bool = True) -> None:
        del warmup
        if (
            super().is_connected
            or (self._reader_thread is not None and self._reader_thread.is_alive())
            or (self._decoder_thread is not None and self._decoder_thread.is_alive())
        ):
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        import zmq

        with self._condition:
            self._frames.clear()
            self._pending_messages.clear()
            self._last_read_frame = None
            self._last_received_sequence = None
            self._received_messages = 0
            self._accepted_frames = 0
            self._invalid_messages = 0
            self._duplicate_or_out_of_order_messages = 0
            self._sequence_gaps = 0
            self._raw_queue_drops = 0
            self._last_accepted_receive_monotonic_ns = None
            self._last_interarrival_ms = None
            self._max_interarrival_ms = 0.0
            self._last_capture_monotonic_ns = None
            self._last_capture_interarrival_ms = None
            self._max_capture_interarrival_ms = 0.0
            self._max_queue_delay_ms = 0.0
            self._max_decode_ms = 0.0
            self._last_error = None
        self._reader_stop.clear()
        self._frame_ready.clear()
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.socket.setsockopt(zmq.RCVTIMEO, min(self.timeout_ms, 100))
            self.socket.setsockopt(zmq.RCVHWM, self.buffer_size)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(f"tcp://{self.server_address}:{self.port}")
            self._connected = True
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name=f"timed_zmq_reader_{self.camera_name}",
                daemon=False,
            )
            self._decoder_thread = threading.Thread(
                target=self._decoder_loop,
                name=f"timed_zmq_decoder_{self.camera_name}",
                daemon=False,
            )
            self._reader_thread.start()
            self._decoder_thread.start()
            if not self._frame_ready.wait(timeout=self.timeout_ms / 1000):
                with self._condition:
                    last_error = self._last_error
                message = f"Timed out waiting for {self.camera_name} on {self}"
                if last_error is not None:
                    message = f"{message}; last_error={last_error}"
                raise TimeoutError(message)
        except BaseException:
            self._disconnect_resources()
            raise

    def _reader_loop(self) -> None:
        import zmq

        while not self._reader_stop.is_set():
            try:
                if self.socket is None:
                    break
                message = self.socket.recv()
                receive_wall_ns = self._wall_time_ns()
                receive_monotonic_ns = self._monotonic_ns()
                with self._condition:
                    self._received_messages += 1
                    if self._last_accepted_receive_monotonic_ns is not None:
                        self._last_interarrival_ms = (
                            receive_monotonic_ns - self._last_accepted_receive_monotonic_ns
                        ) / 1_000_000
                        self._max_interarrival_ms = max(self._max_interarrival_ms, self._last_interarrival_ms)
                    self._last_accepted_receive_monotonic_ns = receive_monotonic_ns
                    if len(self._pending_messages) == self.buffer_size:
                        self._pending_messages.popleft()
                        self._raw_queue_drops += 1
                    self._pending_messages.append(
                        _ReceivedZMQMessage(message, receive_wall_ns, receive_monotonic_ns)
                    )
                    self._condition.notify_all()
            except zmq.Again:
                continue
            except BaseException as exc:
                if self._reader_stop.is_set():
                    break
                with self._condition:
                    self._last_error = exc
                logger.warning("Timed ZMQ receive failed for %s: %s", self.camera_name, exc)

    def _decoder_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: bool(self._pending_messages) or self._reader_stop.is_set())
                if not self._pending_messages and self._reader_stop.is_set():
                    break
                received = self._pending_messages.popleft()
            try:
                frame = self._decode_message(
                    received.payload,
                    received.receive_wall_ns,
                    received.receive_monotonic_ns,
                )
                with self._condition:
                    previous_sequence = self._last_received_sequence
                    if previous_sequence is not None and frame.sequence <= previous_sequence:
                        self._duplicate_or_out_of_order_messages += 1
                        raise ZMQCameraProtocolError(
                            f"{self.camera_name} sequence must increase: "
                            f"previous={previous_sequence}, received={frame.sequence}"
                        )
                    gap = 0 if previous_sequence is None else frame.sequence - previous_sequence - 1
                    frame = replace(frame, sequence_gap=gap)
                    capture_monotonic_ns = int(frame.camera_timing["capture_monotonic_ns"])
                    if self._last_capture_monotonic_ns is not None:
                        self._last_capture_interarrival_ms = (
                            capture_monotonic_ns - self._last_capture_monotonic_ns
                        ) / 1_000_000
                        self._max_capture_interarrival_ms = max(
                            self._max_capture_interarrival_ms,
                            self._last_capture_interarrival_ms,
                        )
                    self._last_capture_monotonic_ns = capture_monotonic_ns
                    self._max_queue_delay_ms = max(
                        self._max_queue_delay_ms,
                        frame.decode_timing["queue_delay_ms"],
                    )
                    self._max_decode_ms = max(
                        self._max_decode_ms,
                        frame.decode_timing["total_decode_ms"],
                    )
                    self._last_received_sequence = frame.sequence
                    self._sequence_gaps += gap
                    self._frames.append(frame)
                    self._accepted_frames += 1
                    self._last_error = None
                    self._frame_ready.set()
                    self._condition.notify_all()
            except BaseException as exc:
                if self._reader_stop.is_set():
                    break
                with self._condition:
                    self._invalid_messages += 1
                    self._last_error = exc
                logger.warning("Rejected timed ZMQ frame for %s: %s", self.camera_name, exc)

    def _decode_message(
        self, message: bytes, receive_wall_ns: int, receive_monotonic_ns: int
    ) -> TimestampedZMQFrame:
        decode_started_monotonic_ns = self._monotonic_ns()
        try:
            data = json.loads(message)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ZMQCameraProtocolError("message must be UTF-8 JSON") from exc
        json_completed_monotonic_ns = self._monotonic_ns()
        if not isinstance(data, dict):
            raise ZMQCameraProtocolError("message must be a JSON object")
        if data.get("protocol") != PROTOCOL or data.get("protocol_version") != PROTOCOL_VERSION:
            raise ZMQCameraProtocolError(
                f"expected {PROTOCOL}/v{PROTOCOL_VERSION}, got "
                f"{data.get('protocol')!r}/v{data.get('protocol_version')!r}"
            )

        images = self._single_camera_mapping(data, "images")
        timestamps = self._single_camera_mapping(data, "timestamps")
        camera_timings = self._single_camera_mapping(data, "camera_timing")
        if (
            set(images) != {self.camera_name}
            or set(timestamps) != {self.camera_name}
            or set(camera_timings) != {self.camera_name}
        ):
            raise ZMQCameraProtocolError(f"message keys must contain only camera {self.camera_name!r}")

        timing = camera_timings[self.camera_name]
        self._validate_camera_timing(timing)
        timestamp = timestamps[self.camera_name]
        if isinstance(timestamp, bool) or not isinstance(timestamp, int | float) or timestamp < 0:
            raise ZMQCameraProtocolError("timestamps entry must be a non-negative number")
        image_b64 = images[self.camera_name]
        if not isinstance(image_b64, str):
            raise ZMQCameraProtocolError("JPEG payload must be base64 text")
        metadata_validated_monotonic_ns = self._monotonic_ns()
        try:
            jpeg = base64.b64decode(image_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ZMQCameraProtocolError("JPEG payload is not valid base64") from exc
        base64_completed_monotonic_ns = self._monotonic_ns()
        if len(jpeg) != timing["payload_bytes"]:
            raise ZMQCameraProtocolError(
                f"JPEG payload size {len(jpeg)} does not match camera_timing {timing['payload_bytes']}"
            )

        decoded_bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            raise ZMQCameraProtocolError("OpenCV failed to decode JPEG")
        jpeg_completed_monotonic_ns = self._monotonic_ns()
        image = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        decode_completed_monotonic_ns = self._monotonic_ns()
        expected_shape = (self.config.height, self.config.width, 3)
        if image.dtype != np.uint8 or image.shape != expected_shape:
            raise ZMQCameraProtocolError(
                f"decoded image shape={image.shape} dtype={image.dtype}, expected={expected_shape}/uint8"
            )
        return TimestampedZMQFrame(
            image=image,
            sequence=timing["sequence"],
            sequence_gap=0,
            receive_wall_ns=receive_wall_ns,
            receive_monotonic_ns=receive_monotonic_ns,
            decode_completed_monotonic_ns=decode_completed_monotonic_ns,
            camera_timing=copy.deepcopy(timing),
            decode_timing={
                "queue_delay_ms": max(0.0, (decode_started_monotonic_ns - receive_monotonic_ns) / 1_000_000),
                "json_decode_ms": (json_completed_monotonic_ns - decode_started_monotonic_ns) / 1_000_000,
                "metadata_validation_ms": (metadata_validated_monotonic_ns - json_completed_monotonic_ns)
                / 1_000_000,
                "base64_decode_ms": (base64_completed_monotonic_ns - metadata_validated_monotonic_ns)
                / 1_000_000,
                "jpeg_decode_ms": (jpeg_completed_monotonic_ns - base64_completed_monotonic_ns) / 1_000_000,
                "color_convert_ms": (decode_completed_monotonic_ns - jpeg_completed_monotonic_ns) / 1_000_000,
                "total_decode_ms": (decode_completed_monotonic_ns - decode_started_monotonic_ns) / 1_000_000,
            },
        )

    @staticmethod
    def _single_camera_mapping(data: dict[str, Any], field: str) -> dict[str, Any]:
        value = data.get(field)
        if not isinstance(value, dict):
            raise ZMQCameraProtocolError(f"{field} must be an object")
        return value

    def _validate_camera_timing(self, timing: Any) -> None:
        if not isinstance(timing, dict):
            raise ZMQCameraProtocolError("camera_timing entry must be an object")
        required_fields = {
            "sequence",
            "timestamp_stage",
            "capture_wall_ns",
            "capture_monotonic_ns",
            "encode_completed_monotonic_ns",
            "width",
            "height",
            "channels",
            "pixel_format",
            "encoding",
            "jpeg_quality",
            "payload_bytes",
        }
        missing = sorted(required_fields - timing.keys())
        if missing:
            raise ZMQCameraProtocolError(f"camera_timing missing fields: {missing}")
        integer_fields = (
            "sequence",
            "capture_wall_ns",
            "capture_monotonic_ns",
            "encode_completed_monotonic_ns",
            "width",
            "height",
            "channels",
            "jpeg_quality",
            "payload_bytes",
        )
        for field in integer_fields:
            value = timing.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ZMQCameraProtocolError(f"camera_timing.{field} must be a non-negative integer")
        if timing["sequence"] <= 0:
            raise ZMQCameraProtocolError("camera_timing.sequence must be positive")
        if timing["timestamp_stage"] != "after_realsense_read_before_jpeg":
            raise ZMQCameraProtocolError("unexpected camera_timing.timestamp_stage")
        if timing["pixel_format"] != "RGB8" or timing["encoding"] != "jpeg":
            raise ZMQCameraProtocolError("camera_timing must describe RGB8/JPEG")
        if (timing["width"], timing["height"], timing["channels"]) != (
            self.config.width,
            self.config.height,
            3,
        ):
            raise ZMQCameraProtocolError("camera_timing dimensions do not match configuration")
        if not 1 <= timing["jpeg_quality"] <= 100:
            raise ZMQCameraProtocolError("camera_timing.jpeg_quality must be in 1..100")
        if timing["encode_completed_monotonic_ns"] < timing["capture_monotonic_ns"]:
            raise ZMQCameraProtocolError("Orin encode time precedes capture time")

    def _read_timed(
        self,
        target_monotonic_ns: int | None = None,
        *,
        max_receive_skew_ms: float | None = None,
        wait_timeout_ms: float = 0,
    ) -> TimestampedZMQFrame:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if max_receive_skew_ms is not None and max_receive_skew_ms <= 0:
            raise ValueError("max_receive_skew_ms must be positive")
        if wait_timeout_ms < 0:
            raise ValueError("wait_timeout_ms must be non-negative")
        deadline = time.monotonic() + wait_timeout_ms / 1000
        with self._condition:
            while True:
                frame = None
                if self._frames:
                    frame = (
                        self._frames[-1]
                        if target_monotonic_ns is None
                        else min(
                            self._frames,
                            key=lambda item: (
                                abs(item.receive_monotonic_ns - target_monotonic_ns),
                                -item.receive_monotonic_ns,
                            ),
                        )
                    )
                within_skew = frame is not None and (
                    max_receive_skew_ms is None
                    or target_monotonic_ns is None
                    or abs(frame.receive_monotonic_ns - target_monotonic_ns) / 1_000_000
                    <= max_receive_skew_ms
                )
                if within_skew:
                    break
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    if frame is None:
                        raise TimeoutError(f"No timed ZMQ frame available for {self.camera_name}")
                    selected_skew_ms = abs(frame.receive_monotonic_ns - target_monotonic_ns) / 1_000_000
                    raise TimeoutError(
                        f"Timed ZMQ camera {self.camera_name} did not produce an aligned frame "
                        f"within {wait_timeout_ms}ms: closest_skew_ms={selected_skew_ms:.3f}, "
                        f"limit_ms={max_receive_skew_ms}, diagnostics={self.diagnostics}"
                    )
                self._condition.wait(timeout=remaining_s)
        age_s = max(0.0, (self._monotonic_ns() - frame.receive_monotonic_ns) / 1_000_000_000)
        if age_s > self.stale_frame_timeout_ms / 1000:
            raise TimeoutError(
                f"Selected ZMQ frame for {self.camera_name} is stale: "
                f"age_s={age_s}, timeout_s={self.stale_frame_timeout_ms / 1000}"
            )
        result = frame.copy()
        with self._condition:
            self._last_read_frame = result
        return result

    def read_timed(self) -> TimestampedZMQFrame:
        return self._read_timed()

    def read_timed_nearest(
        self,
        target_monotonic_ns: int,
        *,
        max_receive_skew_ms: float | None = None,
        wait_timeout_ms: float = 0,
    ) -> TimestampedZMQFrame:
        if isinstance(target_monotonic_ns, bool) or not isinstance(target_monotonic_ns, int):
            raise ValueError("target_monotonic_ns must be an integer")
        return self._read_timed(
            target_monotonic_ns,
            max_receive_skew_ms=max_receive_skew_ms,
            wait_timeout_ms=wait_timeout_ms,
        )

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        if color_mode not in (None, ColorMode.RGB):
            raise ValueError("TimestampedZMQCamera only returns RGB")
        return self.read_timed().image

    def async_read(self, timeout_ms: float = 10000) -> np.ndarray:
        del timeout_ms
        return self.read()

    def _disconnect_resources(self) -> None:
        self._reader_stop.set()
        with self._condition:
            self._condition.notify_all()
        threads = [self._reader_thread, self._decoder_thread]
        for thread in threads:
            if thread is not None:
                thread.join(timeout=max(1.0, self.timeout_ms / 1000 + 1.0))
        live_threads = [thread.name for thread in threads if thread is not None and thread.is_alive()]
        self._reader_thread = None
        self._decoder_thread = None
        self._frame_ready.clear()
        self._cleanup()
        if live_threads:
            raise RuntimeError(f"Timed ZMQ workers did not stop for {self.camera_name}: {live_threads}")

    def disconnect(self) -> None:
        if not super().is_connected and self._reader_thread is None and self._decoder_thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")
        self._disconnect_resources()


__all__ = [
    "PROTOCOL",
    "PROTOCOL_VERSION",
    "TIMESTAMP_STAGE",
    "TimestampedZMQCamera",
    "TimestampedZMQFrame",
    "ZMQCameraProtocolError",
]
