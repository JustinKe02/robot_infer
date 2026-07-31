#!/usr/bin/env python

from __future__ import annotations

import argparse
import base64
import json
import math
import multiprocessing
import queue
import re
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from my_devs.orin_session.orin.camera_sink import MarkerDrivenCameraSink
from my_devs.orin_session.orin.readiness import publish_ready, remove_ready


PROTOCOL_VERSION = 1
ZMQ_SEND_HIGH_WATER_MARK = 64
CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DEFAULT_MAX_RECONNECT_ATTEMPTS = 3
DEFAULT_RECONNECT_BACKOFF_S = 0.25
MAX_RECONNECT_BACKOFF_S = 5.0
COLOR_CONFIG_SCHEMA_VERSION = 1
DIAGNOSTIC_SAMPLE_ORDINALS = frozenset({1, 150, 300})


def empty_camera_metrics() -> dict[str, int]:
    return {
        "frames_captured": 0,
        "frames_processed": 0,
        "frames_previewed": 0,
        "reconnect_generation": 0,
        "reconnect_attempts": 0,
        "reconnect_successes": 0,
        "reconnect_failures": 0,
        "reconnect_drop_count": 0,
        "device_frame_drop_count": 0,
        "formal_drop_evidence_count": 0,
        "preview_drop_count": 0,
        "first_capture_monotonic_ns": 0,
        "last_capture_monotonic_ns": 0,
        "capture_queue_depth": 0,
        "preview_queue_depth": 0,
        "capture_interval_p95_us": 0,
        "jpeg_encode_p95_us": 0,
    }


@dataclass(frozen=True)
class DirectCameraPreset:
    camera_id: str
    serial_number: str
    width: int
    height: int
    fps: int
    port: int
    jpeg_quality: int | None = None
    white_balance: float | None = None


@dataclass(frozen=True)
class ColorConfiguration:
    auto_white_balance: float
    white_balance: float
    white_balance_source: str
    white_balance_min: float
    white_balance_max: float
    white_balance_step: float
    white_balance_default: float

    def timing_fields(self) -> dict[str, Any]:
        return {
            "auto_white_balance": self.auto_white_balance,
            "white_balance": self.white_balance,
            "white_balance_source": self.white_balance_source,
            "color_config_schema_version": COLOR_CONFIG_SCHEMA_VERSION,
        }


def _option_range_text(option_range: Any | None) -> str:
    if option_range is None:
        return "unavailable"
    return (
        f"[{float(option_range.min):g},{float(option_range.max):g}] "
        f"step={float(option_range.step):g} default={float(option_range.default):g}"
    )


def _color_config_error(
    preset: DirectCameraPreset,
    *,
    option: str,
    requested: Any,
    actual: Any,
    option_range: Any | None,
    reason: str,
) -> RuntimeError:
    return RuntimeError(
        "RealSense color configuration failed "
        f"camera_id={preset.camera_id} serial={preset.serial_number} option={option} "
        f"requested={requested} actual={actual} supported_range={_option_range_text(option_range)} "
        f"reason={reason}"
    )


def configure_realsense_color_sensor(
    camera: Any,
    preset: DirectCameraPreset,
    *,
    requested_white_balance: float | None = None,
    white_balance_source: str | None = None,
) -> ColorConfiguration:
    profile = getattr(camera, "rs_profile", None)
    if profile is None:
        raise _color_config_error(
            preset,
            option="color_sensor",
            requested="active_color_sensor",
            actual="unavailable",
            option_range=None,
            reason="camera has no active RealSense pipeline profile",
        )
    awb_option = rs.option.enable_auto_white_balance
    white_balance_option = rs.option.white_balance
    device = profile.get_device()
    try:
        sensor = device.first_color_sensor()
    except Exception as first_color_error:
        candidates = []
        candidate_names = []
        try:
            for candidate in device.query_sensors():
                if candidate.supports(awb_option) and candidate.supports(white_balance_option):
                    candidates.append(candidate)
                    try:
                        candidate_names.append(candidate.get_info(rs.camera_info.name))
                    except Exception:
                        candidate_names.append("unknown")
        except Exception as query_error:
            raise _color_config_error(
                preset,
                option="color_sensor",
                requested="unique_sensor_supporting_awb_and_white_balance",
                actual="unavailable",
                option_range=None,
                reason=(
                    f"first_color_sensor={type(first_color_error).__name__}: {first_color_error}; "
                    f"query_sensors={type(query_error).__name__}: {query_error}"
                ),
            ) from query_error
        if len(candidates) != 1:
            raise _color_config_error(
                preset,
                option="color_sensor",
                requested="unique_sensor_supporting_awb_and_white_balance",
                actual=f"candidate_count={len(candidates)} names={candidate_names}",
                option_range=None,
                reason=f"first_color_sensor={type(first_color_error).__name__}: {first_color_error}",
            ) from first_color_error
        sensor = candidates[0]
    for option, option_name in (
        (awb_option, "enable_auto_white_balance"),
        (white_balance_option, "white_balance"),
    ):
        try:
            supported = bool(sensor.supports(option))
        except Exception as exc:
            raise _color_config_error(
                preset,
                option=option_name,
                requested=0 if option == awb_option else requested_white_balance,
                actual="unavailable",
                option_range=None,
                reason=f"supports check failed: {type(exc).__name__}: {exc}",
            ) from exc
        if not supported:
            raise _color_config_error(
                preset,
                option=option_name,
                requested=0 if option == awb_option else requested_white_balance,
                actual="unsupported",
                option_range=None,
                reason="option is not supported by the active color sensor",
            )

    try:
        awb_range = sensor.get_option_range(awb_option)
        white_balance_range = sensor.get_option_range(white_balance_option)
    except Exception as exc:
        raise _color_config_error(
            preset,
            option="white_balance",
            requested=requested_white_balance,
            actual="unavailable",
            option_range=None,
            reason=f"range query failed: {type(exc).__name__}: {exc}",
        ) from exc

    try:
        sensor.set_option(awb_option, 0.0)
        awb_actual = float(sensor.get_option(awb_option))
    except Exception as exc:
        raise _color_config_error(
            preset,
            option="enable_auto_white_balance",
            requested=0,
            actual="unavailable",
            option_range=awb_range,
            reason=f"set/readback failed: {type(exc).__name__}: {exc}",
        ) from exc
    if abs(awb_actual) > 1e-6:
        raise _color_config_error(
            preset,
            option="enable_auto_white_balance",
            requested=0,
            actual=awb_actual,
            option_range=awb_range,
            reason="readback remained enabled",
        )

    requested = preset.white_balance if requested_white_balance is None else requested_white_balance
    source = white_balance_source or ("configured" if requested is not None else "device_current")
    if requested is not None:
        requested = float(requested)
        minimum = float(white_balance_range.min)
        maximum = float(white_balance_range.max)
        step = float(white_balance_range.step)
        if not math.isfinite(requested) or requested < minimum or requested > maximum:
            raise _color_config_error(
                preset,
                option="white_balance",
                requested=requested,
                actual="not_set",
                option_range=white_balance_range,
                reason="requested value is outside the device range",
            )
        if step > 0 and abs((requested - minimum) / step - round((requested - minimum) / step)) > 1e-6:
            raise _color_config_error(
                preset,
                option="white_balance",
                requested=requested,
                actual="not_set",
                option_range=white_balance_range,
                reason="requested value is not aligned to the device step",
            )
        try:
            sensor.set_option(white_balance_option, requested)
        except Exception as exc:
            raise _color_config_error(
                preset,
                option="white_balance",
                requested=requested,
                actual="unavailable",
                option_range=white_balance_range,
                reason=f"set failed: {type(exc).__name__}: {exc}",
            ) from exc

    try:
        white_balance_actual = float(sensor.get_option(white_balance_option))
    except Exception as exc:
        raise _color_config_error(
            preset,
            option="white_balance",
            requested=requested,
            actual="unavailable",
            option_range=white_balance_range,
            reason=f"readback failed: {type(exc).__name__}: {exc}",
        ) from exc
    minimum = float(white_balance_range.min)
    maximum = float(white_balance_range.max)
    if not math.isfinite(white_balance_actual) or not minimum <= white_balance_actual <= maximum:
        raise _color_config_error(
            preset,
            option="white_balance",
            requested=requested,
            actual=white_balance_actual,
            option_range=white_balance_range,
            reason="readback is outside the device range",
        )
    if requested is not None and abs(white_balance_actual - requested) > max(
        1e-6, abs(float(white_balance_range.step)) * 0.51
    ):
        raise _color_config_error(
            preset,
            option="white_balance",
            requested=requested,
            actual=white_balance_actual,
            option_range=white_balance_range,
            reason="readback does not match the requested fixed value",
        )

    configuration = ColorConfiguration(
        auto_white_balance=awb_actual,
        white_balance=white_balance_actual,
        white_balance_source=source,
        white_balance_min=minimum,
        white_balance_max=maximum,
        white_balance_step=float(white_balance_range.step),
        white_balance_default=float(white_balance_range.default),
    )
    print(
        "[direct realsense zmq] color_config "
        f"camera_id={preset.camera_id} serial={preset.serial_number} "
        f"auto_white_balance_requested=0 auto_white_balance_actual={awb_actual:g} "
        f"white_balance_requested={requested if requested is not None else 'device_current'} "
        f"white_balance_actual={white_balance_actual:g} "
        f"white_balance_min={minimum:g} white_balance_max={maximum:g} "
        f"white_balance_step={float(white_balance_range.step):g} "
        f"white_balance_default={float(white_balance_range.default):g} "
        f"white_balance_source={source} color_config_schema_version={COLOR_CONFIG_SCHEMA_VERSION}",
        flush=True,
    )
    return configuration


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    image: np.ndarray
    capture_wall_ns: int
    capture_monotonic_ns: int
    read_enter_monotonic_ns: int
    read_return_monotonic_ns: int
    capture_interval_ns: int | None
    capture_thread_interval_ns: int | None
    reconnect_generation: int
    reconnect_drop_count_since_previous: int
    realsense_frame_number: int | None
    realsense_frame_gap: int | None
    realsense_device_timestamp_ms: float | None
    realsense_device_interval_ms: float | None
    realsense_timestamp_domain: str | None
    formal_episode_context: dict[str, Any] | None
    color_configuration: ColorConfiguration | None

    @property
    def drop_count_since_previous(self) -> int:
        return self.reconnect_drop_count_since_previous + (self.realsense_frame_gap or 0)


@dataclass(frozen=True)
class PreviewFrame:
    captured: CapturedFrame
    jpeg: bytes
    jpeg_quality: int
    publisher_dequeue_monotonic_ns: int
    encode_started_monotonic_ns: int
    encode_completed_monotonic_ns: int
    formal_spool_completed_monotonic_ns: int


CAMERA_PRESETS = {
    "camera_head": DirectCameraPreset("camera_head", "318122304464", 1280, 720, 30, 5555),
    "camera_left": DirectCameraPreset("camera_left", "230422272306", 640, 480, 30, 5556),
    "camera_right": DirectCameraPreset("camera_right", "230322272819", 640, 480, 30, 5557),
}


def encode_rgb_jpeg(image: np.ndarray, quality: int) -> bytes:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected uint8 RGB image, got shape={image.shape} dtype={image.dtype}")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode JPEG")
    return encoded.tobytes()


def save_color_diagnostic_sample(
    output_dir: str | Path,
    preset: DirectCameraPreset,
    *,
    sequence: int,
    rgb: np.ndarray,
    jpeg: bytes,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{preset.camera_id}_seq_{sequence}"
    rgb_path = directory / f"{stem}_encoder_input_rgb.png"
    jpeg_path = directory / f"{stem}_sent.jpg"
    if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write diagnostic RGB PNG: {rgb_path}")
    jpeg_path.write_bytes(jpeg)
    return rgb_path, jpeg_path


def build_message(
    preset: DirectCameraPreset,
    *,
    sequence: int,
    capture_wall_ns: int,
    capture_monotonic_ns: int,
    encode_completed_monotonic_ns: int,
    jpeg: bytes,
    jpeg_quality: int,
    image_b64: str | None = None,
    extra_timing: dict[str, Any] | None = None,
) -> str:
    if image_b64 is None:
        image_b64 = base64.b64encode(jpeg).decode("ascii")
    timing = {
        "sequence": sequence,
        "timestamp_stage": "after_realsense_read_before_jpeg",
        "capture_wall_ns": capture_wall_ns,
        "capture_monotonic_ns": capture_monotonic_ns,
        "encode_completed_monotonic_ns": encode_completed_monotonic_ns,
        "width": preset.width,
        "height": preset.height,
        "channels": 3,
        "pixel_format": "RGB8",
        "encoding": "jpeg",
        "jpeg_quality": jpeg_quality,
        "payload_bytes": len(jpeg),
    }
    if extra_timing:
        timing.update(extra_timing)
    return json.dumps(
        {
            "protocol": "jz_realsense_zmq",
            "protocol_version": PROTOCOL_VERSION,
            "timestamps": {preset.camera_id: capture_wall_ns / 1_000_000_000},
            "images": {preset.camera_id: image_b64},
            "camera_timing": {preset.camera_id: timing},
        },
        separators=(",", ":"),
    )


class CameraPublisherWorker:
    def __init__(
        self,
        preset: DirectCameraPreset,
        context: zmq.Context,
        *,
        bind_host: str,
        jpeg_quality: int,
        count: int,
        stop_event: threading.Event,
        preview_enabled: bool = True,
        preview_jpeg_quality: int | None = None,
        camera_factory: Any = RealSenseCamera,
        enforce_color_configuration: bool | None = None,
        max_reconnect_attempts: int = DEFAULT_MAX_RECONNECT_ATTEMPTS,
        reconnect_backoff_s: float = DEFAULT_RECONNECT_BACKOFF_S,
        trace_every_frame: bool = False,
        session_marker: str | None = None,
        raw_root: str | None = None,
        spool_chunk_size_bytes: int = 256 * 1024 * 1024,
        spool_fsync_each_frame: bool = False,
        quiesce_ack_root: str | None = None,
        diagnostic_dir: str | None = None,
    ) -> None:
        self.preset = preset
        self.context = context
        self.bind_host = bind_host
        self.jpeg_quality = preset.jpeg_quality if preset.jpeg_quality is not None else jpeg_quality
        self.preview_enabled = preview_enabled
        self.preview_jpeg_quality = (
            jpeg_quality if preview_jpeg_quality is None else preview_jpeg_quality
        )
        self.count = count
        self.stop_event = stop_event
        self.camera_factory = camera_factory
        self.enforce_color_configuration = (
            camera_factory is RealSenseCamera
            if enforce_color_configuration is None
            else bool(enforce_color_configuration)
        )
        self.color_configuration: ColorConfiguration | None = None
        if (
            isinstance(max_reconnect_attempts, bool)
            or not isinstance(max_reconnect_attempts, int)
            or max_reconnect_attempts < 0
        ):
            raise ValueError("max_reconnect_attempts must be a non-negative integer")
        if (
            isinstance(reconnect_backoff_s, bool)
            or not isinstance(reconnect_backoff_s, (int, float))
            or not math.isfinite(float(reconnect_backoff_s))
            or reconnect_backoff_s < 0
        ):
            raise ValueError("reconnect_backoff_s must be a finite non-negative number")
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_backoff_s = float(reconnect_backoff_s)
        self.trace_every_frame = trace_every_frame
        if (session_marker is None) != (raw_root is None):
            raise ValueError("session_marker and raw_root must be configured together")
        self.session_marker = session_marker
        self.raw_root = raw_root
        self.spool_chunk_size_bytes = spool_chunk_size_bytes
        self.spool_fsync_each_frame = spool_fsync_each_frame
        self.quiesce_ack_root = quiesce_ack_root
        self.diagnostic_dir = diagnostic_dir
        self.ready = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name=f"jz-realsense-zmq-{preset.camera_id}",
            daemon=False,
        )
        self.exception: BaseException | None = None
        self.frames_sent = 0
        self.frames_previewed = 0
        self.preview_drops = 0
        self.frames_captured = 0
        self.reconnect_generation = 0
        self.reconnect_attempts = 0
        self.reconnect_successes = 0
        self.reconnect_failures = 0
        self.reconnect_drop_count = 0
        self.device_frame_drop_count = 0
        self.formal_drop_evidence_count = 0
        self.first_capture_monotonic_ns = 0
        self.last_capture_monotonic_ns = 0
        self.capture_queue: queue.Queue[CapturedFrame] = queue.Queue(maxsize=8)
        self.preview_queue: queue.Queue[PreviewFrame] = queue.Queue(maxsize=1)
        self.formal_done = threading.Event()
        self.capture_intervals_ms: deque[float] = deque(maxlen=300)
        self.read_durations_ms: deque[float] = deque(maxlen=300)
        self.encode_durations_ms: deque[float] = deque(maxlen=300)
        self.metrics_lock = threading.Lock()

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)

    def metrics_snapshot(self) -> dict[str, int]:
        with self.metrics_lock:
            capture_p95_us = round(self._percentile(self.capture_intervals_ms, 0.95) * 1000)
            encode_p95_us = round(self._percentile(self.encode_durations_ms, 0.95) * 1000)
        return {
            "frames_captured": self.frames_captured,
            "frames_processed": self.frames_sent,
            "frames_previewed": self.frames_previewed,
            "reconnect_generation": self.reconnect_generation,
            "reconnect_attempts": self.reconnect_attempts,
            "reconnect_successes": self.reconnect_successes,
            "reconnect_failures": self.reconnect_failures,
            "reconnect_drop_count": self.reconnect_drop_count,
            "device_frame_drop_count": self.device_frame_drop_count,
            "formal_drop_evidence_count": self.formal_drop_evidence_count,
            "preview_drop_count": self.preview_drops,
            "first_capture_monotonic_ns": self.first_capture_monotonic_ns,
            "last_capture_monotonic_ns": self.last_capture_monotonic_ns,
            "capture_queue_depth": self.capture_queue.qsize(),
            "preview_queue_depth": self.preview_queue.qsize(),
            "capture_interval_p95_us": capture_p95_us,
            "jpeg_encode_p95_us": encode_p95_us,
        }

    def _run(self) -> None:
        capture_ready = threading.Event()
        formal_ready = threading.Event()
        preview_ready = threading.Event()
        formal_sink = (
            MarkerDrivenCameraSink(
                marker_path=self.session_marker,
                raw_root=self.raw_root,
                camera_id=self.preset.camera_id,
                camera_serial=self.preset.serial_number,
                configured_fps=self.preset.fps,
                width=self.preset.width,
                height=self.preset.height,
                pixel_format="rgb8",
                jpeg_quality=self.jpeg_quality,
                chunk_size_bytes=self.spool_chunk_size_bytes,
                fsync_each_frame=self.spool_fsync_each_frame,
                quiesce_ack_root=self.quiesce_ack_root,
            )
            if self.session_marker is not None and self.raw_root is not None
            else None
        )
        capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(capture_ready, formal_sink),
            name=f"jz-realsense-capture-{self.preset.camera_id}",
            daemon=False,
        )
        formal_thread = threading.Thread(
            target=self._publisher_loop,
            args=(formal_ready, formal_sink),
            name=f"jz-realsense-formal-{self.preset.camera_id}",
            daemon=False,
        )
        preview_thread = threading.Thread(
            target=self._preview_loop,
            args=(preview_ready,),
            name=f"jz-realsense-preview-{self.preset.camera_id}",
            daemon=False,
        )
        try:
            capture_thread.start()
            formal_thread.start()
            preview_thread.start()
            deadline = time.monotonic() + 20
            while not (
                capture_ready.is_set() and formal_ready.is_set() and preview_ready.is_set()
            ):
                if self.exception is not None:
                    raise self.exception
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out starting {self.preset.camera_id} workers")
                time.sleep(0.02)
            self.ready.set()
            preview_endpoint = (
                f"tcp://{self.bind_host}:{self.preset.port}"
                if self.preview_enabled
                else "disabled"
            )
            print(
                f"[direct realsense zmq] ready camera={self.preset.camera_id} "
                f"serial={self.preset.serial_number} "
                f"preview={preview_endpoint} "
                f"format=RGB8 size={self.preset.width}x{self.preset.height} fps={self.preset.fps} "
                "workers=capture+formal+preview formal_queue_size=8 preview_queue_size=1 "
                f"reconnect_generation={self.reconnect_generation} "
                f"reconnect_attempts={self.reconnect_attempts} "
                f"reconnect_failures={self.reconnect_failures} "
                f"reconnect_drop_count={self.reconnect_drop_count} "
                f"auto_white_balance={None if self.color_configuration is None else self.color_configuration.auto_white_balance} "
                f"white_balance={None if self.color_configuration is None else self.color_configuration.white_balance} "
                f"white_balance_source={None if self.color_configuration is None else self.color_configuration.white_balance_source}",
                flush=True,
            )
            while not self.stop_event.is_set():
                if self.exception is not None:
                    raise self.exception
                if self.count > 0 and self.frames_sent >= self.count:
                    break
                time.sleep(0.02)
        except BaseException as exc:
            self.exception = exc
            self.stop_event.set()
        finally:
            capture_thread.join(timeout=5)
            formal_thread.join(timeout=5)
            preview_thread.join(timeout=5)

    def _make_camera(self) -> Any:
        return self.camera_factory(
            RealSenseCameraConfig(
                serial_number_or_name=self.preset.serial_number,
                fps=self.preset.fps,
                width=self.preset.width,
                height=self.preset.height,
                color_mode=ColorMode.RGB,
                use_depth=False,
                warmup_s=1,
            )
        )

    def _read_frame(self, camera: Any) -> tuple[np.ndarray, int | None, float | None, str | None]:
        pipeline = getattr(camera, "rs_pipeline", None)
        if pipeline is None or not isinstance(camera, RealSenseCamera):
            return camera.read(timeout_ms=500), None, None, None

        ok, frames = pipeline.try_wait_for_frames(timeout_ms=500)
        if not ok or frames is None:
            raise RuntimeError(f"{self.preset.camera_id} RealSense read failed status={ok}")
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"{self.preset.camera_id} RealSense frameset has no color frame")
        frame_number = int(color_frame.get_frame_number())
        device_timestamp_ms = float(color_frame.get_timestamp())
        timestamp_domain = str(color_frame.get_frame_timestamp_domain())
        image = np.asanyarray(color_frame.get_data())
        return camera._postprocess_image(image), frame_number, device_timestamp_ms, timestamp_domain

    def _safe_disconnect(self, camera: Any | None) -> None:
        if camera is None:
            return
        try:
            disconnect = getattr(camera, "disconnect", None)
            if callable(disconnect):
                disconnect()
        except Exception as exc:
            print(
                f"[direct realsense zmq] disconnect_error camera={self.preset.camera_id} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )

    def _record_capture_failure(
        self,
        *,
        phase: str,
        error: Exception,
        consecutive_failures: int,
    ) -> int:
        failures = consecutive_failures + 1
        self.reconnect_failures += 1
        self.reconnect_drop_count += 1
        print(
            f"[direct realsense zmq] capture_recoverable_error camera={self.preset.camera_id} "
            f"phase={phase} consecutive_failures={failures} "
            f"max_reconnect_attempts={self.max_reconnect_attempts} "
            f"error={type(error).__name__}: {error}",
            flush=True,
        )
        return failures

    def _wait_reconnect_backoff(self, consecutive_failures: int) -> bool:
        delay_s = min(
            self.reconnect_backoff_s * (2 ** min(8, max(0, consecutive_failures - 1))),
            MAX_RECONNECT_BACKOFF_S,
        )
        return bool(self.stop_event.wait(delay_s)) if delay_s > 0 else self.stop_event.is_set()

    def _reconnect_exhausted_error(
        self,
        *,
        phase: str,
        consecutive_failures: int,
        error: Exception,
    ) -> RuntimeError:
        return RuntimeError(
            f"{self.preset.camera_id} capture reconnect exhausted after "
            f"{consecutive_failures} consecutive {phase} failure(s); "
            f"max_reconnect_attempts={self.max_reconnect_attempts}; "
            f"last_error={type(error).__name__}: {error}"
        )

    def _capture_loop(
        self,
        ready: threading.Event,
        formal_sink: MarkerDrivenCameraSink | None,
    ) -> None:
        camera: Any | None = None
        previous_capture_ns: int | None = None
        previous_loop_enter_ns: int | None = None
        previous_frame_number: int | None = None
        previous_device_timestamp_ms: float | None = None
        consecutive_failures = 0
        pending_reconnect_drops = 0
        recovering = False
        try:
            while not self.stop_event.is_set() and (
                self.count <= 0 or self.frames_captured < self.count
            ):
                if camera is None:
                    if recovering:
                        self.reconnect_attempts += 1
                        if self._wait_reconnect_backoff(consecutive_failures):
                            break
                    candidate: Any | None = None
                    try:
                        candidate = self._make_camera()
                        candidate.connect()
                        if not getattr(candidate, "is_connected", False):
                            raise RuntimeError("camera connect returned without is_connected=true")
                        if self.enforce_color_configuration:
                            locked_white_balance = (
                                None
                                if self.color_configuration is None
                                else self.color_configuration.white_balance
                            )
                            source = (
                                "configured"
                                if self.preset.white_balance is not None
                                else "device_current"
                            )
                            configured = configure_realsense_color_sensor(
                                candidate,
                                self.preset,
                                requested_white_balance=locked_white_balance,
                                white_balance_source=source,
                            )
                            if (
                                self.color_configuration is not None
                                and configured.white_balance != self.color_configuration.white_balance
                            ):
                                raise _color_config_error(
                                    self.preset,
                                    option="white_balance",
                                    requested=self.color_configuration.white_balance,
                                    actual=configured.white_balance,
                                    option_range=None,
                                    reason="fixed value changed across reconnect",
                                )
                            self.color_configuration = configured
                    except Exception as exc:
                        self._safe_disconnect(candidate)
                        consecutive_failures = self._record_capture_failure(
                            phase="connect",
                            error=exc,
                            consecutive_failures=consecutive_failures,
                        )
                        pending_reconnect_drops += 1
                        recovering = True
                        if consecutive_failures > self.max_reconnect_attempts:
                            raise self._reconnect_exhausted_error(
                                phase="connect",
                                consecutive_failures=consecutive_failures,
                                error=exc,
                            ) from exc
                        continue
                    camera = candidate
                    self.reconnect_generation += 1
                    if recovering:
                        self.reconnect_successes += 1
                    ready.set()

                loop_enter_ns = time.monotonic_ns()
                read_enter_ns = loop_enter_ns
                marker_before_read = (
                    formal_sink.capture_marker() if formal_sink is not None else None
                )
                try:
                    image, frame_number, device_timestamp_ms, timestamp_domain = self._read_frame(
                        camera
                    )
                    expected_shape = (self.preset.height, self.preset.width, 3)
                    if image.dtype != np.uint8 or image.shape != expected_shape:
                        raise RuntimeError(
                            f"unexpected {self.preset.camera_id} frame shape={image.shape} "
                            f"dtype={image.dtype} expected={expected_shape}/uint8"
                        )
                except Exception as exc:
                    self._safe_disconnect(camera)
                    camera = None
                    consecutive_failures = self._record_capture_failure(
                        phase="read",
                        error=exc,
                        consecutive_failures=consecutive_failures,
                    )
                    pending_reconnect_drops += 1
                    recovering = True
                    previous_frame_number = None
                    previous_device_timestamp_ms = None
                    if consecutive_failures > self.max_reconnect_attempts:
                        raise self._reconnect_exhausted_error(
                            phase="read",
                            consecutive_failures=consecutive_failures,
                            error=exc,
                        ) from exc
                    continue
                read_return_ns = time.monotonic_ns()
                capture_wall_ns = time.time_ns()
                formal_episode_context = (
                    formal_sink.bind_capture_context(
                        marker_before_read=marker_before_read,
                        capture_monotonic_ns=read_return_ns,
                    )
                    if formal_sink is not None
                    else None
                )
                sequence = self.frames_captured + 1
                frame_gap = (
                    None
                    if previous_frame_number is None or frame_number is None
                    else max(0, frame_number - previous_frame_number - 1)
                )
                captured = CapturedFrame(
                    sequence=sequence,
                    image=image,
                    capture_wall_ns=capture_wall_ns,
                    capture_monotonic_ns=read_return_ns,
                    read_enter_monotonic_ns=read_enter_ns,
                    read_return_monotonic_ns=read_return_ns,
                    capture_interval_ns=(
                        None if previous_capture_ns is None else read_return_ns - previous_capture_ns
                    ),
                    capture_thread_interval_ns=(
                        None if previous_loop_enter_ns is None else loop_enter_ns - previous_loop_enter_ns
                    ),
                    reconnect_generation=self.reconnect_generation,
                    reconnect_drop_count_since_previous=pending_reconnect_drops,
                    realsense_frame_number=frame_number,
                    realsense_frame_gap=frame_gap,
                    realsense_device_timestamp_ms=device_timestamp_ms,
                    realsense_device_interval_ms=(
                        None
                        if previous_device_timestamp_ms is None or device_timestamp_ms is None
                        else device_timestamp_ms - previous_device_timestamp_ms
                    ),
                    realsense_timestamp_domain=timestamp_domain,
                    formal_episode_context=formal_episode_context,
                    color_configuration=self.color_configuration,
                )
                try:
                    self.capture_queue.put(captured, timeout=0.5)
                except queue.Full as exc:
                    raise RuntimeError(f"{self.preset.camera_id} publisher queue remained full") from exc
                self.frames_captured = sequence
                if self.first_capture_monotonic_ns == 0:
                    self.first_capture_monotonic_ns = read_return_ns
                self.last_capture_monotonic_ns = read_return_ns
                self.device_frame_drop_count += frame_gap or 0
                pending_reconnect_drops = 0
                consecutive_failures = 0
                recovering = False
                previous_capture_ns = read_return_ns
                previous_loop_enter_ns = loop_enter_ns
                previous_frame_number = frame_number
                previous_device_timestamp_ms = device_timestamp_ms
        except BaseException as exc:
            self.exception = exc
            self.stop_event.set()
        finally:
            self._safe_disconnect(camera)

    @staticmethod
    def _percentile(values: deque[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
        return ordered[index]

    def _publisher_loop(
        self,
        ready: threading.Event,
        formal_sink: MarkerDrivenCameraSink | None,
    ) -> None:
        started_ns = time.monotonic_ns()
        ready.set()
        try:
            while not self.stop_event.is_set() or not self.capture_queue.empty():
                if self.count > 0 and self.frames_sent >= self.count:
                    break
                try:
                    captured = self.capture_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                dequeue_ns = time.monotonic_ns()
                encode_start_ns = dequeue_ns
                formal_active = (
                    formal_sink is not None and captured.formal_episode_context is not None
                )
                if formal_sink is not None and not formal_active:
                    formal_sink.transition_to_context(None)
                formal_jpeg = (
                    encode_rgb_jpeg(captured.image, self.jpeg_quality) if formal_active else None
                )
                encode_completed_ns = time.monotonic_ns()
                formal_spool_completed_ns = encode_completed_ns
                if formal_active:
                    assert formal_sink is not None
                    assert formal_jpeg is not None
                    formal_sink.append_bound_frame(
                        captured.formal_episode_context,
                        formal_jpeg,
                        camera_sequence=captured.sequence,
                        reconnect_generation=captured.reconnect_generation,
                        realsense_frame_number=(
                            captured.sequence
                            if captured.realsense_frame_number is None
                            else captured.realsense_frame_number
                        ),
                        realsense_device_timestamp_ms=(
                            captured.capture_monotonic_ns / 1_000_000
                            if captured.realsense_device_timestamp_ms is None
                            else captured.realsense_device_timestamp_ms
                        ),
                        realsense_timestamp_domain=(
                            "unknown"
                            if captured.realsense_timestamp_domain is None
                            else captured.realsense_timestamp_domain
                        ),
                        read_enter_monotonic_ns=captured.read_enter_monotonic_ns,
                        read_return_monotonic_ns=captured.read_return_monotonic_ns,
                        capture_wall_ns=captured.capture_wall_ns,
                        encode_completed_monotonic_ns=encode_completed_ns,
                        drop_count_since_previous=captured.drop_count_since_previous,
                    )
                    self.formal_drop_evidence_count += captured.drop_count_since_previous
                    formal_spool_completed_ns = time.monotonic_ns()
                preview_jpeg: bytes | None = None
                effective_preview_quality = self.preview_jpeg_quality
                preview_encode_completed_ns = formal_spool_completed_ns
                if self.preview_enabled:
                    if formal_jpeg is not None:
                        preview_jpeg = formal_jpeg
                        effective_preview_quality = self.jpeg_quality
                    else:
                        preview_jpeg = encode_rgb_jpeg(
                            captured.image,
                            self.preview_jpeg_quality,
                        )
                    preview_encode_completed_ns = time.monotonic_ns()
                if preview_jpeg is not None:
                    preview_item = PreviewFrame(
                        captured=captured,
                        jpeg=preview_jpeg,
                        jpeg_quality=effective_preview_quality,
                        publisher_dequeue_monotonic_ns=dequeue_ns,
                        encode_started_monotonic_ns=encode_start_ns,
                        encode_completed_monotonic_ns=preview_encode_completed_ns,
                        formal_spool_completed_monotonic_ns=formal_spool_completed_ns,
                    )
                    try:
                        self.preview_queue.put_nowait(preview_item)
                    except queue.Full:
                        try:
                            dropped = self.preview_queue.get_nowait()
                        except queue.Empty:
                            dropped = None
                        self.preview_drops += 1
                        self.preview_queue.put_nowait(preview_item)
                        dropped_sequence = (
                            "unknown" if dropped is None else str(dropped.captured.sequence)
                        )
                        print(
                            f"[direct realsense zmq] preview_replace camera={self.preset.camera_id} "
                            f"dropped_sequence={dropped_sequence} latest_sequence={captured.sequence} "
                            "formal_spool=preserved",
                            flush=True,
                        )
                diagnostic_jpeg = preview_jpeg or formal_jpeg
                if (
                    self.diagnostic_dir is not None
                    and captured.sequence in DIAGNOSTIC_SAMPLE_ORDINALS
                ):
                    if diagnostic_jpeg is None:
                        diagnostic_jpeg = encode_rgb_jpeg(
                            captured.image, self.preview_jpeg_quality
                        )
                    rgb_path, jpeg_path = save_color_diagnostic_sample(
                        self.diagnostic_dir,
                        self.preset,
                        sequence=captured.sequence,
                        rgb=captured.image,
                        jpeg=diagnostic_jpeg,
                    )
                    print(
                        f"[direct realsense zmq] diagnostic_sample camera={self.preset.camera_id} "
                        f"sequence={captured.sequence} rgb_png={rgb_path} sent_jpeg={jpeg_path}",
                        flush=True,
                    )
                stage_completed_ns = time.monotonic_ns()
                self.frames_sent = captured.sequence
                self._log_stage_trace(
                    captured,
                    dequeue_ns=dequeue_ns,
                    encode_start_ns=encode_start_ns,
                    encode_completed_ns=preview_encode_completed_ns,
                    base64_start_ns=stage_completed_ns,
                    base64_completed_ns=stage_completed_ns,
                    json_start_ns=stage_completed_ns,
                    json_completed_ns=stage_completed_ns,
                    send_start_ns=stage_completed_ns,
                    send_completed_ns=stage_completed_ns,
                    jpeg_bytes=len(preview_jpeg or formal_jpeg or b""),
                    started_ns=started_ns,
                )
        except BaseException as exc:
            self.exception = exc
            self.stop_event.set()
        finally:
            self.formal_done.set()
            if formal_sink is not None:
                formal_sink.close()

    def _preview_loop(self, ready: threading.Event) -> None:
        if not self.preview_enabled:
            ready.set()
            return
        preview_socket = self.context.socket(zmq.PUB)
        preview_socket.setsockopt(zmq.SNDHWM, ZMQ_SEND_HIGH_WATER_MARK)
        preview_socket.setsockopt(zmq.LINGER, 0)
        preview_socket.bind(f"tcp://{self.bind_host}:{self.preset.port}")
        ready.set()
        try:
            while not self.formal_done.is_set() or not self.preview_queue.empty():
                try:
                    item = self.preview_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                base64_start_ns = time.monotonic_ns()
                image_b64 = base64.b64encode(item.jpeg).decode("ascii")
                base64_completed_ns = time.monotonic_ns()
                json_start_ns = base64_completed_ns
                captured = item.captured
                extra_timing = {
                    "read_enter_monotonic_ns": captured.read_enter_monotonic_ns,
                    "read_return_monotonic_ns": captured.read_return_monotonic_ns,
                    "capture_interval_ns": captured.capture_interval_ns,
                    "capture_thread_interval_ns": captured.capture_thread_interval_ns,
                    "publisher_dequeue_monotonic_ns": item.publisher_dequeue_monotonic_ns,
                    "encode_started_monotonic_ns": item.encode_started_monotonic_ns,
                    "formal_encode_completed_monotonic_ns": item.encode_completed_monotonic_ns,
                    "preview_encode_completed_monotonic_ns": item.encode_completed_monotonic_ns,
                    "formal_spool_completed_monotonic_ns": (
                        item.formal_spool_completed_monotonic_ns
                    ),
                    "base64_started_monotonic_ns": base64_start_ns,
                    "base64_completed_monotonic_ns": base64_completed_ns,
                    "json_started_monotonic_ns": json_start_ns,
                    "reconnect_generation": captured.reconnect_generation,
                    "reconnect_drop_count_since_previous": (
                        captured.reconnect_drop_count_since_previous
                    ),
                    "drop_count_since_previous": captured.drop_count_since_previous,
                    "realsense_frame_number": captured.realsense_frame_number,
                    "realsense_frame_gap": captured.realsense_frame_gap,
                    "realsense_device_timestamp_ms": captured.realsense_device_timestamp_ms,
                    "realsense_device_interval_ms": captured.realsense_device_interval_ms,
                    "realsense_timestamp_domain": captured.realsense_timestamp_domain,
                    "preview_queue_replaced_total": self.preview_drops,
                }
                if captured.color_configuration is not None:
                    extra_timing.update(captured.color_configuration.timing_fields())
                message = build_message(
                    self.preset,
                    sequence=captured.sequence,
                    capture_wall_ns=captured.capture_wall_ns,
                    capture_monotonic_ns=captured.capture_monotonic_ns,
                    encode_completed_monotonic_ns=item.encode_completed_monotonic_ns,
                    jpeg=item.jpeg,
                    jpeg_quality=item.jpeg_quality,
                    image_b64=image_b64,
                    extra_timing=extra_timing,
                )
                send_start_ns = time.monotonic_ns()
                try:
                    preview_socket.send_string(message, flags=zmq.NOBLOCK)
                except zmq.Again:
                    self.preview_drops += 1
                    print(
                        f"[direct realsense zmq] preview_drop camera={self.preset.camera_id} "
                        f"sequence={captured.sequence} formal_spool=preserved",
                        flush=True,
                    )
                else:
                    self.frames_previewed += 1
                send_completed_ns = time.monotonic_ns()
                if self.trace_every_frame and send_completed_ns - send_start_ns > 0:
                    print(
                        "[direct realsense zmq] preview_stage="
                        + json.dumps(
                            {
                                "camera": self.preset.camera_id,
                                "sequence": captured.sequence,
                                "base64_ms": (base64_completed_ns - base64_start_ns) / 1_000_000,
                                "json_ms": (send_start_ns - json_start_ns) / 1_000_000,
                                "send_ms": (send_completed_ns - send_start_ns) / 1_000_000,
                                "preview_drops": self.preview_drops,
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
        except BaseException as exc:
            self.exception = exc
            self.stop_event.set()
        finally:
            preview_socket.close()

    def _log_stage_trace(
        self,
        captured: CapturedFrame,
        *,
        dequeue_ns: int,
        encode_start_ns: int,
        encode_completed_ns: int,
        base64_start_ns: int,
        base64_completed_ns: int,
        json_start_ns: int,
        json_completed_ns: int,
        send_start_ns: int,
        send_completed_ns: int,
        jpeg_bytes: int,
        started_ns: int,
    ) -> None:
        def ms(end_ns: int, start_ns: int) -> float:
            return (end_ns - start_ns) / 1_000_000

        capture_interval_ms = (
            None if captured.capture_interval_ns is None else captured.capture_interval_ns / 1_000_000
        )
        read_ms = ms(captured.read_return_monotonic_ns, captured.read_enter_monotonic_ns)
        encode_ms = ms(encode_completed_ns, encode_start_ns)
        base64_ms = ms(base64_completed_ns, base64_start_ns)
        json_ms = ms(json_completed_ns, json_start_ns)
        send_ms = ms(send_completed_ns, send_start_ns)
        queue_ms = ms(dequeue_ns, captured.capture_monotonic_ns)
        with self.metrics_lock:
            if capture_interval_ms is not None:
                self.capture_intervals_ms.append(capture_interval_ms)
            self.read_durations_ms.append(read_ms)
            self.encode_durations_ms.append(encode_ms)
        slow = (
            max(capture_interval_ms or 0.0, read_ms, encode_ms, base64_ms, json_ms, send_ms, queue_ms) > 50
            or captured.drop_count_since_previous > 0
        )
        if self.trace_every_frame or slow:
            trace = {
                "event": "camera_stage_trace",
                "slow": slow,
                "camera": self.preset.camera_id,
                "sequence": captured.sequence,
                "reconnect_generation": captured.reconnect_generation,
                "reconnect_drop_count_since_previous": (
                    captured.reconnect_drop_count_since_previous
                ),
                "drop_count_since_previous": captured.drop_count_since_previous,
                "realsense_frame_number": captured.realsense_frame_number,
                "realsense_frame_gap": captured.realsense_frame_gap,
                "realsense_device_timestamp_ms": captured.realsense_device_timestamp_ms,
                "realsense_device_interval_ms": captured.realsense_device_interval_ms,
                "capture_interval_ms": capture_interval_ms,
                "capture_thread_interval_ms": None
                if captured.capture_thread_interval_ns is None
                else captured.capture_thread_interval_ns / 1_000_000,
                "read_ms": read_ms,
                "queue_ms": queue_ms,
                "encode_ms": encode_ms,
                "base64_ms": base64_ms,
                "json_ms": json_ms,
                "send_ms": send_ms,
                "jpeg_bytes": jpeg_bytes,
            }
            print(f"[direct realsense zmq] stage={json.dumps(trace, separators=(',', ':'))}", flush=True)
        if captured.sequence == 1 or captured.sequence % (self.preset.fps * 10) == 0:
            elapsed_s = (time.monotonic_ns() - started_ns) / 1_000_000_000
            print(
                f"[direct realsense zmq] camera={self.preset.camera_id} sequence={captured.sequence} "
                f"capture_hz={captured.sequence / elapsed_s:.3f} jpeg_bytes={jpeg_bytes} "
                f"capture_interval_ms_p95={self._percentile(self.capture_intervals_ms, 0.95):.3f} "
                f"capture_interval_ms_max={max(self.capture_intervals_ms, default=0.0):.3f} "
                f"read_ms_p95={self._percentile(self.read_durations_ms, 0.95):.3f} "
                f"read_ms_max={max(self.read_durations_ms, default=0.0):.3f} "
                f"encode_ms_p95={self._percentile(self.encode_durations_ms, 0.95):.3f} "
                f"encode_ms_max={max(self.encode_durations_ms, default=0.0):.3f} "
                f"reconnect_generation={self.reconnect_generation} "
                f"reconnect_attempts={self.reconnect_attempts} "
                f"reconnect_successes={self.reconnect_successes} "
                f"reconnect_failures={self.reconnect_failures} "
                f"formal_drop_evidence={self.formal_drop_evidence_count} "
                f"preview_drops={self.preview_drops}",
                flush=True,
            )


def _run_camera_process(
    preset: DirectCameraPreset,
    bind_host: str,
    jpeg_quality: int,
    preview_enabled: bool,
    preview_jpeg_quality: int,
    max_reconnect_attempts: int,
    reconnect_backoff_s: float,
    count: int,
    trace_every_frame: bool,
    stop_event: Any,
    ready_queue: Any,
    metrics_queue: Any,
    error_queue: Any,
    session_marker: str | None,
    raw_root: str | None,
    spool_chunk_size_bytes: int,
    spool_fsync_each_frame: bool,
    quiesce_ack_root: str | None,
    diagnostic_dir: str | None,
) -> None:
    context = zmq.Context()
    worker = CameraPublisherWorker(
        preset,
        context,
        bind_host=bind_host,
        jpeg_quality=jpeg_quality,
        preview_enabled=preview_enabled,
        preview_jpeg_quality=preview_jpeg_quality,
        max_reconnect_attempts=max_reconnect_attempts,
        reconnect_backoff_s=reconnect_backoff_s,
        count=count,
        stop_event=stop_event,
        trace_every_frame=trace_every_frame,
        session_marker=session_marker,
        raw_root=raw_root,
        spool_chunk_size_bytes=spool_chunk_size_bytes,
        spool_fsync_each_frame=spool_fsync_each_frame,
        quiesce_ack_root=quiesce_ack_root,
        diagnostic_dir=diagnostic_dir,
    )
    ready_reported = False
    last_metrics: dict[str, int] | None = None
    last_metrics_report_monotonic = 0.0
    try:
        worker.start()
        while worker.thread.is_alive():
            if worker.ready.is_set() and not ready_reported:
                last_metrics = worker.metrics_snapshot()
                ready_queue.put({"camera_id": preset.camera_id, "metrics": last_metrics})
                ready_reported = True
            if worker.exception is not None:
                raise worker.exception
            worker.join(timeout=0.05)
            if ready_reported:
                now = time.monotonic()
                metrics = worker.metrics_snapshot()
                evidence_keys = (
                    "reconnect_generation",
                    "reconnect_attempts",
                    "reconnect_successes",
                    "reconnect_failures",
                    "reconnect_drop_count",
                    "device_frame_drop_count",
                    "formal_drop_evidence_count",
                    "preview_drop_count",
                )
                evidence_changed = last_metrics is None or any(
                    metrics[key] != last_metrics[key] for key in evidence_keys
                )
                periodic = now - last_metrics_report_monotonic >= 1.0
                if periodic or (
                    evidence_changed and now - last_metrics_report_monotonic >= 0.25
                ):
                    metrics_queue.put((preset.camera_id, metrics))
                    last_metrics = metrics
                    last_metrics_report_monotonic = now
        if worker.exception is not None:
            raise worker.exception
        if not ready_reported:
            raise RuntimeError(f"{preset.camera_id} exited before readiness")
    except BaseException as exc:
        error_queue.put((preset.camera_id, f"{type(exc).__name__}: {exc}"))
        stop_event.set()
        raise
    finally:
        if ready_reported:
            metrics_queue.put((preset.camera_id, worker.metrics_snapshot()))
        if worker.exception is not None:
            stop_event.set()
        worker.join(timeout=5)
        context.term()


class DirectRealSenseZmqServer:
    def __init__(
        self,
        *,
        bind_host: str = "*",
        jpeg_quality: int = 95,
        preview_enabled: bool = True,
        preview_jpeg_quality: int | None = None,
        max_reconnect_attempts: int = DEFAULT_MAX_RECONNECT_ATTEMPTS,
        reconnect_backoff_s: float = DEFAULT_RECONNECT_BACKOFF_S,
        count: int = 0,
        trace_every_frame: bool = False,
        presets: tuple[DirectCameraPreset, ...] | None = None,
        session_marker: str | None = None,
        raw_root: str | None = None,
        spool_chunk_size_bytes: int = 256 * 1024 * 1024,
        spool_fsync_each_frame: bool = False,
        ready_file: str | None = None,
        quiesce_ack_root: str | None = None,
        diagnostic_dir: str | None = None,
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        selected_preview_quality = (
            jpeg_quality if preview_jpeg_quality is None else preview_jpeg_quality
        )
        if not 1 <= selected_preview_quality <= 100:
            raise ValueError("preview_jpeg_quality must be between 1 and 100")
        if (
            isinstance(max_reconnect_attempts, bool)
            or not isinstance(max_reconnect_attempts, int)
            or max_reconnect_attempts < 0
        ):
            raise ValueError("max_reconnect_attempts must be a non-negative integer")
        if (
            isinstance(reconnect_backoff_s, bool)
            or not isinstance(reconnect_backoff_s, (int, float))
            or not math.isfinite(float(reconnect_backoff_s))
            or reconnect_backoff_s < 0
        ):
            raise ValueError("reconnect_backoff_s must be a finite non-negative number")
        self.bind_host = bind_host
        self.jpeg_quality = jpeg_quality
        self.preview_enabled = preview_enabled
        self.preview_jpeg_quality = selected_preview_quality
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_backoff_s = float(reconnect_backoff_s)
        self.count = count
        self.trace_every_frame = trace_every_frame
        if (session_marker is None) != (raw_root is None):
            raise ValueError("session_marker and raw_root must be configured together")
        if spool_chunk_size_bytes <= 0:
            raise ValueError("spool_chunk_size_bytes must be positive")
        self.session_marker = session_marker
        self.raw_root = raw_root
        self.spool_chunk_size_bytes = spool_chunk_size_bytes
        self.spool_fsync_each_frame = spool_fsync_each_frame
        self.ready_file = ready_file
        self.quiesce_ack_root = quiesce_ack_root
        self.diagnostic_dir = diagnostic_dir
        self.presets = tuple(CAMERA_PRESETS.values()) if presets is None else tuple(presets)
        _validate_presets(self.presets, default_jpeg_quality=jpeg_quality)
        self.mp_context = multiprocessing.get_context("spawn")
        self.stop_event = self.mp_context.Event()
        self.ready_queue = self.mp_context.Queue()
        self.metrics_queue = self.mp_context.Queue()
        self.error_queue = self.mp_context.Queue()
        self.camera_metrics = {
            preset.camera_id: empty_camera_metrics() for preset in self.presets
        }
        self.processes = tuple(
            self.mp_context.Process(
                target=_run_camera_process,
                args=(
                    preset,
                    bind_host,
                    jpeg_quality,
                    preview_enabled,
                    selected_preview_quality,
                    max_reconnect_attempts,
                    self.reconnect_backoff_s,
                    count,
                    trace_every_frame,
                    self.stop_event,
                    self.ready_queue,
                    self.metrics_queue,
                    self.error_queue,
                    session_marker,
                    raw_root,
                    spool_chunk_size_bytes,
                    spool_fsync_each_frame,
                    quiesce_ack_root,
                    diagnostic_dir,
                ),
                name=f"jz-realsense-process-{preset.camera_id}",
                daemon=False,
            )
            for preset in self.presets
        )

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> int:
        for process in self.processes:
            process.start()
        try:
            deadline = time.monotonic() + 20
            ready_cameras: set[str] = set()
            while len(ready_cameras) < len(self.processes):
                self._raise_if_failed()
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out opening all RealSense cameras")
                try:
                    report = self.ready_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if not isinstance(report, dict) or set(report) != {"camera_id", "metrics"}:
                    raise RuntimeError(f"invalid camera readiness report: {report!r}")
                camera_id = str(report["camera_id"])
                if camera_id not in self.camera_metrics:
                    raise RuntimeError(f"unknown camera readiness report: {camera_id}")
                self.camera_metrics[camera_id] = dict(report["metrics"])
                ready_cameras.add(camera_id)
            print("[direct realsense zmq] all cameras ready", flush=True)
            publish_ready(
                self.ready_file,
                role="camera_server",
                details=self.ready_details(),
            )
            while not self.stop_event.is_set():
                self._raise_if_failed()
                if self._drain_metrics():
                    publish_ready(
                        self.ready_file,
                        role="camera_server",
                        details=self.ready_details(),
                    )
                if self.count > 0 and all(not process.is_alive() for process in self.processes):
                    break
                time.sleep(0.05)
            self._raise_if_failed()
            return 0
        finally:
            remove_ready(self.ready_file)
            self.stop_event.set()
            for process in self.processes:
                process.join(timeout=5)
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
            self.ready_queue.close()
            self.metrics_queue.close()
            self.error_queue.close()

    def ready_details(self) -> dict[str, Any]:
        return {
            "camera_presets": [
                camera_preset_ready_details(
                    preset,
                    default_jpeg_quality=self.jpeg_quality,
                )
                for preset in self.presets
            ],
            "bind_host": self.bind_host,
            "preview_enabled": self.preview_enabled,
            "preview_jpeg_quality": self.preview_jpeg_quality,
            "formal_local_spool": self.session_marker is not None,
            "diagnostic_dir": self.diagnostic_dir,
            "capture_reconnect": {
                "max_attempts": self.max_reconnect_attempts,
                "backoff_s": self.reconnect_backoff_s,
                "max_backoff_s": MAX_RECONNECT_BACKOFF_S,
            },
            "camera_metrics": {
                camera_id: dict(self.camera_metrics[camera_id])
                for camera_id in sorted(self.camera_metrics)
            },
        }

    def _drain_metrics(self) -> bool:
        changed = False
        while True:
            try:
                camera_id, metrics = self.metrics_queue.get_nowait()
            except queue.Empty:
                return changed
            if camera_id not in self.camera_metrics or not isinstance(metrics, dict):
                raise RuntimeError(f"invalid camera metrics report: {camera_id!r} {metrics!r}")
            normalized = {str(key): int(value) for key, value in metrics.items()}
            if normalized != self.camera_metrics[camera_id]:
                self.camera_metrics[camera_id] = normalized
                changed = True

    def _raise_if_failed(self) -> None:
        try:
            camera_id, error = self.error_queue.get_nowait()
        except queue.Empty:
            camera_id = None
            error = None
        if camera_id is not None:
            raise RuntimeError(f"camera process failed: {camera_id}: {error}")
        for process in self.processes:
            if process.exitcode not in (None, 0):
                raise RuntimeError(
                    f"camera process failed: name={process.name} exitcode={process.exitcode}"
                )


def camera_preset_ready_details(
    preset: DirectCameraPreset,
    *,
    default_jpeg_quality: int,
) -> dict[str, Any]:
    return {
        "camera_id": preset.camera_id,
        "serial_number": preset.serial_number,
        "width": preset.width,
        "height": preset.height,
        "fps": preset.fps,
        "preview_port": preset.port,
        "pixel_format": "rgb8",
        "jpeg_quality": (
            default_jpeg_quality if preset.jpeg_quality is None else preset.jpeg_quality
        ),
        "white_balance_requested": preset.white_balance,
    }


def parse_camera_spec(value: str) -> DirectCameraPreset:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"camera spec is not valid JSON: {exc}") from exc
    required = {
        "camera_id",
        "serial_number",
        "width",
        "height",
        "fps",
        "preview_port",
        "jpeg_quality",
    }
    optional = {"white_balance"}
    if not isinstance(decoded, dict) or not required <= set(decoded) or set(decoded) - required - optional:
        actual = set(decoded) if isinstance(decoded, dict) else set()
        raise argparse.ArgumentTypeError(
            "camera spec keys mismatch: "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required - optional)}"
        )
    for name in ("camera_id", "serial_number"):
        if not isinstance(decoded[name], str) or not decoded[name]:
            raise argparse.ArgumentTypeError(f"camera spec {name} must be a non-empty string")
    for name in ("width", "height", "fps", "preview_port", "jpeg_quality"):
        if isinstance(decoded[name], bool) or not isinstance(decoded[name], int):
            raise argparse.ArgumentTypeError(f"camera spec {name} must be an integer")
    white_balance = decoded.get("white_balance")
    if white_balance is not None and (
        isinstance(white_balance, bool)
        or not isinstance(white_balance, (int, float))
        or not math.isfinite(float(white_balance))
    ):
        raise argparse.ArgumentTypeError("camera spec white_balance must be a finite number or null")
    preset = DirectCameraPreset(
        camera_id=decoded["camera_id"],
        serial_number=decoded["serial_number"],
        width=decoded["width"],
        height=decoded["height"],
        fps=decoded["fps"],
        port=decoded["preview_port"],
        jpeg_quality=decoded["jpeg_quality"],
        white_balance=None if white_balance is None else float(white_balance),
    )
    try:
        _validate_presets((preset,), default_jpeg_quality=95)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return preset


def _validate_presets(
    presets: tuple[DirectCameraPreset, ...],
    *,
    default_jpeg_quality: int,
) -> None:
    camera_ids: list[str] = []
    serials: list[str] = []
    ports: list[int] = []
    for preset in presets:
        if (
            not isinstance(preset.camera_id, str)
            or CAMERA_ID_PATTERN.fullmatch(preset.camera_id) is None
            or not isinstance(preset.serial_number, str)
            or not preset.serial_number
        ):
            raise ValueError("camera preset id must be safe and serial number must be non-empty")
        for name, value in (
            ("width", preset.width),
            ("height", preset.height),
            ("fps", preset.fps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"camera preset {preset.camera_id} {name} must be a positive integer")
        if (
            isinstance(preset.port, bool)
            or not isinstance(preset.port, int)
            or not 1 <= preset.port <= 65535
        ):
            raise ValueError(f"camera preset {preset.camera_id} preview_port must be in 1..65535")
        quality = default_jpeg_quality if preset.jpeg_quality is None else preset.jpeg_quality
        if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError(f"camera preset {preset.camera_id} jpeg_quality must be in 1..100")
        if preset.white_balance is not None and (
            isinstance(preset.white_balance, bool)
            or not isinstance(preset.white_balance, (int, float))
            or not math.isfinite(float(preset.white_balance))
        ):
            raise ValueError(
                f"camera preset {preset.camera_id} white_balance must be a finite number or null"
            )
        camera_ids.append(preset.camera_id)
        serials.append(preset.serial_number)
        ports.append(preset.port)
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("camera preset ids must be unique")
    if len(serials) != len(set(serials)):
        raise ValueError("camera preset serial numbers must be unique")
    if len(ports) != len(set(ports)):
        raise ValueError("camera preset preview ports must be unique")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Orin RealSense RGB frames over ZMQ")
    parser.add_argument("--bind-host", default="*")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--preview-jpeg-quality", type=int)
    parser.add_argument("--camera-head-white-balance", type=float)
    parser.add_argument("--camera-left-white-balance", type=float)
    parser.add_argument("--camera-right-white-balance", type=float)
    parser.add_argument("--disable-preview", action="store_true")
    parser.add_argument(
        "--max-reconnect-attempts",
        type=int,
        default=DEFAULT_MAX_RECONNECT_ATTEMPTS,
    )
    parser.add_argument(
        "--reconnect-backoff-s",
        type=float,
        default=DEFAULT_RECONNECT_BACKOFF_S,
    )
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--trace-every-frame", action="store_true")
    parser.add_argument("--session-marker")
    parser.add_argument("--raw-root")
    parser.add_argument("--spool-chunk-size-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--spool-fsync-each-frame", action="store_true")
    parser.add_argument("--ready-file")
    parser.add_argument("--quiesce-ack-root")
    parser.add_argument("--diagnostic-dir")
    parser.add_argument(
        "--camera-spec",
        action="append",
        type=parse_camera_spec,
        default=[],
        help="repeatable JSON camera preset; omitted for legacy built-in presets",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    presets = tuple(args.camera_spec) if args.camera_spec else tuple(CAMERA_PRESETS.values())
    white_balance_overrides = {
        "camera_head": args.camera_head_white_balance,
        "camera_left": args.camera_left_white_balance,
        "camera_right": args.camera_right_white_balance,
    }
    unknown_overrides = sorted(
        camera_id
        for camera_id, value in white_balance_overrides.items()
        if value is not None and camera_id not in {preset.camera_id for preset in presets}
    )
    if unknown_overrides:
        raise ValueError(f"white balance override has no matching camera preset: {unknown_overrides}")
    presets = tuple(
        replace(
            preset,
            white_balance=(
                preset.white_balance
                if white_balance_overrides.get(preset.camera_id) is None
                else white_balance_overrides[preset.camera_id]
            ),
        )
        for preset in presets
    )
    server = DirectRealSenseZmqServer(
        bind_host=args.bind_host,
        jpeg_quality=args.jpeg_quality,
        preview_enabled=not args.disable_preview,
        preview_jpeg_quality=args.preview_jpeg_quality,
        max_reconnect_attempts=args.max_reconnect_attempts,
        reconnect_backoff_s=args.reconnect_backoff_s,
        count=args.count,
        trace_every_frame=args.trace_every_frame,
        session_marker=args.session_marker,
        raw_root=args.raw_root,
        spool_chunk_size_bytes=args.spool_chunk_size_bytes,
        spool_fsync_each_frame=args.spool_fsync_each_frame,
        ready_file=args.ready_file,
        quiesce_ack_root=args.quiesce_ack_root,
        diagnostic_dir=args.diagnostic_dir,
        presets=presets,
    )
    signal.signal(signal.SIGINT, lambda _signum, _frame: server.request_stop())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: server.request_stop())
    return server.run()


if __name__ == "__main__":
    raise SystemExit(main())
