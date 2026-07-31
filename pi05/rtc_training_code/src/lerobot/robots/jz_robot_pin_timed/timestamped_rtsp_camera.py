#!/usr/bin/env python

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from lerobot.robots.jz_robot_udp.config_jz_robot_udp import RTSPCameraConfig

logger = logging.getLogger(__name__)
TIMESTAMP_STAGE = "decoder_output_before_pixel_conversion"


@dataclass(frozen=True)
class TimestampedFrame:
    image: Any
    decoder_pts_ns: int | None
    receive_wall_ns: int
    receive_monotonic_ns: int
    decoder_sequence: int
    reconnect_generation: int

    def copy(self) -> TimestampedFrame:
        return replace(self, image=self.image.copy())

    def timing(self, now_monotonic_ns: int) -> dict[str, int | float | str | None]:
        return {
            "timestamp_stage": TIMESTAMP_STAGE,
            "decoder_pts_ns": self.decoder_pts_ns,
            "receive_wall_ns": self.receive_wall_ns,
            "receive_monotonic_ns": self.receive_monotonic_ns,
            "decoder_sequence": self.decoder_sequence,
            "reconnect_generation": self.reconnect_generation,
            "age_ms": max(0.0, (now_monotonic_ns - self.receive_monotonic_ns) / 1_000_000),
        }


class TimestampedRTSPCamera:
    """PyAV RTSP receiver with PTS and local pre-conversion decoder-output timing.

    Decoder PTS is stream-relative. Local timestamps are sampled when the decoder yields a frame,
    before pixel conversion; neither timestamp represents network arrival or sensor exposure.
    """

    def __init__(
        self,
        config: RTSPCameraConfig,
        *,
        buffer_size: int = 8,
        reconnect_delay_ms: int = 250,
        wall_time_ns: Callable[[], int] = time.time_ns,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ValueError("buffer_size must be a positive integer")
        if (
            isinstance(reconnect_delay_ms, bool)
            or not isinstance(reconnect_delay_ms, int)
            or reconnect_delay_ms < 0
        ):
            raise ValueError("reconnect_delay_ms must be a non-negative integer")
        self.config = config
        self.buffer_size = buffer_size
        self.reconnect_delay_ms = reconnect_delay_ms
        self._wall_time_ns = wall_time_ns
        self._monotonic_ns = monotonic_ns

        self._condition = threading.Condition()
        self._frames: deque[TimestampedFrame] = deque(maxlen=buffer_size)
        self._last_read_frame: TimestampedFrame | None = None
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._decoder_sequence = 0
        self._reconnect_generation = 0
        self._reconnect_count = 0
        self._read_failures = 0
        self._last_error: Exception | None = None

    @property
    def is_connected(self) -> bool:
        return (
            not self._stop_event.is_set()
            and self._thread is not None
            and self._thread.is_alive()
            and self._frame_ready.is_set()
        )

    @property
    def frame_age_s(self) -> float | None:
        with self._condition:
            if not self._frames:
                return None
            receive_monotonic_ns = self._frames[-1].receive_monotonic_ns
        return max(0.0, (self._monotonic_ns() - receive_monotonic_ns) / 1_000_000_000)

    @property
    def diagnostics(self) -> dict[str, int | float | str | None]:
        with self._condition:
            latest = self._frames[-1] if self._frames else None
            last_error = self._last_error
            return {
                "frames_read": self._decoder_sequence,
                "read_failures": self._read_failures,
                "reconnect_count": self._reconnect_count,
                "reconnect_generation": self._reconnect_generation,
                "buffered_frames": len(self._frames),
                "latest_decoder_pts_ns": None if latest is None else latest.decoder_pts_ns,
                "latest_decoder_sequence": None if latest is None else latest.decoder_sequence,
                "frame_age_s": self.frame_age_s,
                "last_error": None if last_error is None else repr(last_error),
            }

    @property
    def last_read_timing(self) -> dict[str, int | float | str | None] | None:
        with self._condition:
            frame = self._last_read_frame
            if frame is None:
                return None
            return frame.timing(self._monotonic_ns())

    def connect(self) -> None:
        if self.is_connected:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"RTSP receiver thread is already running for {self.config.url}")

        with self._condition:
            self._frames.clear()
            self._last_read_frame = None
            self._last_error = None
        self._frame_ready.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"timed_rtsp_reader_{self.config.url.rsplit('/', maxsplit=1)[-1]}",
            daemon=False,
        )
        self._thread.start()

        try:
            timeout_s = self.config.timeout_ms / 1000
            if not self._frame_ready.wait(timeout=timeout_s):
                with self._condition:
                    last_error = self._last_error
                message = f"Timed out waiting for the first timed RTSP frame: {self.config.url}"
                if last_error is not None:
                    message = f"{message}; last_error={last_error}"
                raise TimeoutError(message)

            if self.config.warmup_frames > 0:
                self._wait_for_additional_frames(self.config.warmup_frames, timeout_s)
        except BaseException:
            self.disconnect()
            raise

    def _wait_for_additional_frames(self, count: int, timeout_s: float) -> None:
        with self._condition:
            start_sequence = self._decoder_sequence
            ready = self._condition.wait_for(
                lambda: self._decoder_sequence >= start_sequence + count or self._stop_event.is_set(),
                timeout=timeout_s,
            )
        if not ready or self._stop_event.is_set():
            raise TimeoutError(f"Timed out warming up RTSP frames: {self.config.url}")

    def _reader_loop(self) -> None:
        opened_once = False
        while not self._stop_event.is_set():
            container = None
            try:
                container = self._open_container()
                if self._stop_event.is_set():
                    continue
                with self._condition:
                    self._reconnect_generation += 1
                    if opened_once:
                        self._reconnect_count += 1
                    opened_once = True
                    generation = self._reconnect_generation
                    self._last_error = None

                decoded_any = False
                for frame in container.decode(video=0):
                    if self._stop_event.is_set():
                        break
                    decoded_any = True
                    self._store_decoded_frame(frame, generation)
                if not self._stop_event.is_set():
                    reason = "ended after frames" if decoded_any else "ended before the first frame"
                    raise EOFError(f"RTSP decoder {reason}: {self.config.url}")
            except Exception as exc:
                if not self._stop_event.is_set():
                    with self._condition:
                        self._read_failures += 1
                        self._last_error = exc
                    logger.warning("Timed RTSP receiver failed for %s: %s", self.config.url, exc)
            finally:
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        logger.debug("Failed closing RTSP container for %s", self.config.url, exc_info=True)

            if not self._stop_event.is_set():
                self._stop_event.wait(self.reconnect_delay_ms / 1000)

    def _open_container(self):
        import av

        options = {"rtsp_transport": self.config.transport}
        options.update(self._parse_capture_options(self.config.ffmpeg_capture_options))
        timeout_s = self.config.timeout_ms / 1000
        return av.open(self.config.url, options=options, timeout=(timeout_s, timeout_s))

    @staticmethod
    def _parse_capture_options(raw_options: str | None) -> dict[str, str]:
        if not raw_options:
            return {}
        options: dict[str, str] = {}
        for item in raw_options.split("|"):
            if not item:
                continue
            key, separator, value = item.partition(";")
            if not separator or not key or not value:
                raise ValueError("ffmpeg_capture_options must use OpenCV's key;value|key;value format")
            options[key] = value
        return options

    def _store_decoded_frame(self, frame: Any, generation: int) -> None:
        # Timestamp the decoder output before resolution-dependent pixel conversion.
        receive_wall_ns = self._wall_time_ns()
        receive_monotonic_ns = self._monotonic_ns()
        pixel_format = "rgb24" if self.config.color_mode == "rgb" else "bgr24"
        image = frame.to_ndarray(format=pixel_format)
        expected_shape = (self.config.height, self.config.width, 3)
        if image.shape != expected_shape:
            raise RuntimeError(
                f"RTSP camera frame shape {image.shape} does not match configured "
                f"{expected_shape} for {self.config.url}"
            )

        decoder_pts_ns = None
        if frame.pts is not None and frame.time_base is not None:
            decoder_pts_ns = int(frame.pts * frame.time_base * 1_000_000_000)

        with self._condition:
            self._decoder_sequence += 1
            timed_frame = TimestampedFrame(
                image=image,
                decoder_pts_ns=decoder_pts_ns,
                receive_wall_ns=receive_wall_ns,
                receive_monotonic_ns=receive_monotonic_ns,
                decoder_sequence=self._decoder_sequence,
                reconnect_generation=generation,
            )
            self._frames.append(timed_frame)
            self._last_error = None
            self._frame_ready.set()
            self._condition.notify_all()

    def _read_timed(self, target_monotonic_ns: int | None = None) -> TimestampedFrame:
        if not self.is_connected:
            raise RuntimeError(f"RTSP camera is not connected: {self.config.url}")
        with self._condition:
            if not self._frames:
                raise TimeoutError(f"No timed RTSP frame available: {self.config.url}")
            if target_monotonic_ns is None:
                frame = self._frames[-1]
            else:
                frame = min(
                    self._frames,
                    key=lambda item: (
                        abs(item.receive_monotonic_ns - target_monotonic_ns),
                        -item.receive_monotonic_ns,
                    ),
                )

        age_s = max(0.0, (self._monotonic_ns() - frame.receive_monotonic_ns) / 1_000_000_000)
        if age_s > self.config.stale_frame_timeout_ms / 1000:
            raise TimeoutError(
                f"Selected timed RTSP frame is stale for {self.config.url}: "
                f"age_s={age_s}, timeout_s={self.config.stale_frame_timeout_ms / 1000}"
            )

        result = frame.copy()
        with self._condition:
            self._last_read_frame = result
        return result

    def read_timed(self) -> TimestampedFrame:
        return self._read_timed()

    def read_timed_nearest(self, target_monotonic_ns: int) -> TimestampedFrame:
        """Return the buffered frame nearest a local monotonic receive timestamp."""
        if isinstance(target_monotonic_ns, bool) or not isinstance(target_monotonic_ns, int):
            raise ValueError("target_monotonic_ns must be an integer")
        return self._read_timed(target_monotonic_ns)

    def read(self) -> Any:
        return self.read_timed().image

    def async_read(self) -> Any:
        return self.read()

    def disconnect(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

        thread = self._thread
        if thread is not None:
            # The reader owns the PyAV container for its full lifetime, including close().
            # PyAV's read timeout interrupts a stalled decode so shutdown never needs to
            # free FFmpeg state concurrently from this thread.
            shutdown_timeout_s = max(1.0, self.config.timeout_ms / 1000 + 1.0)
            thread.join(timeout=shutdown_timeout_s)
            if thread.is_alive():
                logger.warning(
                    "Timed RTSP receiver exceeded its read-timeout shutdown window for %s; "
                    "waiting for the reader to release its container",
                    self.config.url,
                )
                thread.join()
            self._thread = None
        self._frame_ready.clear()


__all__ = ["TimestampedFrame", "TimestampedRTSPCamera"]
