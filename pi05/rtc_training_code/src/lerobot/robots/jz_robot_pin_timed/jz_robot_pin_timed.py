#!/usr/bin/env python

from __future__ import annotations

import copy
import json
import logging
import uuid
from pathlib import Path
from typing import Any, TextIO

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.jz_robot_pin.jz_robot_pin import JZRobotPin
from lerobot.robots.jz_robot_udp.config_jz_robot_udp import RTSPCameraConfig
from lerobot.robots.jz_robot_udp.protocol import ProtocolError, validate_source_timing
from lerobot.robots.jz_robot_udp.state_cache import CachedState

from .config_jz_robot_pin_timed import JZRobotPinTimedConfig
from .timestamped_rtsp_camera import TimestampedRTSPCamera
from .timestamped_zmq_camera import TimestampedZMQCamera
from .training_schema import (
    TRAINING_SCHEMA_FILENAME,
    JZPinTrainingSchema,
    build_training_schema_manifest,
    write_training_schema_manifest,
)

logger = logging.getLogger(__name__)


class JZRobotPinTimed(JZRobotPin):
    """Pin robot with timestamped camera reception and per-frame timing sidecars."""

    config_class = JZRobotPinTimedConfig
    name = "jz_robot_pin_timed"

    def __init__(self, config: JZRobotPinTimedConfig):
        super().__init__(config)
        self.config = config
        if config.zmq_cameras:
            self.cameras = {
                key: TimestampedZMQCamera(
                    camera_config,
                    buffer_size=config.camera_buffer_size,
                    stale_frame_timeout_ms=config.camera_stale_frame_timeout_ms,
                )
                for key, camera_config in config.zmq_cameras.items()
            }
        self._observation_sequence = 0
        self._timing_session_id = uuid.uuid4().hex
        self._last_camera_sequences: dict[str, int] = {}
        self._last_observation_timing: dict[str, Any] | None = None
        self._last_command_timing: dict[str, Any] | None = None
        self._timing_files: dict[tuple[Path, int], TextIO] = {}
        self._last_observation_state_revision: int | None = None
        self._last_observation_state_seq: int | None = None
        self._last_gripper_source_generations: dict[str, int] = {}

    def _make_camera(self, key: str, config: RTSPCameraConfig) -> TimestampedRTSPCamera:
        return TimestampedRTSPCamera(
            config,
            buffer_size=self.config.camera_buffer_size,
            reconnect_delay_ms=self.config.camera_reconnect_delay_ms,
        )

    def _read_camera(
        self,
        key: str,
        camera: TimestampedRTSPCamera | TimestampedZMQCamera,
        state: CachedState,
    ) -> Any:
        state_receive_monotonic_ns = int(state.received_monotonic_s * 1_000_000_000)
        if isinstance(camera, TimestampedZMQCamera) and self.config.enforce_camera_state_receive_skew:
            max_skew_ms = float(self.config.max_camera_state_receive_skew_ms)
            return camera.read_timed_nearest(
                state_receive_monotonic_ns,
                max_receive_skew_ms=max_skew_ms,
                wait_timeout_ms=max_skew_ms,
            ).image
        return camera.read_timed_nearest(state_receive_monotonic_ns).image

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        configs = self.config.zmq_cameras or self.config.rtsp_cameras
        return {key: (cfg.height, cfg.width, 3) for key, cfg in configs.items()}

    def _get_proprioceptive_observation(self) -> tuple[CachedState, RobotObservation]:
        previous_revision = self._last_observation_state_revision
        if self.config.require_state_advance_per_observation and previous_revision is not None:
            wait_timeout_s = min(
                float(self.config.state_advance_timeout_s),
                float(self.config.state_timeout_s),
                float(self.config.max_camera_state_receive_skew_ms) / 1000,
            )
            state = self._state_cache.wait_after_revision(
                timeout_s=wait_timeout_s,
                after_revision=previous_revision,
            )
            if state is None:
                latest = self._state_cache.latest()
                latest_seq = None if latest is None else latest.packet.get("seq")
                latest_age_s = self._state_cache.age_s(latest)
                raise TimeoutError(
                    "Timed robot state did not advance before observation: "
                    f"last_seq={self._last_observation_state_seq}, latest_seq={latest_seq}, "
                    f"latest_age_s={latest_age_s}, timeout_s={wait_timeout_s}"
                )

        state, observation = super()._get_proprioceptive_observation()
        self._last_observation_state_revision = state.revision
        self._last_observation_state_seq = int(state.packet["seq"])
        return state, observation

    @property
    def last_observation_timing(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._last_observation_timing)

    @property
    def last_command_timing(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._last_command_timing)

    def inhibit_command_sending(self) -> None:
        super().inhibit_command_sending()
        self._last_command_timing = None

    def _after_action_sent(
        self,
        *,
        packet: dict[str, Any],
        action: RobotAction,
        send_completed_wall_ns: int,
        send_completed_monotonic_ns: int,
    ) -> None:
        self._last_command_timing = {
            "observation_sequence": self._observation_sequence,
            "packet_seq": packet["seq"],
            "packet_stamp_ns": packet["stamp_ns"],
            "mode": packet["mode"],
            "transport": self.config.send_action_transport,
            "send_completed_wall_ns": send_completed_wall_ns,
            "send_completed_monotonic_ns": send_completed_monotonic_ns,
            "action_key_count": len(action),
        }

    def _state_source_timing(self, state: CachedState) -> Any:
        source_timing = state.packet.get("source_timing")
        configured_sources = {
            "left": self.config.left_gripper_observation_source,
            "right": self.config.right_gripper_observation_source,
        }
        sides_requiring_freshness = [
            side for side, source in configured_sources.items() if source != "unavailable"
        ]
        if self.config.require_state_source_timing or sides_requiring_freshness:
            try:
                validate_source_timing(source_timing)
            except ProtocolError as exc:
                raise RuntimeError(
                    "Timed state packet must contain valid source_timing v1 before control; "
                    "configured gripper feedback is never replaced with a cached value"
                ) from exc
        next_generations: dict[str, int] = {}
        for side in sides_requiring_freshness:
            source_name = f"{side}_gripper"
            generation = int(source_timing["sources"][source_name]["generation"])
            previous_generation = self._last_gripper_source_generations.get(side)
            if previous_generation is not None and generation <= previous_generation:
                raise TimeoutError(
                    f"Timed {side} gripper source did not advance: generation={generation}, "
                    f"previous_generation={previous_generation}; cached opening is not accepted as new feedback"
                )
            next_generations[side] = generation
        self._last_gripper_source_generations.update(next_generations)
        return source_timing

    def _after_control_observation(self, state: CachedState, observation: RobotObservation) -> None:
        del observation
        self._state_source_timing(state)

    def _after_observation(self, state: CachedState, observation: RobotObservation) -> None:
        del observation
        source_timing = self._state_source_timing(state)
        observation_sequence = self._observation_sequence + 1
        state_receive_monotonic_ns = int(state.received_monotonic_s * 1_000_000_000)
        cameras: dict[str, dict[str, Any]] = {}
        camera_sequences: dict[str, int] = {}
        violations: dict[str, float] = {}

        for key, camera in self.cameras.items():
            timing = camera.last_read_timing
            if timing is None:
                raise RuntimeError(f"Timed camera {key!r} did not expose timing for the returned frame")
            sequence_key = "sequence" if timing.get("protocol") == "jz_realsense_zmq" else "decoder_sequence"
            camera_sequence = int(timing[sequence_key])
            reused = self._last_camera_sequences.get(key) == camera_sequence
            camera_sequences[key] = camera_sequence
            receive_delta_ms = (int(timing["receive_monotonic_ns"]) - state_receive_monotonic_ns) / 1_000_000
            receive_skew_ms = abs(receive_delta_ms)
            timing.update(
                {
                    "reused_by_observation_loop": reused,
                    "state_receive_delta_ms": receive_delta_ms,
                    "state_receive_skew_ms": receive_skew_ms,
                }
            )
            cameras[key] = timing
            if receive_skew_ms > self.config.max_camera_state_receive_skew_ms:
                violations[key] = receive_skew_ms
            if reused and self.config.reject_reused_camera_frames:
                raise TimeoutError(f"Timed camera {key!r} reused sequence {camera_sequence}")

        if violations and self.config.enforce_camera_state_receive_skew:
            raise TimeoutError(
                "Timed camera/state receive skew exceeds limit "
                f"{self.config.max_camera_state_receive_skew_ms}ms: {violations}"
            )

        self._observation_sequence = observation_sequence
        self._last_camera_sequences.update(camera_sequences)
        state_timing = {
            "packet_seq": state.packet["seq"],
            "packet_stamp_ns": state.packet["stamp_ns"],
            "receive_wall_ns": state.received_wall_ns,
            "receive_monotonic_ns": state_receive_monotonic_ns,
        }
        if "source_timing" in state.packet:
            state_timing["source_timing"] = copy.deepcopy(source_timing)
        self._last_observation_timing = {
            "session_id": self._timing_session_id,
            "observation_sequence": observation_sequence,
            "state": state_timing,
            "cameras": cameras,
        }

        if self.config.timing_log_every_n > 0 and (
            self._observation_sequence == 1
            or self._observation_sequence % self.config.timing_log_every_n == 0
        ):
            logger.info(
                "%s timing observation=%s state_seq=%s cameras=%s",
                self,
                self._observation_sequence,
                state.packet["seq"],
                {
                    key: {
                        "seq": value.get("sequence", value.get("decoder_sequence")),
                        "protocol": value.get("protocol", "rtsp"),
                        "age_ms": round(float(value["age_ms"]), 3),
                        "state_skew_ms": round(float(value["state_receive_skew_ms"]), 3),
                        "reused": value["reused_by_observation_loop"],
                    }
                    for key, value in cameras.items()
                },
            )

    @property
    def training_schema(self) -> JZPinTrainingSchema:
        return JZPinTrainingSchema(
            build_training_schema_manifest(
                left_observation_source=self.config.left_gripper_observation_source,
                right_observation_source=self.config.right_gripper_observation_source,
                left_observation_raw_closed=self.config.left_gripper_observation_raw_closed,
                left_observation_raw_open=self.config.left_gripper_observation_raw_open,
                right_observation_raw_closed=self.config.right_gripper_observation_raw_closed,
                right_observation_raw_open=self.config.right_gripper_observation_raw_open,
                left_action_raw_closed=self.config.left_gripper_action_raw_closed,
                left_action_raw_open=self.config.left_gripper_action_raw_open,
                right_action_raw_closed=self.config.right_gripper_action_raw_closed,
                right_action_raw_open=self.config.right_gripper_action_raw_open,
                left_command_force=self.config.left_gripper_training_command_force,
                right_command_force=self.config.right_gripper_training_command_force,
                provenance={
                    "created_by": "jz_robot_pin_timed_x86_recorder",
                    "raw_dataset_preserved": True,
                    "source_semantics": "explicit_robot_config",
                },
            )
        )

    def save_training_schema_manifest(self, dataset_root: Path) -> None:
        write_training_schema_manifest(
            Path(dataset_root) / "meta" / TRAINING_SCHEMA_FILENAME,
            self.training_schema.to_dict(),
        )

    def save_frame_timing(
        self,
        *,
        dataset_root: Path,
        episode_index: int,
        frame_index: int,
        action_timing: dict[str, Any] | None,
    ) -> None:
        dataset_root = Path(dataset_root)
        if frame_index == 0:
            self.save_training_schema_manifest(dataset_root)
        if not self.config.timing_sidecar:
            return
        timing = self.last_observation_timing
        if timing is None:
            raise RuntimeError("Cannot save timed frame metadata before collecting an observation")
        if not isinstance(action_timing, dict):
            raise RuntimeError("Cannot save timed frame metadata without target-action timing")
        command_timing = self.last_command_timing
        if command_timing is None:
            raise RuntimeError("Cannot save timed frame metadata before sending the matching command")
        if command_timing.get("observation_sequence") != timing["observation_sequence"]:
            raise RuntimeError(
                "Cannot save timed frame metadata with a command from another observation: "
                f"observation_sequence={timing['observation_sequence']} "
                f"command_observation_sequence={command_timing.get('observation_sequence')}"
            )
        timing.update(
            {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "action": copy.deepcopy(action_timing),
                "command": command_timing,
            }
        )

        key = (dataset_root, episode_index)
        output = self._timing_files.get(key)
        path = dataset_root / "meta" / "timing" / f"episode-{episode_index:06d}.jsonl"
        if frame_index == 0:
            if output is not None:
                output.close()
            path.parent.mkdir(parents=True, exist_ok=True)
            output = path.open("w", encoding="utf-8", buffering=1)
            self._timing_files[key] = output
        elif output is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            output = path.open("a", encoding="utf-8", buffering=1)
            self._timing_files[key] = output
        output.write(json.dumps(timing, sort_keys=True) + "\n")
        if frame_index == 0 or (
            self.config.timing_log_every_n > 0 and (frame_index + 1) % self.config.timing_log_every_n == 0
        ):
            output.flush()

    def _close_timing_files(self) -> None:
        for output in self._timing_files.values():
            output.close()
        self._timing_files.clear()

    def disconnect(self) -> None:
        try:
            super().disconnect()
        finally:
            self._close_timing_files()
            self._last_camera_sequences.clear()
            self._last_observation_timing = None
            self._last_command_timing = None
            self._last_observation_state_revision = None
            self._last_observation_state_seq = None
            self._last_gripper_source_generations.clear()
            self._observation_sequence = 0
            self._timing_session_id = uuid.uuid4().hex


__all__ = ["JZRobotPinTimed"]
