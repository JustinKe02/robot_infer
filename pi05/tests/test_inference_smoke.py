from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lerobot.robots.jz_robot_pin_timed.training_schema import RAW_FEATURE_NAMES
from tk_infer.pi05 import run_robot_client
from tk_infer.pi05.runtime import client_runtime
from tk_infer.pi05.runtime.client_runtime import (
    ClientRuntimeConfig,
    ClientRuntimeState,
    run_inference_smoke,
)
from tk_infer.pi05.runtime.protocol import (
    InferenceRequest,
    InferenceResponse,
)
from tk_infer.pi05.runtime.safety import ActionSafety


class FakeRobotIO:
    robot_type = "jz_robot_pin_timed"

    def __init__(self) -> None:
        self.send_calls = 0

    def get_observation(self) -> dict[str, Any]:
        return {"fake": "observation"}

    def send_action(self, _action: Any) -> None:
        self.send_calls += 1
        raise AssertionError("inference smoke must never call send_action")


class FakeRemotePolicy:
    def __init__(self, *, steps: int = 4) -> None:
        self.steps = steps
        self.requests: list[InferenceRequest] = []

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        request.validate()
        self.requests.append(request)
        raw = np.zeros((self.steps, 16), dtype=np.float32)
        processed = np.zeros((self.steps, 18), dtype=np.float32)
        processed[:, 15] = 80.0
        processed[:, 17] = 80.0
        return InferenceResponse(
            request_id=request.request_id,
            mode=request.mode,
            raw_actions=raw,
            processed_actions=processed,
            server_latency_s=0.08,
            model_latency_s=0.07,
            raw_action_shape=raw.shape,
            processed_action_shape=processed.shape,
        )


def _runtime_config() -> ClientRuntimeConfig:
    return ClientRuntimeConfig(
        task="jz robot pin timed vr teleoperation",
        mode="single_step",
        sensor_fps=5,
        control_fps=5,
    )


def _dataset_features() -> dict[str, dict[str, Any]]:
    return {"action": {"names": RAW_FEATURE_NAMES}}


def test_inference_smoke_validates_robot_action_without_sending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        client_runtime,
        "build_live_observation_frame",
        lambda **_kwargs: {"observation.state": np.zeros(18, dtype=np.float32)},
    )
    robot_io = FakeRobotIO()
    remote_policy = FakeRemotePolicy()
    clock = iter((10.0, 10.1))

    result = run_inference_smoke(
        config=_runtime_config(),
        remote_policy=remote_policy,  # type: ignore[arg-type]
        robot_io=robot_io,  # type: ignore[arg-type]
        dataset_features=_dataset_features(),
        robot_action_processor=None,
        robot_observation_processor=None,
        safety=ActionSafety(),
        perf_counter=lambda: next(clock),
    )

    assert robot_io.send_calls == 0
    assert len(remote_policy.requests) == 1
    assert remote_policy.requests[0].mode == "single_step"
    assert remote_policy.requests[0].prev_chunk_left_over is None
    assert result.raw_action_shape == (4, 16)
    assert result.processed_action_shape == (4, 18)
    assert result.dropped_steps == 1
    assert result.selected_action_index == 1
    assert result.robot_action_keys == tuple(sorted(RAW_FEATURE_NAMES))
    assert result.processed_action_names == tuple(RAW_FEATURE_NAMES)
    assert result.raw_actions.shape == (4, 16)
    assert result.processed_actions.shape == (4, 18)
    assert result.observation_frame["observation.state"].shape == (18,)

    metadata_path, observation_path = run_robot_client.write_inference_smoke_artifacts(
        output_dir=tmp_path / "smoke",
        runtime_config=_runtime_config(),
        health={"ok": True},
        result=result,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observation = np.load(observation_path)

    assert metadata["send_action_called"] is False
    assert np.asarray(metadata["raw_actions_model16"]).shape == (4, 16)
    assert np.asarray(metadata["processed_actions_raw18"]).shape == (4, 18)
    assert observation["observation.state"].shape == (18,)


def test_inference_smoke_rejects_fully_stale_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        client_runtime,
        "build_live_observation_frame",
        lambda **_kwargs: {"observation.state": np.zeros(18, dtype=np.float32)},
    )
    clock = iter((10.0, 20.0))

    with pytest.raises(RuntimeError, match="fully stale"):
        run_inference_smoke(
            config=_runtime_config(),
            remote_policy=FakeRemotePolicy(),  # type: ignore[arg-type]
            robot_io=FakeRobotIO(),  # type: ignore[arg-type]
            dataset_features=_dataset_features(),
            robot_action_processor=None,
            robot_observation_processor=None,
            safety=ActionSafety(),
            perf_counter=lambda: next(clock),
        )


def test_runtime_state_does_not_call_actuator_after_stop() -> None:
    state = ClientRuntimeState()
    state.request_stop("test stop")
    calls = 0

    def callback() -> str:
        nonlocal calls
        calls += 1
        return "sent"

    did_call, result = state.call_if_running(callback)

    assert did_call is False
    assert result is None
    assert calls == 0


def test_runtime_stop_is_not_blocked_by_inflight_actuator_call() -> None:
    state = ClientRuntimeState()
    started = threading.Event()
    release = threading.Event()

    def callback() -> str:
        started.set()
        assert release.wait(timeout=2.0)
        return "sent"

    worker = threading.Thread(target=lambda: state.call_if_running(callback), daemon=True)
    worker.start()
    assert started.wait(timeout=1.0)

    state.request_stop("operator stop")

    assert state.running is False
    release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (("--execution=armed",), "requires execution=dry_run"),
        (("--mode=rtc",), "requires mode=single_step"),
    ],
)
def test_python_inference_smoke_refuses_armed_or_rtc(
    extra_args: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_robot_client.main(["--inference-smoke=true", *extra_args])


def test_main_inference_smoke_disconnects_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRobot:
        robot_type = "jz_robot_pin_timed"

        def __init__(self) -> None:
            self.is_connected = False
            self.disconnect_calls = 0

        def connect(self) -> None:
            self.is_connected = True

        def disconnect(self) -> None:
            self.disconnect_calls += 1
            self.is_connected = False

    class FakeRemote:
        def health(self) -> dict[str, Any]:
            return {
                "ok": True,
                "protocol_version": run_robot_client.PROTOCOL_VERSION,
                "policy_type": "pi05",
                "model_state_dim": 16,
                "model_action_dim": 16,
                "wire_action_dim": 18,
                "schema_id": "jz_pin_opening16_v1",
                "schema_version": 1,
                "camera_profile": "three_camera",
                "complete_step": True,
                "checkpoint_step": 15705,
                "configured_steps": 15705,
                "camera_keys": list(run_robot_client.EXPECTED_CAMERA_KEYS),
                "camera_shapes": {
                    key: list(value) for key, value in run_robot_client.EXPECTED_CAMERA_SHAPES.items()
                },
                "supported_modes": ["single_step", "rtc"],
            }

    robot = FakeRobot()
    import lerobot.robots

    monkeypatch.setattr(lerobot.robots, "make_robot_from_config", lambda _config: robot)
    monkeypatch.setattr(run_robot_client, "RemotePolicyClient", lambda *_args, **_kwargs: FakeRemote())
    monkeypatch.setattr(
        run_robot_client,
        "build_dataset_artifacts",
        lambda _robot, **_kwargs: ({}, None, None),
    )
    monkeypatch.setattr(
        run_robot_client,
        "run_inference_smoke",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("smoke validation failed")),
    )

    with pytest.raises(RuntimeError, match="smoke validation failed"):
        run_robot_client.main(
            [
                "--server-url=http://127.0.0.1:8088",
                "--mode=single_step",
                "--execution=dry_run",
                "--inference-smoke=true",
            ]
        )

    assert robot.disconnect_calls == 1
    assert robot.is_connected is False


def test_smoke_wrapper_forces_read_only_single_step_and_redacts_token() -> None:
    script = Path(run_robot_client.__file__).resolve().parent / "run_inference_smoke.sh"
    env = os.environ.copy()
    env.update(
        {
            "CONDA_PYTHON": sys.executable,
            "PRINT_COMMAND_ONLY": "true",
            "SERVER_URL": "http://10.0.0.2:8088",
            "SERVER_AUTH_TOKEN": "smoke-test-secret",
            "MODE": "rtc",
            "EXECUTION": "armed",
            "JZ_ROBOT_PIN_ARMED": "1",
            "I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT": "1",
            "JZ_POLICY_INFERENCE_ARMED": "1",
            "SMOKE_SENSOR_FPS": "3",
            "SMOKE_CONTROL_FPS": "4",
            "RUN_STAMP": "pytest_inference_smoke",
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=run_robot_client.REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "--mode=single_step" in output
    assert "--execution=dry_run" in output
    assert "--sensor-fps=3" in output
    assert "--control-fps=4" in output
    assert "--inference-smoke=true" in output
    assert "inference_smoke=read-only send_action=disabled" in output
    assert "smoke-test-secret" not in output


def test_smoke_wrapper_rejects_non_low_rate() -> None:
    script = Path(run_robot_client.__file__).resolve().parent / "run_inference_smoke.sh"
    env = os.environ.copy()
    env["SMOKE_CONTROL_FPS"] = "11"

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=run_robot_client.REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "SMOKE_CONTROL_FPS must be an integer in 1..10" in completed.stderr
