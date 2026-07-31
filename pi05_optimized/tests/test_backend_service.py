from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tk_infer.pi05_optimized.backends.torch_backend import TorchPolicyBackend
from tk_infer.pi05_optimized.runtime.paired_trajectory import PairedTrajectory
from tk_infer.pi05_optimized.runtime.policy_service import OptimizedPolicyService

from .helpers import FakeReferenceService, make_request, make_response


def _optimized_service(reference: FakeReferenceService | None = None) -> OptimizedPolicyService:
    backend = TorchPolicyBackend(reference or FakeReferenceService())  # type: ignore[arg-type]
    return OptimizedPolicyService(backend=backend)


@pytest.mark.parametrize("mode", ["single_step", "rtc"])
def test_torch_adapter_and_pass_through_preserve_reference_actions(mode: str) -> None:
    request = make_request(mode=mode)
    expected = make_response(request)
    reference = FakeReferenceService()
    service = _optimized_service(reference)

    actual = service.infer(request)

    assert reference.infer_requests == [request]
    np.testing.assert_array_equal(actual.raw_actions, expected.raw_actions)
    np.testing.assert_array_equal(actual.processed_actions, expected.processed_actions)
    assert actual.request_id == expected.request_id
    assert actual.mode == expected.mode
    assert actual.model_latency_s == expected.model_latency_s
    health = service.health()
    assert health["optimized_runtime"] is True
    assert health["optimized_runtime_phase"] == 1
    assert health["backend"] == "torch"
    assert health["trajectory_processor"] == "pass_through"
    assert health["optimized_inference_count"] == 1
    assert health["optimized_failure_count"] == 0
    assert health["optimized_metrics"]["success_count"] == 1


def test_service_rejects_backend_request_identity_mismatch() -> None:
    def change_request_id(response: object) -> None:
        response.request_id = 99  # type: ignore[attr-defined]

    service = _optimized_service(FakeReferenceService(response_mutator=change_request_id))
    with pytest.raises(ValueError, match="request_id mismatch"):
        service.infer(make_request(request_id=7))


def test_service_rejects_backend_mode_mismatch() -> None:
    def change_mode(response: object) -> None:
        response.mode = "rtc"  # type: ignore[attr-defined]

    service = _optimized_service(FakeReferenceService(response_mutator=change_mode))
    with pytest.raises(ValueError, match="mode mismatch"):
        service.infer(make_request(mode="single_step"))


def test_service_rejects_invalid_force_before_returning_action() -> None:
    def change_force(response: object) -> None:
        response.processed_actions[0, 15] = 0.0  # type: ignore[attr-defined]

    service = _optimized_service(FakeReferenceService(response_mutator=change_force))
    with pytest.raises(ValueError, match="force slots"):
        service.infer(make_request())


def test_service_rejects_processor_identity_changes() -> None:
    class IdentityChangingProcessor:
        phase = 0
        allows_action_changes = False

        @property
        def name(self) -> str:
            return "invalid_test_processor"

        def process(self, trajectory: PairedTrajectory) -> PairedTrajectory:
            return PairedTrajectory(
                model_actions=trajectory.model_actions,
                robot_actions=trajectory.robot_actions,
                request_id=trajectory.request_id + 1,
                mode=trajectory.mode,
                source_observation_seq=trajectory.source_observation_seq,
                predicted_delay_steps=trajectory.predicted_delay_steps,
            )

    backend = TorchPolicyBackend(FakeReferenceService())  # type: ignore[arg-type]
    service = OptimizedPolicyService(backend=backend, trajectory_processor=IdentityChangingProcessor())
    with pytest.raises(ValueError, match="changed immutable identity fields"):
        service.infer(make_request())


def test_service_rejects_processor_action_changes_in_phase0() -> None:
    class ActionChangingProcessor:
        phase = 0
        allows_action_changes = False

        @property
        def name(self) -> str:
            return "invalid_test_processor"

        def process(self, trajectory: PairedTrajectory) -> PairedTrajectory:
            changed_model = trajectory.model_actions.copy()
            changed_model[0, 0] += 1.0
            return PairedTrajectory(
                model_actions=changed_model,
                robot_actions=trajectory.robot_actions,
                request_id=trajectory.request_id,
                mode=trajectory.mode,
                source_observation_seq=trajectory.source_observation_seq,
                predicted_delay_steps=trajectory.predicted_delay_steps,
            )

    backend = TorchPolicyBackend(FakeReferenceService())  # type: ignore[arg-type]
    service = OptimizedPolicyService(backend=backend, trajectory_processor=ActionChangingProcessor())
    with pytest.raises(ValueError, match="changed model16 action values"):
        service.infer(make_request())


@pytest.mark.parametrize("latency", [-1.0, float("nan"), True, "0.1"])
def test_service_rejects_invalid_backend_latency(latency: object) -> None:
    def change_latency(response: object) -> None:
        response.model_latency_s = latency  # type: ignore[attr-defined]

    service = _optimized_service(FakeReferenceService(response_mutator=change_latency))
    with pytest.raises(ValueError, match="model_latency_s must be"):
        service.infer(make_request())


def test_service_rejects_backend_error_response() -> None:
    def add_error(response: object) -> None:
        response.error = "reference backend failed"  # type: ignore[attr-defined]

    service = _optimized_service(FakeReferenceService(response_mutator=add_error))
    with pytest.raises(RuntimeError, match="reference backend failed"):
        service.infer(make_request())

    health = service.health()
    assert health["optimized_inference_count"] == 0
    assert health["optimized_failure_count"] == 1
    assert health["optimized_metrics"]["failure_count"] == 1


def test_service_uses_injected_monotonic_clock_for_stage_metrics() -> None:
    clock = iter((0.0, 0.1, 0.2, 0.5, 0.5, 0.6, 0.6, 0.8))
    backend = TorchPolicyBackend(FakeReferenceService())  # type: ignore[arg-type]
    service = OptimizedPolicyService(backend=backend, clock=lambda: next(clock))

    response = service.infer(make_request())

    assert response.server_latency_s == pytest.approx(0.8)
    stages = service.metrics.snapshot().stages
    assert stages["total_s"].latest == pytest.approx(0.8)
    assert stages["lock_wait_s"].latest == pytest.approx(0.1)
    assert stages["backend_s"].latest == pytest.approx(0.3)
    assert stages["trajectory_s"].latest == pytest.approx(0.1)
    assert stages["response_s"].latest == pytest.approx(0.2)
    assert stages["backend_reported_server_s"].latest == pytest.approx(0.01)
    assert stages["backend_reported_model_s"].latest == pytest.approx(0.005)


def test_service_emits_failure_trace_without_action_payload(tmp_path: Path) -> None:
    from tk_infer.pi05_optimized.runtime.trace import JsonlTraceWriter

    path = tmp_path / "service-failure.jsonl"

    def change_force(response: object) -> None:
        response.processed_actions[0, 15] = 0.0  # type: ignore[attr-defined]

    backend = TorchPolicyBackend(FakeReferenceService(response_mutator=change_force))  # type: ignore[arg-type]
    service = OptimizedPolicyService(backend=backend, trace_recorder=JsonlTraceWriter(path))
    with pytest.raises(ValueError, match="force slots"):
        service.infer(make_request())

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["event"] == "failure"
    assert event["error_type"] == "ValueError"
    assert "observation_frame" not in event
    assert "raw_actions" not in event


def test_failure_clock_error_does_not_mask_original_action_validation_error() -> None:
    def change_force(response: object) -> None:
        response.processed_actions[0, 15] = 0.0  # type: ignore[attr-defined]

    clock = iter((0.0, 0.1, 0.2, 0.3, 0.4))
    backend = TorchPolicyBackend(FakeReferenceService(response_mutator=change_force))  # type: ignore[arg-type]
    service = OptimizedPolicyService(backend=backend, clock=lambda: next(clock))

    with pytest.raises(ValueError, match="force slots") as captured:
        service.infer(make_request())

    diagnostics = captured.value._pi05_optimized_diagnostics  # type: ignore[attr-defined]
    assert any("failure duration could not be measured" in note for note in diagnostics)
