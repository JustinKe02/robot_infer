from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from tk_infer.pi05_optimized.tools.offline_phase2_benchmark import (
    DEFAULT_VARIANTS,
    _compare_actions,
    _parse_variants,
    _validate_args,
)

from .helpers import make_request, make_response


def test_variant_parser_is_ordered_unique_and_rejects_unknown_values() -> None:
    assert _parse_variants("reference,inference_mode,reference") == (
        "reference",
        "inference_mode",
    )
    assert set(_parse_variants(",".join(DEFAULT_VARIANTS))) == set(DEFAULT_VARIANTS)
    with pytest.raises(ValueError, match="unknown variants"):
        _parse_variants("reference,cuda_graph")
    with pytest.raises(ValueError, match="at least one"):
        _parse_variants(" , ")


def test_fp32_and_reduced_precision_correctness_gates() -> None:
    request = make_request()
    expected = make_response(request)
    exact = make_response(request)

    exact_report = _compare_actions(expected, exact, reduced_precision=False)
    assert exact_report["gate_passed"] is True
    assert exact_report["force_slots_exact_80"] is True

    within_bf16 = make_response(request)
    within_bf16.processed_actions[:, 0] += 0.004
    within_bf16.raw_actions[:, 0] += 0.004
    bf16_report = _compare_actions(expected, within_bf16, reduced_precision=True)
    assert bf16_report["gate_passed"] is True

    outside_bf16 = make_response(request)
    outside_bf16.processed_actions[:, 0] += 0.02
    outside_bf16.raw_actions[:, 0] += 0.02
    failed_report = _compare_actions(expected, outside_bf16, reduced_precision=True)
    assert failed_report["gate_passed"] is False
    assert any("joint max" in failure for failure in failed_report["gate_failures"])


def test_correctness_gate_rejects_force_and_nonfinite_outputs() -> None:
    request = make_request()
    expected = make_response(request)
    actual = make_response(request)
    actual.processed_actions[0, 15] = 79
    actual.raw_actions[0, 0] = np.nan

    report = _compare_actions(expected, actual, reduced_precision=True)

    assert report["gate_passed"] is False
    assert report["force_slots_exact_80"] is False
    assert report["finite"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("warmup", -1, "warmup must be"),
        ("repetitions", 0, "repetitions must be positive"),
        ("stage_repetitions", -1, "stage_repetitions must be"),
        ("task", " ", "task must be"),
    ],
)
def test_benchmark_argument_validation(field: str, value: object, message: str) -> None:
    values = {
        "warmup": 1,
        "repetitions": 1,
        "stage_repetitions": 1,
        "task": "task",
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_args(Namespace(**values))
