from __future__ import annotations

import numpy as np
import pytest

from tk_infer.pi05_optimized.tools.offline_reference_parity import action_error_metrics, build_parser


def test_action_error_metrics_reports_independent_max_and_mean() -> None:
    expected = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    actual = np.array([[0.0, 0.5], [2.25, 3.0]], dtype=np.float32)

    metrics = action_error_metrics(expected, actual)

    assert metrics == {"max_abs": 0.5, "mean_abs": 0.1875}


def test_action_error_metrics_rejects_shape_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        action_error_metrics(np.zeros((1, 16)), np.zeros((2, 16)))
    with pytest.raises(ValueError, match="non-finite"):
        action_error_metrics(np.array([np.nan]), np.array([0.0]))


def test_offline_parity_defaults_cover_single_step_and_rtc() -> None:
    args = build_parser().parse_args([])

    assert args.mode == "both"
    assert args.device == "cuda"
    assert args.atol == 0.0
    assert args.require_complete_step is False
