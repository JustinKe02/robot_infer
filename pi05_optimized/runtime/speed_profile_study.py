from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Literal, TypeAlias

FIXED_SPEED_PROFILES = (1.0, 1.25, 1.5)
DEFAULT_MIN_TRIALS_PER_PROFILE = 10

StudyStatus: TypeAlias = Literal["PASS", "BLOCKED"]


@dataclass(frozen=True, slots=True)
class LabeledSpeedTrial:
    profile: float
    trial_id: str
    task_success: bool
    cycle_time_s: float
    label_source: str
    checkpoint_fingerprint: str
    task_id: str

    def __post_init__(self) -> None:
        profile = _fixed_profile(self.profile)
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("trial_id must be a non-empty string")
        if not isinstance(self.task_success, bool):
            raise ValueError("task_success must be boolean")
        cycle_time_s = _finite_positive("cycle_time_s", self.cycle_time_s)
        for field in ("label_source", "checkpoint_fingerprint", "task_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
            object.__setattr__(self, field, value.strip())
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "trial_id", self.trial_id.strip())
        object.__setattr__(self, "cycle_time_s", cycle_time_s)


@dataclass(frozen=True, slots=True)
class SpeedProfileCurvePoint:
    profile: float
    trial_count: int
    success_count: int
    task_success_rate: float
    success_wilson95_low: float
    success_wilson95_high: float
    cycle_time_mean_s: float
    cycle_time_median_s: float
    cycle_time_p95_s: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpeedProfileStudyResult:
    status: StudyStatus
    curves: tuple[SpeedProfileCurvePoint, ...]
    blocking_reasons: tuple[str, ...]
    learned_adaptation_enabled: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "curves": [curve.to_dict() for curve in self.curves],
            "blocking_reasons": list(self.blocking_reasons),
            "learned_adaptation_enabled": self.learned_adaptation_enabled,
        }


def evaluate_labeled_speed_trials(
    trials: list[LabeledSpeedTrial] | tuple[LabeledSpeedTrial, ...],
    *,
    min_trials_per_profile: int = DEFAULT_MIN_TRIALS_PER_PROFILE,
) -> SpeedProfileStudyResult:
    if (
        isinstance(min_trials_per_profile, bool)
        or not isinstance(min_trials_per_profile, int)
        or min_trials_per_profile <= 0
    ):
        raise ValueError("min_trials_per_profile must be a positive integer")
    if not isinstance(trials, list | tuple) or any(
        not isinstance(trial, LabeledSpeedTrial) for trial in trials
    ):
        raise TypeError("trials must contain LabeledSpeedTrial values")
    identifiers = [trial.trial_id for trial in trials]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("trial_id values must be unique")
    checkpoint_fingerprints = {trial.checkpoint_fingerprint for trial in trials}
    task_ids = {trial.task_id for trial in trials}
    blocking_reasons = []
    if len(checkpoint_fingerprints) > 1:
        blocking_reasons.append("all speed profiles must use one checkpoint fingerprint")
    if len(task_ids) > 1:
        blocking_reasons.append("all speed profiles must use one task_id")
    curves = []
    for profile in FIXED_SPEED_PROFILES:
        profile_trials = [trial for trial in trials if trial.profile == profile]
        if len(profile_trials) < min_trials_per_profile:
            blocking_reasons.append(
                f"profile {profile:g}x requires at least {min_trials_per_profile} labeled trials"
            )
            continue
        successes = sum(trial.task_success for trial in profile_trials)
        success_rate = successes / len(profile_trials)
        low, high = _wilson_interval(successes, len(profile_trials))
        cycle_times = sorted(trial.cycle_time_s for trial in profile_trials)
        curves.append(
            SpeedProfileCurvePoint(
                profile=profile,
                trial_count=len(profile_trials),
                success_count=successes,
                task_success_rate=success_rate,
                success_wilson95_low=low,
                success_wilson95_high=high,
                cycle_time_mean_s=sum(cycle_times) / len(cycle_times),
                cycle_time_median_s=_percentile(cycle_times, 50.0),
                cycle_time_p95_s=_percentile(cycle_times, 95.0),
            )
        )
    return SpeedProfileStudyResult(
        status="PASS" if not blocking_reasons else "BLOCKED",
        curves=tuple(curves),
        blocking_reasons=tuple(blocking_reasons),
    )


def trial_from_dict(value: object) -> LabeledSpeedTrial:
    if not isinstance(value, dict):
        raise ValueError("each speed trial must be an object")
    required = {
        "profile",
        "trial_id",
        "task_success",
        "cycle_time_s",
        "label_source",
        "checkpoint_fingerprint",
        "task_id",
    }
    if set(value) != required:
        raise ValueError(
            f"speed trial fields must be exactly {sorted(required)}, got {sorted(value)}"
        )
    return LabeledSpeedTrial(**value)  # type: ignore[arg-type]


def _fixed_profile(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("profile must be a real number")
    converted = float(value)
    for profile in FIXED_SPEED_PROFILES:
        if math.isclose(converted, profile, rel_tol=0.0, abs_tol=1e-12):
            return profile
    raise ValueError(f"profile must be one of {FIXED_SPEED_PROFILES}")


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    ratio = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * ratio


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


__all__ = [
    "DEFAULT_MIN_TRIALS_PER_PROFILE",
    "FIXED_SPEED_PROFILES",
    "LabeledSpeedTrial",
    "SpeedProfileCurvePoint",
    "SpeedProfileStudyResult",
    "evaluate_labeled_speed_trials",
    "trial_from_dict",
]
