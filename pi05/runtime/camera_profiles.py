from __future__ import annotations

from dataclasses import dataclass

CAMERA_FEATURE_PREFIX = "observation.images."
DEFAULT_CAMERA_PROFILE = "three_camera"


@dataclass(frozen=True, slots=True)
class CameraSpec:
    port: int
    width: int
    height: int

    @property
    def chw_shape(self) -> tuple[int, int, int]:
        return (3, self.height, self.width)

    @property
    def hwc_shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)


CAMERA_SPECS: dict[str, CameraSpec] = {
    "camera_head": CameraSpec(port=5555, width=1280, height=720),
    "camera_left": CameraSpec(port=5556, width=640, height=480),
    "camera_right": CameraSpec(port=5557, width=640, height=480),
}
CAMERA_PROFILES: dict[str, tuple[str, ...]] = {
    "three_camera": ("camera_head", "camera_left", "camera_right"),
    "head_right": ("camera_head", "camera_right"),
}
SUPPORTED_CAMERA_PROFILES = tuple(CAMERA_PROFILES)


def camera_names(profile: str) -> tuple[str, ...]:
    try:
        return CAMERA_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported camera profile {profile!r}; expected one of {list(SUPPORTED_CAMERA_PROFILES)}"
        ) from exc


def camera_feature_keys(profile: str) -> tuple[str, ...]:
    return tuple(f"{CAMERA_FEATURE_PREFIX}{name}" for name in camera_names(profile))


def camera_feature_shapes(profile: str) -> dict[str, tuple[int, int, int]]:
    return {f"{CAMERA_FEATURE_PREFIX}{name}": CAMERA_SPECS[name].chw_shape for name in camera_names(profile)}


def camera_dataset_shapes(profile: str) -> dict[str, tuple[int, int, int]]:
    return {f"{CAMERA_FEATURE_PREFIX}{name}": CAMERA_SPECS[name].hwc_shape for name in camera_names(profile)}


def infer_camera_profile(feature_keys: tuple[str, ...]) -> str:
    actual = tuple(sorted(feature_keys))
    for profile in SUPPORTED_CAMERA_PROFILES:
        if actual == tuple(sorted(camera_feature_keys(profile))):
            return profile
    raise ValueError(
        "Checkpoint cameras do not match a supported profile; "
        f"got={list(actual)} supported="
        f"{ {profile: list(camera_feature_keys(profile)) for profile in SUPPORTED_CAMERA_PROFILES} }"
    )


__all__ = [
    "CAMERA_FEATURE_PREFIX",
    "CAMERA_PROFILES",
    "CAMERA_SPECS",
    "DEFAULT_CAMERA_PROFILE",
    "SUPPORTED_CAMERA_PROFILES",
    "CameraSpec",
    "camera_dataset_shapes",
    "camera_feature_keys",
    "camera_feature_shapes",
    "camera_names",
    "infer_camera_profile",
]
