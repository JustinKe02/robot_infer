#!/usr/bin/env python

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    DataProcessorPipeline,
    NormalizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.robots.jz_robot_pin_timed.training_schema import (
    RAW_DIM,
    RAW_FEATURE_NAMES,
    TRAINING_ACTION_NAMES,
    TRAINING_DIM,
    TRAINING_OBSERVATION_NAMES,
    TRAINING_SCHEMA_VERSION,
    JZPinProjectedMetadata,
    JZPinRaw18ToTraining16ProcessorStep,
    JZPinTraining16ToRaw18ActionProcessorStep,
    JZPinTrainingDatasetView,
    JZPinTrainingSchema,
    JZPinTrainingSchemaError,
    build_training_schema_manifest,
    load_training_schema_from_local_checkpoint,
    write_training_schema_manifest,
)
from my_devs.jz_robot_pin_timed.data_check.check_training_projection import run_check
from my_devs.jz_robot_pin_timed.train.train_act_resized import insert_schema_steps

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATASET_ROOT = REPOSITORY_ROOT / "tests/outputs/jz_robot_pin_timed_real_20260711_190502"


def make_manifest(
    *,
    left_source: str = "measured_opening",
    right_source: str = "commanded_opening",
) -> dict[str, Any]:
    return build_training_schema_manifest(
        left_observation_source=left_source,
        right_observation_source=right_source,
        left_observation_raw_closed=0.0,
        left_observation_raw_open=100.0,
        right_observation_raw_closed=100.0,
        right_observation_raw_open=0.0,
        left_action_raw_closed=100.0,
        left_action_raw_open=0.0,
        right_action_raw_closed=100.0,
        right_action_raw_open=0.0,
        left_command_force=73.0,
        right_command_force=61.0,
        provenance={"source_audit": "offline_test_fixture"},
    )


@pytest.fixture
def schema() -> JZPinTrainingSchema:
    return JZPinTrainingSchema(make_manifest())


def make_raw_features() -> dict[str, dict[str, Any]]:
    vector_feature = {
        "dtype": "float32",
        "shape": [RAW_DIM],
        "names": list(RAW_FEATURE_NAMES),
    }
    return {
        "observation.state": dict(vector_feature),
        "action": dict(vector_feature),
        "observation.images.camera_head": {
            "dtype": "video",
            "shape": [3, 224, 224],
            "names": ["channels", "height", "width"],
        },
    }


def make_raw_stats() -> dict[str, dict[str, np.ndarray]]:
    base = np.arange(RAW_DIM, dtype=np.float64)

    def vector_stats(offset: float) -> dict[str, np.ndarray]:
        shifted = base + offset
        return {
            "min": shifted.copy(),
            "max": shifted + 20.0,
            "mean": shifted + 5.0,
            "std": np.full(RAW_DIM, 2.5, dtype=np.float64),
            "count": np.asarray([32], dtype=np.int64),
            "q01": shifted + 1.0,
            "q10": shifted + 2.0,
            "q50": shifted + 5.0,
            "q90": shifted + 8.0,
            "q99": shifted + 9.0,
        }

    return {
        "observation.state": vector_stats(0.0),
        "action": vector_stats(10.0),
        "observation.images.camera_head": {
            "mean": np.asarray([0.1, 0.2, 0.3]),
            "std": np.asarray([0.4, 0.5, 0.6]),
        },
    }


def make_raw_vector(*, temporal_shape: tuple[int, ...] = ()) -> torch.Tensor:
    vector = torch.arange(RAW_DIM, dtype=torch.float32)
    vector[14] = 20.0
    vector[15] = 987_654.0
    vector[16] = 80.0
    vector[17] = -987_654.0
    if not temporal_shape:
        return vector
    return vector.expand(*temporal_shape, RAW_DIM).clone()


def identity_transition(transition):
    return transition


def write_checker_dataset(root: Path, manifest: dict[str, Any]) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "meta/timing").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {"features": make_raw_features()}
    stats = {
        feature_key: {name: value.tolist() for name, value in values.items()}
        for feature_key, values in make_raw_stats().items()
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/stats.json").write_text(json.dumps(stats), encoding="utf-8")
    raw = make_raw_vector().numpy().tolist()
    pd.DataFrame(
        {
            "episode_index": [0],
            "frame_index": [0],
            "observation.state": [raw],
            "action": [raw],
        }
    ).to_parquet(root / "data/chunk-000/file-000.parquet")
    snapshot_ns = 2_000_000_000
    sources = {}
    for offset, source_name in enumerate(("left_joints", "right_joints", "left_gripper", "right_gripper")):
        receive_ns = snapshot_ns - (offset + 1) * 1_000_000
        sources[source_name] = {
            "generation": offset + 1,
            "recv_wall_ns": 3_000_000_000 + receive_ns,
            "recv_monotonic_ns": receive_ns,
            "header_stamp_ns": 4_000_000_000 + offset if source_name.endswith("joints") else None,
            "age_ms": (snapshot_ns - receive_ns) / 1_000_000,
        }
    timing_record = {
        "session_id": "a" * 32,
        "episode_index": 0,
        "frame_index": 0,
        "state": {
            "source_timing": {
                "schema_version": 1,
                "source_skew_ms": 3.0,
                "sources": sources,
            }
        },
    }
    (root / "meta/timing/episode-000000.jsonl").write_text(
        json.dumps(timing_record) + "\n",
        encoding="utf-8",
    )
    manifest_path = root / "meta/jz_pin_training_schema.json"
    write_training_schema_manifest(manifest_path, manifest)
    return manifest_path


def test_named_projection_keeps_exact_indices_and_excludes_force_sentinels(
    schema: JZPinTrainingSchema,
) -> None:
    raw = make_raw_vector()
    before = raw.clone()

    projected = schema.project_observation(raw)

    assert schema.keep_indices == {
        "observation.state": (*range(15), 16),
        "action": (*range(15), 16),
    }
    assert schema.dropped_force_indices == {
        "observation.state": (15, 17),
        "action": (15, 17),
    }
    assert projected.shape == (TRAINING_DIM,)
    torch.testing.assert_close(projected[:14], raw[:14])
    assert projected[14:].tolist() == pytest.approx([20.0, 20.0])
    assert 987_654.0 not in projected.tolist()
    assert -987_654.0 not in projected.tolist()
    torch.testing.assert_close(raw, before)


def test_projection_rejects_metadata_with_same_fields_in_wrong_order(
    schema: JZPinTrainingSchema,
) -> None:
    features = make_raw_features()
    action_names = features["action"]["names"]
    action_names[14], action_names[15] = action_names[15], action_names[14]

    with pytest.raises(JZPinTrainingSchemaError, match="Raw feature order mismatch"):
        schema.project_features(features)


def test_temporal_action_projection_and_expansion_use_explicit_direction_and_force(
    schema: JZPinTrainingSchema,
) -> None:
    raw = make_raw_vector(temporal_shape=(2, 3))
    raw_before = raw.clone()

    model_action = schema.project_action(raw)
    expanded = schema.expand_action(model_action)

    assert model_action.shape == (2, 3, TRAINING_DIM)
    assert expanded.shape == (2, 3, RAW_DIM)
    torch.testing.assert_close(model_action[..., :14], raw[..., :14])
    assert torch.all(model_action[..., 14] == 80.0)
    assert torch.all(model_action[..., 15] == 20.0)
    torch.testing.assert_close(expanded[..., :14], raw[..., :14])
    assert torch.all(expanded[..., 14] == 20.0)
    assert torch.all(expanded[..., 15] == 73.0)
    assert torch.all(expanded[..., 16] == 80.0)
    assert torch.all(expanded[..., 17] == 61.0)
    torch.testing.assert_close(raw, raw_before)


def test_canonical_action_endpoints_are_zero_closed_and_one_hundred_open(
    schema: JZPinTrainingSchema,
) -> None:
    model_action = torch.zeros(TRAINING_DIM, dtype=torch.float32)
    model_action[14] = 0.0
    model_action[15] = 100.0

    expanded = schema.expand_action(model_action)

    assert expanded[14].item() == pytest.approx(100.0)
    assert expanded[16].item() == pytest.approx(0.0)
    assert expanded[15].item() == pytest.approx(73.0)
    assert expanded[17].item() == pytest.approx(61.0)


def test_projected_features_and_stats_are_model16_and_reverse_order_statistics(
    schema: JZPinTrainingSchema,
) -> None:
    raw_features = make_raw_features()
    raw_stats = make_raw_stats()
    raw_stats_before = {
        key: {name: value.copy() for name, value in values.items()} for key, values in raw_stats.items()
    }

    features = schema.project_features(raw_features)
    stats = schema.project_stats(raw_stats)

    assert features["observation.state"]["shape"] == [TRAINING_DIM]
    assert features["observation.state"]["names"] == list(TRAINING_OBSERVATION_NAMES)
    assert features["action"]["shape"] == [TRAINING_DIM]
    assert features["action"]["names"] == list(TRAINING_ACTION_NAMES)
    assert raw_features["observation.state"]["shape"] == [RAW_DIM]
    assert raw_features["action"]["names"] == list(RAW_FEATURE_NAMES)

    raw_observation_stats = raw_stats["observation.state"]
    observation_stats = stats["observation.state"]
    assert observation_stats["min"].shape == (TRAINING_DIM,)
    assert observation_stats["min"][14] == pytest.approx(raw_observation_stats["min"][14])
    assert observation_stats["min"][15] == pytest.approx(100.0 - raw_observation_stats["max"][16])
    assert observation_stats["max"][15] == pytest.approx(100.0 - raw_observation_stats["min"][16])
    assert observation_stats["q01"][15] == pytest.approx(100.0 - raw_observation_stats["q99"][16])
    assert observation_stats["q10"][15] == pytest.approx(100.0 - raw_observation_stats["q90"][16])
    assert observation_stats["q50"][15] == pytest.approx(100.0 - raw_observation_stats["q50"][16])
    assert observation_stats["std"][15] == pytest.approx(raw_observation_stats["std"][16])
    np.testing.assert_array_equal(observation_stats["count"], np.asarray([32]))
    np.testing.assert_array_equal(
        stats["observation.images.camera_head"]["mean"],
        raw_stats["observation.images.camera_head"]["mean"],
    )
    for feature_key, feature_stats in raw_stats_before.items():
        for statistic_name, original_value in feature_stats.items():
            np.testing.assert_array_equal(raw_stats[feature_key][statistic_name], original_value)


def test_unavailable_source_stays_auditable_but_cannot_be_trained() -> None:
    schema = JZPinTrainingSchema(make_manifest(right_source="unavailable"))
    assert schema.observation_sources == {
        "left": "measured_opening",
        "right": "unavailable",
    }

    with pytest.raises(JZPinTrainingSchemaError, match="unavailable.*right"):
        schema.ensure_trainable()
    with pytest.raises(JZPinTrainingSchemaError, match="unavailable.*right"):
        schema.project_observation(make_raw_vector())


def test_manifest_roundtrip_preserves_measured_and_commanded_semantics(tmp_path: Path) -> None:
    path = tmp_path / "meta/jz_pin_training_schema.json"
    manifest = make_manifest()

    write_training_schema_manifest(path, manifest)
    loaded = JZPinTrainingSchema.from_file(path)

    assert loaded.to_dict()["schema_version"] == TRAINING_SCHEMA_VERSION
    assert loaded.observation_sources == {
        "left": "measured_opening",
        "right": "commanded_opening",
    }
    assert loaded.to_dict()["grippers"]["left"]["action"]["source"] == "commanded_opening"
    assert loaded.to_dict()["grippers"]["right"]["action"]["source"] == "commanded_opening"
    assert loaded.to_dict()["provenance"] == {"source_audit": "offline_test_fixture"}
    assert json.loads(path.read_text(encoding="utf-8")) == manifest


def test_manifest_rejects_force_that_is_presented_as_model_or_feedback_data() -> None:
    manifest = make_manifest()
    manifest["grippers"]["left"]["wire_force"]["source"] = "measured_opening"

    with pytest.raises(JZPinTrainingSchemaError, match="explicit_x86_boundary_config"):
        JZPinTrainingSchema(manifest)


def test_manifest_cannot_swap_left_and_right_training_semantics() -> None:
    manifest = make_manifest()
    manifest["grippers"]["left"]["observation"]["training_field"] = "right_gripper.opening"
    manifest["grippers"]["right"]["observation"]["training_field"] = "left_gripper.opening"

    with pytest.raises(JZPinTrainingSchemaError, match="left observation training_field"):
        JZPinTrainingSchema(manifest)


@pytest.mark.parametrize(("index", "value"), [(14, float("nan")), (16, float("inf"))])
def test_missing_or_nonfinite_gripper_feedback_fails_explicitly(
    schema: JZPinTrainingSchema,
    index: int,
    value: float,
) -> None:
    raw = make_raw_vector()
    raw[index] = value

    with pytest.raises(JZPinTrainingSchemaError, match="missing or non-finite"):
        schema.project_observation(raw)

    with pytest.raises(JZPinTrainingSchemaError, match="last dimension 18"):
        schema.project_observation(torch.zeros(17))


def test_dataset_view_preserves_raw18_and_exposes_only_model16(schema: JZPinTrainingSchema) -> None:
    raw_observation = make_raw_vector()
    raw_action = make_raw_vector()
    raw_item = {
        "observation.state": raw_observation,
        "action": raw_action,
        "timestamp": torch.tensor(0.0),
    }
    raw_meta = SimpleNamespace(
        features=make_raw_features(),
        stats=make_raw_stats(),
        robot_type="jz_robot_pin_timed",
    )

    class RawDataset:
        def __init__(self):
            self.meta = raw_meta
            self.samples = [raw_item]
            self.repo_id = "local/raw18"

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    raw_dataset = RawDataset()
    view = JZPinTrainingDatasetView(raw_dataset, schema)

    model_item = view[0]

    assert model_item["observation.state"].shape == (TRAINING_DIM,)
    assert model_item["action"].shape == (TRAINING_DIM,)
    assert view.meta.features["observation.state"]["shape"] == [TRAINING_DIM]
    assert view.meta.features["action"]["shape"] == [TRAINING_DIM]
    assert view.meta.stats["observation.state"]["mean"].shape == (TRAINING_DIM,)
    assert view.repo_id == "local/raw18"
    assert raw_dataset.meta.features["observation.state"]["shape"] == [RAW_DIM]
    assert raw_dataset.samples[0]["observation.state"].shape == (RAW_DIM,)
    assert raw_dataset.samples[0]["action"].shape == (RAW_DIM,)
    torch.testing.assert_close(raw_dataset.samples[0]["observation.state"], raw_observation)
    torch.testing.assert_close(raw_dataset.samples[0]["action"], raw_action)


def test_dataset_view_rejects_absent_observation_feedback(schema: JZPinTrainingSchema) -> None:
    raw_meta = SimpleNamespace(features=make_raw_features(), stats=make_raw_stats())

    class MissingFeedbackDataset:
        meta = raw_meta

        def __len__(self):
            return 1

        def __getitem__(self, index):
            del index
            return {"action": make_raw_vector()}

    view = JZPinTrainingDatasetView(MissingFeedbackDataset(), schema)
    with pytest.raises(JZPinTrainingSchemaError, match="feedback is unavailable"):
        view[0]


def test_policy_metadata_view_can_load_a_new_raw18_recording_dataset_without_stats(
    schema: JZPinTrainingSchema,
) -> None:
    raw_meta = SimpleNamespace(features=make_raw_features(), stats=None)

    policy_meta = JZPinProjectedMetadata(raw_meta, schema, require_stats=False)

    assert policy_meta.features["observation.state"]["shape"] == [TRAINING_DIM]
    assert policy_meta.features["action"]["shape"] == [TRAINING_DIM]
    assert policy_meta.stats is None


def test_processor_pipelines_serialize_projection_semantics_and_boundary_shapes(
    schema: JZPinTrainingSchema,
    tmp_path: Path,
) -> None:
    preprocessor = DataProcessorPipeline(
        steps=[JZPinRaw18ToTraining16ProcessorStep(schema.to_dict())],
        name="JZ projection test preprocessor",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    postprocessor = DataProcessorPipeline(
        steps=[JZPinTraining16ToRaw18ActionProcessorStep(schema.to_dict())],
        name="JZ projection test postprocessor",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    preprocessor.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")

    loaded_preprocessor = DataProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    loaded_postprocessor = DataProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    raw_observation = make_raw_vector(temporal_shape=(2,))
    raw_action = make_raw_vector(temporal_shape=(2, 3))
    transition = {
        TransitionKey.OBSERVATION: {"observation.state": raw_observation},
        TransitionKey.ACTION: raw_action,
    }

    model_transition = loaded_preprocessor(transition)
    wire_transition = loaded_postprocessor({TransitionKey.ACTION: model_transition[TransitionKey.ACTION]})

    assert model_transition[TransitionKey.OBSERVATION]["observation.state"].shape == (
        2,
        TRAINING_DIM,
    )
    assert model_transition[TransitionKey.ACTION].shape == (2, 3, TRAINING_DIM)
    assert wire_transition[TransitionKey.ACTION].shape == (2, 3, RAW_DIM)
    assert torch.all(wire_transition[TransitionKey.ACTION][..., 15] == 73.0)
    assert torch.all(wire_transition[TransitionKey.ACTION][..., 17] == 61.0)
    assert loaded_preprocessor.steps[0].schema.observation_sources == schema.observation_sources
    assert loaded_postprocessor.steps[0].schema.to_dict() == schema.to_dict()
    assert load_training_schema_from_local_checkpoint(tmp_path).to_dict() == schema.to_dict()
    assert transition[TransitionKey.OBSERVATION]["observation.state"].shape[-1] == RAW_DIM
    assert transition[TransitionKey.ACTION].shape[-1] == RAW_DIM


def test_training_wrapper_places_projection_around_model16_normalization(
    schema: JZPinTrainingSchema,
) -> None:
    normalizer = NormalizerProcessorStep(features={}, norm_map={}, stats={})
    unnormalizer = UnnormalizerProcessorStep(features={}, norm_map={}, stats={})
    preprocessor = SimpleNamespace(steps=[normalizer])
    postprocessor = SimpleNamespace(steps=[unnormalizer])

    insert_schema_steps(preprocessor, postprocessor, schema)
    insert_schema_steps(preprocessor, postprocessor, schema)

    assert len(preprocessor.steps) == 2
    assert len(postprocessor.steps) == 2
    assert isinstance(preprocessor.steps[0], JZPinRaw18ToTraining16ProcessorStep)
    assert preprocessor.steps[1] is normalizer
    assert postprocessor.steps[0] is unnormalizer
    assert isinstance(postprocessor.steps[1], JZPinTraining16ToRaw18ActionProcessorStep)


def test_projection_checker_passes_only_with_available_explicit_sources(tmp_path: Path) -> None:
    manifest_path = write_checker_dataset(tmp_path, make_manifest())

    report = run_check(
        Namespace(
            dataset_root=tmp_path,
            manifest=manifest_path,
            allow_unavailable=False,
            report_json=None,
        )
    )

    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report["force_exclusion"]["observation.state"]["dropped_force_indices"] == [15, 17]
    assert report["sample"]["observation.state"]["raw18_preserved"] is True


def test_projection_checker_marks_unavailable_as_fail_or_explicit_audit(tmp_path: Path) -> None:
    manifest_path = write_checker_dataset(tmp_path, make_manifest(right_source="unavailable"))
    common = {"dataset_root": tmp_path, "manifest": manifest_path, "report_json": None}

    strict_report = run_check(Namespace(**common, allow_unavailable=False))
    audit_report = run_check(Namespace(**common, allow_unavailable=True))

    assert strict_report["status"] == "FAIL"
    assert "unavailable" in strict_report["errors"][0]
    assert audit_report["status"] == "AUDIT"
    assert audit_report["unavailable_sides"] == ["right"]
    assert audit_report["warnings"]


def test_projection_checker_scans_later_rows_for_missing_gripper_feedback(tmp_path: Path) -> None:
    manifest_path = write_checker_dataset(tmp_path, make_manifest())
    valid = make_raw_vector().numpy()
    missing = valid.copy()
    missing[16] = np.nan
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "observation.state": [valid.tolist(), missing.tolist()],
            "action": [valid.tolist(), valid.tolist()],
        }
    ).to_parquet(tmp_path / "data/chunk-000/file-000.parquet")

    report = run_check(
        Namespace(
            dataset_root=tmp_path,
            manifest=manifest_path,
            allow_unavailable=False,
            report_json=None,
        )
    )

    assert report["status"] == "FAIL"
    assert "missing or non-finite" in report["errors"][0]


def test_projection_checker_rejects_reused_gripper_generation(tmp_path: Path) -> None:
    manifest_path = write_checker_dataset(tmp_path, make_manifest())
    valid = make_raw_vector().numpy().tolist()
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "observation.state": [valid, valid],
            "action": [valid, valid],
        }
    ).to_parquet(tmp_path / "data/chunk-000/file-000.parquet")
    timing_path = tmp_path / "meta/timing/episode-000000.jsonl"
    first = json.loads(timing_path.read_text(encoding="utf-8"))
    second = json.loads(json.dumps(first))
    second["frame_index"] = 1
    timing_path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    report = run_check(
        Namespace(
            dataset_root=tmp_path,
            manifest=manifest_path,
            allow_unavailable=False,
            report_json=None,
        )
    )

    assert report["status"] == "FAIL"
    assert "generation did not strictly advance" in report["errors"][0]


def test_projection_checker_requires_exact_timing_to_parquet_frame_keys(tmp_path: Path) -> None:
    manifest_path = write_checker_dataset(tmp_path, make_manifest())
    valid = make_raw_vector().numpy().tolist()
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "observation.state": [valid, valid],
            "action": [valid, valid],
        }
    ).to_parquet(tmp_path / "data/chunk-000/file-000.parquet")
    timing_path = tmp_path / "meta/timing/episode-000000.jsonl"
    first = json.loads(timing_path.read_text(encoding="utf-8"))
    second = json.loads(json.dumps(first))
    second["frame_index"] = 99
    for source in second["state"]["source_timing"]["sources"].values():
        source["generation"] += 1
    timing_path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    report = run_check(
        Namespace(
            dataset_root=tmp_path,
            manifest=manifest_path,
            allow_unavailable=False,
            report_json=None,
        )
    )

    assert report["status"] == "FAIL"
    assert "frame keys must match Parquet exactly" in report["errors"][0]


def test_legacy_real_episode_remains_raw18_and_timing_sidecar_is_readable() -> None:
    if not LEGACY_DATASET_ROOT.exists():
        pytest.skip("Local legacy JZ Pin timed compatibility artifact is not present")

    info = json.loads((LEGACY_DATASET_ROOT / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((LEGACY_DATASET_ROOT / "meta/stats.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(
        LEGACY_DATASET_ROOT / "data/chunk-000/file-000.parquet",
        columns=["observation.state", "action"],
    )
    timing_lines = (
        (LEGACY_DATASET_ROOT / "meta/timing/episode-000000.jsonl").read_text(encoding="utf-8").splitlines()
    )
    first_timing_record = json.loads(timing_lines[0])
    dataset = LeRobotDataset(
        "local/jz_robot_pin_timed_real_20260711_190502",
        root=LEGACY_DATASET_ROOT,
        video_backend="pyav",
    )

    assert len(dataset) == 894
    assert dataset.meta.features["observation.state"]["shape"] == (RAW_DIM,)
    assert dataset.meta.features["action"]["shape"] == (RAW_DIM,)
    assert info["features"]["observation.state"] == {
        "dtype": "float32",
        "shape": [RAW_DIM],
        "names": list(RAW_FEATURE_NAMES),
    }
    assert info["features"]["action"] == {
        "dtype": "float32",
        "shape": [RAW_DIM],
        "names": list(RAW_FEATURE_NAMES),
    }
    assert len(stats["observation.state"]["mean"]) == RAW_DIM
    assert len(stats["action"]["mean"]) == RAW_DIM
    assert np.asarray(frame.iloc[0]["observation.state"]).shape == (RAW_DIM,)
    assert np.asarray(frame.iloc[0]["action"]).shape == (RAW_DIM,)
    assert first_timing_record["command"]["action_key_count"] == RAW_DIM
