# YBD JZ PI0.5 inference

This directory isolates the current YBD deployment from the existing audited
JZ PI0.5 inference runtime.  Code in this directory should reuse the existing
runtime and keep checkpoint-specific behavior local.

## Fixed checkpoint

```text
/mnt/data2/ybd/vla_act/cqy/my_devs/jz_robot_pin_timed/pi05/outputs/
pi05_jz100_model16_head_right_expert_a_e10_seed1000/
checkpoints/010600/pretrained_model
```

The checkpoint must be loaded with its own `policy_preprocessor.json` and
`policy_postprocessor.json`; loading only `model.safetensors` is not allowed.

## Fixed inference contract

- Task: `Put the bottle on the right into the basket on the left.`
- The existing client connects and samples all three live cameras.
- Before checkpoint preprocessing, the YBD service removes
  `observation.images.camera_left`; the model receives only
  `observation.images.camera_head` and `observation.images.camera_right`.
- State input: unchanged raw18 observation containing both arms.
- Model output: normalized model16, retained unchanged for diagnostics and the
  existing protocol.
- Execution: right arm and right gripper only.
- The checkpoint's complete postprocessor runs before the YBD execution
  guard, preserving action unnormalization and model16-to-raw18 expansion.
- On the final raw18 chunk, YBD replaces indices `0:7` with the observed
  left-arm joints and converts the observed left-gripper opening through the
  checkpoint schema before replacing index `14`.
- Final raw18 indices `7:14` and `16` remain the model's right-arm and
  right-gripper commands; force indices `15` and `17` are set by the schema to
  80.

## Safety status

No YBD armed launcher has been added. Offline inference and a no-`send_action`
dry-run must pass before any on-robot command is considered. Armed execution,
including a fixed left-side hold setpoint at the client send boundary,
requires a separate implementation review and on-site authorization.
