# Step 010600 Inference Profile

This directory is the fixed deployment profile for the epoch-10 checkpoint
described in `tk_infer/pi05/CHECKPOINTS.md`. It ports the generic three-camera
inference architecture documented by `INFERENCE_MODULE_REPORT.md` and changes
only the camera feature set required by this checkpoint.

## Weight Contract

```text
policy path:      tk_infer/pi05/checkpoints/010600/pretrained_model
checkpoint step: 10600
configured steps:15900
complete:         false
policy:           pi05 full model.safetensors, not a PEFT adapter
cameras:          camera_head:5555 + camera_right:5557
model boundary:   state16/action16 for both arms
wire boundary:    raw18 for both arms
chunk size:       50
task:             jz robot pin timed vr teleoperation
```

The checkpoint has no serialized `training_state`, optimizer, scheduler, or
global-step file. Its actual step is locked by the verified source directory,
all seven full SHA-256 hashes, path-derived step metadata, and the checkpoint
fingerprint. `verify_checkpoint.py` checks all of these before a real server
start or `CHECK_POLICY_LOAD`.

## Camera Adaptation

The generic runtime retains the original `three_camera` profile. This fixed
profile selects `CAMERA_PROFILE=head_right`, so it constructs only:

```text
observation.images.camera_head   ZMQ 5555   3x720x1280
observation.images.camera_right  ZMQ 5557   3x480x640
```

It does not construct `camera_left:5556`, because that key is absent from the
checkpoint input features. The state/action path is otherwise unchanged:

```text
raw18 observation -> checkpoint preprocessor -> model16
model16 PI0.5 chunk -> checkpoint postprocessor -> raw18 action
```

No executable-action transformation is applied. Both arms remain
model-controlled according to the checkpoint's full model16/raw18 contract.
The generic raw18 finite/shape/force checks and Robot driver safety checks
remain active.

## Required Configuration

Every profile command requires explicit acknowledgement that this is an
intermediate checkpoint:

```bash
TK_PI05_010600_INTERMEDIATE_CONFIRMED=1
```

## Entrypoints

| Stage | Entrypoint | Robot command behavior |
| --- | --- | --- |
| Server | `run_policy_server.sh` | No robot access |
| Offline smoke | `run_offline_smoke.sh` | Recorded dataset, one real CUDA inference, no robot |
| Health | `run_health_check.sh` | HTTP `/health` only |
| Connect smoke | `run_connect_smoke.sh` | Reads live state/cameras, no inference or action |
| Inference smoke | `run_inference_smoke.sh` | One live inference, no `send_action` |
| Single dry-run | `run_single_step_dry_run.sh` | Local transport, no command UDP |
| Single armed | `run_single_step_armed.sh` | UDP, 5 Hz, 1 second, at most one action |
| Async dry-run | `run_async_single_step_dry_run.sh` | Local transport |
| Async armed | `run_async_single_step_armed.sh` | Requires prior armed single-step confirmation |
| RTC dry-run | `run_rtc_dry_run.sh` | Local transport |
| RTC armed | `run_rtc_armed.sh` | Requires prior armed single-step confirmation |

Armed entrypoints retain the three global confirmations. Async and RTC armed
entrypoints additionally require:

```bash
JZ_PI05_SINGLE_STEP_ARMED_PASSED=1
```

## Acceptance Order

1. `PRINT_COMMAND_ONLY=true` on the fixed launcher.
2. `CHECK_POLICY_LOAD=true` on the fixed server launcher.
3. Fixed-profile offline smoke.
4. Fixed-profile health check.
5. Connect smoke.
6. Inference smoke.
7. Single-step dry-run.
8. On-site authorized single-step armed run, limited to one action.
9. Async single-step dry-run and then separately authorized armed validation.
10. RTC dry-run and then separately authorized armed validation.

Passing a checkpoint identity check proves which files are loaded. It does not
prove physical task semantics, approve robot motion, or replace the emergency
stop and on-site operator.
