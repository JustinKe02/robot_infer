# tk_infer PI0.5 Inference

This directory contains the complete project-specific PI0.5 inference runtime
for `jz_robot_pin_timed`. It is independent of
`my_devs/jz_robot_pin_timed/pi05/rtc_infer` and does not import code from that
directory.

The runtime still uses the public/core LeRobot implementations in
`src/lerobot` for the policy, processors, dataset feature conversion, cameras,
and robot driver.

## Runtime Architecture

```text
selected live cameras + timed raw18 robot state
                    |
                    v
          tk_infer/pi05 robot client
                    |
       InferenceRequest over authenticated HTTP
                    |
                    v
          tk_infer/pi05 policy server
                    |
 raw18 -> model16 preprocessor -> PI0.5 predict_action_chunk
                    |
 model16 -> raw18 postprocessor, fixed gripper force=80
                    |
       InferenceResponse: model16 + raw18 chunks
                    |
                    v
   single-step selection or asynchronous RTC action queue
                    |
  ActionSafety -> Robot safety/clamp -> local dry-run or UDP armed
```

The server supports both `single_step` and `rtc` requests in one process. The
client also supports `async_single_step`, which uses the asynchronous queues
without RTC leftover guidance.

## Directory Boundary

- `run_policy_server.py`: checkpoint validation, model load, RTC installation,
  `/health`, and `/infer` server entrypoint.
- `run_robot_client.py`: health contract, robot configuration, smoke modes,
  live runtime, cleanup, and summary output.
- `runtime/checkpoint.py`: complete-step, processor, raw18/model16 schema,
  camera, tokenizer, and checkpoint fingerprint validation.
- `runtime/protocol.py`: restricted pickle protocol and 16D/18D request and
  response contracts.
- `runtime/policy_service.py`: serialized model inference and RTC mode switch.
- `runtime/client_runtime.py`: sensor, producer, actor, single-step, and RTC
  loops.
- `runtime/action_queue.py`: model16 leftovers and executable raw18 queue.
- `runtime/robot_builder.py`: timed UDP state, three ZMQ cameras, dry-run/armed
  transport, timing constraints, and Robot safety defaults.
- `runtime/safety.py`: triple armed confirmation and raw18 action validation.
- `runtime/http_server.py`, `runtime/remote_client.py`: authenticated transport
  with body limits and content-type checks.
- `run_state/`, `logs/`, `outputs/`: local runtime artifacts; ignored by Git
  and constrained to this directory.
- `profiles/step_010600/`: fixed epoch-10 checkpoint identity and staged
  single-step, async-single-step, and RTC deployment launchers.

## 1. No-Hardware Configuration Checks

This command does not read a checkpoint, load a model, open a server socket,
connect to the robot, read cameras, or send an action:

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
bash tk_infer/pi05/run_config_checks.sh
```

Any launcher can print its final command without starting it:

```bash
PRINT_COMMAND_ONLY=true \
SERVER_AUTH_TOKEN='<TOKEN>' \
SERVER_URL=http://<HIGH3_IP>:8088 \
bash tk_infer/pi05/run_single_step_dry_run.sh
```

## 2. Policy Server

The server requires an explicit complete `pretrained_model` directory. It does
not silently choose one of the incomplete local checkpoints.

### Current uploaded checkpoint

The current uploaded checkpoint has been validated and selected through
`checkpoints/current`. It is the epoch-10 intermediate checkpoint copied from
the training host's `checkpoints/010600/pretrained_model` directory; it is not
the configured final step:

```text
resolved path: checkpoints/010600/pretrained_model
epoch:         10/15
step:          10600/15900 (intermediate)
camera profile: head_right
inputs:        camera_head + camera_right + model16 state
output:        model16 action -> raw18 wire action
fingerprint:   4698315f6936f9e9ef19017cfdb873588eba771fdb23595879ce2a7703b4c8dd
```

See [`CHECKPOINTS.md`](CHECKPOINTS.md) for the source path, file hashes, naming
rules, and the inventory used when future checkpoints are added.

Start the fixed profile. It verifies all seven file hashes before loading and
still reports `complete_step=false`:

```bash
CONDA_ROOT=/home/luzhuang/miniconda3 \
TK_PI05_010600_INTERMEDIATE_CONFIRMED=1 \
SERVER_HOST=0.0.0.0 \
SERVER_PORT=8088 \
SERVER_AUTH_TOKEN='<TOKEN>' \
bash tk_infer/pi05/run_current_server.sh
```

Use the fixed profile for client health. It exports the exact step, configured
steps, fingerprint, absolute path, incomplete status, camera profile, and
camera keys before contacting the server:

```bash
TK_PI05_010600_INTERMEDIATE_CONFIRMED=1 \
SERVER_AUTH_TOKEN='<TOKEN>' \
SERVER_URL=http://<HIGH3_IP>:8088 \
bash tk_infer/pi05/profiles/step_010600/run_health_check.sh
```

The fixed profile sets only the checkpoint-specific camera selection:

```bash
CAMERA_PROFILE=head_right
```

The generic runtime still supports the original `three_camera` profile. For
this checkpoint, `head_right` creates only the `camera_head` and
`camera_right` ZMQ clients on ports 5555 and 5557; it does not create or wait
for `camera_left` on port 5556. No action transform is applied: the checkpoint
continues to predict the complete model16 chunk for both arms, and its
serialized postprocessor expands that chunk directly to executable raw18.

For loopback-only development:

```bash
POLICY_PATH=/absolute/path/checkpoints/<final-step>/pretrained_model \
TOKENIZER_PATH=/home/luzhuang/cqy/aaa/flexible_lerobot/assets/modelscope/google/paligemma-3b-pt-224 \
CONDA_ROOT=/home/luzhuang/miniconda3 \
bash tk_infer/pi05/run_server.sh
```

For a trusted-LAN server, a token is mandatory:

```bash
POLICY_PATH=/absolute/path/checkpoints/<final-step>/pretrained_model \
TOKENIZER_PATH=/home/luzhuang/cqy/aaa/flexible_lerobot/assets/modelscope/google/paligemma-3b-pt-224 \
CONDA_ROOT=/home/luzhuang/miniconda3 \
SERVER_HOST=0.0.0.0 \
SERVER_PORT=8088 \
SERVER_AUTH_TOKEN='<TOKEN>' \
bash tk_infer/pi05/run_server.sh
```

Use `CHECK_POLICY_LOAD=true` to load and validate the model/processors and then
exit without listening.

## 3. Server Health Only

This contacts only `/health`; it does not construct or connect a robot:

```bash
CONDA_ROOT=/home/luzhuang/miniconda3 \
TK_PI05_010600_INTERMEDIATE_CONFIRMED=1 \
SERVER_AUTH_TOKEN='<TOKEN>' \
SERVER_URL=http://<HIGH3_IP>:8088 \
bash tk_infer/pi05/profiles/step_010600/run_health_check.sh
```

The client requires protocol version 3, PI0.5, model16 state/action, raw18 wire
actions, the `jz_pin_opening16_v1` schema, the selected camera profile, and a
complete checkpoint unless an explicit audited checkpoint contract is
provided.

## 4. Read-Only Live Inference Smoke

This mode connects to timed state and the cameras selected by
`CAMERA_PROFILE`, sends one inference request, validates the selected raw18
action, and disconnects. It never calls `robot.send_action()` and does not use
command UDP:

```bash
CONDA_ROOT=/home/luzhuang/miniconda3 \
TK_PI05_010600_INTERMEDIATE_CONFIRMED=1 \
SERVER_AUTH_TOKEN='<TOKEN>' \
SERVER_URL=http://<HIGH3_IP>:8088 \
ORIN_IP=192.168.1.81 \
bash tk_infer/pi05/profiles/step_010600/run_inference_smoke.sh
```

This is read-only with respect to actions, but it still accesses live state and
cameras and therefore is not part of automated testing.

## 5. Single-Step Dry-Run

Dry-run reads live state/cameras and performs policy inference. The Robot uses
the local transport, so no command UDP packet is sent:

```bash
CONDA_ROOT=/home/luzhuang/miniconda3 \
TK_PI05_010600_INTERMEDIATE_CONFIRMED=1 \
SERVER_AUTH_TOKEN='<TOKEN>' \
SERVER_URL=http://<HIGH3_IP>:8088 \
ORIN_IP=192.168.1.81 \
STATE_BIND_IP=0.0.0.0 \
STATE_PORT=39010 \
COMMAND_PORT=39020 \
bash tk_infer/pi05/profiles/step_010600/run_single_step_dry_run.sh
```

## 6. Single-Step Armed

This is the isolated equivalent of the supplied PI0.5 armed command. It can
send UDP robot commands and must not be run without on-site authorization,
physical emergency-stop readiness, correct initial pose, and confirmed state
and camera services:

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

CONDA_ROOT=/home/luzhuang/miniconda3 \
TK_PI05_010600_INTERMEDIATE_CONFIRMED=1 \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
JZ_POLICY_INFERENCE_ARMED=1 \
SERVER_AUTH_TOKEN='<TOKEN>' \
SERVER_URL=http://<HIGH3_IP>:8088 \
ORIN_IP=192.168.1.81 \
STATE_BIND_IP=0.0.0.0 \
STATE_PORT=39010 \
COMMAND_PORT=39020 \
bash tk_infer/pi05/profiles/step_010600/run_single_step_armed.sh
```

Armed mode is gated in the profile shell, Python client, and Robot driver. The
first profile run is fixed to 5 Hz, one second, and at most one action. The
shared Robot initial and per-step joint delta limits remain at `0.02 rad`; this
task does not inherit the ACT example's `10.0 rad` overrides.

## 7. Async Single-Step And RTC

Use the same environment variables with one of:

```bash
bash tk_infer/pi05/profiles/step_010600/run_async_single_step_dry_run.sh
bash tk_infer/pi05/profiles/step_010600/run_rtc_dry_run.sh
```

Armed async/RTC entrypoints additionally require
`JZ_PI05_SINGLE_STEP_ARMED_PASSED=1` and use the corresponding
`run_async_single_step_armed.sh` or `run_rtc_armed.sh` profile launcher.

RTC keeps normalized model16 leftovers for the next request and separately
queues postprocessed raw18 actions for execution. The default execution
horizon is 10, queue low watermark is 30, maximum queue size is 50, and the
empty queue strategy is fail-closed `stop`.

## Tests

All automated tests are no-hardware tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
conda run -n lerobot_flex pytest -q tk_infer/pi05/tests
```
