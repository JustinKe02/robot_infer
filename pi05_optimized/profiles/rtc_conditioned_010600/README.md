# PI0.5 RTC-conditioned 010600 profile

This profile locks the existing training-time RTC checkpoint and the independent
`torch_rtc_conditioned` backend.

```text
policy:             tk_infer/pi05/checkpoints/pi05_jz100_model16_head_left_right_expert_b_rtc_e10_seed1000_010600/pretrained_model
configured steps:   10600
path-proven step:   unavailable because the transferred parent directory is non-numeric
camera profile:     three_camera
model/wire actions: model16/raw18
chunk size:         50
max trained delay:  10
min postfix steps:  1
task:               Put the bottle on the right into the basket on the left.
backend:            torch_rtc_conditioned
default port:       18089
fingerprint:        039ef411871f75e8504b7b72ccb299c29c4cdf3a99e7bfbc241a3daae7bfaa57
```

The backend uses clean action prefixes and never installs inference-time VJP
RTC. It rejects standard checkpoints, missing learned RTC parameters, prefix
overflow, and nonzero delay without enough model16 leftover. The profile
verifier pins the exact seven-file checkpoint and four-file tokenizer contents
with SHA-256 before loading the backend.

Configuration-only checks do not load a model or open a socket:

```bash
CONFIG_ONLY=true \
bash tk_infer/pi05_optimized/profiles/rtc_conditioned_010600/run_policy_server.sh

PRINT_COMMAND_ONLY=true \
bash tk_infer/pi05_optimized/profiles/rtc_conditioned_010600/run_policy_server.sh
```

Checkpoint load-only validation does not run inference or open a socket:

```bash
CHECK_POLICY_LOAD=true \
bash tk_infer/pi05_optimized/profiles/rtc_conditioned_010600/run_policy_server.sh
```

Starting the policy server is not a robot action path. Do not start a robot
client without a separate, explicit authorization and the required safety
gates.

## Robot client stages

The robot-capable client profile is kept outside the optimized server tree at
`tk_infer/pi05/profiles/rtc_conditioned_010600/`. It locks the policy URL to
`127.0.0.1:18089`, the exact
path-unproven checkpoint identity, the three-camera input contract, and the
training task. Every client command requires:

```bash
TK_PI05_RTC_CONDITIONED_010600_CONFIRMED=1
```

Run the stages in this order:

```text
run_inference_smoke.sh      one live observation and inference; no send_action
run_single_step_dry_run.sh  local transport; no command UDP
run_single_step_armed.sh    UDP, 5 Hz, one second, at most one action
run_rtc_dry_run.sh          local transport at 20 Hz
run_rtc_armed.sh            UDP, one second, at most ten actions
```

Armed entrypoints retain the three global robot confirmations. RTC armed also
requires `JZ_PI05_RTC_CONDITIONED_SINGLE_STEP_ARMED_PASSED=1`, which may be set
only after the bounded single-step armed result has been reviewed. Joint-delta
checks remain enabled in every profile command.
