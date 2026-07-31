# JZ PI0.5 Training-Time RTC Implementation

## Status

Implemented on 2026-07-29. CPU tests plus single-GPU, two-camera four-GPU, and three-camera four-GPU RTC smoke
training have passed. A three-camera 10-epoch B pilot has been started; no policy-server dry-run or robot
execution was started.

The original A training process that was already running before these edits was unaffected: it loaded its
Python modules and model weights before the source files changed. It completed at step 10600 with `status=PASS`
and checkpoints at steps 5300 and 10600.

The existing distributed metrics tracker reports `smpl`/`epch` four times too large because the effective batch
is multiplied by the process count twice. This is a logging-only issue: optimizer steps, save steps, and the
10-epoch step calculation are correct. It was not changed as part of the RTC implementation.

## Backup

The pre-change source snapshot is:

```text
my_devs/jz_robot_pin_timed/pi05/backups/pre_training_rtc_20260728.tar.gz
SHA256 19c806705387ab432c9a082c0a44f888b08b2f3626186bc99ee36f02e582e9f8
```

## Implemented Contract

- `rtc_training.enabled=false` is the default and preserves the standard PI0.5 model structure and flow path.
- RTC training samples one integer delay for each batch item.
- A committed prefix uses clean actions and `token_flow_time=0`.
- The existing sample-level global flow time still conditions the action expert AdaRMS path.
- Local token time and an explicit prefix-mask embedding condition each action token.
- The local-condition residual output and prefix embedding are zero-initialized so an RTC-enabled model starts
  from the base action-token embedding before learning the new condition.
- Flow-matching loss is computed only on postfix elements.
- Every sample is normalized by its own valid postfix count before the batch mean, preventing long-prefix
  samples from receiving less weight.
- Delay zero retains full-chunk generation training.
- An optional measured delay histogram can provide 90% of the sampling distribution; the remaining 10% is
  uniform coverage by default.
- PyTorch inference accepts `action_prefix` and `prefix_length`, clamps the prefix before and after every Euler
  update, and zeros its predicted velocity. This path does not use VJP/autograd.
- Training-time prefix clamp and the existing inference-time VJP RTC guidance are mutually exclusive.
- Standard PI0.5 checkpoints load non-strictly only when RTC training is explicitly enabled, allowing the new
  condition modules to initialize while retaining all base weights.

## JZ Training Switch

The same script now exposes two methods:

```text
TRAINING_MODE=standard  # A: original flow matching, default
TRAINING_MODE=rtc       # B: clean-prefix RTC training
```

For B, the script passes:

```text
--policy.rtc_training.enabled=true
--policy.rtc_training.max_delay=${RTC_MAX_DELAY}
--policy.rtc_training.min_postfix_steps=${RTC_MIN_POSTFIX_STEPS}
```

The original three-camera B pilot must continue to use `FINETUNE_MODE=expert_only` when it is reproduced.
It is not a controlled comparison against the working head-right/full checkpoint. The separate strict A/B
rerun intentionally uses `FINETUNE_MODE=full`, `CAMERA_MODE=head_right`, single-process batch 32, and
`RTC_MAX_DELAY=5` so the only
training-method change from that baseline is RTC conditioning. `RTC_MAX_DELAY` must be locked from the target
runtime end-to-end latency distribution; the original value `10` is only the historical pilot setting.

## Verification Completed

Executed only in conda environment `lerobot_flex`:

```text
bash -n my_devs/jz_robot_pin_timed/pi05/train_pi05.sh             PASS
pytest tests/policies/rtc/test_configuration_rtc.py \
       tests/policies/pi0_pi05/test_pi05_training_rtc.py          15 passed
```

The installed `pi05_base/config.json` was also loaded through `PreTrainedConfig.from_pretrained` with the three
RTC CLI overrides; it produced `PI05Config` with `enabled=true`, `max_delay=6`, and `min_postfix_steps=2`.

GPU smoke results on RTX 4090 D:

```text
single GPU 3: batch=1, steps=2, loss=0.701 -> 0.379, status=PASS
four GPU 4-7: per-device batch=8, global batch=32, steps=2,
              loss=0.912 -> 0.519, status=PASS
three-camera four GPU 4-7: per-device batch=8, global batch=32, steps=2,
                           loss=0.933 -> 0.522, status=PASS
trainable parameters: 695,523,360
total parameters:     3,618,858,768
```

The base checkpoint reported exactly six missing keys: the known tied PaliGemma embedding key and the five new
RTC condition parameters. No non-RTC key was missing.

Four-GPU peak memory below includes another user's 1.5-1.7 GiB VLM-server allocation on every card:

```text
GPU4 18152 MiB used, 5941 MiB free
GPU5 18044 MiB used, 6049 MiB free
GPU6 18120 MiB used, 5973 MiB free
GPU7 17970 MiB used, 6123 MiB free
```

After that service moved away from GPU4-7, the three-camera smoke peaked at 16265-16461 MiB per card and kept
7632-7828 MiB free.

Smoke artifacts:

```text
logs/pi05_jz100_model16_head_right_expert_b_rtc_1gpu_b1_smoke/
logs/pi05_jz100_model16_head_right_expert_b_rtc_4gpu_b8_smoke/
logs/pi05_jz100_model16_head_left_right_expert_b_rtc_4gpu_b8_smoke/
```

The current pilot runs in tmux session `pi05_rtc_3cam_e10` with run name
`pi05_jz100_model16_head_left_right_expert_b_rtc_e10_seed1000` on GPU4-7. It uses all three cameras, 10600
optimizer steps, and saves checkpoints at steps 5300 and 10600.

Ruff and Black are not installed in `lerobot_flex`; no package was downloaded. Formatting therefore requires
review during tomorrow's acceptance.

## Acceptance Still Required

1. Confirm the standard disabled path against the existing base checkpoint with fixed inputs/noise.
2. Add a targeted assertion that the new RTC condition parameters receive finite gradients.
3. Verify prefix immutability across all ten Euler steps with a real B checkpoint after training.
4. Measure end-to-end runtime delay and replace the pilot `RTC_MAX_DELAY=10` before confirmatory B training.
5. Do not use a model output on the physical robot until the separate dry-run and field authorization gates
   pass.
