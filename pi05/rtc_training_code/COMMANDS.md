# PI0.5 Training-Time RTC Commands

Run all commands in conda environment `lerobot_flex`. These commands do not
connect to or command the robot.

## Paths on the source server

```bash
cd /mnt/data2/ybd/vla_act/cqy
export POLICY_PATH=/mnt/data2/ybd/vla_act/cqy/my_devs/jz_robot_pin_timed/pi05/outputs/pi05_jz100_model16_head_left_right_expert_b_rtc_e10_seed1000/checkpoints/010600/pretrained_model
export PALIGEMMA_TOKENIZER_PATH=/mnt/data2/ybd/vla_act/cqy/assets/modelscope/google/paligemma-3b-pt-224
```

Replace both paths after transferring to another machine.

## Checkpoint load only

This loads the policy and serialized raw18/model16 processors, prints health
metadata, performs no inference, opens no listening socket, and performs no
robot I/O.

```bash
CUDA_VISIBLE_DEVICES=4 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run -n lerobot_flex python \
  my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_policy_server.py \
  --policy-path "$POLICY_PATH" \
  --tokenizer-path "$PALIGEMMA_TOKENIZER_PATH" \
  --device cuda \
  --host 127.0.0.1 \
  --port 0 \
  --check-policy-load
```

## Offline synthetic inference

This loads the B checkpoint and runs three deterministic, synthetic-observation
inferences. It verifies that `prefix_len=0` matches an ordinary call and that a
five-step clean prefix is exactly clamped. It does not read a dataset or access
a robot.

```bash
CUDA_VISIBLE_DEVICES=4 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run -n lerobot_flex python \
  pi05_rtc_training_code_transfer/tools/offline_training_rtc_inference.py \
  --policy-path "$POLICY_PATH" \
  --tokenizer-path "$PALIGEMMA_TOKENIZER_PATH" \
  --device cuda \
  --prefix-len 5 \
  --seed 1000
```

## Training-time RTC dry run

This validates the 100-episode, three-camera, raw18-to-model16 training setup
and prints the command. `DRY_RUN=true` prevents the trainer from starting.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_PROCESSES=4 \
BATCH_SIZE=8 \
EPOCHS=10 \
CHECKPOINT_EVERY_EPOCHS=5 \
CAMERA_MODE=three \
FINETUNE_MODE=expert_only \
TRAINING_MODE=rtc \
RTC_MAX_DELAY=10 \
RTC_MIN_POSTFIX_STEPS=1 \
RUN_NAME=pi05_jz100_model16_head_left_right_expert_b_rtc_dry_run \
DRY_RUN=true \
bash my_devs/jz_robot_pin_timed/pi05/train_pi05.sh
```

## Focused source tests

```bash
conda run -n lerobot_flex pytest -q \
  tests/policies/rtc/test_configuration_rtc.py \
  tests/policies/pi0_pi05/test_pi05_training_rtc.py \
  tests/robots/test_jz_robot_pin_timed_training_schema.py
```

Do not run `rtc_infer/run_rtc_armed.sh` for this checkpoint. That script uses
the older inference-time VJP RTC contract, while the B checkpoint uses
training-time clean-prefix RTC.

## Strict head-right/full RTC rerun

This is a source-training-host-only controlled rerun against the working
head-right/full PI0.5 baseline. It keeps the original 100 episodes unchanged
and locks `max_delay=5` from the measured warm-runtime P95.

Contract inspection, with no data/model/GPU/trainer access:

```bash
PRINT_CONTRACT_ONLY=true \
bash my_devs/jz_robot_pin_timed/pi05/train_rtc_strict_ab_full_head_right_15_epochs.sh
```

Full source-host preflight, without starting the trainer:

```bash
CUDA_VISIBLE_DEVICES=0 \
DRY_RUN=true \
bash my_devs/jz_robot_pin_timed/pi05/train_rtc_strict_ab_full_head_right_15_epochs.sh
```

After preflight, run the locked two-step capacity smoke on the source host. It
uses the formal per-device batch and four-process topology but saves no
checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
SOURCE_SMOKE_ONLY=true \
RUN_STAMP=capacity_smoke \
bash my_devs/jz_robot_pin_timed/pi05/train_rtc_strict_ab_full_head_right_15_epochs.sh
```

Formal source-host training removes `DRY_RUN=true` and leaves
`SOURCE_SMOKE_ONLY` unset only after the preflight output confirms
`head_right`, `full`, `rtc`, 15 epochs, single-process batch 32, task text, base
path, and `RTC_MAX_DELAY=5`, and the capacity smoke completes without OOM.
