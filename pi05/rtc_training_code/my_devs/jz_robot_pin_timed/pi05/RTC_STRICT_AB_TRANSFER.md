# Strict Full PI0.5 Training-Time RTC Transfer

Copy these three files into the target repository directory
`my_devs/jz_robot_pin_timed/pi05/`:

```text
rtc_strict_ab_full_head_right_15e_d5_seed1000.json
train_rtc_strict_ab_full_head_right_15_epochs.sh
RTC_STRICT_AB_TRANSFER.md
```

The target repository must already contain the training-time RTC implementation
and the existing `train_pi05.sh`. This package does not replace model source
files or transfer checkpoints.

The configuration reproduces the working Full 15-epoch PI0.5 setup with the
same 100 episodes, head/right cameras, task, single-process batch 32,
optimizer, scheduler, seed, and checkpoint cadence. It only enables
training-time clean-prefix RTC with `max_delay=5` and `min_postfix_steps=1`.
It does not enable inference-time VJP RTC.

## 1. Contract Only

This command does not access data, models, GPUs, or the trainer:

```bash
cd /mnt/data2/ybd/vla_act/cqy

CUDA_VISIBLE_DEVICES=<GPU_ID> \
PRINT_CONTRACT_ONLY=true \
RUN_STAMP=contract_check \
bash my_devs/jz_robot_pin_timed/pi05/train_rtc_strict_ab_full_head_right_15_epochs.sh
```

## 2. Source-Host Preflight

This validates data, assets, schema, statistics, task, and the final trainer
command without starting training:

```bash
cd /mnt/data2/ybd/vla_act/cqy

CUDA_VISIBLE_DEVICES=<GPU_ID> \
DRY_RUN=true \
RUN_STAMP=preflight \
bash my_devs/jz_robot_pin_timed/pi05/train_rtc_strict_ab_full_head_right_15_epochs.sh
```

## 3. Two-Step Capacity Smoke

This performs two real training steps with formal batch size 32 and saves no
checkpoint:

```bash
cd /mnt/data2/ybd/vla_act/cqy

CUDA_VISIBLE_DEVICES=<GPU_ID> \
SOURCE_SMOKE_ONLY=true \
RUN_STAMP=capacity_smoke \
bash my_devs/jz_robot_pin_timed/pi05/train_rtc_strict_ab_full_head_right_15_epochs.sh
```

Do not start the formal run if this step reports OOM, NaN/Inf, a non-finite
loss, or a distributed process error.

## 4. Formal 15-Epoch Run

Leave `PRINT_CONTRACT_ONLY`, `DRY_RUN`, `SOURCE_SMOKE_ONLY`, and
`STEPS_OVERRIDE` unset:

```bash
cd /mnt/data2/ybd/vla_act/cqy

CUDA_VISIBLE_DEVICES=<GPU_ID> \
RUN_STAMP=formal_v1 \
bash my_devs/jz_robot_pin_timed/pi05/train_rtc_strict_ab_full_head_right_15_epochs.sh
```

Expected checkpoints:

```text
checkpoints/005300/pretrained_model
checkpoints/010600/pretrained_model
checkpoints/015900/pretrained_model
```

Transfer all three checkpoints back for offline prefix-0/prefix-5 evaluation.
Do not run an armed robot client directly from a newly transferred checkpoint.
