# JZ Robot Pin Timed - PI0.5

本目录是 `jz_robot_pin_timed` 的 PI0.5 独立训练模块。所有新增训练脚本、缓存、日志和输出均限制在本目录；脚本只读取原始数据、已有 PI0.5 基模和 tokenizer，不会连接或操控机器人。

## 固定输入

- 原始数据集：`/mnt/data2/ybd/vla_act/cqy/jz_robot_pin_timed_merged_100eps_20260728`
- 两相机 A 视图：`/mnt/data2/ybd/vla_act/cqy/jz_robot_pin_timed_merged_100eps_20260728_pi05_head_right`
- 三相机 B 视图：`/mnt/data2/ybd/vla_act/cqy/jz_robot_pin_timed_merged_100eps_20260728_pi05_head_left_right`
- 数据规模：100 episodes、33898 frames、20 FPS
- 当前 B 训练相机：`camera_head`、`camera_left`、`camera_right`
- 语言任务：`Put the bottle on the right into the basket on the left.`
- 原始边界：18D state/action
- 模型边界：16D state/action（丢弃两路 force，并按 schema 统一夹爪 opening 方向）
- Conda 环境：`lerobot_flex`
- PI0.5 基模：`/mnt/data2/ybd/vla_act/cqy/assets/modelscope/lerobot/pi05_base`
- PaliGemma tokenizer：`/mnt/data2/ybd/vla_act/cqy/assets/modelscope/google/paligemma-3b-pt-224`

## Smoke train

```bash
cd /mnt/data2/ybd/vla_act/cqy
bash my_devs/jz_robot_pin_timed/pi05/smoke_train.sh
```

默认用 batch size 1 跑 2 个真实的 forward/backward/update step，不保存 14GB checkpoint。日志写入本目录的 `logs/`，训练运行目录写入 `outputs/`。

## 正式训练 15 epochs

```bash
cd /mnt/data2/ybd/vla_act/cqy
bash my_devs/jz_robot_pin_timed/pi05/train_15_epochs.sh
```

默认模式为 `expert`：冻结 PaliGemma，训练 action expert 和 PI0.5 动作/时间投影。默认 batch size 32：`ceil(33898 / 32) = 1060` steps/epoch，总计 `15900` steps；每 5 epochs（5300 steps）保存一次 checkpoint。

## 标准训练与训练时 RTC

`train_pi05.sh` 默认 `TRAINING_MODE=standard`，保持原始 PI0.5 flow matching。只有显式设置
`TRAINING_MODE=rtc` 才启用 clean prefix、逐样本 delay、局部 token time 和 postfix-only loss。

当前三相机 RTC pilot 配置：

```bash
cd /mnt/data2/ybd/vla_act/cqy

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_PROCESSES=4 \
BATCH_SIZE=8 \
EPOCHS=10 \
CHECKPOINT_EVERY_EPOCHS=5 \
FINETUNE_MODE=expert_only \
TRAINING_MODE=rtc \
RTC_MAX_DELAY=10 \
RTC_MIN_POSTFIX_STEPS=1 \
CAMERA_MODE=three \
RUN_NAME=pi05_jz100_model16_head_left_right_expert_b_rtc_e10_seed1000 \
  bash my_devs/jz_robot_pin_timed/pi05/train_pi05.sh
```

### Strict A/B baseline-preservation rerun

The existing three-camera expert-only pilot is not a controlled comparison
against the working head-right/full PI0.5 checkpoint. The strict rerun keeps
the original 100 episodes unchanged and locks `head_right`, full finetuning,
15 epochs, single-process batch 32, seed 1000, and the exact bottle task. The only
method change is training-time RTC with the measured warm-runtime P95 delay
bound of five 20 Hz control steps.

This launcher is for the four-GPU source training host only. On another host,
only its contract-only mode may be used:

```bash
PRINT_CONTRACT_ONLY=true \
bash my_devs/jz_robot_pin_timed/pi05/train_rtc_strict_ab_full_head_right_15_epochs.sh
```

Source-host preflight without starting training:

```bash
CUDA_VISIBLE_DEVICES=0 \
DRY_RUN=true \
bash my_devs/jz_robot_pin_timed/pi05/train_rtc_strict_ab_full_head_right_15_epochs.sh
```

`CAMERA_MODE=head_right` 保留昨晚 A 的两相机边界；`CAMERA_MODE=three` 使用三路相机并创建独立视图，
不会覆盖 A 的数据元信息或 checkpoint。

`RTC_MAX_DELAY=10` 只是接口示例。正式 B 训练前必须用目标 runtime 的端到端 P95 延迟换算 delay，
再锁定该值；不得根据训练结果反向选择。训练时 RTC 和现有 inference-time VJP RTC guidance 互斥。

三组正式训练使用相同的 100 episodes 和 15 epochs，按 `full -> expert -> LoRA` 顺序串行执行：

```bash
bash my_devs/jz_robot_pin_timed/pi05/train_three_modes_15_epochs.sh
```

三组训练相互独立，都从原始 `pi05_base` 初始化，不串接上一组的 checkpoint：

| 顺序 | 模式 | 默认 batch | steps/epoch | 总 steps | 可训练范围 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `full` | 32 | 1060 | 15900 | PI0.5 全部参数 |
| 2 | `expert` | 32 | 1060 | 15900 | action expert 和动作/时间投影 |
| 3 | `lora` | 32 | 1060 | 15900 | rank-16 LoRA adapter |

分别启动：

```bash
bash my_devs/jz_robot_pin_timed/pi05/train_full_15_epochs.sh
bash my_devs/jz_robot_pin_timed/pi05/train_expert_15_epochs.sh
bash my_devs/jz_robot_pin_timed/pi05/train_lora_15_epochs.sh
```

总入口的 batch size 可以分别覆盖：

```bash
FULL_BATCH_SIZE=4 \
EXPERT_BATCH_SIZE=16 \
LORA_BATCH_SIZE=16 \
  bash my_devs/jz_robot_pin_timed/pi05/train_three_modes_15_epochs.sh
```

只检查三组参数换算、资产、数据解码和最终命令，而不启动大模型训练：

```bash
DRY_RUN=true bash my_devs/jz_robot_pin_timed/pi05/train_three_modes_15_epochs.sh
```

## 关键处理

1. 原始 100-episode 数据集保持不变。`prepare_training_view.py` 按 `CAMERA_MODE` 生成元数据派生视图；
   当前 B 视图链接原始 parquet 及 head/left/right 三路视频。
2. 100 个 episode 全部使用，不混入旧 200 条数据。episode 93 已知开头约 10 帧 action/state 对齐较差，本方案按当前要求仍明确包含该 episode。
3. 训练视图把任务统一标注为 `Put the bottle on the right into the basket on the left.`，同时把 episode 元数据中的任务记录一并更新。
4. 训练时通过只读 view 将 raw18 投影为 model16，保留兼容边界；左臂状态和动作维度仍存在，但数据中保持静止，主要学习目标是右臂。
5. 合并数据集自带的 quantile 是 episode quantile 的聚合值，不等于全量帧 quantile。本模块每次启动前从 parquet 重算全量 model16 统计到 dataset-specific 的 `runtime/*_model16_stats.json`，不回写数据集。
6. 默认使用 PI0.5 推荐的 `QUANTILES` normalization；如需复刻旧 SO101 脚本，可设置 `NORMALIZATION_MODE=MEAN_STD`。
7. tokenizer 通过本目录 `runtime/google/` 下的软链接离线路由，不在仓库其他位置创建文件。
8. checkpoint 的 pre/post processor 会保存 raw18→model16 和 model16→raw18 边界，后续部署仍需先离线验证，未经许可不得上真机。
9. 当前基模省略了与 `lm_head` tied 的 `embed_tokens.weight` 独立副本，加载器会打印一条 missing-key 警告；两者在 PaliGemma 配置中共享权重，smoke forward/backward 已验证可用。
