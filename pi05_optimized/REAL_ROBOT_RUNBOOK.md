# PI0.5 步骤 015900 真机运行手册

本文记录已经在真机上验证成功的链路：

```text
完整 step-015900 checkpoint
  -> torch_optimized 策略服务（RTC 保留 autograd）
  -> 旧版已审计 JZ Robot 客户端
  -> head/right 真实相机 + raw18 真实 state
  -> UDP 192.168.1.81:39020
```

优化目录目前没有独立的真机 adapter，因此策略服务来自 `pi05_optimized`，机器人连接和动作发送边界
来自 `tk_infer/pi05`。这正是 2026-07-30 已完成真机验证的组合。

## 1. 固定配置

```text
仓库：             /home/luzhuang/cqy/aaa/flexible_lerobot
Conda 环境：       lerobot_flex
策略服务：         http://127.0.0.1:18088
Orin：             192.168.1.81
head 相机：        tcp://192.168.1.81:5555
right 相机：       tcp://192.168.1.81:5557
state：            udp://0.0.0.0:39010
command：          udp://192.168.1.81:39020
checkpoint：       tk_infer/pi05/checkpoints/015900/pretrained_model
step：             15900/15900
checkpoint 指纹：  9d6d37f6111a034209c9bdc2899423a3258cc35070cb8294194c9c594197b58a
相机配置：         head_right
动作边界：         model16 -> raw18
控制频率：         20 Hz
```

## 2. 现场前置条件

1. 机器人、Orin 相机 publisher、state publisher 和 command executor 已启动。
2. 操作员在机器人旁，急停按钮可立即触发。
3. 工作区无人、无障碍物。
4. 没有其他 recorder、teleop、replay 或 robot client 占用 `39010`。
5. 没有其他 PI0.5 服务占用 GPU 或 `18088`。

启动前执行：

```bash
ss -ltnup '( sport = :39010 or sport = :39020 )'
pgrep -af 'run_robot_client|policy_service.py client|teleop|replay|record'
```

只要发现其他机器人客户端或 `39010` 已被占用，就不要启动本手册中的客户端。UDP state 端口不能由
两个控制客户端共享；同时运行两个 armed 客户端还会向同一 command executor 发送相互冲突的动作。

物理急停会停止机器人运动，但不保证自动结束 X86 客户端。按下急停后，还要在客户端终端按
`Ctrl+C`，停止继续产生动作请求。

## 3. 终端 A：启动优化策略服务

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
source /home/luzhuang/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex

PI05_OPT_BACKEND=torch_optimized \
PI05_OPT_TORCH_INFERENCE_MODE=false \
PI05_OPT_TRAJECTORY_PROCESSOR=pass_through \
bash tk_infer/pi05_optimized/profiles/step_015900/run_policy_server.sh
```

RTC guidance 需要 `torch.autograd.grad`，所以不能把 `PI05_OPT_TORCH_INFERENCE_MODE` 设置为 `true`。
等待输出同时包含：

```text
checkpoint_step=15900
complete_step=true
supported_modes=[single_step, rtc]
Listening on http://127.0.0.1:18088
```

## 4. 终端 B：健康检查

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
source /home/luzhuang/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex

bash tk_infer/pi05_optimized/profiles/step_015900/run_health_check.sh
```

确认健康信息中的 checkpoint、指纹、`head_right`、`torch_optimized` 和 `rtc` 均正确。

## 5. 终端 B：设置非危险公共参数

下列变量只绑定 checkpoint 和通信地址，不会单独触发动作：

```bash
export SERVER_URL=http://127.0.0.1:18088
export CAMERA_PROFILE=head_right
export ORIN_IP=192.168.1.81
export STATE_BIND_IP=0.0.0.0
export STATE_PORT=39010
export COMMAND_PORT=39020
export JZ_PI05_EXPECTED_CHECKPOINT_STEP=15900
export JZ_PI05_EXPECTED_CONFIGURED_STEPS=15900
export JZ_PI05_EXPECTED_CHECKPOINT_FINGERPRINT=9d6d37f6111a034209c9bdc2899423a3258cc35070cb8294194c9c594197b58a
export JZ_PI05_EXPECTED_CHECKPOINT_PATH=/home/luzhuang/cqy/aaa/flexible_lerobot/tk_infer/pi05/checkpoints/015900/pretrained_model
export JZ_PI05_EXPECTED_COMPLETE_STEP=true
```

## 6. 单动作真机验证

下面是已经验证成功的单动作命令。它以 5 Hz 获取一次观测并推理，最多向机器人发送一个 raw18
动作，然后立即停止。该命令按现场授权关闭 initial/per-step joint-delta 检查；raw18 finite、force=80、
gripper 范围、state freshness、sender identity 和 checkpoint 校验仍然保留。

```bash
MODE=single_step \
EXECUTION=armed \
SENSOR_FPS=5 \
CONTROL_FPS=5 \
RUN_TIME_S=1 \
MAX_SENT_ACTIONS=1 \
EMPTY_QUEUE_STRATEGY=stop \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
JZ_POLICY_INFERENCE_ARMED=1 \
JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=1 \
I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=1 \
bash tk_infer/pi05/run_client.sh
```

成功日志必须包含：

```text
execution=armed transport=udp
joint_delta_checks=disabled initial=0.0 step=0.0
PASS sent=1 requests=1
robot disconnected
```

## 7. 20 Hz、10 秒 RTC 真机验证

只有上一节确实发送一个动作并正常结束后，才运行：

```bash
MODE=rtc \
EXECUTION=armed \
SENSOR_FPS=20 \
CONTROL_FPS=20 \
RUN_TIME_S=10 \
QUEUE_LOW_WATERMARK=30 \
MAX_QUEUE_SIZE=50 \
FIRST_CHUNK_TIMEOUT_S=60 \
RTC_EXECUTION_HORIZON=10 \
REQUEST_TIMEOUT_S=120 \
EMPTY_QUEUE_STRATEGY=stop \
FULLY_STALE_CHUNK_LIMIT=3 \
METRICS_LOG_INTERVAL_S=2 \
MAX_SENT_ACTIONS=0 \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
JZ_POLICY_INFERENCE_ARMED=1 \
JZ_PI05_SINGLE_STEP_ARMED_PASSED=1 \
JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=1 \
I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=1 \
bash tk_infer/pi05/run_client.sh
```

2026-07-30 的实测结果为：

```text
运行时间：        10 秒
sensor ticks：    201
actor ticks：     200
真实 UDP 动作：   195
模型请求：        11
请求 p95：        226.24 ms
drop/pred delay： 4-5 steps
最终 queue：      41
empty/stale/error：0
```

报告位于：

```text
tk_infer/pi05/outputs/client/rtc_armed_20260730_142509/client_summary.json
```

## 8. 持续 20 Hz RTC

先完成并检查 10 秒运行。确认没有其他客户端占用 `39010` 后，持续运行使用下面的完整命令：

```bash
MODE=rtc \
EXECUTION=armed \
SENSOR_FPS=20 \
CONTROL_FPS=20 \
RUN_TIME_S=0 \
QUEUE_LOW_WATERMARK=30 \
MAX_QUEUE_SIZE=50 \
FIRST_CHUNK_TIMEOUT_S=60 \
RTC_EXECUTION_HORIZON=10 \
REQUEST_TIMEOUT_S=120 \
EMPTY_QUEUE_STRATEGY=stop \
FULLY_STALE_CHUNK_LIMIT=3 \
METRICS_LOG_INTERVAL_S=2 \
MAX_SENT_ACTIONS=0 \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
JZ_POLICY_INFERENCE_ARMED=1 \
JZ_PI05_SINGLE_STEP_ARMED_PASSED=1 \
JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=1 \
I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=1 \
bash tk_infer/pi05/run_client.sh
```

`RUN_TIME_S=0` 和 `MAX_SENT_ACTIONS=0` 表示持续运行，直到按 `Ctrl+C`、客户端触发 fail-closed 错误，
或人工急停并随后终止客户端。

## 9. 停止和结束审计

1. 在机器人客户端终端按 `Ctrl+C`。
2. 确认日志出现 `robot disconnected`。
3. 检查动作端口和残留客户端：

```bash
ss -ltnup '( sport = :39010 or sport = :39020 )'
pgrep -af 'run_robot_client|run_client.sh'
```

4. 不再需要模型时，在策略服务终端按 `Ctrl+C`。
5. 每次运行检查 `client_summary.json` 中的：

```text
stop_reason
sent_actions
inference_requests
metrics.p95_request_s
queue.empty_events
queue.stop_events
queue.dropped_all_chunks
```

## 10. 已知边界

- “20 Hz 控制”指 Actor 每 50 ms 消费一个动作，不要求模型每 50 ms 完成一次推理。
- 当前稳态 RTC 请求约 177-226 ms，由 50-step chunk、队列和 4-5 step 延迟补偿维持 20 Hz 动作输出。
- `p95 <= 50 ms` 是逐周期推理的性能优化目标，不是 chunked RTC 功能运行的必要条件。
- `torch.inference_mode=true` 虽然单步更快，但与基于 autograd 的 RTC guidance 不兼容。
- 当前 step-015900 checkpoint 没有 training-time RTC conditioning 元数据；这里使用 inference-time RTC。
- 本手册的真机命令关闭了 joint-delta 检查，复现的是 2026-07-30 已验证现场模式。
