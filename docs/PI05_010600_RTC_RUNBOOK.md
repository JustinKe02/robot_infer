# PI0.5 010600 RTC 上机运行命令

本文档记录当前已经实际打通的 PI0.5 RTC 上机链路。服务端和机器人客户端运行在同一台 X86
机器上，策略服务使用 `127.0.0.1:8088`，机器人 state、相机和 command 服务位于
`192.168.1.81`。

## 当前配置

```text
checkpoint:      tk_infer/pi05/checkpoints/010600/pretrained_model
checkpoint step: 10600/15900 (epoch 10/15, intermediate)
camera profile:  head_right
head camera:     192.168.1.81:5555
right camera:    192.168.1.81:5557
state:           0.0.0.0:39010
command:         192.168.1.81:39020
policy server:   127.0.0.1:8088
sensor FPS:      20
control FPS:     20
mode:            rtc
action boundary: complete dual-arm raw18
```

当前命令会持续发送完整双臂和双夹爪动作，并显式关闭 initial/per-step joint-delta 检查。
运行时必须有人在机器人旁边，确认工作空间无人并随时准备按急停。

## 1. 启动策略服务

在终端 A 执行：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

CONDA_ROOT=/home/luzhuang/miniconda3 \
TK_PI05_010600_INTERMEDIATE_CONFIRMED=1 \
SERVER_HOST=127.0.0.1 \
SERVER_PORT=8088 \
bash tk_infer/pi05/run_current_server.sh
```

等待服务端完成模型加载并显示：

```text
[tk_infer/pi05/server] Listening on http://127.0.0.1:8088; modes=single_step,rtc
```

保持终端 A 运行。

## 2. 持续运行 RTC

在终端 B 执行：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

CONDA_ROOT=/home/luzhuang/miniconda3 \
TK_PI05_010600_INTERMEDIATE_CONFIRMED=1 \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
JZ_POLICY_INFERENCE_ARMED=1 \
JZ_PI05_SINGLE_STEP_ARMED_PASSED=1 \
JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=1 \
I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=1 \
JZ_PI05_EXPECTED_CHECKPOINT_STEP=10600 \
JZ_PI05_EXPECTED_CONFIGURED_STEPS=15900 \
JZ_PI05_EXPECTED_CHECKPOINT_FINGERPRINT=4698315f6936f9e9ef19017cfdb873588eba771fdb23595879ce2a7703b4c8dd \
JZ_PI05_EXPECTED_CHECKPOINT_PATH=/home/luzhuang/cqy/aaa/flexible_lerobot/tk_infer/pi05/checkpoints/010600/pretrained_model \
JZ_PI05_EXPECTED_COMPLETE_STEP=false \
SERVER_URL=http://127.0.0.1:8088 \
MODE=rtc \
EXECUTION=armed \
CAMERA_PROFILE=head_right \
ORIN_IP=192.168.1.81 \
STATE_BIND_IP=0.0.0.0 \
STATE_PORT=39010 \
COMMAND_PORT=39020 \
CONNECT_TIMEOUT_S=15 \
STATE_TIMEOUT_S=1 \
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
bash tk_infer/pi05/run_client.sh
```

`RUN_TIME_S=0` 和 `MAX_SENT_ACTIONS=0` 表示持续运行，直到用户按 `Ctrl+C` 或运行时发生
fail-closed 错误。物理急停会停止机器人运动，但不保证自动终止 X86 客户端；按下急停后仍需在
终端 B 按 `Ctrl+C` 停止动作请求。

正常启动时应显示：

```text
mode=rtc
execution=armed transport=udp
camera_profile=head_right
sensor_fps=20 control_fps=20
joint_delta_checks=disabled initial=0.0 step=0.0
server health PASS
robot connected type=jz_robot_pin_timed
```

运行中每约 2 秒输出一次指标：

```text
[jz/pi05/rtc-client] sensor=... actor=... sent=... requests=... queue=... \
request_ms=... p95_ms=... server_ms=... drop=... pred_delay=...
```

已经验证过的正常稳态范围为：

```text
sensor/actor:     approximately 20 Hz
inference:        approximately 1 request/s
request latency:  approximately 177-227 ms after warmup
queue depth:      approximately 27-46
drop steps:       4-5
predicted delay:  5 after warmup
```

## 3. 停止运行

先在终端 B 按：

```text
Ctrl+C
```

等待客户端显示：

```text
robot disconnected
```

再到终端 A 按 `Ctrl+C` 关闭策略服务。

最后可以检查是否存在残留进程或端口占用：

```bash
ss -ltnp '( sport = :8088 )'
ss -lunp | rg ':39010' || true
pgrep -af 'run_policy_server.py|run_robot_client.py' || true
```

## 运行边界

- 当前 checkpoint 只接受 `camera_head` 和 `camera_right`，不会连接 `camera_left:5556`。
- RTC 使用 model16 leftover 做跨 chunk guidance，并以 raw18 向机器人发送完整双臂动作。
- `EMPTY_QUEUE_STRATEGY=stop`，动作队列耗尽时客户端会 fail closed。
- 当前命令关闭了 initial/per-step joint-delta 检查，不包含动作跳变限制或额外平滑。
- UDP command 没有应用层 ACK；客户端的 `sent` 表示动作已交给 UDP sender，不等同于机器人已准确执行。
- 动作方向错误、持续抖动、夹爪异常或未按预期停止时，应立即按急停并终止客户端。
