# JZ Robot Pin Timed PI0.5 RTC 推理

本目录提供 `jz_robot_pin_timed` 的分布式 PI0.5 PyTorch 推理运行时：训练/GPU 服务器运行 policy
server，上机侧运行 robot client。它复用 LeRobot 的 Real-Time Chunking（RTC），但**不包含 TensorRT
后端**。

当前只完成代码与非硬件配置入口。本目录的脚本不会启动、停止或修改 Orin state/camera/command
服务，不包含自动 reset、回零、编舞或急停替代逻辑。未经现场授权不要运行 armed 脚本。

## 目录与边界

- `run_policy_server.py`：加载 PI0.5 checkpoint，以同一个 PyTorch 服务执行 `single_step`/`rtc`。
- `run_robot_client.py`：读取 `jz_robot_pin_timed` 的 timed state/三路 ZMQ 相机并消费 action queue。
- `run_server.sh`、`run_client.sh`：统一 Conda、资产、缓存、日志和安全参数。
- `run_single_step_*.sh`：顺序请求普通 chunk，每次按 observation age 只执行第一条未过期动作；
  不使用 RTC leftover/queue。
- `run_rtc_*.sh`：异步 RTC queue 入口。
- `run_config_checks.sh`：只解析/校验配置，不连接机器人、相机或 server。

现场 robot/dataset 边界保持 raw18；checkpoint preprocessor 将 raw18 state 投影为 model16，checkpoint
postprocessor 在 unnormalization 后将 model16 action 展开回 raw18，并写入显式的两侧 gripper force。
RTC queue 会分别保存 model16 raw chunk 和 raw18 processed chunk。

## 固定环境与默认模型

所有入口默认使用 `lerobot_flex`：

```text
/home/cqy/miniconda3/envs/lerobot_flex/bin/python
```

默认 checkpoint 是本次 15 epoch 正式训练 run：

```text
my_devs/jz_robot_pin_timed/pi05/outputs/
  pi05_jz_robot_pin_timed_curated_42eps_20260713_e15_b8_20260714_202432/
  checkpoints/last/pretrained_model
```

训练完成前该目录可能不存在，正常启动 server 会明确拒绝。切换模型必须指向完整的
`pretrained_model/`：

```bash
POLICY_PATH=/absolute/path/checkpoints/last/pretrained_model \
  bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_server.sh
```

tokenizer 只读引用既有绝对资产：

```text
/data/cqy_workspace/flexible_lerobot/assets/modelscope/google/paligemma-3b-pt-224
```

server 通过 processor override 直接使用这个绝对路径，不创建 tokenizer 软链接；robot client 不需要
tokenizer。Hugging Face、Torch、临时文件、日志和运行 summary 分别写入本目录的 `runtime/`、
`logs/`、`outputs/`，不会向仓库其他位置写缓存。
shell 与 Python client 都会拒绝把 runtime/cache/log/summary 路径指向 `rtc_infer` 之外。

## 先做无硬件检查

下面的命令不会读取 checkpoint/tokenizer、加载模型、连接 server、state UDP、相机或 command UDP；
因此 15 epoch checkpoint 尚未保存完成时也可运行：

```bash
cd /data/cqy_workspace/jz_robot/flexible_lerobot
bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_config_checks.sh
```

任何入口均支持只打印最终命令：

```bash
PRINT_COMMAND_ONLY=true \
  bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_rtc_dry_run.sh
```

也可单独设置 `CONFIG_ONLY=true`，让对应 Python 入口完成参数/安全边界校验后退出。

## 启动 PyTorch policy server

默认仅监听 localhost：

```bash
bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_server.sh
```

不需要分别启动两套 server：该进程同时接受 `single_step` 与 `rtc` 请求，并在模型锁内切换 RTC。
启动时会校验 checkpoint 的 model16 state/action、wire18 postprocessor、三路相机键和精确分辨率、
JZ schema 语义、processor state 文件以及最终训练 step。`/health` 使用明确的
`model_action_dim=16`、`wire_action_dim=18` 握手字段。

训练结束且 GPU 可用后，可先只做一次模型/processor 加载检查；它不会监听端口或执行 inference：

```bash
CHECK_POLICY_LOAD=true \
  bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_server.sh
```

robot client 位于另一台机器时，只能在可信局域网中显式开放监听地址，并设置共享 token、将
server 端口限制在可信来源：

```bash
SERVER_AUTH_TOKEN='<现场生成的长随机值>' \
SERVER_HOST=0.0.0.0 SERVER_PORT=8088 \
  bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_server.sh
```

client 必须使用相同的 `SERVER_AUTH_TOKEN`。token 通过环境变量传递，不会出现在打印的命令行中。
协议用于内部 Python runtime，不应暴露到公网。默认 `POLICY_DEVICE=cuda`、RTC execution horizon
为 10、prefix schedule 为 `LINEAR`、max guidance weight 为 10。

server 启动后，可从上机侧只验证鉴权和模型契约，不连接机器人：

```bash
HEALTH_ONLY=true \
SERVER_AUTH_TOKEN='<与 server 相同的值>' \
SERVER_URL=http://<policy-server-ip>:8088 \
  bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_rtc_dry_run.sh
```

## 上机侧 dry-run

`dry_run` 会读取现场 state/相机并请求策略，但 robot action 固定走 `local + dry_run`，不会向 Orin
command UDP 发送。它仍要求现场人员先按已有流程启动正确的 state bridge 和三路相机服务。

普通单步闭环验收（`RUN_TIME_S` 内可重复请求；每个请求最多发送一条未过期动作）：

```bash
SERVER_AUTH_TOKEN='<与 server 相同的值>' \
SERVER_URL=http://<policy-server-ip>:8088 \
  bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_single_step_dry_run.sh
```

短时 RTC dry-run：

```bash
SERVER_AUTH_TOKEN='<与 server 相同的值>' \
SERVER_URL=http://<policy-server-ip>:8088 RUN_TIME_S=10 \
  bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_rtc_dry_run.sh
```

默认 state source/command target 为 `192.168.1.81`，state 端口 39010、command 端口 39020，
sensor/control 均为 20 FPS。常用覆盖包括：

```bash
ORIN_IP=192.168.1.81
STATE_BIND_IP=0.0.0.0
STATE_TIMEOUT_S=1.0
CONNECT_TIMEOUT_S=300
SENSOR_FPS=20
CONTROL_FPS=20
QUEUE_LOW_WATERMARK=30
MAX_QUEUE_SIZE=50
FIRST_CHUNK_TIMEOUT_S=60
RTC_EXECUTION_HORIZON=10
EMPTY_QUEUE_STRATEGY=stop
FULLY_STALE_CHUNK_LIMIT=3
```

默认 task 与训练数据唯一任务一致：`jz robot pin timed vr teleoperation`。RTC low watermark 默认 30，
用于在 20 Hz 控制频率下覆盖 PI0.5 推理延迟和 10-step execution horizon；不要直接照搬 SO101 的
4-step 水位。

按 `Ctrl+C` 停止 client/server；client 会通知 worker 退出并写 summary，server 会关闭监听，二者都不会
自动 reset 机器人。

## Armed 命令（仅现场授权后）

armed 会实际通过 UDP 发送机器人动作。shell 和 Python client 均要求三重确认；缺少任意一项都会
在连接机器人前拒绝：

```bash
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
JZ_POLICY_INFERENCE_ARMED=1 \
SERVER_AUTH_TOKEN='<与 server 相同的值>' \
SERVER_URL=http://<policy-server-ip>:8088 \
  bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_single_step_armed.sh
```

只有单步验收、物理急停、初始姿态和 action delta 均确认后，才考虑短时 RTC：

```bash
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
JZ_POLICY_INFERENCE_ARMED=1 \
SERVER_AUTH_TOKEN='<与 server 相同的值>' \
SERVER_URL=http://<policy-server-ip>:8088 \
RUN_TIME_S=10 \
  bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_rtc_armed.sh
```

这些脚本不负责启动 Orin command executor，也不会绕过 `JZRobotPin` 的 armed env、fresh state、
initial joint delta、per-step joint delta 和 gripper clamp 检查。

## 统一 client 入口

四个便捷脚本最终都调用：

```bash
MODE=single_step|rtc \
EXECUTION=dry_run|armed \
SERVER_AUTH_TOKEN='<远程 server 的共享 token>' \
SERVER_URL=http://<policy-server-ip>:8088 \
  bash my_devs/jz_robot_pin_timed/pi05/rtc_infer/run_client.sh
```

不要把 `EXECUTION=armed` 当作配置开关试跑；查看 armed 命令请结合三重确认变量使用
`CONFIG_ONLY=true` 或 `PRINT_COMMAND_ONLY=true`，且确认当前 shell 不会继续启动真实 client。
