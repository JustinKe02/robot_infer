# 当前 PI0.5 052050 普通 PyTorch RTC 推理说明

本文档记录当前已经实际运行的 PI0.5 推理链路，包括权重位置、权重加载方式、启动命令、
HTTP 接口、RTC action-prefix 条件路径，以及从机器人观测到 raw18 动作下发的完整调用过程。

本文只描述以下实现：

- 普通 `tk_infer/pi05` PyTorch backend；
- checkpoint `052050/pretrained_model`；
- `camera_head + camera_right` 两路相机；
- `sensor_fps=20`、`control_fps=20`；
- 连续 `mode=rtc`；
- X86 上的策略服务与机器人客户端使用本机 `127.0.0.1:8088` 通信；
- 机器人状态、相机和动作命令通过 `192.168.1.81` 侧的服务通信。

这里没有使用 SmolVLA、pass-through、TensorRT、Triton、`realtime_vla`，也没有使用
`tk_infer/pi05_optimized`。服务端启动时会明确输出：

```text
[tk_infer/pi05] backend=pytorch (TensorRT is not used)
```

## 1. 当前已确认的配置

| 项目 | 当前值 |
|---|---|
| checkpoint | `tk_infer/pi05/checkpoints/052050/pretrained_model` |
| checkpoint step | `52050/52050`，完整最终 step |
| checkpoint fingerprint | `10bc4821f71f66c4b698fc51072e60ddfe251b52dc6f8cbfdeede630134aeaec` |
| `model.safetensors` | `14,467,170,664` bytes，约 `13.47 GiB` |
| `model.safetensors` SHA-256 | `b2a9b589189ecc818dec5a656b854996b7a733113d58af3527249e2cdc3577bf` |
| 模型类型 | PI0.5，PaliGemma `gemma_2b` + action expert `gemma_300m` |
| 微调方式 | 全量微调，不是 LoRA |
| 数据 | 380 episodes，15 epochs，111,018 帧 |
| checkpoint 导出 dtype | FP32 safetensors；配置中的模型计算 dtype 为 BF16 |
| 相机 | `camera_head`、`camera_right` |
| 相机 profile | `head_right` |
| 观测历史 | 1 帧 |
| 模型状态维度 | 16，内部补齐至 32 |
| 模型动作维度 | 16，内部补齐至 32 |
| 机器人动作边界 | raw18 |
| chunk size | 50 步，即 20 Hz 下 2.5 秒 |
| flow-matching steps | 10 |
| RTC execution horizon | 10 步，即 0.5 秒 |
| RTC guidance | `LINEAR`，最大 weight `10.0` |
| sensor/control FPS | `20/20` |
| 队列低水位/最大长度 | `30/50` |
| 策略服务 | `http://127.0.0.1:8088` |
| 机器人 IP | `192.168.1.81` |
| state | X86 监听 `0.0.0.0:39010`，只接受机器人 IP |
| command | UDP `192.168.1.81:39020` |
| head camera | ZMQ `192.168.1.81:5555` |
| right camera | ZMQ `192.168.1.81:5557` |
| 任务指令 | `Put the bottle on the right into the basket on the left.` |

最近一次 `052050` 连续 RTC 日志确认了以下健康信息：

```text
policy_type=pi05
checkpoint_step=52050
configured_steps=52050
complete_step=true
camera_profile=head_right
model_state_dim=16
model_action_dim=16
wire_action_dim=18
supported_modes=single_step,rtc
rtc_execution_horizon=10
rtc_max_guidance_weight=10.0
```

## 2. 当前权重目录

完整路径：

```text
/home/luzhuang/cqy/aaa/flexible_lerobot/tk_infer/pi05/checkpoints/052050/pretrained_model
```

目录中当前包含：

| 文件 | 作用 |
|---|---|
| `config.json` | PI0.5 模型结构、输入输出特征、chunk、flow-matching 和 RTC 配置 |
| `model.safetensors` | 完整全量微调模型权重 |
| `train_config.json` | 数据集、训练 step、优化器和训练策略配置 |
| `tk_rtc_training_config.json` | 训练期 action-prefix RTC 采样和条件契约 |
| `policy_preprocessor.json` | 图像、文本、raw18 状态到 model16、Quantile 归一化处理链 |
| `policy_preprocessor_step_3_normalizer_processor.safetensors` | STATE/ACTION 归一化统计 |
| `policy_postprocessor.json` | model16 动作反归一化并恢复 raw18 的处理链 |
| `policy_postprocessor_step_0_unnormalizer_processor.safetensors` | ACTION Quantile 反归一化统计 |

checkpoint 不包含 PaliGemma tokenizer。当前离线 tokenizer 位于：

```text
/home/luzhuang/cqy/aaa/flexible_lerobot/assets/modelscope/google/paligemma-3b-pt-224
```

其中必须至少存在：

```text
tokenizer.json
tokenizer_config.json
```

注意：`tk_infer/pi05/checkpoints/current` 目前仍指向 `010600`。因此加载 `052050` 时不要执行
`run_current_server.sh`，而要显式设置 `POLICY_PATH` 后执行通用的 `run_server.sh`。

## 3. 快速使用命令

所有 Python 代码都通过 `lerobot_flex` Conda 环境运行。以下命令不会修改 checkpoint。

### 3.1 检查权重文件

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

POLICY_PATH="$PWD/tk_infer/pi05/checkpoints/052050/pretrained_model"
test -f "$POLICY_PATH/model.safetensors"
test -f "$POLICY_PATH/config.json"
test -f "$POLICY_PATH/train_config.json"
test -f "$POLICY_PATH/policy_preprocessor.json"
test -f "$POLICY_PATH/policy_postprocessor.json"
du -h "$POLICY_PATH/model.safetensors"
sha256sum "$POLICY_PATH/model.safetensors"
```

预期 SHA-256：

```text
b2a9b589189ecc818dec5a656b854996b7a733113d58af3527249e2cdc3577bf
```

### 3.2 只加载模型后退出

该命令会读取完整权重、在 CUDA 上构造模型和 processor，但不会监听端口、不会连接机器人、
不会执行一次模型推理：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

CONDA_ROOT=/home/luzhuang/miniconda3 \
CONDA_ENV=lerobot_flex \
POLICY_PATH="$PWD/tk_infer/pi05/checkpoints/052050/pretrained_model" \
TOKENIZER_PATH="$PWD/assets/modelscope/google/paligemma-3b-pt-224" \
POLICY_DEVICE=cuda \
RTC_EXECUTION_HORIZON=10 \
RTC_MAX_GUIDANCE_WEIGHT=10.0 \
RTC_PREFIX_ATTENTION_SCHEDULE=LINEAR \
REQUIRE_COMPLETE_STEP=true \
CHECK_POLICY_LOAD=true \
bash tk_infer/pi05/run_server.sh
```

成功时应该看到：

```text
checkpoint_step: 52050
configured_steps: 52050
complete_step: true
camera_profile: head_right
CHECK_POLICY_LOAD passed
```

### 3.3 终端 A：启动普通 PyTorch 策略服务

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

CONDA_ROOT=/home/luzhuang/miniconda3 \
CONDA_ENV=lerobot_flex \
POLICY_PATH="$PWD/tk_infer/pi05/checkpoints/052050/pretrained_model" \
TOKENIZER_PATH="$PWD/assets/modelscope/google/paligemma-3b-pt-224" \
SERVER_HOST=127.0.0.1 \
SERVER_PORT=8088 \
POLICY_DEVICE=cuda \
RTC_EXECUTION_HORIZON=10 \
RTC_MAX_GUIDANCE_WEIGHT=10.0 \
RTC_PREFIX_ATTENTION_SCHEDULE=LINEAR \
REQUIRE_COMPLETE_STEP=true \
bash tk_infer/pi05/run_server.sh
```

模型加载完成后等待：

```text
[tk_infer/pi05/server] Listening on http://127.0.0.1:8088; modes=single_step,rtc
```

保持终端 A 运行。服务端启动本身不会做一次 warm inference；第一次客户端请求可能明显慢于稳态。

### 3.4 终端 B：只检查服务健康状态

该命令访问 `GET /health`，不连接机器人：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

CONDA_ROOT=/home/luzhuang/miniconda3 \
CONDA_ENV=lerobot_flex \
SERVER_URL=http://127.0.0.1:8088 \
MODE=rtc \
EXECUTION=dry_run \
CAMERA_PROFILE=head_right \
HEALTH_ONLY=true \
REQUEST_TIMEOUT_S=120 \
bash tk_infer/pi05/run_client.sh
```

成功时会显示：

```text
server health PASS checkpoint=.../checkpoints/052050/pretrained_model
HEALTH_ONLY PASS; no robot connection was made
```

### 3.5 终端 B：RTC dry-run 预热

该命令会连接 state 和两路相机并运行 RTC，但 `EXECUTION=dry_run` 使用本地动作 transport，
不会向 `192.168.1.81:39020` 发送 UDP 动作。它会走真实的 RTC 请求、action-prefix、后处理和
动作队列路径，适合在 armed 前完成模型预热：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

CONDA_ROOT=/home/luzhuang/miniconda3 \
CONDA_ENV=lerobot_flex \
SERVER_URL=http://127.0.0.1:8088 \
MODE=rtc \
EXECUTION=dry_run \
CAMERA_PROFILE=head_right \
ORIN_IP=192.168.1.81 \
STATE_BIND_IP=0.0.0.0 \
STATE_PORT=39010 \
COMMAND_PORT=39020 \
CONNECT_TIMEOUT_S=15 \
STATE_TIMEOUT_S=1 \
TASK='Put the bottle on the right into the basket on the left.' \
SENSOR_FPS=20 \
CONTROL_FPS=20 \
RUN_TIME_S=8 \
QUEUE_LOW_WATERMARK=30 \
MAX_QUEUE_SIZE=50 \
FIRST_CHUNK_TIMEOUT_S=60 \
RTC_EXECUTION_HORIZON=10 \
REQUEST_TIMEOUT_S=120 \
EMPTY_QUEUE_STRATEGY=stop \
FULLY_STALE_CHUNK_LIMIT=3 \
METRICS_LOG_INTERVAL_S=2 \
bash tk_infer/pi05/run_client.sh
```

正常情况下，第一次请求可能处于冷启动；后续日志应逐步回到约 `180-220 ms`，并出现非空的
`prev_chunk_left_over` 条件路径。日志不会直接打印 leftover 张量，但 `requests` 持续增加、
`mode=rtc`、`pred_delay` 和 `drop` 有效变化，说明 RTC producer 正在工作。

### 3.6 终端 B：连续 RTC 上机

下面的命令复现当前实际使用的两路相机、20 Hz、普通 PyTorch RTC 上机配置。它显式关闭
initial/per-step joint-delta 检查；模型输出经过 raw18 后处理后会直接通过 UDP 发送。

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

CONDA_ROOT=/home/luzhuang/miniconda3 \
CONDA_ENV=lerobot_flex \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
JZ_POLICY_INFERENCE_ARMED=1 \
JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=1 \
I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=1 \
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
TASK='Put the bottle on the right into the basket on the left.' \
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

`RUN_TIME_S=0` 和 `MAX_SENT_ACTIONS=0` 表示持续运行，直到 `Ctrl+C` 或运行时错误触发停止。

正常启动信息应包含：

```text
backend=pytorch-client (TensorRT is not used)
mode=rtc
execution=armed transport=udp
sensor_fps=20 control_fps=20
camera_profile=head_right
schema=raw18->model16->raw18 force=80
joint_delta_checks=disabled
server health PASS checkpoint=.../checkpoints/052050/pretrained_model
robot connected type=jz_robot_pin_timed
```

### 3.7 可选：让客户端精确锁定 052050

客户端即使不设置以下变量，也会通过 `/health` 检查 PI0.5 类型、协议版本、相机、model16/raw18
维度、JZ schema，并要求服务端加载的是完整最终 step。

如果还需要确保服务端加载的恰好是当前这份 `052050` 文件，可以在客户端命令中追加以下五项：

```bash
JZ_PI05_EXPECTED_CHECKPOINT_STEP=52050 \
JZ_PI05_EXPECTED_CONFIGURED_STEPS=52050 \
JZ_PI05_EXPECTED_CHECKPOINT_FINGERPRINT=10bc4821f71f66c4b698fc51072e60ddfe251b52dc6f8cbfdeede630134aeaec \
JZ_PI05_EXPECTED_CHECKPOINT_PATH=/home/luzhuang/cqy/aaa/flexible_lerobot/tk_infer/pi05/checkpoints/052050/pretrained_model \
JZ_PI05_EXPECTED_COMPLETE_STEP=true \
```

这五项只控制客户端和 `/health` 返回值的身份校验，不参与模型加载、图像预处理、RTC
action-prefix 或动作生成。若不需要精确锁定，应当五项全部不设置；不能只设置其中一部分。

### 3.8 停止顺序

1. 先在终端 B 按 `Ctrl+C`，停止客户端动作循环。
2. 等待客户端打印 `robot disconnected`。
3. 再在终端 A 按 `Ctrl+C`，关闭策略服务。

检查残留进程和端口：

```bash
ss -ltnp '( sport = :8088 )'
ss -lunp | rg ':39010' || true
pgrep -af 'run_policy_server.py|run_robot_client.py' || true
```

## 4. 权重是怎么加载的

### 4.1 Shell 入口

服务端命令进入 [`pi05/run_server.sh`](pi05/run_server.sh)。脚本完成以下工作：

1. 从 `POLICY_PATH` 取得 `pretrained_model` 目录；
2. 从 `TOKENIZER_PATH` 取得离线 PaliGemma tokenizer；
3. 固定使用 `lerobot_flex` 环境中的 Python；
4. 设置 Hugging Face/Transformers 离线模式和本地 cache；
5. 调用 [`pi05/run_policy_server.py`](pi05/run_policy_server.py)；
6. 打印 `backend=pytorch (TensorRT is not used)`。

### 4.2 Checkpoint 校验

`run_policy_server.py` 先调用
[`inspect_checkpoint()`](pi05/runtime/checkpoint.py) 校验：

- checkpoint 路径必须是目录；
- `config.json` 中 `type` 必须是 `pi05`；
- 必须存在 `model.safetensors`；
- 必须存在 preprocessor/postprocessor JSON 及其状态文件；
- 必须存在 `train_config.json`；
- 输入相机必须能解析成 `head_right`；
- JZ 边界必须是已审计的 `raw18 -> model16 -> raw18` schema；
- 路径中的 step `052050` 必须等于 `train_config.steps=52050`；
- tokenizer 必须是绝对目录，并包含 `tokenizer.json` 和 `tokenizer_config.json`。

校验同时计算两种不同的指纹：

- `model.safetensors` SHA-256：完整权重文件的普通文件哈希；
- checkpoint fingerprint：schema、配置文件、权重大小及权重开头 1 MiB 的运行时身份指纹。

两者用途不同，不应该互相替换。

### 4.3 构造 PI0.5 Policy

[`PolicyService.from_config()`](pi05/runtime/policy_service.py) 调用
[`load_policy_bundle()`](pi05/runtime/checkpoint.py)，加载过程为：

1. `PreTrainedConfig.from_pretrained(..., local_files_only=True)` 读取 `config.json`；
2. `get_policy_class("pi05")` 取得 PI0.5 策略类；
3. `policy_class.from_pretrained(..., strict=False)` 读取 `model.safetensors`；
4. `policy.to("cuda")`；
5. `policy.eval()`；
6. 从 `policy_preprocessor.json` 构造输入 processor；
7. 将本地 `TOKENIZER_PATH` 覆盖到 tokenizer processor；
8. 从 `policy_postprocessor.json` 构造动作后处理器；
9. postprocessor 保持在 CPU，将机器人可执行 raw18 数组送过 HTTP；
10. `install_policy_rtc()` 把运行时 RTC 配置安装到 policy。

因为 `052050` 中没有 `adapter_config.json`，加载器走完整模型分支，不会加载 PEFT base model 或
`adapter_model.safetensors`。

服务日志中可能出现 `paligemma.lm_head.weight` missing key 提示；当前权重已经成功加载并通过
完整的 `/health`、单步和 RTC 实际运行验证，该提示没有阻止 PI0.5 连续动作输出。

## 5. 完整推理架构

```text
JZ Robot / Orin (192.168.1.81)
  camera_head : ZMQ 5555 -----------+
  camera_right: ZMQ 5557 -----------+--> Sensor thread (20 Hz)
  raw18 state : UDP -> X86:39010 ----+        |
                                               v
                                      latest FrameBuffer
                                               |
                                               v
                                      Producer thread
                                      queue <= 30 时请求
                                               |
                        HTTP POST /infer, protocol v3, mode=rtc
                        observation + task + predicted_delay
                        + previous model16 leftover
                                               |
                                               v
                                      PI0.5 PolicyService (CUDA)
                                      preprocess -> predict chunk
                                      -> postprocess
                                               |
                        model16 (RTC leftover) + raw18 (robot action)
                                               |
                                               v
                                      ActionChunkQueue (最多 50)
                                      丢弃过时 prefix
                                               |
                                               v
                                      Actor thread (20 Hz)
                                               |
  command UDP 39020 <---------------- raw18 complete dual-arm action
```

运行时由三个并行线程组成：

| 线程 | 入口 | 频率/触发 | 职责 |
|---|---|---|---|
| `JZPI05Sensor` | `run_sensor_loop()` | 20 Hz | 读取 raw18 state 和两路相机，保留最新观测 |
| `JZPI05Producer` | `run_producer_loop()` | queue depth <= 30 | 构造 RTC 请求、调用模型、合并新 chunk |
| `JZPI05Actor` | `run_actor_loop()` | 20 Hz | 从队列弹出一个 raw18 动作并调用机器人接口 |

这三个线程由 [`run_rtc_runtime()`](pi05/runtime/client_runtime.py) 创建。

## 6. 一次完整模型推理调用了哪些接口

### 6.1 读取机器人观测

客户端入口是 [`run_robot_client.py`](pi05/run_robot_client.py)。连接成功后构造机器人驱动和数据
特征，再调用 `run_client_runtime()`。`mode=rtc` 会进入 `run_rtc_runtime()`。

Sensor 线程的调用链：

```text
run_sensor_loop()
  -> SerializedRobotIO.get_observation()
  -> JZRobotPinTimed.get_observation()
  -> JZRobotUDP._get_proprioceptive_observation()
  -> camera_head.read()
  -> camera_right.read()
  -> build_live_observation_frame()
  -> robot_observation_processor(default live processor)
  -> build_dataset_frame()
  -> FrameBuffer.update()
```

原始 state 包含 18 维：14 个双臂关节、左右夹爪 width、左右夹爪 force。processor 删除两个
force 维度并映射夹爪语义的步骤不在客户端完成：客户端构造的 observation frame 仍保留 raw18
state，服务端 checkpoint preprocessor 再把它投影成训练使用的 model16 state。图像保持模型声明的
key，进入服务端模型 processor 后缩放到 `224x224`。

### 6.2 构造 RTC 请求

当动作队列深度不大于 `QUEUE_LOW_WATERMARK=30` 时，Producer 获取最新观测并调用
[`make_request()`](pi05/runtime/client_runtime.py) 构造 `InferenceRequest`：

| 字段 | 当前内容 |
|---|---|
| `request_id` | 单调增加的请求编号 |
| `mode` | `rtc` |
| `observation_frame` | raw18 state、`camera_head`、`camera_right` |
| `task` | 当前英文任务指令 |
| `robot_type` | `jz_robot_pin_timed` |
| `obs_sequence_id` | 最新观测序列号 |
| `predicted_delay_steps` | 最近 100 次请求耗时 P95 除以 50 ms 后向上取整 |
| `prev_chunk_left_over` | 当前队列尚未执行的归一化 model16 动作，形状 `(T,16)` |
| `execution_horizon` | 10 |

第一份 chunk 产生之前，队列中没有 leftover，因此第一次 RTC 请求的
`prev_chunk_left_over=None`。从第二次请求开始，客户端通过
[`ActionChunkQueue.get_raw_leftover()`](pi05/runtime/action_queue.py) 取出上一 chunk 中尚未执行的
model16 动作。这就是推理期调用训练时 action-prefix 条件路径的关键输入。

### 6.3 HTTP 接口

[`RemotePolicyClient`](pi05/runtime/remote_client.py) 暴露两个实际网络调用：

| HTTP 接口 | 方法 | 编码 | 用途 |
|---|---|---|---|
| `/health` | `GET` | JSON | 检查模型、step、相机、维度、schema、RTC 和协议版本 |
| `/infer` | `POST` | `application/x-python-pickle` | 发送 `InferenceRequest`，接收 `InferenceResponse` |

wire protocol 位于 [`runtime/protocol.py`](pi05/runtime/protocol.py)：

```text
PROTOCOL_VERSION=3
MODEL_ACTION_DIM=16
WIRE_ACTION_DIM=18
MAX_ACTION_CHUNK_STEPS=50
```

服务端 [`PolicyRequestHandler`](pi05/runtime/http_server.py) 校验请求后调用：

```text
PolicyService.infer(request)
```

同一 `PolicyService` 使用锁串行执行 CUDA 模型推理，避免多个 HTTP 请求同时操作同一 policy。

### 6.4 模型预处理

服务端 [`run_policy_chunk_inference()`](pi05/runtime/policy_service.py) 首先执行：

```text
prepare_observation_for_inference()
  -> 添加 batch 维度
  -> 添加 task prompt
  -> 添加 robot_type
  -> preprocessor(batch)
```

preprocessor 完成：

- raw18 state 到 model16 的边界处理；
- STATE Quantile 归一化；
- 两路图像格式转换和 `224x224` resize；
- VISUAL Identity 归一化策略；
- PaliGemma tokenizer 文本编码；
- 将模型输入移动到 CUDA。

### 6.5 显式进入 RTC action-prefix 条件路径

当 `request.mode == "rtc"` 时，服务端构造：

```python
predict_kwargs = {
    "inference_delay": predicted_delay_steps,
    "prev_chunk_left_over": left_over_tensor,
    "execution_horizon": 10,
}
```

随后在 RTC enabled 上下文中调用：

```python
policy.predict_action_chunk(preprocessed_batch, **predict_kwargs)
```

因此当前推理确实调用了训练时 action-prefix 对应的条件路径，而不是只加载一份“训练过 RTC”
的权重后继续使用普通 `select_action()`。

训练时的 RTC 契约是：

```text
prefix steps: uniform 0..5
prefix action space: normalized model16
internal boundary: model16 padded to 32
loss: postfix only
prefix clamp: after every denoise step
```

推理时的 RTC 契约是：

```text
prev_chunk_left_over: 上一 chunk 尚未执行的 normalized model16
inference_delay: 客户端根据 P95 请求耗时估算
execution_horizon: 10
attention schedule: LINEAR
max guidance weight: 10
```

checkpoint 中没有名为 `action_prefix` 或 `prefix_length` 的独立权重。训练把条件语义写入原有
模型参数；真正决定是否走这条路径的是推理请求中的 `mode=rtc`、`prev_chunk_left_over` 和
`inference_delay`，以及服务端把这些参数传给 `predict_action_chunk()`。

### 6.6 模型输出和后处理

PI0.5 每次输出一个最多 50 步的归一化 model16 chunk：

```text
raw_actions.shape = (50, 16)
```

服务端把副本送到 CPU postprocessor：

```text
ACTION Quantile 反归一化
model16 -> raw18
左右夹爪 force 固定回填 80
```

最终响应同时保留：

```text
raw_actions:       (T,16)，供下一次 RTC leftover 使用
processed_actions: (T,18)，供机器人执行
```

不能用 raw18 作为下一次 RTC prefix，因为训练期 prefix 位于归一化 model16 空间。

### 6.7 延迟丢弃和队列续接

客户端收到响应后，根据从观测采集到响应就绪的实际耗时计算：

```text
drop_steps = ceil((ready_time - observation_time) / 0.05)
```

[`ActionChunkQueue.merge_rtc()`](pi05/runtime/action_queue.py) 丢弃新 chunk 前部已经过时的动作，
最多保留 50 步，并用新 chunk 替换执行队列。队列同时保存 model16/raw18 两个时间对齐的版本：

- model16 用于下一次 `prev_chunk_left_over`；
- raw18 用于 20 Hz Actor 下发。

如果连续 3 个 chunk 全部过时，客户端停止；如果执行队列为空且
`EMPTY_QUEUE_STRATEGY=stop`，客户端同样停止。

### 6.8 20 Hz 动作下发

Actor 每 50 ms 执行一次：

```text
ActionChunkQueue.pop_processed_action()
  -> build_robot_action()
  -> ActionSafety.check_tensor()
  -> make_robot_action()
  -> robot_action_processor()
  -> SerializedRobotIO.send_action()
  -> JZRobotUDP.send_action()
  -> UDP 192.168.1.81:39020
```

这里的 20 Hz 是机器人动作执行频率，不是大模型请求频率。一次模型请求返回 50 步动作，Producer
只在队列下降到 30 步或以下时请求新 chunk。最近 `052050` 日志约为：

```text
机器人动作发送: 约 20 Hz
模型请求:       约 1 Hz
稳态请求耗时:   约 180-220 ms（无其他 GPU 负载时）
稳态 drop:       4-5 步
稳态 pred_delay: 5 步
```

因此看到模型每秒只请求约一次，不表示控制频率降成了 1 Hz 或 5 Hz。

## 7. 三种 mode 的区别

| 客户端 mode | 服务端 wire mode | 动作执行方式 | RTC action-prefix |
|---|---|---|---|
| `single_step` | `single_step` | 请求一次后选一个未过时动作 | 不使用 |
| `async_single_step` | `single_step` | Producer/Actor 异步队列 | 不使用 |
| `rtc` | `rtc` | Producer/Actor 异步队列 | 使用上一 chunk 的 model16 leftover |

当前连续上机使用的是第三种 `rtc`。

## 8. 日志与结果文件

Shell 启动器自动写入：

```text
tk_infer/pi05/logs/server/server_<timestamp>.log
tk_infer/pi05/logs/client/client_<timestamp>.log
tk_infer/pi05/outputs/client/rtc_armed_<timestamp>/client_summary.json
```

RTC 指标行：

```text
[jz/pi05/rtc-client] sensor=... actor=... sent=... requests=... queue=... \
request_ms=... p95_ms=... server_ms=... drop=... pred_delay=...
```

字段含义：

| 字段 | 含义 |
|---|---|
| `sensor` | 已采集观测 tick 数 |
| `actor` | Actor 控制 tick 数 |
| `sent` | 已交给 action transport 的动作数 |
| `requests` | 已完成的模型请求数 |
| `queue` | 当前未执行动作数 |
| `request_ms` | 最近一次端到端模型请求耗时 |
| `p95_ms` | 最近最多 100 次请求的 P95 |
| `server_ms` | 服务端最近一次推理处理耗时 |
| `drop` | 最近一次响应实际丢弃的过时动作步数 |
| `pred_delay` | 下一次 RTC 请求提供给模型的预测延迟步数 |

## 9. 当前实现边界

- 当前 checkpoint 只接受 `camera_head` 和 `camera_right`；不能把 client 改成 `three_camera`。
- 当前 tokenizer 不在 checkpoint 内，离线部署时必须保留单独的 tokenizer 目录。
- 当前 `checkpoints/current` 不是 `052050`，因此必须显式设置 `POLICY_PATH`。
- 当前 armed 命令关闭关节 delta 检查，不会额外裁剪 checkpoint 产生的大动作跳变。
- UDP command 没有应用层 ACK；`sent` 只表示动作已交给 UDP sender。
- 服务端启动不做 warm inference；建议先运行 RTC dry-run，使 CUDA 和 RTC 路径进入稳态。
- 同一 GPU 上的其他训练/推理进程会拉高 `p95_ms`、`pred_delay` 和 `drop`，但不会改变 20 Hz Actor
  目标频率。
- 训练期 RTC 和推理期 RTC 缺一不可：checkpoint 学到 action-prefix 条件语义，`mode=rtc` 负责在
  运行时真正提供 prefix、delay 和 horizon。

## 10. 关键代码索引

| 模块 | 关键入口 |
|---|---|
| Shell 服务端入口 | [`pi05/run_server.sh`](pi05/run_server.sh) |
| Shell 客户端入口 | [`pi05/run_client.sh`](pi05/run_client.sh) |
| Python 服务端入口 | [`pi05/run_policy_server.py`](pi05/run_policy_server.py) |
| Python 客户端入口 | [`pi05/run_robot_client.py`](pi05/run_robot_client.py) |
| checkpoint 校验和加载 | [`pi05/runtime/checkpoint.py`](pi05/runtime/checkpoint.py) |
| PolicyService 与 RTC kwargs | [`pi05/runtime/policy_service.py`](pi05/runtime/policy_service.py) |
| 三线程 runtime | [`pi05/runtime/client_runtime.py`](pi05/runtime/client_runtime.py) |
| action queue 和 leftover | [`pi05/runtime/action_queue.py`](pi05/runtime/action_queue.py) |
| HTTP client | [`pi05/runtime/remote_client.py`](pi05/runtime/remote_client.py) |
| HTTP server | [`pi05/runtime/http_server.py`](pi05/runtime/http_server.py) |
| protocol v3 | [`pi05/runtime/protocol.py`](pi05/runtime/protocol.py) |
| JZ UDP 驱动 | [`pi05/robot_driver/jz_robot_udp/jz_robot_udp.py`](pi05/robot_driver/jz_robot_udp/jz_robot_udp.py) |
| 训练期 RTC 契约 | [`pi05/checkpoints/052050/pretrained_model/tk_rtc_training_config.json`](pi05/checkpoints/052050/pretrained_model/tk_rtc_training_config.json) |
