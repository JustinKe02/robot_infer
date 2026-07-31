# PI0.5 优化推理链路

本目录保存新的 PI0.5 推理优化链路。可信基线仍位于 `tk_infer/pi05/`，优化实现不替换基线代码。

当前真机组合为：

```text
pi05_optimized 策略服务
  + tk_infer/pi05 已审计机器人客户端
  + step-015900 完整 checkpoint
  + head/right 真实相机和 raw18 state
```

完整上机命令见 [PI0.5 步骤 015900 真机运行手册](REAL_ROBOT_RUNBOOK.md)。

## 数据和执行边界

- 策略和 RTC 空间：归一化 `model16`。
- 机器人执行空间：物理量 `raw18`。
- 两种表示在所有变换后必须保持相同时间长度和一一对应。
- `raw18[0:14]` 为关节，`raw18[14]`、`raw18[16]` 为夹爪宽度。
- `raw18[15]`、`raw18[17]` 为力槽，在执行边界必须精确等于 `80`。
- 默认 chunk 长度为 `50`，目标 Actor 控制频率为 `20 Hz`。
- RTC leftover 必须来自未消费的 model16，并与 raw18 队列对应。

## 当前实现范围

### 阶段 0：保持基线的独立框架

阶段 0 通过轻量 `TorchPolicyBackend` 复用可信 PyTorch `PolicyService`。优化服务校验成对的
model16/raw18 轨迹，并默认使用严格恒等的 `pass_through` 处理器。协议 v3、受限反序列化、HTTP
认证、checkpoint 校验和原模型 processor 均沿用基线实现。

### 阶段 1：计时和可观测性

实现了显式源时间戳、可注入单调时钟、有界 p50/p95/p99 指标，以及不含图像 payload 的 JSONL
推理/失败 trace。服务指标、客户端请求、队列、丢步、延迟、Sensor/Actor 周期和抖动均使用有界窗口。

### 阶段 2：优化 PyTorch 后端

`torch_optimized` backend 提供独立运行清单和分阶段计时。所有优化开关默认关闭：

- `PI05_OPT_TORCH_INFERENCE_MODE`：仅支持 single-step；RTC guidance 需要 autograd。
- `PI05_OPT_TORCH_BF16_AUTOCAST`：显式 CUDA BF16 autocast。
- `PI05_OPT_TORCH_PINNED_MEMORY`：每请求使用 pinned CPU 输入。
- `PI05_OPT_TORCH_NON_BLOCKING_COPIES`：要求同时启用 pinned memory。
- `PI05_OPT_TORCH_WARMUP_ITERATIONS`、`PI05_OPT_TORCH_WARMUP_SEED`：确定性启动预热。
- `PI05_OPT_TORCH_STATIC_BUFFERS`、`PI05_OPT_TORCH_CUDA_GRAPH`：当前因动态 observation/token、KV cache、
  RTC 分支和 denoising loop 而在启动时拒绝，不做静默回退。

同 checkpoint 基准中没有配置达到 p95 改善 20% 的目标。step-015900 真实相机/state A/B 中，
single-step `inference_mode` 最快，request p95 为 `156.19 ms`；reference 为 `162.94 ms`，改善
`4.14%`。它仍未达到逐周期推理 `50 ms` 目标，并且不能用于 RTC。

### 阶段 3：Realtime-VLA Triton 评估

固定 Realtime-VLA v1 Triton 实现仅接受已审计本地 safetensors。转换器输出 safetensors 和版本化
manifest，不生成或加载 pickle。当前 specialization 固定为 `head_right`、双视角、chunk 50、内部
action32 映射到已证明的 0..15、BF16 和 single-step。

Triton 基准将 p95 从 `184.36 ms` 降至 `122.94 ms`，改善 `33.31%`，但仍未达到 `50 ms`，并且
不支持 RTC，所以只作为评估 backend。

### 阶段 4：传输评估

协议 v3 的 3.69 MiB 请求在 500 次认证回环测试中 p95 为 `4.56 ms`，仅占最小 reference 模型 p95
的 `2.48%`。传输不是主要瓶颈，因此没有启动协议 v4 设计。

### 阶段 5：时间戳对齐

时间戳对齐目前是客户端 shadow observer。相机和 state 源时间必须来自显式配置的同一时钟域；接收
和构建的单调时间仍是进程本地指标。有界 raw18 历史执行括号线性插值，并拒绝外推、时间回退、
过大 skew、混合时钟域和非有限值。shadow 报告固定为 `changed_policy_input=false`。

### 阶段 6：成对时间轨迹处理

显式 `paired_temporal` 处理器与默认 `pass_through` 并存。它从 raw18 的 0..13 关节维度生成单调
源位置映射，并同时作用于 model16/raw18；夹爪独立插值，力槽恢复为精确 `80`。当前只实现
`0.02 rad` chunk 内关节步长边界，不声明加速度或 jerk 优化。

异常、不可行、超时、非法映射、NaN/Inf 和约束违反均使请求失败，不回退到 pass-through。可选依赖
固定为 SciPy `1.15.3`、OSQP `1.0.4`；当前 `lerobot_flex` 环境未安装它们，确定性速度映射不依赖
这些包，未来 QP 模式必须版本完全匹配后才能启动。

### 阶段 7：本地跟踪器和可选 MPC

`LocalActionTracker` 保存有界单调 raw18 state 历史，只限制 0..13 关节维度，并估计一阶滞后。
contact innovation 只能降低速度，不作为安全传感器。tracker/MPC 异常、deadline、非有限值、force
错误或 rate 违反会停止客户端、清空待执行 RTC 队列并禁止后续 sink 写入。

MPC 还要求 tracker replay 通过、SciPy/OSQP 精确版本和已审计 solver。当前缺少这些条件，因此 MPC
保持 `BLOCKED`，不做隐式回退。

### 阶段 8：训练期动作条件化

现有三相机 010600 checkpoint 已包含 training-time action conditioning：hard prefix、token-wise
flow timestep、postfix-only loss 和独立 learned RTC 参数。它通过独立
`torch_rtc_conditioned` backend 接入，不安装 inference-time `RTCProcessor`，也不执行 VJP。

backend 将现有 RTC 请求中的 `predicted_delay_steps` 映射为 `prefix_length`，并从未消费 model16
leftover 的开头提取同长度 clean prefix。delay 超过 checkpoint 的 `max_delay=10`、prefix 缺失或
长度不足时显式失败，不裁剪后静默继续。首个队列为空的请求使用 prefix length 0。

锁定 profile 使用三相机、实际训练 task 和独立端口 18089：

```bash
CONFIG_ONLY=true \
bash tk_infer/pi05_optimized/profiles/rtc_conditioned_010600/run_policy_server.sh

conda run -n lerobot_flex python \
  tk_infer/pi05_optimized/tools/offline_rtc_conditioned_backend_smoke.py
```

真实 GPU smoke 已证明 prefix length 0 与普通 full chunk 精确一致，prefix length 5 在 model16 和
raw18 两个空间均精确 clamp，force 槽为 80。当前仍缺 recorded paired replay、逐请求 overflow
trace、真实输入 dry-run 和机器人门禁；离线接入通过不等于已授权上机。

### 阶段 9：固定速度配置研究

当前仅评估 `1.0x`、`1.25x`、`1.5x` 固定映射，没有启用 learned throttle。scheduler 完成、关节
约束、pairing、force 和 finiteness 与任务成功率、物理周期分开报告；没有每档至少 10 次带来源标注
的现场 trial 时，不得把 scheduler 结果表述为任务成功。

## 20 Hz 的正确含义

当前 RTC 的 20 Hz 指 Actor 每 50 ms 消费并发送一个动作，不要求模型每 50 ms 完成一次完整推理。
模型每次生成 50-step chunk，Producer 在队列降低到 low watermark 后补充新 chunk，并根据请求 p95
计算 `predicted_delay_steps`。

2026-07-30 真机稳态数据：

```text
Sensor：             约 20 Hz
Actor：              20 Hz
模型请求：           约 1.1 request/s
请求 p95：           226.24 ms
drop/pred delay：    4-5 steps
queue：              无空队列事件
```

`p95 <= 50 ms` 是“每控制周期完成一次完整推理”的性能优化目标，不是 chunked RTC 维持 20 Hz Actor
输出的必要条件。两类门禁必须分开报告。

## 无硬件检查

所有 Python 测试必须使用 `lerobot_flex` Conda 环境：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

bash tk_infer/pi05_optimized/run_config_checks.sh

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
conda run -n lerobot_flex python -m pytest -q tk_infer/pi05_optimized/tests
```

真实 checkpoint 数值一致性，不访问网络或机器人：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_reference_parity.py
```

加速可观测性 soak：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_observability_soak.py --iterations=10000
```

正式 30 分钟 wall-clock soak：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_observability_soak.py --duration-s=1800 --rate-hz=20
```

阶段 2 模型和阶段基准：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_phase2_benchmark.py \
  --warmup=20 --repetitions=100 --stage-repetitions=10
```

协议 v3 传输基准：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_transport_benchmark.py \
  --warmup=50 --repetitions=500
```

阶段 3 转换校验和 artifact 生成：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/convert_pi05_safetensors_to_realtime_vla.py \
  --validate-only --report-json=tk_infer/pi05_optimized/outputs/phase3_conversion_validation.json

conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/convert_pi05_safetensors_to_realtime_vla.py
```

阶段 3 同 seed 一致性和性能基准：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_phase3_triton_benchmark.py \
  --warmup=20 --repetitions=100
```

阶段 5 时间戳对齐 replay：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_phase5_alignment_replay.py \
  --duration-s=60 --rate-hz=20 --camera-delay-s=0.03 --readout-delay-s=0.005
```

阶段 6 合成和 recorded replay：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_phase6_temporal_replay.py \
  --source=synthetic --speed-factor=1.0 --max-joint-step-rad=0.02 \
  --output-json=tk_infer/pi05_optimized/outputs/phase6_temporal_synthetic_replay.json

conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_phase6_temporal_replay.py \
  --source=recorded --speed-factor=1.0 --max-joint-step-rad=0.02 \
  --output-json=tk_infer/pi05_optimized/outputs/phase6_temporal_recorded_replay.json
```

阶段 7 tracker replay：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_phase7_tracker_replay.py \
  --duration-s=60 --rate-hz=20 --max-joint-step-rad=0.02
```

阶段 8 训练决策门禁：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_phase8_training_gate.py
```

阶段 9 固定速度 replay：

```bash
conda run -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_phase9_speed_profiles.py \
  --source=recorded --control-hz=20 --max-joint-step-rad=0.02 \
  --output-json=tk_infer/pi05_optimized/outputs/phase9_fixed_speed_recorded_profiles.json
```

## 策略服务

优化服务默认端口为 `18088`，与基线端口 `8088` 隔离。环境变量使用 `PI05_OPT_*` 或
`JZ_PI05_OPT_*` 前缀。默认 trace 每个文件最多 16 MiB，并保留两个轮转备份。

只校验配置：

```bash
CONFIG_ONLY=true bash tk_infer/pi05_optimized/run_server.sh
```

只加载模型并校验，不推理、不监听：

```bash
PI05_OPT_POLICY_PATH=/absolute/path/to/pretrained_model \
CHECK_POLICY_LOAD=true \
bash tk_infer/pi05_optimized/run_server.sh
```

部署时优先使用锁定的完整 15-epoch 配置。single-step 只读评估可启用 inference mode：

```bash
PI05_OPT_BACKEND=torch_optimized \
PI05_OPT_TORCH_INFERENCE_MODE=true \
bash tk_infer/pi05_optimized/profiles/step_015900/run_policy_server.sh
```

RTC 服务必须关闭 inference mode：

```bash
PI05_OPT_BACKEND=torch_optimized \
PI05_OPT_TORCH_INFERENCE_MODE=false \
PI05_OPT_TRAJECTORY_PROCESSOR=pass_through \
bash tk_infer/pi05_optimized/profiles/step_015900/run_policy_server.sh
```

健康检查：

```bash
bash tk_infer/pi05_optimized/profiles/step_015900/run_health_check.sh
```

非回环地址绑定必须提供 `JZ_PI05_OPT_SERVER_AUTH_TOKEN`。token 只从环境读取，不接受启动器命令行参数。

## 真实相机/state 只读基准

```bash
SERVER_URL=http://127.0.0.1:18088 \
WARMUP_REQUESTS=3 \
MEASURE_REQUESTS=30 \
CONTROL_HZ=5 \
bash tk_infer/pi05_optimized/run_live_readonly_benchmark.sh
```

该工具只创建带时间戳的 ZMQ 相机 subscriber 和 UDP state receiver，不创建 Robot 或 command
transport。策略输出只写入进程内 recording sink，聚合 JSON 位于 `outputs/live/`。

汇总固定 step-015900 A/B 报告：

```bash
conda run --no-capture-output -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/live_backend_ab_summary.py
```

默认报告为 `outputs/live/live_backend_ab_step015900_20260730.json`。它校验请求数量、trace、相机、
state、raw18、force、接收器停止状态和 `action_sent=false`。

## P5 阶段软件门禁和真机现状

`P5SingleActionGuardedSink` 是依赖注入的内存 guard，不是机器人 adapter。它校验短时授权、fresh
state、robot/sender identity、raw18、force、gripper 和最大 `0.02 rad` 初始关节差，并在任何失败后
永久 latch。当前实现拒绝宣称 hardware、command transport 或 armed capability 的 delegate，因此
它本身不能发送机器人动作。

只读评估命令：

```bash
conda run --no-capture-output -n lerobot_flex \
  python tk_infer/pi05_optimized/tools/offline_p5_readiness_gate.py
```

`outputs/p5_software_readiness.json` 仍会根据逐周期性能门禁返回 `BLOCKED`。这与已经完成的现场授权
验证是两个不同事实：软件 guard 没有 hardware adapter；现场验证则明确使用旧版已审计硬件客户端。

2026-07-30 已完成：

- 5 Hz single-step armed：一个请求、精确一个 UDP 动作、正常断开。
- 20 Hz RTC armed：10 秒、201 Sensor ticks、200 Actor ticks、195 个 UDP 动作、11 个模型请求。
- request p95 `226.24 ms`，delay/drop `4-5`，最终 queue `41`。
- empty、hold、skip、stop、fully stale、backend、trace 错误全部为 `0`。
- 本次现场模式按明确授权关闭 initial/per-step joint-delta 检查，其他边界继续启用。

直接运行命令和停止步骤见 [真机运行手册](REAL_ROBOT_RUNBOOK.md)。

## 可写目录

所有运行时可写路径限制在本目录：

- `run_state/`
- `logs/`
- `outputs/`
- `artifacts/`

这些路径由 `tk_infer/.gitignore` 忽略。
