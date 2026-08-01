# PI0.5 推理优化任务台账

更新时间：2026-07-31

本文档记录 PI0.5 优化推理链路的实施和验证证据。可信基线 `tk_infer/pi05/` 保持不变；新的优化实现
位于 `tk_infer/pi05_optimized/`。直接真机命令见
[`tk_infer/pi05_optimized/REAL_ROBOT_RUNBOOK.md`](pi05_optimized/REAL_ROBOT_RUNBOOK.md)。

## 状态说明

- `[x]`：已完成并验证。
- `[ ]`：尚未开始。
- `[~]`：进行中。
- `[!]`：被明确前置条件、依赖或现场授权阻塞。

## 固定边界

- 构建优化链路时不修改或替换 `tk_infer/pi05/` 基线。
- 所有 Python 测试必须通过 `conda run -n lerobot_flex` 执行。
- Phase 0–3 不连接机器人、相机服务、state UDP 或 command UDP。
- 普通 `dry_run` 仍可能读取真实机器人和相机，它不等于无硬件模式。
- backend、优化器、时间戳、队列或 tracker 失败必须显式停止。
- 运行中不得静默切换 backend、动作表示、优化模式或执行频率。
- 所有可执行 raw18 动作必须经过 `ActionSafety` 和 JZ driver 边界。
- 真机现场可以在明确授权下复用已审计旧客户端；优化目录自身仍不实现 Robot adapter。
- 2026-07-30 接手后的默认工作范围是 `tk_infer/`；未经新的明确授权，不启动客户端、不创建动作
  transport、不连接机器人，也不停止当前监听 `127.0.0.1:18088` 的只含策略服务进程。

## 不可变数据契约

- 策略/RTC 空间：归一化 `model16`。
- 机器人执行空间：物理量 `raw18`。
- 两种表示在每次变换后必须保持相同时间长度和对应关系。
- `raw18[0:14]` 为关节。
- `raw18[14]`、`raw18[16]` 为夹爪宽度。
- `raw18[15]`、`raw18[17]` 为力槽，执行边界必须精确等于 `80`。
- 默认 chunk 长度为 `50`，目标 Actor 控制频率为 `20 Hz`。
- RTC leftover 必须来自尚未消费、且与 raw18 对应的 model16。

## 阶段 10：抓取行为整改（进行中）

目标：先把 checkpoint、prompt、随机采样和运行时状态混杂的问题拆开，再引入任何夹爪控制约束。

- [x] 复核训练脚本默认提示为 `Put the bottle on the right into the basket on the left.`，但实际
  100-episode 数据集 `tasks.parquet` 的唯一训练 task 是 `jz robot pin timed vr teleoperation`。
- [x] 当前目标明确恢复为原训练任务，不包含 plate；RTC-conditioned profile 锁定数据集中实际
  task，不能用脚本默认值替代 dataset task 证据。
- [x] 确认右夹爪 raw18 语义为 `0=open`、`100=closed`；force 槽不是接触反馈。
- [x] 首次离线 smoke 证明 postprocessed 夹爪预测可能越过执行范围；行为报告保留原值并单独统计
  越界，真正执行仍由既有安全边界 fail-closed。
- [~] 固定同一 recorded observation、seed、checkpoint 和 prompt，对 reference 与真实
  `torch_optimized` backend 分别前向，并记录右夹爪完整 chunk、开闭转换、重复闭合和独立 chunk
  边界翻转。
- [ ] 在实际训练 task 下分别汇总 standard step-010600、step-015900 和 RTC-conditioned
  step-010600 的多 seed 行为差异。
- [ ] 实现 one-shot grasp commitment 纯运行时组件；默认关闭，不依据固定 force=80 判断抓取成功。
- [ ] 增加显式任务完成终止条件；完成后清空队列并禁止后续 action sink 写入。
- [x] 已审计现有 RTC-conditioned 010600 checkpoint：三相机、model16/raw18、`max_delay=10`，
  并包含 5 个普通 checkpoint 不存在的 learned RTC 参数。
- [x] 已建立独立 `torch_rtc_conditioned` backend，显式关闭 VJP，使用 delay-sized clean
  prefix，拒绝 untrained overflow、prefix 缺失和非 conditioned checkpoint。
- [x] 真实 GPU 离线 smoke 通过：prefix=0 与普通 full chunk 零误差，prefix=5 在 model16/raw18
  均精确 clamp，force 槽保持 80。
- [!] conditioned checkpoint 尚未完成 recorded paired replay、真实输入 dry-run 和上机门禁。
- [!] Realtime-VLA v2 继续作为独立接入阶段；当前 Phase 0-9 和 v1 single-step 不得称为 v2。

## 阶段 11：Realtime-VLA v2 训练期 RTC（等待新 checkpoint 验收）

目标：保留本地协议 v3、model16/raw18 processor 和已审计队列语义，只接入 v2 RTC Triton kernel；
不得以源码接入或 config-only 通过替代效果验收。

- [x] 固定 `dexmal/realtime-vla-v2@a36d02a7b241de1129af2048e749de58f95ead9c` 的最小 kernel
  文件、MIT LICENSE、Git blob、上游 SHA-256 和修改后 SHA-256。
- [x] 未引入上游不受限 pickle server/checkpoint loader、AIRBOT observer/actuator、time-axis optimizer
  或 MPC。
- [x] 扩展 v2 PI0.5 RTC kernel，消费本地训练架构的 5 个 learned token-flow/prefix tensor；prefix
  token 使用 flow time 0 和 prefix embedding，AdaRMS 继续使用当前全局 denoise time。
- [x] 实现 51-tensor BF16 `safetensors + manifest.json` converter、严格 artifact loader 和
  `realtime_vla_v2` backend；拒绝错误 commit/header/hash/key/shape/dtype/task/camera/action/RTC contract。
- [x] backend 将 delay-sized model16 leftover 补零到 internal32，拒绝超过训练 max_delay、缺失/过短
  leftover、非有限值和 kernel prefix clamp 失败；raw18 force 槽必须精确为 80。
- [x] 配置、独立 artifact 路径、shell launcher 和 config-only 隔离已接入；config-only 不读取
  checkpoint/artifact，不构造模型或 socket。
- [x] v2 provenance、artifact 篡改、5 tensor 映射、token condition、prefix、force、配置和 launcher
  CPU 测试通过。
- [!] 新 RTC checkpoint 正在外部重新训练，尚未生成正式 v2 artifact。
- [!] 尚未执行同 observation/task/noise 的 Torch-conditioned 与 v2 GPU parity；必须覆盖 prefix
  `0/1/5/max_delay` 和至少 20 个固定 seed。
- [!] 尚未通过行为门禁：prefix=0 至少 `18/20` 合理闭合，prefix=5 至少 `16/20`。
- [!] 尚未通过性能门禁：p95 `<=50 ms`，或相对同 checkpoint Torch 至少改善 20%，且 p99
  `<=100 ms`。
- [!] 通过全部离线/只读门禁前，不启动 v2 inference smoke、动作 transport 或 armed 控制。

当前无硬件验收命令：

```bash
PI05_OPT_BACKEND=realtime_vla_v2 \
PI05_OPT_RTC_CONDITIONED_TASK='jz robot pin timed vr teleoperation' \
CONFIG_ONLY=true \
bash tk_infer/pi05_optimized/run_server.sh

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lerobot_flex \
  python -m pytest -q tk_infer/pi05_optimized/tests/test_realtime_vla_v2.py \
  tk_infer/pi05_optimized/tests/test_config.py \
  tk_infer/pi05_optimized/tests/test_launcher_isolation.py
```

离线行为 A/B 入口：

```bash
conda run -n lerobot_flex python \
  tk_infer/pi05_optimized/tools/offline_gripper_behavior_ab.py
```

默认矩阵不会创建 socket 或 Robot adapter；历史报告保留 prompt sensitivity 诊断，但后续正式比较
只使用 `tasks.parquet` 中的实际训练 task。45/55 阈值只用于报告中的滞回分类，不是执行控制
或抓取成功判据。

## 外部参考

- Realtime-VLA v1：`Dexmal/realtime-vla@b86a942a073ea241f9bd6916a705f81906f4638b`
- Realtime-VLA v2：`dexmal/realtime-vla-v2@a36d02a7b241de1129af2048e749de58f95ead9c`
- Running VLAs at Real-time Speed：arXiv `2510.26742`
- Realtime-VLA V2：arXiv `2603.26360`
- Training-Time Action Conditioning for Efficient Real-Time Chunking：arXiv `2512.05964`

## 阶段 0：保持基线的独立框架

目标：建立可独立启动、默认数值等价于可信 PyTorch 路径的优化框架，本阶段不声明性能提升。

- [x] 创建并持续维护本任务台账。
- [x] 建立隔离的 package、runtime、backend、profile、test 和 artifact 路径。
- [x] 建立不可变 `OptimizedRuntimeConfig`，默认 `torch + pass_through`。
- [x] 建立带健康检查和推理契约的 `PolicyBackend` 接口。
- [x] 用 `TorchPolicyBackend` 适配已审计 PI0.5 `PolicyService`。
- [x] 校验 model16/raw18 的维度、时间对齐、有限值和力槽。
- [x] 实现返回防御性副本的严格恒等轨迹处理器。
- [x] 复用协议 v3、受限反序列化、Bearer 认证、checkpoint 校验和安全边界。
- [x] 使用独立端口 `18088` 和隔离的可写目录。
- [x] 证明 import/config-only 不打开 socket、不加载 checkpoint、不连接硬件。
- [x] 使用 checkpoint 010600 和离线样本证明 single-step/RTC 精确一致。

验收命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n lerobot_flex \
  python -m pytest -q tk_infer/pi05_optimized/tests

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n lerobot_flex \
  python -m pytest -q tk_infer/pi05/tests tk_infer/tests

conda run -n lerobot_flex python -m ruff check tk_infer/pi05_optimized
```

## 阶段 1：带时间的观测和指标

目标：选择优化前测量完整延迟预算。

- [x] 定义 request、observation、preprocess、model、postprocess、response、queue 和 actor 的单调时间。
- [x] 分离源时间戳和接收时间戳，禁止相互伪造。
- [x] 在无 payload trace 和客户端 telemetry 中记录请求 ID、观测序列、backend、mode、chunk、drop 和 queue。
- [x] 实现有界 JSONL trace，不默认记录图像。
- [x] 实现 p50/p95/p99、queue starvation、stale chunk、重复帧和 Actor jitter 指标。
- [x] 实现可注入时钟、并发、轮转和失败路径测试。
- [x] 完成 30 分钟无硬件 soak：36,000 次、20 Hz、1800.0015 秒、无失败、无泄漏线程。

## 阶段 2：优化 PyTorch 后端

目标：不改变模型语义，移除可避免的框架和内存开销。

- [x] 独立实现 inference mode、BF16、pinned memory、non-blocking copy、static buffer 和 warmup 开关。
- [x] 明确拒绝当前不兼容的 static buffer/CUDA Graph，不静默回退。
- [x] 分别测量 observation prepare、preprocess、模型、device copy、postprocess 和 transport。
- [x] 每种优化使用相同 checkpoint、样本、请求、RTC leftover 和 seed 对比 reference。
- [x] 启动时拒绝不支持的 device/dtype/shape。
- [x] 明确 `inference_mode` 仅支持 single-step；RTC guidance 需要 autograd。
- [x] 使用 step `15900/15900`、真实 `head_right` 和 raw18 state 完成 3 次 warmup + 30 次测量。
- [x] 汇总请求、trace、相机、state、raw18、force 和无动作完整性门禁。

数值门禁：FP32 raw18 最大绝对误差 `<=1e-5`，平均绝对误差 `<=1e-6`；BF16 采用 Phase 3 精度门禁。

## 阶段 3：Triton PI0.5 后端

目标：通过显式 backend 标志评估 Realtime-VLA v1 kernel。

- [x] 固定所需 PI0/PI0.5 kernel、MIT notice、源码哈希、Git blob 和 commit。
- [x] 实现安全的 `safetensors -> safetensors + manifest.json`，不使用 pickle。
- [x] manifest 记录 checkpoint、转换器、SHA-256、Torch/CUDA/Triton、dtype、视角、chunk、action 和 mode。
- [x] 当前只支持 `head_right`、chunk 50、model16 specialization。
- [x] 证明 action32 的 0..15 到 model16 的映射，不通过猜测截断。
- [x] Triton 仅声明 single-step；RTC 继续使用 PyTorch，拒绝时不回退。

低精度门禁：关节 p99 `<0.005 rad`、最大误差 `<0.01 rad`、夹爪 p99 `<0.5`、force 精确 `80`、
无 NaN/Inf。

## 阶段 4：版本化传输评估

- [x] 协议 v3 保持可回退。
- [x] 完成 500 请求认证回环基准：3,689,702 字节请求，p95/p99 为 `4.56/6.20 ms`。
- [!] 传输仅占 reference 模型 p95 的 `2.48%`，未触发协议 v4 设计。
- [!] 未来 v4 必须固定 schema、dtype、shape、请求上限、认证、超时和请求 ID，且不得任意反序列化 pickle。

## 阶段 5：时间戳对齐

- [x] 实现带相机源、state 源、接收和构建时间的 `TimedObservation`。
- [x] 保存有界 raw18 历史，并在校准后的图像时间执行无外推插值。
- [x] 显式配置相机、state、readout delay，校验时钟域、单调性、有限值和 skew。
- [x] 先以 shadow 方式运行，只记录诊断，不替换策略输入。
- [!] 机器人运动延迟标定需要独立现场实验。

## 阶段 6：成对时间轨迹优化

- [x] 固定可选 SciPy `1.15.3`、OSQP `1.0.4`；当前环境未安装，QP 路径 fail-closed。
- [x] 默认 `speed_factor=1.0`，报告中明确关闭加速度和 jerk 目标。
- [x] 使用同一单调插值映射同时处理 model16 和 raw18。
- [x] 只优化 0..13 关节，独立处理夹爪，force 恢复为精确 `80`。
- [x] solver、不可行、超时、非法映射、非有限值和步长违反均 fail-closed。
- [x] 完成真实 recorded observation 和 checkpoint 生成 50-step 轨迹 replay。
- [x] 使用最佳 Phase 2 backend 完成真实只读 shadow；33/33 成功，但 p95 回退，因此保持 opt-in。

## 阶段 7：本地跟踪器和可选 MPC

- [x] 实现确定性有界 state replay、关节 rate limit 和一阶 lag estimator。
- [x] contact innovation 只作为减速信号，不作为安全传感器。
- [x] tracker 通过两次 1,200-cycle/20 Hz 确定性 replay，action SHA-256 一致。
- [!] 缺少精确 SciPy/OSQP 和已审计 solver，MPC 保持 `BLOCKED`。
- [x] tracker/MPC 失败会停止客户端、清空执行队列、复位 tracker 并禁止后续动作。

## 阶段 8：训练期动作条件化

- [x] 实测 `delay_steps=4-5`，满足训练期条件化的触发条件。
- [x] 已有训练实现包含随机 delay、hard action prefix、token-wise flow timestep 和 postfix-only
  loss，并已有单独训练完成的 010600 checkpoint。
- [x] checkpoint 明确 `rtc_training.enabled=true`、`max_delay=10`、
  `min_postfix_steps=1`、`rtc_config=null`，权重含独立 learned RTC 参数。
- [x] 选择训练期条件化时拒绝缺少 `rtc_training.enabled=true` 和 contract tag 的旧 checkpoint。
- [x] 独立 `torch_rtc_conditioned` backend 和锁定 profile 已建立，不安装 inference-time VJP。
- [x] 真实 checkpoint 离线 prefix=0/prefix=5 contract smoke 通过。
- [!] 统一 A/B/C recorded replay、逐请求 prefix/overflow trace 和真实输入 dry-run 尚未完成。

## 阶段 9：固定速度和学习式节流研究

- [x] 当前只实现 `1.0x/1.25x/1.5x` 固定映射，不启用 learned throttle。
- [!] 每档至少需要 10 次带来源标签的现场 trial，才能建立成功率/周期曲线。
- [!] throttle rollout 需要独立机器人实验授权。

## 跨阶段失败注入

- [x] HTTP 不可用、500、畸形响应、错误 content type、慢超时。
- [x] 认证失败、协议版本错误、body 超限和 request content type 错误。
- [x] 相机 stale/reuse、state 不前进、时间 skew、观测序列回退。
- [x] 动作 shape、model16/raw18 时间不一致、NaN/Inf、force 非 80。
- [x] 空/满 queue、连续 3 个 fully stale chunk。
- [x] optimizer、solver 不可行/超时、tracker/MPC deadline。

要求：必须记录明确 `stop_reason`，停止后不得继续发送 UDP，保留诊断，禁止自动语义回退。

## 两类性能门禁

逐周期完整推理优化目标：

- request p95 `<=50 ms`，p99 `<=100 ms`。
- 同硬件 p95 至少比 reference 改善 20%。
- `predicted_delay_steps` p99 `<=2`。
- stale/repeated frame rate `<0.1%`。

chunked RTC 功能门禁：

- Sensor/Actor 接近 `20 Hz`。
- queue 不为空，无 stop、skip、hold 和 fully stale。
- delay/drop 在 50-step chunk 和训练/运行覆盖范围内。
- 动作输出 finite，model16/raw18 配对，force 精确 `80`。
- 客户端按时间边界或人工停止后正常断开。

`p95>50 ms` 表示没有实现“每 50 ms 做一次完整模型推理”，不能单独证明 chunked RTC 无法以 20 Hz
发送动作。2026-07-30 真机已经验证 request p95 `226.24 ms` 时 Actor 仍稳定为 20 Hz。

## 机器人授权阶梯

- P0：静态、配置、manifest，无硬件。
- P1：离线数值一致性和 recorded replay，无硬件。
- P2：假服务回环协议和失败注入，无硬件。
- P3：性能和 30 分钟 soak，无机器人 transport。
- P4：真实相机/state 只读和 inference smoke，已授权并完成。
- P5：一个 armed 动作，已于 2026-07-30 通过旧客户端 + 优化 server 完成。
- P6：短时 armed RTC，已完成 20 Hz/10 秒试验；更长时间仍需按上一轮日志逐级确认。
- P7：shadow/A-B 和长时评估，等待带任务标签的现场试验。

## P5 阶段软件保护和真实硬件边界

`P5SingleActionGuardedSink` 为未来独立 adapter 准备内存契约，不是 Robot 实现：

- [x] 校验 fresh raw18、robot/sender、短时授权、force、finite、gripper 和 `0.02 rad` 初始差。
- [x] 任意校验/delegate 失败后 latch；delegate 异常视为未知交付并禁止重试。
- [x] 完成授权、急停、workspace、operator、state、shape、force、gripper、delta、并发和一次成功测试。
- [x] 离线 P5 报告消费 step-015900 A/B 和 tracker 证据。
- [x] 优化目录的 guard/launcher 仍不创建 Robot、socket 或 action transport。

因此 `outputs/p5_software_readiness.json` 的 `BLOCKED` 描述的是“优化 guard 没有可执行 hardware adapter
和逐周期性能候选”，不否定已经通过明确现场授权、使用旧版已审计硬件客户端完成的真机验证。

## 关键进展记录

### 2026-07-29

- 完成 Realtime-VLA v1/v2 源码和论文适用性分析，开始 Phase 0。
- Phase 0 Ruff、shell、config-only 和测试通过（`44 passed`）；真实 checkpoint single-step/RTC 精确一致。
- 基线回归除一个已有隔离失败外通过（`85 passed, 1 skipped`）。该失败来自 training snapshot 导入旧私有 runtime。
- Phase 1 完成时钟域、`TimedObservation`、有界指标、JSONL trace、队列/Actor telemetry 和 30 分钟 soak。
- Phase 2 完成 optimized Torch backend。20 warmup + 100 measurement 数值门禁通过，但无配置达到 20% p95 改善。
- Phase 3 安全转换校验 810 个源 tensor，生成 6,706,745,632 字节、46 tensor 的 BF16 artifact。
- Triton p95 从 `184.36` 降到 `122.94 ms`，精度门禁通过，但仍未达到 `50 ms` 且不支持 RTC。
- Phase 5 完成 60 秒、20 Hz、1,200 observation shadow replay；无失败，最大插值误差 `9.5367e-7`。
- Phase 6 recorded replay 保持 model16/raw18 配对和 force=80，速度 1.0 映射为恒等。
- Phase 7 tracker replay 两次 action hash 一致，最大关节步约 `0.02000001 rad`；MPC 仍阻塞。
- Phase 8 返回 `BLOCKED`，没有训练进程；Phase 9 固定 profile scheduler replay 通过。
- 跨阶段无硬件失败注入和最终回归通过（当时 `185` 个优化测试）。

### 2026-07-30：真实输入和只读推理

- P4 state-only：3.016 秒、90 包、29.84 Hz、raw18 精确、无序列/时间回退。
- 启动 Orin 三路相机后，10 秒 FPS 为 head/left/right `29.67/30.07/30.17`，无 gap/drop/invalid/duplicate。
- 联合相机/state：head/left/right `29.78/29.98/30.08 Hz`，state `29.91 Hz`、300 包，无动作路径。
- 旧 server 一次真实 inference smoke：模型/server `182.95/182.99 ms`，model16 `[50,16]`、raw18 `[50,18]`。
- 新增锁定 step-015900 profile：固定 7 文件、完整 SHA-256、指纹、step、camera、schema 和维度。
- 优化 server 两次真实只读请求通过；cold/warm `676.18/181.94 ms`，无 trace/backend 失败。
- 新增真实只读 benchmark；完整优化测试增至 `209 passed`，随后 P5 测试增至 `247 passed`。

### 2026-07-30：step-015900 A/B

- reference 真实 3+30：request p50/p95/p99 `158.54/162.94/166.55 ms`。
- `torch_optimized` plain p95 `167.79 ms`。
- inference mode p95 `156.19 ms`，改善 `4.14%`，但 RTC 不兼容。
- inference mode + BF16/pinned/pinned-nonblocking p95 分别 `163.07/158.51/163.05 ms`。
- 所有运行 33/33 成功，raw18/force、相机/state/trace/stale/repeated/queue 完整性通过，无动作发送。
- `paired_temporal` p95 `164.38 ms`，比 inference-mode pass-through 回退 `5.24%`，保持 opt-in。
- 正式 A/B 报告没有逐周期性能合格候选，但选出 inference mode 作为 single-step 只读候选。

### 2026-07-30：真机动作验证

- 用户明确授权现场真机、确认物理急停，并明确接受关闭 initial/per-step joint-delta 检查。
- `torch_optimized` server 关闭 inference mode，恢复 autograd RTC；第二次热请求 `177.14 ms`，冷请求
  `603.52 ms` 被确认是无 warmup 的首请求开销。
- single-step armed：5 Hz、1 秒、最多一个动作；精确发送一个 raw18 UDP 动作并正常断开。
- 20 Hz RTC armed：限定 10 秒，完成 201 Sensor ticks、200 Actor ticks、195 个 UDP 动作、11 个模型请求。
- request p95 `226.24 ms`，delay/drop `4-5`，最终 queue `41`。
- empty、hold、skip、stop、fully stale、backend、optimized runtime 和 trace 失败全部为 `0`。
- 客户端按 10 秒时间边界停止并断开，`39010/39020` 释放；策略服务保留在 `18088`。
- 单动作证据：`tk_infer/pi05/outputs/client/single_step_armed_20260730_142431/client_summary.json`。
- RTC 证据：`tk_infer/pi05/outputs/client/rtc_armed_20260730_142509/client_summary.json`。
