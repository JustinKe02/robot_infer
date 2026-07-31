# JZ Robot Pin Timed PI0.5 RTC 对比实验方案

## 1. 文档状态

- 状态：训练时 RTC 核心代码、轻量单测、单卡/四卡 GPU smoke 和单 seed B 训练已完成；现有 B
  checkpoint 已通过独立 PyTorch conditioned backend 的真实离线 prefix-clamp smoke。正式多 seed
  A/B/C 实验、recorded replay 和现场实验尚未执行。
- 适用对象：`jz_robot_pin_timed`、PI0.5、20 Hz 控制、三路相机、raw18/model16 数据边界。
- Conda 环境：`lerobot_flex`。所有训练、测试、导出和 benchmark 必须使用该环境。
- 安全边界：本文只定义实验与验收流程。未经现场明确授权，不得运行 armed 入口或向机器人发送动作。

## 2. 实验目标

本实验回答两个独立问题。

### 2.1 方法效果问题

在相同 JZ 数据、PI0.5 基模、训练预算、异步调度和控制频率下，对比以下三种方法：

| ID | 名称 | 训练方式 | 推理方式 | 额外反向/VJP |
| --- | --- | --- | --- | --- |
| A | 普通 PI0.5 | 标准 flow matching | 普通异步 full-chunk 推理 | 否 |
| B | 训练时 RTC | clean prefix 条件训练，postfix masked loss | prefix clamp，只更新 postfix | 否 |
| C | 训练后 RTC | 与 A 共用普通 checkpoint | 推理时 RTC guidance | 是 |

C 不产生独立训练 checkpoint。C 必须严格加载 A 的 checkpoint，以隔离“训练策略变化”和“推理 guidance”两种影响。

### 2.2 推理加速问题

仅针对训练时 RTC checkpoint B，对比两种推理后端：

| ID | Checkpoint | 后端 | RTC 语义 |
| --- | --- | --- | --- |
| B1 | B | PyTorch | clean prefix + postfix Euler 更新 |
| B2 | B | TensorRT | 与 B1 完全相同 |

B2 不是第四种算法。B2 的目标是验证同一个 B checkpoint 在 TensorRT 下能否降低完整推理链路延迟，同时保持 B1 的数值和 RTC 语义。

注意：B 的训练张量接口、PyTorch prefix-clamp 模型接口、单 seed checkpoint 和独立 conditioned
backend 已完成。正式 A/B/C 共享的 full-chunk 对比 runtime、recorded replay 和 B2 TensorRT engine
尚未实现，因此当前仍不能执行 E1-E4 或上机动作。

## 3. 核心假设

- H1：在存在推理延迟时，B 和 C 的 chunk 边界连续性优于 A。
- H2：B 不需要逐 denoise step 的 VJP，因此端到端延迟低于 C。
- H3：B 在训练 delay 分布覆盖部署 delay 时，任务效果不劣于 C，并具有更低的 queue underflow 风险。
- H4：B2 与 B1 数值一致，在相同硬件和输入下具有更低的 denoise loop P95 和端到端 P95。
- H5：当部署 delay 超出 B 的训练范围时，B 的收益会下降；该退化必须通过 delay 分桶显式报告。

## 4. 当前仓库基线

### 4.1 数据与模型边界

当前训练模块固定使用：

- 数据集：`data/jz_robot_pin_timed_curated_42eps_20260713`
- 规模：42 episodes、8370 frames、20 FPS、3 cameras
- 原始 state/action：18D
- PI0.5 模型 state/action：16D
- 默认 action chunk：50 steps，即 20 Hz 下覆盖 2.5 秒
- 默认 flow inference steps：10
- normalization：`QUANTILES`
- 基模默认路径：`/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base`

训练脚本允许通过 `PI05_BASE_PATH` 覆盖基模路径。若当前工作副本不在 `/data/cqy_workspace/flexible_lerobot`，必须显式提供可访问的绝对路径，不能假定仓库内存在相对 `assets/` 目录。

训练入口：

```bash
bash my_devs/jz_robot_pin_timed/pi05/smoke_train.sh
bash my_devs/jz_robot_pin_timed/pi05/train_15_epochs.sh
```

### 4.2 Flow 时间约定

本仓库 PI0.5 使用：

```python
x_t = time * noise + (1 - time) * clean_action
target_velocity = noise - clean_action
```

因此：

- `time=1.0` 表示纯噪声；
- `time=0.0` 表示 clean action；
- 推理从 `1.0` 向 `0.0` 积分。

训练时 RTC 的 clean prefix 必须使用 `token_time=0.0`。不得照搬将 `time=1.0` 定义为 clean action 的其他实现。

### 4.3 当前推理模式不能直接用于正式 A/B/C

现有 JZ client 提供 `single_step`、`async_single_step` 和 `rtc` 三种客户端运行模式；server wire protocol 只有 `single_step` 和 `rtc`，其中 `async_single_step` 在 wire 上仍使用 `single_step`。这些客户端模式的 action queue、leftover 和 chunk 消费语义不同。直接比较现有三个入口会同时改变 RTC 方法和调度策略，结论不成立。

正式实验必须增加统一的 full-chunk 异步评测路径：

```text
相同 observation producer
相同 full-chunk action queue
相同 queue watermark
相同 stale-action 丢弃规则
相同控制线程
相同 latency 统计
        │
        ├── A: 不使用 prefix 条件或 guidance
        ├── B: clean prefix 条件 + clamp
        └── C: inference-time RTC guidance
```

三种方法只能在 action chunk 生成逻辑上不同。

优化运行时现已增加独立 `torch_rtc_conditioned` backend：它把现有请求的 model16 leftover 前
`predicted_delay_steps` 映射为 clean prefix，并拒绝 VJP、overflow 和 prefix 缺失。该路径可用于
离线接入验证，但正式 A/B/C 仍缺共享同一 method switch、anchor/delay manifest 和逐请求 trace 的
统一评测 runtime。E1 及之后的实验不得用语义不同的旧入口替代正式统一路径。

## 5. 实验变量控制

### 5.1 必须固定的训练变量

A 和 B 从同一个 PI0.5 base checkpoint 独立分叉，不允许先训练 A、再从 A 继续训练 B。

固定以下变量：

- 数据集版本、episode split manifest 和数据顺序；
- raw18 到 model16 schema；
- 图像预处理、语言 task 和 normalization stats；
- optimizer、learning rate、scheduler、batch size、gradient accumulation；
- 总 optimizer steps 和 checkpoint 保存频率；
- dtype、gradient checkpointing 和冻结策略；
- base checkpoint fingerprint；
- 可复现随机种子。

A/B 唯一的训练方法差异：

- A 使用标准 flow matching；
- B 采样 committed prefix，并且只在 postfix 计算 loss。

训练时间、samples/s、峰值显存和 GPU 利用率也要记录，避免忽略 B 的训练成本。

### 5.2 数据划分

不得按 frame 随机切分，因为同一 episode 的相邻帧高度相关。正式实验前创建不可变的 episode split manifest。

建议初始划分：

- train：34 episodes
- validation：4 episodes
- offline test：4 episodes

如 episode 包含不同初始姿态、物体位置或录制批次，应按这些因素分层；不能只按 episode ID 顺序切分。split manifest 一旦生成，在 A/B/C 和所有 seed 间保持不变。

若 42 episodes 的覆盖不足以支持独立 test split，应在正式训练前补录数据，不能通过把 test episode 混回 train 来提高结果。

### 5.3 训练重复

- 正式目标：A 和 B 各训练至少 3 个 seed。
- 单 seed 只作为管线 pilot，不用于宣称方法优劣。
- A/B 必须使用配对 seed，例如 `101/202/303`。
- 每个 seed 保存 base fingerprint、训练配置、最终 step 和 checkpoint fingerprint。

考虑到 PI0.5 训练成本较高，可先用单 seed 完成端到端 pilot；只有 pilot 通过后再启动剩余 seed。

## 6. 训练时 RTC 定义

### 6.1 张量接口

训练时 RTC 至少需要以下张量：

```text
actions:          [B, H, D]
noise:            [B, H, D]
delay:            [B]
prefix_mask:      [B, H]
global_flow_time: [B]
token_flow_time:  [B, H]
loss_mask:        [B, H, D]
```

每个 batch 样本独立采样 delay，不能让整个 batch 共用一个 delay。

### 6.2 训练逻辑

按当前仓库 flow 约定，概念逻辑为：

```python
delay = sample_delay_per_sample(batch_size)
prefix_mask = step_index < delay[:, None]

global_flow_time = sample_flow_time(batch_size)
token_flow_time = where(prefix_mask, 0.0, global_flow_time[:, None])

x_t = (
    token_flow_time[..., None] * noise
    + (1 - token_flow_time[..., None]) * ground_truth_actions
)

pred_velocity = model(
    observation,
    x_t,
    global_flow_time=global_flow_time,
    token_flow_time=token_flow_time,
    prefix_mask=prefix_mask,
)

target_velocity = noise - ground_truth_actions
squared_error = (pred_velocity - target_velocity) ** 2
loss = masked_mean(squared_error, ~prefix_mask)
```

`masked_mean` 必须除以有效 postfix 元素数量。不能把 prefix loss 置零后仍然除以固定的 `B*H*D`，否则 delay 大的样本会被隐式降低权重。

### 6.3 PI0.5 时间条件兼容

当前 PI0.5 的 AdaRMS 使用样本级 timestep `[B]`。不能机械地把它替换为 `[B,H]`。

初始实现采用双时间条件：

- `global_flow_time [B]` 保留现有 AdaRMS 路径；
- `token_flow_time [B,H]` 作为 action-token 局部条件；
- `prefix_mask [B,H]` 显式标识 committed prefix；
- RTC 关闭时必须走原路径，并与当前 checkpoint 行为严格一致。

正式实现前用单元测试验证 shape、广播、checkpoint loading 和 disabled-path parity。

### 6.4 Delay 分布

`max_delay` 不由主观经验指定。先在目标 GPU、PyTorch B1 路径上测量完整链路：

```text
observation capture
image preprocessing
request serialization/transport
prefix cache
10-step denoise loop
response transport
client merge
```

控制周期为：

```text
control_dt = 1 / 20 Hz = 50 ms
```

初始定义：

```text
p95_delay_steps = ceil(P95_end_to_end_latency / control_dt)
max_delay = min(p95_delay_steps, H - min_postfix_steps)
```

训练分布建议由两部分组成：

- 90% 从实测 delay histogram 采样；
- 10% 在 `[0, max_delay]` 均匀采样，补足长尾覆盖；
- 保留非零 `delay=0` 概率，避免模型丧失完整 chunk 生成能力。

必须记录部署中 `delay > max_delay` 的 overflow rate。overflow 不能静默裁剪后不报告。

## 7. 三方法统一推理协议

### 7.1 公共状态

所有方法使用：

- 同一个 observation 及 observation timestamp；
- 同一个原始 noise seed；
- 同一个 action horizon 和 inference steps；
- 同一个 raw18/model16 pre/post processor；
- 同一个异步 producer/consumer；
- 同一个 queue low watermark 和 max size；
- 同一个控制频率；
- 同一个实际 latency 计算方法；
- 同一个 fully-stale chunk 处理策略。

每次请求记录：

```text
observation_sequence_id
observation_timestamp
request_start/ready timestamps
measured_delay_steps
predicted_delay_steps
prefix_len
queue cursor/depth
noise seed
checkpoint fingerprint
method ID
backend ID
```

### 7.2 A：普通异步 PI0.5

```text
1. 使用 observation 生成普通完整 chunk。
2. 根据响应时 observation age 计算真实 delay d。
3. 丢弃已经过时的前 d 个动作。
4. 将剩余动作按统一 merge 规则接入队列。
```

A 不接收 action prefix，不做 guidance。A 仍必须使用 full-chunk 异步队列，不能用当前 single-step 入口代替。

### 7.3 B：训练时 RTC

```text
1. 请求开始时，从已提交旧 chunk 提取 action_prefix。
2. 传入 prefix、prefix_mask 和预计 delay。
3. 每个 Euler step 前恢复 clean prefix。
4. prefix token_time 固定为 0，postfix 使用当前 denoise time。
5. 只更新 postfix。
6. 响应到达后按实测 delay 丢弃已执行部分，仅合并有效 postfix。
```

B 不得调用现有 inference-time `RTCProcessor`，不得产生 VJP 或 guidance backward。

### 7.4 C：训练后 RTC

```text
1. 加载与 A 完全相同的普通 checkpoint。
2. 使用与 B 相同来源的 leftover prefix 和 delay。
3. 每个 denoise step 调用现有 RTC guidance。
4. 响应到达后使用与 A/B 相同的 stale/merge 规则。
```

C 的 `execution_horizon`、guidance weight 和 prefix schedule 作为预注册参数。正式结果不能在 test 或上机结果出来后只保留最优参数。

## 8. 实验矩阵

### 8.1 方法效果主矩阵

| Run ID | Train | Checkpoint | Inference | Backend | 用途 |
| --- | --- | --- | --- | --- | --- |
| A-PT | 普通 | A | async full chunk，无 RTC | PyTorch | 普通基线 |
| B-PT | 训练时 RTC | B | prefix clamp | PyTorch | 目标方法 |
| C-PT | 普通 | A | inference-time RTC/VJP | PyTorch | 训练后 RTC |

### 8.2 推理加速矩阵

| Run ID | Checkpoint | Backend | 输入与 seed | 用途 |
| --- | --- | --- | --- | --- |
| B-PT | B | PyTorch | 固定 | 数值和速度基准 |
| B-TRT | B | TensorRT | 与 B-PT 完全相同 | 加速效果 |

### 8.3 可选消融，不计入三方法主排名

- A-sync：普通 checkpoint，同步 full-chunk，用于估计异步调度收益。
- B-delay0：B checkpoint，但 prefix length 固定为 0，用于检查完整 chunk 能力。
- B-uniform：B 使用 uniform delay 训练，用于比较 empirical delay 分布。
- C-LINEAR/C-EXP：只在 validation 上选择一次 schedule，之后锁定。
- B-compile：B checkpoint + `torch.compile`，作为 PyTorch 与 TensorRT 之间的中间基线。

## 9. 评测阶段

### 9.1 E0：实现契约与单元测试

必须覆盖：

- RTC disabled 时 loss、velocity 和普通推理与原实现一致；
- `delay=0` 退化为普通 flow matching；
- 每个样本独立 delay；
- prefix 为 clean action；
- prefix 不参与 loss；
- 有效 postfix denominator 正确；
- `max_delay < H` 且保留最小 postfix；
- 推理每个 Euler step 后 prefix 完全不变；
- B 路径不触发 autograd/VJP；
- A/B/C 使用同一个 queue merge 实现；
- shape、NaN/Inf、空 postfix 和 fully-stale chunk 均有显式错误或统计。

E0 全部通过之前不得启动大模型正式训练。

### 9.2 E1：离线 paired replay

从 offline test episodes 创建固定 anchor manifest。建议：

- 至少 200 个 observation anchors；
- 覆盖 episode 开始、中段、接触前后和结束段；
- 每个 anchor 使用至少 5 个固定 noise seeds；
- 对所有方法复用相同 observation、prefix 来源、noise 和 delay manifest；
- delay 覆盖 `0..max_delay`，并额外测试 P99 与 overflow delay。

配对 manifest 为每个 case 保存同一个 committed prefix，但消费语义不同：B/C 实际消费该 prefix，A 明确忽略它并只使用相同的 observation、noise 和 delay 条件。A 仍按相同实测 delay 丢弃 stale actions。所谓 paired comparison 是共享同一环境条件和随机条件，不表示三种方法的模型输入张量完全相同。

输出必须包含逐请求 JSONL/Parquet，不能只输出汇总均值。

### 9.3 E2：live-sensor dry-run

该阶段允许读取现场 state/相机和请求 policy server，但 `send_action` 必须保持未调用。

每种方法至少运行：

- 3 次短时 60 秒；
- 1 次持续 10 分钟；
- 相同 sensor/control FPS；
- 尽量相同现场网络和 GPU 负载。

记录完整 observation-to-response 延迟、queue 模拟状态和动作输出。dry-run 只验证系统时序和输出安全性，不代表真实任务成功率。

### 9.4 E3：TensorRT 数值与速度验证

先做 B-PT/B-TRT 离线 paired comparison，再做 dry-run。两者使用同一个 checkpoint、输入、prefix、mask、noise seed 和 Euler steps。

逐步比较：

- prefix cache；
- 每个 denoise step 的 velocity；
- 每个 Euler step 的 actions；
- 最终完整 chunk；
- 有效 postfix；
- prefix immutability。

### 9.5 E4：真实机器人 A/B/C

只有 E0-E3 通过，且得到现场明确授权后才能进入。

推荐正式目标：

- 每种方法每个训练 seed 至少 10 episodes；
- 理想目标为每种方法 20 episodes；
- A/B/C 使用随机区组顺序，例如每个 block 随机排列一次 A/B/C；
- 跨不同时间段重复，记录操作者、初始姿态、物体位置和环境条件；
- 失败、急停和安全拒绝都必须保留，不得从结果中删除。

首次上机不属于正式统计，应按安全验收逐级进行：只读 smoke、dry-run、低频短时、单臂/小范围、完整任务。

## 10. 指标定义

### 10.1 任务指标

- `task_success`：按预先定义的成功条件取 0/1；
- `completion_time_s`：从 episode 开始到成功或超时；
- `timeout_rate`；
- `safety_stop_count`；
- `operator_intervention_count`；
- `collision_or_limit_event_count`。

### 10.2 Chunk 连续性指标

设切换前最后一个实际发送动作为 `a_old`，新 chunk 第一个实际发送动作为 `a_new`：

```text
boundary_action_jump = ||a_new - a_old||_2
```

对关节和夹爪分别报告，并增加：

- 每维最大绝对跳变；
- 一阶差分跳变；
- 二阶差分跳变；
- overlap prefix L2/cosine error；
- postfix 第 1、2、4、8 step continuation error；
- delay 分桶后的上述指标。

所有指标同时在 normalized model16 和 postprocessed raw18 空间报告；真实机器人安全结论以 raw18 为准。

### 10.3 时延与队列指标

- image preprocessing P50/P95/P99；
- prefix-cache P50/P95/P99；
- 单 denoise-step P50/P95/P99；
- 完整 denoise loop P50/P95/P99；
- server model latency P50/P95/P99；
- request round-trip P50/P95/P99；
- observation-to-queue-ready P50/P95/P99；
- measured/predicted delay steps；
- prediction delay error；
- queue depth distribution；
- queue underflow count/rate；
- fully-stale chunk count/rate；
- dropped action count；
- hold/skip/stop count；
- effective action send frequency；
- deadline miss rate。

C 的 VJP/autograd 时间必须计入 denoise loop 和端到端时间，不能只比较模型 forward。

### 10.4 资源指标

- GPU 峰值显存；
- GPU utilization；
- CPU utilization；
- server/client 网络吞吐；
- 可测量时记录 GPU 功耗与单 chunk 能耗；
- 训练 steps/s 和推理 chunks/s。

## 11. 统计方案

### 11.1 预注册主终点

- 方法效果主终点：真实机器人 `task_success`。
- 连续性关键次终点：`boundary_action_jump` 和二阶差分跳变。
- 系统关键次终点：queue underflow rate 和 observation-to-queue-ready P95。
- 加速主终点：B-TRT 相对 B-PT 的完整 denoise loop P95 与端到端 P95。
- 加速非劣终点：B-TRT 与 B-PT 的最终 postfix 数值误差和任务成功率。

### 11.2 汇总与置信区间

- 成功率报告 Wilson 95% CI；
- 连续指标报告 median、mean、P95、P99 和 bootstrap 95% CI；
- 离线 A/B/C 使用 paired bootstrap，配对单位为相同 anchor/noise/delay/prefix-source manifest；A 按协议忽略 prefix，B/C 消费 prefix；
- 上机结果按训练 seed、episode 和实验日期分层报告；
- 多个次终点同时检验时使用 Holm 校正；
- 不只报告“最佳 seed”或“最佳 episode”。

单 seed 或每方法少于 10 个上机 episode 的结果只能标记为 pilot observation，不能写成确定性结论。

## 12. TensorRT 设计与验收

### 12.1 推荐 engine 边界

```text
prefix-cache engine:
  images
  image masks
  language tokens
  language masks
  -> KV cache

denoise-step engine:
  KV cache
  noisy actions [B,H,D]
  global flow time [B]
  token flow time [B,H]
  prefix mask [B,H]
  -> velocity [B,H,D]
```

engine 外保留：

- tokenizer；
- 图像预处理；
- Euler 循环；
- action prefix clamp；
- 动作反归一化和 raw18 展开；
- latency 估计；
- 异步 action queue。

prefix action 本身无需强制作为 engine 输入；若 clamp 在 engine 外完成，engine 只需看到已 clamp 的 noisy actions、token time 和 prefix mask。该边界比把队列逻辑导入 engine 更容易验证和维护。

### 12.2 数值验收

正式阈值在 pilot 后、主实验前锁定。至少报告：

- 每步 velocity 最大绝对误差、RMSE 和 cosine similarity；
- 每步 action 最大绝对误差和 RMSE；
- 最终 postfix 最大绝对误差和 RMSE；
- prefix 最大绝对误差，理论上 clamp 后应为 0；
- chunk boundary 指标差异；
- NaN/Inf 数量。

不能只验证 engine 能运行，也不能只比较最终 action，因为逐步误差可能在 Euler 循环中累积。

### 12.3 性能验收

同时测量：

- 冷启动；
- 充分 warmup 后的稳定状态；
- 单请求 latency；
- 连续请求 P50/P95/P99；
- 完整 client/server 链路；
- queue underflow 和 deadline miss。

只有 engine forward 变快但端到端 P95、delay steps 和 queue underflow 没有改善时，不认定为有效部署加速。

## 13. 阶段验收门槛

### Gate 0：配置与实现

- 所有 E0 测试通过；
- RTC disabled parity 通过；
- A/B/C 统一 runtime 可用；
- 所有配置和 checkpoint fingerprint 可追溯。

### Gate 1：训练 pilot

- A/B 均能完成相同步数；
- 无 NaN/Inf；
- B 的有效 postfix loss denominator 正确；
- B 的 `delay=0` validation 不发生明显退化；
- 保存完整训练配置、日志和 checkpoint。

### Gate 2：离线 paired replay

- 所有方法使用同一 anchor/noise/delay manifest；
- B/C 在目标 delay 区间的连续性指标可量化；
- 不存在 prefix 漂移、shape 错误或未报告 overflow；
- 结果包含逐请求 artifact 和统计脚本输出。

### Gate 3：TensorRT

- B-PT/B-TRT 逐步数值误差满足预注册阈值；
- TensorRT 无 NaN/Inf；
- prefix clamp 语义保持；
- 端到端 P95、delay steps 或 queue underflow 至少一项有实质改善，且其余不恶化。

### Gate 4：dry-run

- `send_action=not-called`；
- 无非有限动作；
- action key、model16/raw18 shape 和 force slots 正确；
- 无未解释的 fully-stale 连续失败；
- summary 和逐步 trace 完整落盘。

### Gate 5：真实机器人

- 已获得现场明确授权；
- 急停和现场监护到位；
- 三重 armed 环境变量只在授权终端设置；
- fresh state、initial delta、per-step delta 和 gripper clamp 保持启用；
- 任意安全拒绝、陈旧状态、越界或通信异常立即停止，不自动重试动作发送。

## 14. Artifact 与命名规范

每次实验使用唯一 `experiment_id`：

```text
YYYYMMDD_pi05_jz_rtc_<method>_<backend>_seed<seed>
```

建议目录：

```text
my_devs/jz_robot_pin_timed/pi05/experiments/<experiment_id>/
  manifest.json
  resolved_config.json
  git_status.txt
  environment.txt
  checkpoint_fingerprint.json
  split_manifest.json
  latency_calibration.json
  requests.jsonl
  actions.parquet
  client_summary.json
  metrics.json
  plots/
  logs/
```

`manifest.json` 至少包含：

- method：`A_baseline`、`B_train_rtc` 或 `C_post_rtc`；
- backend：`pytorch` 或 `tensorrt`；
- checkpoint path、step 和 fingerprint；
- base checkpoint fingerprint；
- dataset/split/schema/stats fingerprint；
- git commit 和 dirty status；
- Conda 环境与关键包版本；
- seed、task、FPS、chunk size、inference steps；
- delay distribution/version；
- RTC training config；
- RTC inference config；
- queue config；
- execution：`offline`、`dry_run` 或 `armed`；
- operator、date、initial-condition ID，仅在上机时填写。

现有 `client_summary.json` 只有聚合计数，不足以复现实验。正式评测必须新增逐请求、逐 chunk 和逐 action trace。

## 15. 实施工作包

### WP1：统一评测基础设施

- 新增 immutable split/anchor/delay manifest；
- 新增统一 full-chunk async runtime；
- 新增 A/B/C method switch；
- 新增逐请求、逐 chunk、逐 action trace；
- 新增 paired metrics 与统计脚本。

### WP2：训练时 RTC

- [x] 增加独立训练配置和 JZ `standard|rtc` 启动开关；
- [x] 实现 per-sample delay、prefix mask、token time 和等样本权重 masked loss；
- [x] 实现模型级 B prefix-clamp Euler 推理接口；
- [x] 完成纯张量 shape、delay、clean prefix、denominator 和 prefix padding 单测；
- [x] 完成单卡 batch1 和四卡每卡 batch8 的两步 GPU smoke；
- [x] 在独立 PyTorch backend 中接入 B prefix 来源并对 overflow fail-closed；
- [ ] 在正式统一 runtime 中补齐逐请求 prefix/overflow trace；
- [ ] 完成大模型 disabled-path parity 和 RTC 新模块逐参数梯度断言；
- [ ] WP2 Gate 通过后再启动正式 B 训练。

### WP3：A/B 训练与离线 A/B/C

- 生成 A/B paired seed checkpoints；
- C 复用 A checkpoint；
- 完成 latency calibration；
- 完成 E1 paired replay；
- 锁定正式 RTC inference 参数和数值阈值。

### WP4：TensorRT

- 导出 prefix-cache 与 denoise-step；
- 构建 B-TRT runtime；
- 完成逐步数值对比；
- 完成 B-PT/B-TRT 性能 benchmark；
- 完成 TensorRT dry-run。

### WP5：现场实验

- 只读 inference smoke；
- A/B/C dry-run；
- 经授权后的低频短时安全验收；
- 随机区组 A/B/C 正式实验；
- 汇总任务、连续性、延迟和安全结果。

## 16. 最终报告必须回答的问题

1. 在相同训练预算下，B 相比 A 是否提高延迟条件下的任务成功率和 chunk 连续性？
2. C 相比 A 的收益是多少，其 VJP 成本是多少？
3. B 相比 C 是否在效果相近时具有更低的端到端 P95 和更少的 queue underflow？
4. B 对训练分布内和分布外 delay 的退化曲线如何？
5. B-TRT 相比 B-PT 的 engine、denoise loop 和完整链路加速比分别是多少？
6. TensorRT 是否保持逐步 velocity、最终 postfix 和 prefix clamp 语义？
7. 离线连续性指标与真实机器人任务成功率是否一致？
8. 所有收益是否跨 seed、episode 和实验日期保持，而不是来自单次偶然结果？

只有上述问题均有可追溯的原始 artifact、统计结果和安全记录时，实验才算完成。
