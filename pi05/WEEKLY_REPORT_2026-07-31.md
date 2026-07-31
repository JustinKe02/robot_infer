# PI0.5 RTC 接入工作周报

**统计时间：截至 2026 年 7 月 31 日**

## 一、本周目标

在 `tk_infer` 中为右臂瓶子抓取任务接入独立的 RTC-conditioned 推理后端，验证 20 Hz 连续控制链路，并定位当前 RTC 权重抓取响应慢、抓取不准的问题。

## 二、已完成工作

- 建立 PI0.5 RTC-conditioned 独立 server/client profile，完成三相机通信、机器人状态接收、动作下发及 RTC 连续运行链路验证。
- 将推理任务提示修正为训练时使用的任务：`Put the bottle on the right into the basket on the left.`。
- 完成 connect smoke、只读 inference smoke、single-step armed 和有限动作数 RTC 测试；增强 smoke 产物，保存模型输入、状态、相机帧及完整动作 chunk，便于离线分析。
- 定位 `QUEUE_LOW_WATERMARK=45` 运行失败原因：预测延迟达到 14 步，超过旧权重训练支持的最大 prefix 10，服务端主动返回 HTTP 500；运行参数已回到 watermark 30。
- 使用相同现场帧对普通 PI0.5 和旧 RTC checkpoint 进行离线 A/B：
  - 普通 PI0.5（prefix=0）：20/20 产生右夹爪闭合，闭合步中位数约为 32。
  - 旧 RTC checkpoint（prefix=0）：6/20 产生闭合，闭合率 30%。
  - 旧 RTC checkpoint（prefix=5）：2/20 产生闭合，闭合率 10%。
- 生成新的严格 A/B RTC 训练配置：沿用成功的 Full 15-epoch PI0.5 基线，只增加 training-time clean-prefix RTC，设置 `max_delay=5`、`min_postfix_steps=1` 和 postfix-only loss。

## 三、当前结论

- 原 100 条训练数据具备抓取能力，普通 PI0.5 已稳定产生抓取闭合动作，因此数据和瓶子可见性不是当前主要问题。
- 旧 RTC checkpoint 在 prefix=0 时已明显退化，加入 prefix 后进一步弱化抓取；队列 watermark 只是问题放大器，不是根因。
- 旧 RTC 实验同时改变了相机数量、expert-only、训练轮数和 RTC loss，无法形成严格对照，不能继续作为正式 RTC 权重使用。
- 无需重新采集或修改数据，但真正的 RTC-conditioned backend 需要使用同一数据重新训练一版学会 prefix/delay 条件的新权重。

## 四、下一步计划

1. 在另一台训练主机执行 contract check、preflight 和 2-step capacity smoke，确认无 OOM、NaN/Inf 后启动 Full 15-epoch RTC 训练；当前本机不执行训练。
2. 回传 `005300`、`010600`、`015900` 三个 checkpoint，逐一进行相同现场帧的 prefix=0/prefix=5 离线验收。
3. 建议验收门槛：prefix=0 至少 18/20 产生合理闭合，prefix=5 至少 16/20 产生合理闭合，并检查闭合时序和右臂轨迹是否接近普通 PI0.5。
4. 通过离线验收后接入独立 RTC backend，按 single-step、有限动作数、连续 20 Hz 的顺序逐级上机验证。

## 五、风险与注意事项

- 正式 RTC 训练必须保持 Full 全参数、head+right 相机、100 条数据、15 epochs、batch size 32、seed 1000 和原任务文本不变，避免再次引入多变量差异。
- 新 checkpoint 未通过离线验收前，不直接启动无限连续 armed 控制。
- 当前本机不启动训练，且未经明确授权不操控机器人。
