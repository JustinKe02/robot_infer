# PI0.5 优化版步骤 015900 配置

这是完成 15 个 epoch 的 PI0.5 checkpoint 对应的锁定策略服务配置。

```text
策略路径：        tk_infer/pi05/checkpoints/015900/pretrained_model
checkpoint step：15900
配置总步数：      15900
是否完整：        true
相机配置：        head_right
模型边界：        state16/action16
线协议边界：      raw18
chunk 大小：      50
指纹：            9d6d37f6111a034209c9bdc2899423a3258cc35070cb8294194c9c594197b58a
```

启动器拒绝覆盖 `PI05_OPT_POLICY_PATH`。模型加载前会校验固定的 7 个文件、所有 SHA-256、checkpoint
指纹、完成状态、相机配置、schema 和维度。

只校验配置和最终命令，不加载 CUDA：

```bash
CONFIG_ONLY=true \
bash tk_infer/pi05_optimized/profiles/step_015900/run_policy_server.sh

PRINT_COMMAND_ONLY=true \
bash tk_infer/pi05_optimized/profiles/step_015900/run_policy_server.sh
```

只有在 GPU 未被其他 PI0.5 服务占用时才启动优化策略服务。RTC 需要 autograd，因此必须关闭
`torch.inference_mode`：

```bash
PI05_OPT_BACKEND=torch_optimized \
PI05_OPT_TORCH_INFERENCE_MODE=false \
PI05_OPT_TRAJECTORY_PROCESSOR=pass_through \
bash tk_infer/pi05_optimized/profiles/step_015900/run_policy_server.sh
```

默认服务地址为 `http://127.0.0.1:18088`。真机命令见 [真机运行手册](../../REAL_ROBOT_RUNBOOK.md)。
