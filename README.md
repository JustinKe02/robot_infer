# tk_infer

`tk_infer` 是一个独立的推理工作区，不调用或导入 `my_devs/.../infer` 下的旧版私有启动器。

当前工作区仅覆盖 PI0.5，暂不包含 ACT 实时推理。

## 目录结构

- `pi05/`：面向 `jz_robot_pin_timed` 的完整分布式 PI0.5 服务端/客户端运行时，包含单步推理、
  异步单步推理、RTC、checkpoint/schema 校验、HTTP 协议、动作队列、机器人构建器、安全门禁、
  启动脚本和无硬件测试。
- `pi05_optimized/`：在保持基线不变的前提下实现的 PI0.5 优化服务端路径。目前已实现 Phase 0
  精确透传一致性和 Phase 1 有界可观测性；该目录暂不提供机器人客户端或 armed 启动脚本。
- `pi05/profiles/step_010600/`：从旧推理模块设计迁移而来的 epoch-10、head-right、right-arm
  checkpoint 锁定配置。
- `offline_infer.py`：用于检查 checkpoint 的通用只读数据集推理工具。
- `run_infer.sh` 和 `run_local_pi05.sh`：离线数据集启动脚本，不会连接机器人。

PI0.5 实时推理实现只依赖 `src/lerobot` 下的公共核心代码，不依赖旧版项目专用推理运行时。

## PI0.5 使用说明

完整使用说明见：

- [`pi05/README.md`](pi05/README.md)

运行全部无硬件配置检查：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
bash tk_infer/pi05/run_config_checks.sh
```

在指定 Conda 环境中运行测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
conda run -n lerobot_flex pytest -q tk_infer/pi05/tests tk_infer/tests
```

运行优化路径的无硬件检查：

```bash
bash tk_infer/pi05_optimized/run_config_checks.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
conda run -n lerobot_flex python -m pytest -q tk_infer/pi05_optimized/tests
```

## 安全边界

`dry_run`、`health_only`、`config_only` 和 `PRINT_COMMAND_ONLY` 不会发送机器人动作。普通
dry-run 仍可能连接实时状态服务和相机服务。

`run_single_step_armed.sh` 和 `run_rtc_armed.sh` 可以发送 UDP 机器人控制命令。它们要求完成
三项显式确认，并且只能用于经过授权的现场操作。测试和配置检查不会自动执行任何 armed 启动脚本。
