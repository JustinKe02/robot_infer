# 第三方代码说明

## Realtime-VLA v1

- 仓库：https://github.com/Dexmal/realtime-vla
- 固定 commit：`b86a942a073ea241f9bd6916a705f81906f4638b`
- 许可证：MIT，副本位于 `realtime_vla/LICENSE`
- 引入文件：`realtime_vla/pi0_infer.py` 和 `realtime_vla/pi05_infer.py`

除包内相对导入、Ruff 排除头和 commit 来源注释外，源码保持固定上游实现。上游使用不安全 pickle
格式的转换脚本没有被引入，也没有在本项目中使用。

## Realtime-VLA v2

- 仓库：https://github.com/dexmal/realtime-vla-v2
- 固定 commit：`a36d02a7b241de1129af2048e749de58f95ead9c`
- 许可证：MIT，副本位于 `realtime_vla_v2/LICENSE`
- 引入文件：`realtime_vla_v2/pi0_infer.py`、`pi05_infer.py` 和 `pi05rtc_infer.py`

除包内相对导入、Ruff 排除头和来源注释外，`pi05rtc_infer.py` 还加入了本地 PI0.5 训练架构所需的
learned token-flow/prefix conditioning；修改后的哈希记录在 `SOURCE_MANIFEST.json`。上游基于
pickle 的服务、checkpoint 适配器和 AIRBOT 客户端均未引入；本项目只允许经严格清单校验的
safetensors 制品。
