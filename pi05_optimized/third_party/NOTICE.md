# 第三方代码说明

## Realtime-VLA v1

- 仓库：https://github.com/Dexmal/realtime-vla
- 固定 commit：`b86a942a073ea241f9bd6916a705f81906f4638b`
- 许可证：MIT，副本位于 `realtime_vla/LICENSE`
- 引入文件：`realtime_vla/pi0_infer.py` 和 `realtime_vla/pi05_infer.py`

除包内相对导入、Ruff 排除头和 commit 来源注释外，源码保持固定上游实现。上游使用不安全 pickle
格式的转换脚本没有被引入，也没有在本项目中使用。
