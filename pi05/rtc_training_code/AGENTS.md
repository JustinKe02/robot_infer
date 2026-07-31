# Repository Guidelines

本文件用于约定本仓库的目录组织、开发命令与贡献流程，方便快速上手并保持一致性。

## 项目结构与模块组织

- `src/lerobot/`: 主 Python 包（configs, datasets, envs, robots, policies, training, utils 等）。
- `tests/`: `pytest` 测试与共享 fixtures；大体积测试工件在 `tests/artifacts/`（Git LFS）。
- `examples/`, `benchmarks/`: 示例脚本与基准测试工具。
- `docs/`: 文档源码在 `docs/source/`（Markdown/MDX）。
- `docker/`, `media/`: Docker 文件与文档/README 使用的静态资源。

## 构建、测试与开发命令

- 开发安装（含开发与测试依赖）：`python -m pip install -e ".[dev,test]"`（需 Python >= 3.10）。
- 可选（使用 `uv` 时）：创建 `.venv` 后执行 `uv pip install -e ".[dev,test]"`（`Makefile` 会优先使用 `.venv/bin/python`）。
- 安装提交前钩子：`pre-commit install`
- 运行格式化/静态检查/安全扫描：`pre-commit run -a`
- 运行单测：`pytest -sv tests`
- 只跑部分用例：`pytest -q tests/ -k <keyword>`
- 端到端冒烟：`make test-end-to-end DEVICE=cpu`（产物写入 `tests/outputs/`）。

## 代码风格与命名约定

- 格式化与 lint 使用 Ruff（`ruff format`、`ruff check`），通过 `pre-commit` 统一执行；配置在 `pyproject.toml`。
- 本地直跑 Ruff：`ruff format .`，`ruff check . --fix`（优先仍用 `pre-commit run -a` 保持一致）。
- 约定：4 空格缩进、行宽 110、双引号；import 排序由 Ruff 处理。
- 命名：函数/变量用 `snake_case`，类用 `PascalCase`；测试按 `test_*.py` 命名。

## 测试指南

- 优先写快速、可重复的测试；可选依赖用 `pytest.importorskip(...)` 保护。
- 部分测试依赖 LFS 工件：先执行 `git lfs install && git lfs pull` 再运行 `pytest`。

## 提交与 PR 指南

- 当前仓库 git 历史较少；既有提交多为简短祈使句（如 "Add ...", "Initial commit"）。
- 建议采用 PR 模板风格：`type(scope): summary`（例：`fix(robots): handle None in sensor parser`）。
- PR 按 `.github/PULL_REQUEST_TEMPLATE.md`：说明动机、具体改动、如何测试（命令/输出），并在适用时关联 Issue。

## 文档开发（可选）

- 安装文档依赖：`pip install -e . -r docs-requirements.txt`（需要 `nodejs`）。
- 生成文档：`doc-builder build lerobot docs/source/ --build_dir ~/tmp/test-build`
- 本地预览：`doc-builder preview lerobot docs/source/`

## 安全与配置提示

- 不要提交密钥；`pre-commit` 中包含 `gitleaks`。令牌通过环境变量提供（如 `HF_TOKEN`、`WANDB_API_KEY`）。
- 大文件避免直接进入 git 历史；需要时使用 Git LFS 存放工件。

## 开发规范
## 开发规范
lerobot 必须使用这个conda环境开发代码 任何代码的测试运行都必须使用这个环境。
没有我的允许，不要随便操控机器人