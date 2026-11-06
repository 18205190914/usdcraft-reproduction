# USDCraft Reproduction on Articraft

基于 **Articraft + build123d + OpenUSD** 的关节资产程序化重建与检查流程。目标是把文字、参考图像或已有表面网格转成可编辑的 Python 建模程序和关节化 USD 资产，并利用检查反馈迭代修订。


## 功能与入口

| 功能 | 实现入口 | 边界 |
| --- | --- | --- |
| 文字 / 单张参考图建模 | `articraft generate`、`api.py`、`agent/` | 由模型写完整 Python；图像需选支持图像的适配器 |
| 源表面几何观测 | `source_geometry.py` | 表面支持网格、六向深度、区域/任意法向测量；未观测部分为 UNKNOWN |
| 自由程序编译、SDK 验收 | `compiler/`、`sdk/` | build123d BRep、独立结构检查、关节与质量属性、USDZ 导出 |
| 源/候选几何复查 | `geometry_recheck.py` | 共用网格覆盖 + libigl 连续距离/法向 + 局部区域诊断 |
| 运动检查和反馈修订 | `usdcraft_motion.py`、`usdcraft_workflow.py` | 运动范围扫描、碰撞/干涉检查；失败反馈回到完整程序修订 |
| 外部智能体文件会话 | `usdcraft_session.py` | prepare → measure → submit；不会自动启动 Codex |
| OpenRouter 整段程序后端 | `usdcraft_program.py` | 显式模型选择、额度预检；此适配器不接收图像像素 |
| 原生 PhysX 验收（可选） | `usdcraft_physx.py`、`scripts/isaac/` | 独立 Windows Isaac Sim；不是 MuJoCo 结果替代 |
| 浏览器查看 / MuJoCo 检查 | `viewer.py`、`simulate.py` | 关节显示与交互；MuJoCo 是独立的辅助后端 |

完整架构与调用关系见 [架构说明](docs/architecture.md)，运行细节见 [运行指南](docs/running.md)。

## 目录

```text
.
├── src/articraft/            # SDK、编译器、模型适配器和 USDCraft 工作流
│   ├── agent/               # 模型—工具—编译循环、记录、工作空间
│   ├── compiler/            # 子进程编译与独立验收
│   ├── sdk/                 # 刚体、关节、材料、导出及只读参考文档
│   ├── prompts/             # 建模提示词
│   ├── source_geometry.py   # 源几何观测
│   ├── geometry_recheck.py  # 全局/局部复查
│   └── usdcraft_*.py        # 完整程序、会话、物理与评估接口
├── configs/                 # 固定几何和 PhysX 验收参数；无本机路径
├── scripts/                 # 离线示例、演示输入生成、原生运行时配置
│   └── isaac/               # 原生 Kit/PhysX 源码；不含 Isaac 安装包
├── examples/                # 三个可编译的 Python 源程序
├── tests/                   # 离线测试与显式启用的 live/volume 测试
├── benchmarks/              # SDK 微基准代码；不含测量输出
├── docs/                    # 安装、运行、架构、开发与发布说明
├── typings/                 # 静态类型辅助
├── .github/                 # 离线 CI、Issue/PR 模板
├── .env.example             # 空白凭证模板
├── pyproject.toml           # Python 包和工具配置
└── uv.lock                  # 保留当前依赖锁定；不要随意升级
```

## 1. 安装

推荐 Ubuntu 24.04 / WSL2、Python 3.12 和 uv。`uv.lock` 是 CAD 环境的版本依据；Isaac Sim 使用另一套 Windows 环境。

在仓库根目录执行：

```bash
uv sync --locked --python 3.12 --group dev --group sim
uv run --locked articraft --help
```

`uv` 自动创建项目的 `.venv`。同名 `.venv` 通过完整路径区分，不会合并。可把整个仓库放在其他盘；WSL 想把环境单独放在 Linux 文件系统时，先设置：

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/usdcraft-reproduction"
uv sync --locked --python 3.12 --group dev --group sim
```

系统依赖、锁定版本与安装排错见 [环境说明](docs/installation.md)。

## 2. 先运行不调用模型的示例

```bash
uv run --locked python scripts/run_example.py
```

命令打印实际 `run` 路径和编译 JSON。它编译 `examples/hinged_box/main.py`，验证结构与质量声明并导出资产；不需要 API Key。把输出中的实际路径赋给 `RUN`，即可查看或辅助仿真：

```bash
RUN="这里替换为上一命令打印的 run 路径"
uv run --locked articraft view "$RUN"
uv run --locked articraft simulate "$RUN" --scenario drop --seconds 3
```

另两个源程序：

```bash
uv run --locked python scripts/run_example.py --source examples/storage_box.py
uv run --locked python scripts/run_example.py --source examples/mesh_knob/main.py
```

这些是 SDK 使用示例，不是智能体盲测结果。

## 3. 接入模型生成

```bash
cp .env.example .env
# 在本机编辑 .env，只填写选用服务的凭证和有效模型 ID。
export MODEL_ID="替换为你账户中可用的模型 ID"
uv run --locked articraft generate --provider openrouter --model "$MODEL_ID" --physics \
  --output-dir outputs/generation \
  "Create a 0.30 m wide, 0.20 m deep, 0.15 m tall hollow storage box with one rear-hinged lid."
```

这会发起真实 API 调用。模型是否免费、是否可用、配额多少由服务商决定；仓库不默认声称某个模型免费可用。参考图像应使用支持图像的 OpenAI/Anthropic/Gemini 适配器，详见 [模型配置](docs/providers.md)。

## 4. 从源网格进入重建—检查—修订

先生成本地合成开发输入，或按 [输入格式](docs/running.md#输入格式) 放入自己的网格和图片：

```bash
uv run --locked python scripts/prepare_demo.py --output outputs/demo
uv run --locked python -m articraft.usdcraft_session prepare \
  --public outputs/demo/inputs/hinged_box \
  --output outputs/sessions --backend external-agent \
  --geometry-profile configs/usdcraft_geometry_local.json
```

然后让外部智能体只读取该公开输入目录、打印的会话目录内 `model-request.json`、`SDK_CONTRACT.md` 和检查反馈，编写完整 `main.py` 再提交。**prepare 不会替你调用模型**。具体 measure / submit 命令、自动 OpenRouter 后端和独立评价方法见 [运行指南](docs/running.md)。

原生验收需把输出放在 Windows 与 WSL 共享盘，完成 [PhysX 安装与配置](docs/native-physx.md)，再给 `prepare` 或 `usdcraft_program` 增加 `--native-config`。

## 开发验证

```bash
uv run --locked ruff check src/articraft tests scripts benchmarks
uv run --locked ruff format --check src/articraft tests scripts
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1 \
  uv run --locked pytest -q -m 'not live and not volume'
uv run --locked python scripts/check_repo_hygiene.py
```

默认 pytest 排除 live 和 volume；仓库 CI 不使用模型密钥、不安装 Isaac、不运行付费模型。训练步骤不适用：当前实现使用现成模型 API 和确定性 CAD/几何/物理工具。

底座为 [Articraft](https://github.com/articraftresearch/Articraft)，保留 [Apache 2.0 许可证](LICENSE) 与 [NOTICE](NOTICE)。仓库组织参考 [PyPA sampleproject](https://github.com/pypa/sampleproject) 的 `src/ + tests/` 分离，以及 [Open3D](https://github.com/isl-org/Open3D) 的代码、文档和示例分工。具体职责、发布步骤和源代码版本来源见 [仓库维护说明](docs/repository-layout.md) 和 [CONTRIBUTING.md](CONTRIBUTING.md)。
