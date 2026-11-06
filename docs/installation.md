# 环境安装

## CAD 与智能体环境

推荐 Linux / Ubuntu 24.04 WSL2、Python 3.12。包声明兼容 Python >=3.11，但本地整理验收使用 3.12。保留 `uv.lock`，不要根据网上教程先升级 numpy、OCP 或 USD。

已有 uv 时，在仓库根目录执行：

```bash
uv sync --locked --python 3.12 --group dev --group sim
uv run --locked python -c "import sys; print(sys.executable); import build123d, trimesh; from pxr import Usd; print(Usd.GetVersion())"
```

还未安装 uv，可按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装，再执行上面的命令。已安装 Python 的环境也可用 `python3 -m pip install uv` 安装 uv。网络只用于安装依赖，离线 CAD 示例和默认测试不调用模型服务。

Ubuntu 如出现 OpenGL / glib 动态库缺失，先安装这两个系统库：

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

依赖来自 `pyproject.toml` 与 `uv.lock`。整理时实际 CAD 环境包括 Python 3.12.3、build123d 0.11.1、numpy 2.5.1、scipy 1.18.0、trimesh 4.12.2、libigl 2.6.2、usd-core 26.5、MuJoCo 3.11.0、pytest 9.1.1；完整的平台解析以锁文件为准。这是环境记录，不是建议强行覆盖锁定解析。

## 环境位置与磁盘

普通安装生成当前项目的 `.venv`。另一个目录的 `.venv` 是独立环境。uv 会管理激活，不必手工 `source`：

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/usdcraft-reproduction"
uv sync --locked --python 3.12 --group dev --group sim
uv run --locked python -c "import sys; print(sys.executable)"
```

`$HOME` 所在的 WSL 虚拟磁盘位置由你的 WSL 部署决定。把仓库放在 `/mnt/f/...` 不会自动迁移 WSL 虚拟磁盘。不要直接复制旧 `.venv` 来迁移电脑；重新 sync。

## 模型与仿真

离线 SDK 示例不需要模型权重、PyTorch 或 API Key。真实建模需要可用的服务商 API，见 [providers.md](providers.md)。

`sim` 组只安装 MuJoCo 辅助仿真。原生 PhysX 需要独立 Windows Isaac 环境，见 [native-physx.md](native-physx.md)。不要把 Isaac Sim 的 USD / numpy 装入 CAD `.venv`，也不要为运行 CAD 示例安装 Isaac。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `.env` 打开是空白 | 确认在仓库根目录，先 `cp .env.example .env`；不要把文件保存成 `.env.txt` |
| API 握手超时 | 先检查 WSL 网络和本机代理；这不表示 CAD 依赖有问题 |
| localhost 测试返回 502 | 按 README 的 `env -u ... NO_PROXY=...` 命令运行，避免 localhost 经过代理 |
| OpenRouter 拒绝图像 | 当前适配器只支持文本；使用其他图像适配器或外部智能体会话 |
| 没有结果文件 | 仓库本来只含源码；运行 `scripts/run_example.py` 生成自己的文件 |
| 原生验收拒绝 Linux 路径 | 会话与原生输出必须在 `/mnt/<盘符>/...`，例如共享盘，而不是 `/home/...` |
| 包依赖版本冲突 | 重新从锁文件创建 CAD 环境；Isaac 与 CAD 保持分离 |
