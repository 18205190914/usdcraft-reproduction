# Windows 原生 PhysX 验收

## 边界

CAD/智能体在 Linux 或 WSL 中运行。原生验收由独立 **Windows Isaac Sim** 的 Kit + `omni.physx` 执行。仓库不分发 Isaac 包、扩展缓存或驱动，不在 CAD 虚拟环境中安装 Isaac。

已使用的环境组合为 Windows Python 3.12、`isaacsim==6.2.0.0`、`psutil==5.9.8`。原生脚本读取 Kit 所带 USD/PhysX；不要向该环境安装 CAD 锁文件的 `usd-core`。CPU 求解是原生 PhysX；浏览器关节动画和 MuJoCo 则不是原生验收。

## 1. 独立安装（Windows PowerShell）

以下目录是可修改的安装示例。选择你自己的大容量盘；不要覆盖已有环境。

```powershell
$IsaacEnv = 'F:\tools\isaacsim-6.2'
py -3.12 -m venv $IsaacEnv
& "$IsaacEnv\Scripts\python.exe" -m pip install "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com
& "$IsaacEnv\Scripts\python.exe" -m pip install "psutil==5.9.8"
& "$IsaacEnv\Scripts\python.exe" -c "import importlib.metadata as m; print(m.version('isaacsim')); print(m.version('psutil'))"
```

这是针对已用版本的安装命令，不保证未来发行版本可互换。安装前确认 [NVIDIA Python 安装说明](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)、平台要求与许可证。初次启动可能需要联网补充扩展；`[all,extscache]` 很大，安装位置与缓存均需预留空间。

## 2. 生成本机配置（WSL）

把仓库和本轮输出放在共享盘，例如 `/mnt/f/projects/USDCraft-reproduction`。以下 `--windows-env` 对应上一步的实际安装路径；`outputs/` 由 Git 忽略：

```bash
uv run --locked python scripts/configure_native.py \
  --windows-env /mnt/f/tools/isaacsim-6.2 \
  --output outputs/native \
  --portable-root outputs/isaac-portable --accept-eula
```

`--accept-eula` 表示你已阅读并接受 NVIDIA Omniverse 许可；脚本才会在该配置中写入 `OMNI_KIT_ACCEPT_EULA=YES`。省略时不会代替你接受许可。

脚本将四个原生 Python 源文件、Kit 配置和固定 PhysX profile 复制到 `outputs/native/`，生成本机 `native-config.json`，将 Windows 环境和输出路径正确转换。代码库内没有写死旧电脑的盘符、用户名、代理或安装位置。共享盘以外的 `/home/...` 输出会被拒绝。

配置中的 `windows_python` 是从 WSL 执行的 `python.exe` 路径；`isaacsim`、`experience`、`portable_root` 是 Windows 路径；`script_dir` 和 `profile` 是 WSL 可读路径。不要把这两种路径混用。更换原生脚本、profile 或配置后重新生成；身份哈希会阻止不同配置错误复用旧收据。

如本机必须使用代理，可在生成后的 `environment` 显式增加自己的代理设置；同时为 localhost 设置 `NO_PROXY`。这些机器相关设置始终保留在被忽略的输出目录，不写回仓库。

## 3. 纳入重建主流程

```bash
uv run --locked python scripts/prepare_demo.py --output outputs/demo
uv run --locked python -m articraft.usdcraft_session prepare \
  --public outputs/demo/inputs/hinged_box \
  --output outputs/sessions --backend external-agent \
  --geometry-profile configs/usdcraft_geometry_local.json \
  --native-config outputs/native/native-config.json
```

随后按 [running.md](running.md) 编写/提交候选。`ModelingTools.evaluate()` 在编译、几何、局部和运动检查后自动调用原生适配与测试，反馈进入同一会话。不需要人工把“模拟成功”改写进报告。

也可给 `python -m articraft.usdcraft_program` 增加同样的 `--native-config`，由其整段程序循环读取实际反馈。

## 验收与适配方法

`configs/usdcraft_physx.json` 是固定的验收参数，不是启动器路径配置。四个场景 `drop`、`half_open_release`、`drive_cycle`、`limit_load` 各重复三次，执行和汇总完整的 4×3 矩阵。检查地板穿透、关节限位/跟踪、锚点误差和完整性；阈值来自 profile。

`apply_compatibility.py` 为当前验收使用标准刚体关节，明确移除 reduced-articulation 标记，并记录原始/适配资产哈希和允许的修改。闭合正交碰撞网格使用 `orthogonal_compound.py` 的精确盒分解；不支持的曲面保留原生近似并明确记录。视觉几何、质量、材料与关节定义不得静默篡改。

`launch_native.py` 记录真实 Windows PID 与创建时间、日志和超时。主流程按资产/配置/适配器/求解脚本身份复用已经完成的验收；未完成或失败的场景不能包装成通过。

## 当前限制

复杂曲面与多壳网格碰撞仍可能不稳定；几何检查通过不保证物理检查通过。当前方案验证的是标准原生刚体关节适配，不能宣称原始 reduced articulation 的相邻碰撞语义已验证。真实尺度、质量、摩擦和运动范围需要外部依据；未知值只能作为演示假设。

这里的最小 Kit 体验用于无窗口物理验收。RTX 视频/扩展部署不是本仓库的快捷运行路径；交互显示使用 `articraft view`。不要为此文档中的测试擅自升级系统显卡驱动。
