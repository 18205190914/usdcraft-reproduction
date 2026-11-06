# 仓库组织与发布

## 组织原则

采用 [PyPA sampleproject](https://github.com/pypa/sampleproject) 的 `src/` 包、`tests/` 和包配置分离；参考 [Open3D](https://github.com/isl-org/Open3D) 将文档、示例和实现分别组织。没有复制这些项目的代码，也没有为了外观改变既有 SDK 导入路径。

- `src/articraft/` 是运行实现；公共导入与 CLI 保留兼容。
- `configs/` 是无机器路径的验收配置；本机 Isaac 路径由脚本写入被忽略的输出目录。
- `examples/` 只含 Python 源码；二进制运行产物由用户生成。
- `scripts/` 是薄运行入口；核心几何和验收逻辑仍位于包内。
- `tests/` 包含确定性回归和显式 live 测试；默认与 CI 均排除 live/volume。
- `docs/` 说明如何安装、生成、检查、修订、评价和维护。

## 源码版本与归属

这是已有 Articraft ZIP 工作副本的独立整理版本，保留其中实际存在的 USDCraft 修改。原副本没有 `.git`，因此没有可保留的提交历史；精确上游 commit 待确认。不能声称这是某个官方 commit 的干净 fork。

保留上游 Apache 2.0 LICENSE、包作者信息和 vendored 来源说明；新增内容与复现性质见 NOTICE。`pyproject.toml` 仍保留 `articraft` 分发/导入名供本地安装，不应直接向 PyPI 发布并占用上游包名。自己的 GitHub 仓库名称可独立命名。

## 什么不进入仓库

不复制旧 `runs/`、`work/`、实验/audit/jobs/evaluation 目录、USD/PLY/视频/结果图、报告、数据、权重、安装包、环境和密钥。SDK 内原本随代码分发的 build123d 参考插图、样例 STEP/STL 是只读文档依赖，并保留其来源说明；它们不是本项目生成结果。

旧电脑的临时渲染/汇报材料、已部署 Isaac 环境与旧本机配置也不作为运行依赖。新环境用 `configure_native.py` 重新生成路径配置。

## 上传 GitHub

仓库可以在此目录直接由 Git 管理；不要将存放其他项目的父目录整体上传。若你解压的是不含 `.git` 的源码 ZIP，先初始化：

```bash
git init -b main
git add .
uv run --locked python scripts/check_repo_hygiene.py
git status --short
git diff --cached --stat
git commit -m "Organize source-only USDCraft reproduction on Articraft"
```

在 GitHub 创建自己的空仓库后，复制 GitHub 页面提供的真实 HTTPS/SSH 地址：

```bash
# 将占位文字替换为你刚创建仓库的真实地址；此处不预设你的账号或仓库名。
git remote add origin "替换为你的实际 GitHub 仓库地址"
git push -u origin main
```

初始化仓库但尚未提交时，`git status` 显示新增文件是正常的。上传前再次运行 hygiene。它检查 Git 已跟踪及未被忽略的新文件，拒绝环境、结果与常见密钥模式；该静态检查不能代替人工确认文件访问权。

`.github/workflows/ci.yml` 仅运行离线测试、格式与 lint、源码卫生检查及 wheel 构建/导入/编译冒烟。没有自动调用付费模型的定时任务，没有 PyPI 发布工作流，也没有旧项目的 CODEOWNERS。

仓库 ZIP 仅用于传输源码，不包含 `.git`、本地输出、测试日志和构建结果。验证日志保存在仓库外的整理工作目录，避免混入源码发布。
