# 运行指南

所有命令在仓库根目录运行。路径示例中 `RUN`、`ASSET`、`MODEL_ID` 明确由本机输出/账户替换；不是附带的结果文件。

## A. 离线编译 SDK 源程序

```bash
uv run --locked python scripts/run_example.py --source examples/hinged_box/main.py
```

脚本创建新的运行目录，复制源程序到 `workspace/main.py`，由 `LocalWorkspace.compile_path()` 启动独立编译子进程。结果 JSON 列出真实路径。编译产物在该运行目录的 `result/`，包括 USDZ、部件网格、汇总与验收记录。具体文件名以编译返回的 artifact/manifest 字段为准。

```bash
RUN="替换为脚本打印的 run 路径"
uv run --locked articraft view "$RUN"
uv run --locked articraft simulate "$RUN" --scenario drop --seconds 3
```

## B. 准备公开输入

### 输入格式

准备一个目录，例如 `data/my_object/`：

```text
data/my_object/
├── input.json
├── source.ply
└── reference.png
```

`input.json` 的实际工作流格式：

```json
{
  "source": "source.ply",
  "image": "reference.png",
  "units": "metres",
  "up_axis": "Z",
  "request": "Reconstruct the observed object with separate rigid parts and an appropriate joint."
}
```

工作流用 `trimesh.load_mesh(..., process=False)` 读源表面。坐标应事先统一为米、Z 向上；`units` 目前是元数据，不会自动缩放，`up_axis` 也不会自动旋转。没有真实尺度时明确标注未标定演示，不能据此报告真实毫米精度或物理参数。开放/缺损表面会保留 UNKNOWN 语义。

最小合成开发输入可直接生成：

```bash
uv run --locked python scripts/prepare_demo.py --output outputs/demo
```

生成 `inputs/` 和独立的 `private_truth/`。后者只供生成冻结后的评估，不交给建模智能体。生成器源码本身包含合成几何定义，因此这不是对开发者保密的论文盲测集。缺损样例属于输入构造，不可把诊断探针/人为错误当成自然生成错误修复成果。

## C. 外部智能体文件会话

先建立会话：

```bash
uv run --locked python -m articraft.usdcraft_session prepare \
  --public outputs/demo/inputs/hinged_box \
  --output outputs/sessions --backend external-agent \
  --geometry-profile configs/usdcraft_geometry_local.json
```

把打印的 `run` 路径赋给 `RUN`。准备阶段保存公开请求、观测、源视图和 SDK 契约，不会调用 Codex，也不会自动让任意模型开始工作。

```bash
RUN="替换为 prepare 打印的 run 路径"
uv run --locked python -m articraft.usdcraft_session measure --run "$RUN"
uv run --locked python -m articraft.usdcraft_session measure --run "$RUN" --axis 2 --side 1
uv run --locked python -m articraft.usdcraft_session measure --run "$RUN" --normal '[0,0,1]'
```

可以再加 `--bounds '[[xmin,ymin,zmin],[xmax,ymax,zmax]]'`，其中数值来自本例实际坐标。

建模者只读取公开输入、`model-request.json`、`SDK_CONTRACT.md` 和反馈，先测量再编写完整 `main.py`。源程序必须导出 `object_model` 和 `run_tests()`。把它放到 `outputs/candidate/main.py`，同时写下实际建模说明：

```bash
mkdir -p outputs/candidate
# 此处由建模者编写 outputs/candidate/main.py；不要复制真值实现充当模型结果。
printf '%s\n' '{"plan":"填写实际测量依据与建模/修订说明"}' > outputs/candidate/notes.json
uv run --locked python -m articraft.usdcraft_session submit \
  --run "$RUN" --source outputs/candidate/main.py \
  --notes outputs/candidate/notes.json --backend external-agent
```

读取 `result/attempt-XX/agent-feedback.json`、`feedback.json`、`views/`、`local-views/` 和编译错误，修订整段源程序后重复 submit。相同源码与相同验收配置会复用已完成验收；基础设施错误需先修基础设施，不要改 CAD 来掩盖。`--backend` 是记录标签，不是一个自动模型调用器。

## D. 自动文本程序后端（需要 API）

```bash
export MODEL_ID="替换为当前通过免费模型预检的 OpenRouter 模型 ID"
uv run --locked python -m articraft.usdcraft_program \
  --root outputs/demo --cases hinged_box --max-calls 2 --model "$MODEL_ID" \
  --geometry-profile configs/usdcraft_geometry_local.json
```

案例输入固定从 `ROOT/inputs/CASE/` 读取，运行记录在 `ROOT/runs/`。`--max-calls` 当前允许 1–4；`--resume-run` 限单一案例。它不发送图像像素、不供应真值、不静默回退其他服务。

## E. 原生验收

按 [native-physx.md](native-physx.md) 生成配置，再从 B/C 或 D 加入：

```text
--native-config outputs/native/native-config.json
```

此时会话输出也必须处于共享盘。原生适配和场景模拟自动纳入 submit/evaluate；未提供配置时不能声称 PhysX 已通过。

## F. 独立开发评估

只有合成开发输入包含本地真值时，可以在候选冻结后运行：

```bash
ASSET="替换为候选编译后返回的 USDZ 实际路径"
uv run --locked python -m articraft.usdcraft_stage --root outputs/demo evaluate \
  --case hinged_box --asset "$ASSET" --out outputs/evaluation/hinged_box
```

代码读取 `private_truth/CASE/assembly.json` 和部件网格，计算本地 F1、PC、AE、LE 等指标，同时保存参考坐标系与独立归一化两种结果。此本地协议的匹配/缺失惩罚仍需官方核验，不能直接与论文数值比较。自己的实拍数据若无真值，只能做源表面一致性、结构/运动/物理检查，不能伪造绝对精度。

`usdcraft_stage prepare` 还会运行人为扰动的仪器诊断探针；它们不是新模型生成结果。通常准备开发数据只需 `scripts/prepare_demo.py`。
