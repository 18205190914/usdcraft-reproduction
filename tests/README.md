# 测试入口

默认测试不使用真实模型；`pyproject.toml` 已排除 live 和 volume。

```bash
uv run --locked pytest -q -m 'not live and not volume'
```

存在代理时按主 README 的 `env -u ... NO_PROXY=...` 命令运行，避免 localhost 请求经代理。

| 测试层 | 主要职责 |
| --- | --- |
| SDK / 几何单元测试 | 刚体关节、源观测、全局局部误差、运动、原生收据身份与完整性 |
| `harness.py` warm compile | 持久测试进程中的真实编译与工具行为；不是模型生成 |
| scripted agent | 预录回复驱动工具循环；不调用 API |
| `test_compile.py` | 新解释器编译、进程清理与编译 payload 契约 |
| live | 显式 `-m live`，真实模型调用，需凭证且可能付费 |
| volume | 显式 `-m volume`，独立并发压力测试 |

不要把 mock/native 收据汇总单元测试当成真实 PhysX 求解。真实原生运行在单独的 Windows 环境中，见 `docs/native-physx.md`。同理，SDK 离线示例与合成开发输入不属于论文模型盲测成绩。

CI 安装 CAD dev/sim 依赖后仅运行离线层，构建 wheel，并对安装后的 wheel 进行导入与真实例子编译冒烟；不上传编译产物。`benchmarks/sdk_benchmark.py` 是 SDK 微基准，输出不提交。
