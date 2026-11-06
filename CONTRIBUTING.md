# 开发约定

先阅读 README、`docs/architecture.md`、`docs/running.md` 和 AGENTS.md。保持 `articraft` 公共导入、SDK 与程序编译契约稳定。

修改应简洁、直接；优先修复已有模块，不为单次实验增加框架。新增依赖必须有实际用途，不能为了重新整理升级锁文件。算法变化应注明哪些是已实现行为，哪些仍是方案。

提交前：

```bash
uv sync --locked --group dev --group sim
uv run --locked ruff check src/articraft tests scripts benchmarks
uv run --locked ruff format --check src/articraft tests scripts
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1 \
  uv run --locked pytest -q -m 'not live and not volume'
uv run --locked python scripts/check_repo_hygiene.py
```

不要在普通测试和 CI 中请求真实模型。`-m live` 明确启用真实 API 测试，会产生服务商费用；volume 是单独的并发压力测试。原生物理验收需独立安装和显式配置，不属于普通离线单元测试。

不提交 `.env`、本机路径配置、输入数据、权重、实验结果、USD/视频、环境和缓存。请把运行产生的文件放在 `outputs/`。保留 LICENSE、NOTICE 与 SDK 文档来源说明。
