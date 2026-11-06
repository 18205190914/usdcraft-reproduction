# 模型服务配置

`src/articraft/settings.py` 定义当前适配器配置。模型 ID 必须来自你所用服务账户，不保证代码里保存的默认 ID 仍可用。

| 服务 | 凭证变量 | 选择方式 | 当前图像接口 |
| --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `--provider openai --model "$MODEL_ID"` | 支持；Responses WebSocket 路径 |
| Anthropic | `ANTHROPIC_API_KEY` | `--provider anthropic --model "$MODEL_ID"` | 支持 |
| Gemini | `GEMINI_API_KEY` | `--provider gemini --model "$MODEL_ID"` | 支持 |
| OpenRouter | `OPENROUTER_API_KEY` | `--provider openrouter --model "$MODEL_ID"` | 当前适配器只支持文本 |

将选用的密钥写入本机 `.env`，或由 shell 提供。不要提交或上传 `.env`。API 按服务商计费；ChatGPT 订阅不由此代码转换为 API 配额。

## 文字生成

```bash
export MODEL_ID="替换为服务商实际支持的模型 ID"
uv run --locked articraft generate --provider openrouter --model "$MODEL_ID" \
  --physics --output-dir outputs/generation "a storage box with a rear-hinged lid"
```

## 参考图片

把你有权使用的图片放入被忽略的 `data/`，然后选择支持图片的适配器：

```bash
uv run --locked articraft generate --provider openai --model "$MODEL_ID" \
  --image data/reference.png --physics --output-dir outputs/image-generation \
  "Reconstruct this object with separate rigid parts and a realistic joint."
```

此命令是原始生成入口；没有自动连接 `usdcraft_session` 的源网格验收。若要同时以源表面为依据修订，请使用 [文件会话](running.md)。不要把“接受图片参数”当成独立图片重建精度已经验证。

## 完整程序 OpenRouter 后端

`python -m articraft.usdcraft_program` 会执行免费模型和额度预检。`--model` 应显式指定当前符合该预检要求的模型 ID；无法通过预检时停止，不自动改用付费模型。代码不保证某个 `:free` 模型永远存在或有额度。

该后端发送 SDK 契约、几何观测和真实反馈，以整段 Python 回复为候选，不调用类别 CAD 模板。它是文本后端，`pixels_sent_to_model` 为 false。文件会话可以由具备图像能力的外部建模者读取源图和局部反馈图，但模型隔离和调用仍由外部编排负责。
