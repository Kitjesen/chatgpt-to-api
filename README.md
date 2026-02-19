# ChatGPT-to-API

将 ChatGPT Plus 网页版转为 **OpenAI 兼容 API**，支持 GPT-5 / GPT-5.1 / GPT-5.2 等最新模型。

核心思路：通过 ChatGPT 的 **Codex Responses API** (`/backend-api/codex/responses`) 端点绕过 Cloudflare Turnstile 人机验证，再用 `curl_cffi` 模拟 Chrome TLS 指纹完成请求。对外暴露标准的 `/v1/chat/completions` 接口，可直接对接任何 OpenAI 兼容客户端。

**v5.0 新增**：Vision（图片输入）、JSON Mode / Structured Outputs、OpenAI 标准错误格式、完整参数支持。

## 背景与发现过程

### 问题

ChatGPT 网页版的传统对话接口 `/backend-api/conversation` 在 2025 年初加上了 **Cloudflare Turnstile** 保护。即使携带有效的 Access Token，直接请求也会返回 `403 Forbidden`（"Unusual activity has been detected"），导致几乎所有 chatgpt-to-api 类项目失效。

### 突破口

通过调研 GitHub 上的 [Securiteru/codex-openai-proxy](https://github.com/Securiteru/codex-openai-proxy) 等项目，发现 ChatGPT 的 **Codex CLI** 使用了另一个端点：

```
POST https://chatgpt.com/backend-api/codex/responses
```

该端点：
- **不受 Cloudflare Turnstile 保护**（返回 400 而非 403，说明请求已经到达后端）
- 使用 OpenAI Responses API 格式（而非传统 Chat Completions 格式）
- 需要特殊请求头：`OpenAI-Beta: responses=experimental` 和 `originator: codex_cli_rs`
- **必须 `stream: true`**（`stream: false` 会返回 400）

### 模型发现

通过系统测试，确认了以下模型在 Codex 端点上可用：

| 请求模型名  | 解析为                     | 说明           |
|-------------|---------------------------|----------------|
| `gpt-5.2`   | `gpt-5.2-2025-12-11`     | 最新版本       |
| `gpt-5.1`   | `gpt-5.1-2025-11-13`     | 次新版本       |
| `gpt-5`     | `gpt-5-2025-08-07`       | 基础版本       |

注意：模型名中使用**点号**（`gpt-5.2`），用短横线（`gpt-5-2`）则不被识别。

## 架构

```
客户端 (OpenClaw / Cursor / 任意 OpenAI 兼容客户端)
    │
    │  POST /v1/chat/completions (OpenAI 格式)
    ▼
┌──────────────────────────────────────┐
│  FastAPI 代理服务器 (server.py)       │
│  - API Key 鉴权                      │
│  - 消息格式转换 (含 Vision)           │
│  - Tool Calling / JSON Mode          │
│  - 流式/非流式响应                    │
│  - 模型名映射                         │
│  - OpenAI 标准错误格式                │
└──────────────────────────────────────┘
    │
    │  curl_cffi (Chrome TLS 指纹)
    │  POST /backend-api/codex/responses
    ▼
┌──────────────────────────────────────┐
│  ChatGPT 后端 (chatgpt.com)          │
│  Codex Responses API                 │
│  绕过 Cloudflare Turnstile           │
└──────────────────────────────────────┘
```

### 关键技术点

1. **`curl_cffi`**：模拟 Chrome 131 的 TLS 指纹（JA3/JA4），避免 Cloudflare 的 TLS 指纹检测
2. **消息格式转换**：OpenAI Chat Completions 的 `messages` 数组 → Codex Responses API 的 `instructions` + `input` 格式
3. **多模态支持**：
   - `text` → `input_text` / `output_text`
   - `image_url`（URL 或 base64）→ `input_image`
4. **角色映射**：
   - `system` / `developer` → `instructions` 字段
   - `user` → `input_text` / `input_image` 类型
   - `assistant` → `output_text` 类型
   - `tool` / `function` 等不支持的角色 → 折叠为 `user`
5. **Token 自动刷新**：使用 Session Token 在 Access Token 过期前自动刷新，并持久化到 `.env`

## 快速开始

### 1. 安装依赖

```bash
cd chatgpt-to-api
pip install -r requirements.txt
```

### 2. 配置 Token

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 Token（任选一种方式）：

**方式 A：Session Token（推荐，支持自动刷新）**

1. 浏览器登录 https://chatgpt.com
2. F12 打开开发者工具 → Application → Cookies → `chatgpt.com`
3. 找到 `__Secure-next-auth.session-token` 的值
4. 填入 `.env`：

```env
CHATGPT_SESSION_TOKEN=你的session_token
```

**方式 B：Access Token（临时使用，约 1 小时过期）**

1. 登录后访问 https://chatgpt.com/api/auth/session
2. 复制 JSON 中 `accessToken` 字段
3. 填入 `.env`：

```env
CHATGPT_ACCESS_TOKEN=你的access_token
```

**方式 C：使用辅助工具**

```bash
python token_fetcher.py
```

交互式引导获取并保存 Token。

### 3. 可选：设置 API Key

为代理服务器设置密钥，防止被未授权访问：

```env
CHATGPT_API_KEY=sk-your-custom-key-here
```

设置后，所有 API 请求需要携带 `Authorization: Bearer sk-your-custom-key-here`。

### 4. 启动服务

```bash
python server.py
```

服务默认监听 `0.0.0.0:8100`。

### 5. 测试

```bash
# 基本文本对话
curl http://127.0.0.1:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5.2", "messages": [{"role": "user", "content": "Hello!"}]}'

# Vision（图片输入）
curl http://127.0.0.1:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.2",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "这张图里有什么？"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
      ]
    }]
  }'

# JSON Mode
curl http://127.0.0.1:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.2",
    "messages": [{"role": "user", "content": "列出三种编程语言，以 JSON 数组返回"}],
    "response_format": {"type": "json_object"}
  }'
```

## API 文档

### `GET /v1/models`

列出可用模型。返回格式与 OpenAI 一致。

### `GET /v1/models/{model_id}`

获取单个模型信息。不存在时返回 404。

### `POST /v1/chat/completions`

OpenAI 兼容的 Chat Completions 接口。

**请求体参数：**

| 参数 | 类型 | 说明 | 状态 |
|------|------|------|------|
| `model` | string | 模型名称 | 转发 |
| `messages` | array | 消息列表，支持文本和图片内容 | 转发 |
| `stream` | boolean | 是否流式返回 | 转发 |
| `temperature` | number | 采样温度 | 转发 |
| `tools` | array | 工具定义列表 | 转发 |
| `tool_choice` | string/object | 工具选择策略 | 转发 |
| `response_format` | object | 输出格式 (`json_object` / `json_schema`) | 转发 |
| `reasoning_effort` | string | 推理力度 (`low`/`medium`/`high`) | 转发 |
| `stop` | string/array | 停止序列 | 转发 |
| `seed` | integer | 确定性采样种子 | 转发 |
| `stream_options` | object | 流式选项 (`include_usage`) | 支持 |
| `max_tokens` | integer | 最大 token 数 | 接受，不转发 |
| `max_completion_tokens` | integer | 最大完成 token 数 | 接受，不转发 |
| `top_p` | number | nucleus 采样 | 接受，不转发 |
| `frequency_penalty` | number | 频率惩罚 | 接受，不转发 |
| `presence_penalty` | number | 存在惩罚 | 接受，不转发 |
| `n` | integer | 候选数量（仅支持 1） | 校验 |
| `logprobs` | boolean | 返回 log 概率 | 接受，不支持 |
| `parallel_tool_calls` | boolean | 并行工具调用 | 接受 |

**消息格式（支持多模态）：**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
      ]
    }
  ]
}
```

图片支持两种方式：
- **URL**：`{"url": "https://example.com/image.jpg"}`
- **Base64**：`{"url": "data:image/jpeg;base64,..."}`

可选 `detail` 参数控制图片精度：`"auto"` / `"low"` / `"high"`

**支持的模型名：**

| 模型名              | 映射到       |
|---------------------|-------------|
| `gpt-5.2`          | `gpt-5.2`  |
| `gpt-5.1`          | `gpt-5.1`  |
| `gpt-5`            | `gpt-5`    |
| `gpt-4o`           | `gpt-5.2`  |
| `gpt-4o-mini`      | `gpt-5.2`  |
| `gpt-4`            | `gpt-5.2`  |
| `gpt-3.5-turbo`    | `gpt-5.2`  |
| `auto`             | `gpt-5.2`  |

未识别的模型名默认映射到 `gpt-5.2`。

**错误响应格式（与 OpenAI 一致）：**

```json
{
  "error": {
    "message": "Invalid API key",
    "type": "authentication_error",
    "param": null,
    "code": null
  }
}
```

### `GET /admin/status`

查看服务状态和 Token 信息。

### `POST /admin/token`

动态更新 Token（无需重启服务）：

```json
{"access_token": "..."}
```

或

```json
{"session_token": "..."}
```

### `POST /admin/refresh`

手动触发 Token 刷新（需要已配置 Session Token）。

## 客户端集成示例

### Python (openai 库)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8100/v1",
    api_key="sk-your-custom-key-here",  # 如果设置了 API Key
)

# 文本对话
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)

# Vision（图片分析）
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "这是什么？"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ],
    }],
)
print(response.choices[0].message.content)

# JSON Mode
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "列出三种水果"}],
    response_format={"type": "json_object"},
)
print(response.choices[0].message.content)
```

### OpenClaw

在 `~/.openclaw/openclaw.json` 中添加 provider：

```json
{
  "models": {
    "providers": {
      "chatgpt-proxy": {
        "baseUrl": "http://127.0.0.1:8100/v1",
        "apiKey": "sk-your-custom-key-here",
        "api": "openai-completions",
        "models": {
          "gpt-5.2": { "name": "GPT-5.2", "reasoning": true, "input": 0.0, "contextWindow": 128000, "maxTokens": 16384 },
          "gpt-5.1": { "name": "GPT-5.1", "reasoning": true, "input": 0.0, "contextWindow": 128000, "maxTokens": 16384 },
          "gpt-5": { "name": "GPT-5", "reasoning": true, "input": 0.0, "contextWindow": 128000, "maxTokens": 16384 }
        }
      }
    }
  }
}
```

然后设置 agent 默认模型为 `chatgpt-proxy/gpt-5.2`。

### Cursor / 任意 OpenAI 兼容客户端

Base URL：`http://127.0.0.1:8100/v1`  
API Key：你在 `.env` 中设置的 `CHATGPT_API_KEY`

## 配置项

所有配置通过环境变量或 `.env` 文件设置，统一前缀 `CHATGPT_`：

| 变量名                    | 默认值                              | 说明                          |
|--------------------------|-------------------------------------|-------------------------------|
| `CHATGPT_ACCESS_TOKEN`   | -                                   | ChatGPT Access Token          |
| `CHATGPT_SESSION_TOKEN`  | -                                   | 用于自动刷新的 Session Token    |
| `CHATGPT_API_KEY`        | -                                   | 代理服务器的 API 密钥           |
| `CHATGPT_HOST`           | `0.0.0.0`                           | 监听地址                       |
| `CHATGPT_PORT`           | `8100`                              | 监听端口                       |
| `CHATGPT_BASE_URL`       | `https://chatgpt.com/backend-api`   | ChatGPT 后端地址               |
| `CHATGPT_PROXY`          | `http://127.0.0.1:7890`            | HTTP 代理（用于科学上网）        |

## 项目结构

```
chatgpt-to-api/
├── server.py            # FastAPI 服务器，对外暴露 OpenAI 兼容 API
├── chatgpt_client.py    # Codex Responses API 客户端，核心请求逻辑
├── config.py            # 配置管理 (pydantic-settings)
├── token_fetcher.py     # Token 获取/验证辅助工具
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量模板
└── README.md
```

## 常见问题

### Q: 为什么不用 `/backend-api/conversation`？

该端点受 Cloudflare Turnstile 保护，无论 TLS 指纹模拟得多好，都会返回 403。Turnstile 在浏览器端运行 JavaScript 挑战，纯后端请求无法通过。

### Q: `curl_cffi` 是必须的吗？

是的。ChatGPT 后端通过 TLS 指纹（JA3/JA4）识别客户端类型。标准 Python HTTP 库（`requests`, `httpx`）的 TLS 指纹与浏览器不同，会被 Cloudflare 拦截。`curl_cffi` 可以模拟 Chrome 的 TLS 指纹。

### Q: 代理（PROXY）是必须的吗？

如果你在中国大陆，需要配置 HTTP 代理才能访问 `chatgpt.com`。默认使用 `http://127.0.0.1:7890`（Clash 默认端口）。在海外服务器上可以置空：`CHATGPT_PROXY=`。

### Q: Token 多久过期？

Access Token 约 1 小时过期。如果配置了 Session Token，服务会在过期前自动刷新。Session Token 本身有效期较长（约 30 天）。

### Q: 支持哪些模型？

目前确认可用的有 `gpt-5`、`gpt-5.1`、`gpt-5.2`。传入旧模型名（如 `gpt-4o`、`gpt-4`）会自动映射到 `gpt-5.2`。

### Q: 支持 Vision（图片输入）吗？

**支持。** v5.0 起，用户消息的 `content` 可以是 `[{type: "text", ...}, {type: "image_url", ...}]` 格式的数组。代理会自动将 `image_url` 转换为 Codex API 的 `input_image` 格式。支持 URL 和 base64 两种图片传入方式。

### Q: 支持 Tool/Function Calling 吗？

**支持。** 代理会：

- 接收并转发客户端的 `tools`、`tool_choice` 到 Codex Responses API（仅转换 `type: "function"` 的工具定义）。
- 将多轮中的 `assistant`（含 `tool_calls`）+ `tool` 消息转换为 Codex 的 `function_call` / `function_call_output` 输入。
- 解析流式事件 `response.output_item.added`、`response.function_call_arguments.delta`/`.done` 以及 `response.completed` 中的 `output`，组装成 OpenAI 格式的 `tool_calls` 并返回（流式与非流式均支持）。

### Q: 支持 JSON Mode / Structured Outputs 吗？

**支持。** 传入 `response_format: {"type": "json_object"}` 启用 JSON 模式，或传入 `response_format: {"type": "json_schema", "json_schema": {...}}` 启用 Structured Outputs。

## 免责声明

本项目仅供学习和研究用途。使用者需拥有有效的 ChatGPT Plus 订阅。请遵守 OpenAI 的使用条款。

## License

MIT
