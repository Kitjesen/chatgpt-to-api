#!/usr/bin/env python3
"""
完整测试 chatgpt-to-api 代理：
1. 健康检查与 models 列表
2. 纯对话（无 tools）非流式 / 流式
3. 带 tools 的请求（模型可选返回 tool_calls）
4. 多轮：assistant + tool_calls + tool 消息
"""
import json
import sys
from pathlib import Path

# 项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx

BASE = "http://127.0.0.1:8100"


def load_api_key() -> str:
    env = Path(__file__).parent / ".env"
    if not env.exists():
        return ""
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("CHATGPT_API_KEY=") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def headers() -> dict:
    key = load_api_key()
    if key:
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    return {"Content-Type": "application/json"}


def test_root():
    print("\n--- 1. GET / 健康检查 ---")
    r = httpx.get(f"{BASE}/", timeout=10)
    assert r.status_code == 200, r.text
    data = r.json()
    print(f"   service: {data.get('service')}, version: {data.get('version')}")
    assert data.get("service") == "ChatGPT-to-API"
    print("   OK")


def test_models():
    print("\n--- 2. GET /v1/models ---")
    r = httpx.get(f"{BASE}/v1/models", headers=headers(), timeout=10)
    assert r.status_code == 200, r.text
    data = r.json()
    ids = [m["id"] for m in data.get("data", [])]
    print(f"   models: {ids}")
    assert "gpt-5.2" in ids
    print("   OK")


def test_chat_no_tools():
    print("\n--- 3. POST /v1/chat/completions 纯对话（非流式）---")
    body = {
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": "Say exactly: Hello from test."}],
        "stream": False,
    }
    r = httpx.post(f"{BASE}/v1/chat/completions", headers=headers(), json=body, timeout=60)
    assert r.status_code == 200, (r.status_code, r.text)
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    print(f"   content: {content[:200]}...")
    assert "Hello" in content or "hello" in content.lower()
    assert "tool_calls" not in data["choices"][0]["message"] or not data["choices"][0]["message"].get("tool_calls")
    print("   OK")


def test_chat_stream_no_tools():
    print("\n--- 4. POST /v1/chat/completions 纯对话（流式）---")
    body = {
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": "Reply with one word: OK"}],
        "stream": True,
    }
    with httpx.stream(
        "POST", f"{BASE}/v1/chat/completions", headers=headers(), json=body, timeout=60
    ) as r:
        assert r.status_code == 200, r.read().decode()
        chunks = []
        for line in r.iter_lines():
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunks.append(json.loads(data_str))
                except json.JSONDecodeError:
                    pass
        content = "".join(
            c.get("choices", [{}])[0].get("delta", {}).get("content") or ""
            for c in chunks
        )
    print(f"   streamed: {repr(content[:100])}")
    assert len(content) >= 1
    print("   OK")


def test_chat_with_tools():
    print("\n--- 5. POST /v1/chat/completions 带 tools（模型可能返回 tool_calls）---")
    body = {
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": "What is 2+2? Reply in one short sentence, no tools."}],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": "auto",
    }
    r = httpx.post(f"{BASE}/v1/chat/completions", headers=headers(), json=body, timeout=60)
    assert r.status_code == 200, (r.status_code, r.text)
    data = r.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    print(f"   content: {content[:150]}...")
    print(f"   tool_calls count: {len(tool_calls)}")
    # 我们只校验请求被接受、有 content 或 tool_calls
    assert content or tool_calls
    print("   OK")


def test_chat_multi_turn_tool_style():
    print("\n--- 6. 多轮：assistant(tool_calls) + tool 消息 ---")
    # 模拟 OpenClaw 发来的第二轮：上轮模型返回了 tool_calls，客户端执行后把结果放在 role=tool
    body = {
        "model": "gpt-5.2",
        "messages": [
            {"role": "user", "content": "Run the command: echo hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "exec", "arguments": '{"command": "echo hello"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc123", "content": "hello\n"},
        ],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "exec",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
    }
    r = httpx.post(f"{BASE}/v1/chat/completions", headers=headers(), json=body, timeout=60)
    # 多轮带 tool 可能返回文本或再次 tool_calls
    assert r.status_code == 200, (r.status_code, r.text)
    data = r.json()
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    tool_calls = msg.get("tool_calls") or []
    print(f"   content: {repr(content[:200])}")
    print(f"   tool_calls: {len(tool_calls)}")
    print("   OK (多轮请求被接受并返回)")


def main():
    print("ChatGPT-to-API 完整测试 (BASE=%s)" % BASE)
    try:
        test_root()
        test_models()
        test_chat_no_tools()
        test_chat_stream_no_tools()
        test_chat_with_tools()
        test_chat_multi_turn_tool_style()
    except httpx.ConnectError as e:
        print("\n[ERROR] 无法连接代理，请先启动: python server.py")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
    print("\n======== 全部通过 ========")


if __name__ == "__main__":
    main()
