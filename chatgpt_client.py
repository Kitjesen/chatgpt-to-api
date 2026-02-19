"""
ChatGPT Codex Responses API Client
- 使用 /backend-api/codex/responses 端点 (绕过 Turnstile)
- curl_cffi + Chrome TLS 指纹
- Token 自动刷新 + 持久化到 .env
"""

import uuid
import json
import time
import asyncio
import logging
from typing import AsyncGenerator, Optional
from pathlib import Path

import httpx

try:
    from curl_cffi.requests import Session as CurlSession
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

from config import settings

logger = logging.getLogger("chatgpt-client")

DEVICE_ID = str(uuid.uuid4())
IMPERSONATE = "chrome131"
ENV_PATH = Path(__file__).parent / ".env"

CODEX_HEADERS = {
    "accept": "text/event-stream",
    "content-type": "application/json",
    "origin": "https://chatgpt.com",
    "referer": "https://chatgpt.com/",
    "oai-language": "en-US",
    "OpenAI-Beta": "responses=experimental",
    "originator": "codex_cli_rs",
    "sec-ch-ua": '"Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}

CODEX_URL = f"{settings.chatgpt_base_url}/codex/responses"


def _persist_env(key: str, value: str):
    try:
        content = ""
        if ENV_PATH.exists():
            content = ENV_PATH.read_text(encoding="utf-8")

        if f"{key}=" in content:
            lines = content.split("\n")
            new_lines = []
            for line in lines:
                if line.strip().startswith(f"{key}="):
                    new_lines.append(f"{key}={value}")
                else:
                    new_lines.append(line)
            ENV_PATH.write_text("\n".join(new_lines), encoding="utf-8")
        else:
            with open(ENV_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n{key}={value}\n")

        logger.info(f"已持久化 {key} 到 .env")
    except Exception as e:
        logger.warning(f"持久化 {key} 失败: {e}")


# ──────────── Token Manager ────────────

class TokenManager:
    def __init__(self):
        self._access_token: str = settings.access_token
        self._session_token: str = settings.session_token
        self._token_expiry: float = 0
        self._lock = asyncio.Lock()
        self._refresh_count = 0

        if self._access_token:
            self._parse_expiry(self._access_token)

    def _parse_expiry(self, token: str):
        try:
            import base64
            parts = token.split(".")
            if len(parts) >= 2:
                payload = parts[1]
                payload += "=" * (4 - len(payload) % 4)
                data = json.loads(base64.urlsafe_b64decode(payload))
                self._token_expiry = data.get("exp", 0)
                remaining = self._token_expiry - time.time()
                logger.info(f"Token 有效期剩余: {remaining/60:.1f} 分钟")
        except Exception as e:
            logger.warning(f"无法解析 token 过期时间: {e}")
            self._token_expiry = time.time() + 3600

    def is_expired(self) -> bool:
        if not self._access_token:
            return True
        return time.time() >= (self._token_expiry - 60)

    def set_access_token(self, token: str):
        self._access_token = token
        self._parse_expiry(token)
        settings.access_token = token
        _persist_env("CHATGPT_ACCESS_TOKEN", token)

    def set_session_token(self, token: str):
        self._session_token = token
        settings.session_token = token
        _persist_env("CHATGPT_SESSION_TOKEN", token)

    async def get_access_token(self) -> str:
        if not self.is_expired() and self._access_token:
            return self._access_token

        if not self._session_token:
            if self._access_token:
                logger.warning("access_token 已过期，但没有 session_token 无法自动刷新")
                return self._access_token
            raise RuntimeError(
                "access_token 已过期且无 session_token。"
                "请设置 CHATGPT_SESSION_TOKEN 以启用自动刷新"
            )

        async with self._lock:
            if not self.is_expired() and self._access_token:
                return self._access_token
            await self._refresh()
            return self._access_token

    async def _refresh(self):
        logger.info("正在刷新 access_token (via httpx)...")
        cookies = {"__Secure-next-auth.session-token": self._session_token}
        headers = {
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    settings.chatgpt_auth_url,
                    cookies=cookies,
                    headers=headers,
                )

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"刷新 token 失败: HTTP {resp.status_code} - {resp.text[:300]}"
                    )

                data = resp.json()
                new_access_token = data.get("accessToken")
                if not new_access_token:
                    raise RuntimeError(f"响应中无 accessToken: {json.dumps(data)[:200]}")

                self._access_token = new_access_token
                self._parse_expiry(new_access_token)
                settings.access_token = new_access_token
                _persist_env("CHATGPT_ACCESS_TOKEN", new_access_token)
                self._refresh_count += 1

                new_session = None
                for cookie in resp.cookies.jar:
                    if cookie.name == "__Secure-next-auth.session-token":
                        new_session = cookie.value
                        break

                if new_session and new_session != self._session_token:
                    self._session_token = new_session
                    settings.session_token = new_session
                    _persist_env("CHATGPT_SESSION_TOKEN", new_session)
                    logger.info("session_token 已轮转并持久化")

                logger.info(
                    f"access_token 刷新成功 (第 {self._refresh_count} 次), "
                    f"有效期至 {time.strftime('%H:%M:%S', time.localtime(self._token_expiry))}"
                )

        except httpx.ConnectError as e:
            raise RuntimeError(f"无法连接 ChatGPT 服务器: {e}")

    @property
    def status(self) -> dict:
        remaining = max(0, self._token_expiry - time.time())
        return {
            "has_access_token": bool(self._access_token),
            "has_session_token": bool(self._session_token),
            "token_expired": self.is_expired(),
            "remaining_seconds": int(remaining),
            "remaining_minutes": round(remaining / 60, 1),
            "refresh_count": self._refresh_count,
            "auto_refresh_enabled": bool(self._session_token),
        }


token_manager = TokenManager()


# ──────────── Format Conversion ────────────

SUPPORTED_ROLES = {"user", "assistant", "system", "developer"}


def _openai_tools_to_codex(tools: list) -> list:
    """Convert OpenAI Chat Completions tools to Responses API format."""
    if not tools:
        return []
    codex_tools = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        codex_tools.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description") or "",
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            "strict": fn.get("strict", False),
        })
    return codex_tools


def _convert_content_parts(content, direction: str = "input") -> list[dict]:
    """Convert OpenAI content (string or list of parts) to Codex Responses API content items.

    direction="input"  -> input_text / input_image  (for user messages)
    direction="output" -> output_text               (for assistant messages)
    """
    if isinstance(content, str):
        text_type = "input_text" if direction == "input" else "output_text"
        return [{"type": text_type, "text": content}] if content else []

    if not isinstance(content, list):
        return []

    parts = []
    for item in content:
        if isinstance(item, str):
            text_type = "input_text" if direction == "input" else "output_text"
            parts.append({"type": text_type, "text": item})
            continue
        if not isinstance(item, dict):
            continue
        part_type = item.get("type", "")

        if part_type == "text":
            text_type = "input_text" if direction == "input" else "output_text"
            parts.append({"type": text_type, "text": item.get("text", "")})

        elif part_type == "image_url":
            image_info = item.get("image_url", {})
            url = image_info.get("url", "") if isinstance(image_info, dict) else str(image_info)
            codex_item = {"type": "input_image", "image_url": url}
            detail = image_info.get("detail") if isinstance(image_info, dict) else None
            if detail and detail != "auto":
                codex_item["detail"] = detail
            parts.append(codex_item)

        elif part_type == "input_audio":
            audio_info = item.get("input_audio", {})
            parts.append({"type": "input_audio", **audio_info})

        else:
            if "text" in item:
                text_type = "input_text" if direction == "input" else "output_text"
                parts.append({"type": text_type, "text": item["text"]})

    return parts


def _extract_text_from_content(content) -> str:
    """Extract plain text from content (string or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(item.get("text", ""))
        return " ".join(t for t in texts if t)
    return ""


def _messages_to_responses_input(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI chat messages to Responses API format.
    Returns (instructions, input_items).
    Handles: system/developer -> instructions; user/assistant/tool -> input items.
    Supports multimodal content (text + image_url) in user messages.
    Assistant with tool_calls + following tool messages -> function_call + function_call_output.
    """
    instructions = "You are a helpful assistant."
    input_items = []
    i = 0

    while i < len(messages):
        msg = messages[i]
        role = msg["role"]
        content = msg.get("content") or ""

        if role in ("system", "developer"):
            instructions = _extract_text_from_content(content)
            i += 1
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                text = _extract_text_from_content(content)
                if text:
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    })
                for tc in tool_calls:
                    call_id = tc.get("id") or tc.get("call_id", "")
                    name = (tc.get("function") or {}).get("name", "")
                    args = (tc.get("function") or {}).get("arguments", "{}")
                    input_items.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": args if isinstance(args, str) else json.dumps(args),
                    })
                j = i + 1
                for tc in tool_calls:
                    call_id = tc.get("id") or tc.get("call_id", "")
                    out_content = ""
                    if j < len(messages) and messages[j].get("role") == "tool":
                        out_content = messages[j].get("content") or ""
                        j += 1
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": out_content if isinstance(out_content, str) else json.dumps(out_content),
                    })
                i = j
                continue
            input_items.append({
                "type": "message",
                "role": "assistant",
                "content": _convert_content_parts(content, "output") or [{"type": "output_text", "text": ""}],
            })
            i += 1
            continue

        if role == "tool":
            text = _extract_text_from_content(content)
            input_items.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            })
            i += 1
            continue

        if role not in SUPPORTED_ROLES:
            role = "user"

        codex_parts = _convert_content_parts(content, "input")
        if not codex_parts:
            codex_parts = [{"type": "input_text", "text": ""}]
        input_items.append({
            "type": "message",
            "role": role,
            "content": codex_parts,
        })
        i += 1

    return instructions, input_items


def _normalize_tool_choice(tool_choice, has_tools: bool):
    """Normalize tool_choice for Codex: str passthrough, dict passthrough, default auto/none."""
    if tool_choice is not None:
        return tool_choice
    return "auto" if has_tools else "none"


def _build_codex_body(
    messages: list[dict],
    model: str = "gpt-5",
    max_tokens: Optional[int] = None,  # accepted but not forwarded; Codex rejects max_output_tokens
    temperature: Optional[float] = None,
    tools: Optional[list] = None,
    tool_choice = None,
    reasoning_effort: Optional[str] = None,
    response_format = None,
    stop = None,
    seed: Optional[int] = None,
) -> dict:
    instructions, input_items = _messages_to_responses_input(messages)
    codex_tools = _openai_tools_to_codex(tools) if tools else []

    body = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "tools": codex_tools,
        "tool_choice": _normalize_tool_choice(tool_choice, bool(codex_tools)),
        "store": False,
        "stream": True,
    }

    if temperature is not None:
        body["temperature"] = temperature
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    if response_format is not None:
        fmt_type = response_format.get("type", "text")
        if fmt_type == "json_object":
            body["text"] = {"format": {"type": "json_object"}}
        elif fmt_type == "json_schema":
            schema_def = response_format.get("json_schema", {})
            body["text"] = {"format": {
                "type": "json_schema",
                "name": schema_def.get("name", "response"),
                "schema": schema_def.get("schema", {}),
                "strict": schema_def.get("strict", False),
            }}
    if stop is not None:
        body["stop"] = stop if isinstance(stop, list) else [stop]
    if seed is not None:
        body["seed"] = seed

    return body


# ──────────── Streaming via curl_cffi (sync, run in thread) ────────────

def _make_tool_call_id() -> str:
    """Generate a tool_call ID in the format OpenClaw expects (call_xxxx)."""
    return f"call_{uuid.uuid4().hex[:24]}"


def _stream_codex_sync(url: str, headers: dict, body: dict, proxy: Optional[str]):
    """Synchronous streaming using curl_cffi (preserves Chrome TLS fingerprint).
    Yields incremental events: content, tool_call_start, tool_call_delta, finish, error.
    """
    with CurlSession(impersonate=IMPERSONATE, proxy=proxy) as session:
        resp = session.post(url, headers=headers, json=body, stream=True, timeout=120)

        if resp.status_code == 401:
            yield {"type": "error", "message": "Token 无效或已过期 (401)，请刷新"}
            return
        if resp.status_code == 403:
            detail = ""
            for chunk in resp.iter_content():
                detail += chunk.decode("utf-8", errors="replace")
                if len(detail) > 300:
                    break
            yield {"type": "error", "message": f"403 Forbidden: {detail[:300]}"}
            return
        if resp.status_code == 429:
            yield {"type": "error", "message": "速率限制 (429)，请稍后重试"}
            return
        if resp.status_code != 200:
            detail = ""
            for chunk in resp.iter_content():
                detail += chunk.decode("utf-8", errors="replace")
                if len(detail) > 500:
                    break
            yield {"type": "error", "message": f"HTTP {resp.status_code}: {detail[:500]}"}
            return

        buffer = ""
        collected_text = ""
        resolved_model = ""
        tool_calls_acc: list[dict] = []
        tool_call_index = -1

        def _emit_tool_calls(output: list) -> list:
            out = []
            for item in output:
                if item.get("type") == "function_call":
                    tc_id = item.get("call_id") or item.get("id") or _make_tool_call_id()
                    out.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}") if isinstance(item.get("arguments"), str) else json.dumps(item.get("arguments", {})),
                        },
                    })
            return out

        for chunk in resp.iter_content():
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="replace")
            buffer += chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    continue
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    yield {"type": "finish", "text": collected_text, "model": resolved_model, "tool_calls": tool_calls_acc}
                    return

                try:
                    evt = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                etype = evt.get("type", "")

                if etype == "response.created":
                    resp_obj = evt.get("response", {})
                    resolved_model = resp_obj.get("model", "")
                    tool_calls_acc = []
                    tool_call_index = -1

                elif etype == "response.output_item.added":
                    item = evt.get("item", {})
                    if item.get("type") == "function_call":
                        tool_call_index += 1
                        tc_id = item.get("id") or item.get("call_id") or _make_tool_call_id()
                        tc_name = item.get("name", "")
                        tc_entry = {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": tc_name, "arguments": ""},
                        }
                        tool_calls_acc.append(tc_entry)
                        yield {
                            "type": "tool_call_start",
                            "index": tool_call_index,
                            "id": tc_id,
                            "name": tc_name,
                        }

                elif etype == "response.function_call_arguments.delta":
                    delta_str = evt.get("delta", "")
                    if delta_str and tool_calls_acc:
                        tool_calls_acc[-1]["function"]["arguments"] += delta_str
                        yield {
                            "type": "tool_call_delta",
                            "index": tool_call_index,
                            "arguments_delta": delta_str,
                        }

                elif etype == "response.function_call_arguments.done":
                    item = evt.get("item", {})
                    if item and tool_calls_acc:
                        final_args = item.get("arguments")
                        if final_args is not None:
                            tool_calls_acc[-1]["function"]["arguments"] = final_args

                elif etype == "response.output_text.delta":
                    delta = evt.get("delta", "")
                    if delta:
                        collected_text += delta
                        yield {"type": "content", "text": delta}

                elif etype == "response.completed":
                    resp_obj = evt.get("response", {})
                    resolved_model = resp_obj.get("model", resolved_model)
                    usage = resp_obj.get("usage", {})
                    output = resp_obj.get("output", [])
                    for item in output:
                        if item.get("type") == "message":
                            for c in item.get("content", []):
                                if c.get("type") == "output_text" and c.get("text"):
                                    collected_text = c["text"]
                    fc_items = [x for x in output if x.get("type") == "function_call"]
                    if fc_items and not tool_calls_acc:
                        tool_calls_acc = _emit_tool_calls(fc_items)
                        for idx, tc in enumerate(tool_calls_acc):
                            yield {"type": "tool_call_start", "index": idx, "id": tc["id"], "name": tc["function"]["name"]}
                            if tc["function"]["arguments"]:
                                yield {"type": "tool_call_delta", "index": idx, "arguments_delta": tc["function"]["arguments"]}
                    if tool_calls_acc:
                        logger.info(f"[codex] tool_calls returned: {[tc['function']['name'] for tc in tool_calls_acc]}")
                    yield {
                        "type": "finish",
                        "text": collected_text,
                        "model": resolved_model,
                        "usage": usage,
                        "tool_calls": tool_calls_acc,
                    }
                    return

                elif etype in ("error", "response.failed"):
                    error_msg = evt.get("error", {}).get("message", json.dumps(evt)[:300])
                    yield {"type": "error", "message": error_msg}
                    return

        if collected_text or tool_calls_acc:
            yield {"type": "finish", "text": collected_text, "model": resolved_model, "tool_calls": tool_calls_acc}


# ──────────── Public API ────────────

MODEL_MAP = {
    # GPT-5.2 (latest, 2025-12-11)
    "gpt-5.2": "gpt-5.2",
    "gpt-5-2": "gpt-5.2",
    "gpt-5-2-thinking": "gpt-5.2",
    "gpt-5.2-thinking": "gpt-5.2",
    # GPT-5.1 (2025-11-13)
    "gpt-5.1": "gpt-5.1",
    "gpt-5-1": "gpt-5.1",
    "gpt-5-1-thinking": "gpt-5.1",
    # GPT-5 (2025-08-07)
    "gpt-5": "gpt-5",
    # Legacy names -> latest
    "gpt-4o": "gpt-5.2",
    "gpt-4o-mini": "gpt-5.2",
    "gpt-4": "gpt-5.2",
    "gpt-3.5-turbo": "gpt-5.2",
    "auto": "gpt-5.2",
}


def _resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, "gpt-5.2")


async def stream_chat(
    messages: list[dict],
    model: str = "gpt-5",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tools: Optional[list] = None,
    tool_choice = None,
    reasoning_effort: Optional[str] = None,
    response_format = None,
    stop = None,
    seed: Optional[int] = None,
) -> AsyncGenerator[dict, None]:
    try:
        access_token = await token_manager.get_access_token()
    except RuntimeError as e:
        yield {"type": "error", "message": str(e)}
        return

    resolved = _resolve_model(model)
    headers = {**CODEX_HEADERS}
    headers["authorization"] = f"Bearer {access_token}"
    headers["oai-device-id"] = DEVICE_ID

    body = _build_codex_body(
        messages, resolved, max_tokens, temperature,
        tools, tool_choice, reasoning_effort,
        response_format=response_format, stop=stop, seed=seed,
    )
    proxy = settings.proxy if settings.proxy else None

    if not HAS_CURL_CFFI:
        yield {"type": "error", "message": "curl_cffi 未安装，无法请求 Codex API"}
        return

    tool_names = [t.get("function", {}).get("name", "?") for t in (tools or []) if t.get("type") == "function"]
    logger.info(f"[codex] model={resolved} proxy={proxy} tools={tool_names or 'none'} reasoning={reasoning_effort}")

    loop = asyncio.get_running_loop()
    gen = _stream_codex_sync(CODEX_URL, headers, body, proxy)

    try:
        while True:
            event = await loop.run_in_executor(None, next, gen, None)
            if event is None:
                break
            yield event
            if event["type"] in ("finish", "error"):
                return
    except StopIteration:
        pass
    finally:
        gen.close()


async def chat_completion(
    messages: list[dict],
    model: str = "gpt-5",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tools: Optional[list] = None,
    tool_choice = None,
    reasoning_effort: Optional[str] = None,
    response_format = None,
    stop = None,
    seed: Optional[int] = None,
) -> dict:
    full_text = ""
    resolved_model = model
    usage = {}
    tool_calls: list = []
    async for event in stream_chat(
        messages, model, max_tokens, temperature, tools, tool_choice,
        reasoning_effort, response_format=response_format, stop=stop, seed=seed,
    ):
        if event["type"] == "content":
            full_text += event["text"]
        elif event["type"] == "finish":
            full_text = event.get("text", full_text)
            resolved_model = event.get("model", resolved_model)
            usage = event.get("usage", {})
            tool_calls = event.get("tool_calls") or []
        elif event["type"] == "error":
            raise RuntimeError(event["message"])
    return {"text": full_text, "model": resolved_model, "usage": usage, "tool_calls": tool_calls}
