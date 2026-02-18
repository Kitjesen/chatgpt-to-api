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

def _messages_to_responses_input(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI chat messages to Responses API format.
    Returns (instructions, input_items).
    system/developer -> instructions; user/assistant -> input items.
    Unsupported roles (tool, function) are folded into user messages.
    """
    instructions = "You are a helpful assistant."
    input_items = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content") or ""

        if role in ("system", "developer"):
            instructions = content
            continue

        if role not in SUPPORTED_ROLES:
            role = "user"

        if role == "assistant":
            content_type = "output_text"
        else:
            content_type = "input_text"

        input_items.append({
            "type": "message",
            "role": role,
            "content": [{"type": content_type, "text": content}],
        })

    return instructions, input_items


def _build_codex_body(
    messages: list[dict],
    model: str = "gpt-5",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> dict:
    instructions, input_items = _messages_to_responses_input(messages)

    body = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "tools": [],
        "tool_choice": "auto",
        "store": False,
        "stream": True,
    }

    if max_tokens is not None:
        body["max_output_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature

    return body


# ──────────── Streaming via curl_cffi (sync, run in thread) ────────────

def _stream_codex_sync(url: str, headers: dict, body: dict, proxy: Optional[str]):
    """Synchronous streaming using curl_cffi (preserves Chrome TLS fingerprint)."""
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
                    yield {"type": "finish", "text": collected_text, "model": resolved_model}
                    return

                try:
                    evt = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                etype = evt.get("type", "")

                if etype == "response.created":
                    resp_obj = evt.get("response", {})
                    resolved_model = resp_obj.get("model", "")

                elif etype == "response.output_text.delta":
                    delta = evt.get("delta", "")
                    if delta:
                        collected_text += delta
                        yield {"type": "content", "text": delta}

                elif etype == "response.completed":
                    resp_obj = evt.get("response", {})
                    resolved_model = resp_obj.get("model", resolved_model)
                    usage = resp_obj.get("usage", {})
                    # Extract final text from completed response
                    output = resp_obj.get("output", [])
                    for item in output:
                        if item.get("type") == "message":
                            for c in item.get("content", []):
                                if c.get("type") == "output_text" and c.get("text"):
                                    collected_text = c["text"]
                    yield {
                        "type": "finish",
                        "text": collected_text,
                        "model": resolved_model,
                        "usage": usage,
                    }
                    return

                elif etype in ("error", "response.failed"):
                    error_msg = evt.get("error", {}).get("message", json.dumps(evt)[:300])
                    yield {"type": "error", "message": error_msg}
                    return

        if collected_text:
            yield {"type": "finish", "text": collected_text, "model": resolved_model}


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

    body = _build_codex_body(messages, resolved, max_tokens, temperature)
    proxy = settings.proxy if settings.proxy else None

    if not HAS_CURL_CFFI:
        yield {"type": "error", "message": "curl_cffi 未安装，无法请求 Codex API"}
        return

    logger.info(f"[codex] model={resolved} proxy={proxy}")

    loop = asyncio.get_event_loop()
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


async def chat_completion(
    messages: list[dict],
    model: str = "gpt-5",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> dict:
    full_text = ""
    resolved_model = model
    usage = {}
    async for event in stream_chat(messages, model, max_tokens, temperature):
        if event["type"] == "content":
            full_text += event["text"]
        elif event["type"] == "finish":
            full_text = event.get("text", full_text)
            resolved_model = event.get("model", resolved_model)
            usage = event.get("usage", {})
        elif event["type"] == "error":
            raise RuntimeError(event["message"])
    return {"text": full_text, "model": resolved_model, "usage": usage}
