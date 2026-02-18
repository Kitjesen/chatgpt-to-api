"""
ChatGPT-to-API: OpenAI 兼容的 API 服务器
通过 Codex Responses API (/backend-api/codex/responses) 实现
支持 access_token 自动刷新 + GPT-5
"""

import uuid
import json
import time
import logging
from typing import Optional, Union

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from config import settings
from chatgpt_client import stream_chat, chat_completion, token_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

app = FastAPI(
    title="ChatGPT-to-API",
    description="将 ChatGPT Plus 网页版转为 OpenAI 兼容 API (via Codex Responses API)",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AVAILABLE_MODELS = [
    {"id": "gpt-5.2", "object": "model", "owned_by": "openai", "created": int(time.time())},
    {"id": "gpt-5.1", "object": "model", "owned_by": "openai", "created": int(time.time())},
    {"id": "gpt-5", "object": "model", "owned_by": "openai", "created": int(time.time())},
    {"id": "gpt-4o", "object": "model", "owned_by": "openai", "created": int(time.time())},
    {"id": "gpt-4", "object": "model", "owned_by": "openai", "created": int(time.time())},
    {"id": "auto", "object": "model", "owned_by": "chatgpt-proxy", "created": int(time.time())},
]


class MessageInput(BaseModel):
    role: str
    content: Union[str, list] = ""

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v):
        if isinstance(v, list):
            parts = []
            for item in v:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("text", ""))
            return " ".join(parts)
        if v is None:
            return ""
        return v


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-5.2"
    messages: list[MessageInput]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    model_config = {"extra": "allow"}


def verify_auth(authorization: Optional[str] = None):
    if not settings.api_key:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.replace("Bearer ", "")
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def make_chat_completion_response(
    model: str,
    content: str,
    finish_reason: str = "stop",
    usage: Optional[dict] = None,
) -> dict:
    u = usage or {}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": u.get("output_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
        },
    }


def make_chunk(
    chat_id: str,
    model: str,
    delta_content: Optional[str] = None,
    finish: bool = False,
) -> str:
    delta = {}
    finish_reason = None
    if finish:
        finish_reason = "stop"
    elif delta_content is not None:
        delta = {"role": "assistant", "content": delta_content}

    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ──────────── API Routes ────────────

@app.get("/")
async def root():
    return {
        "service": "ChatGPT-to-API",
        "version": "3.0.0",
        "backend": "Codex Responses API",
        "token_status": token_manager.status,
    }


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    verify_auth(authorization)
    return {"object": "list", "data": AVAILABLE_MODELS}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    verify_auth(authorization)
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    model = request.model

    if request.stream:
        return StreamingResponse(
            _stream_response(messages, model, request.max_tokens, request.temperature),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await chat_completion(messages, model, request.max_tokens, request.temperature)
        return JSONResponse(make_chat_completion_response(
            result.get("model", model),
            result["text"],
            usage=result.get("usage"),
        ))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


async def _stream_response(
    messages: list[dict],
    model: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
):
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

    first_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

    async for event in stream_chat(messages, model, max_tokens, temperature):
        if event["type"] == "content":
            yield make_chunk(chat_id, event.get("model", model), delta_content=event["text"])
        elif event["type"] == "finish":
            resolved = event.get("model", model)
            yield make_chunk(chat_id, resolved, finish=True)
            yield "data: [DONE]\n\n"
            return
        elif event["type"] == "error":
            error_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n\n[ERROR] {event['message']}"},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

    yield make_chunk(chat_id, model, finish=True)
    yield "data: [DONE]\n\n"


# ──────────── Token Management ────────────

@app.post("/admin/token")
async def update_token(body: dict, authorization: Optional[str] = Header(None)):
    verify_auth(authorization)
    if "access_token" in body:
        token_manager.set_access_token(body["access_token"])
    if "session_token" in body:
        token_manager.set_session_token(body["session_token"])
    if "access_token" not in body and "session_token" not in body:
        raise HTTPException(status_code=400, detail="需要 access_token 或 session_token")
    return {"status": "ok", "token_status": token_manager.status}


@app.get("/admin/status")
async def admin_status(authorization: Optional[str] = Header(None)):
    verify_auth(authorization)
    return {
        "token": token_manager.status,
        "models": [m["id"] for m in AVAILABLE_MODELS],
        "backend": "Codex Responses API (/backend-api/codex/responses)",
    }


@app.post("/admin/refresh")
async def force_refresh(authorization: Optional[str] = Header(None)):
    verify_auth(authorization)
    if not token_manager._session_token:
        raise HTTPException(status_code=400, detail="未设置 session_token，无法刷新")
    token_manager._token_expiry = 0
    try:
        await token_manager.get_access_token()
        return {"status": "ok", "token_status": token_manager.status}
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


# ──────────── Startup ────────────

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  ChatGPT-to-API Proxy v3.0 (Codex Responses API)")
    print(f"  http://{settings.host}:{settings.port}")
    print(f"  API: http://{settings.host}:{settings.port}/v1/chat/completions")
    print(f"  Backend: /backend-api/codex/responses")
    print("=" * 60)

    status = token_manager.status
    if status["has_session_token"]:
        print(f"\n  [OK] session_token 已配置，自动刷新已启用")
    if status["has_access_token"]:
        print(f"  [OK] access_token 有效，剩余 {status['remaining_minutes']} 分钟")
    if not status["has_access_token"] and not status["has_session_token"]:
        print("\n  [!] 未配置任何 token!")
        print("  方式1: .env 设置 CHATGPT_SESSION_TOKEN (推荐，自动刷新)")
        print("  方式2: .env 设置 CHATGPT_ACCESS_TOKEN (需手动刷新)")
        print("  方式3: POST /admin/token 动态设置")

    print()
    uvicorn.run(app, host=settings.host, port=settings.port)
