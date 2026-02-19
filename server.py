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
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator

from config import settings
from chatgpt_client import stream_chat, chat_completion, token_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

app = FastAPI(
    title="ChatGPT-to-API",
    description="将 ChatGPT Plus 网页版转为 OpenAI 兼容 API (via Codex Responses API)，支持 Vision / Tool Calling / JSON Mode",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_FINGERPRINT = "fp_chatgpt_proxy"


def _make_error_response(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    param: str = None,
    code: str = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_type = "invalid_request_error"
    if exc.status_code == 401:
        error_type = "authentication_error"
    elif exc.status_code == 404:
        error_type = "not_found_error"
    elif exc.status_code == 429:
        error_type = "rate_limit_error"
    elif exc.status_code >= 500:
        error_type = "server_error"
    return _make_error_response(exc.status_code, str(exc.detail), error_type)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = first.get("loc", [])
    param = ".".join(str(l) for l in loc) if loc else None
    msg = first.get("msg", str(exc))
    return _make_error_response(400, msg, "invalid_request_error", param)


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
    content: Union[str, list, None] = ""
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v):
        if v is None:
            return ""
        return v


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-5.2"
    messages: list[MessageInput]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None
    stream_options: Optional[dict] = None
    reasoning_effort: Optional[str] = None
    store: Optional[bool] = None
    response_format: Optional[dict] = None
    stop: Optional[Union[str, list]] = None
    seed: Optional[int] = None
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None

    model_config = {"extra": "allow"}

    @property
    def resolved_max_tokens(self) -> Optional[int]:
        return self.max_completion_tokens or self.max_tokens

    @property
    def include_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.get("include_usage"))


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
    tool_calls: Optional[list] = None,
) -> dict:
    u = usage or {}
    message: dict = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": "tool_calls" if tool_calls else finish_reason,
        }],
        "usage": {
            "prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": u.get("output_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
        },
    }


def _make_sse(chunk_dict: dict) -> str:
    return f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"


def _base_chunk(chat_id: str, model: str) -> dict:
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
    }


def make_content_chunk(chat_id: str, model: str, content: str) -> str:
    c = _base_chunk(chat_id, model)
    c["choices"] = [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
    return _make_sse(c)


def make_tool_call_start_chunk(chat_id: str, model: str, tc_index: int, tc_id: str, tc_name: str) -> str:
    """First chunk for a tool call: includes id, type, function name, empty arguments."""
    c = _base_chunk(chat_id, model)
    c["choices"] = [{"index": 0, "delta": {
        "tool_calls": [{
            "index": tc_index,
            "id": tc_id,
            "type": "function",
            "function": {"name": tc_name, "arguments": ""},
        }],
    }, "finish_reason": None}]
    return _make_sse(c)


def make_tool_call_args_chunk(chat_id: str, model: str, tc_index: int, args_delta: str) -> str:
    """Incremental argument fragment for a tool call."""
    c = _base_chunk(chat_id, model)
    c["choices"] = [{"index": 0, "delta": {
        "tool_calls": [{"index": tc_index, "function": {"arguments": args_delta}}],
    }, "finish_reason": None}]
    return _make_sse(c)


def make_finish_chunk(chat_id: str, model: str, has_tool_calls: bool) -> str:
    """Empty delta with finish_reason."""
    c = _base_chunk(chat_id, model)
    c["choices"] = [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if has_tool_calls else "stop"}]
    return _make_sse(c)


def make_usage_chunk(chat_id: str, model: str, usage: dict) -> str:
    """Final chunk with usage info and empty choices (per OpenAI spec)."""
    c = _base_chunk(chat_id, model)
    c["choices"] = []
    c["usage"] = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    return _make_sse(c)


# ──────────── API Routes ────────────

@app.get("/")
async def root():
    return {
        "service": "ChatGPT-to-API",
        "version": "5.0.0",
        "backend": "Codex Responses API",
        "token_status": token_manager.status,
    }


@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    verify_auth(authorization)
    return {"object": "list", "data": AVAILABLE_MODELS}


@app.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str, authorization: Optional[str] = Header(None)):
    verify_auth(authorization)
    for m in AVAILABLE_MODELS:
        if m["id"] == model_id:
            return m
    return _make_error_response(404, f"The model '{model_id}' does not exist", "not_found_error", "model_id")


def _request_messages_to_list(request: ChatCompletionRequest) -> list[dict]:
    out = []
    for m in request.messages:
        msg: dict = {"role": m.role, "content": m.content}
        if m.tool_calls is not None:
            msg["tool_calls"] = m.tool_calls
        if m.tool_call_id is not None:
            msg["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            msg["name"] = m.name
        out.append(msg)
    return out


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    verify_auth(authorization)

    if request.n is not None and request.n > 1:
        return _make_error_response(400, "n > 1 is not supported by this proxy", "invalid_request_error", "n")

    messages = _request_messages_to_list(request)
    model = request.model
    max_tokens = request.resolved_max_tokens

    extra_params = {
        "response_format": request.response_format,
        "stop": request.stop,
        "seed": request.seed,
    }

    if request.stream:
        return StreamingResponse(
            _stream_response(
                messages, model, max_tokens, request.temperature,
                request.tools, request.tool_choice,
                request.reasoning_effort, request.include_usage,
                **extra_params,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await chat_completion(
            messages, model, max_tokens, request.temperature,
            request.tools, request.tool_choice,
            request.reasoning_effort,
            **extra_params,
        )
        return JSONResponse(make_chat_completion_response(
            result.get("model", model),
            result["text"],
            usage=result.get("usage"),
            tool_calls=result.get("tool_calls"),
        ))
    except RuntimeError as e:
        return _make_error_response(502, str(e), "server_error")


async def _stream_response(
    messages: list[dict],
    model: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tools: Optional[list] = None,
    tool_choice = None,
    reasoning_effort: Optional[str] = None,
    include_usage: bool = False,
    response_format: Optional[dict] = None,
    stop = None,
    seed: Optional[int] = None,
):
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    has_tool_calls = False

    role_chunk = _base_chunk(chat_id, model)
    role_chunk["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    yield _make_sse(role_chunk)

    async for event in stream_chat(
        messages, model, max_tokens, temperature,
        tools, tool_choice, reasoning_effort,
        response_format=response_format, stop=stop, seed=seed,
    ):
        etype = event["type"]

        if etype == "content":
            yield make_content_chunk(chat_id, event.get("model", model), event["text"])

        elif etype == "tool_call_start":
            has_tool_calls = True
            yield make_tool_call_start_chunk(
                chat_id, model, event["index"], event["id"], event["name"],
            )

        elif etype == "tool_call_delta":
            yield make_tool_call_args_chunk(
                chat_id, model, event["index"], event["arguments_delta"],
            )

        elif etype == "finish":
            resolved = event.get("model", model)
            usage = event.get("usage") or {}
            if event.get("tool_calls"):
                has_tool_calls = True
            yield make_finish_chunk(chat_id, resolved, has_tool_calls)
            if include_usage:
                yield make_usage_chunk(chat_id, resolved, usage)
            yield "data: [DONE]\n\n"
            return

        elif etype == "error":
            err = _base_chunk(chat_id, model)
            err["choices"] = [{
                "index": 0,
                "delta": {"content": f"\n\n[ERROR] {event['message']}"},
                "finish_reason": "stop",
            }]
            yield _make_sse(err)
            yield "data: [DONE]\n\n"
            return

    yield make_finish_chunk(chat_id, model, has_tool_calls)
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
    print("  ChatGPT-to-API Proxy v5.0 (OpenAI API Compatible)")
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
