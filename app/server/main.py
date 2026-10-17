"""FastAPI 服务入口（SSE 版：流式输出 + 事件回调架构）。

运行：.venv/Scripts/python -m uvicorn app.server.main:app --host 127.0.0.1 --port 8000
（单 worker —— 进程内 Agent dict 不支持多 worker）

POST /api/chat 现在返回 SSE 事件流（text/event-stream）：agent 的每一步
（思考/工具调用/工具结果/待确认/最终答复/错误）都变成一个 {type: ...} JSON 事件逐帧推送。
敏感操作另有 POST /api/confirm（不受聊天限流）。调试页 GET /demo。

线程模型：
- agent.achat() 跑在 uvicorn 事件循环上：LLM 调用 await AsyncOpenAI，互不占用线程
- 构造 Agent / PG 读写 / 同步工具 下沉 asyncio.to_thread
- 事件仍通过 call_soon_threadsafe 入队（同一 loop 里也安全，工具线程回调也不乱序）
- CLI / 测试继续调同步 chat()（内部 asyncio.run）

同一用户的同一会话只允许一个操作在执行，冲突请求返回 409。
"""

from __future__ import annotations

import asyncio
import json
import inspect
import logging
import time
from contextlib import asynccontextmanager

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from app.server.auth import authenticated_user, check_claim
from app.server.identifiers import validate_id
from fastapi.responses import FileResponse, StreamingResponse

from app.config.settings import settings
from app.observability import request_id, new_request_id, failure, stage, event
from app.agent.maintenance import drain_maintenance, close_resources
from app.server.schemas import (
    ChatRequest,
    ConfirmRequest,
    ConfirmResponse,
    DeleteSessionResponse,
    HealthResponse,
    SessionMessagesResponse,
)
from app.server.sessions import SESSION_DIR, SessionRegistry, SessionCapacityError

registry = SessionRegistry()
_active_tasks: set[asyncio.Task] = set()
_stopping = False

# SSE 流结束哨兵：用对象而非 dict，和事件本身区分；走同一队列保证 final/error 先到
_END = object()


def _sse_line(event: dict) -> str:
    """把事件 dict 序列化成一条 SSE 帧。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    mode = "Multi-Agent" if settings.multi_agent_enabled else "ReAct"
    from app.server.cache import get_cache
    redis_status = get_cache().status()
    print(
        f"🟢 服务启动 | mode={mode} | model={settings.model_name} | redis={redis_status}"
    )
    global _stopping
    _stopping = False
    registry.open()
    logging.getLogger("yier.service").setLevel(logging.INFO)
    if not logging.getLogger("yier.service").handlers:
        logging.getLogger("yier.service").addHandler(logging.StreamHandler())
    async def reap():
        while True:
            await asyncio.sleep(max(0.1, settings.session_reap_interval))
            try:
                await registry.reap_idle()
            except Exception as exc:
                failure("session_reaper_failed", exc)
    reaper = asyncio.create_task(reap())
    try:
        yield
    finally:
        _stopping = True
        reaper.cancel()
        deadline = time.monotonic() + settings.shutdown_timeout
        tasks = set(_active_tasks)
        tasks.add(reaper)
        _, pending = await asyncio.wait(tasks, timeout=max(0, deadline-time.monotonic()))
        for task in pending:
            task.cancel()
        if pending:
            event("shutdown_work_timeout", count=len(pending))
        await registry.close_all(timeout=max(0, deadline-time.monotonic()))
        from app.server.repository import aclose_engine
        from app.server.cache import aclose_cache
        cleanup = asyncio.create_task(aclose_engine())
        _, pending = await asyncio.wait({cleanup}, timeout=max(0, deadline-time.monotonic()))
        for task in pending:
            task.cancel()
        aclose_cache()
        event("shutdown_complete")


app = FastAPI(
    title="一二商城 · 智能客服「一二」服务化",
    version="0.9.0",
    lifespan=lifespan,
)

_DEMO_HTML = Path(__file__).resolve().parents[2] / "web" / "demo.html"


@app.get("/demo")
def demo_page():
    """调试页：SSE 事件流 + 退款确认按钮（非产品 UI）。"""
    if not _DEMO_HTML.is_file():
        raise HTTPException(status_code=404, detail="web/demo.html 不存在")
    return FileResponse(_DEMO_HTML, media_type="text/html")


@app.get("/health", response_model=HealthResponse)
def health():
    from app.server.cache import get_cache
    return HealthResponse(
        status="ok",
        resident_sessions=len(registry),
        multi_agent_enabled=settings.multi_agent_enabled,
        redis=get_cache().status(),
    )


def _reserve(session_id: str, user: str) -> None:
    if _stopping:
        raise HTTPException(503, "Service is shutting down")
    try:
        validate_id(session_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    if not registry.acquire(session_id, user):
        raise HTTPException(409, "Session is busy; retry after the current operation completes")


def _track(task):
    _active_tasks.add(task)
    def done(t):
        _active_tasks.discard(t)
        if not t.cancelled():
            t.exception()  # Retrieve failures even after a client disconnects.
    task.add_done_callback(done)
    return task


@app.post("/api/chat")
async def chat(req: ChatRequest, user: str = Depends(authenticated_user)):
    check_claim(req.user_id, user)
    from app.server.ratelimit import get_limiter
    if not await get_limiter().acquire(user):
        raise HTTPException(429, "请求过于频繁，请稍后再试", headers={"Retry-After": "60"})
    _reserve(req.session_id, user)
    # Start before returning the response: the task owns the lease even if the
    # browser disconnects before consuming its first event.
    rid = new_request_id()
    stream = _start_chat(req.model_copy(update={"user_id": user}), rid)
    return StreamingResponse(stream, media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-ID": rid,
    })


def _start_chat(req: ChatRequest, rid=None):
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()
    connected = True

    def enqueue(event):
        if connected:
            q.put_nowait(event)

    def emit(event):
        loop.call_soon_threadsafe(enqueue, event)

    async def run():
        context_token = request_id.set(rid or new_request_id())
        agent = None
        try:
            if registry.get(req.session_id, req.user_id) is None:
                await registry.reap_idle(force_room=True)
            with stage("session_init"):
                agent = await asyncio.to_thread(registry.get_or_create, req.session_id, req.user_id)
            agent.on_event = emit
            emit({"type": "session", "session_id": req.session_id, "user_id": req.user_id})
            with stage("chat_total"):
                await agent.achat(req.message)
        except SessionCapacityError:
            emit({"type": "error", "message": "服务会话已满，请稍后重试", "code": "session_capacity"})
        except Exception as exc:
            failure("chat_failed", exc)
            emit({"type": "error", "message": "本轮处理失败，请稍后重试", "request_id": request_id.get()})
        finally:
            if agent is not None:
                agent.on_event = None
            registry.release(req.session_id, req.user_id)
            emit(_END)
            request_id.reset(context_token)

    _track(asyncio.create_task(run()))

    async def stream():
        nonlocal connected
        try:
            while True:
                event = await q.get()
                if event is _END:
                    break
                yield _sse_line(event)
        finally:
            # Continue the accepted turn and persistence; do not cancel a tool
            # with external effects just because the transport disconnected.
            connected = False
    return stream()


async def _exclusive(session_id, user, operation):
    _reserve(session_id, user)
    async def run():
        context_token = request_id.set(new_request_id())
        try:
            agent = registry.get(session_id, user)
            if agent is None:
                await registry.reap_idle(force_room=True)
            if agent is not None:
                await drain_maintenance(agent)
            if inspect.iscoroutinefunction(operation):
                return await operation()
            return await asyncio.to_thread(operation)
        except SessionCapacityError:
            raise HTTPException(503, "Resident session capacity reached") from None
        except HTTPException:
            raise
        except Exception as exc:
            failure("session_operation_failed", exc)
            raise HTTPException(500, {"message": "操作失败", "request_id": request_id.get()}) from None
        finally:
            request_id.reset(context_token)
            registry.release(session_id, user)
    return await asyncio.shield(_track(asyncio.create_task(run())))


@app.post("/api/confirm", response_model=ConfirmResponse)
async def confirm(req: ConfirmRequest, user: str = Depends(authenticated_user)):
    check_claim(req.user_id, user)
    async def operation():
        entry = await asyncio.to_thread(registry.get_or_restore, req.session_id, user)
        if entry is None:
            raise HTTPException(404, "会话不存在")
        agent, _ = entry
        info = await agent.aresolve_hitl(req.token, req.approved)
        if not info.get("ok"):
            raise HTTPException(400, info.get("error") or "确认失败")
        return ConfirmResponse(**{k: v for k, v in info.items() if k in ConfirmResponse.model_fields})
    return await _exclusive(req.session_id, user, operation)


@app.get("/api/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session(session_id: str, user: str = Depends(authenticated_user)):
    def operation():
        entry = registry.get_or_restore(session_id, user)
        if entry is None:
            raise HTTPException(404, "会话不存在")
        agent, _ = entry
        return SessionMessagesResponse(session_id=session_id, user_id=user,
            history_size=len(agent.raw_messages), messages=list(agent.raw_messages))
    return await _exclusive(session_id, user, operation)


@app.delete("/api/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str, user: str = Depends(authenticated_user)):
    async def operation():
        entry = await asyncio.to_thread(registry.get_or_restore, session_id, user)
        if entry is None:
            raise HTTPException(404, "会话不存在")
        agent, _ = entry
        tasks = list(getattr(agent, "_ltm_tasks", ()))
        # Drain existing writes before deletion; do not run a new consolidation.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.to_thread(agent.reset)
        registry.remove(session_id, user)
        await close_resources(agent)
        return DeleteSessionResponse(deleted=True, session_id=session_id)
    return await _exclusive(session_id, user, operation)


if __name__ == "__main__":
    uvicorn.run("app.server.main:app", host="127.0.0.1", port=8000, reload=False)
